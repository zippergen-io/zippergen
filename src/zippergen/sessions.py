"""Experimental local session scheduler for detached global spawn.

One instance belongs to one physical participant. Static code is shared, while
residuals, visit counters, and generated control variables belong to a session.
Bindings name participant-local cells. Only selected cells are copied at a fork.
The complete state is encoded in the existing role-state transaction.
"""
from __future__ import annotations

import copy
import base64
import binascii
import json
import re
from collections.abc import MutableMapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, cast

from zippergen.locator import _children
from zippergen.syntax import (
    EmptyStmt, ForkStmt, AllocateSessionStmt, BoundSendStmt, BoundReceiveStmt,
    BoundForkStmt, IfRecvStmt, WhileRecvStmt, LocalStmt, VarExpr,
)
from zippergen.value_codec import encode_value, decode_value

if TYPE_CHECKING:
    from zippergen.runtime import PendingExternal


def has_forks(stmt) -> bool:
    return isinstance(stmt, (ForkStmt, AllocateSessionStmt, BoundSendStmt,
                             BoundReceiveStmt, BoundForkStmt)) or any(
                                 has_forks(c) for c in _children(stmt))


_ENDPOINT = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,127}\Z")
_FRESH_PREFIX = "~spawn:"


def fresh_endpoint(child: str, role: str) -> str:
    """Injective internal address, disjoint from the root-role namespace."""
    payload = json.dumps([child, role], separators=(",", ":"), ensure_ascii=False)
    return _FRESH_PREFIX + base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


def validate_root_endpoint(endpoint: str) -> str:
    if not isinstance(endpoint, str) or endpoint.startswith(_FRESH_PREFIX):
        raise ValueError(f"Root role name {endpoint!r} uses the reserved fresh endpoint namespace")
    return validate_endpoint(endpoint)


def validate_endpoint(endpoint: object) -> str:
    if isinstance(endpoint, str) and endpoint.startswith(_FRESH_PREFIX):
        try:
            encoded = endpoint[len(_FRESH_PREFIX):]
            values = json.loads(base64.urlsafe_b64decode(
                encoded + "=" * (-len(encoded) % 4)).decode())
            if (isinstance(values, list) and len(values) == 2
                    and all(isinstance(value, str) for value in values)
                    and fresh_endpoint(*values) == endpoint):
                return endpoint
        except (ValueError, UnicodeDecodeError, binascii.Error):
            pass
    if not isinstance(endpoint, str) or not _ENDPOINT.fullmatch(endpoint):
        raise ValueError(f"Invalid endpoint identifier {endpoint!r}")
    return endpoint


def _control_names(stmt) -> set[str]:
    names = set()
    if isinstance(stmt, (IfRecvStmt, WhileRecvStmt)):
        binding = stmt.bindings[0]
        assert isinstance(binding, VarExpr)
        names.add(binding.var.name)
    for child in _children(stmt):
        names.update(_control_names(child))
    return names


@dataclass
class Session:
    name: str
    residual: LocalStmt
    bindings: dict[str, str] = field(default_factory=dict)
    visits: dict[str, int] = field(default_factory=dict)
    steps: int = 0
    waiting: dict[str, dict] = field(default_factory=dict)
    role: str | None = None
    role_endpoints: dict[str, str] = field(default_factory=dict)
    template: str | None = None
    pending_descriptors: dict[str, dict] = field(default_factory=dict)


class CellEnv(MutableMapping):
    def __init__(self, root, cells, bindings):
        self.root, self.cells, self.bindings = root, cells, bindings

    def _store(self, key):
        owner = self.bindings.get(key, "0")
        if owner == "0":
            return self.root
        if owner not in self.cells:
            raise RuntimeError(f"Missing cell allocation for session {owner!r}")
        return self.cells[owner]

    def __getitem__(self, key):
        return self._store(key)[key]

    def __setitem__(self, key, value):
        self._store(key)[key] = value

    def __delitem__(self, key):
        del self._store(key)[key]

    def __iter__(self):
        candidates = set(self.root) | set(self.bindings)
        return iter(key for key in candidates if key in self._store(key))

    def __len__(self):
        return sum(1 for _ in self)


class ScopedChannel:
    """Logical routes retain role names; delivery uses this session's endpoints."""
    session_steps = True

    def __init__(self, channel, state, session, root_env, human_handler=None):
        self.channel, self.state, self.session = channel, state, session
        self.root_env = root_env
        self.human_handler = human_handler

    def route(self, channel):
        return channel if self.session.name == "0" else f"session:{self.session.name}|{channel}"

    def put(self, sender, receiver, channel, *args, **kwargs):
        return self.channel.put(self.session.role_endpoints.get(sender, sender),
                                self.session.role_endpoints.get(receiver, receiver),
                                self.route(f"{sender}>{receiver}|{channel}")
                                if self.state.bound else self.route(channel), *args, **kwargs)

    def try_get(self, sender, receiver, channel):
        return self.channel.try_get(self.session.role_endpoints.get(sender, sender),
                                    self.session.role_endpoints.get(receiver, receiver),
                                    self.route(f"{sender}>{receiver}|{channel}")
                                    if self.state.bound else self.route(channel))

    def get(self, sender, receiver, channel, **kwargs):
        return self.channel.get(self.session.role_endpoints.get(sender, sender),
                                self.session.role_endpoints.get(receiver, receiver),
                                self.route(f"{sender}>{receiver}|{channel}")
                                if self.state.bound else self.route(channel), **kwargs)

    def _record(self, descriptor):
        conn = getattr(self.channel, "conn", None)
        if conn is not None:
            from zippergen.store import record_spawn_instance
            record_spawn_instance(conn, descriptor["child"], descriptor)
        else:
            self.channel.register_spawn_instance(descriptor)

    def _activate(self, descriptor, role):
        conn = getattr(self.channel, "conn", None)
        if conn is not None:
            from zippergen.store import record_spawn_activation
            first = record_spawn_activation(conn, descriptor["child"], role, self.state.endpoint)
            self.state.activations[(descriptor["child"], role)] = descriptor
            return first
        key = (descriptor["child"], role)
        old = self.state.activations.get(key)
        if old is not None:
            if old != descriptor:
                raise ValueError(f"Conflicting activation for {key!r}")
            return False
        self.state.activations[key] = descriptor
        return True

    def allocate(self, stmt):
        parent = self.session
        visit = parent.visits.get(stmt.site, 0) + 1
        child = f"{parent.name}/{stmt.site}:{visit}"
        retained = {role.name for role in stmt.retain}
        role_endpoints = {
            role.name: (parent.role_endpoints[role.name] if role.name in retained
                        else fresh_endpoint(child, role.name))
            for role in stmt.interface
        }
        if len(set(role_endpoints.values())) != len(role_endpoints):
            raise ValueError("Fresh endpoint allocation collided with a retained address")
        descriptor = {"child": child, "parent": parent.name, "site": stmt.site,
                      "template": stmt.template, "owner": stmt.owner.name,
                      "bindings": role_endpoints,
                      "retain": sorted(retained)}
        self._record(descriptor)
        self.state.creations[child] = descriptor
        parent.visits[stmt.site] = visit
        return descriptor

    def _current_descriptor(self, site):
        visit = self.session.visits.get(site, 0)
        if not visit:
            raise RuntimeError(f"No selected spawn at site {site!r}")
        child = f"{self.session.name}/{site}:{visit}"
        descriptor = self.state.creations.get(child) or self.session.pending_descriptors.get(site)
        if descriptor is None or descriptor["child"] != child:
            raise RuntimeError(f"Missing recorded descriptor for {child!r}")
        return descriptor

    def send_init(self, stmt):
        descriptor = self._current_descriptor(stmt.site)
        recipient = stmt.recipient.name
        endpoint = descriptor["bindings"][recipient]
        if stmt.kind == "desc":
            sender = self.state.endpoint
            channel = f"init:desc:{self.session.name}:{stmt.site}:{stmt.owner.name}:{recipient}"
        else:
            sender = self.state.endpoint
            channel = f"init:invite:{descriptor['child']}:{recipient}"
        self.channel.put(sender, endpoint, channel, (descriptor,))
        return descriptor

    def receive_init(self, stmt):
        parent = self.session
        channel = f"init:desc:{parent.name}:{stmt.site}:{stmt.owner.name}:{stmt.role.name}"
        item = self.channel.try_get(parent.role_endpoints[stmt.owner.name],
                                    self.state.endpoint, channel)
        if item is None:
            return None
        descriptor = item[1][0]
        expected = f"{parent.name}/{stmt.site}:{parent.visits.get(stmt.site, 0) + 1}"
        self.state.validate_descriptor(descriptor, stmt.template, stmt.role.name,
                                       expected_child=expected, retained=True)
        if descriptor.get("parent") != parent.name or descriptor.get("site") != stmt.site \
                or descriptor.get("owner") != stmt.owner.name:
            raise ValueError("Retained descriptor names a different parent spawn")
        for role in descriptor["retain"]:
            if descriptor["bindings"][role] != parent.role_endpoints.get(role):
                raise ValueError(f"Retained role {role!r} changed endpoint")
        conn = getattr(self.channel, "conn", None)
        if conn is not None:
            from zippergen.store import load_spawn_instance
            if load_spawn_instance(conn, descriptor["child"]) != descriptor:
                raise ValueError("Retained descriptor conflicts with creation record")
        parent.pending_descriptors[stmt.site] = descriptor
        parent.visits[stmt.site] = parent.visits.get(stmt.site, 0) + 1
        return descriptor

    def fork_bound(self, stmt):
        descriptor = self._current_descriptor(stmt.site)
        self.state.validate_descriptor(descriptor, stmt.template, stmt.role.name,
                                       retained=True)
        child = descriptor["child"]
        if not self._activate(descriptor, stmt.role.name):
            return child
        parent = self.session
        template = self.state.templates[(stmt.site, stmt.role.name)]
        body = template.program
        env = CellEnv(self.root_env, self.state.cells, parent.bindings)
        values = copy.deepcopy({name: env[name] for name in stmt.fresh if name in env})
        bindings = dict(parent.bindings)
        for name in (*stmt.fresh, *_control_names(body)):
            bindings[name] = child
        self.state.cells[child] = values
        self.state.sessions[child] = Session(
            child, body, bindings, role=stmt.role.name,
            role_endpoints=dict(descriptor["bindings"]), template=stmt.template,
        )
        parent.pending_descriptors.pop(stmt.site, None)
        return child

    def effect_ready(self, node):
        ready = getattr(self.channel, "effect_ready", None)
        return True if ready is None else ready(node, self.session.name)

    def fork(self, stmt):
        parent = self.session
        visit = parent.visits.get(stmt.site, 0) + 1
        child = f"{parent.name}/{stmt.site}:{visit}"
        if child in self.state.sessions:
            raise RuntimeError("Duplicate local child activation")
        template = self.state.templates.get((stmt.site, parent.role)) if self.state.bound else None
        body = template.program if template is not None else stmt.body
        env = CellEnv(self.root_env, self.state.cells, parent.bindings)
        # Take every source before installing any target. Missing is distinct
        # from None, and copying preserves no mutable reference to the parent.
        values = copy.deepcopy({name: env[name] for name in stmt.fresh if name in env})
        bindings = dict(parent.bindings)
        for name in (*stmt.fresh, *_control_names(body)):
            bindings[name] = child
        self.state.cells[child] = values
        parent.visits[stmt.site] = visit
        self.state.sessions[child] = Session(
            child, body, bindings, role=parent.role,
            role_endpoints=dict(parent.role_endpoints),
            template=template.template if template is not None else None,
        )
        return child


@dataclass
class SessionState:
    sessions: dict[str, Session]
    cells: dict[str, dict] = field(default_factory=dict)
    cursor: int = 0
    pending_session: str | None = None
    dirty: bool = False
    endpoint: str | None = None
    templates: dict = field(default_factory=dict, repr=False)
    creations: dict[str, dict] = field(default_factory=dict)
    activations: dict[tuple[str, str], dict] = field(default_factory=dict)
    bound: bool = False
    keep_alive: bool = False
    invite_turn: bool = True

    @classmethod
    def start(cls, root, *, role=None, endpoint=None, templates=None,
              root_bindings=None, keep_alive=False):
        bound = templates is not None
        sessions = {} if role is None else {
            "0": Session("0", root, role=role,
                         role_endpoints=dict(root_bindings or {role: endpoint or role}))
        }
        if role is None and not bound:
            sessions = {"0": Session("0", root)}
        return cls(sessions, endpoint=endpoint, templates=templates or {},
                   bound=bound, keep_alive=keep_alive)

    @property
    def done(self):
        try:
            sessions = tuple(self.sessions.values())
        except RuntimeError:
            # An endpoint thread installed a child during this diagnostic read.
            return False
        return all(isinstance(s.residual, EmptyStmt) for s in sessions)

    def validate_descriptor(self, descriptor, template, role, *, expected_child=None,
                            retained=False):
        if not isinstance(descriptor, dict) or descriptor.get("template") != template:
            raise ValueError("Missing or incompatible bound-spawn template")
        child = descriptor.get("child")
        if expected_child is not None and child != expected_child:
            raise ValueError(f"Unexpected child visit: {child!r}; expected {expected_child!r}")
        binding = descriptor.get("bindings")
        if not isinstance(binding, dict) or binding.get(role) != self.endpoint:
            raise ValueError("Bound-spawn descriptor address does not match this endpoint")
        if (descriptor.get("site"), role) not in self.templates:
            raise ValueError(f"Missing installed template for spawn site {descriptor.get('site')!r}, role {role!r}")
        installed = self.templates[(descriptor["site"], role)]
        if installed.template != template:
            raise ValueError("Incompatible installed bound-spawn template")
        expected_roles = {member.name for member in installed.interface}
        if set(binding) != expected_roles or len(set(binding.values())) != len(binding):
            raise ValueError("Bound-spawn descriptor has invalid role binding")
        if set(descriptor.get("retain", ())) != {member.name for member in installed.retain}:
            raise ValueError("Bound-spawn descriptor has invalid retained roles")
        if retained != (role in descriptor.get("retain", ())):
            raise ValueError("Bound-spawn retention disagrees with descriptor")
        for member, endpoint in binding.items():
            validate_endpoint(endpoint)
            if member not in descriptor["retain"] and endpoint != fresh_endpoint(child, member):
                raise ValueError("Fresh role has an inconsistent endpoint identity")

    def accept_invitation(self, channel):
        if not self.bound or self.endpoint is None:
            return False
        getter = getattr(channel, "try_get_invitation", None)
        if getter is None:
            return False
        item = getter(self.endpoint)
        if item is None:
            return False
        descriptor = item[1][0]
        if not isinstance(descriptor, dict):
            raise ValueError("Invalid spawn invitation payload")
        binding = descriptor.get("bindings", {})
        roles = [role for role, endpoint in binding.items()
                 if endpoint == self.endpoint]
        if len(roles) != 1:
            raise ValueError("Invitation does not identify one role at this endpoint")
        role = roles[0]
        self.validate_descriptor(descriptor, descriptor.get("template"), role,
                                 retained=False)
        child = descriptor["child"]
        key = (child, role)
        existing = self.activations.get(key)
        if existing is not None:
            if existing != descriptor:
                raise ValueError(f"Conflicting repeated invitation for {child!r}")
            complete = getattr(channel, "complete_invitation", None)
            if complete is not None:
                complete()
            return True
        conn = getattr(channel, "conn", None)
        if conn is not None:
            from zippergen.store import load_spawn_instance, record_spawn_activation
            canonical = load_spawn_instance(conn, child)
            if canonical != descriptor:
                raise ValueError(f"Invitation conflicts with creation record for {child!r}")
            first = record_spawn_activation(conn, child, role, self.endpoint)
            if not first:
                self.activations[key] = descriptor
                return True
        template = self.templates[(descriptor["site"], role)]
        from zippergen.syntax import occurring_variable_declarations
        names = {var.name for var in occurring_variable_declarations(template.program)}
        names.update(_control_names(template.program))
        self.cells[child] = {}
        self.sessions[child] = Session(
            child, template.program, {name: child for name in names},
            role=role, role_endpoints=dict(binding), template=descriptor["template"],
        )
        self.activations[key] = descriptor
        complete = getattr(channel, "complete_invitation", None)
        if complete is not None:
            complete()
        return True

    def collect(self):
        # Descendants can keep cells after their allocating session finishes.
        # A finished session can never fork again, so its visit counters need
        # not survive collection. Existing child names retain the parent's ID.
        for name, session in list(self.sessions.items()):
            if name != "0" and isinstance(session.residual, EmptyStmt):
                del self.sessions[name]
        live = {owner for s in self.sessions.values() for owner in s.bindings.values()}
        self.cells = {owner: values for owner, values in self.cells.items() if owner in live}

    def step(self, env, channel, ns, llm_backend, human_backend, trace, stop,
             *, durable=False, assistant_backend=None, resolved=None, human_handler=None,
             monitor=None, formula_conditions=None
             ) -> tuple[SessionState | EmptyStmt | PendingExternal, bool]:
        from zippergen.runtime import _step, PendingExternal
        if monitor is not None or formula_conditions:
            raise ValueError("CPL monitoring is not supported in spawn sessions")
        if resolved is None and self.invite_turn and self.accept_invitation(channel):
            self.invite_turn = False
            return self, True
        names = list(self.sessions)
        if resolved is not None and self.pending_session is not None:
            names = [self.pending_session]
        else:
            if names:
                cut = self.cursor % len(names)
                names = names[cut:] + names[:cut]
        self.dirty = False
        for name in names:
            session = self.sessions[name]
            if isinstance(session.residual, EmptyStmt):
                continue
            local_env = CellEnv(env, self.cells, session.bindings)
            scoped = ScopedChannel(channel, self, session, env, human_handler)
            def session_trace(event):
                if trace:
                    event = {**event, "session": name}
                    if self.bound:
                        event["endpoint"] = self.endpoint
                        event.setdefault("role", session.role)
                    if "channel" in event:
                        event["channel"] = scoped.route(event["channel"])
                    trace(event)
            result, progressed = _step(
                session.residual, local_env, scoped, ns, llm_backend, human_backend,
                None, session_trace if trace else None, {}, stop, durable=durable,
                assistant_backend=assistant_backend, resolved=resolved,
            )
            if isinstance(result, PendingExternal):
                self.pending_session = name
                start = None if result.trace_start is None else {**result.trace_start, "session": name}
                return replace(result, session=name, trace_start=start), False
            if progressed or self.dirty:
                session.residual = result
                session.steps += 1
                self.pending_session = None
                all_names = list(self.sessions)
                next_index = (all_names.index(name) + 1) % len(all_names)
                next_name = all_names[next_index]
                self.collect()
                self.cursor = list(self.sessions).index(next_name) if next_name in self.sessions else 0
                self.invite_turn = True
                return (EmptyStmt() if self.done and not self.keep_alive else self), True
        if self.accept_invitation(channel):
            self.invite_turn = False
            return self, True
        return (EmptyStmt() if self.done and not self.keep_alive else self), False

    def encode(self, root):
        from zippergen.control import encode_control
        if self.bound:
            return {"k": "sessions", "v": 2, "cursor": self.cursor,
                    "endpoint": self.endpoint, "bound": True,
                    "invite_turn": self.invite_turn,
                    "cells": encode_value(self.cells),
                    "creations": list(self.creations.values()),
                    "activations": [
                        {"child": child, "role": role, "descriptor": descriptor}
                        for (child, role), descriptor in self.activations.items()],
                    "sessions": [
                        {"name": s.name, "role": s.role, "template": s.template,
                         "control": encode_control(
                             root if s.name == "0" else
                             self.templates[(self._site_from_session(s.name), s.role)].program,
                             s.residual),
                         "bindings": s.bindings, "role_endpoints": s.role_endpoints,
                         "visits": s.visits, "steps": s.steps,
                         "pending_descriptors": s.pending_descriptors,
                         "waiting": encode_value(s.waiting)} for s in self.sessions.values()]}
        return {"k": "sessions", "v": 1, "cursor": self.cursor,
                "cells": encode_value(self.cells),
                "sessions": [
                    {"name": s.name, "control": encode_control(root, s.residual),
                     "bindings": s.bindings, "visits": s.visits, "steps": s.steps,
                     "waiting": encode_value(s.waiting)} for s in self.sessions.values()]}

    @staticmethod
    def _site_from_session(name):
        return name.rsplit("/", 1)[1].rsplit(":", 1)[0]

    @classmethod
    def decode(cls, root, data, *, templates=None, endpoint=None,
               root_bindings=None, keep_alive=False):
        from zippergen.control import decode_control, ControlError
        if data.get("v") == 2:
            if templates is None or endpoint != data.get("endpoint"):
                raise ControlError("Bound session state requires the same endpoint and templates")
            sessions = {}
            for entry in data["sessions"]:
                name, role = entry["name"], entry["role"]
                if name in sessions:
                    raise ControlError("Duplicate persisted session")
                token = entry["template"]
                if name == "0":
                    program = root
                else:
                    template = templates.get((cls._site_from_session(name), role))
                    if template is None or template.template != token:
                        raise ControlError(f"Missing or incompatible installed template for {name!r}")
                    program = template.program
                residual = decode_control(program, entry["control"])
                waiting = decode_value(entry["waiting"])
                if isinstance(residual, SessionState) or not isinstance(waiting, dict):
                    raise ControlError("Invalid nested session control")
                sessions[name] = Session(
                    name, residual, dict(entry["bindings"]), dict(entry["visits"]),
                    entry["steps"], cast(dict[str, dict], waiting), role,
                    dict(entry["role_endpoints"]), token,
                    dict(entry.get("pending_descriptors", {})),
                )
            cells = decode_value(data["cells"])
            if not isinstance(cells, dict):
                raise ControlError("Invalid bound session cells")
            if any(owner != "0" and owner not in cells
                   for session in sessions.values()
                   for owner in session.bindings.values()):
                raise ControlError("Missing persisted bound session cells")
            activations = {(entry["child"], entry["role"]): entry["descriptor"]
                           for entry in data.get("activations", ())}
            creations = {d["child"]: d for d in data.get("creations", ())}
            return cls(sessions, cells, data["cursor"], endpoint=endpoint,
                       templates=templates,
                       creations=creations, activations=activations,
                       bound=True, keep_alive=keep_alive,
                       invite_turn=bool(data.get("invite_turn", True)))
        if data.get("v") != 1:
            raise ControlError("Unsupported spawn session-state version")
        sessions = {}
        for entry in data["sessions"]:
            name = entry["name"]
            if name in sessions:
                raise ControlError("Duplicate persisted session")
            residual = decode_control(root, entry["control"])
            waiting = decode_value(entry["waiting"])
            if isinstance(residual, SessionState) or not isinstance(waiting, dict):
                raise ControlError("Invalid nested session control")
            sessions[name] = Session(name, residual, entry["bindings"], entry["visits"],
                                     entry["steps"], cast(dict[str, dict], waiting))
        if "0" not in sessions:
            raise ControlError("Missing root session")
        cells = decode_value(data["cells"])
        if not isinstance(cells, dict) or any(not isinstance(v, dict) for v in cells.values()):
            raise ControlError("Invalid persisted session cells")
        if any(owner != "0" and owner not in cells for s in sessions.values()
               for owner in s.bindings.values()):
            raise ControlError("Missing persisted session cells")
        return cls(sessions, cast(dict[str, dict], cells), data["cursor"])

    def frontier(self, root):
        from zippergen.control import frontier_paths
        return [path for s in self.sessions.values()
                for path in frontier_paths(
                    root if s.name == "0" or not self.bound else
                    self.templates[(self._site_from_session(s.name), s.role)].program,
                    s.residual)]
