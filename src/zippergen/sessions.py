"""Experimental local session scheduler for detached global spawn.

One instance belongs to one physical participant. Static code is shared, while
residuals, visit counters, and generated control variables belong to a session.
Bindings name participant-local cells. Only selected cells are copied at a fork.
The complete state is encoded in the existing role-state transaction.
"""
from __future__ import annotations

import copy
from collections.abc import MutableMapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, cast

from zippergen.locator import _children
from zippergen.syntax import EmptyStmt, ForkStmt, IfRecvStmt, WhileRecvStmt, LocalStmt, VarExpr
from zippergen.value_codec import encode_value, decode_value

if TYPE_CHECKING:
    from zippergen.runtime import PendingExternal


def has_forks(stmt) -> bool:
    return isinstance(stmt, ForkStmt) or any(has_forks(c) for c in _children(stmt))


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
    """Physical endpoints stay fixed. Only the logical route changes."""
    session_steps = True

    def __init__(self, channel, state, session, root_env, human_handler=None):
        self.channel, self.state, self.session = channel, state, session
        self.root_env = root_env
        self.human_handler = human_handler

    def route(self, channel):
        return channel if self.session.name == "0" else f"session:{self.session.name}|{channel}"

    def put(self, sender, receiver, channel, *args, **kwargs):
        return self.channel.put(sender, receiver, self.route(channel), *args, **kwargs)

    def try_get(self, sender, receiver, channel):
        return self.channel.try_get(sender, receiver, self.route(channel))

    def get(self, sender, receiver, channel, **kwargs):
        return self.channel.get(sender, receiver, self.route(channel), **kwargs)

    def fork(self, stmt):
        parent = self.session
        visit = parent.visits.get(stmt.site, 0) + 1
        child = f"{parent.name}/{stmt.site}:{visit}"
        if child in self.state.sessions:
            raise RuntimeError("Duplicate local child activation")
        env = CellEnv(self.root_env, self.state.cells, parent.bindings)
        # Take every source before installing any target. Missing is distinct
        # from None, and copying preserves no mutable reference to the parent.
        values = copy.deepcopy({name: env[name] for name in stmt.fresh if name in env})
        bindings = dict(parent.bindings)
        for name in (*stmt.fresh, *_control_names(stmt.body)):
            bindings[name] = child
        self.state.cells[child] = values
        parent.visits[stmt.site] = visit
        self.state.sessions[child] = Session(child, stmt.body, bindings)
        return child


@dataclass
class SessionState:
    sessions: dict[str, Session]
    cells: dict[str, dict] = field(default_factory=dict)
    cursor: int = 0
    pending_session: str | None = None
    dirty: bool = False

    @classmethod
    def start(cls, root):
        return cls({"0": Session("0", root)})

    @property
    def done(self):
        return all(isinstance(s.residual, EmptyStmt) for s in self.sessions.values())

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
        names = list(self.sessions)
        if resolved is not None and self.pending_session is not None:
            names = [self.pending_session]
        else:
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
                return (EmptyStmt() if self.done else self), True
        return (EmptyStmt() if self.done else self), False

    def encode(self, root):
        from zippergen.control import encode_control
        return {"k": "sessions", "v": 1, "cursor": self.cursor,
                "cells": encode_value(self.cells),
                "sessions": [
                    {"name": s.name, "control": encode_control(root, s.residual),
                     "bindings": s.bindings, "visits": s.visits, "steps": s.steps,
                     "waiting": encode_value(s.waiting)} for s in self.sessions.values()]}

    @classmethod
    def decode(cls, root, data):
        from zippergen.control import decode_control, ControlError
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
        return [path for s in self.sessions.values() for path in frontier_paths(root, s.residual)]
