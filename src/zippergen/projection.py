"""
Layer 4: Projection engine. π_A(P) — given a global Workflow and a lifeline A,
returns the LocalStmt A must execute. See paper Tables for the projection rules.
"""

from __future__ import annotations

from typing import cast
from dataclasses import replace, dataclass

from zippergen.syntax import (
    EmptyStmt, MsgStmt, ActStmt, SkipStmt, SeqStmt, IfStmt, WhileStmt,
    ParallelStmt, SpawnStmt, ForkStmt,
    BoundSelectStmt, BoundSendStmt, BoundReceiveStmt, BoundForkStmt,
    SendStmt, RecvStmt, SelfAssignStmt, IfRecvStmt, WhileRecvStmt,
    ParallelLocalStmt,
    Lifeline, LocalStmt, AnyStmt, Var, VarExpr, LitExpr,
    make_kappa_ctrl, canonical_construct_key, participation_set,
    occurring_variables, seq,
    Workflow, contains_spawn,
)

__all__ = ["project"]


@dataclass(frozen=True)
class SpawnTemplate:
    """Statically projected role code installed by a child creation site."""

    program: LocalStmt
    template: str
    interface: tuple[Lifeline, ...]
    retain: tuple[Lifeline, ...]
    recruited: bool
    fresh: tuple[str, ...]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _receivers(
    p_left: AnyStmt,
    p_right: AnyStmt,
    owner: Lifeline,
    context: "_Context",
) -> list[Lifeline]:
    """
    Compute R = (L(p_left) ∪ L(p_right)) - {owner}, sorted by name (the ≺ order).
    """
    combined = context.participants(p_left) | context.participants(p_right)
    return sorted(combined - {owner}, key=lambda l: l.name)


def _ctrl_sends(
    owner: Lifeline,
    value: bool,
    receivers: list[Lifeline],
    tag: LitExpr,
    channel: str,
) -> list[SendStmt]:
    """
    Generate  send owner(⊤/⊥, κ_ctrl^P) → C  for each C in receivers.
    tag is the per-construct control tag for this if/while construct.
    These are the control-broadcast sends prepended by the owner in each branch.
    """
    lit = LitExpr(value, bool)
    return [SendStmt(owner, (lit, tag), C, channel) for C in receivers]


class _Context:
    """The whole-program facts the recursion carries: fresh names, and who
    participates where.

    Both are properties of the entire workflow rather than of one step, so
    both are established once per projection and consulted, never recomputed.

    The paper requires each generated variable to be fresh -- not occurring in
    P. Freshness is a property of the whole program, not of the allocation
    order, so the occupied names are collected once up front and every
    generated name skips them. A counter alone is not enough: an author may
    write a variable called ``_ctrl1``, and binding the received branch
    decision over it changes the receiver's value with no error anywhere.

    Allocation stays deterministic, so every process projecting the same
    workflow generates the same names.
    """

    def __init__(self, occupied: frozenset[str], sessions: bool = False) -> None:
        self.sessions = sessions
        self._occupied = occupied
        self._issued = 0
        self._participation: dict[int, frozenset[Lifeline]] = {}

    def participants(self, stmt: AnyStmt) -> frozenset[Lifeline]:
        """L(stmt), computed once per node for this projection."""

        return participation_set(stmt, self._participation)

    def fresh(self) -> Var:
        while True:
            self._issued += 1
            name = f"_ctrl{self._issued}"
            if name not in self._occupied:
                return Var(name, bool)


def _parallel_channel(stmt: ParallelStmt, branch_index: int, parent_channel: str) -> str:
    """Return the private FIFO channel namespace for one parallel branch.

    Keyed on the region's content (canonical_construct_key), not id(stmt), so the
    branch's sender and receiver — which project in separate processes — agree on
    the same channel name."""
    return f"{parent_channel}/par-{canonical_construct_key(stmt)}-{branch_index + 1}"


# ---------------------------------------------------------------------------
# Core projection — structural recursion on Stmt
# ---------------------------------------------------------------------------

def _project(stmt: AnyStmt, A: Lifeline, context: _Context, channel: str = "main", site: str = "0") -> LocalStmt:
    """π_A(stmt) — one step of the structural recursion."""

    match stmt:

        # ε
        case EmptyStmt():
            return EmptyStmt()

        # msg X(xs) → Y(ys)
        case MsgStmt(sender=X, payload=xs, receiver=Y, bindings=ys):
            if X == Y:
                # Self-send: no channel needed — project as local assignment for X.
                return SelfAssignStmt(X, xs, ys) if A == X else EmptyStmt()
            elif A == X:
                return SendStmt(A, xs, Y, channel)
            elif A == Y:
                return RecvStmt(A, ys, X, channel)
            else:
                return EmptyStmt()

        # act X(ys) := f(xs)
        case ActStmt(lifeline=X):
            return (replace(stmt) if context.sessions else stmt) if A == X else EmptyStmt()

        # skip_X
        case SkipStmt(lifeline=X):
            return (replace(stmt) if context.sessions else stmt) if A == X else EmptyStmt()

        # P1 ; P2
        case SeqStmt(first=p1, second=p2):
            return cast(LocalStmt, seq(_project(p1, A, context, channel, site + ".0"), _project(p2, A, context, channel, site + ".1")))

        case SpawnStmt(body=body, fresh=fresh, bindings=bindings, owner=owner, retain=retain):
            interface = tuple(sorted(context.participants(body), key=lambda role: role.name))
            if bindings is None and A not in interface:
                return EmptyStmt()
            if bindings is not None and A != owner and A not in (retain or ()):
                return EmptyStmt()
            names = tuple(sorted(field.name for field in fresh if field.lifeline == A))
            # Guard-only fields and declared but currently unused scratch cells
            # are valid selections too. They need not occur in an action.
            if any(name not in context._occupied for name in names):
                raise ValueError(f"spawn fresh selects an undeclared field at {A.name}")
            if bindings is None:
                child = _project(body, A, context, "main", site + ".0")
                return ForkStmt(A, site, child, names)
            retained = tuple(sorted(retain or (), key=lambda role: role.name))
            template = f"bound-v1:{site}:{canonical_construct_key(stmt)}"
            if A == owner:
                instructions: list[LocalStmt] = [
                    BoundSelectStmt(owner, site, bindings, interface, retained, template),
                ]
                for role in sorted((set(interface) - ({owner} if owner in retained else set())),
                                   key=lambda role: role.name):
                    kind = "desc" if role in retained else "invite"
                    instructions.append(BoundSendStmt(owner, site, role, kind))
                if owner in retained:
                    child = _project(body, A, _Context(context._occupied, True),
                                     "main", site + ".0")
                    instructions.append(BoundForkStmt(A, site, child, names, template,
                                                      interface, retained))
                return cast(LocalStmt, seq(*instructions))
            child = _project(body, A, _Context(context._occupied, True),
                             "main", site + ".0")
            return cast(LocalStmt, seq(
                BoundReceiveStmt(A, site, owner, template),
                BoundForkStmt(A, site, child, names, template, interface, retained),
            ))

        # parallel { P_i }_i
        case ParallelStmt(branches=branches):
            local_branches: list[LocalStmt] = []
            branch_indices: list[int] = []
            for i, branch in enumerate(branches):
                if A not in context.participants(branch):
                    continue
                branch_channel = f"par-{site}-{i + 1}" if context.sessions else _parallel_channel(stmt, i, channel)
                local_branches.append(_project(branch, A, context, branch_channel, site + f".{i}"))
                branch_indices.append(i)
            if not local_branches:
                return EmptyStmt()
            return ParallelLocalStmt(tuple(local_branches), tuple(branch_indices))

        # if c@B then P_⊤ else P_⊥
        case IfStmt(condition=c, owner=B, branch_true=p_true, branch_false=p_false):
            r_if = _receivers(p_true, p_false, B, context)
            tag = make_kappa_ctrl(site if context.sessions else canonical_construct_key(stmt))

            if A == B:
                # Owner: prepend control broadcasts to each branch, then recurse.
                return IfStmt(
                    condition=c,
                    owner=B,
                    branch_true=seq(
                        *_ctrl_sends(B, True,  r_if, tag, channel),
                        _project(p_true, B, context, channel, site + ".0"),
                    ),
                    branch_false=seq(
                        *_ctrl_sends(B, False, r_if, tag, channel),
                        _project(p_false, B, context, channel, site + ".1"),
                    ),
                )
            elif A in frozenset(r_if):
                # Receiver: wait for B's decision, branch accordingly.
                ctrl = context.fresh()
                return IfRecvStmt(
                    lifeline=A,
                    bindings=(VarExpr(ctrl), tag),
                    sender=B,
                    branch_true=_project(p_true, A, context, channel, site + ".0"),
                    branch_false=_project(p_false, A, context, channel, site + ".1"),
                    channel=channel,
                )
            else:
                return EmptyStmt()

        # while c@B do P_body exit P_exit
        case WhileStmt(condition=c, owner=B, body=p_body, exit_body=p_exit):
            r_while = _receivers(p_body, p_exit, B, context)
            tag = make_kappa_ctrl(site if context.sessions else canonical_construct_key(stmt))

            if A == B:
                # Owner: prepend control broadcasts, then recurse.
                return WhileStmt(
                    condition=c,
                    owner=B,
                    body=seq(
                        *_ctrl_sends(B, True,  r_while, tag, channel),
                        _project(p_body, B, context, channel, site + ".0"),
                    ),
                    exit_body=seq(
                        *_ctrl_sends(B, False, r_while, tag, channel),
                        _project(p_exit, B, context, channel, site + ".1"),
                    ),
                )
            elif A in frozenset(r_while):
                # Receiver: loop decision comes from B each iteration.
                ctrl = context.fresh()
                return WhileRecvStmt(
                    lifeline=A,
                    bindings=(VarExpr(ctrl), tag),
                    sender=B,
                    body=_project(p_body, A, context, channel, site + ".0"),
                    exit_body=_project(p_exit, A, context, channel, site + ".1"),
                    channel=channel,
                )
            else:
                return EmptyStmt()

        case _:
            # Reached only when a global IR member gains no projection rule.
            # The type model cannot catch this: SeqStmt, IfStmt and WhileStmt
            # carry AnyStmt children because they are shared by global and
            # local programs, so `stmt` cannot be narrowed to Stmt without
            # making those nodes generic. `test_projection.py` closes the gap
            # by projecting every member of Stmt.
            raise TypeError(f"Unknown statement type: {type(stmt).__name__}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def project(wf: Workflow, lifeline: Lifeline) -> LocalStmt:
    """
    Project a global Workflow onto a single lifeline.

    Returns the local program (a LocalStmt) that `lifeline` must execute.
    The result is a faithful implementation of  π_lifeline(wf.body)
    as defined in the paper.
    """
    # Every name the author used, so no generated control variable can land on
    # one. Declared inputs and locals are included even when the body never
    # mentions them: they still occupy the lifeline's environment.
    occupied = (
        occurring_variables(wf.body)
        | {name for name, _type, _lifeline in wf.inputs}
        | {var.name for var in wf.vars}
        | {var.name for var, _lifeline in wf.outputs}
    )
    sessions = contains_spawn(wf.body)
    if sessions:
        # A declared field may occur only in a guard, including one inside a
        # fragment. It is still a valid copy-on-fork selection.
        occupied |= {value.name for value in wf.ns.values() if isinstance(value, Var)}
    local = _project(wf.body, lifeline, _Context(frozenset(occupied), sessions))
    if sessions:
        from zippergen.runtime import _collect_formula_guards
        guards, _ = _collect_formula_guards(wf.body, wf.ns)
        if guards:
            raise ValueError("Experimental spawn does not yet support CPL guards")
    return local


def project_templates(wf: Workflow) -> dict[tuple[str, str], SpawnTemplate]:
    """Compile every child role once, including roles recruited at nested sites.

    Keys are ``(static_spawn_site, role_name)``. Session instances reuse this
    code with immutable runtime binding descriptors; no endpoint-specific code
    is generated when an invitation arrives.
    """

    occupied = (
        occurring_variables(wf.body)
        | {name for name, _type, _lifeline in wf.inputs}
        | {var.name for var in wf.vars}
        | {var.name for var, _lifeline in wf.outputs}
        | {value.name for value in wf.ns.values() if isinstance(value, Var)}
    )
    registry: dict[tuple[str, str], SpawnTemplate] = {}

    def walk(node: AnyStmt, site: str) -> None:
        if isinstance(node, SpawnStmt):
            interface = tuple(sorted(participation_set(node.body), key=lambda role: role.name))
            retained = (interface if node.bindings is None else
                        tuple(sorted(node.retain or (), key=lambda role: role.name)))
            template = (f"bound-v1:{site}:{canonical_construct_key(node)}" if node.bindings is not None
                        else f"inherited-v1:{site}:{canonical_construct_key(node)}")
            for role in interface:
                program = _project(node.body, role, _Context(frozenset(occupied), True),
                                   "main", site + ".0")
                registry[(site, role.name)] = SpawnTemplate(
                    program, template, interface, retained, role not in retained,
                    tuple(sorted(field.name for field in node.fresh if field.lifeline == role)),
                )
            walk(node.body, site + ".0")
        elif isinstance(node, SeqStmt):
            walk(node.first, site + ".0")
            walk(node.second, site + ".1")
        elif isinstance(node, IfStmt):
            walk(node.branch_true, site + ".0")
            walk(node.branch_false, site + ".1")
        elif isinstance(node, WhileStmt):
            walk(node.body, site + ".0")
            walk(node.exit_body, site + ".1")
        elif isinstance(node, ParallelStmt):
            for index, branch in enumerate(node.branches):
                walk(branch, site + f".{index}")

    walk(wf.body, "0")
    return registry
