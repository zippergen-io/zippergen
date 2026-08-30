"""
Layer 4: Projection engine. π_A(P) — given a global Workflow and a lifeline A,
returns the LocalStmt A must execute. See paper Tables for the projection rules.
"""

from __future__ import annotations

from typing import cast

from zippergen.syntax import (
    EmptyStmt, MsgStmt, ActStmt, SkipStmt, SeqStmt, IfStmt, WhileStmt,
    ParallelStmt,
    SendStmt, RecvStmt, SelfAssignStmt, IfRecvStmt, WhileRecvStmt,
    ParallelLocalStmt,
    Lifeline, LocalStmt, AnyStmt, Var, VarExpr, LitExpr,
    make_kappa_ctrl, canonical_construct_key, participation_set,
    occurring_variables, seq,
    Workflow,
)

__all__ = ["project"]


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

    def __init__(self, occupied: frozenset[str]) -> None:
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

def _project(stmt: AnyStmt, A: Lifeline, context: _Context, channel: str = "main") -> LocalStmt:
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
            return stmt if A == X else EmptyStmt()

        # skip_X
        case SkipStmt(lifeline=X):
            return stmt if A == X else EmptyStmt()

        # P1 ; P2
        case SeqStmt(first=p1, second=p2):
            return cast(LocalStmt, seq(_project(p1, A, context, channel), _project(p2, A, context, channel)))

        # parallel { P_i }_i
        case ParallelStmt(branches=branches):
            local_branches: list[LocalStmt] = []
            branch_indices: list[int] = []
            for i, branch in enumerate(branches):
                if A not in context.participants(branch):
                    continue
                branch_channel = _parallel_channel(stmt, i, channel)
                local_branches.append(_project(branch, A, context, branch_channel))
                branch_indices.append(i)
            if not local_branches:
                return EmptyStmt()
            return ParallelLocalStmt(tuple(local_branches), tuple(branch_indices))

        # if c@B then P_⊤ else P_⊥
        case IfStmt(condition=c, owner=B, branch_true=p_true, branch_false=p_false):
            r_if = _receivers(p_true, p_false, B, context)
            tag = make_kappa_ctrl(canonical_construct_key(stmt))   # κ_ctrl^P: keyed on construct content

            if A == B:
                # Owner: prepend control broadcasts to each branch, then recurse.
                return IfStmt(
                    condition=c,
                    owner=B,
                    branch_true=seq(
                        *_ctrl_sends(B, True,  r_if, tag, channel),
                        _project(p_true,  B, context, channel),
                    ),
                    branch_false=seq(
                        *_ctrl_sends(B, False, r_if, tag, channel),
                        _project(p_false, B, context, channel),
                    ),
                )
            elif A in frozenset(r_if):
                # Receiver: wait for B's decision, branch accordingly.
                ctrl = context.fresh()
                return IfRecvStmt(
                    lifeline=A,
                    bindings=(VarExpr(ctrl), tag),
                    sender=B,
                    branch_true=_project(p_true,  A, context, channel),
                    branch_false=_project(p_false, A, context, channel),
                    channel=channel,
                )
            else:
                return EmptyStmt()

        # while c@B do P_body exit P_exit
        case WhileStmt(condition=c, owner=B, body=p_body, exit_body=p_exit):
            r_while = _receivers(p_body, p_exit, B, context)
            tag = make_kappa_ctrl(canonical_construct_key(stmt))   # κ_ctrl^P: keyed on construct content

            if A == B:
                # Owner: prepend control broadcasts, then recurse.
                return WhileStmt(
                    condition=c,
                    owner=B,
                    body=seq(
                        *_ctrl_sends(B, True,  r_while, tag, channel),
                        _project(p_body, B, context, channel),
                    ),
                    exit_body=seq(
                        *_ctrl_sends(B, False, r_while, tag, channel),
                        _project(p_exit, B, context, channel),
                    ),
                )
            elif A in frozenset(r_while):
                # Receiver: loop decision comes from B each iteration.
                ctrl = context.fresh()
                return WhileRecvStmt(
                    lifeline=A,
                    bindings=(VarExpr(ctrl), tag),
                    sender=B,
                    body=_project(p_body, A, context, channel),
                    exit_body=_project(p_exit, A, context, channel),
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
    return _project(wf.body, lifeline, _Context(frozenset(occupied)))
