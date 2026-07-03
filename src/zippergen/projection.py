"""
Layer 4: Projection engine. π_A(P) — given a global Workflow and a lifeline A,
returns the LocalStmt A must execute. See paper Tables for the projection rules.
"""

from __future__ import annotations

from typing import cast

from zippergen.syntax import (
    EmptyStmt, MsgStmt, CoregionStmt, ActStmt, SkipStmt, SeqStmt, IfStmt, WhileStmt,
    ParallelStmt,
    SendStmt, RecvStmt, ReceiveAnyStmt, SelfAssignStmt, IfRecvStmt, WhileRecvStmt,
    ParallelLocalStmt,
    Lifeline, LocalStmt, AnyStmt, Var, VarExpr, LitExpr,
    make_kappa_ctrl, canonical_construct_key, participation_set, seq,
    Workflow,
)

__all__ = ["project"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _receivers(p_left: AnyStmt, p_right: AnyStmt, owner: Lifeline) -> list[Lifeline]:
    """
    Compute R = (L(p_left) ∪ L(p_right)) - {owner}, sorted by name (the ≺ order).
    """
    combined = participation_set(p_left) | participation_set(p_right)
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


def _fresh_ctrl(counter: list[int]) -> Var:
    """Allocate a fresh Bool variable _ctrl1, _ctrl2, … for a receive-guard."""
    counter[0] += 1
    return Var(f"_ctrl{counter[0]}", bool)


def _parallel_channel(stmt: ParallelStmt, branch_index: int, parent_channel: str) -> str:
    """Return the private FIFO channel namespace for one parallel branch.

    Keyed on the region's content (canonical_construct_key), not id(stmt), so the
    branch's sender and receiver — which project in separate processes — agree on
    the same channel name."""
    return f"{parent_channel}/par-{canonical_construct_key(stmt)}-{branch_index + 1}"


# ---------------------------------------------------------------------------
# Core projection — structural recursion on Stmt
# ---------------------------------------------------------------------------

def _project(stmt: AnyStmt, A: Lifeline, counter: list[int], channel: str = "main") -> LocalStmt:
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

        # coregion { msg X_i(xs_i) → Y(ys_i) }_i
        case CoregionStmt(messages=messages):
            receiver = messages[0].receiver
            if A == receiver:
                return ReceiveAnyStmt(
                    lifeline=A,
                    receives=tuple((msg.sender, msg.bindings) for msg in messages),
                    channel=channel,
                )
            return cast(LocalStmt, seq(*(_project(msg, A, counter, channel) for msg in messages)))

        # act X(ys) := f(xs)
        case ActStmt(lifeline=X):
            return stmt if A == X else EmptyStmt()

        # skip_X
        case SkipStmt(lifeline=X):
            return stmt if A == X else EmptyStmt()

        # P1 ; P2
        case SeqStmt(first=p1, second=p2):
            return cast(LocalStmt, seq(_project(p1, A, counter, channel), _project(p2, A, counter, channel)))

        # parallel { P_i }_i
        case ParallelStmt(branches=branches):
            local_branches: list[LocalStmt] = []
            branch_indices: list[int] = []
            for i, branch in enumerate(branches):
                if A not in participation_set(branch):
                    continue
                branch_channel = _parallel_channel(stmt, i, channel)
                local_branches.append(_project(branch, A, counter, branch_channel))
                branch_indices.append(i)
            if not local_branches:
                return EmptyStmt()
            return ParallelLocalStmt(tuple(local_branches), tuple(branch_indices))

        # if c@B then P_⊤ else P_⊥
        case IfStmt(condition=c, owner=B, branch_true=p_true, branch_false=p_false):
            r_if = _receivers(p_true, p_false, B)
            tag = make_kappa_ctrl(canonical_construct_key(stmt))   # κ_ctrl^P: keyed on construct content

            if A == B:
                # Owner: prepend control broadcasts to each branch, then recurse.
                return IfStmt(
                    condition=c,
                    owner=B,
                    branch_true=seq(
                        *_ctrl_sends(B, True,  r_if, tag, channel),
                        _project(p_true,  B, counter, channel),
                    ),
                    branch_false=seq(
                        *_ctrl_sends(B, False, r_if, tag, channel),
                        _project(p_false, B, counter, channel),
                    ),
                )
            elif A in frozenset(r_if):
                # Receiver: wait for B's decision, branch accordingly.
                ctrl = _fresh_ctrl(counter)
                return IfRecvStmt(
                    lifeline=A,
                    bindings=(VarExpr(ctrl), tag),
                    sender=B,
                    branch_true=_project(p_true,  A, counter, channel),
                    branch_false=_project(p_false, A, counter, channel),
                    channel=channel,
                )
            else:
                return EmptyStmt()

        # while c@B do P_body exit P_exit
        case WhileStmt(condition=c, owner=B, body=p_body, exit_body=p_exit):
            r_while = _receivers(p_body, p_exit, B)
            tag = make_kappa_ctrl(canonical_construct_key(stmt))   # κ_ctrl^P: keyed on construct content

            if A == B:
                # Owner: prepend control broadcasts, then recurse.
                return WhileStmt(
                    condition=c,
                    owner=B,
                    body=seq(
                        *_ctrl_sends(B, True,  r_while, tag, channel),
                        _project(p_body, B, counter, channel),
                    ),
                    exit_body=seq(
                        *_ctrl_sends(B, False, r_while, tag, channel),
                        _project(p_exit, B, counter, channel),
                    ),
                )
            elif A in frozenset(r_while):
                # Receiver: loop decision comes from B each iteration.
                ctrl = _fresh_ctrl(counter)
                return WhileRecvStmt(
                    lifeline=A,
                    bindings=(VarExpr(ctrl), tag),
                    sender=B,
                    body=_project(p_body, A, counter, channel),
                    exit_body=_project(p_exit, A, counter, channel),
                    channel=channel,
                )
            else:
                return EmptyStmt()

        case _:
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
    return _project(wf.body, lifeline, [0])
