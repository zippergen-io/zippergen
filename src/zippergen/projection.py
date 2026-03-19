"""
ZipperGen — Layer 4: Projection engine.

Design notes
------------
**What this module is.**
This module implements the syntax-directed projection  π_A(P)  from the paper
(Tables tab:projection-base, tab:projection-if, tab:projection-while).
Given a global Proc and a lifeline A, it returns the local program (a
LocalStmt) that A must execute.

**The three cases for if/while.**
Every if/while has an owner B (the decider).  For each other lifeline A:
  - A = B      → owner case: B broadcasts the branch decision to all
                 participants, then executes its own projected branch.
  - A ∈ R      → receiver case: A waits for B's decision and branches
                 accordingly (IfRecvStmt / WhileRecvStmt).
  - A ∉ R ∪{B} → bystander: A does not participate at all → ε.

R_if    = (L(P_top) ∪ L(P_bot)) - {B}
R_while = (L(P_body) ∪ L(P_exit)) - {B}

**Control broadcasts.**
The owner sends  (⊤/⊥, κ_ctrl)  to every C ∈ R in a fixed total order ≺
(here: alphabetical by lifeline name).  κ_ctrl is the reserved literal that
tags these sends so receivers can filter them from data messages.

**Fresh control variables.**
Each IfRecvStmt / WhileRecvStmt on a non-owner lifeline needs a fresh local
Bool variable to bind the received branch decision.  A single-element list
[counter] is threaded through the recursion to generate unique names
_ctrl1, _ctrl2, …  These names are local to each lifeline's program and
never clash with user-declared variables (which must not start with _ctrl).

**Reuse of global nodes.**
ActStmt and SkipStmt nodes for the owning lifeline are returned unchanged
(frozen dataclasses are safe to share).  SeqStmt nodes are reconstructed
via seq() so that ε-elimination is applied automatically.
"""

from __future__ import annotations

from zippergen.syntax import (
    EmptyStmt, MsgStmt, ActStmt, SkipStmt, SeqStmt, IfStmt, WhileStmt,
    SendStmt, RecvStmt, IfRecvStmt, WhileRecvStmt,
    Lifeline, LocalStmt, Stmt, Var, VarExpr, LitExpr,
    kappa_ctrl, participation_set, seq,
    Proc,
)

__all__ = ["project"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _receivers(p_left: Stmt, p_right: Stmt, owner: Lifeline) -> list[Lifeline]:
    """
    Compute R = (L(p_left) ∪ L(p_right)) - {owner}, sorted by name (the ≺ order).
    """
    combined = participation_set(p_left) | participation_set(p_right)
    return sorted(combined - {owner}, key=lambda l: l.name)


def _ctrl_sends(owner: Lifeline, value: bool, receivers: list[Lifeline]) -> list[SendStmt]:
    """
    Generate  send owner(⊤/⊥, κ_ctrl) → C  for each C in receivers.
    These are the control-broadcast sends prepended by the owner in each branch.
    """
    lit = LitExpr(value, bool)
    return [SendStmt(owner, (lit, kappa_ctrl), C) for C in receivers]


def _fresh_ctrl(counter: list[int]) -> Var:
    """Allocate a fresh Bool variable _ctrl1, _ctrl2, … for a receive-guard."""
    counter[0] += 1
    return Var(f"_ctrl{counter[0]}", bool)


# ---------------------------------------------------------------------------
# Core projection — structural recursion on Stmt
# ---------------------------------------------------------------------------

def _project(stmt: Stmt, A: Lifeline, counter: list[int]) -> LocalStmt:
    """π_A(stmt) — one step of the structural recursion."""

    match stmt:

        # ε
        case EmptyStmt():
            return EmptyStmt()

        # msg X(x⃗) → Y(y⃗)
        case MsgStmt(sender=X, payload=xs, receiver=Y, bindings=ys):
            if A == X:
                return SendStmt(A, xs, Y)
            elif A == Y:
                return RecvStmt(A, ys, X)
            else:
                return EmptyStmt()

        # act X(y⃗) := f(x⃗)
        case ActStmt(lifeline=X):
            return stmt if A == X else EmptyStmt()

        # skip_X
        case SkipStmt(lifeline=X):
            return stmt if A == X else EmptyStmt()

        # P1 ; P2
        case SeqStmt(first=p1, second=p2):
            return seq(_project(p1, A, counter), _project(p2, A, counter))

        # if c@B then P_⊤ else P_⊥
        case IfStmt(condition=c, owner=B, branch_true=p_true, branch_false=p_false):
            r_if = _receivers(p_true, p_false, B)

            if A == B:
                # Owner: prepend control broadcasts to each branch, then recurse.
                return IfStmt(
                    condition=c,
                    owner=B,
                    branch_true=seq(
                        *_ctrl_sends(B, True,  r_if),
                        _project(p_true,  B, counter),
                    ),
                    branch_false=seq(
                        *_ctrl_sends(B, False, r_if),
                        _project(p_false, B, counter),
                    ),
                )
            elif A in frozenset(r_if):
                # Receiver: wait for B's decision, branch accordingly.
                ctrl = _fresh_ctrl(counter)
                return IfRecvStmt(
                    lifeline=A,
                    bindings=(VarExpr(ctrl), kappa_ctrl),
                    sender=B,
                    branch_true=_project(p_true,  A, counter),
                    branch_false=_project(p_false, A, counter),
                )
            else:
                return EmptyStmt()

        # while c@B do P_body exit P_exit
        case WhileStmt(condition=c, owner=B, body=p_body, exit_body=p_exit):
            r_while = _receivers(p_body, p_exit, B)

            if A == B:
                # Owner: prepend control broadcasts, then recurse.
                return WhileStmt(
                    condition=c,
                    owner=B,
                    body=seq(
                        *_ctrl_sends(B, True,  r_while),
                        _project(p_body, B, counter),
                    ),
                    exit_body=seq(
                        *_ctrl_sends(B, False, r_while),
                        _project(p_exit, B, counter),
                    ),
                )
            elif A in frozenset(r_while):
                # Receiver: loop decision comes from B each iteration.
                ctrl = _fresh_ctrl(counter)
                return WhileRecvStmt(
                    lifeline=A,
                    bindings=(VarExpr(ctrl), kappa_ctrl),
                    sender=B,
                    body=_project(p_body, A, counter),
                    exit_body=_project(p_exit, A, counter),
                )
            else:
                return EmptyStmt()

        case _:
            raise TypeError(f"Unknown statement type: {type(stmt).__name__}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def project(proc: Proc, lifeline: Lifeline) -> LocalStmt:
    """
    Project a global Proc onto a single lifeline.

    Returns the local program (a LocalStmt) that `lifeline` must execute.
    The result is a faithful implementation of  π_lifeline(proc.body)
    as defined in the paper.
    """
    return _project(proc.body, lifeline, [0])
