"""Tests for Layer 4: projection engine (π_A).

IR nodes are constructed directly — no builder — so these tests are
independent of Layer 3 and test projection semantics in isolation.
"""

import pytest

from zippergen.syntax import (
    EmptyStmt, MsgStmt, ActStmt, SkipStmt, SeqStmt, IfStmt, WhileStmt,
    SendStmt, RecvStmt, IfRecvStmt, WhileRecvStmt,
    Lifeline, Var, VarExpr, LitExpr,
    Workflow, seq, kappa_ctrl,
)
from zippergen.actions import pure
from zippergen.projection import project


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

A = Lifeline("A")
B = Lifeline("B")
C = Lifeline("C")

x = Var("x", int)
y = Var("y", int)
z = Var("z", int)


def _make_workflow(body, name="test") -> Workflow:
    return Workflow(
        name=name,
        inputs=(),
        output_type=str,
        vars=(),
        body=body,
        ns={},
    )


# ---------------------------------------------------------------------------
# Base cases
# ---------------------------------------------------------------------------

def test_project_empty():
    wf = _make_workflow(EmptyStmt())
    assert project(wf, A) == EmptyStmt()


def test_project_msg_sender():
    stmt = MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),))
    wf = _make_workflow(stmt)
    result = project(wf, A)
    assert isinstance(result, SendStmt)
    assert result.lifeline == A
    assert result.receiver == B
    assert result.payload == (VarExpr(x),)


def test_project_msg_receiver():
    stmt = MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),))
    wf = _make_workflow(stmt)
    result = project(wf, B)
    assert isinstance(result, RecvStmt)
    assert result.lifeline == B
    assert result.sender == A
    assert result.bindings == (VarExpr(y),)


def test_project_msg_bystander():
    stmt = MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),))
    wf = _make_workflow(stmt)
    assert project(wf, C) == EmptyStmt()


def test_project_act_owner():
    @pure
    def f(v: int) -> int:
        return v
    stmt = ActStmt(A, f, (VarExpr(x),), (y,))
    wf = _make_workflow(stmt)
    result = project(wf, A)
    assert result is stmt  # frozen node reused unchanged


def test_project_act_nonowner():
    @pure
    def f(v: int) -> int:
        return v
    stmt = ActStmt(A, f, (VarExpr(x),), (y,))
    wf = _make_workflow(stmt)
    assert project(wf, B) == EmptyStmt()


def test_project_skip_owner():
    stmt = SkipStmt(A)
    wf = _make_workflow(stmt)
    assert project(wf, A) is stmt


def test_project_skip_nonowner():
    stmt = SkipStmt(A)
    wf = _make_workflow(stmt)
    assert project(wf, B) == EmptyStmt()


# ---------------------------------------------------------------------------
# Sequential composition
# ---------------------------------------------------------------------------

def test_project_seq_distributes():
    s1 = MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),))
    s2 = MsgStmt(B, (VarExpr(y),), A, (VarExpr(z),))
    wf = _make_workflow(SeqStmt(s1, s2))

    # A: send then recv
    result_A = project(wf, A)
    assert isinstance(result_A, SeqStmt)
    assert isinstance(result_A.first, SendStmt)
    assert isinstance(result_A.second, RecvStmt)

    # B: recv then send
    result_B = project(wf, B)
    assert isinstance(result_B, SeqStmt)
    assert isinstance(result_B.first, RecvStmt)
    assert isinstance(result_B.second, SendStmt)


def test_project_seq_epsilon_elimination():
    """If one side projects to ε, seq collapses to the other side."""
    s1 = MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),))
    s2 = SkipStmt(C)
    wf = _make_workflow(SeqStmt(s1, s2))

    # A is only in s1 — s2 projects to ε for A, result should be just SendStmt
    result = project(wf, A)
    assert isinstance(result, SendStmt)


# ---------------------------------------------------------------------------
# If — owner
# ---------------------------------------------------------------------------

def test_project_if_owner_gets_ifstmt():
    """The owner of an if receives an IfStmt with control broadcasts prepended."""
    cond = lambda _e: True
    body_true = MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),))
    body_false = SkipStmt(A)
    stmt = IfStmt(condition=cond, owner=A, branch_true=body_true, branch_false=body_false)
    wf = _make_workflow(stmt)

    result = project(wf, A)
    assert isinstance(result, IfStmt)
    assert result.owner == A
    assert result.condition is cond


def test_project_if_owner_broadcasts_to_receivers():
    """Owner's true/false branches start with a SendStmt(kappa_ctrl) to each receiver."""
    cond = lambda _e: True
    # B and C both appear in branches → both are receivers
    body_true  = MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),))
    body_false = SkipStmt(C)
    stmt = IfStmt(condition=cond, owner=A, branch_true=body_true, branch_false=body_false)
    wf = _make_workflow(stmt)

    result = project(wf, A)
    assert isinstance(result, IfStmt)

    # true branch: broadcast True to B and C (sorted), then projected body
    true_branch = result.branch_true
    # Unwrap the SeqStmt chain to find the first SendStmt
    def _collect_seq(s):
        stmts = []
        while isinstance(s, SeqStmt):
            stmts.append(s.first)
            s = s.second
        stmts.append(s)
        return stmts

    true_stmts = _collect_seq(true_branch)
    ctrl_sends = [s for s in true_stmts if isinstance(s, SendStmt) and kappa_ctrl in s.payload]
    assert len(ctrl_sends) == 2  # one for B, one for C
    receivers = {s.receiver for s in ctrl_sends}
    assert receivers == {B, C}


# ---------------------------------------------------------------------------
# If — receiver
# ---------------------------------------------------------------------------

def test_project_if_receiver_gets_ifrecvstmt():
    cond = lambda _e: True
    body_true  = MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),))
    body_false = SkipStmt(B)
    stmt = IfStmt(condition=cond, owner=A, branch_true=body_true, branch_false=body_false)
    wf = _make_workflow(stmt)

    result = project(wf, B)
    assert isinstance(result, IfRecvStmt)
    assert result.sender == A
    assert result.lifeline == B


def test_project_if_receiver_fresh_ctrl_var():
    cond = lambda _e: True
    body_true  = MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),))
    body_false = SkipStmt(B)
    stmt = IfStmt(condition=cond, owner=A, branch_true=body_true, branch_false=body_false)
    wf = _make_workflow(stmt)

    result = project(wf, B)
    assert isinstance(result, IfRecvStmt)
    ctrl_var_expr = result.bindings[0]
    assert isinstance(ctrl_var_expr, VarExpr)
    assert ctrl_var_expr.var.name.startswith("_ctrl")
    assert ctrl_var_expr.var.type is bool


def test_project_if_receiver_nested_fresh_ctrl_names_differ():
    """Two nested ifs produce two distinct _ctrl variables."""
    cond = lambda _e: True
    inner = IfStmt(
        condition=cond, owner=A,
        branch_true=SkipStmt(B),
        branch_false=SkipStmt(B),
    )
    outer = IfStmt(
        condition=cond, owner=A,
        branch_true=inner,
        branch_false=SkipStmt(B),
    )
    wf = _make_workflow(outer)
    result = project(wf, B)
    assert isinstance(result, IfRecvStmt)
    outer_ctrl = result.bindings[0].var.name
    inner_result = result.branch_true
    assert isinstance(inner_result, IfRecvStmt)
    inner_ctrl = inner_result.bindings[0].var.name
    assert outer_ctrl != inner_ctrl


# ---------------------------------------------------------------------------
# If — bystander
# ---------------------------------------------------------------------------

def test_project_if_bystander():
    cond = lambda _e: True
    body_true  = MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),))
    body_false = SkipStmt(A)
    stmt = IfStmt(condition=cond, owner=A, branch_true=body_true, branch_false=body_false)
    wf = _make_workflow(stmt)

    # C does not appear in any branch and is not the owner
    assert project(wf, C) == EmptyStmt()


# ---------------------------------------------------------------------------
# While — owner / receiver / bystander
# ---------------------------------------------------------------------------

def test_project_while_owner_gets_whilestmt():
    cond = lambda _e: False
    body = MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),))
    exit_body = SkipStmt(A)
    stmt = WhileStmt(condition=cond, owner=A, body=body, exit_body=exit_body)
    wf = _make_workflow(stmt)

    result = project(wf, A)
    assert isinstance(result, WhileStmt)
    assert result.owner == A


def test_project_while_receiver_gets_whilerecvstmt():
    cond = lambda _e: False
    body = MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),))
    exit_body = EmptyStmt()
    stmt = WhileStmt(condition=cond, owner=A, body=body, exit_body=exit_body)
    wf = _make_workflow(stmt)

    result = project(wf, B)
    assert isinstance(result, WhileRecvStmt)
    assert result.sender == A
    assert result.lifeline == B


def test_project_while_bystander():
    cond = lambda _e: False
    body = MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),))
    exit_body = EmptyStmt()
    stmt = WhileStmt(condition=cond, owner=A, body=body, exit_body=exit_body)
    wf = _make_workflow(stmt)

    assert project(wf, C) == EmptyStmt()
