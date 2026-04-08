"""Tests for Layer 3: @workflow builder and DSL syntax.

Each test verifies that the @workflow decorator produces the expected IR
from a given DSL pattern.
"""

import pytest

from zippergen.syntax import (
    EmptyStmt, MsgStmt, ActStmt, SkipStmt, SeqStmt, IfStmt, WhileStmt,
    SendStmt, RecvStmt,
    Lifeline, Var, VarExpr, LitExpr,
    Workflow, participation_set,
)
from zippergen.actions import pure, llm
from zippergen.builder import workflow


# ---------------------------------------------------------------------------
# Lifelines used across tests
# ---------------------------------------------------------------------------

P = Lifeline("P")
Q = Lifeline("Q")
R = Lifeline("R")


# ---------------------------------------------------------------------------
# @pure and @llm produce correct IR nodes
# ---------------------------------------------------------------------------

def test_pure_produces_pure_action():
    from zippergen.syntax import PureAction
    @pure
    def add_one(x: int) -> int:
        return x + 1
    assert isinstance(add_one, PureAction)
    assert add_one.name == "add_one"
    assert add_one.inputs == (("x", int),)
    assert add_one.outputs == (("add_one", int),)


def test_llm_produces_llm_action():
    from zippergen.syntax import LLMAction
    @llm(system="sys", user="user: {x}", parse="json", outputs=(("result", str),))
    def my_action(x: str) -> None: ...
    assert isinstance(my_action, LLMAction)
    assert my_action.name == "my_action"
    assert my_action.inputs == (("x", str),)
    assert my_action.outputs == (("result", str),)
    assert my_action.system_prompt == "sys"
    assert my_action.parse_format == "json"


def test_pure_missing_annotation_raises():
    with pytest.raises(TypeError):
        @pure
        def bad(x) -> int:
            return x


def test_pure_bad_return_type_raises():
    with pytest.raises(TypeError):
        @pure
        def bad(x: int) -> list:
            return [x]


def test_pure_union_return_type_raises():
    with pytest.raises(TypeError):
        @pure
        def bad(x: str) -> bool | None:
            return None


# ---------------------------------------------------------------------------
# @workflow produces a Workflow node
# ---------------------------------------------------------------------------

@pure
def _echo(v: int) -> int:
    return v


@workflow
def simple_send(n: int @ P) -> int:
    P(n) >> Q(n)
    return n @ Q


def test_workflow_is_workflow_node():
    assert isinstance(simple_send, Workflow)
    assert simple_send.name == "simple_send"


def test_workflow_inputs_parsed():
    assert len(simple_send.inputs) == 1
    name, ztype, lifeline = simple_send.inputs[0]
    assert name == "n"
    assert ztype is int
    assert lifeline == P


def test_workflow_output_type():
    assert simple_send.output_type is int


def test_workflow_output_var_and_lifeline():
    assert simple_send.output_var is not None
    assert simple_send.output_var.name == "n"
    assert simple_send.output_lifeline == Q


# ---------------------------------------------------------------------------
# DSL syntax → IR: message passing ( >> )
# ---------------------------------------------------------------------------

def test_arrow_produces_msg_in_body():
    body = simple_send.body
    # Simple send: P(n) >> Q(n) — should be a MsgStmt (or first of a SeqStmt)
    def _find_msg(stmt):
        if isinstance(stmt, MsgStmt):
            return stmt
        if isinstance(stmt, SeqStmt):
            return _find_msg(stmt.first) or _find_msg(stmt.second)
        return None

    msg = _find_msg(body)
    assert msg is not None
    assert msg.sender == P
    assert msg.receiver == Q


# ---------------------------------------------------------------------------
# DSL syntax → IR: with block → sequence of ActStmts
# ---------------------------------------------------------------------------

@workflow
def two_acts(n: int @ P) -> int:
    with P:
        n = _echo(n)
        n = _echo(n)
    return n @ P


def test_with_block_produces_act_stmts():
    def _collect_acts(stmt):
        acts = []
        if isinstance(stmt, ActStmt):
            acts.append(stmt)
        elif isinstance(stmt, SeqStmt):
            acts.extend(_collect_acts(stmt.first))
            acts.extend(_collect_acts(stmt.second))
        return acts

    acts = _collect_acts(two_acts.body)
    assert len(acts) == 2
    for a in acts:
        assert isinstance(a, ActStmt)
        assert a.lifeline == P


# ---------------------------------------------------------------------------
# DSL syntax → IR: if/else
# ---------------------------------------------------------------------------

@workflow
def conditional(n: int @ P) -> int:
    if (n > 0) @ P:
        P(n) >> Q(n)
    else:
        pass
    return n @ P


def test_if_produces_if_stmt():
    def _find_if(stmt):
        if isinstance(stmt, IfStmt):
            return stmt
        if isinstance(stmt, SeqStmt):
            return _find_if(stmt.first) or _find_if(stmt.second)
        return None

    if_stmt = _find_if(conditional.body)
    assert if_stmt is not None
    assert if_stmt.owner == P


def test_if_true_branch_has_msg():
    def _find_if(stmt):
        if isinstance(stmt, IfStmt):
            return stmt
        if isinstance(stmt, SeqStmt):
            return _find_if(stmt.first) or _find_if(stmt.second)
        return None

    if_stmt = _find_if(conditional.body)
    assert participation_set(if_stmt.branch_true) == frozenset({P, Q})


def test_if_false_branch_is_empty():
    def _find_if(stmt):
        if isinstance(stmt, IfStmt):
            return stmt
        if isinstance(stmt, SeqStmt):
            return _find_if(stmt.first) or _find_if(stmt.second)
        return None

    if_stmt = _find_if(conditional.body)
    assert if_stmt.branch_false == EmptyStmt()


# ---------------------------------------------------------------------------
# DSL syntax → IR: while / else (exit body)
# ---------------------------------------------------------------------------

@pure
def _dec(v: int) -> int:
    return v - 1


@workflow
def countdown(n: int @ P) -> int:
    while (n != 0) @ P:
        with P:
            n = _dec(n)
    else:
        pass
    return n @ P


def test_while_produces_while_stmt():
    def _find_while(stmt):
        if isinstance(stmt, WhileStmt):
            return stmt
        if isinstance(stmt, SeqStmt):
            return _find_while(stmt.first) or _find_while(stmt.second)
        return None

    while_stmt = _find_while(countdown.body)
    assert while_stmt is not None
    assert while_stmt.owner == P


def test_while_body_has_act():
    def _find_while(stmt):
        if isinstance(stmt, WhileStmt):
            return stmt
        if isinstance(stmt, SeqStmt):
            return _find_while(stmt.first) or _find_while(stmt.second)
        return None

    while_stmt = _find_while(countdown.body)
    assert isinstance(while_stmt.body, ActStmt)
    assert while_stmt.body.lifeline == P


# ---------------------------------------------------------------------------
# Participation set computed correctly from built workflow
# ---------------------------------------------------------------------------

@workflow
def three_party(n: int @ P) -> int:
    P(n) >> Q(n)
    Q(n) >> R(n)
    return n @ R


def test_participation_set_from_built_workflow():
    ps = participation_set(three_party.body)
    assert ps == frozenset({P, Q, R})


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_workflow_missing_annotation_raises():
    with pytest.raises(TypeError):
        @workflow
        def bad(n) -> int:
            P(n) >> Q(n)
            return n @ Q


def test_workflow_bad_return_type_raises():
    with pytest.raises(TypeError):
        @workflow
        def bad(n: int @ P) -> list:
            return n @ P
