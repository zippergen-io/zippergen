"""Tests for Layer 1: IR nodes, participation_set, and seq."""

import pytest

from zippergen.syntax import (
    EmptyStmt, MsgStmt, ActStmt, SkipStmt, SeqStmt, IfStmt, WhileStmt,
    SendStmt, RecvStmt, IfRecvStmt, WhileRecvStmt,
    Lifeline, Var, VarExpr, LitExpr,
    participation_set, seq,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

A = Lifeline("A")
B = Lifeline("B")
C = Lifeline("C")

x = Var("x", int)
y = Var("y", str)


# ---------------------------------------------------------------------------
# Lifeline and Var equality (frozen dataclasses)
# ---------------------------------------------------------------------------

def test_lifeline_equality():
    assert Lifeline("A") == Lifeline("A")
    assert Lifeline("A") != Lifeline("B")


def test_lifeline_hashable():
    s = {Lifeline("A"), Lifeline("A"), Lifeline("B")}
    assert len(s) == 2


def test_var_equality():
    assert Var("x", int) == Var("x", int)
    assert Var("x", int) != Var("x", str)
    assert Var("x", int) != Var("y", int)


# ---------------------------------------------------------------------------
# seq — right-associative fold with EmptyStmt identity
# ---------------------------------------------------------------------------

def test_seq_no_args():
    assert seq() == EmptyStmt()


def test_seq_single():
    s = SkipStmt(A)
    assert seq(s) is s


def test_seq_two():
    s1 = SkipStmt(A)
    s2 = SkipStmt(B)
    result = seq(s1, s2)
    assert isinstance(result, SeqStmt)
    assert result.first is s1
    assert result.second is s2


def test_seq_three_is_right_associative():
    s1 = SkipStmt(A)
    s2 = SkipStmt(B)
    s3 = SkipStmt(C)
    result = seq(s1, s2, s3)
    assert isinstance(result, SeqStmt)
    assert result.first is s1
    assert isinstance(result.second, SeqStmt)
    assert result.second.first is s2
    assert result.second.second is s3


def test_seq_drops_empty_left():
    s = SkipStmt(A)
    assert seq(EmptyStmt(), s) is s


def test_seq_drops_empty_right():
    s = SkipStmt(A)
    assert seq(s, EmptyStmt()) is s


def test_seq_all_empty():
    assert seq(EmptyStmt(), EmptyStmt()) == EmptyStmt()


# ---------------------------------------------------------------------------
# participation_set — L(P)
# ---------------------------------------------------------------------------

def test_participation_empty():
    assert participation_set(EmptyStmt()) == frozenset()


def test_participation_msg():
    stmt = MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),))
    assert participation_set(stmt) == frozenset({A, B})


def test_participation_act():
    from zippergen.actions import pure
    @pure
    def f(v: int) -> int:
        return v
    stmt = ActStmt(A, f, (VarExpr(x),), (x,))
    assert participation_set(stmt) == frozenset({A})


def test_participation_skip():
    assert participation_set(SkipStmt(A)) == frozenset({A})


def test_participation_seq():
    s1 = SkipStmt(A)
    s2 = SkipStmt(B)
    assert participation_set(SeqStmt(s1, s2)) == frozenset({A, B})


def test_participation_if():
    stmt = IfStmt(
        condition=lambda _e: True,
        owner=A,
        branch_true=SkipStmt(B),
        branch_false=SkipStmt(C),
    )
    assert participation_set(stmt) == frozenset({A, B, C})


def test_participation_if_owner_only():
    """Owner is always included even if branches are empty."""
    stmt = IfStmt(
        condition=lambda _e: True,
        owner=A,
        branch_true=EmptyStmt(),
        branch_false=EmptyStmt(),
    )
    assert participation_set(stmt) == frozenset({A})


def test_participation_while():
    stmt = WhileStmt(
        condition=lambda _e: False,
        owner=A,
        body=SkipStmt(B),
        exit_body=SkipStmt(C),
    )
    assert participation_set(stmt) == frozenset({A, B, C})


def test_participation_send():
    stmt = SendStmt(A, (VarExpr(x),), B)
    assert participation_set(stmt) == frozenset({A})


def test_participation_recv():
    stmt = RecvStmt(A, (VarExpr(y),), B)
    assert participation_set(stmt) == frozenset({A})
