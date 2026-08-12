"""Paths name statements, and durable control state is built from paths."""

from zippergen.syntax import (
    SeqStmt, WhileStmt, EmptyStmt, SendStmt, Lifeline, VarExpr, Var, seq,
    ActStmt, ParallelLocalStmt,
)
from zippergen.locator import resolve_path, statement_node_paths
from zippergen.control import frontier_paths
from zippergen.actions import pure

A = Lifeline("A"); B = Lifeline("B")
x = Var("x", int)


def _while(body):
    return WhileStmt(condition=lambda _e: True, owner=A, body=body, exit_body=EmptyStmt())


def test_the_root_has_the_empty_path():
    w = _while(SendStmt(A, (VarExpr(x),), B))
    assert statement_node_paths(w)[id(w)] == []
    assert resolve_path(w, []) is w


def test_a_statement_after_a_prefix_is_index_one():
    w = _while(SendStmt(A, (VarExpr(x),), B))
    prog = seq(SendStmt(A, (VarExpr(x),), B), w)
    assert statement_node_paths(prog)[id(w)] == [1]
    assert resolve_path(prog, [1]) is w


def test_resolve_out_of_range_returns_none():
    w = _while(SendStmt(A, (VarExpr(x),), B))
    assert resolve_path(w, [5]) is None
    assert resolve_path(w, [0, 0, 0]) is None


@pure
def f(x: int) -> int:
    return x


def _act():
    y = Var("x", int)
    return ActStmt(A, f, (VarExpr(y),), (y,))


def test_every_path_resolves_back_to_the_same_object():
    """Identity is what lets a decoded control state re-find real nodes."""

    a1, a2 = _act(), _act()
    root = SeqStmt(a1, SeqStmt(a2, ParallelLocalStmt((_act(),), (0,))))
    for node_id, path in statement_node_paths(root).items():
        assert id(resolve_path(root, path)) == node_id


def test_frontier_covers_every_parallel_branch():
    first = SendStmt(A, (VarExpr(x),), B)
    second = SendStmt(A, (VarExpr(x),), B, channel="other")
    root = ParallelLocalStmt((first, second), (0, 1))

    paths = statement_node_paths(root)
    assert paths[id(first)] == [0]
    assert paths[id(second)] == [1]
    assert frontier_paths(root, root) == [[0], [1]]


def test_frontier_descends_through_a_residual_sequence():
    first, second = _act(), _act()
    root = SeqStmt(first, second)
    residual = SeqStmt(EmptyStmt(), second)

    assert frontier_paths(root, residual) == [[1]]
