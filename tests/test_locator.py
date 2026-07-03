from zippergen.syntax import (
    SeqStmt, WhileStmt, EmptyStmt, SendStmt, Lifeline, VarExpr, Var, seq,
)
from zippergen.locator import loop_node_paths, resolve_path

A = Lifeline("A"); B = Lifeline("B")
x = Var("x", int)


def _while(body):
    return WhileStmt(condition=lambda _e: True, owner=A, body=body, exit_body=EmptyStmt())


def test_whole_program_is_loop_has_empty_path():
    w = _while(SendStmt(A, (VarExpr(x),), B))
    paths = loop_node_paths(w)
    assert paths == {id(w): []}
    assert resolve_path(w, []) is w


def test_prefix_then_loop_path_is_index_one():
    w = _while(SendStmt(A, (VarExpr(x),), B))
    prog = seq(SendStmt(A, (VarExpr(x),), B), w)   # SeqStmt(first=send, second=while)
    paths = loop_node_paths(prog)
    assert paths[id(w)] == [1]
    assert resolve_path(prog, [1]) is w


def test_resolve_out_of_range_returns_none():
    w = _while(SendStmt(A, (VarExpr(x),), B))
    assert resolve_path(w, [5]) is None
    assert resolve_path(w, [0, 0, 0]) is None
