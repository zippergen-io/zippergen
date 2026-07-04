from zippergen.syntax import (
    SeqStmt, WhileStmt, EmptyStmt, SendStmt, Lifeline, VarExpr, Var, seq,
    ActStmt, ParallelLocalStmt, IfStmt,
)
from zippergen.locator import loop_node_paths, resolve_path, action_node_paths
from zippergen.actions import pure

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


# action_node_paths tests

A_action = Lifeline("A")

@pure
def f(x: int) -> int:
    return x

def _act():
    x = Var("x", int)
    return ActStmt(A_action, f, (VarExpr(x),), (x,))

def test_action_paths_resolve_back_to_same_nodes():
    a1, a2 = _act(), _act()
    root = SeqStmt(a1, SeqStmt(a2, ParallelLocalStmt((_act(),), (0,))))
    paths = action_node_paths(root)
    # every recorded id resolves back to the identical object via its path
    for node_id, path in paths.items():
        assert id(resolve_path(root, path)) == node_id
    # all three acts are indexed (incl. the one inside the parallel branch)
    assert sum(isinstance(resolve_path(root, p), ActStmt) for p in paths.values()) == 3

def test_action_paths_index_owner_loop():
    body = _act()
    root = WhileStmt(condition=lambda _e: True, owner=A_action, body=body, exit_body=_act())
    paths = action_node_paths(root)
    assert id(root) in paths                    # owner WhileStmt indexed (decision)
    assert id(body) in paths                    # act inside the loop body indexed
