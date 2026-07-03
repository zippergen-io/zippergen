"""Locate a loop node in a projected local program by a child-index path.

Used by snapshots: the residual at a loop-iteration boundary is (by identity)
one of these nodes, and the path re-finds it in a freshly-projected program so
the (unserializable) continuation never has to be persisted.
"""
from __future__ import annotations

from zippergen.syntax import (
    SeqStmt, IfStmt, IfRecvStmt, WhileStmt, WhileRecvStmt,
)


def _children(node) -> list:
    # Canonical, stable child ordering per node type. Leaf nodes have no children.
    match node:
        case SeqStmt(first=a, second=b):
            return [a, b]
        case IfStmt(branch_true=t, branch_false=f):
            return [t, f]
        case IfRecvStmt(branch_true=t, branch_false=f):
            return [t, f]
        case WhileStmt(body=b, exit_body=x):
            return [b, x]
        case WhileRecvStmt(body=b, exit_body=x):
            return [b, x]
        case _:
            return []


def loop_node_paths(root) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}

    def walk(node, path: list[int]) -> None:
        if isinstance(node, (WhileStmt, WhileRecvStmt)):
            out[id(node)] = list(path)
        for i, child in enumerate(_children(node)):
            walk(child, path + [i])

    walk(root, [])
    return out


def resolve_path(root, path: list[int]):
    node = root
    for i in path:
        children = _children(node)
        if not isinstance(i, int) or i < 0 or i >= len(children):
            return None
        node = children[i]
    return node
