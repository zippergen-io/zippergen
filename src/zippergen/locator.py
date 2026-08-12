"""Locate nodes in a projected local program by child-index paths.

A path is the stable name of a statement: walk the canonical child ordering
from the root. Durable control state is built from these paths, so a role's
position survives a restart by being re-resolved against a freshly projected
program. See ``control.py`` for the control language itself.
"""
from __future__ import annotations

from zippergen.syntax import (
    SeqStmt, IfStmt, IfRecvStmt, WhileStmt, WhileRecvStmt,
    ParallelLocalStmt,
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
        case ParallelLocalStmt(branches=branches):
            return list(branches)
        case _:
            return []


def resolve_path(root, path: list[int]):
    node = root
    for i in path:
        children = _children(node)
        if not isinstance(i, int) or i < 0 or i >= len(children):
            return None
        node = children[i]
    return node


def statement_node_paths(root) -> dict[int, list[int]]:
    """Map every node identity in a projected program to its stable path."""

    out: dict[int, list[int]] = {}

    def walk(node, path: list[int]) -> None:
        out[id(node)] = list(path)
        for i, child in enumerate(_children(node)):
            walk(child, path + [i])

    walk(root, [])
    return out
