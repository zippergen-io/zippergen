"""Locate nodes in a projected local program by child-index paths.

Used by snapshots: the residual at a loop-iteration boundary is (by identity)
one of these nodes, and the path re-finds it in a freshly-projected program so
the (unserializable) continuation never has to be persisted.

The same stable paths also support execution observation.  Observation remains
separate from snapshots: a current-statement pointer is useful diagnostic state,
not a recovery boundary.
"""
from __future__ import annotations

from zippergen.syntax import (
    SeqStmt, IfStmt, IfRecvStmt, WhileStmt, WhileRecvStmt,
    ActStmt, EmptyStmt, ParallelLocalStmt, ReceiveAnyStmt,
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


def action_node_paths(root) -> dict[int, list[int]]:
    """Map id(node) -> child-index path for every act and owner if/while node.

    These are the statements whose non-deterministic result is journaled; the
    path re-finds the node in a freshly-projected program (same trick as
    loop_node_paths). Leaf/owner identity survives _step, so id() is a stable key
    for the journal locator."""
    out: dict[int, list[int]] = {}

    def walk(node, path: list[int]) -> None:
        if isinstance(node, (ActStmt, IfStmt, WhileStmt)):
            out[id(node)] = list(path)
        for i, child in enumerate(_children(node)):
            walk(child, path + [i])

    walk(root, [])
    return out


def statement_node_paths(root) -> dict[int, list[int]]:
    """Map every node identity in a projected program to its stable path."""

    out: dict[int, list[int]] = {}

    def walk(node, path: list[int]) -> None:
        out[id(node)] = list(path)
        for i, child in enumerate(_children(node)):
            walk(child, path + [i])

    walk(root, [])
    return out


def _receive_any_matches(current: ReceiveAnyStmt, original: object) -> bool:
    """Recognize a residual receive-any after some senders were consumed."""

    if not isinstance(original, ReceiveAnyStmt):
        return False
    if (
        current.lifeline != original.lifeline
        or current.channel != original.channel
    ):
        return False
    current_senders = {sender.name for sender, _bindings in current.receives}
    original_senders = {sender.name for sender, _bindings in original.receives}
    return bool(current_senders) and current_senders <= original_senders


def execution_frontier_paths(root, residual) -> list[list[int]]:
    """Return stable paths for the residual's currently executable leaves.

    Sequential residuals have one frontier.  A local parallel region can have
    one frontier per unfinished branch.  Most residual leaves retain their
    identity from ``root``; ``ReceiveAnyStmt`` is the exception because the
    runtime creates a smaller residual after each received message.
    """

    paths = statement_node_paths(root)
    by_path: list[tuple[list[int], object]] = []

    def collect(node, path: list[int]) -> None:
        by_path.append((path, node))
        for i, child in enumerate(_children(node)):
            collect(child, path + [i])

    collect(root, [])

    def locate(node) -> list[int] | None:
        direct = paths.get(id(node))
        if direct is not None:
            return list(direct)
        if isinstance(node, ReceiveAnyStmt):
            matches = [
                path
                for path, original in by_path
                if _receive_any_matches(node, original)
            ]
            if len(matches) == 1:
                return list(matches[0])
        return None

    def frontier(node) -> list[list[int]]:
        if isinstance(node, EmptyStmt):
            return []
        if isinstance(node, SeqStmt):
            first = frontier(node.first)
            return first if first else frontier(node.second)
        if isinstance(node, ParallelLocalStmt):
            out: list[list[int]] = []
            for branch in node.branches:
                out.extend(frontier(branch))
            return out
        path = locate(node)
        return [path] if path is not None else []

    return frontier(residual)
