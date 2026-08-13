"""Explicit, serializable control state for a projected local program.

A role's position in its program is an ordinary IR value: the residual the
interpreter has left to run. Most residual nodes are nodes of the static
program, so a child-index path names them exactly. The interpreter builds only
three shapes fresh, and each gets one constructor here:

    done                     nothing left to run
    at   [path]              run the static node at this path
    seq  a b                 run a, then b
    any  [path] [senders]    a coregion receive with some senders still pending
    par  [branches] [labels] a local parallel region, one control per branch

That is the whole control language. It is closed under one interpreter step,
which is exactly why storing the current state is sufficient and why nothing
has to be replayed. A role resumes by decoding this value against a freshly
projected program and continuing.

The encoding is small and readable on purpose. ``zg run inspect`` renders it,
and a person debugging a stuck deployment should be able to read it straight
out of SQLite without a tool.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias, cast

from zippergen.locator import _children, resolve_path, statement_node_paths
from zippergen.syntax import (
    AnyStmt,
    EmptyStmt,
    LitExpr,
    LocalStmt,
    ParallelLocalStmt,
    ReceiveAnyStmt,
    SeqStmt,
    VarExpr,
)

__all__ = [
    "ControlError",
    "PartialReceiveAny",
    "Residual",
    "decode_control",
    "encode_control",
    "frontier_paths",
    "program_fingerprint",
]


class ControlError(Exception):
    """Durable control state does not fit the program it was decoded against."""


@dataclass(frozen=True)
class PartialReceiveAny:
    """A coregion receive that has taken some of its messages already.

    The interpreter cannot simply shrink the static ``ReceiveAnyStmt``, because
    then the residual would no longer be nameable by a path. Instead it keeps a
    reference to the static node and the set of senders still outstanding, so
    the control state stays exactly representable.
    """

    origin: ReceiveAnyStmt
    remaining: tuple[str, ...]

    @property
    def receives(self) -> tuple:
        pending = set(self.remaining)
        return tuple(
            (sender, bindings)
            for sender, bindings in self.origin.receives
            if sender.name in pending
        )

    @property
    def lifeline(self):
        return self.origin.lifeline

    @property
    def channel(self) -> str:
        return self.origin.channel


# Projection produces ``LocalStmt``. The interpreter additionally creates a
# partial coregion receive while it consumes that static node one sender at a
# time. This is the complete residual language held in memory.
Residual: TypeAlias = LocalStmt | PartialReceiveAny


def encode_control(root: LocalStmt, residual: Residual) -> dict:
    """Represent a residual as plain JSON-safe data."""

    paths = statement_node_paths(root)

    def encode(node: Residual) -> dict:
        if isinstance(node, EmptyStmt):
            return {"k": "done"}
        path = paths.get(id(node))
        if path is not None:
            return {"k": "at", "p": list(path)}
        if isinstance(node, PartialReceiveAny):
            origin_path = paths.get(id(node.origin))
            if origin_path is None:
                raise ControlError(
                    "a coregion receive in progress is not part of this program"
                )
            return {
                "k": "any",
                "p": list(origin_path),
                "s": list(node.remaining),
            }
        if isinstance(node, SeqStmt):
            return {
                "k": "seq",
                "a": encode(cast(Residual, node.first)),
                "b": encode(cast(Residual, node.second)),
            }
        if isinstance(node, ParallelLocalStmt):
            labels = node.branch_indices or tuple(range(len(node.branches)))
            return {
                "k": "par",
                "b": [encode(branch) for branch in node.branches],
                "i": list(labels),
            }
        raise ControlError(
            f"cannot represent residual of type {type(node).__name__}"
        )

    return encode(residual)


def decode_control(root: LocalStmt, data: dict) -> Residual:
    """Rebuild a residual from encoded control state and a projected program."""

    def decode(node: object) -> Residual:
        if not isinstance(node, dict):
            raise ControlError(f"control state is not an object: {node!r}")
        kind = node.get("k")
        if kind == "done":
            return EmptyStmt()
        if kind == "at":
            target = resolve_path(root, list(node.get("p") or []))
            if target is None:
                raise ControlError(
                    f"control path {node.get('p')!r} does not exist in this program"
                )
            return target
        if kind == "any":
            origin = resolve_path(root, list(node.get("p") or []))
            if not isinstance(origin, ReceiveAnyStmt):
                raise ControlError(
                    f"control path {node.get('p')!r} is not a coregion receive"
                )
            remaining = tuple(str(name) for name in node.get("s") or ())
            known = {sender.name for sender, _bindings in origin.receives}
            if not remaining or not set(remaining) <= known:
                raise ControlError(
                    "pending coregion senders do not belong to this receive"
                )
            return PartialReceiveAny(origin, remaining)
        if kind == "seq":
            return SeqStmt(
                cast(AnyStmt, decode(node.get("a"))),
                cast(AnyStmt, decode(node.get("b"))),
            )
        if kind == "par":
            branches = tuple(decode(branch) for branch in node.get("b") or ())
            if not branches:
                raise ControlError("a parallel region needs at least one branch")
            labels = tuple(int(index) for index in node.get("i") or ())
            return ParallelLocalStmt(cast(tuple[LocalStmt, ...], branches), labels)
        raise ControlError(f"unknown control constructor {kind!r}")

    return decode(data)


def frontier_paths(root: LocalStmt, residual: Residual) -> list[list[int]]:
    """Paths of the leaves this residual would try to run next.

    Diagnostic only. Recovery uses the control state itself.
    """

    paths = statement_node_paths(root)

    def walk(node: Residual) -> list[list[int]]:
        if isinstance(node, EmptyStmt):
            return []
        if isinstance(node, SeqStmt):
            first = walk(cast(Residual, node.first))
            return first if first else walk(cast(Residual, node.second))
        if isinstance(node, ParallelLocalStmt):
            out: list[list[int]] = []
            for branch in node.branches:
                out.extend(walk(branch))
            return out
        if isinstance(node, PartialReceiveAny):
            path = paths.get(id(node.origin))
            return [list(path)] if path is not None else []
        path = paths.get(id(node))
        return [list(path)] if path is not None else []

    return walk(residual)


def _type_name(kind: object) -> str:
    return getattr(kind, "__name__", str(kind))


def _variable_shape(var: object) -> list[str]:
    return [
        str(getattr(var, "name", "?")),
        _type_name(getattr(var, "type", None)),
    ]


def _expression_shape(expr: object) -> list:
    """Describe the part of an expression that gives durable state meaning."""

    if isinstance(expr, VarExpr):
        return ["var", *_variable_shape(expr.var)]
    if isinstance(expr, LitExpr):
        return ["lit", type(expr.value).__name__, _plain(expr.value)]
    raise ControlError(f"cannot fingerprint expression {expr!r}")


def _named_type_shape(items) -> list[list[str]]:
    return [
        [str(name), _type_name(kind)]
        for name, kind in (items or ())
    ]


def _action_signature(action: object) -> dict[str, list[list[str]]]:
    outputs = _named_type_shape(getattr(action, "outputs", ()))
    if not outputs and hasattr(action, "output") and hasattr(action, "output_type"):
        outputs = [
            [
                str(getattr(action, "output")),
                _type_name(getattr(action, "output_type")),
            ]
        ]
    return {
        "inputs": _named_type_shape(getattr(action, "inputs", ())),
        "outputs": outputs,
    }


def _shape(node) -> list:
    """A structural description of one statement and its children.

    Control state is child-index paths, so what matters is the shape of the tree
    and which statement sits at each position. Guard closures are deliberately
    excluded: two programs that differ only in what a condition computes still
    have the same positions, and their control state stays meaningful. A
    condition's repr also carries its memory address, which would make this
    differ between processes for no reason.
    """

    kind = type(node).__name__
    fields: list = []
    for name in (
        "lifeline",
        "owner",
        "sender",
        "receiver",
        "channel",
        "branch_indices",
    ):
        value = getattr(node, name, None)
        if value is not None:
            fields.append(
                [name, getattr(value, "name", None) or _plain(value)]
            )
    action = getattr(node, "action", None)
    if action is not None:
        # The action kind, name and declared interface affect how the durable
        # environment is interpreted. The implementation body and any prompt
        # text remain deliberately excluded: changing those changes future work,
        # but does not make already-committed state unreadable.
        fields.append(
            [
                "action",
                type(action).__name__,
                getattr(action, "name", type(action).__name__),
            ]
        )
        fields.append(
            ["signature", _action_signature(action)]
        )
    for name in ("payload", "inputs", "bindings"):
        value = getattr(node, name, None)
        if value:
            fields.append([name, [_expression_shape(item) for item in value]])
    outputs = getattr(node, "outputs", None)
    if outputs:
        fields.append(["outputs", [_variable_shape(item) for item in outputs]])
    receives = getattr(node, "receives", None)
    if receives:
        fields.append(
            [
                "receives",
                sorted(
                    [
                        sender.name,
                        [_expression_shape(binding) for binding in bindings],
                    ]
                    for sender, bindings in receives
                ),
            ]
        )
    return [kind, fields, [_shape(child) for child in _children(node)]]


def _plain(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _plain(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return str(value)


def program_fingerprint(local_programs: Mapping[str, object]) -> str:
    """Identify the projected code durable state may be resumed against.

    The stored environment and control paths must mean the same thing to the
    new program. This therefore covers both statement positions and the names
    and types of values read or written at those positions. Implementation
    bodies, prompts and guard computations remain deliberately outside this
    compatibility identity. The hash is stable across processes.
    """

    payload = json.dumps(
        {role: _shape(program) for role, program in sorted(local_programs.items())},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]
