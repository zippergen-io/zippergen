"""Lossless JSON storage for ZipperGen coordination values.

Plain JSON cannot distinguish a tuple from a list.  That matters for durable
state: ``tuple`` is a declared coordination type, so resuming a computation
must not silently turn one into the other.  This module adds a small tagged
encoding while keeping the database columns ordinary, inspectable JSON text.

The codec is deliberately strict. A durable boundary must either preserve a
value exactly or reject it; silently accepting an untagged JSON value would
reintroduce the tuple/list ambiguity this module exists to remove.
"""

from __future__ import annotations

import json
import math
from typing import cast

__all__ = [
    "decode_value",
    "dumps_value",
    "encode_value",
    "json_value_error",
    "loads_value",
    "portable_value_error",
]


_ENVELOPE = "__zippergen_typed_value_v1__"
_MAX_DEPTH = 128


def portable_value_error(
    value: object,
    *,
    path: str = "$",
    active: set[int] | None = None,
    depth: int = 0,
    json_only: bool = False,
) -> str | None:
    """Explain why a value cannot cross a durable boundary, if it cannot.

    ``json_only`` excludes tuples for the public ``Json`` coordination type.
    Ordinary tuple-typed coordination values use the same recursion, depth,
    cycle, key and finite-number checks but retain tuple/list distinctions.
    """

    value_type = type(value)
    if value is None or value_type in {bool, int, str}:
        return None
    if value_type is float:
        return (
            None
            if math.isfinite(cast(float, value))
            else f"{path} is not a finite number"
        )
    allowed_containers = {list, dict} if json_only else {tuple, list, dict}
    if value_type not in allowed_containers:
        expected = (
            "a built-in null, boolean, number, string, list, or dictionary"
            if json_only
            else "a portable scalar, tuple, list, or dictionary"
        )
        return f"{path} has type {value_type.__name__}; expected {expected}"
    if depth >= _MAX_DEPTH:
        return f"{path} nests deeper than {_MAX_DEPTH} levels"

    active = set() if active is None else active
    identity = id(value)
    if identity in active:
        return f"{path} contains a circular reference"
    active.add(identity)
    try:
        if value_type in {tuple, list}:
            sequence = cast(tuple[object, ...] | list[object], value)
            for index, item in enumerate(sequence):
                error = portable_value_error(
                    item,
                    path=f"{path}[{index}]",
                    active=active,
                    depth=depth + 1,
                    json_only=json_only,
                )
                if error is not None:
                    return error
            return None

        mapping = cast(dict[object, object], value)
        for key, item in mapping.items():
            if type(key) is not str:
                return (
                    f"{path} has a {type(key).__name__} key; "
                    "dictionary keys must be strings"
                )
            error = portable_value_error(
                item,
                path=f"{path}.{key}",
                active=active,
                depth=depth + 1,
                json_only=json_only,
            )
            if error is not None:
                return error
        return None
    finally:
        active.remove(identity)


def json_value_error(value: object) -> str | None:
    """Explain why a value is not a strict JSON value, if it is not."""

    return portable_value_error(value, json_only=True)


def _encode(value: object) -> object:
    value_type = type(value)
    if value is None or value_type in {bool, int, float, str}:
        return ["scalar", value]
    if value_type is tuple:
        sequence = cast(tuple[object, ...], value)
        return ["tuple", [_encode(item) for item in sequence]]
    if value_type is list:
        sequence = cast(list[object], value)
        return ["list", [_encode(item) for item in sequence]]
    if value_type is dict:
        mapping = cast(dict[object, object], value)
        if any(type(key) is not str for key in mapping):
            raise TypeError("ZipperGen value dictionaries must have string keys.")
        items: list[list[object]] = []
        keys = sorted(cast(list[str], list(mapping)))
        for key in keys:
            items.append([key, _encode(mapping[key])])
        return ["dict", items]
    raise TypeError(
        f"{type(value).__name__} is not a portable ZipperGen value."
    )


def _decode(encoded: object, *, depth: int = 0) -> object:
    if not (
        type(encoded) is list
        and len(encoded) == 2
        and type(encoded[0]) is str
    ):
        raise ValueError("Malformed encoded ZipperGen value.")
    kind, payload = encoded
    if kind == "scalar":
        if payload is None or type(payload) in {bool, int, float, str}:
            if type(payload) is float and not math.isfinite(cast(float, payload)):
                raise ValueError("Malformed non-finite scalar ZipperGen value.")
            return payload
        raise ValueError("Malformed scalar ZipperGen value.")
    if kind in {"tuple", "list"}:
        if type(payload) is not list:
            raise ValueError(f"Malformed {kind} ZipperGen value.")
        if depth >= _MAX_DEPTH:
            raise ValueError(
                f"Encoded ZipperGen value nests deeper than {_MAX_DEPTH} levels."
            )
        values = [_decode(item, depth=depth + 1) for item in payload]
        return tuple(values) if kind == "tuple" else values
    if kind == "dict":
        if type(payload) is not list:
            raise ValueError("Malformed dictionary ZipperGen value.")
        if depth >= _MAX_DEPTH:
            raise ValueError(
                f"Encoded ZipperGen value nests deeper than {_MAX_DEPTH} levels."
            )
        result: dict[str, object] = {}
        for item in payload:
            if not (
                type(item) is list
                and len(item) == 2
                and type(item[0]) is str
                and item[0] not in result
            ):
                raise ValueError("Malformed dictionary entry in ZipperGen value.")
            result[item[0]] = _decode(item[1], depth=depth + 1)
        return result
    raise ValueError(f"Unknown encoded ZipperGen value kind {kind!r}.")


def encode_value(value: object) -> dict[str, object]:
    """Return the JSON-safe typed envelope for one portable value."""

    error = portable_value_error(value)
    if error is not None:
        raise TypeError(f"Value is not portable: {error}.")
    return {_ENVELOPE: _encode(value)}


def decode_value(encoded: object) -> object:
    """Decode one typed envelope, rejecting plain or malformed JSON."""

    if (
        type(encoded) is not dict
        or set(cast(dict[object, object], encoded)) != {_ENVELOPE}
    ):
        raise ValueError("Value is not a typed ZipperGen value.")
    return _decode(cast(dict[str, object], encoded)[_ENVELOPE])


def dumps_value(value: object) -> str:
    """Return deterministic JSON that preserves every supported container type."""

    return json.dumps(
        encode_value(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def loads_value(payload: str) -> object:
    """Decode typed JSON produced by :func:`dumps_value`."""

    return decode_value(json.loads(payload))
