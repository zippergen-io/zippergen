"""Lossless JSON storage for ZipperGen coordination values.

Plain JSON cannot distinguish a tuple from a list.  That matters for durable
state: ``tuple`` is a declared coordination type, so resuming a computation
must not silently turn one into the other.  This module adds a small tagged
encoding while keeping the database columns ordinary, inspectable JSON text.

``loads_value`` also accepts the untagged JSON written by older stores.  Those
values have already lost any tuple information, but remaining readable is
better than making an otherwise valid store unusable.
"""

from __future__ import annotations

import json
from typing import cast

__all__ = ["dumps_value", "loads_value"]


_ENVELOPE = "__zippergen_typed_value_v1__"


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


def _decode(encoded: object) -> object:
    if not (
        type(encoded) is list
        and len(encoded) == 2
        and type(encoded[0]) is str
    ):
        raise ValueError("Malformed encoded ZipperGen value.")
    kind, payload = encoded
    if kind == "scalar":
        if payload is None or type(payload) in {bool, int, float, str}:
            return payload
        raise ValueError("Malformed scalar ZipperGen value.")
    if kind in {"tuple", "list"}:
        if type(payload) is not list:
            raise ValueError(f"Malformed {kind} ZipperGen value.")
        values = [_decode(item) for item in payload]
        return tuple(values) if kind == "tuple" else values
    if kind == "dict":
        if type(payload) is not list:
            raise ValueError("Malformed dictionary ZipperGen value.")
        result: dict[str, object] = {}
        for item in payload:
            if not (
                type(item) is list
                and len(item) == 2
                and type(item[0]) is str
                and item[0] not in result
            ):
                raise ValueError("Malformed dictionary entry in ZipperGen value.")
            result[item[0]] = _decode(item[1])
        return result
    raise ValueError(f"Unknown encoded ZipperGen value kind {kind!r}.")


def dumps_value(value: object) -> str:
    """Return deterministic JSON that preserves every supported container type."""

    return json.dumps(
        {_ENVELOPE: _encode(value)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def loads_value(payload: str) -> object:
    """Decode a typed value, accepting untagged JSON from an older store."""

    raw = json.loads(payload)
    if type(raw) is dict and set(raw) == {_ENVELOPE}:
        return _decode(raw[_ENVELOPE])
    return raw
