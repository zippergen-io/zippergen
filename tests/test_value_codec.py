"""Tests for the strict, lossless durable-value codec."""

import pytest

from zippergen.value_codec import decode_value, dumps_value, loads_value


def test_codec_preserves_tuple_and_list_identity():
    value = ((1, 2), [1, 2], {"nested": (True, None)})

    restored = loads_value(dumps_value(value))

    assert restored == value
    assert type(restored) is tuple
    assert type(restored[0]) is tuple
    assert type(restored[1]) is list


def test_decoder_rejects_excessive_nesting_before_recursing_unboundedly():
    encoded: object = ["scalar", None]
    for _ in range(129):
        encoded = ["list", [encoded]]

    with pytest.raises(ValueError, match="nests deeper than 128 levels"):
        decode_value({"__zippergen_typed_value_v1__": encoded})
