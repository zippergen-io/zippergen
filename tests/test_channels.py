import queue
import pytest
from zippergen.channels import InProcessChannel

def test_put_then_try_get_roundtrip():
    ch = InProcessChannel()
    seq = ch.put("A", "B", "main", (1, 2))
    assert seq == 0
    item = ch.try_get("A", "B", "main")
    assert item is not None
    got_seq, values, vc, view, field_view = item
    assert got_seq == 0 and values == (1, 2) and vc is None

def test_try_get_empty_returns_none():
    ch = InProcessChannel()
    assert ch.try_get("A", "B", "main") is None

def test_fifo_order_per_channel():
    ch = InProcessChannel()
    ch.put("A", "B", "main", (1,))
    ch.put("A", "B", "main", (2,))
    assert ch.try_get("A", "B", "main")[1] == (1,)
    assert ch.try_get("A", "B", "main")[1] == (2,)

def test_channels_are_isolated_by_key():
    ch = InProcessChannel()
    ch.put("A", "B", "main", (1,))
    assert ch.try_get("A", "C", "main") is None
    assert ch.try_get("A", "B", "ctrl") is None
