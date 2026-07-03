import queue
import threading
import time
import pytest
from zippergen.channels import InProcessChannel
import zippergen.channels as _channels_mod

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


def test_concurrent_first_touch_delivers(monkeypatch):
    """A sender and receiver first-touching the SAME channel key at the same
    instant must share one underlying queue. A get-or-create that isn't atomic
    binds them to two different queues, the item lands in the one nobody reads,
    and the receiver blocks forever — the root cause of the intermittent
    multi-process/threaded deadlock. We widen the per-queue construction window
    so the race is hit every run instead of one time in thousands."""
    orig_init = _channels_mod._SeqQueue.__init__

    def slow_init(self):
        time.sleep(0.005)          # stand in for the GIL switch inside queue.Queue()
        orig_init(self)

    monkeypatch.setattr(_channels_mod._SeqQueue, "__init__", slow_init)

    ch = InProcessChannel()
    key = ("A", "B", "main")
    barrier = threading.Barrier(2)
    got: list = []

    def send():
        barrier.wait()
        ch.put(*key, (42,))

    def recv():
        barrier.wait()
        stop = threading.Event()
        threading.Timer(2.0, stop.set).start()   # fail fast instead of hanging
        try:
            got.append(ch.get(*key, stop=stop))
        except RuntimeError:
            pass                                  # orphaned queue -> nothing ever arrives

    ts = threading.Thread(target=send)
    tr = threading.Thread(target=recv)
    ts.start(); tr.start()
    ts.join(timeout=5); tr.join(timeout=5)

    assert got and got[0][1] == (42,), (
        "message lost: sender and receiver bound different queues for the same key"
    )
