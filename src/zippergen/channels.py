"""Channel abstraction shared by the in-process and durable runtimes.

The interpreter (`_step` / `_exec`) touches channels only through three
operations: put (send), try_get (non-blocking recv), get (blocking recv).
Items are 5-tuples ``(seq, values, vc, view, field_view)``; vc/view/field_view
are the sender's monitor snapshot, or None when monitoring is inactive.
"""
from __future__ import annotations

import queue
import threading
from collections import defaultdict

Item = tuple[int, tuple, "dict | None", "dict | None", "dict | None"]


class _SeqQueue:
    """FIFO queue that auto-stamps each item with a per-channel sequence number."""

    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue()
        self._next = 0

    def put(self, values: tuple, vc: dict | None = None, view: dict | None = None,
            field_view: dict | None = None) -> int:
        seq = self._next
        self._next += 1
        self._q.put((seq, values, vc, view, field_view))
        return seq

    def get(self, *, stop: threading.Event | None = None) -> Item:
        if stop is None:
            return self._q.get()
        while True:
            try:
                return self._q.get(timeout=0.5)
            except queue.Empty:
                if stop.is_set():
                    raise RuntimeError("Workflow cancelled: another lifeline failed")

    def get_nowait(self) -> Item:
        return self._q.get_nowait()


class InProcessChannel:
    """In-memory channel map keyed by (sender, receiver, channel)."""

    def __init__(self) -> None:
        self._qs: dict[tuple[str, str, str], _SeqQueue] = defaultdict(_SeqQueue)

    def put(self, sender: str, receiver: str, channel: str, values: tuple,
            vc: dict | None = None, view: dict | None = None,
            field_view: dict | None = None) -> int:
        return self._qs[(sender, receiver, channel)].put(values, vc, view, field_view)

    def try_get(self, sender: str, receiver: str, channel: str) -> Item | None:
        try:
            return self._qs[(sender, receiver, channel)].get_nowait()
        except queue.Empty:
            return None

    def get(self, sender: str, receiver: str, channel: str, *,
            stop: threading.Event | None = None) -> Item:
        return self._qs[(sender, receiver, channel)].get(stop=stop)
