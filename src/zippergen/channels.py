"""Channel abstraction shared by the in-process and durable runtimes.

The interpreter (`_step` / `_exec`) touches channels through put (send),
try_get (non-blocking recv), get (blocking recv), and optional try_get_any
(deterministic receive-any when a channel can provide a global order).
Items are 5-tuples ``(seq, values, vc, view, field_view)``; vc/view/field_view
are the sender's monitor snapshot, or None when monitoring is inactive.
"""
from __future__ import annotations

import queue
import threading

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
        self._qs: dict[tuple[str, str, str], _SeqQueue] = {}
        self._lock = threading.Lock()

    def _queue(self, key: tuple[str, str, str]) -> _SeqQueue:
        """Atomically get-or-create the queue for a key.

        A plain ``defaultdict`` is NOT safe here: its ``__missing__`` runs the
        ``_SeqQueue`` factory (which constructs a ``queue.Queue``, i.e. Python
        code that can drop the GIL), so a sender and receiver first-touching the
        same key at the same instant can each build a *separate* queue — the
        item lands in one and the reader blocks forever on the other. The lock
        makes creation-or-lookup a single atomic step; it is never held across a
        blocking ``get``."""
        with self._lock:
            q = self._qs.get(key)
            if q is None:
                q = _SeqQueue()
                self._qs[key] = q
            return q

    def put(self, sender: str, receiver: str, channel: str, values: tuple,
            vc: dict | None = None, view: dict | None = None,
            field_view: dict | None = None) -> int:
        return self._queue((sender, receiver, channel)).put(values, vc, view, field_view)

    def try_get(self, sender: str, receiver: str, channel: str) -> Item | None:
        try:
            return self._queue((sender, receiver, channel)).get_nowait()
        except queue.Empty:
            return None

    def get(self, sender: str, receiver: str, channel: str, *,
            stop: threading.Event | None = None) -> Item:
        return self._queue((sender, receiver, channel)).get(stop=stop)
