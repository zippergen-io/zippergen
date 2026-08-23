"""One line saying what is running, for a terminal that would otherwise look idle.

A workflow that calls a model or a coding assistant can be busy for minutes
with nothing to print. Silence and a hang look identical from the outside, so a
person watching a correct run has no way to tell it is still working.

This renders a single line on the standard error stream while an action is in
flight, and erases it when the action finishes. Standard output is left alone,
so a result can still be piped.

Two rules keep it out of the way:

* It never draws while a person is being asked something. A human action prints
  its own prompt and reads a reply; a spinner writing underneath that would
  corrupt what the person is typing.
* It draws only into an interactive terminal. Redirected output, a deployment
  log and a test all get nothing.
"""

from __future__ import annotations

import itertools
import threading
import time
from typing import TextIO

__all__ = ["ActivityIndicator", "format_duration"]

_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_INTERVAL_SECONDS = 0.1


def format_duration(milliseconds: float) -> str:
    """Render an elapsed time the way a person reads it."""

    seconds = milliseconds / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(int(seconds), 60)
    return f"{minutes}m{remainder:02d}s"


class ActivityIndicator:
    """Show what is running now, and how long it has been running."""

    def __init__(self, stream: TextIO, *, interactive: bool) -> None:
        self._stream = stream
        self._interactive = interactive
        self._lock = threading.Lock()
        self._running: dict[int, tuple[str, float]] = {}
        self._people_waiting = 0
        self._frames = itertools.cycle(_FRAMES)
        self._width = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- what the workflow tells it ---------------------------------------

    def started(self, key: int, label: str) -> None:
        with self._lock:
            self._running[key] = (label, time.monotonic())
        self._ensure_thread()

    def finished(self, key: int) -> None:
        with self._lock:
            self._running.pop(key, None)
            if not self._running:
                self._erase_locked()

    def person_waiting(self, waiting: bool) -> None:
        """A person is being asked something, so stop drawing until they answer."""

        with self._lock:
            self._people_waiting += 1 if waiting else -1
            self._people_waiting = max(0, self._people_waiting)
            if self._people_waiting:
                self._erase_locked()

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
        with self._lock:
            self._running.clear()
            self._erase_locked()

    # -- drawing ----------------------------------------------------------

    def _ensure_thread(self) -> None:
        if not self._interactive or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._draw_until_stopped,
            name="zippergen-activity",
            daemon=True,
        )
        self._thread.start()

    def _draw_until_stopped(self) -> None:
        while not self._stop.wait(_INTERVAL_SECONDS):
            with self._lock:
                self._draw_locked()

    def _draw_locked(self) -> None:
        if self._people_waiting or not self._running:
            return
        label, since = min(self._running.values(), key=lambda item: item[1])
        elapsed = format_duration((time.monotonic() - since) * 1000)
        others = len(self._running) - 1
        also = f" (+{others} more)" if others else ""
        self._write_locked(f"{next(self._frames)} {label} · {elapsed}{also}")

    def _write_locked(self, line: str) -> None:
        padding = " " * max(0, self._width - len(line))
        self._stream.write(f"\r{line}{padding}")
        self._stream.flush()
        self._width = len(line)

    def _erase_locked(self) -> None:
        if not self._interactive or not self._width:
            return
        self._stream.write("\r" + " " * self._width + "\r")
        self._stream.flush()
        self._width = 0
