"""Small dependency-free full-screen display for live CLI views."""

from __future__ import annotations

import os
import shutil
import sys
import time
from collections.abc import Callable
from typing import TextIO

_ENTER_SCREEN = "\033[?1049h\033[?25l\033[2J\033[H"
_LEAVE_SCREEN = "\033[?25h\033[?1049l"
_CLEAR_SCREEN = "\033[2J\033[H"


def live_display_available(stream: TextIO | None = None) -> bool:
    """Return whether *stream* can support an in-place terminal view."""

    output: TextIO = sys.stdout if stream is None else stream
    return bool(getattr(output, "isatty", lambda: False)()) and (
        os.environ.get("TERM") != "dumb"
    )


def _viewport(lines: list[str], height: int) -> list[str]:
    """Keep the current program pointer visible in a short terminal."""

    if len(lines) <= height:
        return lines
    pointer = next(
        (index for index in range(len(lines) - 1, -1, -1) if "▶" in lines[index]),
        None,
    )
    if pointer is None:
        return [*lines[: height - 1], "↓ more"]

    projection = next(
        (
            index
            for index in range(pointer, -1, -1)
            if lines[index].endswith(" local projection")
        ),
        None,
    )
    if projection is not None:
        body_start = min(len(lines), projection + 2)
        prefix = lines[:body_start]
        body_height = height - len(prefix)
        if body_height >= 5:
            body = lines[body_start:]
            body_pointer = max(0, pointer - body_start)
            return prefix + _pointer_window(body, body_pointer, body_height)

    return _pointer_window(lines, pointer, height)


def _pointer_window(lines: list[str], pointer: int, height: int) -> list[str]:
    """Return a bounded window around *pointer*, with continuation marks."""

    if len(lines) <= height:
        return lines
    before = height // 2
    start = max(0, min(pointer - before, len(lines) - height))
    end = min(len(lines), start + height)
    window = lines[start:end]
    if start:
        window[0] = "↑ more"
    if end < len(lines):
        window[-1] = "↓ more"
    return window


def _screen_lines(frame: str, columns: int, rows: int) -> list[str]:
    """Fit a logical frame into physical terminal rows without wrapping."""

    width = max(1, columns - 1)
    height = max(1, rows - 1)
    lines = [line[:width] for line in frame.splitlines()]
    return _viewport(lines or [""], height)


def _write_changes(
    stream: TextIO,
    previous: list[str],
    current: list[str],
    *,
    reset: bool,
) -> None:
    if reset:
        stream.write(_CLEAR_SCREEN)
        previous = []
    for index in range(max(len(previous), len(current))):
        old = previous[index] if index < len(previous) else None
        new = current[index] if index < len(current) else ""
        if old == new:
            continue
        stream.write(f"\033[{index + 1};1H\033[2K{new}")
    stream.flush()


def watch_frames(
    frame: Callable[[int], str],
    *,
    interval: float = 1.0,
    stream: TextIO | None = None,
    terminal_size: Callable[[], os.terminal_size] = shutil.get_terminal_size,
    sleep: Callable[[float], object] = time.sleep,
) -> bool:
    """Display refreshed frames until Ctrl-C.

    ``frame`` receives the current terminal width. The return value is true
    when the user closed the display with Ctrl-C.
    """

    output: TextIO = sys.stdout if stream is None else stream
    previous: list[str] = []
    previous_size: os.terminal_size | None = None
    interrupted = False
    output.write(_ENTER_SCREEN)
    output.flush()
    try:
        while True:
            size = terminal_size()
            current = _screen_lines(frame(size.columns), size.columns, size.lines)
            _write_changes(
                output,
                previous,
                current,
                reset=previous_size is not None and size != previous_size,
            )
            previous = current
            previous_size = size
            sleep(interval)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        output.write(_LEAVE_SCREEN)
        output.flush()
    return interrupted


__all__ = ["live_display_available", "watch_frames"]
