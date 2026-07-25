"""Shared terminal rendering primitives for interactive ZipperGen surfaces."""

from __future__ import annotations

import os
import re
import shutil
import sys
import textwrap
from collections.abc import Callable
from typing import Literal

OutputFunc = Callable[[str], object]
StatusKind = Literal["success", "warning", "error", "info"]

STATUS_MARKS: dict[StatusKind, str] = {
    "success": "✓",
    "warning": "⚠",
    "error": "✗",
    "info": "•",
}
_STATUS_COLORS: dict[StatusKind, str] = {
    "success": "32",
    "warning": "33",
    "error": "31",
    "info": "36",
}
_DEFAULT_OUTPUT_COLUMNS = 100
_MAX_OUTPUT_COLUMNS = 108
_MIN_OUTPUT_COLUMNS = 60


class TerminalRenderer:
    """Render status, key/value data, and columns with one visual contract."""

    def __init__(
        self,
        output: OutputFunc = print,
        *,
        color: bool | None = None,
        columns: Callable[[], int] | None = None,
    ) -> None:
        self.output = output
        self.color = (
            output is print
            and bool(getattr(sys.stdout, "isatty", lambda: False)())
            and "NO_COLOR" not in os.environ
            and os.environ.get("TERM") != "dumb"
            if color is None
            else color
        )
        self._columns = columns

    def emit(self, value: object = "") -> None:
        self.output(str(value))

    @staticmethod
    def visible_width(value: str) -> int:
        return len(re.sub(r"\x1b\[[0-9;]*m", "", value))

    def output_columns(self) -> int:
        if self._columns is not None:
            return max(
                _MIN_OUTPUT_COLUMNS,
                min(_MAX_OUTPUT_COLUMNS, self._columns()),
            )
        return self.detected_output_columns()

    def detected_output_columns(self) -> int:
        if self.output is print and bool(
            getattr(sys.stdout, "isatty", lambda: False)()
        ):
            columns = shutil.get_terminal_size(
                fallback=(_DEFAULT_OUTPUT_COLUMNS, 24)
            ).columns
        else:
            columns = _DEFAULT_OUTPUT_COLUMNS
        return max(_MIN_OUTPUT_COLUMNS, min(_MAX_OUTPUT_COLUMNS, columns))

    @staticmethod
    def wrapped_lines(value: object, width: int) -> list[str]:
        lines: list[str] = []
        for source_line in str(value).splitlines() or [""]:
            lines.extend(
                textwrap.wrap(
                    source_line,
                    width=max(1, width),
                    break_long_words=True,
                    break_on_hyphens=False,
                    replace_whitespace=False,
                    drop_whitespace=True,
                )
                or [""]
            )
        return lines

    def status_mark(self, kind: StatusKind) -> str:
        mark = STATUS_MARKS[kind]
        if self.color:
            return f"\033[{_STATUS_COLORS[kind]}m{mark}\033[0m"
        return mark

    def status(
        self,
        kind: StatusKind,
        message: str,
        *,
        indent: int = 0,
    ) -> None:
        self.emit(f"{' ' * indent}{self.status_mark(kind)} {message}")

    def section(self, title: str, *, major: bool = False) -> None:
        self.emit(title)
        self.emit(("═" if major else "─") * self.visible_width(title))

    def wrapped_field(
        self,
        label: str,
        value: object,
        *,
        label_width: int = 7,
    ) -> None:
        prefix = f"  {label:<{label_width}}  "
        continuation = " " * self.visible_width(prefix)
        lines = self.wrapped_lines(
            value,
            self.output_columns() - self.visible_width(prefix),
        )
        self.emit(prefix + lines[0])
        for line in lines[1:]:
            self.emit(continuation + line)

    def pad_cell(
        self,
        value: object,
        width: int,
        *,
        right: bool = False,
    ) -> str:
        text = str(value)
        padding = " " * max(0, width - self.visible_width(text))
        return padding + text if right else text + padding

    def columns(
        self,
        title: str,
        headers: tuple[str, ...],
        rows: list[tuple[object, ...]],
        *,
        right_aligned: frozenset[int] = frozenset(),
    ) -> None:
        if not headers:
            raise ValueError("A column table requires at least one heading.")
        if any(len(row) != len(headers) for row in rows):
            raise ValueError("Every column-table row must match its headings.")
        rendered = [tuple(str(value) for value in row) for row in rows]
        natural_widths: list[int] = []
        for index, heading in enumerate(headers):
            candidates = [self.visible_width(heading)]
            candidates.extend(self.visible_width(row[index]) for row in rendered)
            natural_widths.append(max(candidates))
        available = self.output_columns() - 2 - 2 * (len(headers) - 1)
        widths = list(natural_widths)
        if sum(widths) > available:
            minimums = [
                min(width, max(5, min(self.visible_width(headers[index]), 12)))
                for index, width in enumerate(natural_widths)
            ]
            contains_spaces = [
                any(" " in row[index] for row in rendered)
                for index in range(len(headers))
            ]
            while sum(widths) > available:
                candidates = [
                    index
                    for index, width in enumerate(widths)
                    if width > minimums[index]
                ]
                if not candidates:
                    break
                selected = max(
                    candidates,
                    key=lambda index: (
                        contains_spaces[index],
                        widths[index] - minimums[index],
                    ),
                )
                widths[selected] -= 1
        self.section(title)

        def emit_row(values: tuple[str, ...]) -> None:
            wrapped = [
                self.wrapped_lines(value, widths[index])
                for index, value in enumerate(values)
            ]
            height = max(len(lines) for lines in wrapped)
            for line_index in range(height):
                self.emit(
                    "  "
                    + "  ".join(
                        self.pad_cell(
                            (
                                wrapped[index][line_index]
                                if line_index < len(wrapped[index])
                                else ""
                            ),
                            widths[index],
                            right=index in right_aligned,
                        )
                        for index in range(len(values))
                    ).rstrip()
                )

        emit_row(headers)
        self.emit("  " + "  ".join("─" * width for width in widths))
        for row in rendered:
            emit_row(row)
        self.emit()

    def next(self, value: object) -> None:
        self.section("Next")
        for line in self.wrapped_lines(value, self.output_columns() - 2):
            self.emit(f"  {line}")
        self.emit()

    def table(
        self,
        title: str,
        rows: list[tuple[str, object, StatusKind | None]],
        *,
        headings: bool = True,
    ) -> None:
        content = [row for row in rows if row[0].casefold() != "next"]
        next_values = [
            value for label, value, _kind in rows if label.casefold() == "next"
        ]
        self.section(title)
        label_width = max(
            len("Field") if headings else 0,
            max(
                (self.visible_width(label) for label, _value, _kind in content),
                default=0,
            ),
        )
        value_width = max(
            len("Value"),
            self.output_columns() - label_width - 4,
        )
        if headings:
            self.emit(f"  {self.pad_cell('Field', label_width)}  Value")
            self.emit(f"  {'─' * label_width}  {'─' * value_width}")
        for label, value, kind in content:
            mark = f"{self.status_mark(kind)} " if kind else ""
            prefix = f"  {label:<{label_width}}  {mark}"
            continuation = " " * self.visible_width(prefix)
            lines = self.wrapped_lines(
                value,
                self.output_columns() - self.visible_width(prefix),
            )
            self.emit(prefix + lines[0])
            for line in lines[1:]:
                self.emit(continuation + line)
        self.emit()
        for value in next_values:
            self.next(value)
