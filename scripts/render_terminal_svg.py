"""Render an unedited terminal capture as a responsive light/dark SVG."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path


ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
CANVAS_COLUMNS = 100
CELL_WIDTH = 8.4
LINE_HEIGHT = 20
MARGIN_X = 18
MARGIN_Y = 22


def _capture_lines(path: Path) -> list[str]:
    text = ANSI.sub("", path.read_text(encoding="utf-8")).replace("\r", "")
    lines = text.splitlines()
    command_starts = [
        index
        for index, line in enumerate(lines[:-1])
        if line.startswith("╭") and "ZipperGen Studio ·" in lines[index + 1]
    ]
    if command_starts:
        start = command_starts[-1]
    else:
        try:
            start = next(
                index for index, line in enumerate(lines) if line.startswith("╭")
            )
        except StopIteration:
            start = 0
    lines = lines[start:]
    while lines and not lines[-1]:
        lines.pop()
    return lines


def _line_class(line: str) -> str:
    if line.startswith(("╭", "│", "╰")):
        return "banner"
    if "✗" in line:
        return "error"
    if "⚠" in line:
        return "warning"
    if "✓" in line:
        return "success"
    if "▶" in line:
        return "pointer"
    if line.startswith("#"):
        return "comment"
    return "text"


def render_svg(
    capture: Path,
    destination: Path,
    title: str,
    *,
    canvas_columns: int = CANVAS_COLUMNS,
) -> None:
    if canvas_columns < 1 or canvas_columns > CANVAS_COLUMNS:
        raise ValueError(
            f"canvas_columns must be between 1 and {CANVAS_COLUMNS}"
        )
    lines = _capture_lines(capture)
    too_wide = [
        (index, line)
        for index, line in enumerate(lines, start=1)
        if len(line) > canvas_columns
    ]
    if too_wide:
        index, line = too_wide[0]
        raise ValueError(
            f"{capture}: line {index} is {len(line)} columns wide. "
            f"Capture Studio at {canvas_columns} columns so it wraps the output."
        )
    width = int(canvas_columns * CELL_WIDTH + 2 * MARGIN_X)
    height = len(lines) * LINE_HEIGHT + 2 * MARGIN_Y
    text_nodes = []
    for index, line in enumerate(lines):
        y = MARGIN_Y + 14 + index * LINE_HEIGHT
        text_nodes.append(
            f'<text class="{_line_class(line)}" x="{MARGIN_X}" y="{y}" '
            f'xml:space="preserve">{html.escape(line)}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" role="img"
  aria-labelledby="title description" width="{width}" height="{height}"
  viewBox="0 0 {width} {height}" preserveAspectRatio="xMinYMin meet">
  <title id="title">{html.escape(title)}</title>
  <desc id="description">Real ZipperGen Studio terminal output.</desc>
  <style>
    .background {{ fill: #f6f8fa; stroke: #d0d7de; }}
    text {{ fill: #1f2328; font: 14px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .banner {{ fill: #0969da; }}
    .success {{ fill: #1a7f37; }}
    .warning {{ fill: #9a6700; }}
    .error {{ fill: #cf222e; }}
    .pointer {{ fill: #8250df; font-weight: 700; }}
    .comment {{ fill: #57606a; }}
    @media (prefers-color-scheme: dark) {{
      .background {{ fill: #0d1117; stroke: #30363d; }}
      text {{ fill: #e6edf3; }}
      .banner {{ fill: #58a6ff; }}
      .success {{ fill: #3fb950; }}
      .warning {{ fill: #d29922; }}
      .error {{ fill: #f85149; }}
      .pointer {{ fill: #d2a8ff; }}
      .comment {{ fill: #8b949e; }}
    }}
  </style>
  <rect class="background" x="0.5" y="0.5" width="{width - 1}"
    height="{height - 1}" rx="8"/>
  {chr(10).join(text_nodes)}
</svg>
"""
    destination.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument(
        "--columns",
        type=int,
        default=CANVAS_COLUMNS,
        help=(
            "SVG terminal width. Use a narrower value only for captures "
            "already wrapped to that width."
        ),
    )
    args = parser.parse_args()
    render_svg(
        args.capture,
        args.destination,
        args.title,
        canvas_columns=args.columns,
    )


if __name__ == "__main__":
    main()
