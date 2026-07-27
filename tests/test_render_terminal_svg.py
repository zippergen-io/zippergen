from pathlib import Path

import pytest

from scripts.render_terminal_svg import CANVAS_COLUMNS, render_svg


def test_terminal_svg_uses_one_fixed_canvas_width(tmp_path: Path):
    capture = tmp_path / "capture.txt"
    destination = tmp_path / "capture.svg"
    capture.write_text(
        "╭──────────────────╮\n"
        "│ ZipperGen Studio │\n"
        "╰──────────────────╯\n"
        "Session context\n"
        "╭────╮\n"
        "│ ZipperGen Studio · current │\n"
        "╰────╯\n"
        "Short output\n",
        encoding="utf-8",
    )

    render_svg(capture, destination, "Example")

    svg = destination.read_text(encoding="utf-8")
    assert 'width="876"' in svg
    assert 'viewBox="0 0 876 ' in svg
    assert 'preserveAspectRatio="xMinYMin meet"' in svg
    assert "Session context" not in svg
    assert "ZipperGen Studio · current" in svg


def test_terminal_svg_rejects_a_capture_wider_than_its_canvas(tmp_path: Path):
    capture = tmp_path / "capture.txt"
    destination = tmp_path / "capture.svg"
    capture.write_text(
        "╭────╮\n" + "x" * (CANVAS_COLUMNS + 1) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="101 columns wide"):
        render_svg(capture, destination, "Too wide")


def test_terminal_svg_can_use_a_narrower_canvas_without_scaling_text(
    tmp_path: Path,
):
    capture = tmp_path / "capture.txt"
    destination = tmp_path / "capture.svg"
    capture.write_text(
        "╭────╮\n"
        "│ ZipperGen Studio · deploy show │\n"
        "╰────╯\n"
        "Deployment  review-demo\n",
        encoding="utf-8",
    )

    render_svg(
        capture,
        destination,
        "Deployment",
        canvas_columns=80,
    )

    svg = destination.read_text(encoding="utf-8")
    assert 'width="708"' in svg
    assert 'viewBox="0 0 708 ' in svg
    assert "font: 14px" in svg
