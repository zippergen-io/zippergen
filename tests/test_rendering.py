from zippergen.rendering import TerminalRenderer


def test_column_renderer_keeps_short_marked_statuses_on_one_line():
    output: list[str] = []
    renderer = TerminalRenderer(
        output.append,
        color=False,
        columns=lambda: 60,
    )

    renderer.columns(
        "Model configurations",
        ("Name", "Provider", "Model", "Status"),
        [
            ("mock", "mock", "built in", "✓ available"),
            ("drafting", "openai", "gpt-4o-mini", "✓ available"),
            (
                "local-qwen2.5-7b",
                "local",
                "qwen2.5:7b",
                "✓ available",
            ),
            (
                "review-mistral",
                "mistral",
                "mistral-small-latest",
                "✓ available",
            ),
        ],
    )

    assert sum("✓ available" in line for line in output) == 4
    assert not any(line.strip() == "available" for line in output)
    assert all(renderer.visible_width(line) <= 60 for line in output)


def test_column_renderer_wraps_descriptions_before_status_phrases():
    output: list[str] = []
    renderer = TerminalRenderer(
        output.append,
        color=False,
        columns=lambda: 60,
    )

    renderer.columns(
        "Provider connections",
        ("Provider", "State", "Details"),
        [
            (
                "anthropic",
                "not configured",
                "⚠ not configured; use models provider configure anthropic",
            ),
            (
                "local",
                "checked",
                "✓ last check succeeded; many model identifiers follow",
            ),
        ],
    )

    assert any("⚠ not configured" in line for line in output)
    assert any("✓ last check succeeded" in line for line in output)
    assert not any(line.rstrip().endswith(("✓", "⚠")) for line in output)
    assert all(renderer.visible_width(line) <= 60 for line in output)


def test_column_renderer_keeps_unmarked_state_values_atomic():
    output: list[str] = []
    renderer = TerminalRenderer(
        output.append,
        color=False,
        columns=lambda: 60,
    )

    renderer.columns(
        "Implementation history",
        ("Request", "Kind", "State", "Created"),
        [
            (
                "20260726-very-long-request",
                "refine",
                "awaiting human review",
                "2026-07-26T12:00:00+0200",
            )
        ],
    )

    assert any("awaiting human review" in line for line in output)
    assert not any(line.strip() in {"human review", "review"} for line in output)
    assert all(renderer.visible_width(line) <= 60 for line in output)


def test_column_renderer_wraps_colored_statuses_by_visible_width():
    output: list[str] = []
    renderer = TerminalRenderer(
        output.append,
        color=True,
        columns=lambda: 60,
    )
    status = f"{renderer.status_mark('success')} available"

    renderer.columns(
        "Model configurations",
        ("Name", "Provider", "Model", "Status"),
        [
            (
                "openai-gpt-4o-mini",
                "openai",
                "gpt-4o-mini",
                status,
            )
        ],
    )

    status_lines = [line for line in output if "available" in line]
    assert len(status_lines) == 1
    assert status in status_lines[0]
    assert all(renderer.visible_width(line) <= 60 for line in output)
