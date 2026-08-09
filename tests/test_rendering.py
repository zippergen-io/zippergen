from zippergen.rendering import TerminalRenderer


def test_table_uses_double_section_rule_and_no_data_indent():
    output: list[str] = []
    renderer = TerminalRenderer(
        output.append,
        color=False,
        columns=lambda: 80,
    )

    renderer.table(
        "Execution context",
        [
            ("Run", "tutorial-review", None),
            ("Status", "waiting", "warning"),
        ],
    )

    assert output[:6] == [
        "Execution context",
        "═" * len("Execution context"),
        "Field   Value",
        "──────  ─────",
        "Run     tutorial-review",
        "Status  ⚠ waiting",
    ]
    assert not any(line.startswith("  ") for line in output if line)


def test_column_table_uses_double_section_rule_and_no_indent():
    output: list[str] = []
    renderer = TerminalRenderer(
        output.append,
        color=False,
        columns=lambda: 80,
    )

    renderer.columns(
        "Participants",
        ("Name", "State"),
        [("Writer", "running")],
    )

    assert output[:5] == [
        "Participants",
        "═" * len("Participants"),
        "Name    State",
        "──────  ───────",
        "Writer  running",
    ]


def test_next_uses_double_section_rule_and_no_indent():
    output: list[str] = []
    renderer = TerminalRenderer(output.append, color=False)

    renderer.next("run")

    assert output[:3] == ["Next", "════", "run"]


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
                "⚠ not configured; use model provider configure anthropic",
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


def test_column_renderer_truncates_identifiers_and_dates_without_splitting():
    output: list[str] = []
    renderer = TerminalRenderer(
        output.append,
        color=False,
        columns=lambda: 60,
    )

    renderer.columns(
        "Durable stores",
        ("Store", "Used by", "State", "Updated"),
        [
            (
                "tutorial_review-20260726-100626-683804000",
                "durable run",
                "✓ done",
                "2026-07-26 10:06",
            )
        ],
    )

    data = [line for line in output if "tutorial_" in line]
    assert len(data) == 1
    assert "…" in data[0]
    assert not any(line.strip() == "6 10:06" for line in output)
    assert all(renderer.visible_width(line) <= 60 for line in output)


def test_column_renderer_uses_more_width_for_data_than_prose_tables():
    renderer = TerminalRenderer(
        lambda _value: None,
        color=False,
        columns=lambda: 160,
    )

    assert renderer.output_columns() == 108
    assert renderer.data_output_columns() == 160
