"""Two assistants work a task; a human decides whether they continue.

The human owns the loop, so both assistants wait on a decision neither of them
evaluates. These pin the endings a person can choose, the unattended mode, and
what each side is actually told.
"""

from typing import Any, cast

from zippergen.runtime import run
from zippergen.serve import load_workflow_spec

SPEC = "examples/pair_programming.py:pair_programming"

APPROVING = "APPROVE\nFINDINGS: 0\nNothing real to report.\n\nLooks correct."
OBJECTING = "REVISE\nFINDINGS: 2\nThe fix is right but the tests do not pin it.\n\n**1.** ..."


def _run(verdicts: list[str], reviews: list[str], *, detail: str, max_rounds: int = 4):
    workflow, module = load_workflow_spec(SPEC)
    remaining_reviews = iter(reviews)
    answers = iter(verdicts)
    seen: dict[str, list[object]] = {"implement": [], "review": []}

    def assistant_backend(action: Any, inputs: dict[str, object]) -> dict[str, object]:
        seen[action.name].append(dict(inputs))
        value = (
            next(remaining_reviews)
            if action.name == "review"
            else "Changed apply_discount.\n\nLong detail nobody reads."
        )
        return {action.outputs[0][0]: value}

    def human_backend(action: Any, inputs: dict[str, object]) -> dict[str, object]:
        seen.setdefault(action.name, []).append(dict(inputs))
        value = next(answers) if action.name == "judge_turn" else ""
        return {action.output: value}

    result = run(
        workflow,
        [module.Human, module.Implementer, module.Reviewer],
        {
            "Human": {
                "task": "Money is sometimes a cent short.",
                "max_rounds": max_rounds,
                "detail": detail,
            }
        },
        assistant_backend=assistant_backend,
        human_backend=human_backend,
        timeout=10,
    )
    return cast(str, result), seen


def test_shipping_ends_the_run_without_another_round():
    result, seen = _run([ "Ship it" ], [OBJECTING], detail="brief")

    assert len(seen["implement"]) == 1, "no implementation happens after shipping"
    assert result.startswith("Task: Money is sometimes a cent short.")
    assert "Shipped after 1 round(s)" in result


def test_abandoning_says_the_work_is_still_in_the_tree():
    result, seen = _run(["Abandon"], [OBJECTING], detail="brief")

    assert len(seen["implement"]) == 1
    assert "Abandoned after 1 round(s)" in result
    assert "still in the working tree" in result


def test_another_round_hands_the_reviewer_findings_to_the_implementer():
    """Sending only the human's note is how an implementer was told nothing."""

    _result, seen = _run(
        ["Another round", "Ship it"],
        [OBJECTING, APPROVING],
        detail="brief",
    )

    assert len(seen["implement"]) == 2
    second_guidance = str(seen["implement"][1]["guidance"])
    assert "FINDINGS: 2" in second_guidance
    assert "the tests do not pin it" in second_guidance


def test_the_round_limit_stops_even_while_the_reviewer_objects():
    result, seen = _run(
        ["Another round", "Another round"],
        [OBJECTING, OBJECTING],
        detail="brief",
        max_rounds=2,
    )

    assert len(seen["implement"]) == 2
    assert "Stopped at the 2-round limit, still unresolved" in result


def test_nobody_is_asked_when_the_reviewer_decides():
    """`detail=auto` runs unattended: the reviewer's verdict ends or continues."""

    result, seen = _run([], [OBJECTING, APPROVING], detail="auto")

    assert "judge_turn" not in seen, "no human task is created"
    assert len(seen["implement"]) == 2, "REVISE continued, APPROVE stopped"
    assert "Shipped after 2 round(s)" in result


def test_a_brief_shows_one_line_from_each_side():
    _result, seen = _run(["Ship it"], [OBJECTING], detail="brief")

    briefing = str(seen["judge_turn"][0]["briefing"])
    assert "Implementer: Changed apply_discount." in briefing
    assert "Reviewer: REVISE (2 finding(s))" in briefing
    assert "Long detail nobody reads" not in briefing


def test_a_review_that_breaks_its_contract_is_not_read_as_approval():
    result, _seen = _run([], ["no header at all"], detail="auto", max_rounds=1)

    assert "Stopped at the 1-round limit" in result, (
        "an unreadable review continues the loop rather than approving"
    )
