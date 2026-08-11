from typing import Any, cast

from zippergen.runtime import run
from zippergen.semantic import workflow_semantics
from zippergen.serve import load_workflow_spec


def _run_review(reviews: list[str], *, max_rounds: int = 3):
    workflow, module = load_workflow_spec(
        "examples/codex_claude_review.py:codex_claude_review"
    )
    remaining = iter(reviews)
    calls: list[str] = []

    def backend(action: Any, inputs: dict[str, object]) -> dict[str, object]:
        calls.append(action.name)
        if action.name == "review_candidate":
            value = next(remaining)
        elif action.name == "finalize_result":
            value = "Codex final report"
        else:
            value = f"{action.name} completed"
        return {action.outputs[0][0]: value}

    result = run(
        workflow,
        [module.TaskOwner, module.Codex, module.Claude],
        {
            "TaskOwner": {
                "task": "Improve the parser.",
                "max_review_rounds": max_rounds,
            }
        },
        assistant_backend=backend,
        timeout=5,
    )
    return cast(str, result), calls


def test_codex_claude_review_finishes_only_after_approval():
    result, calls = _run_review(
        ["REVISE\nAdd a regression test.", "APPROVE\nAll checks pass."]
    )

    assert result == "Codex final report"
    assert calls == [
        "implement_task",
        "review_candidate",
        "revise_candidate",
        "review_candidate",
        "finalize_result",
    ]


def test_codex_claude_review_reports_unresolved_exhaustion():
    result, calls = _run_review(
        ["REVISE\nConcern one.", "REVISE\nConcern remains."],
        max_rounds=2,
    )

    assert "not accepted after 2 review round(s)" in result
    assert "Concern remains." in result
    assert "finalize_result" not in calls


def test_codex_claude_review_has_enforced_roles_and_owned_control():
    workflow, module = load_workflow_spec(
        "examples/codex_claude_review.py:codex_claude_review"
    )
    model = cast(dict[str, Any], workflow_semantics(workflow, module))
    actions = model["action_definitions"]

    assert model["lifelines"] == ["TaskOwner", "Codex", "Claude"]
    assert {control["owner"] for control in model["controls"]} == {"Codex"}
    assert actions["implement_task"].get("access", "write") == "write"
    assert actions["implement_task"]["external_tools"] == "none"
    assert actions["implement_task"]["shell"] == "restricted"
    assert actions["review_candidate"]["access"] == "read-only"
    assert actions["review_candidate"]["external_tools"] == "none"
    assert actions["review_candidate"]["shell"] == "restricted"
    assert actions["finalize_result"]["access"] == "read-only"
