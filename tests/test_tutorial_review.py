from typing import Any, cast

from zippergen.runtime import mock_llm, run
from zippergen.semantic import workflow_semantics
from zippergen.serve import load_workflow_spec


def _run_tutorial(*, decisions: list[bool], max_retries: int = 2) -> str:
    workflow, module = load_workflow_spec(
        "examples/tutorial_review.py:tutorial_review"
    )
    remaining = iter(decisions)

    def human_backend(action: Any, inputs: dict[str, object]) -> dict[str, bool]:
        try:
            decision = next(remaining)
        except StopIteration as exc:
            raise AssertionError("the workflow requested an unexpected review") from exc
        return {action.output: decision}

    return cast(
        str,
        run(
            workflow,
            [module.Requester, module.Writer, module.Reviewer],
            {
                "Requester": {
                    "request": "Explain the sky.",
                    "max_retries": max_retries,
                }
            },
            llm_backend=mock_llm,
            human_backend=human_backend,
        ),
    )


def test_tutorial_review_approved_path_returns_first_draft():
    assert _run_tutorial(decisions=[True]) == "[draft_reply:draft]"


def test_tutorial_review_rejection_presents_a_revision_for_review():
    assert _run_tutorial(decisions=[False, True]) == "[revise_reply:draft]"


def test_tutorial_review_stops_after_the_retry_budget_is_exhausted():
    assert _run_tutorial(decisions=[False, False, False]) == (
        "Review stopped after 2 revision attempt(s) without approval. "
        "Last draft: [revise_reply:draft]"
    )


def test_tutorial_review_zero_retries_stops_after_initial_rejection():
    assert _run_tutorial(decisions=[False], max_retries=0) == (
        "Review stopped after 0 revision attempt(s) without approval. "
        "Last draft: [draft_reply:draft]"
    )


def test_tutorial_review_has_inspectable_deployment_contract():
    workflow, module = load_workflow_spec(
        "examples/tutorial_review.py:tutorial_review"
    )
    model = cast(dict[str, Any], workflow_semantics(workflow, module))

    assert model["lifelines"] == ["Requester", "Writer", "Reviewer"]
    assert {control["owner"] for control in model["controls"]} == {"Reviewer"}
    assert model["action_definitions"]["approve_reply"]["kind"] == "human"
    assert model["action_definitions"]["assess_draft"]["kind"] == "llm"
    assert model["deployment"]["name"] == "tutorial-review"
    assert model["deployment"]["fields"]["max_retries"]["default"] == 2
    assert model["deployment"]["fields"]["openai_api_key"]["secret"] is True
    assert model["deployment"]["fields"]["anthropic_api_key"]["secret"] is True
