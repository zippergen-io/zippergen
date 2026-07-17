from zippergen.runtime import mock_llm, run
from zippergen.semantic import workflow_semantics
from zippergen.serve import load_workflow_spec


def _run_tutorial(*, approved: bool) -> str:
    workflow, module = load_workflow_spec(
        "examples/tutorial_review.py:tutorial_review"
    )

    def human_backend(action, inputs):
        return {action.output: approved}

    return run(
        workflow,
        [module.Requester, module.Writer, module.Reviewer],
        {"Requester": {"request": "Explain the sky."}},
        llm_backend=mock_llm,
        human_backend=human_backend,
    )


def test_tutorial_review_approved_path_returns_first_draft():
    assert _run_tutorial(approved=True) == "[draft_reply:draft]"


def test_tutorial_review_rejected_path_returns_revision():
    assert _run_tutorial(approved=False) == "[revise_reply:draft]"


def test_tutorial_review_has_inspectable_deployment_contract():
    workflow, module = load_workflow_spec(
        "examples/tutorial_review.py:tutorial_review"
    )
    model = workflow_semantics(workflow, module)

    assert model["lifelines"] == ["Requester", "Writer", "Reviewer"]
    assert model["controls"][0]["owner"] == "Reviewer"
    assert model["action_definitions"]["approve_reply"]["kind"] == "human"
    assert model["deployment"]["name"] == "tutorial-review"
    assert model["deployment"]["fields"]["openai_api_key"]["secret"] is True
