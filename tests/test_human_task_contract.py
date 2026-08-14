import pytest

from zippergen.actions import human
from zippergen.human_tasks import (
    build_human_task_spec,
    human_task_result_from_value,
    validate_human_action_result,
    validate_human_task_spec,
)


@human(kind="ack", outputs=["seen: bool"], instruction="Read {notice}")
def acknowledge(notice: str) -> None: ...


@human(kind="select", outputs=["choice: str"], prefill="red\ngreen")
def choose() -> None: ...


def test_action_builds_complete_canonical_task_spec():
    spec = build_human_task_spec(acknowledge, {"notice": "the update"})

    assert spec == {
        "kind": "ack",
        "output": "seen",
        "output_type": "bool",
        "rendered": {
            "context": None,
            "instruction": "Read the update",
            "prefill": None,
        },
        "submit_label": None,
        "cancel_label": None,
    }


def test_task_spec_requires_the_response_type():
    with pytest.raises(ValueError, match="output_type"):
        validate_human_task_spec({"kind": "confirm", "output": "approved"})


def test_acknowledgement_cannot_be_declined_by_any_backend():
    with pytest.raises(ValueError, match="only be completed affirmatively"):
        validate_human_action_result(acknowledge, {"notice": "x"}, {"seen": False})


def test_select_response_must_be_one_of_the_rendered_options():
    spec = build_human_task_spec(choose, {})

    assert human_task_result_from_value(spec, "2") == {"choice": "green"}
    with pytest.raises(ValueError, match="Choose a number between 1 and 2"):
        human_task_result_from_value(spec, "blue")
