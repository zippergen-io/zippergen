"""Execution tests for runtime-generated planner workflows."""

import pytest

from zippergen.actions import planner, pure
from zippergen.builder import workflow
from zippergen.syntax import Lifeline


Planner = Lifeline("Planner")


@pure
def subtract_float(a: float, b: float) -> float:
    return a - b


@planner(
    description="Generate a numeric calculator workflow.",
    actions=[subtract_float],
    lifelines=["Calculator"],
    max_retries=1,
)
def plan_float(request: str) -> float: ...


@planner(
    description="Generate a text workflow with a schema-checked LLM action.",
    actions=[],
    lifelines=["Writer"],
    allow=["llm"],
    max_retries=1,
)
def plan_text(request: str) -> str: ...


@workflow
def planned_float(expression: str @ Planner) -> float:
    Planner: result = plan_float(expression)
    return result @ Planner


@workflow
def planned_text(request: str @ Planner) -> str:
    Planner: result = plan_text(request)
    return result @ Planner


def test_planner_preserves_float_intermediate_types():
    spec = """\
@workflow
def generated_workflow() -> float:
    Calculator: sub1 = subtract_float(2.0, 4.0)
    Calculator(sub1) >> Planner(result)
    return result @ Planner
"""

    def backend(action, inputs):
        assert action.name == "_generate_spec"
        return {"workflow_spec": spec}

    planned_float.configure(backend=backend, timeout=5)

    assert plan_float.outputs == (("plan_float", float),)
    assert planned_float(expression="2 - 4") == -2.0


def test_planner_executes_schema_checked_generated_llm_action():
    spec = '''\
@llm(
    system="Draft safely.",
    user="{request}",
    parse="text",
    outputs=(("draft", str),),
)
def draft(request: str): ...

@workflow
def generated_workflow(request: str @ Planner) -> str:
    Planner(request) >> Writer(request)
    Writer: draft_value = draft(request)
    Writer(draft_value) >> Planner(draft_value)
    return draft_value @ Planner
'''

    def backend(action, inputs):
        if action.name == "_generate_spec":
            return {"workflow_spec": spec}
        assert action.name == "draft"
        assert inputs == {"request": "Write a safe answer."}
        return {"draft": "Safe answer."}

    planned_text.configure(backend=backend, timeout=5)

    assert planned_text(request="Write a safe answer.") == "Safe answer."


def test_planner_does_not_validate_extra_candidate_after_attempt_budget():
    invalid_spec = """\
@workflow
def wrong_name() -> float:
    Calculator: sub1 = subtract_float(2.0, 4.0)
    Calculator(sub1) >> Planner(result)
    return result @ Planner
"""
    valid_second_spec = """\
@workflow
def generated_workflow() -> float:
    Calculator: sub1 = subtract_float(2.0, 4.0)
    Calculator(sub1) >> Planner(result)
    return result @ Planner
"""
    calls = []

    def backend(action, inputs):
        calls.append(action.name)
        if len(calls) == 1:
            return {"workflow_spec": invalid_spec}
        return {"workflow_spec": valid_second_spec}

    planned_float.configure(backend=backend, timeout=5)

    with pytest.raises(RuntimeError) as exc:
        planned_float(expression="2 - 4")

    assert "Planner failed after 1 attempt" in str(exc.value)
    assert calls == ["_generate_spec"]


def test_planner_rejects_module_code_without_executing_it(tmp_path):
    marker = tmp_path / "planner-owned.txt"
    spec = f'''\
open({str(marker)!r}, "w").write("owned")

@workflow
def generated_workflow() -> float:
    Calculator: sub1 = subtract_float(2.0, 4.0)
    Calculator(sub1) >> Planner(result)
    return result @ Planner
'''

    def backend(action, inputs):
        assert action.name == "_generate_spec"
        return {"workflow_spec": spec}

    planned_float.configure(backend=backend, timeout=5)

    with pytest.raises(RuntimeError, match="Unsafe generated planner code"):
        planned_float(expression="2 - 4")

    assert not marker.exists()


def test_planner_rejects_generated_pure_extension():
    with pytest.raises(ValueError, match="generated @pure actions are disabled"):

        @planner(
            description="Generate an unsafe helper.",
            actions=[],
            lifelines=["Worker"],
            allow=["pure"],
        )
        def unsafe_planner(request: str) -> str: ...
