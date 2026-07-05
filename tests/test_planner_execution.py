"""Execution tests for runtime-generated planner workflows."""

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


@workflow
def planned_float(expression: str @ Planner) -> float:
    Planner: result = plan_float(expression)
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

    planned_float.configure(backend=backend, ui=False, timeout=5)

    assert plan_float.outputs == (("plan_float", float),)
    assert planned_float(expression="2 - 4") == -2.0
