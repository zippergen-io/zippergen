from zippergen.syntax import HumanAction
import pytest

def test_human_action_fields():
    action = HumanAction(
        name="review_plan",
        inputs=(("plan", str),),
        output="approved",
        output_type=bool,
        prompt="Approve this plan?\n\n{plan}",
        options=None,
    )
    assert action.name == "review_plan"
    assert action.inputs == (("plan", str),)
    assert action.output == "approved"
    assert action.output_type is bool
    assert action.prompt == "Approve this plan?\n\n{plan}"
    assert action.options is None

def test_human_action_choice():
    action = HumanAction(
        name="choose",
        inputs=(("plan", str),),
        output="decision",
        output_type=str,
        prompt="Choose: {plan}",
        options=("approve", "reject", "escalate"),
    )
    assert action.options == ("approve", "reject", "escalate")
    assert action.output_type is str

def test_human_action_repr():
    action = HumanAction(
        name="review_plan",
        inputs=(("plan", str),),
        output="approved",
        output_type=bool,
        prompt="Approve? {plan}",
    )
    r = repr(action)
    assert r.startswith("HumanAction(")
    assert "review_plan" in r
    assert "approved: bool" in r

def test_human_action_immutable():
    action = HumanAction(
        name="review_plan",
        inputs=(("plan", str),),
        output="approved",
        output_type=bool,
        prompt="Approve? {plan}",
    )
    with pytest.raises((AttributeError, TypeError)):
        action.name = "changed"  # type: ignore

def test_human_action_invalid_output_type():
    with pytest.raises(ValueError, match="output_type must be bool or str"):
        HumanAction(
            name="bad",
            inputs=(),
            output="result",
            output_type=int,
            prompt="Hello",
        )

def test_human_decorator_bool():
    from zippergen.actions import human

    @human(prompt="Approve this plan?\n\n{plan}", outputs=["approved: bool"])
    def review_plan(plan: str): pass

    assert isinstance(review_plan, HumanAction)
    assert review_plan.name == "review_plan"
    assert review_plan.inputs == (("plan", str),)
    assert review_plan.output == "approved"
    assert review_plan.output_type is bool
    assert review_plan.prompt == "Approve this plan?\n\n{plan}"
    assert review_plan.options is None

def test_human_decorator_text():
    from zippergen.actions import human

    @human(prompt="Add a comment about {plan}:", outputs=["comment: str"])
    def add_comment(plan: str): pass

    assert add_comment.output == "comment"
    assert add_comment.output_type is str
    assert add_comment.options is None

def test_human_decorator_choice():
    from zippergen.actions import human

    @human(
        prompt="Choose an action for {plan}:",
        options=["approve", "reject", "escalate"],
        outputs=["decision: str"],
    )
    def choose_action(plan: str): pass

    assert choose_action.output == "decision"
    assert choose_action.output_type is str
    assert choose_action.options == ("approve", "reject", "escalate")

def test_human_decorator_bad_placeholder():
    from zippergen.actions import human

    with pytest.raises(TypeError, match="unknown variables"):
        @human(prompt="Approve {typo}?", outputs=["approved: bool"])
        def bad_action(plan: str): pass

def test_human_decorator_options_requires_str():
    from zippergen.actions import human

    with pytest.raises(TypeError, match="options.*str"):
        @human(
            prompt="Choose: {plan}",
            options=["a", "b"],
            outputs=["decision: bool"],
        )
        def bad_choice(plan: str): pass


# CLI Backend Tests
from unittest.mock import patch
from zippergen.human_backends import make_cli_human_backend


def _make_action(output_type, options=None):
    return HumanAction(
        name="ask",
        inputs=(("plan", str),),
        output="result",
        output_type=output_type,
        prompt="Question: {plan}",
        options=options,
    )


def test_cli_backend_bool_yes():
    backend = make_cli_human_backend()
    action = _make_action(bool)
    with patch("builtins.input", return_value="y"):
        result = backend(action, {"plan": "do something"})
    assert result == {"result": True}


def test_cli_backend_bool_no():
    backend = make_cli_human_backend()
    action = _make_action(bool)
    with patch("builtins.input", return_value="n"):
        result = backend(action, {"plan": "do something"})
    assert result == {"result": False}


def test_cli_backend_text():
    backend = make_cli_human_backend()
    action = _make_action(str)
    with patch("builtins.input", return_value="looks good"):
        result = backend(action, {"plan": "do something"})
    assert result == {"result": "looks good"}


def test_cli_backend_choice():
    backend = make_cli_human_backend()
    action = _make_action(str, options=("approve", "reject", "escalate"))
    with patch("builtins.input", side_effect=["99", "2"]):
        result = backend(action, {"plan": "do something"})
    assert result == {"result": "reject"}


# Runtime integration tests
from zippergen.syntax import Lifeline, Var
from zippergen.actions import human, pure
from zippergen.builder import workflow
from zippergen.runtime import run

_Human = Lifeline("Human")
_Planner = Lifeline("Planner")

# Intermediate variables must be declared as Var objects in module scope
# so the @workflow DSL can resolve them by name during AST execution.
_plan = Var("plan", str)
_approved = Var("approved", bool)

@pure
def make_task(n: int) -> str:
    return f"task-{n}"

@human(prompt="Approve: {plan}?", outputs=["approved: bool"])
def review_plan(plan: str): pass

@workflow
def approval_flow(n: int @ _Planner) -> bool:
    with _Planner:
        _plan = make_task(n)
    _Planner(_plan) >> _Human(_plan)
    with _Human:
        _approved = review_plan(_plan)
    return _approved @ _Human

def test_runtime_human_action():
    def mock_human_backend(action, inputs):
        return {action.output: True}

    result = run(
        approval_flow,
        [_Planner, _Human],
        {"Planner": {"n": 42}},
        human_backend=mock_human_backend,
    )
    assert result is True

def test_runtime_human_action_reject():
    def mock_human_backend(action, inputs):
        return {action.output: False}

    result = run(
        approval_flow,
        [_Planner, _Human],
        {"Planner": {"n": 1}},
        human_backend=mock_human_backend,
    )
    assert result is False


def test_runtime_human_trace_kind():
    events = []

    def mock_human_backend(action, inputs):
        return {action.output: True}

    result = run(
        approval_flow,
        [_Planner, _Human],
        {"Planner": {"n": 7}},
        human_backend=mock_human_backend,
        trace=events.append,
    )
    assert result is True
    human_events = [
        event for event in events
        if event["type"] in {"act_start", "act"} and event["action"] == "review_plan"
    ]
    assert len(human_events) == 2
    assert {event["action_kind"] for event in human_events} == {"human"}
