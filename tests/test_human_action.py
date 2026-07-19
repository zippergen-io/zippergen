from zippergen.syntax import HumanAction
import pytest

def test_human_action_fields():
    action = HumanAction(
        name="review_plan",
        inputs=(("plan", str),),
        output="approved",
        output_type=bool,
        kind="confirm",
        instruction="Approve this plan?",
    )
    assert action.name == "review_plan"
    assert action.inputs == (("plan", str),)
    assert action.output == "approved"
    assert action.output_type is bool
    assert action.kind == "confirm"
    assert action.instruction == "Approve this plan?"
    assert action.context is None
    assert action.prefill is None


def test_cli_human_backend_claims_resumed_pending_tasks():
    from zippergen.human_backends import make_cli_human_backend

    backend = make_cli_human_backend()

    assert getattr(backend, "claims_pending_human_tasks", False) is True
    assert getattr(backend, "requires_main_thread", False) is True

def test_human_action_select():
    action = HumanAction(
        name="choose",
        inputs=(("plan", str),),
        output="decision",
        output_type=str,
        kind="select",
        context="{plan}",
        prefill="approve\nreject\nescalate",
    )
    assert action.kind == "select"
    assert action.output_type is str
    assert action.prefill == "approve\nreject\nescalate"

def test_human_action_repr():
    action = HumanAction(
        name="review_plan",
        inputs=(("plan", str),),
        output="approved",
        output_type=bool,
        kind="confirm",
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
        kind="confirm",
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
            kind="confirm",
        )

def test_human_decorator_confirm():
    from zippergen.actions import human

    @human(kind="confirm", instruction="Approve this plan?", outputs=["approved: bool"])
    def review_plan(plan: str): pass

    assert isinstance(review_plan, HumanAction)
    assert review_plan.name == "review_plan"
    assert review_plan.inputs == (("plan", str),)
    assert review_plan.output == "approved"
    assert review_plan.output_type is bool
    assert review_plan.kind == "confirm"
    assert review_plan.instruction == "Approve this plan?"

def test_human_decorator_edit():
    from zippergen.actions import human

    @human(kind="edit", context="{plan}", prefill="{plan}",
           instruction="Edit this plan:", outputs=["comment: str"])
    def edit_plan(plan: str): pass

    assert edit_plan.output == "comment"
    assert edit_plan.output_type is str
    assert edit_plan.kind == "edit"
    assert edit_plan.context == "{plan}"
    assert edit_plan.prefill == "{plan}"

def test_human_decorator_select():
    from zippergen.actions import human

    @human(kind="select", context="{plan}",
           prefill="approve\nreject\nescalate", outputs=["decision: str"])
    def choose_action(plan: str): pass

    assert choose_action.output == "decision"
    assert choose_action.output_type is str
    assert choose_action.prefill == "approve\nreject\nescalate"

def test_human_decorator_ack():
    from zippergen.actions import human

    @human(kind="ack", context="{event}", instruction="Event created.",
           outputs=["ack: bool"], submit_label="Noted")
    def acknowledge(event: str): pass

    assert acknowledge.output == "ack"
    assert acknowledge.output_type is bool
    assert acknowledge.kind == "ack"
    assert acknowledge.submit_label == "Noted"
    assert acknowledge.cancel_label is None

def test_human_decorator_input():
    from zippergen.actions import human

    @human(kind="input", instruction="Any additional notes?",
           outputs=["notes: str"])
    def add_notes(): pass

    assert add_notes.output == "notes"
    assert add_notes.output_type is str
    assert add_notes.kind == "input"

def test_human_decorator_bad_placeholder():
    from zippergen.actions import human

    with pytest.raises(TypeError, match="unknown variables"):
        @human(kind="confirm", context="{typo}", instruction="Approve?",
               outputs=["approved: bool"])
        def bad_action(plan: str): pass

def test_human_decorator_confirm_requires_bool():
    from zippergen.actions import human

    with pytest.raises(TypeError, match="bool"):
        @human(kind="confirm", outputs=["decision: str"])
        def bad_confirm(plan: str): pass

def test_human_decorator_ack_requires_bool():
    from zippergen.actions import human

    with pytest.raises(TypeError, match="bool"):
        @human(kind="ack", outputs=["ack: str"])
        def bad_ack(): pass

def test_human_decorator_edit_requires_str():
    from zippergen.actions import human

    with pytest.raises(TypeError, match="str"):
        @human(kind="edit", outputs=["decision: bool"])
        def bad_edit(plan: str): pass

def test_human_decorator_bad_kind():
    from zippergen.actions import human

    with pytest.raises(ValueError, match="unsupported kind"):
        @human(kind="unknown", outputs=["x: str"])
        def bad_kind(): pass


# CLI Backend Tests
from unittest.mock import patch
from zippergen.human_backends import make_cli_human_backend


def _make_confirm(name="ask"):
    return HumanAction(
        name=name,
        inputs=(("plan", str),),
        output="result",
        output_type=bool,
        kind="confirm",
        instruction="Question: {plan}",
    )

def _make_edit(name="ask"):
    return HumanAction(
        name=name,
        inputs=(("plan", str),),
        output="result",
        output_type=str,
        kind="edit",
        instruction="Edit this:",
        prefill="{plan}",
    )

def _make_select(name="ask"):
    return HumanAction(
        name=name,
        inputs=(("plan", str),),
        output="result",
        output_type=str,
        kind="select",
        instruction="Choose:",
        prefill="approve\nreject\nescalate",
    )

def _make_ack(name="ask"):
    return HumanAction(
        name=name,
        inputs=(("event", str),),
        output="result",
        output_type=bool,
        kind="ack",
        instruction="Event created.",
    )


def test_cli_backend_confirm_yes():
    backend = make_cli_human_backend()
    action = _make_confirm()
    with patch("builtins.input", return_value="y"):
        result = backend(action, {"plan": "do something"})
    assert result == {"result": True}


def test_cli_backend_confirm_no():
    backend = make_cli_human_backend()
    action = _make_confirm()
    with patch("builtins.input", return_value="n"):
        result = backend(action, {"plan": "do something"})
    assert result == {"result": False}


def test_cli_backend_edit():
    backend = make_cli_human_backend()
    action = _make_edit()
    with patch("builtins.input", return_value="looks good"):
        result = backend(action, {"plan": "do something"})
    assert result == {"result": "looks good"}


def test_cli_backend_edit_keep_prefill():
    backend = make_cli_human_backend()
    action = _make_edit()
    with patch("builtins.input", return_value=""):
        result = backend(action, {"plan": "do something"})
    assert result == {"result": "do something"}


def test_cli_backend_select():
    backend = make_cli_human_backend()
    action = _make_select()
    with patch("builtins.input", side_effect=["99", "2"]):
        result = backend(action, {"plan": "do something"})
    assert result == {"result": "reject"}


def test_cli_backend_ack():
    backend = make_cli_human_backend()
    action = _make_ack()
    with patch("builtins.input", return_value=""):
        result = backend(action, {"event": "Meeting at 10am"})
    assert result == {"result": True}


# Runtime integration tests
from zippergen.syntax import Lifeline, Var
from zippergen.actions import human, pure
from zippergen.builder import workflow
from zippergen.runtime import run

_Human = Lifeline("Human")
_Planner = Lifeline("Planner")

_plan = Var("plan", str)
_approved = Var("approved", bool)

@pure
def make_task(n: int) -> str:
    return f"task-{n}"

@human(kind="confirm", instruction="Approve: {plan}?", outputs=["approved: bool"])
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


def test_runtime_cli_human_input_runs_only_on_main_thread():
    import threading

    input_threads = []
    backend = make_cli_human_backend(
        input_func=lambda prompt: input_threads.append(threading.current_thread())
        or "y",
        output_func=lambda line: None,
    )

    result = run(
        approval_flow,
        [_Planner, _Human],
        {"Planner": {"n": 42}},
        human_backend=backend,
    )

    assert result is True
    assert input_threads == [threading.main_thread()]


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
