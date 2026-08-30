"""The per-lifeline ownership rule is shared by every workflow source."""

import copy
import json

import pytest

from zippergen.actions import pure
from zippergen.availability import (
    AvailabilityViolation,
    workflow_availability_error,
)
from zippergen.planner import _validate_planner_spec
from zippergen.runtime import run
from zippergen.serve import main
from zippergen.syntax import (
    ActStmt,
    EmptyStmt,
    IfStmt,
    Json,
    Lifeline,
    LitExpr,
    MsgStmt,
    ParallelStmt,
    SeqStmt,
    SkipStmt,
    Var,
    VarExpr,
    WhileStmt,
    Workflow,
)

A = Lifeline("A")
B = Lifeline("B")
C = Lifeline("C")


@pure
def make_text() -> str:
    return "value"


@pure
def echo_text(value: str) -> str:
    return value


@pure
def echo_json(value: Json) -> Json:
    return value


@pure
def join_text(left: str, right: str) -> str:
    return f"{left}:{right}"


def _workflow(body, *, inputs=(), outputs=(), ns=None) -> Workflow:
    return Workflow(
        name="availability_test",
        inputs=inputs,
        output_type=(outputs[0][0].type if len(outputs) == 1 else type(None)),
        vars=(),
        body=body,
        outputs=outputs,
        ns=ns or {},
    )


def test_handwritten_and_generated_workflows_share_the_missing_receive_rule():
    value = Var("value", str)
    result = Var("result", str)
    handwritten = _workflow(
        SeqStmt(
            ActStmt(A, make_text, (), (value,)),
            ActStmt(B, echo_text, (VarExpr(value),), (result,)),
        ),
        outputs=((result, B),),
    )
    generated = """\
@workflow
def generated_workflow() -> str:
    A: value = make_text()
    B: result = echo_text(value)
    return result @ B
"""

    handwritten_error = workflow_availability_error(handwritten)
    generated_error = _validate_planner_spec(
        generated,
        "A",
        {"make_text": 1, "echo_text": 1},
        allowed_lifelines={"A", "B"},
    )

    assert handwritten_error is not None
    assert generated_error is not None
    for error in (handwritten_error, generated_error):
        assert "B" in error
        assert "value" in error


def test_branch_join_keeps_only_values_available_on_both_paths():
    flag = Var("flag", bool)
    value = Var("value", str)
    result = Var("result", str)
    body = SeqStmt(
        IfStmt(
            lambda env: env.flag,
            A,
            ActStmt(A, make_text, (), (value,)),
            SkipStmt(A),
        ),
        ActStmt(A, echo_text, (VarExpr(value),), (result,)),
    )
    workflow = _workflow(
        body,
        inputs=(("flag", bool, A),),
        outputs=((result, A),),
    )

    error = workflow_availability_error(workflow)

    assert error is not None
    assert "value" in error


def test_loop_body_does_not_establish_a_value_when_zero_iterations_are_possible():
    flag = Var("flag", bool)
    value = Var("value", str)
    result = Var("result", str)
    body = SeqStmt(
        WhileStmt(
            lambda env: env.flag,
            A,
            ActStmt(A, make_text, (), (value,)),
            EmptyStmt(),
        ),
        ActStmt(A, echo_text, (VarExpr(value),), (result,)),
    )
    workflow = _workflow(
        body,
        inputs=(("flag", bool, A),),
        outputs=((result, A),),
    )

    error = workflow_availability_error(workflow)

    assert error is not None
    assert "value" in error


def test_guard_must_be_available_at_its_owner():
    flag = Var("flag", bool)
    workflow = _workflow(
        IfStmt(lambda env: env.flag, B, SkipStmt(A), SkipStmt(A)),
        inputs=(("flag", bool, A),),
    )

    error = workflow_availability_error(workflow)

    assert error is not None
    assert "if guard" in error
    assert "flag" in error
    assert "B" in error


def test_generated_guard_uses_the_same_availability_state():
    generated = """\
@workflow
def generated_workflow(flag: bool @ A) -> str:
    if flag @ B:
        A: result = make_text()
    else:
        A: result = make_text()
    return result @ A
"""

    error = _validate_planner_spec(
        generated,
        "A",
        {"make_text": 1},
        allowed_lifelines={"A", "B"},
    )

    assert error is not None
    assert "if guard" in error
    assert "flag" in error
    assert "B" in error


def test_explicit_json_none_default_is_available_and_distinct_from_no_default():
    payload = Var("payload", Json, default=None)
    result = Var("result", Json)
    workflow = _workflow(
        ActStmt(B, echo_json, (VarExpr(payload),), (result,)),
        outputs=((result, B),),
        ns={"payload": payload},
    )

    assert payload.has_default is True
    missing = Var("missing", Json)
    assert missing.has_default is False
    assert copy.deepcopy(missing).has_default is False
    assert workflow_availability_error(workflow) is None
    assert run(workflow, [B], {}) is None


def test_parallel_branches_may_contribute_independent_values_after_the_join():
    left = Var("left", str)
    right = Var("right", str)
    result = Var("result", str)
    body = SeqStmt(
        ParallelStmt(
            (
                MsgStmt(A, (LitExpr("L", str),), C, (VarExpr(left),)),
                MsgStmt(B, (LitExpr("R", str),), C, (VarExpr(right),)),
            )
        ),
        ActStmt(C, join_text, (VarExpr(left), VarExpr(right)), (result,)),
    )
    workflow = _workflow(body, outputs=((result, C),))

    assert workflow_availability_error(workflow) is None


def test_runtime_backstop_refuses_an_invalid_direct_ir_workflow():
    value = Var("value", str)
    result = Var("result", str)
    workflow = _workflow(
        ActStmt(B, echo_text, (VarExpr(value),), (result,)),
        outputs=((result, B),),
    )

    with pytest.raises(AvailabilityViolation, match="value"):
        run(workflow, [B], {})


def test_cli_validate_rejects_the_missing_receive(tmp_path, capsys):
    source = tmp_path / "missing_receive.py"
    source.write_text(
        """\
from zippergen import Lifeline, pure, workflow

A = Lifeline("A")
B = Lifeline("B")

@pure
def produce() -> str:
    return "real value"

@pure
def consume(value: str) -> str:
    return value

@workflow
def missing_receive() -> str:
    A: value = produce()
    B: result = consume(value)
    return result @ B
"""
    )

    rc = main(["validate", f"{source}:missing_receive", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert rc == 1
    assert payload["valid"] is False
    check = next(
        item for item in payload["checks"]
        if item["name"] == "variable availability"
    )
    assert check["status"] == "fail"
    assert "consume" in check["detail"]
    assert "value" in check["detail"]


def test_cli_run_reports_the_violation_without_a_traceback(tmp_path, capsys):
    """A run refuses an invalid workflow the way validation reports it.

    The rule lives in one place; the CLI only has to present it. An author
    mistake must read as a message, not as an interpreter stack trace.
    """

    source = tmp_path / "missing_receive_run.py"
    source.write_text(
        """\
from zippergen import Lifeline, pure, workflow

A = Lifeline("A")
B = Lifeline("B")

@pure
def produce() -> str:
    return "real value"

@pure
def consume(value: str) -> str:
    return value

@workflow
def missing_receive() -> str:
    A: value = produce()
    B: result = consume(value)
    return result @ B
"""
    )

    with pytest.raises(SystemExit) as raised:
        main(["run", "--workflow", f"{source}:missing_receive"])

    message = str(raised.value)
    assert "variable availability" in message
    assert "consume" in message
    assert "value" in message
    assert "Traceback" not in message
