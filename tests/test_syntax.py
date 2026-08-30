"""Tests for Layer 1: IR nodes, participation_set, and seq."""

import inspect
import math
import pytest

from zippergen.syntax import (
    EmptyStmt, MsgStmt, ActStmt, SkipStmt, SeqStmt, IfStmt, WhileStmt,
    SendStmt, RecvStmt, IfRecvStmt, WhileRecvStmt,
    Json, Lifeline, Var, VarExpr, LitExpr, Workflow,
    is_json_value, participation_set, seq, validate_zvalue,
)


def test_workflow_configure_has_no_retired_browser_options():
    parameters = inspect.signature(Workflow.configure).parameters

    assert "ui" not in parameters
    assert "show_decisions" not in parameters


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

A = Lifeline("A")
B = Lifeline("B")
C = Lifeline("C")

x = Var("x", int)
y = Var("y", str)


# ---------------------------------------------------------------------------
# Lifeline and Var equality (frozen dataclasses)
# ---------------------------------------------------------------------------

def test_lifeline_equality():
    assert Lifeline("A") == Lifeline("A")
    assert Lifeline("A") != Lifeline("B")


def test_lifeline_hashable():
    s = {Lifeline("A"), Lifeline("A"), Lifeline("B")}
    assert len(s) == 2


def test_var_equality():
    assert Var("x", int) == Var("x", int)
    assert Var("x", int) != Var("x", str)
    assert Var("x", int) != Var("y", int)


def test_json_is_a_first_class_coordination_type():
    value = {
        "title": "Call",
        "attempts": 2,
        "approved": False,
        "notes": None,
        "slots": ["Thursday", {"hour": 11.5}],
    }

    annotation = Json @ A
    variable = Var("payload", Json, default=value)

    assert annotation.type is Json
    assert annotation.lifeline == A
    assert is_json_value(value)
    assert validate_zvalue(value, Json) is value
    assert hash(variable) == hash(Var("payload", Json, default=value))


@pytest.mark.parametrize(
    "value",
    [
        {"bad": (1, 2)},
        {1: "non-string key"},
        {"number": math.inf},
        object(),
    ],
)
def test_json_rejects_values_that_do_not_round_trip_portably(value):
    assert not is_json_value(value)
    with pytest.raises(TypeError, match="valid Json value"):
        validate_zvalue(value, Json)


def test_json_rejects_circular_values():
    value = []
    value.append(value)

    assert not is_json_value(value)
    with pytest.raises(TypeError, match="circular reference"):
        Var("payload", Json, default=value)


def test_json_nesting_is_bounded_without_recursion_error():
    accepted = None
    for _ in range(128):
        accepted = [accepted]
    too_deep = [accepted]

    assert is_json_value(accepted)
    assert not is_json_value(too_deep)
    with pytest.raises(TypeError, match="nests deeper than 128 levels"):
        validate_zvalue(too_deep, Json)


def test_json_rejects_container_subclasses():
    class CustomList(list):
        pass

    with pytest.raises(TypeError, match="expected a built-in"):
        validate_zvalue(CustomList([1, 2]), Json)


def test_scalar_float_must_be_finite():
    with pytest.raises(TypeError, match="not a finite number"):
        validate_zvalue(math.nan, float)


def test_tuple_values_must_be_portable_recursively():
    value = ("ok", [1, {"nested": (True, None)}])

    assert validate_zvalue(value, tuple) is value
    with pytest.raises(TypeError, match="not portable"):
        validate_zvalue((object(),), tuple)


# ---------------------------------------------------------------------------
# seq — right-associative fold with EmptyStmt identity
# ---------------------------------------------------------------------------

def test_seq_no_args():
    assert seq() == EmptyStmt()


def test_seq_single():
    s = SkipStmt(A)
    assert seq(s) is s


def test_seq_two():
    s1 = SkipStmt(A)
    s2 = SkipStmt(B)
    result = seq(s1, s2)
    assert isinstance(result, SeqStmt)
    assert result.first is s1
    assert result.second is s2


def test_seq_three_is_right_associative():
    s1 = SkipStmt(A)
    s2 = SkipStmt(B)
    s3 = SkipStmt(C)
    result = seq(s1, s2, s3)
    assert isinstance(result, SeqStmt)
    assert result.first is s1
    assert isinstance(result.second, SeqStmt)
    assert result.second.first is s2
    assert result.second.second is s3


def test_seq_drops_empty_left():
    s = SkipStmt(A)
    assert seq(EmptyStmt(), s) is s


def test_seq_drops_empty_right():
    s = SkipStmt(A)
    assert seq(s, EmptyStmt()) is s


def test_seq_all_empty():
    assert seq(EmptyStmt(), EmptyStmt()) == EmptyStmt()


# ---------------------------------------------------------------------------
# participation_set — L(P)
# ---------------------------------------------------------------------------

def test_participation_empty():
    assert participation_set(EmptyStmt()) == frozenset()


def test_participation_msg():
    stmt = MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),))
    assert participation_set(stmt) == frozenset({A, B})


def test_participation_act():
    from zippergen.actions import pure
    @pure
    def f(v: int) -> int:
        return v
    stmt = ActStmt(A, f, (VarExpr(x),), (x,))
    assert participation_set(stmt) == frozenset({A})


def test_participation_skip():
    assert participation_set(SkipStmt(A)) == frozenset({A})


def test_participation_seq():
    s1 = SkipStmt(A)
    s2 = SkipStmt(B)
    assert participation_set(SeqStmt(s1, s2)) == frozenset({A, B})


def test_participation_if():
    stmt = IfStmt(
        condition=lambda _e: True,
        owner=A,
        branch_true=SkipStmt(B),
        branch_false=SkipStmt(C),
    )
    assert participation_set(stmt) == frozenset({A, B, C})


def test_participation_if_owner_only():
    """Owner is always included even if branches are empty."""
    stmt = IfStmt(
        condition=lambda _e: True,
        owner=A,
        branch_true=EmptyStmt(),
        branch_false=EmptyStmt(),
    )
    assert participation_set(stmt) == frozenset({A})


def test_participation_while():
    stmt = WhileStmt(
        condition=lambda _e: False,
        owner=A,
        body=SkipStmt(B),
        exit_body=SkipStmt(C),
    )
    assert participation_set(stmt) == frozenset({A, B, C})


def test_participation_send():
    stmt = SendStmt(A, (VarExpr(x),), B)
    assert participation_set(stmt) == frozenset({A})


def test_participation_recv():
    stmt = RecvStmt(A, (VarExpr(y),), B)
    assert participation_set(stmt) == frozenset({A})


# ---------------------------------------------------------------------------
# A guard decides; it does not compute
# ---------------------------------------------------------------------------


def test_a_guard_that_calls_out_is_refused(tmp_path):
    """A guard runs inside the durable write transaction.

    `role_runner` rolls that transaction back before every external action, on
    purpose. A condition has no way to ask for the same treatment, so a guard
    that reaches outside holds the write lock for the length of the call and
    blocks every other participant. A deployed mailbox poller did exactly that
    on every cycle.
    """

    source = '''
from zippergen import Lifeline, Var, effect, workflow

Mailbox = Lifeline("Mailbox")
seen = Var("seen", int, default=0)


@effect
def mail_present() -> bool:
    return True


@effect
def take_one() -> int:
    return 1


@workflow
def poller() -> int:
    if mail_present() @ Mailbox:
        Mailbox: seen = take_one()
    return seen @ Mailbox
'''
    path = tmp_path / "computed_guard.py"
    path.write_text(source)

    import importlib.util

    spec = importlib.util.spec_from_file_location("computed_guard", path)
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(TypeError) as caught:
        spec.loader.exec_module(module)

    message = str(caught.value)
    assert "calls mail_present()" in message
    # The refusal must name the idiom that works.
    assert "Compute it in an action first" in message


def test_a_guard_over_variables_is_accepted(tmp_path):
    source = '''
from zippergen import Lifeline, Var, effect, workflow

Mailbox = Lifeline("Mailbox")
seen = Var("seen", int, default=0)
has_mail = Var("has_mail", bool, default=False)


@effect
def mail_present() -> bool:
    return True


@effect
def take_one() -> int:
    return 1


@workflow
def poller() -> int:
    Mailbox: has_mail = mail_present()
    if has_mail @ Mailbox:
        Mailbox: seen = take_one()
    return seen @ Mailbox
'''
    path = tmp_path / "plain_guard.py"
    path.write_text(source)

    import importlib.util

    spec = importlib.util.spec_from_file_location("plain_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.poller.name == "poller"


def test_a_causal_past_formula_guard_is_still_accepted(tmp_path):
    """At[A](atom(...)) is a call, but the monitor evaluates it, not the step."""

    source = '''
from zippergen import At, Lifeline, Var, atom, effect, workflow

Owner = Lifeline("Owner")
seen = Var("seen", int, default=0)


@effect
def take_one() -> int:
    return 1


@workflow
def watched() -> int:
    if At[Owner](atom(lambda env: env.get("seen", 0) > 0, version="v1")) @ Owner:
        Owner: seen = take_one()
    return seen @ Owner
'''
    path = tmp_path / "formula_guard.py"
    path.write_text(source)

    import importlib.util

    spec = importlib.util.spec_from_file_location("formula_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.watched.name == "watched"
