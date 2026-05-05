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
