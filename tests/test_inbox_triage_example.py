"""A deployed service: watch a mailbox, classify a message, record it.

The point of this example is that a workflow does not grow because it is
deployed. These pin what can be checked without a Google credential: the shape
each participant runs, the decisions, and what stays out of the trace.

Running the loop itself needs live connectors. That is a real gap -- it is why
a larger example carries a hand-written fake-services mode -- and not something
this example should grow a harness for.
"""

from zippergen.serve import load_workflow_spec
from zippergen.view import ViewOptions, render_workflow

SPEC = "examples/inbox_triage.py:inbox_triage"


def _projection(agent: str) -> str:
    workflow, _module = load_workflow_spec(SPEC)
    return render_workflow(
        workflow, options=ViewOptions(detail="protocol", agent=agent)
    )


def test_the_classifier_waits_on_decisions_it_never_makes():
    """Mailbox owns the loop and the branch; the others are told."""

    text = _projection("Classifier")

    assert "while recv_decision('Mailbox')" in text
    assert "if recv_decision('Mailbox')" in text
    assert "still_working" not in text, "the classifier never evaluates the guard"
    assert "mailbox_has_mail" not in text


def test_the_recorder_runs_only_when_there_was_mail():
    text = _projection("Records")

    assert "record_message" in text
    assert "while recv_decision('Mailbox')" in text


def test_the_mailbox_owns_both_decisions():
    text = _projection("Mailbox")

    assert "send_decision('Classifier'" in text
    assert "send_decision('Records'" in text
    assert "still_working" in text


def test_an_unexpected_answer_becomes_other_rather_than_a_new_kind():
    """A model asked for one word may answer with a sentence."""

    _workflow, module = load_workflow_spec(SPEC)

    assert module.known_kind.fn("Request") == "request"
    assert module.known_kind.fn("  NOTICE  ") == "notice"
    assert module.known_kind.fn("I think this is a request.") == "other"
    assert module.known_kind.fn("") == "other"


def test_a_limit_of_zero_is_the_service_case():
    """A deployed poller has no message count at which it should stop."""

    _workflow, module = load_workflow_spec(SPEC)

    assert module.still_working.fn(0, 0) is True
    assert module.still_working.fn(10_000, 0) is True
    assert module.still_working.fn(2, 2) is False
    assert module.still_working.fn(1, 2) is True


def test_an_idle_poll_is_kept_out_of_the_trace():
    """A minute-by-minute poll would otherwise evict the events worth reading."""

    _workflow, module = load_workflow_spec(SPEC)

    assert module.mailbox_has_mail.visible is False
    assert module.wait_for_mail.visible is False
    assert module.record_message.visible is True, "real work stays visible"


def test_every_outside_action_declares_the_connector_it_uses():
    _workflow, module = load_workflow_spec(SPEC)

    assert module.take_one_message.connector == "incoming-mail"
    assert module.finish_message.connector == "incoming-mail"
    assert module.record_message.connector == "triage-records"
    assert {r.name for r in module.zippergen_connectors} == {
        "incoming-mail",
        "triage-records",
    }
