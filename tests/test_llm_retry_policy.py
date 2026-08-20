"""Retry and fallback for @llm actions.

One policy governs a whole LLM attempt -- transport, parsing, coercion and the
declared output types -- because a well-formed answer of the wrong type has
failed just as surely as a refused connection, and asking again may fix
either. These tests pin what is retried, what is not, and what a declared
fallback is allowed to do.
"""

import json
import threading
import time
from pathlib import Path

import pytest

from zippergen.actions import llm, pure
from zippergen.builder import workflow
from zippergen.llm_policy import (
    FOREVER,
    LLMInvalidResponseError,
    LLMPermanentError,
    LLMTransientError,
    RetryCancelled,
    attempt_llm_action,
    retry_delays,
)
from zippergen.syntax import Json, Lifeline


# ---------------------------------------------------------------------------
# Backends that fail in specific, recognised ways
# ---------------------------------------------------------------------------

def _failing_backend(failures, *, error, then=None):
    """Fail ``failures`` times with ``error``, then return ``then``."""

    calls = {"n": 0}

    def backend(action, inputs):
        calls["n"] += 1
        if calls["n"] <= failures:
            raise error()
        return dict(then or {})

    backend.calls = calls  # type: ignore[attr-defined]
    return backend


@pytest.fixture(autouse=True)
def _no_real_waiting(monkeypatch):
    """Keep the schedule honest but instant.

    The delays are asserted directly in their own test; everywhere else they
    would only make the suite slow.
    """

    monkeypatch.setattr("zippergen.llm_policy._FIRST_DELAY", 0.0)
    monkeypatch.setattr("zippergen.llm_policy._MAX_DELAY", 0.0)


def _action(**kwargs):
    defaults = dict(
        system="s", user="{message}", parse="text", outputs=[("draft", str)]
    )
    defaults.update(kwargs)

    @llm(**defaults)
    def draft_reply(message: str): ...

    return draft_reply


# ---------------------------------------------------------------------------
# 1. retries="forever"
# ---------------------------------------------------------------------------

def test_forever_survives_more_failures_than_the_old_fixed_limit():
    """The old backend budget was three. "forever" must not stop there."""

    action = _action(retries=FOREVER)
    attempts = {"n": 0}

    def call():
        attempts["n"] += 1
        if attempts["n"] <= 12:
            raise LLMTransientError("connection refused")
        return {"draft": "done"}

    assert attempt_llm_action(action, call) == {"draft": "done"}
    assert attempts["n"] == 13


# ---------------------------------------------------------------------------
# 2. cancellation
# ---------------------------------------------------------------------------

def test_cancellation_interrupts_an_unbounded_retry_wait(monkeypatch):
    """A stopped deployment must not sit out its backoff first."""

    monkeypatch.setattr("zippergen.llm_policy._FIRST_DELAY", 30.0)
    monkeypatch.setattr("zippergen.llm_policy._MAX_DELAY", 30.0)
    action = _action(retries=FOREVER)
    stop = threading.Event()

    def call():
        stop.set()
        raise LLMTransientError("still down")

    started = time.monotonic()
    with pytest.raises(RetryCancelled):
        attempt_llm_action(action, call, stop=stop)

    assert time.monotonic() - started < 5.0, "the wait was not interrupted"


def test_cancellation_never_produces_the_fallback():
    """Stopping is not failing, so it must not look like a model failure."""

    action = _action(retries=FOREVER, fallback="")
    stop = threading.Event()

    def call():
        stop.set()
        raise LLMTransientError("still down")

    with pytest.raises(RetryCancelled):
        attempt_llm_action(action, call, stop=stop)


# ---------------------------------------------------------------------------
# 3 and 4. fallbacks
# ---------------------------------------------------------------------------

def test_a_bounded_budget_returns_a_typed_single_output_fallback():
    action = _action(parse="bool", outputs=[("accepted", bool)], retries=3, fallback=False)
    call = _always(LLMInvalidResponseError("not a boolean"))

    result = attempt_llm_action(action, call)

    assert result == {"accepted": False}
    assert type(result["accepted"]) is bool
    assert call.calls["n"] == 4, "one first attempt plus three retries"


def test_a_multiple_output_fallback_is_validated_and_returned():
    action = _action(
        parse="json",
        outputs=[("draft", str), ("usable", bool)],
        retries=1,
        fallback={"draft": "", "usable": False},
    )

    result = attempt_llm_action(action, _always(LLMInvalidResponseError("bad")))

    assert result == {"draft": "", "usable": False}


def test_a_single_json_output_may_have_a_dict_fallback():
    """For one output the fallback is the value, even when that value is a dict."""

    from zippergen.llm_policy import decode_fallback

    action = _action(parse="json", outputs=[("record", Json)], fallback={"a": 1})

    assert decode_fallback(action) == {"record": {"a": 1}}


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (dict(retries=-1), "non-negative"),
        (dict(retries="sometimes"), "non-negative"),
        (dict(fallback=3), "expected str"),
        (
            dict(outputs=[("a", str), ("b", str)], parse="json", fallback="x"),
            "must be a mapping",
        ),
        (
            dict(outputs=[("a", str), ("b", str)], parse="json", fallback={"a": "x"}),
            "Missing: b",
        ),
        (
            dict(
                outputs=[("a", str), ("b", str)],
                parse="json",
                fallback={"a": "x", "b": "y", "c": "z"},
            ),
            "Unexpected: c",
        ),
    ],
    ids=["negative", "unknown-literal", "wrong-type", "not-a-mapping", "missing", "extra"],
)
def test_a_bad_declaration_fails_when_the_workflow_is_written(kwargs, expected):
    """The day a model misbehaves is the wrong day to discover a typo."""

    with pytest.raises((TypeError, ValueError), match=expected):
        _action(**kwargs)


# ---------------------------------------------------------------------------
# 5, 6, 7, 8. what is retried and what is not
# ---------------------------------------------------------------------------

def _always(error):
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        raise error

    call.calls = calls  # type: ignore[attr-defined]
    return call


def test_a_malformed_model_answer_is_retried():
    """An unusable answer is a failed call, not a failed workflow."""

    action = _action(retries=2)
    attempts = {"n": 0}

    def call():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise LLMInvalidResponseError("returned invalid JSON")
        return {"draft": "second time lucky"}

    assert attempt_llm_action(action, call) == {"draft": "second time lucky"}
    assert attempts["n"] == 2


def test_a_permanent_failure_is_not_retried():
    """A rejected key answers the same way however often it is asked."""

    action = _action(retries=FOREVER)
    call = _always(LLMPermanentError("401 invalid api key"))

    with pytest.raises(LLMPermanentError):
        attempt_llm_action(action, call)

    assert call.calls["n"] == 1


def test_a_permanent_failure_may_still_take_a_declared_fallback():
    action = _action(retries=FOREVER, fallback="")
    call = _always(LLMPermanentError("model does not exist"))

    assert attempt_llm_action(action, call) == {"draft": ""}
    assert call.calls["n"] == 1


def test_without_a_fallback_the_final_error_is_raised():
    action = _action(retries=2)
    call = _always(LLMTransientError("connection refused"))

    with pytest.raises(LLMTransientError, match="connection refused"):
        attempt_llm_action(action, call)

    assert call.calls["n"] == 3


@pytest.mark.parametrize(
    "error",
    [AttributeError("typo"), KeyError("missing"), ZeroDivisionError()],
    ids=["attribute", "key", "arithmetic"],
)
def test_a_defect_is_never_converted_into_a_fallback(error):
    """A bug in ZipperGen or a workflow must stay visible.

    Catching Exception here would turn a typo into a silent wrong answer, or
    into an unbounded wait, which is worse than a crash.
    """

    action = _action(retries=FOREVER, fallback="")

    with pytest.raises(type(error)):
        attempt_llm_action(action, _always(error))


# ---------------------------------------------------------------------------
# retry timing
# ---------------------------------------------------------------------------

def test_the_backoff_schedule_is_two_four_eight_capped_at_thirty(monkeypatch):
    monkeypatch.undo()

    assert [retry_delays(None, n) for n in range(6)] == [2, 4, 8, 16, 30, 30]


def test_a_providers_retry_after_wins_over_the_schedule(monkeypatch):
    monkeypatch.undo()

    assert retry_delays(5.0, 0) == 5.0
    assert retry_delays(120.0, 0) == 120.0, (
        "a provider instruction is honoured, not capped: ignoring a request "
        "for two minutes only earns another rate limit"
    )
    assert retry_delays(None, 9) == 30.0, "our own schedule is still capped"


# ---------------------------------------------------------------------------
# 11. concurrency
# ---------------------------------------------------------------------------

def test_two_lifelines_do_not_share_a_retry_budget():
    """The counter lives on the call stack, not on the action or backend."""

    action = _action(retries=2)
    seen: dict[str, int] = {}
    lock = threading.Lock()

    def run(name: str) -> None:
        attempts = {"n": 0}

        def call():
            attempts["n"] += 1
            if attempts["n"] <= 2:
                raise LLMTransientError("busy")
            return {"draft": name}

        result = attempt_llm_action(action, call)
        with lock:
            seen[name] = attempts["n"]
            assert result == {"draft": name}

    threads = [
        threading.Thread(target=run, args=(f"role-{index}",)) for index in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert seen == {f"role-{index}": 3 for index in range(4)}, (
        "each invocation must get its own three attempts"
    )


# ---------------------------------------------------------------------------
# 12. unchanged declarations
# ---------------------------------------------------------------------------

def test_an_undeclared_action_keeps_the_previous_budget_and_stays_loud():
    action = _action()

    assert action.retries == 3
    assert action.fallback_json is None

    call = _always(LLMTransientError("connection refused"))
    with pytest.raises(LLMTransientError):
        attempt_llm_action(action, call)
    assert call.calls["n"] == 4, "three retries, as the backend used to allow"


# ---------------------------------------------------------------------------
# 9. durable execution
# ---------------------------------------------------------------------------

Caller = Lifeline("RetryCaller")


@llm(
    system="Classify.",
    user="{message}",
    parse="bool",
    outputs=[("accepted", bool)],
    retries=1,
    fallback=False,
)
def classify_with_fallback(message: str): ...


@pure
def note(accepted: bool) -> str:
    return f"decided:{accepted}"


@workflow
def retry_fallback_workflow(message: str @ Caller) -> str:
    Caller: accepted = classify_with_fallback(message)
    Caller: decision = note(accepted)
    return decision @ Caller


def test_a_durable_run_commits_the_fallback_and_carries_on(tmp_path):
    """A fallback is an ordinary result: committed, and the run continues.

    The control state must not advance until a value is chosen, so the step
    after the action sees the fallback exactly as it would see a real answer.
    """

    from zippergen.sqlite_runner import run_sqlite

    def always_unusable(action, inputs):
        raise LLMInvalidResponseError("model returned prose, not a boolean")

    result = run_sqlite(
        retry_fallback_workflow,
        [Caller],
        {"RetryCaller": {"message": "please"}},
        store_path=str(tmp_path / "fallback.sqlite"),
        llm_backend=always_unusable,
        timeout=20,
    )

    assert result == "decided:False", "the run continued past the action"

    from zippergen.store import list_role_states, open_store

    conn = open_store(str(tmp_path / "fallback.sqlite"))
    try:
        states = {row["role"]: row for row in list_role_states(conn)}
        assert states["RetryCaller"]["status"] == "done", (
            "the role finished, so the fallback was committed like any result"
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 10. fingerprints
# ---------------------------------------------------------------------------

@llm(system="C.", user="{m}", parse="bool", outputs=[("ok", bool)])
def policy_plain(m: str): ...


@llm(system="C.", user="{m}", parse="bool", outputs=[("ok", bool)], retries=9)
def policy_more_retries(m: str): ...


@llm(
    system="C.", user="{m}", parse="bool", outputs=[("ok", bool)],
    retries=9, fallback=False,
)
def policy_with_fallback(m: str): ...


@llm(system="C.", user="{m}", parse="bool", outputs=[("ok", bool)], retries=FOREVER)
def policy_forever(m: str): ...


def test_the_semantic_record_changes_when_the_policy_changes():
    """A different retry or fallback declaration is a different action."""

    from zippergen.semantic import _action_definition

    plain = _action_definition(policy_plain)
    more_retries = _action_definition(policy_more_retries)
    with_fallback = _action_definition(policy_with_fallback)
    forever = _action_definition(policy_forever)

    assert plain != more_retries, "retries= must be part of the record"
    assert more_retries != with_fallback, "fallback= must be part of the record"
    assert plain != forever
    assert _action_definition(policy_plain) == plain, "stable for one declaration"


# ---------------------------------------------------------------------------
# The backend boundary classifies; it no longer retries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (429, LLMTransientError),
        (500, LLMTransientError),
        (503, LLMTransientError),
        (401, LLMPermanentError),
        (403, LLMPermanentError),
        (404, LLMPermanentError),
        (400, LLMPermanentError),
    ],
    ids=["rate-limited", "server", "unavailable", "unauthorized", "forbidden",
         "unknown-model", "bad-request"],
)
def test_http_status_decides_transient_from_permanent(status, expected, monkeypatch):
    """Provider-specific knowledge stays here; the policy stays elsewhere."""

    import io
    from urllib import request as urlrequest
    from urllib.error import HTTPError

    from zippergen.backends import _json_request

    def raising(req, timeout=None):
        raise HTTPError("u", status, "m", {"Retry-After": "7"}, io.BytesIO(b"detail"))

    monkeypatch.setattr("urllib.request.urlopen", raising)

    with pytest.raises(expected) as caught:
        _json_request(urlrequest.Request("http://example.invalid"), timeout=1)

    if expected is LLMTransientError:
        assert caught.value.retry_after == 7.0, "Retry-After must be honoured"


def test_the_backend_makes_exactly_one_attempt(monkeypatch):
    """The nested loop is gone: one call, one classification, one return."""

    from urllib import request as urlrequest
    from urllib.error import URLError

    from zippergen.backends import _json_request

    calls = {"n": 0}

    def raising(req, timeout=None):
        calls["n"] += 1
        raise URLError("refused")

    monkeypatch.setattr("urllib.request.urlopen", raising)

    with pytest.raises(LLMTransientError):
        _json_request(urlrequest.Request("http://example.invalid"), timeout=1)

    assert calls["n"] == 1, "the backend must not retry behind the policy"


# ---------------------------------------------------------------------------
# One namespace, and types that survive the fallback
# ---------------------------------------------------------------------------

Two = Lifeline("TwoOutputs")


@llm(
    system="s",
    user="{m}",
    parse="json",
    outputs=[("draft", str), ("score", int)],
    retries=0,
    fallback={"draft": "none", "score": 7},
)
def two_outputs(m: str): ...


@pure
def show_two(score: str, result: int) -> str:
    return f"{score!r}/{result!r}"


@workflow
def two_output_flow(m: str @ Two) -> str:
    with Two:
        score, result = two_outputs(m)
        text = show_two(score, result)
    return text @ Two


def test_a_fallback_lands_in_the_right_variables(tmp_path):
    """Outputs and destination variables are two namespaces, not one.

    Here the *second* declared output is called `score` and the *first*
    destination variable is also called `score`. Guessing which namespace a
    result arrived in put the wrong value in both.
    """

    from zippergen.sqlite_runner import run_sqlite

    def unusable(action, inputs):
        raise LLMInvalidResponseError("nothing usable")

    result = run_sqlite(
        two_output_flow,
        [Two],
        {"TwoOutputs": {"m": "x"}},
        store_path=str(tmp_path / "ns.sqlite"),
        llm_backend=unusable,
        timeout=20,
    )

    assert result == "'none'/7", (
        "the first variable takes the first declared output, in order"
    )


def test_a_declared_tuple_fallback_stays_a_tuple():
    """JSON has no tuples, so the declared type has to put them back."""

    from zippergen.llm_policy import decode_fallback

    @llm(system="s", user="{m}", parse="json", outputs=[("pair", tuple)], fallback=(1, 2))
    def pairs(m: str): ...

    decoded = decode_fallback(pairs)

    assert decoded == {"pair": (1, 2)}
    assert type(decoded["pair"]) is tuple, "a list would enter durable state untyped"


def test_a_tuple_fallback_preserves_nested_lists_and_tuples():
    """Only actual tuples are tuples; the codec never guesses from JSON arrays."""

    from zippergen.llm_policy import decode_fallback

    fallback = ([1, 2], (3, 4), {"items": [5, 6]})

    @llm(system="s", user="{m}", parse="json", outputs=[("value", tuple)], fallback=fallback)
    def structured(m: str): ...

    assert decode_fallback(structured) == {"value": fallback}


def test_the_fallback_is_checked_like_any_other_answer():
    """Both roads meet the same gate before anything is committed."""

    from zippergen.llm_policy import checked_llm_outputs

    action = _action(parse="json", outputs=[("pair", tuple)], retries=0, fallback=(1, 2))

    result = attempt_llm_action(
        action,
        _always(LLMInvalidResponseError("bad")),
        check=lambda values: checked_llm_outputs(action, values),
    )

    assert type(result["pair"]) is tuple


# ---------------------------------------------------------------------------
# Trace contract, cancellation classification, envelopes
# ---------------------------------------------------------------------------

def test_a_retry_event_matches_the_trace_contract():
    """Every event is keyed by "type"; console_trace reads it directly."""

    from zippergen.llm_policy import retry_reporter
    from zippergen.runtime import console_trace

    events: list[dict] = []
    retry_reporter(events.append, _action())("draft_reply: retrying in 2s")

    assert events[0]["type"] == "llm_retry", "not 'kind'"
    console_trace({**events[0], "lifeline": "Writer"})  # must not raise


def test_a_cancelled_wait_is_classified_as_cancellation():
    """The runners tell a stop from a fault by this phrase."""

    assert "Workflow cancelled" in str(RetryCancelled())


@pytest.mark.parametrize(
    "body",
    [{}, {"choices": []}, {"choices": [{"message": {}}]},
     {"choices": [{"message": {"content": 7}}]}, None],
    ids=["empty", "no-choices", "no-content", "non-string", "not-a-dict"],
)
def test_a_malformed_envelope_is_an_invalid_response(body):
    """Valid JSON can still be structurally unusable, and that is retryable."""

    from zippergen.backends import _chat_completion_text

    with pytest.raises(LLMInvalidResponseError):
        _chat_completion_text(body, provider="OpenAI")


def test_a_malformed_anthropic_envelope_is_an_invalid_response():
    from zippergen.backends import _anthropic_text

    with pytest.raises(LLMInvalidResponseError):
        _anthropic_text({"content": "not a list of blocks"})


@pytest.mark.parametrize(
    ("header", "expected"),
    [("120", 120.0), ("0", 0.0), ("soon", None)],
    ids=["seconds", "zero", "unparseable"],
)
def test_retry_after_is_read_in_both_standard_forms(header, expected):
    from zippergen.backends import _retry_after_seconds

    assert _retry_after_seconds({"Retry-After": header}) == expected


def test_retry_after_accepts_an_http_date():
    from datetime import datetime, timedelta, timezone
    from email.utils import format_datetime

    from zippergen.backends import _retry_after_seconds

    when = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=90))

    assert 80 <= (_retry_after_seconds({"Retry-After": when}) or 0) <= 95


def test_planner_generation_retries_a_transient_failure():
    """PlannerAction.max_retries counts bad candidates, not network blips."""

    from zippergen.planner import _planner_llm_call
    from zippergen.syntax import LLMAction

    action = LLMAction(
        name="_generate_spec",
        inputs=(),
        outputs=(("workflow_spec", str),),
        system_prompt="s",
        user_prompt="u",
        parse_format="text",
    )
    calls = {"n": 0}

    def backend(node, inputs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise LLMTransientError("connection refused")
        return {"workflow_spec": "generated"}

    assert _planner_llm_call(backend, action) == {"workflow_spec": "generated"}
    assert calls["n"] == 2, "a network blip must not cost a candidate"


def test_planner_generation_reports_retries_through_the_trace():
    """Planner's internal model calls obey the ordinary LLM trace contract."""

    from zippergen.planner import _planner_llm_call
    from zippergen.syntax import LLMAction

    action = LLMAction(
        name="_generate_spec",
        inputs=(),
        outputs=(("workflow_spec", str),),
        system_prompt="s",
        user_prompt="u",
        parse_format="text",
    )
    events: list[dict] = []
    calls = {"n": 0}

    def backend(node, inputs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise LLMTransientError("connection refused", retry_after=0)
        return {"workflow_spec": "generated"}

    assert _planner_llm_call(
        backend, action, trace=events.append
    ) == {"workflow_spec": "generated"}
    assert events == [{
        "type": "llm_retry",
        "action": "_generate_spec",
        "action_kind": "llm",
        "detail": (
            "_generate_spec: connection refused — retrying in 0s (3 left)."
        ),
    }]


def test_generated_fallback_types_follow_declared_output_order():
    """A set's hash order must never pair output names with the wrong types."""

    import ast

    from zippergen.planner import _planner_llm_definition_error

    base = ["draft", "score"]
    names = base if list(set(base)) != base else list(reversed(base))
    first, second = names
    source = f'''@llm(
    system="s", user="{{m}}", parse="json",
    outputs=[("{first}", str), ("{second}", int)],
    fallback={{"{first}": "none", "{second}": 7}},
)
def assess(m: str): ...
'''

    assert _planner_llm_definition_error(ast.parse(source).body[0]) is None


def test_planner_generation_can_be_cancelled_mid_wait():
    """A long Retry-After during @planner must end when the run is stopped."""

    from zippergen.planner import _planner_llm_call
    from zippergen.syntax import LLMAction

    action = LLMAction(
        name="_generate_spec",
        inputs=(),
        outputs=(("workflow_spec", str),),
        system_prompt="s",
        user_prompt="u",
        parse_format="text",
    )
    stop = threading.Event()

    def backend(node, inputs):
        stop.set()
        raise LLMTransientError("rate limited", retry_after=600.0)

    started = time.monotonic()
    with pytest.raises(RetryCancelled):
        _planner_llm_call(backend, action, stop=stop)

    assert time.monotonic() - started < 5.0, "the wait was not interrupted"


def test_a_cancelled_retry_is_a_workflow_cancellation():
    """The runners classify by type now, so the type has to be right."""

    from zippergen.errors import WorkflowCancelled

    assert issubclass(RetryCancelled, WorkflowCancelled)
