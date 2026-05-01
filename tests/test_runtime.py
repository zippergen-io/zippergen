"""Tests for Layer 5: runtime executor.

All tests use pure actions or a deterministic mock backend — no real LLM calls.
Conditions in @workflow bodies are boolean expressions, not PureAction calls.
"""

import pytest

from zippergen.syntax import Lifeline, Var
from zippergen.actions import pure
from zippergen.builder import workflow
from zippergen.runtime import run, mock_llm


# ---------------------------------------------------------------------------
# Lifelines
# ---------------------------------------------------------------------------

User    = Lifeline("User")
Compute = Lifeline("Compute")
Owner   = Lifeline("Owner")
Worker  = Lifeline("Worker")


# ---------------------------------------------------------------------------
# Shared pure actions
# ---------------------------------------------------------------------------

@pure
def increment(v: int) -> int:
    return v + 1


@pure
def double(v: int) -> int:
    return v * 2


@pure
def add(a: int, b: int) -> int:
    return a + b


@pure
def decrement(v: int) -> int:
    return v - 1


# ---------------------------------------------------------------------------
# Simple message passing
# ---------------------------------------------------------------------------

@workflow
def pass_int(n: int @ User) -> int:
    User(n) >> Compute(n)
    return n @ Compute


def test_message_passing():
    assert pass_int(n=42) == 42


def test_literal_receive_binding_mismatch_raises():
    from zippergen.syntax import LitExpr, MsgStmt, Workflow

    Sender = Lifeline("LiteralSender")
    Receiver = Lifeline("LiteralReceiver")
    body = MsgStmt(
        Sender,
        (LitExpr("actual", str),),
        Receiver,
        (LitExpr("expected", str),),
    )
    wf = Workflow(
        name="literal_mismatch",
        inputs=(),
        output_type=object,
        vars=(),
        body=body,
        ns={},
    )

    with pytest.raises(RuntimeError, match="does not match literal binding"):
        run(wf, [Sender, Receiver], {}, timeout=1.0)


@workflow
def pass_and_transform(n: int @ User) -> int:
    User(n) >> Compute(n)
    with Compute:
        n = increment(n)
    return n @ Compute


def test_pass_and_transform():
    assert pass_and_transform(n=10) == 11


# ---------------------------------------------------------------------------
# Pure actions — correct computation
# ---------------------------------------------------------------------------

@workflow
def double_then_increment(n: int @ Compute) -> int:
    with Compute:
        n = double(n)
        n = increment(n)
    return n @ Compute


def test_pure_action_sequence():
    assert double_then_increment(n=3) == 7  # 3*2 + 1


# ---------------------------------------------------------------------------
# Increment example from README (end-to-end)
# ---------------------------------------------------------------------------

@workflow
def increment_workflow(number: int @ User) -> int:
    User(number) >> Compute(number)
    with Compute:
        number = increment(number)
        number = double(number)
    Compute(number) >> User(number)
    return number @ User


def test_increment_example():
    assert increment_workflow(number=1) == 4   # (1+1)*2
    assert increment_workflow(number=0) == 2   # (0+1)*2
    assert increment_workflow(number=5) == 12  # (5+1)*2


# ---------------------------------------------------------------------------
# Conditional — uses boolean expression as condition
# ---------------------------------------------------------------------------

@workflow
def conditional_workflow(n: int @ Owner) -> int:
    if (n > 0) @ Owner:
        Owner(n) >> Worker(n)
        with Worker:
            n = double(n)
        Worker(n) >> Owner(n)
    else:
        pass
    return n @ Owner


def test_conditional_true_branch():
    assert conditional_workflow(n=3) == 6


def test_conditional_false_branch():
    assert conditional_workflow(n=-1) == -1


# ---------------------------------------------------------------------------
# While loop — runs until condition is false
# ---------------------------------------------------------------------------

@workflow
def countdown(n: int @ Owner) -> int:
    while (n > 0) @ Owner:
        with Owner:
            n = decrement(n)
    else:
        pass
    return n @ Owner


def test_while_loop_counts_down():
    assert countdown(n=3) == 0
    assert countdown(n=0) == 0


# ---------------------------------------------------------------------------
# While with message exchange
# ---------------------------------------------------------------------------

@workflow
def ping_pong(n: int @ Owner) -> int:
    while (n > 0) @ Owner:
        Owner(n) >> Worker(n)
        with Worker:
            n = decrement(n)
        Worker(n) >> Owner(n)
    else:
        pass
    return n @ Owner


def test_while_with_message_exchange():
    assert ping_pong(n=3) == 0


# ---------------------------------------------------------------------------
# Output var correctly returned
# ---------------------------------------------------------------------------

@workflow
def two_inputs(a: int @ User, b: int @ User) -> int:
    User(a) >> Compute(a)
    User(b) >> Compute(b)
    with Compute:
        a = add(a, b)
    return a @ Compute


def test_output_var_returned():
    assert two_inputs(a=3, b=4) == 7


# ---------------------------------------------------------------------------
# Timeout raises TimeoutError
# ---------------------------------------------------------------------------

def test_timeout_raises():
    """A backend that never returns causes TimeoutError after the deadline."""
    import threading
    from zippergen.syntax import LLMAction, ActStmt, VarExpr, Var, Workflow

    Slow = Lifeline("Slow")
    n = Var("n", int)
    action = LLMAction(
        name="slow",
        inputs=(("n", int),),
        outputs=(("n", int),),
        system_prompt="",
        user_prompt="{n}",
        parse_format="json",
    )
    body = ActStmt(Slow, action, (VarExpr(n),), (n,))
    wf = Workflow(name="slow_wf", inputs=(), output_type=int, vars=(), body=body, ns={})

    block = threading.Event()  # never set — backend blocks forever

    def blocking_backend(act, inputs):
        block.wait()
        return {}

    with pytest.raises(TimeoutError):
        run(wf, [Slow], {"Slow": {"n": 0}}, llm_backend=blocking_backend, timeout=0.2)


def test_failed_thread_unblocks_waiters():
    """When one lifeline's backend raises, waiting lifelines cancel quickly."""
    import time
    from zippergen.syntax import LLMAction, ActStmt, MsgStmt, VarExpr, Var, Workflow, SeqStmt

    Sender = Lifeline("Sender")
    Receiver = Lifeline("Receiver")
    n = Var("n", int)
    action = LLMAction(
        name="fail",
        inputs=(("n", int),),
        outputs=(("n", int),),
        system_prompt="",
        user_prompt="{n}",
        parse_format="json",
    )
    # Sender does an LLM action (will fail), then sends to Receiver.
    # Receiver waits for a message from Sender that will never arrive.
    body = SeqStmt(
        ActStmt(Sender, action, (VarExpr(n),), (n,)),
        MsgStmt(Sender, (VarExpr(n),), Receiver, (VarExpr(n),)),
    )
    wf = Workflow(name="fail_wf", inputs=(), output_type=int, vars=(), body=body, ns={})

    def failing_backend(act, inputs):
        raise RuntimeError("deliberate failure")

    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="deliberate failure"):
        run(wf, [Sender, Receiver], {"Sender": {"n": 0}, "Receiver": {"n": 0}},
            llm_backend=failing_backend, timeout=10.0)
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0, f"Cancellation took {elapsed:.1f}s — expected < 2s"
