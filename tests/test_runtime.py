"""Tests for Layer 5: runtime executor.

All tests use pure actions or a deterministic mock backend — no real LLM calls.
Conditions in @workflow bodies are boolean expressions, not PureAction calls.
"""

import pytest

from zippergen.syntax import Lifeline, Var
from zippergen.actions import pure
from zippergen.builder import workflow
from zippergen.runtime import run, mock_llm

# Module-level Var declarations for output variables used in CPL tests.
# Annotated-assignment DSL syntax (Lifeline: out = action()) requires output
# variable names to be Var objects in the module's global namespace.
verdict = Var("verdict", str)
flag    = Var("flag",    bool)
out     = Var("out",     str)
nh_out  = Var("nh_out",  str)


# ---------------------------------------------------------------------------
# Lifelines
# ---------------------------------------------------------------------------

User    = Lifeline("User")
Compute = Lifeline("Compute")
Owner   = Lifeline("Owner")
Worker  = Lifeline("Worker")
CoreA   = Lifeline("CoreA")
CoreB   = Lifeline("CoreB")
CoreR   = Lifeline("CoreR")


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
def slow_identity(v: int) -> int:
    import time
    time.sleep(0.05)
    return v


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
# Co-region receive-any
# ---------------------------------------------------------------------------

@workflow
def coregion_collect(a: int @ CoreA, b: int @ CoreB) -> tuple:
    with CoreA:
        a = slow_identity(a)
    with coregion:
        CoreA(a) >> CoreR(a)
        CoreB(b) >> CoreR(b)
    return (a @ CoreR, b @ CoreR)


def test_coregion_receiver_accepts_messages_in_arrival_order():
    events = []
    result = run(coregion_collect, [CoreA, CoreB, CoreR], {"CoreA": {"a": 1}, "CoreB": {"b": 2}}, trace=events.append)
    recv_events = [event for event in events if event["type"] == "recv" and event["to"] == "CoreR"]
    assert result == (1, 2)
    assert [event["from"] for event in recv_events] == ["CoreB", "CoreA"]


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


def test_trace_includes_pure_action_kind():
    events = []
    result = run(
        double_then_increment,
        [Compute],
        {"Compute": {"n": 3}},
        trace=events.append,
    )
    assert result == 7
    act_events = [event for event in events if event["type"] in {"act_start", "act"}]
    assert act_events
    assert {event["action_kind"] for event in act_events} == {"pure"}


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


# ---------------------------------------------------------------------------
# Causal Past Logic guard integration tests
# ---------------------------------------------------------------------------

from zippergen.formula import atom, Y, P, on


_CPLPlanner2  = Lifeline("CPLPlanner2")
_CPLExecutor2 = Lifeline("CPLExecutor2")

_approved_atom2 = atom(lambda env: env.get("approved", False))
_ya_guard2      = Y[_CPLPlanner2](_approved_atom2)


@pure
def _approved_str() -> str:
    return "approved"


@pure
def _rejected_str() -> str:
    return "rejected"


@workflow
def _ya_workflow2(approved: bool @ _CPLPlanner2) -> str:
    _CPLPlanner2(approved) >> _CPLExecutor2(approved)
    if _ya_guard2 @ _CPLExecutor2:
        _CPLExecutor2: verdict = _approved_str()
    else:
        _CPLExecutor2: verdict = _rejected_str()
    return verdict @ _CPLExecutor2


def test_ya_guard_routes_true():
    assert _ya_workflow2(approved=True) == "approved"


def test_ya_guard_routes_false():
    assert _ya_workflow2(approved=False) == "rejected"


# --- Y (previous local event) ---

_YOwner = Lifeline("YLocalOwner")
_flag_atom = atom(lambda env: env.get("flag", False))
_prev_flag = Y(_flag_atom)


@pure
def _set_flag_true() -> bool:
    return True


@pure
def _set_flag_false() -> bool:
    return False


@pure
def _yes_str(x: bool) -> str:
    return "yes"


@pure
def _no_str(x: bool) -> str:
    return "no"


@workflow
def _y_local_workflow(dummy: bool @ _YOwner) -> str:
    _YOwner: flag = _set_flag_true()
    if _prev_flag @ _YOwner:
        _YOwner: out = _yes_str(flag)
    else:
        _YOwner: out = _no_str(flag)
    return out @ _YOwner


def test_y_local_guard_true_after_prior_act():
    assert _y_local_workflow(dummy=True) == "yes"


# --- Y false when no prior event ---

_NoHistOwner = Lifeline("NoHistOwner")
_nh_guard = Y(atom(lambda env: True))


@workflow
def _no_history_workflow(x: bool @ _NoHistOwner) -> str:
    if _nh_guard @ _NoHistOwner:
        _NoHistOwner: nh_out = _yes_str(x)
    else:
        _NoHistOwner: nh_out = _no_str(x)
    return nh_out @ _NoHistOwner


def test_y_guard_false_with_no_prior_event():
    assert _no_history_workflow(x=True) == "no"


# --- Inline formula discovery ---

_InlinePlanner = Lifeline("InlinePlanner")
_InlineExecutor = Lifeline("InlineExecutor")


@workflow
def _inline_formula_workflow(approved: bool @ _InlinePlanner) -> str:
    _InlinePlanner(approved) >> _InlineExecutor(approved)
    if Y[_InlinePlanner](atom(lambda env: env.get("approved", False))) @ _InlineExecutor:
        _InlineExecutor: out = _yes_str(approved)
    else:
        _InlineExecutor: out = _no_str(approved)
    return out @ _InlineExecutor


def test_inline_formula_guard_is_discovered_before_execution():
    assert _inline_formula_workflow(approved=True) == "yes"


# --- P (strict causal past) ---

_PastOwner = Lifeline("PastOwner")
_past_guard = P(atom(lambda env: env.get("flag", False)))


@workflow
def _past_workflow(dummy: bool @ _PastOwner) -> str:
    _PastOwner: flag = _set_flag_true()
    if _past_guard @ _PastOwner:
        _PastOwner: out = _yes_str(flag)
    else:
        _PastOwner: out = _no_str(flag)
    return out @ _PastOwner


def test_past_guard_sees_prior_local_event():
    assert _past_workflow(dummy=True) == "yes"


# --- Cross-lifeline P ---

_RemotePastPlanner = Lifeline("RemotePastPlanner")
_RemotePastExecutor = Lifeline("RemotePastExecutor")
_remote_approved = on(_RemotePastPlanner) & atom(lambda env: env.get("approved", False))
_remote_past_guard = P(_remote_approved)


@workflow
def _remote_past_workflow(approved: bool @ _RemotePastPlanner) -> str:
    _RemotePastPlanner(approved) >> _RemotePastExecutor(approved)
    if _remote_past_guard @ _RemotePastExecutor:
        _RemotePastExecutor: out = _yes_str(approved)
    else:
        _RemotePastExecutor: out = _no_str(approved)
    return out @ _RemotePastExecutor


def test_past_guard_sees_remote_causal_past():
    assert _remote_past_workflow(approved=True) == "yes"
    assert _remote_past_workflow(approved=False) == "no"


# --- Stale relay status ---

_RelayDevice = Lifeline("RelayDevice")
_Relay1 = Lifeline("Relay1")
_Relay2 = Lifeline("Relay2")
_Indicator = Lifeline("Indicator")
_latest_device_on = Y[_RelayDevice](atom(lambda env: env.get("flag", False), src="flag"))


@workflow
def _stale_relay_workflow() -> tuple:
    _RelayDevice: flag = _set_flag_true()
    _RelayDevice(flag) >> _Relay1(flag)
    _Relay1(flag) >> _Indicator(flag)
    if _latest_device_on @ _Indicator:
        pass
    else:
        pass

    _RelayDevice: flag = _set_flag_false()
    _RelayDevice(flag) >> _Relay1(flag)
    _RelayDevice: flag = _set_flag_true()
    _RelayDevice(flag) >> _Relay2(flag)
    _Relay2(flag) >> _Indicator(flag)
    if _latest_device_on @ _Indicator:
        pass
    else:
        pass

    _Relay1(flag) >> _Indicator(flag)
    if _latest_device_on @ _Indicator:
        pass
    else:
        pass


def test_causal_guard_ignores_stale_relay_status():
    events = []
    final_envs = run(_stale_relay_workflow, [_RelayDevice, _Relay1, _Relay2, _Indicator], {}, trace=events.append)
    decisions = [e["value"] for e in events if e["type"] == "decision"]
    assert decisions == [True, True, True]
    assert final_envs["Indicator"]["flag"] is False


# --- Event metadata atoms ---

_MetaPlanner = Lifeline("MetaPlanner")
_MetaExecutor = Lifeline("MetaExecutor")
_fresh_receive = Y(atom(
    lambda env, event: (
        event.kind == "recv"
        and event.message_vc is not None
        and event.message_vc["MetaPlanner"] == event.vc["MetaPlanner"]
    ),
    src="fresh_receive",
))


@workflow
def _metadata_formula_workflow(approved: bool @ _MetaPlanner) -> str:
    _MetaPlanner(approved) >> _MetaExecutor(approved)
    if _fresh_receive @ _MetaExecutor:
        _MetaExecutor: out = _yes_str(approved)
    else:
        _MetaExecutor: out = _no_str(approved)
    return out @ _MetaExecutor


def test_atom_can_use_receive_metadata_in_guard_history():
    assert _metadata_formula_workflow(approved=True) == "yes"


def test_monitor_trace_includes_vector_clock_metadata():
    events = []
    result = run(
        _metadata_formula_workflow,
        [_MetaPlanner, _MetaExecutor],
        {"MetaPlanner": {"approved": True}},
        trace=events.append,
    )
    assert result == "yes"
    recv_event = next(e for e in events if e["type"] == "recv")
    decision_event = next(e for e in events if e["type"] == "decision")
    assert recv_event["vc"]["MetaPlanner"] == 1
    assert recv_event["message_vc"]["MetaPlanner"] == 1
    assert decision_event["vc"]["MetaExecutor"] == 2
