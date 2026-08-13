"""Durable-state semantics, tested at the crash boundaries themselves.

Every test here drives roles through ``RoleRunner`` directly and then throws the
runner away, which is exactly what a crash looks like: the process stops between
committed steps and a new one starts against the same store. Nothing is mocked
about the persistence.

The rule under test throughout:

    The committed role state describes what is known to have completed. The
    step the control state points at may run again after a crash.
"""

import json
import sqlite3
import threading

import pytest

from zippergen import Lifeline, Var, branch, parallel, workflow
from zippergen.actions import effect, human, llm, pure
from zippergen.formula import atom
from zippergen.control import (
    ControlError,
    decode_control,
    encode_control,
    program_fingerprint,
)
from zippergen.projection import project
from zippergen.role_runner import RoleRunner
from zippergen.runtime import _build_formula_monitors
from zippergen.sqlite_runner import run_sqlite
from zippergen.store import (
    RoleStateConflict,
    WorkflowIdentityError,
    claim_workflow_identity,
    list_outstanding_messages,
    list_role_states,
    load_role_state,
    open_store,
)
from zippergen.syntax import EmptyStmt, _ordered_workflow_lifelines


# ---------------------------------------------------------------------------
# Harness: run a role for a bounded number of committed steps, then abandon it
# ---------------------------------------------------------------------------


class Crash(Exception):
    """Raised to stop a role at a chosen point, standing in for a crash."""


class _BoundedRunner(RoleRunner):
    """A RoleRunner that dies after ``budget`` committed steps."""

    budget = 10**9

    def _commit_state(self, residual, status, detail=None):
        if self.budget <= 0:
            raise Crash("crashed before committing this step")
        super()._commit_state(residual, status, detail)
        self.budget -= 1


def _runner(store, wf, lifeline, env, *, budget=10**9, cls=_BoundedRunner, **kwargs):
    conn = open_store(store)
    runner = cls(
        conn,
        lifeline.name,
        project(wf, lifeline),
        env,
        wf.ns,
        **kwargs,
    )
    runner.budget = budget
    return conn, runner


def _drive(store, wf, lifeline, env, *, budget=10**9, cls=_BoundedRunner, **kwargs):
    """Run one role until it finishes, blocks with nothing to do, or crashes."""

    conn, runner = _runner(store, wf, lifeline, env, budget=budget, cls=cls, **kwargs)
    try:
        try:
            runner.run()
        except Crash:
            return "crashed"
        except RuntimeError as exc:
            if "cancelled" in str(exc):
                return "cancelled"
            raise
        return "done"
    finally:
        conn.close()


def _drive_until_blocked(store, wf, lifeline, env, *, budget=10**9, **kwargs):
    """Run a role but give up once it has nothing to do, instead of spinning."""

    stop = threading.Event()

    class _StopWhenIdle(_BoundedRunner):
        def _sleep_after_idle_step(self):
            stop.set()

    return _drive(
        store, wf, lifeline, env, budget=budget, cls=_StopWhenIdle, stop=stop, **kwargs
    )


def _state(store, role):
    conn = open_store(store)
    try:
        return load_role_state(conn, role)
    finally:
        conn.close()


def _messages(store):
    conn = open_store(store)
    try:
        return list_outstanding_messages(conn)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Workflows under test
# ---------------------------------------------------------------------------

A = Lifeline("A")
B = Lifeline("B")

calls: dict[str, int] = {}


def _bump(name: str) -> int:
    calls[name] = calls.get(name, 0) + 1
    return calls[name]


@pure
def double(n: int) -> int:
    return n * 2


@pure
def add_one(n: int) -> int:
    return n + 1


@effect
def risky_effect(n: int) -> int:
    _bump("risky_effect")
    return n * 10


@llm(system="s", user="{n}", parse="text", outputs=(("answer", str),))
def ask_model(n: int) -> str: ...


@workflow
def sequential() -> int:
    A: a = double(2)
    A: b = add_one(a)
    A: c = add_one(b)
    return c @ A


@workflow
def two_role(n: int @ A) -> int:
    A: sent = double(n)
    A(sent) >> B(sent)
    B: got = add_one(sent)
    return got @ B


@pure
def below(total: int, limit: int) -> bool:
    return total < limit


@workflow
def loop_workflow(limit: int @ A) -> int:
    total = Var("total", int, default=0)
    A: keep = below(total, limit)
    while keep @ A:
        A: total = add_one(total)
        A: keep = below(total, limit)
    return total @ A


Worker = Lifeline("Worker")
Sink = Lifeline("Sink")


@workflow
def par_workflow(n: int @ Worker) -> int:
    with parallel:
        with branch:
            Worker: left = double(n)
            Worker(left) >> Sink(left)
        with branch:
            Worker: right = add_one(n)
            Worker(right) >> Sink(right)
    Sink: total = add_one(n)
    return total @ Sink


@workflow
def effect_workflow(n: int @ A) -> int:
    A: out = risky_effect(n)
    return out @ A


@workflow
def model_workflow(n: int @ A) -> str:
    A: answer = ask_model(n)
    return answer @ A


# A CPL-monitored loop: the guard is a formula, so the role carries monitor and
# vector-clock state that must survive a restart.
monitored_guard = atom(
    lambda env: env.get("total", 0) < env.get("limit", 0),
    src="total < limit",
)


@workflow
def monitored_loop(limit: int @ A) -> int:
    total = Var("total", int, default=0)
    while monitored_guard @ A:
        A: total = add_one(total)
    return total @ A


# A monitored two-role workflow, so sends carry a causal stamp.
sent_guard = atom(lambda env: env.get("sent", 0) > 0, src="sent > 0")


@workflow
def stamped_two_role(n: int @ A) -> int:
    A: sent = double(n)
    if sent_guard @ A:
        A(sent) >> B(sent)
    else:
        A(sent) >> B(sent)
    B: got = add_one(sent)
    return got @ B


# ---------------------------------------------------------------------------
# 1-2. Deterministic sequential execution, and crashing across a step
# ---------------------------------------------------------------------------


def test_sequential_execution_completes_and_records_its_state(tmp_path):
    store = str(tmp_path / "run.sqlite")

    assert _drive(store, sequential, A, {}) == "done"

    state = _state(store, "A")
    assert state["control"] == {"k": "done"}
    assert state["env"]["c"] == 6
    assert state["status"] == "done"


def test_a_crash_resumes_from_the_last_committed_step(tmp_path):
    store = str(tmp_path / "run.sqlite")

    assert _drive(store, sequential, A, {}, budget=1) == "crashed"
    partial = _state(store, "A")
    assert partial["control"] != {"k": "done"}
    assert partial["env"].get("c") is None

    # A brand-new runner, as a restarted process would build.
    assert _drive(store, sequential, A, {}) == "done"
    assert _state(store, "A")["env"]["c"] == 6


def test_repeated_restarts_at_every_step_still_reach_the_same_answer(tmp_path):
    """Restart after each committed step, over and over, from step 0 upward."""

    store = str(tmp_path / "run.sqlite")
    for budget in range(0, 6):
        outcome = _drive(store, sequential, A, {}, budget=budget)
        if outcome == "done":
            break
    assert _state(store, "A")["env"]["c"] == 6
    assert _state(store, "A")["control"] == {"k": "done"}


def test_the_starting_environment_is_ignored_once_state_exists(tmp_path):
    """A resumed role uses its committed variables, not the caller's arguments."""

    store = str(tmp_path / "run.sqlite")
    assert _drive(store, two_role, A, {"n": 5}, budget=1) == "crashed"

    # Restart with a different argument. The committed state must win.
    _drive_until_blocked(store, two_role, A, {"n": 999})
    outstanding = _messages(store)
    assert [message["payload"] for message in outstanding] == [[10]]


# ---------------------------------------------------------------------------
# 3-5. Outstanding messages
# ---------------------------------------------------------------------------


def test_a_sent_message_outlives_the_sender(tmp_path):
    store = str(tmp_path / "run.sqlite")

    _drive_until_blocked(store, two_role, A, {"n": 4})

    outstanding = _messages(store)
    assert len(outstanding) == 1
    assert outstanding[0]["sender"] == "A"
    assert outstanding[0]["receiver"] == "B"
    assert outstanding[0]["payload"] == [8]


def test_a_stale_second_runner_cannot_duplicate_a_committed_send(tmp_path):
    """The step counter is a compare-and-swap version, not just a metric."""

    store = str(tmp_path / "run.sqlite")
    assert _drive(store, two_role, A, {"n": 4}, budget=1) == "crashed"

    first_conn, first = _runner(store, two_role, A, {})
    second_conn, second = _runner(store, two_role, A, {})
    try:
        first.run()
        with pytest.raises(RoleStateConflict, match="another runner"):
            second.run()
    finally:
        first_conn.close()
        second_conn.close()

    outstanding = _messages(store)
    assert len(outstanding) == 1
    assert outstanding[0]["payload"] == [8]
    assert _state(store, "A")["status"] == "done"


def test_consuming_a_message_and_advancing_the_receiver_are_one_transaction(tmp_path):
    """If the receiver's step does not commit, the message is still outstanding."""

    store = str(tmp_path / "run.sqlite")
    _drive_until_blocked(store, two_role, A, {"n": 4})
    assert len(_messages(store)) == 1

    # Budget 0: the receive executes but its commit never happens.
    assert _drive(store, two_role, B, {}, budget=0) == "crashed"

    assert len(_messages(store)) == 1, "a rolled-back receive must not eat the message"
    assert _state(store, "B") is None or _state(store, "B")["env"].get("got") is None

    assert _drive(store, two_role, B, {}) == "done"
    assert _messages(store) == []
    assert _state(store, "B")["env"]["got"] == 9


def test_messages_on_one_route_stay_in_order(tmp_path):
    store = str(tmp_path / "run.sqlite")
    conn = open_store(store)
    try:
        conn.execute("BEGIN IMMEDIATE")
        for value in (1, 2, 3):
            conn.execute(
                "INSERT INTO outstanding_messages(sender,receiver,channel,payload) "
                "VALUES('A','B','main',?)",
                (json.dumps([value]),),
            )
        conn.execute("COMMIT")
        from zippergen.store import DurableChannel

        channel = DurableChannel(conn, "B")
        taken = [
            channel.try_get("A", "B", "main")[1][0] for _ in range(3)
        ]
        assert taken == [1, 2, 3]
    finally:
        conn.close()


def test_a_coregion_receive_takes_the_earliest_send_across_routes(tmp_path):
    store = str(tmp_path / "run.sqlite")
    conn = open_store(store)
    try:
        conn.execute("BEGIN IMMEDIATE")
        for sender, value in (("C", 30), ("A", 10), ("B", 20)):
            conn.execute(
                "INSERT INTO outstanding_messages(sender,receiver,channel,payload) "
                "VALUES(?,'R','main',?)",
                (sender, json.dumps([value])),
            )
        conn.execute("COMMIT")
        from zippergen.store import DurableChannel

        channel = DurableChannel(conn, "R")
        order = [
            channel.try_get_any("R", {"A", "B", "C"}, "main")[0] for _ in range(3)
        ]
        assert order == ["C", "A", "B"], "send order decides, not sender name"
    finally:
        conn.close()


def test_a_send_and_the_senders_advance_commit_together(tmp_path):
    """Either both or neither. This is what makes 'never duplicated' true.

        not committed:  no message,  old sender position
        committed:      message,     new sender position
    """

    store = str(tmp_path / "run.sqlite")

    # Crash on the commit that would both insert the message and advance A.
    # `double` commits first, so budget=1 lands exactly on the send.
    assert _drive(store, two_role, A, {"n": 4}, budget=1) == "crashed"

    before_control = _state(store, "A")["control"]
    assert _messages(store) == [], "an uncommitted send left no message"

    _drive_until_blocked(store, two_role, A, {"n": 4})

    after = _state(store, "A")
    assert len(_messages(store)) == 1, "the committed send left exactly one"
    assert after["control"] != before_control, "and the sender moved past it"


def test_the_causal_stamp_is_written_with_the_senders_own_state(tmp_path):
    """CPL's send side, not just its receive side.

    The outgoing message carries the stamp, and the sender's monitor state
    advances, in the same commit. A crash cannot leave a stamped message whose
    sender never recorded the event, or the reverse.
    """

    store = str(tmp_path / "run.sqlite")
    monitors, conditions = _build_formula_monitors(
        stamped_two_role, _ordered_workflow_lifelines(stamped_two_role)
    )
    _drive_until_blocked(
        store,
        stamped_two_role,
        A,
        {"n": 4},
        monitor=monitors["A"],
        formula_conditions=conditions,
    )

    conn = open_store(store)
    try:
        stamp = conn.execute(
            "SELECT causal_stamp FROM outstanding_messages ORDER BY id LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    assert stamp is not None and stamp[0] is not None, "the send carried a stamp"

    sender_state = _state(store, "A")
    assert sender_state["monitor"] is not None
    sent_clock = json.loads(stamp[0])["vc"]
    assert sender_state["monitor"]["vc"]["A"] >= sent_clock["A"], (
        "the sender's own clock is at least as far along as what it stamped"
    )


def test_a_crashed_sender_does_not_send_its_message_twice(tmp_path):
    """The classic duplicate-send hazard, checked directly.

    A crashes after committing its send. On restart its control state is past
    that send, so it does not repeat it, and the receiver sees exactly one.
    """

    store = str(tmp_path / "run.sqlite")

    assert _drive(store, two_role, A, {"n": 4}) == "done"
    assert len(_messages(store)) == 1

    # Restart the sender against the same store, as a supervisor would.
    assert _drive(store, two_role, A, {"n": 4}) == "done"
    assert len(_messages(store)) == 1, "a resumed sender must not send again"

    assert _drive(store, two_role, B, {}) == "done"
    assert _state(store, "B")["env"]["got"] == 9
    assert _messages(store) == []


# ---------------------------------------------------------------------------
# 6-8. Control constructs across restarts
# ---------------------------------------------------------------------------


def test_a_loop_resumes_mid_iteration_without_any_loop_snapshot(tmp_path):
    store = str(tmp_path / "run.sqlite")

    assert _drive(store, loop_workflow, A, {"limit": 3}, budget=4) == "crashed"
    midway = _state(store, "A")
    assert midway["control"] != {"k": "done"}

    assert _drive(store, loop_workflow, A, {"limit": 3}) == "done"
    assert _state(store, "A")["env"]["total"] == 3


def test_a_branch_resumes_after_the_decision(tmp_path):
    store = str(tmp_path / "run.sqlite")
    result = run_sqlite(
        two_role, None, {"A": {"n": 7}}, store_path=store, timeout=30
    )
    assert result == 15


def test_parallel_branches_each_keep_their_own_position(tmp_path):
    """A role inside `parallel` has several positions, and all of them persist."""

    store = str(tmp_path / "run.sqlite")
    assert _drive(store, par_workflow, Worker, {"n": 4}, budget=1) == "crashed"

    control = _state(store, "Worker")["control"]
    assert control["k"] in {"par", "seq"}, control
    encoded = json.dumps(control)
    assert "par" in encoded, "an in-flight parallel region must be representable"

    _drive_until_blocked(store, par_workflow, Worker, {"n": 4})
    payloads = sorted(message["payload"] for message in _messages(store))
    assert payloads == [[5], [8]]


# ---------------------------------------------------------------------------
# 9-12. External actions: the honest, weaker guarantee
# ---------------------------------------------------------------------------


def test_a_committed_effect_is_not_repeated_on_restart(tmp_path):
    store = str(tmp_path / "run.sqlite")
    calls.clear()

    assert _drive(store, effect_workflow, A, {"n": 3}) == "done"
    assert calls["risky_effect"] == 1
    assert _state(store, "A")["env"]["out"] == 30

    # Restarting a finished role must not run the effect again.
    assert _drive(store, effect_workflow, A, {"n": 3}) == "done"
    assert calls["risky_effect"] == 1


def test_an_effect_whose_result_never_committed_runs_again(tmp_path):
    """The documented weaker guarantee, pinned as a test rather than a claim.

    The side effect really happened, but nothing durable recorded it, so the
    control state still points at the effect and it runs a second time.
    """

    store = str(tmp_path / "run.sqlite")
    calls.clear()

    assert _drive(store, effect_workflow, A, {"n": 3}, budget=0) == "crashed"
    assert calls["risky_effect"] == 1, "the effect did happen"
    assert _state(store, "A")["env"].get("out") is None, "but nothing recorded it"

    assert _drive(store, effect_workflow, A, {"n": 3}) == "done"
    assert calls["risky_effect"] == 2, "so it runs again; this is by design"
    assert _state(store, "A")["env"]["out"] == 30


def test_a_model_call_whose_result_committed_is_not_repeated(tmp_path):
    store = str(tmp_path / "run.sqlite")
    seen: list[int] = []

    def backend(action, inputs):
        seen.append(1)
        return {"answer": "hello"}

    assert _drive(store, model_workflow, A, {"n": 1}, llm_backend=backend) == "done"
    assert len(seen) == 1
    assert _drive(store, model_workflow, A, {"n": 1}, llm_backend=backend) == "done"
    assert len(seen) == 1
    assert _state(store, "A")["env"]["answer"] == "hello"


def test_a_model_call_is_repeated_when_its_result_did_not_commit(tmp_path):
    store = str(tmp_path / "run.sqlite")
    seen: list[int] = []

    def backend(action, inputs):
        seen.append(1)
        return {"answer": "hello"}

    assert (
        _drive(store, model_workflow, A, {"n": 1}, budget=0, llm_backend=backend)
        == "crashed"
    )
    assert len(seen) == 1
    assert _drive(store, model_workflow, A, {"n": 1}, llm_backend=backend) == "done"
    assert len(seen) == 2, "an uncommitted model call is made again"


# ---------------------------------------------------------------------------
# 13-14. Human tasks
# ---------------------------------------------------------------------------


@human(kind="confirm", instruction="Approve {n}?", outputs=["ok: bool"])
def approve(n: int) -> bool: ...


@workflow
def human_workflow(n: int @ A) -> bool:
    A: ok = approve(n)
    return ok @ A


def test_a_pending_human_task_survives_a_restart(tmp_path):
    """The question is durable even though nothing is running to hold it."""

    store = str(tmp_path / "run.sqlite")

    class _Sqlite:
        uses_sqlite_human_tasks = True

        def __call__(self, action, inputs):  # pragma: no cover - never called
            raise AssertionError("the sqlite backend must not prompt inline")

    stop = threading.Event()

    class _GiveUpWaiting(_BoundedRunner):
        def _wait_for_human_task(self, task_id):
            stop.set()
            raise RuntimeError("Workflow cancelled")

    _drive(
        store,
        human_workflow,
        A,
        {"n": 2},
        cls=_GiveUpWaiting,
        human_backend=_Sqlite(),
        stop=stop,
    )

    conn = open_store(store)
    try:
        tasks = conn.execute(
            "SELECT task_id, status, action FROM human_tasks"
        ).fetchall()
    finally:
        conn.close()
    assert len(tasks) == 1
    assert tasks[0][1] == "pending"
    assert tasks[0][2] == "approve"


def test_an_answered_human_task_is_picked_up_after_a_restart(tmp_path):
    store = str(tmp_path / "run.sqlite")

    class _Sqlite:
        uses_sqlite_human_tasks = True

        def __call__(self, action, inputs):  # pragma: no cover
            raise AssertionError("the sqlite backend must not prompt inline")

    stop = threading.Event()

    class _GiveUpWaiting(_BoundedRunner):
        def _wait_for_human_task(self, task_id):
            stop.set()
            raise RuntimeError("Workflow cancelled")

    _drive(
        store,
        human_workflow,
        A,
        {"n": 2},
        cls=_GiveUpWaiting,
        human_backend=_Sqlite(),
        stop=stop,
    )

    from zippergen.store import complete_human_task

    conn = open_store(store)
    try:
        task_id = conn.execute("SELECT task_id FROM human_tasks").fetchone()[0]
        conn.execute("BEGIN IMMEDIATE")
        complete_human_task(conn, task_id, {"ok": True})
        conn.execute("COMMIT")
    finally:
        conn.close()

    assert _drive(store, human_workflow, A, {"n": 2}, human_backend=_Sqlite()) == "done"
    assert _state(store, "A")["env"]["ok"] is True


# ---------------------------------------------------------------------------
# 15. CPL monitor and vector clock
# ---------------------------------------------------------------------------


def test_monitor_and_vector_clock_state_survive_a_restart(tmp_path):
    store = str(tmp_path / "run.sqlite")
    monitors, conditions = _build_formula_monitors(
        monitored_loop, _ordered_workflow_lifelines(monitored_loop)
    )
    monitor = monitors["A"]

    assert _drive(
        store,
        monitored_loop,
        A,
        {"limit": 3},
        budget=3,
        monitor=monitor,
        formula_conditions=conditions,
    ) == "crashed"
    stored = _state(store, "A")["monitor"]
    assert stored is not None and "vc" in stored

    # A fresh monitor, as a restarted process would build, must be restored
    # from the store rather than starting from zero.
    fresh_monitors, fresh_conditions = _build_formula_monitors(
        monitored_loop, _ordered_workflow_lifelines(monitored_loop)
    )
    assert _drive(
        store,
        monitored_loop,
        A,
        {"limit": 3},
        monitor=fresh_monitors["A"],
        formula_conditions=fresh_conditions,
    ) == "done"
    assert _state(store, "A")["env"]["total"] == 3
    assert fresh_monitors["A"].snapshot_state()["vc"]["A"] > 0


def test_monitor_state_round_trips_through_the_store(tmp_path):
    """Whatever the monitor needs is persisted verbatim, vector clock included."""

    store = str(tmp_path / "run.sqlite")
    monitors, _conditions = _build_formula_monitors(
        monitored_loop, _ordered_workflow_lifelines(monitored_loop)
    )
    monitor = monitors["A"]
    before = monitor.snapshot_state()

    conn = open_store(store)
    try:
        from zippergen.store import write_role_state

        conn.execute("BEGIN IMMEDIATE")
        write_role_state(
            conn,
            "A",
            env={},
            control={"k": "done"},
            monitor=before,
            steps=0,
            status="running",
        )
        conn.execute("COMMIT")
        after = load_role_state(conn, "A")["monitor"]
    finally:
        conn.close()
    assert after["vc"] == before["vc"]


# ---------------------------------------------------------------------------
# 16-17. Workflow identity and results
# ---------------------------------------------------------------------------


def test_a_finished_workflow_keeps_its_result(tmp_path):
    store = str(tmp_path / "run.sqlite")
    first = run_sqlite(two_role, None, {"A": {"n": 6}}, store_path=store, timeout=30)
    second = run_sqlite(two_role, None, {"A": {"n": 6}}, store_path=store, timeout=30)
    assert first == second == 13


def test_an_incompatible_workflow_is_refused(tmp_path):
    store = str(tmp_path / "run.sqlite")
    conn = open_store(store)
    try:
        claim_workflow_identity(conn, "demo", "fingerprint-one")
        claim_workflow_identity(conn, "demo", "fingerprint-one")  # same code is fine
        with pytest.raises(WorkflowIdentityError, match="workflow changed"):
            claim_workflow_identity(conn, "demo", "fingerprint-two")
        with pytest.raises(WorkflowIdentityError, match="not 'other'"):
            claim_workflow_identity(conn, "other", "fingerprint-one")
    finally:
        conn.close()


def test_the_fingerprint_is_the_same_in_a_different_process():
    """Otherwise every restart would look like an incompatible edit.

    An earlier version hashed repr(program), which embeds the memory address of
    each guard closure. That was stable within one process and different in the
    next one, which is exactly the wrong way round.
    """

    import subprocess
    import sys

    code = (
        "import sys; sys.path.insert(0, 'tests')\n"
        "from zippergen.projection import project\n"
        "from zippergen.control import program_fingerprint\n"
        "from test_durable_state import loop_workflow, A\n"
        "print(program_fingerprint({'A': project(loop_workflow, A)}))\n"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        ).stdout.strip()
        for _ in range(2)
    }
    assert len(runs) == 1 and "" not in runs


def test_adding_a_statement_changes_the_fingerprint():
    """A structural edit moves the paths, so the stored control state is stale."""

    @workflow
    def before_edit(n: int @ A) -> int:
        A: a = double(n)
        return a @ A

    @workflow
    def after_edit(n: int @ A) -> int:
        A: a = double(n)
        A: a = add_one(a)
        return a @ A

    assert program_fingerprint(
        {"A": project(before_edit, A)}
    ) != program_fingerprint({"A": project(after_edit, A)})


def test_exactly_what_the_fingerprint_covers():
    """Pin the guarantee, both halves, so nobody has to guess it.

    The fingerprint answers one question: are the stored control positions still
    meaningful? Positions are paths, so it covers the shape of the program and
    which statement sits at each position, plus the declared output types,
    because committed variables are stored under those names. It does not cover
    an action's implementation or an LLM's prompt, which cannot move a path.
    """

    from zippergen.syntax import (
        ActStmt,
        HumanAction,
        LLMAction,
        PureAction,
        RecvStmt,
        SelfAssignStmt,
        SendStmt,
        VarExpr,
    )

    x, y, z = Var("x", int), Var("y", int), Var("z", int)

    def fingerprint(action):
        return program_fingerprint({"A": ActStmt(A, action, (VarExpr(x),), (y,))})

    base = PureAction("calc", (("x", int),), (("y", int),), lambda v: v * 2)

    # Not covered: the same action computing something else.
    other_body = PureAction("calc", (("x", int),), (("y", int),), lambda v: v * 999)
    assert fingerprint(base) == fingerprint(other_body)

    # Not covered: a rewritten prompt.
    prompt_one = LLMAction("ask", (("x", int),), (("y", str),), "one", "{x}", "text")
    prompt_two = LLMAction("ask", (("x", int),), (("y", str),), "TWO", "OTHER {x}", "text")
    assert fingerprint(prompt_one) == fingerprint(prompt_two)

    # Covered: a different action in that position.
    renamed = PureAction("calc_v2", (("x", int),), (("y", int),), lambda v: v * 2)
    assert fingerprint(base) != fingerprint(renamed)

    # Covered: a changed output type, because the committed value has that type.
    retyped = PureAction("calc", (("x", int),), (("y", str),), lambda v: str(v))
    assert fingerprint(base) != fingerprint(retyped)

    # Covered: every expression that reads or writes the durable environment.
    assert program_fingerprint(
        {"A": SendStmt(A, (VarExpr(x),), B)}
    ) != program_fingerprint({"A": SendStmt(A, (VarExpr(z),), B)})
    assert program_fingerprint(
        {"A": ActStmt(A, base, (VarExpr(x),), (y,))}
    ) != program_fingerprint({"A": ActStmt(A, base, (VarExpr(z),), (y,))})
    assert program_fingerprint(
        {"A": SelfAssignStmt(A, (VarExpr(x),), (VarExpr(y),))}
    ) != program_fingerprint(
        {"A": SelfAssignStmt(A, (VarExpr(z),), (VarExpr(y),))}
    )

    # Binding types and HumanAction's singular output declaration also matter.
    text_y = Var("y", str)
    assert program_fingerprint(
        {"A": RecvStmt(A, (VarExpr(y),), B)}
    ) != program_fingerprint({"A": RecvStmt(A, (VarExpr(text_y),), B)})
    confirm = HumanAction("ask", (), "answer", bool, "confirm")
    edit = HumanAction("ask", (), "answer", str, "edit")
    assert program_fingerprint(
        {"A": ActStmt(A, confirm, (), (y,))}
    ) != program_fingerprint({"A": ActStmt(A, edit, (), (y,))})


def test_changing_a_workflow_changes_its_fingerprint():
    original = program_fingerprint(
        {name: project(two_role, lifeline)
         for name, lifeline in (("A", A), ("B", B))}
    )
    same = program_fingerprint(
        {name: project(two_role, lifeline)
         for name, lifeline in (("A", A), ("B", B))}
    )
    different = program_fingerprint(
        {name: project(sequential, lifeline) for name, lifeline in (("A", A),)}
    )
    assert original == same
    assert original != different


# ---------------------------------------------------------------------------
# 18. History is not part of recovery
# ---------------------------------------------------------------------------


def test_deleting_all_history_does_not_affect_recovery(tmp_path):
    """The invariant that keeps history a free choice rather than a commitment."""

    store = str(tmp_path / "run.sqlite")
    assert _drive(store, sequential, A, {}, budget=1) == "crashed"

    conn = open_store(store)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM history")
        conn.execute("COMMIT")
        assert conn.execute("SELECT COUNT(*) FROM history").fetchone()[0] == 0
    finally:
        conn.close()

    assert _drive(store, sequential, A, {}) == "done"
    assert _state(store, "A")["env"]["c"] == 6


def test_deleting_history_from_a_finished_run_keeps_the_result(tmp_path):
    store = str(tmp_path / "run.sqlite")
    run_sqlite(two_role, None, {"A": {"n": 6}}, store_path=store, timeout=30)

    from zippergen.storage_maintenance import prune_store_history

    prune_store_history(store, keep=0)
    assert run_sqlite(
        two_role, None, {"A": {"n": 6}}, store_path=store, timeout=30
    ) == 13


# ---------------------------------------------------------------------------
# 19. Control state encoding
# ---------------------------------------------------------------------------


def test_control_state_round_trips_for_every_shape(tmp_path):
    local = project(sequential, A)
    for control in (
        {"k": "done"},
        encode_control(local, local),
    ):
        residual = decode_control(local, control)
        assert encode_control(local, residual) == control


def test_control_state_from_another_program_is_rejected(tmp_path):
    local = project(sequential, A)
    with pytest.raises(ControlError):
        decode_control(local, {"k": "at", "p": [9, 9, 9]})
    with pytest.raises(ControlError):
        decode_control(local, {"k": "nonsense"})


def test_a_replay_era_store_is_refused_with_advice(tmp_path):
    store = tmp_path / "old.sqlite"
    conn = sqlite3.connect(store)
    conn.execute("CREATE TABLE events (rowid INTEGER PRIMARY KEY, kind TEXT)")
    conn.commit()
    conn.close()

    from zippergen.store import StoreSchemaError

    with pytest.raises(StoreSchemaError, match="deploy reset"):
        open_store(str(store))


# ---------------------------------------------------------------------------
# 20. Whole-workflow runs through the supervisor
# ---------------------------------------------------------------------------


def test_supervisor_runs_a_loop_workflow_to_completion(tmp_path):
    store = str(tmp_path / "run.sqlite")
    assert run_sqlite(
        loop_workflow, None, {"A": {"limit": 4}}, store_path=store, timeout=30
    ) == 4
    states = {row["role"]: row for row in list_role_states(open_store(store))}
    assert states["A"]["status"] == "done"
    assert states["A"]["control"] == {"k": "done"}
