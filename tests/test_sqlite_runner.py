from zippergen.runtime import run
from zippergen.sqlite_runner import LocalSupervisor, run_sqlite
from zippergen.store import (
    complete_human_task,
    DurableChannel,
    ensure_human_task,
    human_task_id,
    list_trace_events,
    load_workflow_result,
    open_store,
)
from zippergen import Lifeline, Var, workflow, branch, parallel
from zippergen.actions import effect, human, llm, pure
from zippergen.formula import atom, At, Here, Y
from zippergen.locator import action_node_paths
from zippergen.projection import project
from zippergen.role_runner import RoleRunner, _begin_immediate
from zippergen.runtime import _input_hash
from zippergen.syntax import ReceiveAnyStmt, VarExpr
from zipperchat import WebTrace
from tests.loop_fixture import counter_loop, A as LoopA, B as LoopB
from tests.test_examples_regression import _two_role_branch_workflow, A, B
import json
import pytest
import sqlite3
import threading
import time

PUser = Lifeline("PUser")
POwner = Lifeline("POwner")
PCompute = Lifeline("PCompute")

PAsk = Lifeline("PAsk")
PAnswer = Lifeline("PAnswer")

PHuman = Lifeline("PHuman")

SQLiteCPLPlanner = Lifeline("SQLiteCPLPlanner")
SQLiteCPLExecutor = Lifeline("SQLiteCPLExecutor")
SQLiteFieldSource = Lifeline("SQLiteFieldSource")
SQLiteFieldGate = Lifeline("SQLiteFieldGate")
SQLiteFormulaLoopOwner = Lifeline("SQLiteFormulaLoopOwner")
SQLiteAnyA = Lifeline("SQLiteAnyA")
SQLiteAnyZ = Lifeline("SQLiteAnyZ")
SQLiteAnyR = Lifeline("SQLiteAnyR")

p_total = Var("p_total", int, default=0)
p_m = Var("p_m", int, default=0)
p_label = Var("p_label", int, default=0)
p_got = Var("p_got", int, default=0)
p_approved = Var("p_approved", bool, default=False)
p_effect_value = Var("p_effect_value", int, default=0)
sqlite_any_a = Var("sqlite_any_a", int, default=0)
sqlite_any_z = Var("sqlite_any_z", int, default=0)


@pure
def p_add(a: int, b: int) -> int:
    return a + b


@llm(system="s", user="{p_m}", parse="json", outputs=[("p_label", int)])
def p_classify(p_m: int) -> int: ...


_effect_calls = {"n": 0}


@effect
def p_external_counter(p_m: int) -> int:
    _effect_calls["n"] += 1
    return p_m + _effect_calls["n"]


@effect(visible=False)
def p_hidden_external_counter(p_m: int) -> int:
    return p_m + 1


@human(kind="confirm", instruction="Approve {prompt}?", outputs=["p_approved: bool"])
def p_review(prompt: str): pass


@pure
def p_yes() -> str:
    return "yes"


@pure
def p_no() -> str:
    return "no"


@pure
def p_inc(n: int) -> int:
    return n + 1


_sqlite_approved_atom = atom(lambda env: env.get("approved", False))
_sqlite_ya_guard = Y[SQLiteCPLPlanner](_sqlite_approved_atom)
_sqlite_field_match_guard = At[SQLiteFieldSource].src == Here.gate
_sqlite_formula_loop_guard = atom(
    lambda env: env.get("loop_n", 0) < env.get("loop_limit", 0),
    src="loop_n < loop_limit",
)


@workflow
def sqlite_parallel_sum(a: int @ PUser, b: int @ POwner):
    with parallel:
        with branch:
            PUser(a) >> PCompute(a)
        with branch:
            POwner(b) >> PCompute(b)
    PCompute: p_total = p_add(a, b)
    return p_total @ PCompute


@workflow
def sqlite_external_round(n: int @ PAsk):
    PAsk(n) >> PAnswer(p_m)
    PAnswer: p_label = p_classify(p_m)
    PAnswer(p_label) >> PAsk(p_got)
    return p_got @ PAsk


@workflow
def sqlite_effect_round(n: int @ PAsk):
    PAsk(n) >> PAnswer(p_m)
    PAnswer: p_effect_value = p_external_counter(p_m)
    PAnswer(p_effect_value) >> PAsk(p_got)
    return p_got @ PAsk


@workflow
def sqlite_hidden_effect_round(n: int @ PAsk):
    PAsk(n) >> PAnswer(p_m)
    PAnswer: p_effect_value = p_hidden_external_counter(p_m)
    PAnswer(p_effect_value) >> PAsk(p_got)
    return p_got @ PAsk


@workflow
def sqlite_human_round(prompt: str @ PHuman):
    PHuman: p_approved = p_review(prompt)
    return p_approved @ PHuman


@workflow
def sqlite_formula_round(approved: bool @ SQLiteCPLPlanner) -> str:
    SQLiteCPLPlanner(approved) >> SQLiteCPLExecutor(approved)
    if _sqlite_ya_guard @ SQLiteCPLExecutor:
        SQLiteCPLExecutor: cpl_out = p_yes()
    else:
        SQLiteCPLExecutor: cpl_out = p_no()
    return cpl_out @ SQLiteCPLExecutor


@workflow
def sqlite_field_term_round(src: str @ SQLiteFieldSource, gate: str @ SQLiteFieldSource) -> str:
    SQLiteFieldSource(gate) >> SQLiteFieldGate(gate)
    if _sqlite_field_match_guard @ SQLiteFieldGate:
        SQLiteFieldGate: cpl_out = p_yes()
    else:
        SQLiteFieldGate: cpl_out = p_no()
    return cpl_out @ SQLiteFieldGate


@workflow
def sqlite_formula_loop(loop_n: int @ SQLiteFormulaLoopOwner, loop_limit: int @ SQLiteFormulaLoopOwner) -> int:
    while _sqlite_formula_loop_guard @ SQLiteFormulaLoopOwner:
        SQLiteFormulaLoopOwner: loop_n = p_inc(loop_n)
    return loop_n @ SQLiteFormulaLoopOwner


def test_run_sqlite_two_role_branch_matches_inprocess():
    wf = _two_role_branch_workflow()
    initial = {"A": {"x": 7}}
    assert run_sqlite(wf, [A, B], initial, timeout=10) == run(wf, [A, B], initial, timeout=10)


def test_run_sqlite_two_role_branch_false_matches_inprocess():
    wf = _two_role_branch_workflow()
    initial = {"A": {"x": -3}}
    assert run_sqlite(wf, [A, B], initial, timeout=10) == run(wf, [A, B], initial, timeout=10)


def test_run_sqlite_loop_matches_inprocess():
    initial = {"A": {"n": 0, "limit": 3}}
    assert run_sqlite(counter_loop, [LoopA, LoopB], initial, timeout=10) == run(
        counter_loop, [LoopA, LoopB], initial, timeout=10
    )


def test_local_supervisor_replays_persistent_store_without_duplicates(tmp_path):
    path = str(tmp_path / "run.sqlite")
    wf = _two_role_branch_workflow()
    initial = {"A": {"x": 7}}
    first = LocalSupervisor(wf, [A, B], initial, store_path=path, timeout=10).run()
    conn = open_store(path)
    before = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    second = LocalSupervisor(wf, [A, B], {"A": {"x": -3}}, store_path=path, timeout=10).run()
    after = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert first is True and second is True
    assert after == before


def test_run_sqlite_persists_trace_events_without_live_trace(tmp_path):
    path = str(tmp_path / "trace.sqlite")
    initial = {"PUser": {"a": 2}, "POwner": {"b": 5}}
    assert run_sqlite(
        sqlite_parallel_sum,
        [PUser, POwner, PCompute],
        initial,
        store_path=path,
        timeout=10,
    ) == 7

    conn = open_store(path)
    events = [item["event"] for item in list_trace_events(conn)]
    event_types = [event["type"] for event in events]
    assert "send" in event_types
    assert "recv" in event_types
    assert "act_start" in event_types
    assert "act" in event_types
    assert any(event.get("action") == "p_add" for event in events)


def test_run_sqlite_replay_does_not_duplicate_trace_events(tmp_path):
    path = str(tmp_path / "trace-replay.sqlite")
    initial = {"PUser": {"a": 2}, "POwner": {"b": 5}}
    assert run_sqlite(
        sqlite_parallel_sum,
        [PUser, POwner, PCompute],
        initial,
        store_path=path,
        timeout=10,
    ) == 7
    conn = open_store(path)
    before = len(list_trace_events(conn))
    assert run_sqlite(
        sqlite_parallel_sum,
        [PUser, POwner, PCompute],
        {"PUser": {"a": 99}, "POwner": {"b": 100}},
        store_path=path,
        timeout=10,
    ) == 7
    assert len(list_trace_events(conn)) == before


def test_begin_immediate_retries_database_locked(monkeypatch):
    sleeps = []
    monkeypatch.setattr("zippergen.role_runner.time.sleep", lambda seconds: sleeps.append(seconds))

    class FlakyConnection:
        def __init__(self):
            self.calls = 0

        def execute(self, sql):
            self.calls += 1
            assert sql == "BEGIN IMMEDIATE"
            if self.calls < 3:
                raise sqlite3.OperationalError("database is locked")

    conn = FlakyConnection()
    _begin_immediate(conn)
    assert conn.calls == 3
    assert sleeps == [0.05, 0.05]


def test_role_runner_idle_backoff_grows_and_resets(monkeypatch, tmp_path):
    sleeps = []
    monkeypatch.setattr("zippergen.role_runner.time.sleep", lambda seconds: sleeps.append(seconds))
    runner = RoleRunner(
        open_store(str(tmp_path / "idle.sqlite")),
        PAsk.name,
        project(sqlite_external_round, PAsk),
        {"n": 1},
        sqlite_external_round.ns,
    )

    for _ in range(8):
        runner._sleep_after_idle_step()

    assert sleeps == pytest.approx([0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0, 1.0])
    runner._reset_idle_backoff()
    runner._sleep_after_idle_step()
    assert sleeps[-1] == pytest.approx(0.02)


def test_workflow_call_uses_sqlite_execution_by_default():
    wf = _two_role_branch_workflow()
    assert wf._execution == "sqlite"
    wf.configure(ui=False, timeout=10)
    assert wf(x=7) is True


def test_workflow_call_can_opt_into_memory_execution():
    wf = _two_role_branch_workflow()
    wf.configure(execution="memory", ui=False, timeout=10)
    assert wf(x=7) is True


def test_workflow_call_sqlite_persistent_store_replays_without_duplicates(tmp_path):
    path = str(tmp_path / "run.sqlite")
    wf = _two_role_branch_workflow()
    wf.configure(execution="sqlite", store_path=path, ui=False, timeout=10)
    assert wf(x=7) is True
    conn = open_store(path)
    before = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert wf(x=-3) is True
    after = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert after == before


def test_workflow_configure_rejects_unknown_execution_mode():
    wf = _two_role_branch_workflow()
    with pytest.raises(ValueError, match="execution"):
        wf.configure(execution="nonsense")


def test_run_sqlite_parallel_matches_inprocess():
    initial = {"PUser": {"a": 2}, "POwner": {"b": 5}}
    assert run_sqlite(sqlite_parallel_sum, [PUser, POwner, PCompute], initial, timeout=10) == run(
        sqlite_parallel_sum, [PUser, POwner, PCompute], initial, timeout=10
    )


def test_role_runner_receive_any_uses_sqlite_rowid_order(tmp_path):
    path = str(tmp_path / "receive-any.sqlite")
    conn = open_store(path)
    z_sender = DurableChannel(conn, SQLiteAnyZ.name)
    a_sender = DurableChannel(conn, SQLiteAnyA.name)
    conn.execute("BEGIN")
    z_rowid = z_sender.put(SQLiteAnyZ.name, SQLiteAnyR.name, "main", (9,))
    z_sender.commit_txn()
    conn.execute("BEGIN")
    a_rowid = a_sender.put(SQLiteAnyA.name, SQLiteAnyR.name, "main", (1,))
    a_sender.commit_txn()

    local_stmt = ReceiveAnyStmt(
        SQLiteAnyR,
        (
            (SQLiteAnyA, (VarExpr(sqlite_any_a),)),
            (SQLiteAnyZ, (VarExpr(sqlite_any_z),)),
        ),
    )
    events = []
    runner = RoleRunner(
        conn,
        SQLiteAnyR.name,
        local_stmt,
        {},
        {"sqlite_any_a": sqlite_any_a, "sqlite_any_z": sqlite_any_z},
        trace=events.append,
    )

    env = runner.run()
    recv_events = [
        event
        for event in events
        if event["type"] == "recv" and event["to"] == SQLiteAnyR.name
    ]

    assert z_rowid < a_rowid
    assert [event["from"] for event in recv_events] == [SQLiteAnyZ.name, SQLiteAnyA.name]
    assert env["sqlite_any_z"] == 9
    assert env["sqlite_any_a"] == 1


def test_run_sqlite_persists_final_result(tmp_path):
    path = str(tmp_path / "result.sqlite")
    initial = {"PUser": {"a": 2}, "POwner": {"b": 5}}
    result = run_sqlite(
        sqlite_parallel_sum,
        [PUser, POwner, PCompute],
        initial,
        store_path=path,
        timeout=10,
    )
    conn = open_store(path)
    assert result == 7
    assert load_workflow_result(conn, "sqlite_parallel_sum") == 7


def test_workflow_call_sqlite_parallel():
    sqlite_parallel_sum.configure(execution="sqlite", ui=False, timeout=10)
    assert sqlite_parallel_sum(a=2, b=5) == 7
    sqlite_parallel_sum.configure(execution="memory", ui=False)


def test_run_sqlite_external_act_replays_without_backend_call(tmp_path):
    path = str(tmp_path / "external.sqlite")
    calls = {"n": 0}

    def backend(action, inputs):
        calls["n"] += 1
        return {"p_label": calls["n"] * 10}

    initial = {"PAsk": {"n": 1}}
    first = run_sqlite(
        sqlite_external_round,
        [PAsk, PAnswer],
        initial,
        store_path=path,
        llm_backend=backend,
        timeout=10,
    )
    conn = open_store(path)
    before = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    second = run_sqlite(
        sqlite_external_round,
        [PAsk, PAnswer],
        {"PAsk": {"n": 999}},
        store_path=path,
        llm_backend=backend,
        timeout=10,
    )
    after = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert first == 10 and second == 10
    assert calls["n"] == 1
    assert after == before


def test_run_sqlite_effect_action_replays_without_python_call(tmp_path):
    path = str(tmp_path / "effect.sqlite")
    _effect_calls["n"] = 0

    first = run_sqlite(
        sqlite_effect_round,
        [PAsk, PAnswer],
        {"PAsk": {"n": 10}},
        store_path=path,
        timeout=10,
    )
    conn = open_store(path)
    before = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    second = run_sqlite(
        sqlite_effect_round,
        [PAsk, PAnswer],
        {"PAsk": {"n": 999}},
        store_path=path,
        timeout=10,
    )
    after = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    assert first == 11 and second == 11
    assert _effect_calls["n"] == 1
    assert after == before


def test_run_sqlite_hidden_effect_does_not_emit_trace_card(tmp_path):
    path = str(tmp_path / "hidden-effect.sqlite")
    assert run_sqlite(
        sqlite_hidden_effect_round,
        [PAsk, PAnswer],
        {"PAsk": {"n": 10}},
        store_path=path,
        timeout=10,
    ) == 11

    conn = open_store(path)
    events = [item["event"] for item in list_trace_events(conn)]
    assert not any(event.get("action") == "p_hidden_external_counter" for event in events)


def test_run_sqlite_human_action_uses_backend_and_replays_result(tmp_path):
    path = str(tmp_path / "human.sqlite")
    calls = {"n": 0}

    def human_backend(action, inputs):
        calls["n"] += 1
        return {action.output: True}

    first = run_sqlite(
        sqlite_human_round,
        [PHuman],
        {"PHuman": {"prompt": "plan"}},
        store_path=path,
        human_backend=human_backend,
        timeout=10,
    )
    second = run_sqlite(
        sqlite_human_round,
        [PHuman],
        {"PHuman": {"prompt": "changed"}},
        store_path=path,
        human_backend=human_backend,
        timeout=10,
    )
    assert first is True and second is True
    assert calls["n"] == 1
    conn = open_store(path)
    assert conn.execute("SELECT COUNT(*) FROM human_tasks").fetchone()[0] == 1
    task = conn.execute("SELECT task_id, status, result FROM human_tasks").fetchone()
    task_id = task[0]
    trace_events = [
        item["event"]
        for item in list_trace_events(conn)
        if item["event"].get("action") == "p_review"
    ]
    assert {event["seq"] for event in trace_events} == {task_id}
    assert task[1] == "done"


def test_run_sqlite_formula_guard_matches_inprocess():
    for approved in (True, False):
        initial = {"SQLiteCPLPlanner": {"approved": approved}}
        assert run_sqlite(
            sqlite_formula_round,
            [SQLiteCPLPlanner, SQLiteCPLExecutor],
            initial,
            timeout=10,
        ) == run(
            sqlite_formula_round,
            [SQLiteCPLPlanner, SQLiteCPLExecutor],
            initial,
            timeout=10,
        )


def test_run_sqlite_field_term_guard_matches_inprocess():
    for gate, expected in (("v1", "yes"), ("v2", "no")):
        initial = {"SQLiteFieldSource": {"src": "v1", "gate": gate}}
        assert run_sqlite(
            sqlite_field_term_round,
            [SQLiteFieldSource, SQLiteFieldGate],
            initial,
            timeout=10,
        ) == expected
        assert run_sqlite(
            sqlite_field_term_round,
            [SQLiteFieldSource, SQLiteFieldGate],
            initial,
            timeout=10,
        ) == run(
            sqlite_field_term_round,
            [SQLiteFieldSource, SQLiteFieldGate],
            initial,
            timeout=10,
        )


def test_run_sqlite_formula_replays_persistent_store_without_duplicates(tmp_path):
    path = str(tmp_path / "formula.sqlite")
    first = run_sqlite(
        sqlite_formula_round,
        [SQLiteCPLPlanner, SQLiteCPLExecutor],
        {"SQLiteCPLPlanner": {"approved": True}},
        store_path=path,
        timeout=10,
    )
    conn = open_store(path)
    before = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    second = run_sqlite(
        sqlite_formula_round,
        [SQLiteCPLPlanner, SQLiteCPLExecutor],
        {"SQLiteCPLPlanner": {"approved": False}},
        store_path=path,
        timeout=10,
    )
    after = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert first == "yes" and second == "yes"
    assert after == before
    assert load_workflow_result(conn, "sqlite_formula_round") == "yes"
    assert conn.execute("SELECT COUNT(*) FROM workflow_results").fetchone()[0] == 1


def test_run_sqlite_formula_loop_replays_without_monitor_unsafe_snapshots(tmp_path):
    path = str(tmp_path / "formula-loop.sqlite")
    first = run_sqlite(
        sqlite_formula_loop,
        [SQLiteFormulaLoopOwner],
        {"SQLiteFormulaLoopOwner": {"loop_n": 0, "loop_limit": 2}},
        store_path=path,
        timeout=10,
    )
    conn = open_store(path)
    before = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    snapshots = conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
    second = run_sqlite(
        sqlite_formula_loop,
        [SQLiteFormulaLoopOwner],
        {"SQLiteFormulaLoopOwner": {"loop_n": 0, "loop_limit": 99}},
        store_path=path,
        timeout=10,
    )
    after = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    assert first == 2 and second == 2
    assert snapshots == 0
    assert after == before


def test_role_runner_waits_for_existing_pending_human_task(tmp_path):
    path = str(tmp_path / "out-of-band-human.sqlite")
    local = project(sqlite_human_round, PHuman)
    locator = next(iter(action_node_paths(local).values()))
    input_hash = _input_hash({"prompt": "plan"})
    task_id = human_task_id("PHuman", locator, input_hash, 0)

    conn = open_store(path)
    conn.execute("BEGIN")
    ensure_human_task(
        conn,
        task_id=task_id,
        role="PHuman",
        locator=locator,
        action="p_review",
        input_hash=input_hash,
        inputs={"prompt": "plan"},
        spec={"kind": "confirm", "output": "p_approved"},
    )
    conn.execute("COMMIT")

    calls = {"n": 0}
    result_box = {}

    def backend(action, inputs):
        calls["n"] += 1
        raise AssertionError("existing pending task should be answered through SQLite")

    def run_role():
        result_box["env"] = RoleRunner(
            open_store(path),
            "PHuman",
            local,
            {"prompt": "plan"},
            sqlite_human_round.ns,
            human_backend=backend,
        ).run()

    thread = threading.Thread(target=run_role)
    thread.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if conn.execute("SELECT status FROM human_tasks WHERE task_id=?", (task_id,)).fetchone()[0] == "pending":
            break
        time.sleep(0.01)

    conn.execute("BEGIN")
    complete_human_task(conn, task_id, {"p_approved": True})
    conn.execute("COMMIT")
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert calls["n"] == 0
    assert result_box["env"]["p_approved"] is True
    assert conn.execute("SELECT COUNT(*) FROM events WHERE kind='act'").fetchone()[0] == 1


def test_role_runner_terminal_backend_claims_existing_pending_human_task(tmp_path):
    path = str(tmp_path / "inline-resumed-human.sqlite")
    local = project(sqlite_human_round, PHuman)
    locator = next(iter(action_node_paths(local).values()))
    input_hash = _input_hash({"prompt": "plan"})
    task_id = human_task_id("PHuman", locator, input_hash, 0)

    conn = open_store(path)
    conn.execute("BEGIN")
    ensure_human_task(
        conn,
        task_id=task_id,
        role="PHuman",
        locator=locator,
        action="p_review",
        input_hash=input_hash,
        inputs={"prompt": "plan"},
        spec={"kind": "confirm", "output": "p_approved"},
    )
    conn.execute("COMMIT")

    calls = {"n": 0}

    def backend(action, inputs):
        calls["n"] += 1
        return {action.output: True}

    setattr(backend, "claims_pending_human_tasks", True)
    result = RoleRunner(
        open_store(path),
        "PHuman",
        local,
        {"prompt": "plan"},
        sqlite_human_round.ns,
        human_backend=backend,
    ).run()

    assert calls["n"] == 1
    assert result["p_approved"] is True
    task = conn.execute(
        "SELECT status, result FROM human_tasks WHERE task_id=?",
        (task_id,),
    ).fetchone()
    assert task[0] == "done"
    assert json.loads(task[1]) == {"p_approved": True}


def test_workflow_call_sqlite_ui_completes_human_task_through_store(tmp_path):
    path = str(tmp_path / "sqlite-ui-human.sqlite")
    trace = WebTrace([PHuman], port=0)
    result_box = {}
    errors = []

    sqlite_human_round.configure(
        execution="sqlite",
        store_path=path,
        ui=True,
        trace=trace,
        timeout=10,
    )

    def run_workflow():
        try:
            result_box["result"] = sqlite_human_round(prompt="plan")
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_workflow)
    conn = open_store(path)
    try:
        thread.start()
        task_id = None
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            row = conn.execute("SELECT task_id FROM human_tasks WHERE status='pending'").fetchone()
            if row is not None:
                task_id = row[0]
                break
            time.sleep(0.05)

        assert task_id is not None
        assert trace._complete_sqlite_human_input(task_id, "true") is True
        thread.join(timeout=10)

        assert not thread.is_alive()
        assert errors == []
        assert result_box["result"] is True
        assert conn.execute("SELECT status FROM human_tasks WHERE task_id=?", (task_id,)).fetchone()[0] == "done"
    finally:
        trace.stop()
        sqlite_human_round.configure(execution="memory", ui=False)


def test_workflow_call_sqlite_without_ui_waits_for_human_task_store(tmp_path):
    path = str(tmp_path / "sqlite-no-ui-human.sqlite")
    result_box = {}
    errors = []

    sqlite_human_round.configure(
        execution="sqlite",
        store_path=path,
        ui=False,
        timeout=10,
    )

    def run_workflow():
        try:
            result_box["result"] = sqlite_human_round(prompt="plan")
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_workflow)
    conn = open_store(path)
    try:
        thread.start()
        task_id = None
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            row = conn.execute("SELECT task_id FROM human_tasks WHERE status='pending'").fetchone()
            if row is not None:
                task_id = row[0]
                break
            time.sleep(0.05)

        assert task_id is not None
        conn.execute("BEGIN")
        complete_human_task(conn, task_id, {"p_approved": True})
        conn.execute("COMMIT")
        thread.join(timeout=10)

        assert not thread.is_alive()
        assert errors == []
        assert result_box["result"] is True
        assert conn.execute("SELECT status FROM human_tasks WHERE task_id=?", (task_id,)).fetchone()[0] == "done"
    finally:
        sqlite_human_round.configure(execution="memory", ui=False)
