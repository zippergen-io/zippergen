import json
import sqlite3
from urllib.request import urlopen

from zipperchat.web import WebTrace, _HTML, _begin_immediate
from zippergen.store import (
    ensure_human_task,
    human_task_id,
    load_human_task,
    open_store,
    record_trace_event,
    write_workflow_result,
)


def test_webtrace_hides_decisions_by_default():
    trace = WebTrace(["A", "B"], port=0)
    assert trace._init_event()["show_decisions"] is False


def test_webtrace_can_show_decisions_explicitly():
    trace = WebTrace(["A", "B"], port=0, show_decisions=True)
    assert trace._init_event()["show_decisions"] is True


def test_dashboard_hides_decisions_by_default():
    trace = WebTrace.dashboard(port=0)
    assert trace._init_event()["show_decisions"] is False


def test_webtrace_html_preserves_view_state_on_reload():
    assert "zc-view-state:" in _HTML
    assert "_restoreAutoOpenSuppressed" in _HTML
    assert "_scheduleRestoreAfterReplay" in _HTML


def test_dashboard_run_scopes_event_paths():
    trace = WebTrace.dashboard(port=0)
    trace.reset()
    run_trace = trace.start_run("alpha", ["A", "B"])
    run_path = run_trace.path

    run_trace({"type": "act", "lifeline": "A"})
    run_trace({"type": "level_push", "path": ["inner"], "lifelines": ["C"]})
    run_trace.done()

    history = trace._bus._history
    assert history[0] == {"type": "init", "dashboard": True, "show_decisions": False}
    assert history[1]["type"] == "run_start"
    assert history[1]["path"] == run_path
    assert history[2]["path"] == run_path
    assert history[3]["path"] == run_path + ["inner"]
    assert history[4] == {"type": "done", "path": run_path}


def test_webtrace_begin_immediate_retries_database_locked(monkeypatch):
    sleeps = []
    monkeypatch.setattr("zipperchat.web.time.sleep", lambda seconds: sleeps.append(seconds))

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


def test_webtrace_publishes_and_completes_sqlite_human_task(tmp_path):
    path = str(tmp_path / "ui.sqlite")
    task_id = human_task_id("Reviewer", [0], "abc", 0)
    conn = open_store(path)
    conn.execute("BEGIN")
    ensure_human_task(
        conn,
        task_id=task_id,
        role="Reviewer",
        locator=[0],
        action="review",
        input_hash="abc",
        inputs={"prompt": "plan"},
        spec={
            "kind": "confirm",
            "output": "approved",
            "output_type": "bool",
            "instruction": "Approve?",
            "rendered": {"instruction": "Approve plan?"},
        },
    )
    conn.execute("COMMIT")

    trace = WebTrace(["Reviewer"], port=0, store_path=path)
    trace._poll_sqlite_human_tasks_once()
    history = trace._bus._history

    assert [event["type"] for event in history] == ["act_start", "human_input_required"]
    assert history[0]["seq"] == task_id
    assert history[0]["inputs"] == {"prompt": "plan"}
    assert history[1]["id"] == task_id
    assert history[1]["instruction"] == "Approve plan?"

    assert trace._complete_sqlite_human_input(task_id, "true") is True
    assert load_human_task(conn, task_id)["result"] == {"approved": True}
    trace._poll_sqlite_human_tasks_once()
    assert trace._bus._history[-1] == {
        "type": "human_input",
        "id": task_id,
        "lifeline": "Reviewer",
        "value": "True",
    }


def test_scoped_webtrace_publishes_sqlite_task_with_path(tmp_path):
    path = str(tmp_path / "ui.sqlite")
    task_id = human_task_id("Reviewer", [0], "abc", 0)
    conn = open_store(path)
    conn.execute("BEGIN")
    ensure_human_task(
        conn,
        task_id=task_id,
        role="Reviewer",
        locator=[0],
        action="review",
        input_hash="abc",
        inputs={},
        spec={"kind": "input", "output": "answer", "output_type": "str", "rendered": {}},
    )
    conn.execute("COMMIT")

    trace = WebTrace.dashboard(port=0)
    run_trace = trace.start_run("run", ["Reviewer"])
    run_trace.use_store(path)
    trace._poll_sqlite_human_tasks_once()

    assert trace._bus._history[-2]["path"] == run_trace.path
    assert trace._bus._history[-1]["path"] == run_trace.path


def test_webtrace_publishes_sqlite_workflow_result(tmp_path):
    path = str(tmp_path / "ui-result.sqlite")
    conn = open_store(path)
    write_workflow_result(conn, "wf", {"answer": 42})

    trace = WebTrace(["Runner"], port=0)
    with trace._store_lock:
        trace._store_bindings[path] = None
    trace._poll_sqlite_human_tasks_once()

    assert trace._bus._history == [{
        "type": "workflow_result",
        "workflow": "wf",
        "value": {"answer": 42},
        "created_at": trace._bus._history[0]["created_at"],
        "updated_at": trace._bus._history[0]["updated_at"],
    }]

    trace._poll_sqlite_human_tasks_once()
    assert len(trace._bus._history) == 1


def test_webtrace_publishes_sqlite_trace_events(tmp_path):
    path = str(tmp_path / "ui-trace.sqlite")
    conn = open_store(path)
    record_trace_event(conn, "A", {"type": "send", "from": "A", "to": "B", "seq": 1})
    record_trace_event(conn, "B", {"type": "recv", "from": "A", "to": "B", "seq": 1})

    trace = WebTrace(["A", "B"], port=0)
    with trace._store_lock:
        trace._store_bindings[path] = None
    trace._poll_sqlite_human_tasks_once()

    assert trace._bus._history == [
        {"type": "send", "from": "A", "to": "B", "seq": 1},
        {"type": "recv", "from": "A", "to": "B", "seq": 1},
    ]

    trace._poll_sqlite_human_tasks_once()
    assert len(trace._bus._history) == 2


def test_scoped_webtrace_publishes_sqlite_trace_events_with_path(tmp_path):
    path = str(tmp_path / "ui-trace.sqlite")
    conn = open_store(path)
    record_trace_event(conn, "A", {"type": "send", "from": "A", "to": "B", "seq": 1})

    trace = WebTrace.dashboard(port=0)
    run_trace = trace.start_run("run", ["A", "B"])
    with trace._store_lock:
        trace._store_bindings[path] = run_trace.path
    trace._poll_sqlite_human_tasks_once()

    assert trace._bus._history[-1] == {
        "type": "send",
        "from": "A",
        "to": "B",
        "seq": 1,
        "path": run_trace.path,
    }


def test_scoped_webtrace_publishes_sqlite_workflow_result_with_path(tmp_path):
    path = str(tmp_path / "ui-result.sqlite")
    conn = open_store(path)
    write_workflow_result(conn, "wf", True)

    trace = WebTrace.dashboard(port=0)
    run_trace = trace.start_run("run", ["Runner"])
    with trace._store_lock:
        trace._store_bindings[path] = run_trace.path
    trace._poll_sqlite_human_tasks_once()

    assert trace._bus._history[-1]["type"] == "workflow_result"
    assert trace._bus._history[-1]["path"] == run_trace.path
    assert trace._bus._history[-1]["value"] is True


def test_webtrace_workflow_results_endpoint(tmp_path):
    path = str(tmp_path / "ui-result.sqlite")
    conn = open_store(path)
    write_workflow_result(conn, "wf", ["done"])

    trace = WebTrace(["Runner"], port=0, store_path=path).start()
    try:
        with urlopen(f"http://127.0.0.1:{trace._port}/workflow-results", timeout=5) as resp:
            data = json.loads(resp.read().decode())
    finally:
        trace.stop()

    assert data == [{
        "workflow": "wf",
        "value": ["done"],
        "created_at": data[0]["created_at"],
        "updated_at": data[0]["updated_at"],
    }]
