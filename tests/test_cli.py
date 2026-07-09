import json

from zippergen.serve import main
from zippergen.store import ensure_human_task, load_human_task, open_store


WORKFLOW_SOURCE = """
from zippergen import Lifeline, pure, workflow

User = Lifeline("User")

@pure
def add_suffix(topic: str) -> str:
    return topic + "!"

@workflow
def hello(topic: str @ User) -> str:
    User: reply = add_suffix(topic)
    return reply @ User
"""

SETUP_WORKFLOW_SOURCE = """
from zippergen import Lifeline, pure, workflow

User = Lifeline("User")
PREFIX = ""

def zippergen_setup(config):
    global PREFIX
    services = config.option("services", "fake")
    prefix = config.option("prefix", "")
    PREFIX = f"{services}:{prefix}:"

@pure
def add_prefix(topic: str) -> str:
    return PREFIX + topic

@workflow
def setup_hello(topic: str @ User) -> str:
    User: reply = add_prefix(topic)
    return reply @ User
"""


def test_run_command_loads_workflow_from_path(tmp_path, capsys):
    workflow_path = tmp_path / "sample_workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    store_path = tmp_path / "run.sqlite"

    rc = main([
        "run",
        f"{workflow_path}:hello",
        "--store",
        str(store_path),
        "--input",
        "topic=deploy",
        "--timeout",
        "10",
    ])

    captured = capsys.readouterr()
    assert rc == 0
    assert store_path.exists()
    assert f"Store: {store_path}" in captured.err
    assert json.loads(captured.out) == {"result": "deploy!"}


def test_run_command_loads_workflow_from_module(tmp_path, monkeypatch, capsys):
    workflow_path = tmp_path / "sample_module_workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    store_path = tmp_path / "module-run.sqlite"
    monkeypatch.syspath_prepend(str(tmp_path))

    rc = main([
        "run",
        "sample_module_workflow:hello",
        "--store",
        str(store_path),
        "--input-json",
        '{"topic": "local"}',
        "--timeout",
        "10",
    ])

    captured = capsys.readouterr()
    assert rc == 0
    assert store_path.exists()
    assert json.loads(captured.out) == {"result": "local!"}


def test_run_command_calls_setup_hook_with_options(tmp_path, capsys):
    workflow_path = tmp_path / "setup_workflow.py"
    workflow_path.write_text(SETUP_WORKFLOW_SOURCE)
    store_path = tmp_path / "setup-run.sqlite"

    rc = main([
        "run",
        f"{workflow_path}:setup_hello",
        "--store",
        str(store_path),
        "--input",
        "topic=deploy",
        "--option",
        "prefix=hook",
        "--services",
        "live",
        "--timeout",
        "10",
    ])

    captured = capsys.readouterr()
    assert rc == 0
    assert store_path.exists()
    assert json.loads(captured.out) == {"result": "live:hook:deploy"}


def test_status_command_reports_completed_run(tmp_path, capsys):
    workflow_path = tmp_path / "status_workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    store_path = tmp_path / "status-run.sqlite"
    main([
        "run",
        f"{workflow_path}:hello",
        "--store",
        str(store_path),
        "--input",
        "topic=status",
        "--timeout",
        "10",
    ])
    capsys.readouterr()

    rc = main(["status", "--store", str(store_path), "--json"])

    captured = capsys.readouterr()
    status = json.loads(captured.out)
    assert rc == 0
    assert status["state"] == "done"
    assert status["event_count"] > 0
    assert status["workflow_results"] == [
        {
            "workflow": "hello",
            "value": "status!",
            "created_at": status["workflow_results"][0]["created_at"],
            "updated_at": status["workflow_results"][0]["updated_at"],
        }
    ]


def test_status_command_reports_pending_human_task(tmp_path, capsys):
    store_path = tmp_path / "pending.sqlite"
    conn = open_store(str(store_path))
    ensure_human_task(
        conn,
        task_id="task-1",
        role="User",
        locator=[0],
        action="approve",
        input_hash=None,
        inputs={"prompt": "Approve?"},
        spec={"kind": "confirm"},
    )
    conn.close()

    rc = main(["status", "--store", str(store_path)])

    captured = capsys.readouterr()
    assert rc == 0
    assert "State: waiting (waiting for 1 human task(s))" in captured.out
    assert "Pending human tasks: 1" in captured.out
    assert "task-1 User.approve" in captured.out


def test_status_command_reports_missing_store(tmp_path, capsys):
    store_path = tmp_path / "missing.sqlite"

    rc = main(["status", "--store", str(store_path), "--json"])

    captured = capsys.readouterr()
    assert rc == 0
    assert json.loads(captured.out) == {
        "store": str(store_path),
        "exists": False,
        "state": "missing",
        "summary": "store does not exist",
    }


def test_tasks_command_lists_pending_tasks(tmp_path, capsys):
    store_path = tmp_path / "tasks.sqlite"
    conn = open_store(str(store_path))
    ensure_human_task(
        conn,
        task_id="task-1",
        role="User",
        locator=[0],
        action="approve",
        input_hash=None,
        inputs={"prompt": "Approve?"},
        spec={
            "kind": "confirm",
            "output": "approved",
            "output_type": "bool",
            "rendered": {"instruction": "Approve this?"},
        },
    )
    conn.close()

    rc = main(["tasks", "--store", str(store_path)])

    captured = capsys.readouterr()
    assert rc == 0
    assert "Pending human tasks: 1" in captured.out
    assert "task-1 User.approve confirm -> approved: bool" in captured.out
    assert "instruction: Approve this?" in captured.out


def test_approve_command_completes_boolean_task(tmp_path, capsys):
    store_path = tmp_path / "approve.sqlite"
    conn = open_store(str(store_path))
    ensure_human_task(
        conn,
        task_id="task-1",
        role="User",
        locator=[0],
        action="approve",
        input_hash=None,
        inputs={"prompt": "Approve?"},
        spec={"kind": "confirm", "output": "approved", "output_type": "bool"},
    )
    conn.close()

    rc = main(["approve", "--store", str(store_path), "--task", "task-1", "--no"])

    captured = capsys.readouterr()
    assert rc == 0
    assert 'Completed human task task-1: {"approved": false}' in captured.out
    conn = open_store(str(store_path))
    try:
        assert load_human_task(conn, "task-1")["result"] == {"approved": False}
    finally:
        conn.close()


def test_approve_command_completes_string_task(tmp_path, capsys):
    store_path = tmp_path / "approve-string.sqlite"
    conn = open_store(str(store_path))
    ensure_human_task(
        conn,
        task_id="task-1",
        role="User",
        locator=[0],
        action="edit_reply",
        input_hash=None,
        inputs={"draft": "Hello"},
        spec={"kind": "edit", "output": "reply", "output_type": "str"},
    )
    conn.close()

    rc = main([
        "approve",
        "--store",
        str(store_path),
        "--task",
        "task-1",
        "--value",
        "Looks good.",
        "--json",
    ])

    captured = capsys.readouterr()
    assert rc == 0
    assert json.loads(captured.out)["result"] == {"reply": "Looks good."}


def test_approve_command_requires_value_for_string_task(tmp_path):
    store_path = tmp_path / "approve-missing-value.sqlite"
    conn = open_store(str(store_path))
    ensure_human_task(
        conn,
        task_id="task-1",
        role="User",
        locator=[0],
        action="edit_reply",
        input_hash=None,
        inputs={"draft": "Hello"},
        spec={"kind": "edit", "output": "reply", "output_type": "str"},
    )
    conn.close()

    try:
        main(["approve", "--store", str(store_path), "--task", "task-1"])
    except SystemExit as exc:
        assert "requires --value" in str(exc)
    else:
        raise AssertionError("approve should reject string tasks without --value")
