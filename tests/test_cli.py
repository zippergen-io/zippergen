import json

from zippergen.serve import main
from zippergen.store import (
    ensure_human_task,
    load_human_task,
    load_human_task_token,
    open_store,
    record_trace_event,
)


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


def test_run_command_zero_timeout_means_no_deadline(tmp_path, capsys):
    workflow_path = tmp_path / "no_deadline_workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    store_path = tmp_path / "no-deadline.sqlite"

    rc = main([
        "run",
        f"{workflow_path}:hello",
        "--store",
        str(store_path),
        "--input",
        "topic=steady",
        "--timeout",
        "0",
    ])

    captured = capsys.readouterr()
    assert rc == 0
    assert store_path.exists()
    assert json.loads(captured.out) == {"result": "steady!"}


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


def test_deploy_local_creates_profile_and_runs_by_name(tmp_path, monkeypatch, capsys):
    workflow_path = tmp_path / "deploy_workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    zippergen_home = tmp_path / "zg-home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(zippergen_home))

    rc = main([
        "deploy-local",
        f"{workflow_path}:hello",
        "--name",
        "hello-prod",
        "--llm",
        "mock",
        "--input",
        "topic=deploy",
        "--timeout",
        "10",
    ])

    captured = capsys.readouterr()
    profile_path = zippergen_home / "deployments" / "hello-prod.json"
    script_path = zippergen_home / "deployments" / "hello-prod.sh"
    service_path = zippergen_home / "deployments" / "zippergen-hello-prod.service"
    store_path = zippergen_home / "runs" / "hello-prod.sqlite"
    profile = json.loads(profile_path.read_text())
    assert rc == 0
    assert "Run: zippergen run-deployment hello-prod" in captured.out
    assert profile["name"] == "hello-prod"
    assert profile["workflow"] == f"{workflow_path}:hello"
    assert profile["store"] == str(store_path)
    assert profile["llm"] == "mock"
    assert profile["inputs"] == {"topic": "deploy"}
    assert script_path.exists()
    assert service_path.exists()

    rc = main(["run-deployment", "hello-prod"])
    captured = capsys.readouterr()
    assert rc == 0
    assert store_path.exists()
    assert json.loads(captured.out) == {"result": "deploy!"}

    rc = main(["status", "hello-prod", "--json"])
    captured = capsys.readouterr()
    status = json.loads(captured.out)
    assert rc == 0
    assert status["store"] == str(store_path)
    assert status["state"] == "done"


def test_start_deployment_dry_run_prints_systemd_commands(tmp_path, monkeypatch, capsys):
    workflow_path = tmp_path / "deploy_workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    zippergen_home = tmp_path / "zg-home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(zippergen_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    main([
        "deploy-local",
        f"{workflow_path}:hello",
        "--name",
        "hello-prod",
    ])
    capsys.readouterr()

    rc = main(["start", "hello-prod", "--enable", "--dry-run"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "Install systemd unit:" in captured.out
    assert "zippergen-hello-prod.service" in captured.out
    assert "systemctl --user daemon-reload" in captured.out
    assert "systemctl --user enable zippergen-hello-prod.service" in captured.out
    assert "systemctl --user start zippergen-hello-prod.service" in captured.out


def test_logs_command_tails_deployment_log(tmp_path, monkeypatch, capsys):
    workflow_path = tmp_path / "deploy_workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    zippergen_home = tmp_path / "zg-home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(zippergen_home))
    main([
        "deploy-local",
        f"{workflow_path}:hello",
        "--name",
        "hello-prod",
    ])
    capsys.readouterr()
    profile = json.loads((zippergen_home / "deployments" / "hello-prod.json").read_text())
    log_path = profile["log"]
    with open(log_path, "w") as f:
        f.write("first\nsecond\nthird\n")

    rc = main(["logs", "hello-prod", "--tail", "2"])

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.splitlines() == ["second", "third"]


def test_status_rejects_deployment_and_store_together(tmp_path, monkeypatch):
    zippergen_home = tmp_path / "zg-home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(zippergen_home))
    (zippergen_home / "deployments").mkdir(parents=True)
    (zippergen_home / "deployments" / "demo.json").write_text(json.dumps({
        "name": "demo",
        "workflow": "missing.py:demo",
        "store": str(tmp_path / "demo.sqlite"),
    }))

    try:
        main(["status", "demo", "--store", str(tmp_path / "other.sqlite")])
    except SystemExit as exc:
        assert "either a deployment name or --store" in str(exc)
    else:
        raise AssertionError("status should reject ambiguous store selection")


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


def test_trace_command_reports_recent_trace_events(tmp_path, capsys):
    store_path = tmp_path / "trace.sqlite"
    conn = open_store(str(store_path))
    first = record_trace_event(
        conn,
        "Writer",
        {"type": "send", "from": "Writer", "to": "User", "channel": "main", "values": ["old"]},
    )
    second = record_trace_event(
        conn,
        "User",
        {
            "type": "recv",
            "from": "Writer",
            "to": "User",
            "channel": "main",
            "bindings": {"draft": "Looks good."},
        },
    )
    conn.close()

    rc = main(["trace", "--store", str(store_path), "--tail", "1"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "Trace events: 1" in captured.out
    assert f"#{second} User recv Writer->User main" in captured.out
    assert "draft" in captured.out
    assert f"#{first}" not in captured.out


def test_trace_command_outputs_json_after_rowid(tmp_path, capsys):
    store_path = tmp_path / "trace-json.sqlite"
    conn = open_store(str(store_path))
    first = record_trace_event(
        conn,
        "Writer",
        {"type": "act_start", "action": "draft", "action_kind": "llm", "inputs": {"topic": "x"}},
    )
    second = record_trace_event(
        conn,
        "Writer",
        {"type": "act", "action": "draft", "action_kind": "llm", "outputs": {"reply": "hello"}},
    )
    conn.close()

    rc = main(["trace", "--store", str(store_path), "--after", str(first), "--json"])

    captured = capsys.readouterr()
    assert rc == 0
    assert json.loads(captured.out) == [
        {
            "rowid": second,
            "role": "Writer",
            "event": {
                "type": "act",
                "action": "draft",
                "action_kind": "llm",
                "outputs": {"reply": "hello"},
            },
        }
    ]


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


def test_tasks_command_generates_stable_channel_tokens(tmp_path, capsys):
    store_path = tmp_path / "task-token.sqlite"
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

    rc = main(["tasks", "--store", str(store_path), "--tokens", "--channel", "email", "--json"])
    captured = capsys.readouterr()
    first = json.loads(captured.out)
    assert rc == 0
    assert first[0]["token"].startswith("zg_")
    assert first[0]["token_channel"] == "email"

    rc = main(["tasks", "--store", str(store_path), "--tokens", "--channel", "email", "--json"])
    captured = capsys.readouterr()
    second = json.loads(captured.out)
    assert rc == 0
    assert second[0]["token"] == first[0]["token"]


def test_approve_command_completes_task_by_token(tmp_path, capsys):
    store_path = tmp_path / "approve-token.sqlite"
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

    main(["tasks", "--store", str(store_path), "--tokens", "--channel", "telegram", "--json"])
    token = json.loads(capsys.readouterr().out)[0]["token"]

    rc = main(["approve", "--store", str(store_path), "--token", token, "--yes"])

    captured = capsys.readouterr()
    assert rc == 0
    assert 'Completed human task task-1: {"approved": true}' in captured.out
    conn = open_store(str(store_path))
    try:
        assert load_human_task(conn, "task-1")["result"] == {"approved": True}
        assert load_human_task_token(conn, token)["used_at"] is not None
    finally:
        conn.close()


def test_notify_stdout_prints_pending_task_with_token(tmp_path, capsys):
    store_path = tmp_path / "notify.sqlite"
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
            "rendered": {
                "instruction": "Approve the deployment?",
                "context": "Production rollout",
            },
        },
    )
    conn.close()

    rc = main([
        "notify",
        "stdout",
        "--store",
        str(store_path),
        "--channel",
        "slack",
    ])

    captured = capsys.readouterr()
    assert rc == 0
    assert "Human task: task-1" in captured.out
    assert "Token: zg_" in captured.out
    assert "Action: User.approve (confirm)" in captured.out
    assert "Approve the deployment?" in captured.out
    assert "Production rollout" in captured.out
    assert "zippergen approve --store" in captured.out
    assert "--token zg_" in captured.out
    assert "--no" in captured.out


def test_notify_stdout_reports_no_pending_tasks(tmp_path, capsys):
    store_path = tmp_path / "notify-empty.sqlite"
    open_store(str(store_path)).close()

    rc = main(["notify", "stdout", "--store", str(store_path)])

    captured = capsys.readouterr()
    assert rc == 0
    assert "No pending human tasks." in captured.out
