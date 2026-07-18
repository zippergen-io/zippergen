import json
from pathlib import Path

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

GUIDED_WORKFLOW_SOURCE = """
import os

from zippergen import DeploymentField, DeploymentSpec, Lifeline, pure, workflow

User = Lifeline("User")
PREFIX = ""

zippergen_deployment = DeploymentSpec(
    name="guided-demo",
    fields=(
        DeploymentField("prefix", "Reply prefix", default="guided", required=True),
        DeploymentField(
            "demo_token",
            "Demo token",
            target="env",
            env="DEMO_TOKEN",
            required=True,
            secret=True,
        ),
        DeploymentField(
            "mode",
            "Demo mode",
            target="env",
            env="DEMO_MODE",
            default="safe",
            required=True,
        ),
    ),
)

def zippergen_setup(config):
    global PREFIX
    PREFIX = str(config.option("prefix", ""))

@pure
def describe(topic: str) -> str:
    token_state = "token" if os.environ.get("DEMO_TOKEN") else "missing"
    return f"{PREFIX}:{os.environ.get('DEMO_MODE')}:{token_state}:{topic}"

@workflow
def guided(topic: str @ User) -> str:
    User: reply = describe(topic)
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


def test_show_command_renders_code_and_agent_projection(tmp_path, capsys):
    workflow_path = tmp_path / "show_workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)

    rc = main(["show", f"{workflow_path}:hello"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "@workflow" in captured.out
    assert "User: reply = add_suffix(topic)" in captured.out

    rc = main(["show", f"{workflow_path}:hello", "--agent", "User", "--format", "json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 0
    assert payload["agent"] == "User"
    assert "Generated local projection for User" in payload["code"]


def test_validate_command_checks_projection_and_deployment_metadata(tmp_path, capsys):
    workflow_path = tmp_path / "validate_workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)

    rc = main(["validate", f"{workflow_path}:hello", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 0
    assert payload["valid"] is True
    assert payload["lifelines"] == ["User"]
    assert "User" in payload["projections"]
    assert all(check["status"] == "ok" for check in payload["checks"])


def test_diff_command_reports_semantic_changes(tmp_path, capsys):
    before_path = tmp_path / "before_workflow.py"
    after_path = tmp_path / "after_workflow.py"
    before_path.write_text(WORKFLOW_SOURCE)
    after_path.write_text(WORKFLOW_SOURCE.replace("topic + \"!\"", "topic + \"?\""))

    rc = main([
        "diff",
        f"{before_path}:hello",
        f"{after_path}:hello",
        "--format",
        "json",
    ])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 0
    assert payload["changed"] is True
    changed = payload["changes"]["action_definitions"]["changed"]
    assert changed[0]["name"] == "add_suffix"
    assert "implementation_hash" in changed[0]["fields"]


def test_diff_command_ignores_action_formatting_and_comments(tmp_path, capsys):
    before_path = tmp_path / "before_workflow.py"
    after_path = tmp_path / "after_workflow.py"
    before_path.write_text(WORKFLOW_SOURCE)
    after_path.write_text(
        WORKFLOW_SOURCE.replace(
            'return topic + "!"',
            'return topic+"!"  # same implementation',
        )
    )

    rc = main([
        "diff",
        f"{before_path}:hello",
        f"{after_path}:hello",
        "--format",
        "json",
    ])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["changed"] is False


def test_snapshot_then_diff_supports_assistant_refinement_loop(tmp_path, capsys):
    workflow_path = tmp_path / "workflow.py"
    snapshot_path = tmp_path / "before.json"
    workflow_path.write_text(WORKFLOW_SOURCE)

    rc = main([
        "snapshot",
        f"{workflow_path}:hello",
        "--output",
        str(snapshot_path),
    ])
    assert rc == 0
    assert json.loads(snapshot_path.read_text())["schema"] == "zippergen.workflow-semantics.v1"
    capsys.readouterr()

    workflow_path.write_text(WORKFLOW_SOURCE.replace("topic + \"!\"", "topic + \"?\""))
    rc = main([
        "diff",
        str(snapshot_path),
        f"{workflow_path}:hello",
        "--format",
        "json",
    ])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["changed"] is True
    assert payload["changes"]["action_definitions"]["changed"][0]["name"] == "add_suffix"


def test_studio_commands_remember_workflow_and_render_code(
    tmp_path, monkeypatch, capsys
):
    workflow_path = tmp_path / "studio_workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    zippergen_home = tmp_path / "zg-home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(zippergen_home))

    rc = main([
        "studio",
        f"{workflow_path}:hello",
        "--project",
        str(tmp_path),
        "--command",
        "current",
        "--command",
        "show communications",
    ])

    captured = capsys.readouterr()
    assert rc == 0
    assert "ZipperGen Studio" in captured.out
    assert f"Workflow: {workflow_path.name}:hello" in captured.out
    assert "def hello(topic: str @ User)" in captured.out
    assert "return reply @ User" in captured.out
    assert "add_suffix(topic)" not in captured.out
    workspace_states = list((zippergen_home / "workspaces").glob("*/workspace.json"))
    assert len(workspace_states) == 1


def test_dev_command_creates_a_managed_durable_run(tmp_path, monkeypatch, capsys):
    workflow_path = tmp_path / "dev_workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    zippergen_home = tmp_path / "zg-home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(zippergen_home))

    rc = main([
        "dev",
        f"{workflow_path}:hello",
        "--project",
        str(tmp_path),
        "--input",
        "topic=durable",
    ])

    captured = capsys.readouterr()
    assert rc == 0
    assert "Workflow hello: valid" in captured.out
    assert "Result: durable!" in captured.out
    run_records = list((zippergen_home / "workspaces").glob("*/runs/*.json"))
    assert len(run_records) == 1
    record = json.loads(run_records[0].read_text())
    assert record["status"] == "done"
    assert Path(record["store"]).exists()


def test_no_command_opens_studio(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "zg-home"))
    responses = iter(["exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(responses))

    rc = main([])

    captured = capsys.readouterr()
    assert rc == 0
    assert "ZipperGen Studio" in captured.out
    assert "No workflow selected." in captured.out


def test_dev_run_id_requires_resume():
    try:
        main(["dev", "--run-id", "old-run"])
    except SystemExit as exc:
        assert str(exc) == "--run-id requires --resume."
    else:
        raise AssertionError("--run-id without --resume should fail")


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
        "--llm-for",
        "User=mock",
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
    assert profile["llms"] == {"User": "mock"}
    assert profile["inputs"] == {"topic": "deploy"}
    assert script_path.exists()
    assert f"ZIPPERGEN_HOME={zippergen_home}" in script_path.read_text()
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
    monkeypatch.setenv("ZIPPERGEN_SERVICE_MANAGER", "systemd")
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


def test_start_deployment_dry_run_prints_launchd_commands(tmp_path, monkeypatch, capsys):
    workflow_path = tmp_path / "deploy_workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    zippergen_home = tmp_path / "zg-home"
    launch_agents = tmp_path / "LaunchAgents"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(zippergen_home))
    monkeypatch.setenv("ZIPPERGEN_LAUNCH_AGENTS_DIR", str(launch_agents))
    monkeypatch.setenv("ZIPPERGEN_SERVICE_MANAGER", "launchd")
    main([
        "deploy-local",
        f"{workflow_path}:hello",
        "--name",
        "hello-prod",
    ])
    capsys.readouterr()

    rc = main(["start", "hello-prod", "--dry-run"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "Install launchd agent:" in captured.out
    assert "io.zippergen.hello-prod.plist" in captured.out
    assert "launchctl bootout" in captured.out
    assert "launchctl bootstrap" in captured.out


def test_guided_deploy_persists_config_and_private_secrets(tmp_path, monkeypatch, capsys):
    workflow_path = tmp_path / "guided_workflow.py"
    workflow_path.write_text(GUIDED_WORKFLOW_SOURCE)
    zippergen_home = tmp_path / "zg-home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(zippergen_home))

    rc = main([
        "deploy",
        f"{workflow_path}:guided",
        "--name",
        "guided-prod",
        "--input",
        "topic=deploy",
        "--set",
        "prefix=hello",
        "--set",
        "demo_token=top-secret",
        "--yes",
        "--no-install",
        "--no-setup",
        "--no-doctor",
        "--no-start",
    ])

    captured = capsys.readouterr()
    profile_path = zippergen_home / "deployments" / "guided-prod.json"
    secrets_path = zippergen_home / "deployments" / "guided-prod.secrets.json"
    profile_text = profile_path.read_text()
    profile = json.loads(profile_text)
    assert rc == 0
    assert "Deployment: guided-prod" in captured.out
    assert profile["options"]["prefix"] == "hello"
    assert profile["environment"] == {"DEMO_MODE": "safe"}
    assert profile["secret_names"] == ["DEMO_TOKEN"]
    assert "top-secret" not in profile_text
    assert json.loads(secrets_path.read_text()) == {"DEMO_TOKEN": "top-secret"}
    assert secrets_path.stat().st_mode & 0o077 == 0
    assert (zippergen_home / "deployments" / "io.zippergen.guided-prod.plist").exists()
    assert Path(profile["bundle"]).exists()

    workflow_path.unlink()
    rc = main(["run-deployment", "guided-prod"])
    captured = capsys.readouterr()
    assert rc == 0
    assert json.loads(captured.out) == {"result": "hello:safe:token:deploy"}


def test_configure_keeps_existing_secret_when_updating_public_field(tmp_path, monkeypatch, capsys):
    workflow_path = tmp_path / "guided_workflow.py"
    workflow_path.write_text(GUIDED_WORKFLOW_SOURCE)
    zippergen_home = tmp_path / "zg-home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(zippergen_home))
    main([
        "deploy",
        f"{workflow_path}:guided",
        "--name",
        "guided-prod",
        "--set",
        "demo_token=top-secret",
        "--yes",
        "--no-install",
        "--no-setup",
        "--no-doctor",
        "--no-start",
    ])
    capsys.readouterr()

    rc = main([
        "configure",
        "guided-prod",
        "--set",
        "prefix=updated",
        "--yes",
        "--no-doctor",
    ])

    capsys.readouterr()
    profile = json.loads((zippergen_home / "deployments" / "guided-prod.json").read_text())
    secrets = json.loads((zippergen_home / "deployments" / "guided-prod.secrets.json").read_text())
    assert rc == 0
    assert profile["options"]["prefix"] == "updated"
    assert secrets == {"DEMO_TOKEN": "top-secret"}


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


def test_doctor_reports_deployment_checks(tmp_path, monkeypatch, capsys):
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

    rc = main(["doctor", "hello-prod", "--json", "--no-systemd"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    checks = {check["name"]: check for check in payload["checks"]}
    assert rc == 0
    assert checks["profile"]["status"] == "ok"
    assert checks["workflow import"]["status"] == "ok"
    assert checks["run script"]["status"] == "ok"
    assert checks["systemd template"]["status"] == "ok"
    assert checks["sqlite store"]["status"] == "warn"


def test_doctor_returns_failure_for_broken_profile(tmp_path, monkeypatch, capsys):
    zippergen_home = tmp_path / "zg-home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(zippergen_home))
    deployments = zippergen_home / "deployments"
    deployments.mkdir(parents=True)
    (deployments / "broken.json").write_text(json.dumps({
        "name": "broken",
        "workflow": "missing.py:hello",
        "cwd": str(tmp_path / "missing-cwd"),
        "store": str(tmp_path / "runs" / "broken.sqlite"),
        "log": str(tmp_path / "logs" / "broken.log"),
        "python": str(tmp_path / "missing-python"),
    }))

    rc = main(["doctor", "broken", "--json", "--no-systemd"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    failures = [check for check in payload["checks"] if check["status"] == "fail"]
    assert rc == 1
    assert any(check["name"] == "working directory" for check in failures)
    assert any(check["name"] == "run script" for check in failures)


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
