import json
import plistlib
import subprocess
from pathlib import Path

import pytest

from zippergen.serve import (
    _launchd_service_status,
    _start_deployment_connector_workers,
    _systemd_boot_status,
    main,
)
from zippergen.store import (
    ensure_human_task,
    load_human_task,
    load_human_task_token,
    open_store,
    record_trace_event,
)
from zippergen.workspace import Workspace


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


def test_connector_authorize_google_emits_checked_private_handoff(
    tmp_path,
    monkeypatch,
    capsys,
):
    from zippergen.google_auth import (
        GOOGLE_GMAIL_READONLY_SCOPE,
        GoogleAuthorization,
        decode_google_authorization,
    )

    client = tmp_path / "google-client.json"
    client.write_text(json.dumps({
        "installed": {
            "client_id": "example.apps.googleusercontent.com",
            "client_secret": "private-client-secret",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }))
    monkeypatch.setattr("builtins.input", lambda prompt: str(client))
    monkeypatch.setattr(
        "zippergen.google_auth.authorize_google_client_result",
        lambda value, *, scopes: GoogleAuthorization(
            authorized_user_json=json.dumps({
                "client_id": "example.apps.googleusercontent.com",
                "refresh_token": "private-refresh-token",
            }),
            granted_scopes=tuple(scopes),
            client_id="example.apps.googleusercontent.com",
        ),
    )

    rc = main([
        "connector",
        "authorize",
        "google",
        "--scopes",
        "gmail.readonly",
    ])

    assert rc == 0
    lines = capsys.readouterr().out.strip().splitlines()
    result = decode_google_authorization(lines[-1])
    assert result.granted_scopes == (GOOGLE_GMAIL_READONLY_SCOPE,)
    assert "private-refresh-token" not in "\n".join(lines[:-1])


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
    assert "│ ZipperGen Studio · current " in captured.out
    assert "│ ZipperGen Studio · language " in captured.out
    assert f"Workflow   ✓ {workflow_path.name}:hello" in captured.out
    assert "def hello(topic: str @ User)" in captured.out
    assert "return reply @ User" in captured.out
    assert "add_suffix(topic)" not in captured.out
    workspace_states = list((zippergen_home / "workspaces").glob("*/workspace.json"))
    assert len(workspace_states) == 1


def test_studio_command_mode_renders_errors_and_fails_fast(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "zg-home"))

    rc = main(
        [
            "studio",
            "--project",
            str(tmp_path),
            "--command",
            "workflow validate",
            "--command",
            "project init MustNotRun",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert "✗ " in captured.out
    assert "workflow" in captured.out.casefold()
    assert not (tmp_path / "zippergen.toml").exists()
    assert captured.err == ""


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
    assert "Workflow   ⚠ none selected" in captured.out


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
    assert store_path.exists()
    connection = open_store(str(store_path))
    assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
    connection.close()

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
    service = (
        zippergen_home
        / "deployments"
        / "zippergen-hello-prod.service"
    ).read_text()
    assert "Restart=on-failure" in service
    assert "Restart=always" not in service


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
    launchd = plistlib.loads(
        (
            zippergen_home
            / "deployments"
            / "io.zippergen.hello-prod.plist"
        ).read_bytes()
    )
    assert launchd["KeepAlive"] == {"SuccessfulExit": False}


def test_systemd_boot_status_reports_server_boot_with_lingering(monkeypatch):
    def fake_run(arguments, **_kwargs):
        if "is-enabled" in arguments:
            return subprocess.CompletedProcess(arguments, 0, stdout="enabled\n")
        assert arguments[0] == "loginctl"
        return subprocess.CompletedProcess(arguments, 0, stdout="yes\n")

    monkeypatch.setattr("zippergen.serve.subprocess.run", fake_run)

    status = _systemd_boot_status("reviewed-answer")

    assert status["state"] == "server-boot"
    assert status["kind"] == "success"
    assert "automatic at server boot" in status["detail"]


def test_systemd_boot_status_explains_when_lingering_is_disabled(monkeypatch):
    def fake_run(arguments, **_kwargs):
        if "is-enabled" in arguments:
            return subprocess.CompletedProcess(arguments, 0, stdout="enabled\n")
        return subprocess.CompletedProcess(arguments, 0, stdout="no\n")

    monkeypatch.setattr("zippergen.serve.subprocess.run", fake_run)

    status = _systemd_boot_status("reviewed-answer")

    assert status["state"] == "user-login"
    assert status["kind"] == "warning"
    assert "requires account lingering" in status["detail"]


def test_systemd_boot_status_explains_how_to_enable_manual_service(monkeypatch):
    monkeypatch.setattr(
        "zippergen.serve.subprocess.run",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(
            arguments,
            1,
            stdout="disabled\n",
        ),
    )

    status = _systemd_boot_status("reviewed-answer")

    assert status["state"] == "manual"
    assert "deploy start reviewed-answer" in status["detail"]


@pytest.mark.parametrize("action", ["start", "restart"])
def test_start_and_restart_refuse_a_deployment_that_fails_readiness(
    action,
    tmp_path,
    monkeypatch,
    capsys,
):
    workflow_path = tmp_path / "deploy_workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    zippergen_home = tmp_path / "zg-home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(zippergen_home))
    monkeypatch.setenv("ZIPPERGEN_SERVICE_MANAGER", "systemd")
    main(
        [
            "deploy-local",
            f"{workflow_path}:hello",
            "--name",
            "hello-openai",
            "--llm",
            "openai:gpt-4o-mini",
        ]
    )
    capsys.readouterr()
    monkeypatch.setattr(
        "zippergen.serve._run_systemctl",
        lambda *args, **kwargs: pytest.fail(
            "the service manager must not run after a failed readiness check"
        ),
    )

    rc = main([action, "hello-openai"])

    captured = capsys.readouterr()
    assert rc == 1
    assert "1 failure(s)" in captured.out
    assert "model credential OPENAI_API_KEY" in captured.out
    assert f"was not {action}ed because readiness checks found failures" in (
        captured.out
    )


def test_launchd_status_distinguishes_a_loaded_crash_loop(monkeypatch):
    monkeypatch.setattr(
        "zippergen.serve.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            stdout=(
                "state = spawn scheduled\n"
                "active count = 0\n"
                "runs = 9\n"
                "last exit code = 1\n"
                "\t\tstate = active\n"
                "\t\tactive count = 1\n"
            ),
            stderr="",
        ),
    )

    status = _launchd_service_status("reviewed-answer")

    assert status["state"] == "restarting"
    assert status["healthy"] is False
    assert status["runs"] == 9
    assert status["last_exit_code"] == 1


def test_guided_deploy_persists_an_implicit_model_provider_secret(
    tmp_path,
    monkeypatch,
    capsys,
):
    workflow_path = tmp_path / "deploy_workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    zippergen_home = tmp_path / "zg-home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(zippergen_home))

    rc = main(
        [
            "deploy",
            f"{workflow_path}:hello",
            "--name",
            "hello-mistral",
            "--llm",
            "mistral:mistral-small-latest",
            "--provider-secret",
            "MISTRAL_API_KEY=private-key",
            "--yes",
            "--no-install",
            "--no-setup",
            "--no-doctor",
            "--no-start",
        ]
    )

    assert rc == 0
    capsys.readouterr()
    secrets = json.loads(
        (
            zippergen_home
            / "deployments"
            / "hello-mistral.secrets.json"
        ).read_text()
    )
    assert secrets == {"MISTRAL_API_KEY": "private-key"}


def test_guided_deploy_preserves_google_connector_credential_json(
    tmp_path,
    monkeypatch,
    capsys,
):
    workflow_path = tmp_path / "deploy_workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    zippergen_home = tmp_path / "zg-home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(zippergen_home))
    credential = json.dumps(
        {
            "client_id": "example.apps.googleusercontent.com",
            "client_secret": "private-client-secret",
            "refresh_token": "private-refresh-token",
            "token_uri": "https://oauth2.googleapis.com/token",
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    rc = main(
        [
            "deploy",
            f"{workflow_path}:hello",
            "--name",
            "hello-google",
            "--connector-secret",
            "ZIPPERGEN_CONNECTOR_MAILBOX_GOOGLE_CREDENTIAL="
            + credential,
            "--yes",
            "--no-install",
            "--no-setup",
            "--no-doctor",
            "--no-start",
        ]
    )

    assert rc == 0
    capsys.readouterr()
    secrets = json.loads(
        (
            zippergen_home
            / "deployments"
            / "hello-google.secrets.json"
        ).read_text()
    )
    stored = secrets[
        "ZIPPERGEN_CONNECTOR_MAILBOX_GOOGLE_CREDENTIAL"
    ]
    assert stored == credential
    assert json.loads(stored)["refresh_token"] == "private-refresh-token"


def test_guided_deploy_persists_a_local_provider_endpoint(
    tmp_path,
    monkeypatch,
    capsys,
):
    workflow_path = tmp_path / "deploy_workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    zippergen_home = tmp_path / "zg-home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(zippergen_home))

    rc = main(
        [
            "deploy",
            f"{workflow_path}:hello",
            "--name",
            "hello-local",
            "--llm",
            "local:qwen2.5:7b",
            "--provider-env",
            "OLLAMA_BASE_URL=http://127.0.0.1:11434/v1",
            "--yes",
            "--no-install",
            "--no-setup",
            "--no-doctor",
            "--no-start",
        ]
    )

    assert rc == 0
    capsys.readouterr()
    profile = json.loads(
        (
            zippergen_home / "deployments" / "hello-local.json"
        ).read_text()
    )
    assert profile["environment"] == {
        "OLLAMA_BASE_URL": "http://127.0.0.1:11434/v1"
    }


def test_guided_deploy_blocks_a_missing_selected_model_credential(
    tmp_path,
    monkeypatch,
    capsys,
):
    workflow_path = tmp_path / "deploy_workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    zippergen_home = tmp_path / "zg-home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(zippergen_home))

    rc = main(
        [
            "deploy",
            f"{workflow_path}:hello",
            "--name",
            "hello-mistral",
            "--llm",
            "mistral:mistral-small-latest",
            "--yes",
            "--no-install",
            "--no-setup",
            "--no-start",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert "FAIL model credential MISTRAL_API_KEY" in captured.out
    assert "configured but not started" in captured.out


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
    assert "ui" not in profile
    assert "show_decisions" not in profile
    assert "top-secret" not in profile_text
    assert json.loads(secrets_path.read_text()) == {"DEMO_TOKEN": "top-secret"}
    assert secrets_path.stat().st_mode & 0o077 == 0
    assert (zippergen_home / "deployments" / "io.zippergen.guided-prod.plist").exists()
    assert Path(profile["bundle"]).exists()
    store_path = Path(profile["store"])
    assert store_path.exists()
    connection = open_store(str(store_path))
    assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
    connection.close()

    workflow_path.unlink()
    rc = main(["run-deployment", "guided-prod"])
    captured = capsys.readouterr()
    assert rc == 0
    assert json.loads(captured.out) == {"result": "hello:safe:token:deploy"}


def test_explicit_redeploy_replaces_the_named_deployment_source(
    tmp_path,
    monkeypatch,
    capsys,
):
    workflow_path = tmp_path / "workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    zippergen_home = tmp_path / "zg-home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(zippergen_home))
    arguments = [
        "deploy",
        f"{workflow_path}:hello",
        "--name",
        "hello-redeploy",
        "--input",
        "topic=updated",
        "--yes",
        "--no-install",
        "--no-setup",
        "--no-doctor",
        "--no-start",
    ]

    assert main(arguments) == 0
    capsys.readouterr()
    first_profile = json.loads(
        (zippergen_home / "deployments" / "hello-redeploy.json").read_text()
    )
    first_bundle = Path(first_profile["bundle"])
    store_path = Path(first_profile["store"])
    connection = open_store(str(store_path))
    connection.execute(
        "INSERT INTO adapter_state(key, value, updated_at) VALUES (?, ?, ?)",
        ("preserved", b"yes", 1.0),
    )
    connection.close()

    workflow_path.write_text(
        WORKFLOW_SOURCE.replace('return topic + "!"', 'return topic + "?"')
    )
    assert main(arguments) == 0
    capsys.readouterr()
    second_profile = json.loads(
        (zippergen_home / "deployments" / "hello-redeploy.json").read_text()
    )
    second_bundle = Path(second_profile["bundle"])

    assert second_bundle != first_bundle
    connection = open_store(str(store_path))
    assert connection.execute(
        "SELECT value FROM adapter_state WHERE key = ?",
        ("preserved",),
    ).fetchone() == (b"yes",)
    connection.close()
    bundled_workflow = str(second_profile["workflow"]).partition(":")[0]
    assert 'return topic + "?"' in (
        second_bundle / bundled_workflow
    ).read_text()
    workflow_path.unlink()

    assert main(["run-deployment", "hello-redeploy"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"result": "updated?"}


def test_deploy_bundle_carries_every_content_checked_accepted_file(
    tmp_path,
    monkeypatch,
    capsys,
):
    project = tmp_path / "project"
    project.mkdir()
    (project / "workflow.py").write_text(WORKFLOW_SOURCE)
    (project / "helper.py").write_text("VALUE = 'reviewed'\n")
    workspace = Workspace(project, home=tmp_path / "studio-home")
    accepted = workspace.capture_accepted_source(
        "workflow.py:hello",
        files=[
            ("workflow.py", "entry point"),
            ("helper.py", "local Python import"),
        ],
        specification="Return the topic with one suffix.",
        git_provenance={
            "available": False,
            "commit": None,
            "dirty": None,
            "status": [],
        },
    )
    accepted_root = Path(str(accepted["root"]))
    deployment_home = tmp_path / "deployment-home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(deployment_home))
    monkeypatch.chdir(accepted_root)

    assert main([
        "deploy",
        "workflow.py:hello",
        "--name",
        "accepted-files",
        "--input",
        "topic=accepted",
        "--yes",
        "--no-install",
        "--no-setup",
        "--no-doctor",
        "--no-start",
    ]) == 0
    capsys.readouterr()

    profile = json.loads(
        (
            deployment_home
            / "deployments"
            / "accepted-files.json"
        ).read_text()
    )
    bundled = set(profile["bundled_files"])
    assert {
        "workflow.py",
        "helper.py",
        "specification.md",
        ".zippergen-accepted.json",
    }.issubset(bundled)


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

    profile_path = zippergen_home / "deployments" / "guided-prod.json"
    legacy_profile = json.loads(profile_path.read_text())
    legacy_profile["ui"] = True
    legacy_profile["show_decisions"] = True
    profile_path.write_text(json.dumps(legacy_profile))

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
    assert "ui" not in profile
    assert "show_decisions" not in profile
    assert secrets == {"DEMO_TOKEN": "top-secret"}


@pytest.mark.parametrize("flag", ["--ui", "--show-decisions"])
def test_run_rejects_retired_browser_flags(flag):
    with pytest.raises(SystemExit) as exc:
        main(["run", "unused.py:workflow", flag])

    assert exc.value.code == 2


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
    assert checks["sqlite store"]["status"] == "ok"
    assert "initialized but empty" in checks["sqlite store"]["detail"]


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


def test_deployment_starts_one_telegram_bridge_for_shared_routes(
    tmp_path,
    monkeypatch,
):
    observed = []
    monkeypatch.setenv("ZIPPERGEN_CONNECTOR_TELEGRAM_TOKEN", "secret")
    monkeypatch.setattr(
        "zippergen.telegram_notify.TelegramDeploymentNotifier.run_forever",
        lambda notifier: observed.append(
            (dict(notifier.assignments), dict(notifier.routes))
        ),
    )
    profile = {
        "store": str(tmp_path / "deployment.sqlite"),
        "connectors": {
            "human:Writer": {
                "type": "human",
                "target": "Writer",
                "kind": "telegram",
                "configuration": "team-chat",
                "chat_id": "123",
                "channel": "telegram:team-chat",
                "token_env": "ZIPPERGEN_CONNECTOR_TELEGRAM_TOKEN",
            },
            "human:Reviewer": {
                "type": "human",
                "target": "Reviewer",
                "kind": "telegram",
                "configuration": "team-chat",
                "chat_id": "123",
                "channel": "telegram:team-chat",
                "token_env": "ZIPPERGEN_CONNECTOR_TELEGRAM_TOKEN",
            },
        },
    }

    threads = _start_deployment_connector_workers(profile)
    for thread in threads:
        thread.join(timeout=1)

    assert len(threads) == 1
    assert observed[0][0] == {
        "Writer": "team-chat",
        "Reviewer": "team-chat",
    }
