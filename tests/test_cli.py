import json
import os
import plistlib
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from zippergen.serve import (
    _deployment_command,
    _launchd_service_status,
    _parse_cli_args,
    _start_deployment_connector_workers,
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
from zippergen import DeploymentField, DeploymentSpec, Lifeline, pure, workflow

User = Lifeline("User")

@pure
def add_suffix(topic: str) -> str:
    return topic + "!"

@workflow
def hello(topic: str @ User) -> str:
    User: reply = add_suffix(topic)
    return reply @ User

zippergen_deployment = DeploymentSpec(fields=(
    DeploymentField(
        "topic", "Topic", target="input", default="deploy", required=True
    ),
))
"""


def _the_deployment(zippergen_home, suffix=".json"):
    """The project's one deployment, whatever name was derived for it."""

    found = sorted((zippergen_home / "deployments").glob(f"*{suffix}"))
    if suffix == ".json":
        found = [p for p in found if not p.name.endswith(".secrets.json")]
    assert len(found) == 1, f"expected one {suffix} deployment file, got {found}"
    return found[0]


def _run_prepared_deployment(zippergen_home) -> int:
    """Exercise the hidden entry point used by generated service scripts."""

    profile = json.loads(_the_deployment(zippergen_home).read_text())
    return main(["__run-deployment", "--profile", str(profile["name"])])


def _deploy_for_test(arguments: list[str]) -> int:
    """Prepare through the public deployment path without external setup."""

    workflow_spec, *deployment_arguments = arguments
    module_path, _separator, _workflow_name = workflow_spec.partition(":")
    root = Path(module_path).resolve().parent
    workspace = Workspace(root)
    if not workspace.manifest_path.exists():
        workspace.initialize_project(name=root.name)
    workspace.select_workflow(workflow_spec, cwd=root)
    previous = Path.cwd()
    try:
        os.chdir(root)
        return main(
            [
                "deploy",
                *deployment_arguments,
                "--no-start",
                "--no-bundle",
                "--no-install",
                "--no-setup",
                "--no-doctor",
                "--yes",
            ]
        )
    finally:
        os.chdir(previous)


def _configure_model_for_test(
    workflow_path: Path,
    home: Path,
    *,
    name: str,
    spec: str,
    credential: tuple[str, str] | None = None,
    base_url: str | None = None,
) -> Workspace:
    """Configure the project's named model through its real persistence API."""

    workspace = Workspace(workflow_path.parent, home=home)
    if not workspace.manifest_path.exists():
        workspace.initialize_project(name=workflow_path.parent.name)
    workflow_spec = f"{workflow_path.name}:hello"
    workspace.select_workflow(workflow_spec, cwd=workflow_path.parent)
    provider, _separator, model = spec.partition(":")
    values = {"provider": provider, "model": model, "spec": spec}
    workspace.save_model_configuration(name, values)
    if base_url:
        workspace.save_provider_profile(provider, {"base_url": base_url})
    workspace.save_model_assignment_profile(
        workflow_spec,
        default=name,
        lifelines={},
        actions={},
    )
    if credential:
        workspace.save_development_credential(*credential)
    return workspace


def _prepared_deployment_store(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> Path:
    """Return the store owned by this temporary project's deployment."""

    workflow_path = tmp_path / "workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    home = tmp_path / "zg-home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    assert _deploy_for_test([f"{workflow_path}:hello"]) == 0
    capsys.readouterr()
    profile = json.loads(_the_deployment(home).read_text())
    return Path(str(profile["store"]))


def test_compact_reports_removed_store_events_and_log_archives(
    tmp_path, monkeypatch, capsys
):
    from zippergen import deployments, serve, storage_maintenance

    store = tmp_path / "run.sqlite"
    home = tmp_path / "zg-home"
    (home / "deployments").mkdir(parents=True)
    name = Workspace(home=home).directory.name
    (home / "deployments" / f"{name}.json").write_text(
        json.dumps({"name": name, "source_cwd": str(Path.cwd()),
                    "store": str(store)})
    )
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    monkeypatch.setattr(
        serve,
        "_load_deployment_profile",
        lambda _name: {"store": str(store), "source_cwd": str(Path.cwd())},
    )
    monkeypatch.setattr(
        storage_maintenance,
        "compact_store",
        lambda _path: SimpleNamespace(
            deleted_total=12,
            before_bytes=4096,
            after_bytes=1024,
        ),
    )
    monkeypatch.setattr(
        deployments,
        "compact_deployment_logs",
        lambda _name, _profile, *, keep_archives: SimpleNamespace(
            removed_archives=2,
            removed_archive_bytes=768,
        ),
    )

    assert main(["deploy", "compact", "--keep-archives", "1"]) == 0

    output = capsys.readouterr().out
    assert "removed events: 12" in output
    assert "reclaimed bytes: 3072" in output
    assert "removed archives: 2 (768 bytes)" in output


def test_deploy_reset_archives_and_recreates_its_owned_store(
    tmp_path, monkeypatch, capsys
):
    store = _prepared_deployment_store(tmp_path, monkeypatch, capsys)
    connection = open_store(str(store))
    connection.execute(
        "INSERT INTO adapter_state(key, value, updated_at) VALUES (?, ?, ?)",
        ("evidence", b"kept", 1.0),
    )
    connection.close()
    stopped = {
        "state": "not-loaded",
        "healthy": False,
        "detail": "not loaded",
    }
    monkeypatch.setattr(
        "zippergen.deployment_platform.deployment_service_status",
        lambda _name: stopped,
    )
    monkeypatch.setattr(
        "zippergen.deployments._deployment_service_status",
        lambda _name: stopped,
    )

    assert main(["deploy", "reset", "--yes"]) == 0

    output = capsys.readouterr().out
    assert "Reset deployment state" in output
    assert "Archived" in output
    connection = open_store(str(store))
    assert connection.execute(
        "SELECT COUNT(*) FROM adapter_state WHERE key='evidence'"
    ).fetchone() == (0,)
    connection.close()
    archives = list(
        (tmp_path / "zg-home" / "trash" / "deployment-stores").glob("*")
    )
    assert len(archives) == 1
    archived_store = archives[0] / store.name
    archived = open_store(str(archived_store))
    assert archived.execute(
        "SELECT value FROM adapter_state WHERE key='evidence'"
    ).fetchone() == (b"kept",)
    archived.close()

SETUP_WORKFLOW_SOURCE = """
from zippergen import DeploymentField, DeploymentSpec, Lifeline, pure, workflow

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

zippergen_deployment = DeploymentSpec(fields=(
    DeploymentField("topic", "Topic", target="input", required=True),
    DeploymentField("prefix", "Prefix", target="option", default=""),
    DeploymentField("services", "Services", target="option", default="fake"),
))
"""

GUIDED_WORKFLOW_SOURCE = """
import os

from zippergen import DeploymentField, DeploymentSpec, Lifeline, pure, workflow

User = Lifeline("User")
PREFIX = ""

zippergen_deployment = DeploymentSpec(
    fields=(
        DeploymentField("prefix", "Reply prefix", default="guided", required=True),
        DeploymentField("topic", "Topic", target="input", required=True),
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


def test_durable_run_records_a_resumable_run_with_an_owned_store(
    tmp_path, monkeypatch, capsys
):
    workflow_path = tmp_path / "sample_workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))

    rc = main([
        "run",
        f"{workflow_path}:hello",
        "--durable",
        "--input",
        "topic=deploy",
        "--timeout",
        "10",
    ])

    captured = capsys.readouterr()
    assert rc == 0
    workspace = Workspace(tmp_path, home=tmp_path / "home")
    record = workspace.current_run()
    assert record is not None
    assert Path(str(record["store"])).exists()
    assert record["status"] == "done"
    assert "Result: deploy!" in captured.out


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
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))

    rc = main([
        "run",
        "sample_module_workflow:hello",
        "--input-json",
        '{"topic": "local"}',
        "--timeout",
        "10",
    ])

    captured = capsys.readouterr()
    assert rc == 0
    assert json.loads(captured.out) == {"result": "local!"}


def test_run_command_zero_timeout_means_no_deadline(tmp_path, monkeypatch, capsys):
    workflow_path = tmp_path / "no_deadline_workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))

    rc = main([
        "run",
        f"{workflow_path}:hello",
        "--input",
        "topic=steady",
        "--timeout",
        "0",
    ])

    captured = capsys.readouterr()
    assert rc == 0
    assert json.loads(captured.out) == {"result": "steady!"}


def test_run_command_calls_setup_hook_with_options(tmp_path, monkeypatch, capsys):
    workflow_path = tmp_path / "setup_workflow.py"
    workflow_path.write_text(SETUP_WORKFLOW_SOURCE)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))

    rc = main([
        "run",
        f"{workflow_path}:setup_hello",
        "--input",
        "topic=deploy",
        "--option",
        "prefix=hook",
        "--option",
        "services=live",
        "--timeout",
        "10",
    ])

    captured = capsys.readouterr()
    assert rc == 0
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


def test_show_rejects_multiple_scope_selectors_in_argparse(tmp_path, capsys):
    workflow_path = tmp_path / "show_workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)

    with pytest.raises(SystemExit) as error:
        main(
            [
                "show",
                f"{workflow_path}:hello",
                "--communications",
                "--agent",
                "User",
            ]
        )

    assert error.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


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
        str(snapshot_path),
        f"{workflow_path}:hello",
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


def test_run_durable_creates_a_managed_durable_run(tmp_path, monkeypatch, capsys):
    workflow_path = tmp_path / "dev_workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    zippergen_home = tmp_path / "zg-home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(zippergen_home))
    monkeypatch.chdir(tmp_path)

    rc = main([
        "run",
        f"{workflow_path}:hello",
        "--durable",
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


def test_no_command_prints_help(capsys):
    """There is no interactive shell to fall into any more."""

    rc = main([])

    captured = capsys.readouterr()
    assert rc == 0
    assert "usage: zippergen" in captured.out
    assert "validate" in captured.out
    assert "Studio" not in captured.out


def test_deploy_prepares_a_profile_that_runs_for_its_project(tmp_path, monkeypatch, capsys):
    workflow_path = tmp_path / "deploy_workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    zippergen_home = tmp_path / "zg-home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(zippergen_home))
    monkeypatch.chdir(tmp_path)

    rc = _deploy_for_test([
        f"{workflow_path}:hello",
        "--set",
        "topic=deploy",
        "--timeout",
        "10",
    ])

    captured = capsys.readouterr()
    profile_path = _the_deployment(zippergen_home)
    script_path = _the_deployment(zippergen_home, ".sh")
    service_path = _the_deployment(zippergen_home, ".service")
    profile = json.loads(profile_path.read_text())
    store_path = Path(profile["store"])
    assert rc == 0
    assert "Status: zippergen deploy status" in captured.out
    assert profile["name"] == Workspace(tmp_path, home=zippergen_home).directory.name
    assert profile["workflow"].endswith("deploy_workflow.py:hello")
    assert profile["store"] == str(store_path)
    assert profile["llm"] == "mock"
    assert profile["llms"] == {}
    assert profile["inputs"] == {"topic": "deploy"}
    assert script_path.exists()
    assert f"ZIPPERGEN_HOME={zippergen_home}" in script_path.read_text()
    assert service_path.exists()
    assert store_path.exists()
    connection = open_store(str(store_path))
    assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
    connection.close()

    deployed = subprocess.run(
        [str(script_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert deployed.returncode == 0, deployed.stderr
    assert store_path.exists()
    assert json.loads(deployed.stdout) == {"result": "deploy!"}

    rc = main(["deploy", "status", "--json"])
    captured = capsys.readouterr()
    status = json.loads(captured.out)
    assert rc == 0
    assert status["store"] == str(store_path)
    assert status["state"] == "done"


def test_two_projects_with_the_same_workflow_get_independent_deployments(
    tmp_path, monkeypatch, capsys
):
    home = tmp_path / "zg-home"
    roots = [tmp_path / "project-a", tmp_path / "project-b"]

    for root in roots:
        root.mkdir()
        (root / "workflow.py").write_text(WORKFLOW_SOURCE)
        workspace = Workspace(root, home=home)
        workspace.initialize_project(name="same-visible-name")
        workspace.select_workflow("workflow.py:hello", cwd=root)
        monkeypatch.chdir(root)
        monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
        assert main([
            "deploy",
            "--set",
            "topic=test",
            "--no-start",
            "--no-bundle",
            "--no-install",
            "--no-setup",
            "--no-doctor",
            "--yes",
        ]) == 0
        capsys.readouterr()

    profiles = [
        json.loads(path.read_text())
        for path in sorted((home / "deployments").glob("*.json"))
        if not path.name.endswith(".secrets.json")
    ]
    assert len(profiles) == 2
    assert {profile["source_cwd"] for profile in profiles} == {
        str(root) for root in roots
    }
    assert len({profile["name"] for profile in profiles}) == 2
    assert len({profile["store"] for profile in profiles}) == 2


def test_start_deployment_dry_run_prints_systemd_commands(tmp_path, monkeypatch, capsys):
    workflow_path = tmp_path / "deploy_workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    zippergen_home = tmp_path / "zg-home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(zippergen_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("ZIPPERGEN_SERVICE_MANAGER", "systemd")
    monkeypatch.chdir(tmp_path)
    _deploy_for_test([
        f"{workflow_path}:hello",
    ])
    capsys.readouterr()
    profile = json.loads(_the_deployment(zippergen_home).read_text())
    name = profile["name"]

    rc = main(["deploy", "start", "--enable", "--dry-run"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "Install systemd unit:" in captured.out
    assert f"zippergen-{name}.service" in captured.out
    assert "systemctl --user daemon-reload" in captured.out
    assert f"systemctl --user enable zippergen-{name}.service" in captured.out
    assert f"systemctl --user start zippergen-{name}.service" in captured.out
    service = (
        zippergen_home
        / "deployments"
        / f"zippergen-{name}.service"
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
    monkeypatch.chdir(tmp_path)
    _deploy_for_test([
        f"{workflow_path}:hello",
    ])
    capsys.readouterr()
    profile = json.loads(_the_deployment(zippergen_home).read_text())
    name = profile["name"]

    rc = main(["deploy", "start", "--dry-run"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "Install launchd agent:" in captured.out
    assert f"io.zippergen.{name}.plist" in captured.out
    assert "launchctl bootout" in captured.out
    assert "launchctl bootstrap" in captured.out
    launchd = plistlib.loads(
        (
            zippergen_home
            / "deployments"
            / f"io.zippergen.{name}.plist"
        ).read_bytes()
    )
    assert launchd["KeepAlive"] == {"SuccessfulExit": False}


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
    monkeypatch.chdir(tmp_path)
    _configure_model_for_test(
        workflow_path,
        zippergen_home,
        name="writer",
        spec="openai:gpt-4o-mini",
    )
    _deploy_for_test(
        [
            f"{workflow_path}:hello",
        ]
    )
    capsys.readouterr()
    monkeypatch.setattr(
        "zippergen.serve._run_systemctl",
        lambda *args, **kwargs: pytest.fail(
            "the service manager must not run after a failed readiness check"
        ),
    )

    rc = main(["deploy", action])

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
    _configure_model_for_test(
        workflow_path,
        zippergen_home,
        name="writer",
        spec="mistral:mistral-small-latest",
        credential=("MISTRAL_API_KEY", "private-key"),
    )

    rc = _deploy_for_test([f"{workflow_path}:hello"])

    assert rc == 0
    capsys.readouterr()
    secrets = json.loads(
        _the_deployment(zippergen_home, ".secrets.json").read_text()
    )
    assert secrets == {"MISTRAL_API_KEY": "private-key"}


def test_guided_deploy_preserves_google_connector_credential_json(
    tmp_path,
    monkeypatch,
    capsys,
):
    from zippergen.google_auth import GOOGLE_SHEETS_SCOPE

    source = Path(__file__).parents[1] / "examples" / "google_sheets_records.py"
    workflow_path = tmp_path / "workflow.py"
    workflow_path.write_text(source.read_text())
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
    workspace = Workspace(tmp_path, home=zippergen_home)
    workspace.initialize_project(name=tmp_path.name)
    workflow_spec = "workflow.py:google_sheet_records"
    workspace.select_workflow(workflow_spec, cwd=tmp_path)
    workspace.save_connector_provider_profile(
        "google", {"granted_scopes": json.dumps([GOOGLE_SHEETS_SCOPE])}
    )
    workspace.save_connector_provider_secret(
        "google", "authorized_user_json", credential
    )
    workspace.save_connector_configuration(
        "records",
        {
            "provider": "google",
            "kind": "google-sheets",
            "spreadsheet_id": "sheet-1",
            "tab": "Calls",
        },
    )
    workspace.bind_connector(workflow_spec, "project-records", "records")

    rc = _deploy_for_test([f"{workflow_path}:google_sheet_records"])

    assert rc == 0
    capsys.readouterr()
    secrets = json.loads(
        _the_deployment(zippergen_home, ".secrets.json").read_text()
    )
    stored = next(value for value in secrets.values() if value == credential)
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
    _configure_model_for_test(
        workflow_path,
        zippergen_home,
        name="writer",
        spec="local:qwen2.5:7b",
        base_url="http://127.0.0.1:11434/v1",
    )

    rc = _deploy_for_test([f"{workflow_path}:hello"])

    assert rc == 0
    capsys.readouterr()
    profile = json.loads(
        (
            _the_deployment(zippergen_home)
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
    _configure_model_for_test(
        workflow_path,
        zippergen_home,
        name="writer",
        spec="mistral:mistral-small-latest",
    )

    previous = Path.cwd()
    try:
        os.chdir(tmp_path)
        rc = main(
            [
                "deploy",
                "--yes",
                "--no-install",
                "--no-setup",
                "--no-start",
            ]
        )
    finally:
        os.chdir(previous)

    captured = capsys.readouterr()
    assert rc == 1
    assert "FAIL model credential MISTRAL_API_KEY" in captured.out
    assert "configured but not started" in captured.out


def test_guided_deploy_persists_config_and_private_secrets(tmp_path, monkeypatch, capsys):
    workflow_path = tmp_path / "guided_workflow.py"
    workflow_path.write_text(GUIDED_WORKFLOW_SOURCE)
    zippergen_home = tmp_path / "zg-home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(zippergen_home))
    workspace = Workspace(tmp_path, home=zippergen_home)
    workspace.initialize_project(name=tmp_path.name)
    workspace.select_workflow("guided_workflow.py:guided", cwd=tmp_path)
    monkeypatch.chdir(tmp_path)

    rc = main([
        "deploy",
        "--set",
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
    profile_path = _the_deployment(zippergen_home)
    secrets_path = _the_deployment(zippergen_home, ".secrets.json")
    profile_text = profile_path.read_text()
    profile = json.loads(profile_text)
    assert rc == 0
    assert f"Deployment: {profile['name']}" in captured.out
    assert profile["options"]["prefix"] == "hello"
    assert profile["environment"] == {"DEMO_MODE": "safe"}
    assert profile["secret_names"] == ["DEMO_TOKEN"]
    assert "ui" not in profile
    assert "show_decisions" not in profile
    assert "top-secret" not in profile_text
    assert json.loads(secrets_path.read_text()) == {"DEMO_TOKEN": "top-secret"}
    assert secrets_path.stat().st_mode & 0o077 == 0
    assert (_the_deployment(zippergen_home, ".plist")).exists()
    assert Path(profile["bundle"]).exists()
    store_path = Path(profile["store"])
    assert store_path.exists()
    connection = open_store(str(store_path))
    assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
    connection.close()

    workflow_path.unlink()
    rc = _run_prepared_deployment(zippergen_home)
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
    workspace = Workspace(tmp_path, home=zippergen_home)
    workspace.initialize_project(name=tmp_path.name)
    workspace.select_workflow("workflow.py:hello", cwd=tmp_path)
    monkeypatch.chdir(tmp_path)
    arguments = [
        "deploy",
        "--set",
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
        (_the_deployment(zippergen_home)).read_text()
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
        (_the_deployment(zippergen_home)).read_text()
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

    assert _run_prepared_deployment(zippergen_home) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"result": "updated?"}


def test_configure_keeps_existing_secret_when_updating_public_field(tmp_path, monkeypatch, capsys):
    workflow_path = tmp_path / "guided_workflow.py"
    workflow_path.write_text(GUIDED_WORKFLOW_SOURCE)
    zippergen_home = tmp_path / "zg-home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(zippergen_home))
    workspace = Workspace(tmp_path, home=zippergen_home)
    workspace.initialize_project(name=tmp_path.name)
    workspace.select_workflow("guided_workflow.py:guided", cwd=tmp_path)
    monkeypatch.chdir(tmp_path)
    main([
        "deploy",
        "--set",
        "topic=deploy",
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
        "deploy",
        "--set",
        "prefix=updated",
        "--yes",
        "--no-start",
        "--no-bundle",
        "--no-install",
        "--no-setup",
        "--no-doctor",
    ])

    capsys.readouterr()
    profile = json.loads((_the_deployment(zippergen_home)).read_text())
    secrets = json.loads((_the_deployment(zippergen_home, ".secrets.json")).read_text())
    assert rc == 0
    assert profile["options"]["prefix"] == "updated"
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
    monkeypatch.chdir(tmp_path)
    _deploy_for_test([
        f"{workflow_path}:hello",
    ])
    capsys.readouterr()
    profile = json.loads((_the_deployment(zippergen_home)).read_text())
    log_path = profile["log"]
    with open(log_path, "w") as f:
        f.write("first\nsecond\nthird\n")

    rc = main(["deploy", "logs", "--tail", "2"])

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.splitlines() == ["second", "third"]


def test_logs_command_shows_only_the_current_log_generation(
    tmp_path,
    monkeypatch,
    capsys,
):
    workflow_path = tmp_path / "deploy_workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    zippergen_home = tmp_path / "zg-home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(zippergen_home))
    monkeypatch.chdir(tmp_path)
    _deploy_for_test([
        f"{workflow_path}:hello",
    ])
    capsys.readouterr()
    profile_path = _the_deployment(zippergen_home)
    profile = json.loads(profile_path.read_text())
    log_path = Path(profile["log"])
    old = b"old failure\n"
    current = b"current start\ncurrent ready\n"
    log_path.write_bytes(old + current)
    profile["log_generation_offset"] = len(old)
    profile_path.write_text(json.dumps(profile))

    rc = main(["deploy", "logs", "--tail", "80"])

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.splitlines() == ["current start", "current ready"]


def test_doctor_reports_deployment_checks(tmp_path, monkeypatch, capsys):
    workflow_path = tmp_path / "deploy_workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    zippergen_home = tmp_path / "zg-home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(zippergen_home))
    monkeypatch.chdir(tmp_path)
    _deploy_for_test([
        f"{workflow_path}:hello",
    ])
    capsys.readouterr()

    rc = main(["deploy", "check", "--json", "--no-systemd"])

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
    assert checks["sqlite integrity"]["status"] == "ok"
    assert checks["sqlite integrity"]["detail"] == "SQLite quick check passed"


def test_doctor_returns_failure_for_broken_profile(tmp_path, monkeypatch, capsys):
    zippergen_home = tmp_path / "zg-home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(zippergen_home))
    deployments = zippergen_home / "deployments"
    deployments.mkdir(parents=True)
    name = Workspace(home=zippergen_home).directory.name
    (deployments / f"{name}.json").write_text(json.dumps({
        "name": name,
        "source_cwd": str(Path.cwd()),
        "workflow": "missing.py:hello",
        "cwd": str(tmp_path / "missing-cwd"),
        "store": str(tmp_path / "runs" / "broken.sqlite"),
        "log": str(tmp_path / "logs" / "broken.log"),
        "python": str(tmp_path / "missing-python"),
    }))

    rc = main(["deploy", "check", "--json", "--no-systemd"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    failures = [check for check in payload["checks"] if check["status"] == "fail"]
    assert rc == 1
    assert any(check["name"] == "working directory" for check in failures)
    assert any(check["name"] == "run script" for check in failures)
def test_status_command_reports_completed_run(tmp_path, monkeypatch, capsys):
    store_path = _prepared_deployment_store(tmp_path, monkeypatch, capsys)
    assert main([
        "deploy",
        "--set",
        "topic=status",
        "--yes",
        "--no-start",
        "--no-bundle",
        "--no-install",
        "--no-setup",
        "--no-doctor",
    ]) == 0
    capsys.readouterr()
    assert _run_prepared_deployment(tmp_path / "zg-home") == 0
    capsys.readouterr()

    rc = main(["deploy", "status", "--json"])

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


def test_status_command_reports_pending_human_task(tmp_path, monkeypatch, capsys):
    store_path = _prepared_deployment_store(tmp_path, monkeypatch, capsys)
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

    rc = main(["deploy", "status"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "State: waiting (waiting for 1 human task(s))" in captured.out
    assert "Pending human tasks: 1" in captured.out
    assert "task-1 User.approve" in captured.out


def test_status_command_reports_missing_store(tmp_path, monkeypatch, capsys):
    store_path = _prepared_deployment_store(tmp_path, monkeypatch, capsys)
    store_path.unlink()

    rc = main(["deploy", "status", "--json"])

    captured = capsys.readouterr()
    assert rc == 0
    assert json.loads(captured.out) == {
        "store": str(store_path),
        "exists": False,
        "state": "missing",
        "summary": "store does not exist",
    }


def test_trace_command_reports_recent_trace_events(tmp_path, monkeypatch, capsys):
    store_path = _prepared_deployment_store(tmp_path, monkeypatch, capsys)
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

    rc = main(["trace", "--deployment", "--tail", "1"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "Trace events: 1" in captured.out
    assert f"#{second} User recv Writer->User main" in captured.out
    assert "draft" in captured.out
    assert f"#{first}" not in captured.out


def test_trace_command_outputs_json_after_rowid(tmp_path, monkeypatch, capsys):
    store_path = _prepared_deployment_store(tmp_path, monkeypatch, capsys)
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

    rc = main(["trace", "--deployment", "--after", str(first), "--json"])

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


def test_tasks_command_lists_pending_tasks(tmp_path, monkeypatch, capsys):
    store_path = _prepared_deployment_store(tmp_path, monkeypatch, capsys)
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

    rc = main(["tasks", "--deployment"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "Pending human tasks: 1" in captured.out
    assert "task-1 User.approve confirm -> approved: bool" in captured.out
    assert "instruction: Approve this?" in captured.out


def test_approve_command_completes_boolean_task(tmp_path, monkeypatch, capsys):
    store_path = _prepared_deployment_store(tmp_path, monkeypatch, capsys)
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

    rc = main(["approve", "--deployment", "--task", "task-1", "--no"])

    captured = capsys.readouterr()
    assert rc == 0
    assert 'Completed human task task-1: {"approved": false}' in captured.out
    conn = open_store(str(store_path))
    try:
        assert load_human_task(conn, "task-1")["result"] == {"approved": False}
    finally:
        conn.close()


def test_approve_command_completes_string_task(tmp_path, monkeypatch, capsys):
    store_path = _prepared_deployment_store(tmp_path, monkeypatch, capsys)
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
        "--deployment",
        "--task",
        "task-1",
        "--value",
        "Looks good.",
        "--json",
    ])

    captured = capsys.readouterr()
    assert rc == 0
    assert json.loads(captured.out)["result"] == {"reply": "Looks good."}


def test_approve_command_requires_value_for_string_task(tmp_path, monkeypatch, capsys):
    store_path = _prepared_deployment_store(tmp_path, monkeypatch, capsys)
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
        main(["approve", "--deployment", "--task", "task-1"])
    except SystemExit as exc:
        assert "requires --value" in str(exc)
    else:
        raise AssertionError("approve should reject string tasks without --value")


def test_tasks_command_generates_stable_channel_tokens(tmp_path, monkeypatch, capsys):
    store_path = _prepared_deployment_store(tmp_path, monkeypatch, capsys)
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

    rc = main(["tasks", "--deployment", "--tokens", "--channel", "email", "--json"])
    captured = capsys.readouterr()
    first = json.loads(captured.out)
    assert rc == 0
    assert first[0]["token"].startswith("zg_")
    assert first[0]["token_channel"] == "email"

    rc = main(["tasks", "--deployment", "--tokens", "--channel", "email", "--json"])
    captured = capsys.readouterr()
    second = json.loads(captured.out)
    assert rc == 0
    assert second[0]["token"] == first[0]["token"]


def test_approve_command_completes_task_by_token(tmp_path, monkeypatch, capsys):
    store_path = _prepared_deployment_store(tmp_path, monkeypatch, capsys)
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

    main(["tasks", "--deployment", "--tokens", "--channel", "telegram", "--json"])
    token = json.loads(capsys.readouterr().out)[0]["token"]

    rc = main(["approve", "--deployment", "--token", token, "--yes"])

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
    assert "zippergen approve --deployment" in captured.out
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


def test_every_command_named_in_output_actually_exists():
    """Messages that suggest a command must suggest a real one.

    Renaming a command silently invalidates every string that mentions it. The
    deployment consolidation left `zippergen logs NAME` in the success message
    for a command that no longer existed, and only a manual run caught it.
    """

    import re
    from pathlib import Path

    from zippergen.serve import _parse_cli_args

    source_root = Path(__file__).resolve().parents[1] / "src" / "zippergen"
    mentioned: dict[str, str] = {}
    for path in source_root.rglob("*.py"):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            for match in re.finditer(r"\b(?:zippergen|zg) ([a-z][a-z-]{2,})", line):
                verb = match.group(1)
                if verb in {"zippergen", "zg"}:
                    continue  # e.g. the completion script's `compdef _zg zg zippergen`
                mentioned.setdefault(verb, f"{path.name}:{number}")

    assert mentioned, "expected the source to suggest at least one command"
    for verb, where in sorted(mentioned.items()):
        try:
            _parse_cli_args([verb, "--help"])
        except SystemExit as exit_code:
            # argparse exits 0 for --help on a real command, 2 for a bad one.
            assert exit_code.code == 0, f"{where} names unknown command {verb!r}"


def test_generated_service_command_matches_the_full_parser():
    """The generated service command must include only accepted arguments."""

    command = shlex.split(
        _deployment_command("private-project-id", python_executable="/python")
    )
    module_index = command.index("zippergen.serve")
    _parser, arguments = _parse_cli_args(command[module_index + 1 :])

    assert arguments.cmd == "__run-deployment"
    assert arguments.profile == "private-project-id"
