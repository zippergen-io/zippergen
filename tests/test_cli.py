import json
import os
import plistlib
import shlex
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from zippergen.deployment_profiles import DEPLOYMENT_PROFILE_SCHEMA_VERSION
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
    record_history,
)
from zippergen.storage_maintenance import inspect_store_storage
from zippergen.workspace import Workspace
from zippergen.value_codec import decode_value, encode_value


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


CONNECTOR_ENV_WORKFLOW_SOURCE = """
import os

from zippergen import Lifeline, pure, workflow

User = Lifeline("User")

@pure
def connector_environment() -> str:
    return os.environ.get("ZIPPERGEN_CONNECTORS_JSON", "missing")

@workflow
def connector_demo() -> str:
    User: value = connector_environment()
    return value @ User
"""


HUMAN_CONNECTOR_WORKFLOW_SOURCE = """
from zippergen import Lifeline, human, workflow

User = Lifeline("User")

@human(
    kind="confirm",
    instruction="Approve this run?",
    outputs=["approved: bool"],
)
def approve() -> None: ...

@workflow
def human_connector_demo() -> bool:
    User: approved = approve()
    return approved @ User
"""


PATH_WORKFLOW_SOURCE = """
from zippergen import DeploymentField, DeploymentSpec, Lifeline, pure, workflow

User = Lifeline("User")

@pure
def identity(value: str) -> str:
    return value

@workflow
def path_demo(directory: str @ User) -> str:
    User: result = identity(directory)
    return result @ User

zippergen_deployment = DeploymentSpec(fields=(
    DeploymentField(
        "directory",
        "External directory",
        target="input",
        required=True,
        path_exists=True,
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
    connection = f"{provider}-test"
    workspace.save_provider_connection(
        connection,
        {"kind": provider, **({"base_url": base_url} if base_url else {})},
    )
    workspace.save_model_configuration(
        name, {"connection": connection, "model": model}
    )
    workspace.save_model_assignment_profile(
        workflow_spec,
        default=name,
        lifelines={},
        actions={},
    )
    if credential:
        workspace.save_provider_secret(connection, "api_key", credential[1])
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


def test_compact_drops_history_and_rotates_logs(tmp_path, monkeypatch, capsys):
    from zippergen import deployments, serve, storage_maintenance

    store = tmp_path / "run.sqlite"
    home = tmp_path / "zg-home"
    (home / "deployments").mkdir(parents=True)
    name = Workspace(home=home).directory.name
    (home / "deployments" / f"{name}.json").write_text(
        json.dumps({"name": name, "source_cwd": str(Path.cwd()),
                    "project_id": Workspace().project_manifest().get("project_id"),
                    "store": str(store)})
    )
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    monkeypatch.setattr(
        serve,
        "_load_deployment_profile",
        lambda _name: {
            "store": str(store),
            "source_cwd": str(Path.cwd()),
            "project_id": Workspace().project_manifest().get("project_id"),
        },
    )
    changed: list[str] = []

    def prune(_path, *, keep=None):
        changed.append("history")
        return SimpleNamespace(
            removed_rows=12,
            before_bytes=4096,
            after_bytes=1024,
        )

    def rotate(_name, _profile, *, keep_archives):
        changed.append("logs")
        return SimpleNamespace(
            removed_archives=2,
            removed_archive_bytes=768,
        )

    monkeypatch.setattr(
        storage_maintenance,
        "prune_store_history",
        prune,
    )
    monkeypatch.setattr(
        deployments,
        "compact_deployment_logs",
        rotate,
    )

    assert main(["deploy", "compact", "--keep-archives", "1"]) == 0

    output = capsys.readouterr().out
    assert "removed history rows: 12" in output
    assert "reclaimed bytes: 3072" in output
    assert "removed archives: 2 (768 bytes)" in output
    assert changed == ["logs", "history"]


def test_history_keep_needs_a_run_that_records_one(tmp_path, monkeypatch):
    """A plain run has no store, so the option would silently do nothing."""

    with pytest.raises(SystemExit, match="requires --durable or --resume"):
        main(["run", "--llm", "mock", "--history-keep", "25"])


def _compact_fixture(tmp_path, monkeypatch, store):
    """Set up the one deployment ``zg deploy compact`` acts on."""

    from zippergen import deployments, serve

    home = tmp_path / "zg-home"
    (home / "deployments").mkdir(parents=True)
    name = Workspace(home=home).directory.name
    profile = {
        "store": str(store),
        "source_cwd": str(Path.cwd()),
        "project_id": Workspace().project_manifest().get("project_id"),
    }
    (home / "deployments" / f"{name}.json").write_text(json.dumps(profile))
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    monkeypatch.setattr(serve, "_load_deployment_profile", lambda _name: dict(profile))
    written: dict[str, object] = {}
    monkeypatch.setattr(
        serve, "_write_deployment_artifacts", lambda p: written.update(p)
    )
    monkeypatch.setattr(
        deployments,
        "compact_deployment_logs",
        lambda _name, _profile, *, keep_archives: SimpleNamespace(
            removed_archives=0, removed_archive_bytes=0
        ),
    )
    return written


def test_bare_compact_trims_to_the_budget_instead_of_emptying_the_store(
    tmp_path,
    monkeypatch,
    capsys,
):
    """A command with no arguments must not throw away the only record of a run.

    ``compact`` used to default to keeping nothing. Trimming to the store's own
    budget is what "tidy this up" means; emptying it is a separate request.
    """

    from zippergen.store import open_store, record_history, write_history_keep

    store = tmp_path / "run.sqlite"
    conn = open_store(str(store))
    try:
        write_history_keep(conn, 25)
        for index in range(400):
            record_history(conn, "A", {"type": "step", "index": index})
    finally:
        conn.close()
    _compact_fixture(tmp_path, monkeypatch, store)

    assert main(["deploy", "compact"]) == 0

    assert inspect_store_storage(str(store)).history_rows == 25
    assert "history budget: 25 of 25 rows kept" in capsys.readouterr().out


def test_setting_the_history_budget_persists_and_applies(
    tmp_path,
    monkeypatch,
    capsys,
):
    from zippergen.store import open_store, read_history_keep, record_history

    store = tmp_path / "run.sqlite"
    conn = open_store(str(store))
    try:
        for index in range(60):
            record_history(conn, "A", {"type": "step", "index": index})
    finally:
        conn.close()
    written = _compact_fixture(tmp_path, monkeypatch, store)

    assert main(["deploy", "compact", "--set-history-keep", "10"]) == 0

    conn = open_store(str(store))
    try:
        assert read_history_keep(conn) == 10
    finally:
        conn.close()
    assert inspect_store_storage(str(store)).history_rows == 10
    # Recorded on the deployment too, so a reset does not lose the choice.
    assert written["history_keep"] == 10
    assert "removed history rows: 50" in capsys.readouterr().out


def test_turning_the_trace_off_is_reported_as_off(tmp_path, monkeypatch, capsys):
    from zippergen.store import open_store

    store = tmp_path / "run.sqlite"
    open_store(str(store)).close()
    _compact_fixture(tmp_path, monkeypatch, store)

    assert main(["deploy", "compact", "--set-history-keep", "0"]) == 0

    assert "history budget: 0 rows (history is off)" in capsys.readouterr().out


def test_compact_refuses_before_changing_a_running_deployment(
    tmp_path, monkeypatch
):
    """The combined maintenance command must not half-complete."""

    from zippergen import (
        deployment_platform,
        deployments,
        serve,
        storage_maintenance,
    )

    store = tmp_path / "run.sqlite"
    home = tmp_path / "zg-home"
    (home / "deployments").mkdir(parents=True)
    name = Workspace(home=home).directory.name
    (home / "deployments" / f"{name}.json").write_text(
        json.dumps({"name": name, "source_cwd": str(Path.cwd()),
                    "project_id": Workspace().project_manifest().get("project_id"),
                    "store": str(store)})
    )
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    monkeypatch.setattr(
        serve,
        "_load_deployment_profile",
        lambda _name: {
            "store": str(store),
            "source_cwd": str(Path.cwd()),
            "project_id": Workspace().project_manifest().get("project_id"),
        },
    )
    changed: list[str] = []
    monkeypatch.setattr(
        deployment_platform,
        "deployment_service_status",
        lambda _name: {
            "state": "running",
            "detail": "service is running",
        },
    )
    monkeypatch.setattr(
        storage_maintenance,
        "prune_store_history",
        lambda _path, *, keep: changed.append("history"),
    )
    monkeypatch.setattr(
        deployments,
        "compact_deployment_logs",
        lambda _name, _profile, *, keep_archives: changed.append("logs"),
    )

    with pytest.raises(SystemExit, match="Stop deployment .* before compacting"):
        main(["deploy", "compact"])

    assert changed == []


def test_reset_names_what_it_discards_before_asking(tmp_path):
    """"Start fresh" hides the only thing worth confirming.

    A reset that is about to throw away an approval somebody already typed must
    say so, in units a person recognises, before the prompt.
    """

    from zippergen import serve
    from zippergen.store import ensure_human_task, open_store, write_role_state

    store = tmp_path / "run.sqlite"
    connection = open_store(str(store))
    try:
        connection.execute("BEGIN IMMEDIATE")
        write_role_state(
            connection,
            "User",
            env={},
            control={"k": "done"},
            monitor=None,
            steps=1,
            status="waiting_human",
        )
        connection.execute(
            "INSERT INTO outstanding_messages(sender,receiver,channel,payload)"
            " VALUES('User','Writer','main','[1]')"
        )
        ensure_human_task(
            connection,
            task_id="task-1",
            role="User",
            locator=[0],
            action="approve",
            input_hash=None,
            inputs={},
            spec={"kind": "confirm", "output": "ok", "output_type": "bool"},
        )
        connection.execute("COMMIT")
    finally:
        connection.close()

    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        serve._print_reset_consequences("demo", {"store": str(store)})
    output = buffer.getvalue()

    assert "participant positions  1" in output
    assert "1 waiting" in output
    assert "Kept:" in output
    # Every category is listed even at zero, so an absent line cannot be
    # mistaken for a category nobody checked.
    assert "messages in flight" in output
    assert "workflow results" in output
    assert "connector progress" in output


def test_reset_lists_every_category_even_at_zero(tmp_path):
    from zippergen import serve
    from zippergen.store import open_store

    store = tmp_path / "run.sqlite"
    open_store(str(store)).close()

    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        serve._print_reset_consequences("demo", {"store": str(store)})

    output = buffer.getvalue()
    assert "participant positions  0" in output
    assert "messages in flight     0" in output
    assert "0 waiting, 0 answered" in output
    assert "left stopped" in output


def test_prune_deletes_stale_archives_but_keeps_the_undo_window(
    tmp_path, monkeypatch, capsys
):
    """Removal archives exist so a mistake can be undone.

    So pruning is decided by age, never by size: today's archive is exactly the
    one somebody may still need.
    """

    import os
    import time

    from zippergen.deployments import list_trash_entries, prune_trash

    home = tmp_path / "zg-home"
    (home / "deployments").mkdir(parents=True)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))

    old = time.time() - 60 * 86400
    for area in ("deployments", "deployment-stores"):
        stale = home / "trash" / area / "demo-20260101-000000"
        stale.mkdir(parents=True)
        (stale / "store.sqlite").write_bytes(b"x" * 5000)
        os.utime(stale, (old, old))
    fresh = home / "trash" / "deployments" / "demo-fresh"
    fresh.mkdir(parents=True)
    (fresh / "store.sqlite").write_bytes(b"y" * 1000)
    log = home / "trash" / "deployment-logs"
    log.mkdir(parents=True)
    rotated = log / "demo-20260101-000000.log"
    rotated.write_bytes(b"z" * 2000)
    os.utime(rotated, (old, old))

    assert len(list_trash_entries()) == 4

    outcome = prune_trash(keep_days=30)

    assert len(outcome.removed) == 3
    assert outcome.removed_bytes == 12_000
    assert [entry.path.name for entry in outcome.kept] == ["demo-fresh"]
    assert fresh.exists(), "an archive inside the undo window must survive"
    assert not rotated.exists()


def test_prune_reports_the_trash_and_is_idempotent(tmp_path, monkeypatch, capsys):
    import os
    import time

    home = tmp_path / "zg-home"
    (home / "deployments").mkdir(parents=True)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    stale = home / "trash" / "deployments" / "demo-20260101-000000"
    stale.mkdir(parents=True)
    (stale / "store.sqlite").write_bytes(b"x" * 4096)
    old = time.time() - 60 * 86400
    os.utime(stale, (old, old))

    assert main(["deploy", "prune", "--yes"]) == 0
    first = capsys.readouterr().out
    assert "Trash: 1 archive(s)" in first
    assert "Deleted 1 archive(s)" in first

    assert main(["deploy", "prune", "--yes"]) == 0
    second = capsys.readouterr().out
    assert "Trash: empty." in second
    assert "Deleted" not in second


def test_prune_keeps_everything_when_the_window_is_wide(tmp_path, monkeypatch, capsys):
    import os
    import time

    home = tmp_path / "zg-home"
    (home / "deployments").mkdir(parents=True)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    archive = home / "trash" / "deployments" / "demo-20260101-000000"
    archive.mkdir(parents=True)
    (archive / "store.sqlite").write_bytes(b"x" * 100)
    os.utime(archive, (time.time() - 60 * 86400, time.time() - 60 * 86400))

    assert main(["deploy", "prune", "--yes", "--keep-days", "365"]) == 0

    assert archive.exists()
    assert "0 older than 365 day(s)" in capsys.readouterr().out


def test_start_does_nothing_when_the_deployment_is_already_running(
    tmp_path, monkeypatch, capsys
):
    """"start" means make sure it is running, not bounce it.

    On launchd the old code ran bootout then bootstrap for start as well as
    restart, so starting a healthy deployment tore it down and back up. That
    interrupts the step in flight, and an interrupted model call or effect can
    run a second time.
    """

    _prepared_deployment_store(tmp_path, monkeypatch, capsys)
    monkeypatch.setattr(
        "zippergen.deployment_platform.deployment_service_status",
        lambda _name: {
            "state": "running",
            "healthy": True,
            "detail": "service is running",
        },
    )
    monkeypatch.setattr(
        "zippergen.serve._run_launchctl",
        lambda *a, **k: pytest.fail("start must not touch a running service"),
    )
    monkeypatch.setattr(
        "zippergen.serve._run_systemctl",
        lambda *a, **k: pytest.fail("start must not touch a running service"),
    )
    capsys.readouterr()

    assert main(["deploy", "start"]) == 0
    assert "is already running" in capsys.readouterr().out


def test_reset_never_starts_the_service(tmp_path, monkeypatch, capsys):
    """Reset is a state operation, so it leaves the running axis alone.

    It has to stop the service to replace the store, and it must not decide on
    your behalf that the service should come back: after a reset the connector
    cursor is gone too, so the next start may re-read a whole mailbox.
    """

    from zippergen import serve

    store = _prepared_deployment_store(tmp_path, monkeypatch, capsys)
    running = {
        "state": "running",
        "healthy": True,
        "detail": "service is running",
    }
    monkeypatch.setattr(
        "zippergen.deployment_platform.deployment_service_status",
        lambda _name: running,
    )
    stopped = {"state": "not-loaded", "healthy": False, "detail": "not loaded"}
    monkeypatch.setattr(
        "zippergen.deployments._deployment_service_status",
        lambda _name: stopped,
    )
    lifecycle: list[str] = []
    monkeypatch.setattr(
        serve,
        "_deployment_lifecycle_command",
        lambda _args, action: lifecycle.append(action) or 0,
    )
    capsys.readouterr()

    assert main(["deploy", "reset", "--yes"]) == 0

    assert lifecycle == ["stop"], "reset must stop, and must not start again"
    output = capsys.readouterr().out
    assert "was running and is now stopped" in output
    assert "zippergen deploy start" in output


def test_remove_names_the_credentials_it_destroys(tmp_path):
    """Listing only what survives hides the irreversible loss.

    Secrets are deleted rather than archived, so removing costs a token
    re-entry and any provider authorization again. That must be on screen
    before the name prompt.
    """

    from dataclasses import dataclass

    from zippergen import serve

    @dataclass
    class Artifact:
        label: str
        retain: bool = False

    artifacts = [
        Artifact("Profile", True),
        Artifact("Private secrets"),
        Artifact("Managed environment"),
        Artifact("Durable store", True),
    ]

    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        serve._print_remove_consequences("demo", artifacts, purge=False)
    output = buffer.getvalue()

    assert "Deleted for good:" in output
    assert "Private secrets" in output
    assert "re-enter tokens" in output
    assert "Archived under" in output
    assert "Durable store" in output

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        serve._print_remove_consequences("demo", artifacts, purge=True)
    purged = buffer.getvalue()

    assert "nothing is kept" in purged
    assert "Archived under" not in purged


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
    Workspace(tmp_path, home=tmp_path / "home").initialize_project()

    rc = main([
        "run",
        "--workflow",
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

    assert main(["run", "trace"]) == 0
    trace_output = capsys.readouterr().out
    assert f"Subject: durable run {record['run_id']}" in trace_output
    assert "Status: done" in trace_output
    assert f"Store: {record['store']}" in trace_output


def test_provider_authorize_google_emits_checked_private_handoff(
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
        "provider",
        "authorize",
        "google-work",
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
    Workspace(tmp_path, home=tmp_path / "home").initialize_project()

    rc = main([
        "run",
        "--workflow",
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
    Workspace(tmp_path, home=tmp_path / "home").initialize_project()

    rc = main([
        "run",
        "--workflow",
        f"{workflow_path}:hello",
        "--input",
        "topic=steady",
        "--timeout",
        "0",
    ])

    captured = capsys.readouterr()
    assert rc == 0
    assert json.loads(captured.out) == {"result": "steady!"}


def test_run_defaults_to_no_deadline():
    _parser, args = _parse_cli_args(["run"])

    assert args.timeout == 0.0


@pytest.mark.parametrize("action", ["status", "reset", "inspect", "trace", "tasks"])
def test_run_subcommands_inherit_project_before_the_action(action):
    _parser, args = _parse_cli_args(["run", "--project", "/tmp/example", action])

    assert args.project == "/tmp/example"


def test_google_authorization_result_is_never_accepted_on_argv():
    with pytest.raises(SystemExit) as exc:
        _parse_cli_args(
            ["provider", "accept", "google-work", "zg-google-v1.secret"]
        )

    assert exc.value.code == 2


@pytest.mark.parametrize("durable", [False, True])
def test_project_run_anchors_relative_effects_to_project_root(
    tmp_path, monkeypatch, capsys, durable
):
    root = tmp_path / "project"
    caller = tmp_path / "caller"
    root.mkdir()
    caller.mkdir()
    (root / "value.txt").write_text("from-project", encoding="utf-8")
    (root / "workflow.py").write_text(
        """
from pathlib import Path
from zippergen import Lifeline, effect, workflow

Worker = Lifeline("Worker")

@effect
def read_value() -> str:
    return Path("value.txt").read_text(encoding="utf-8")

@workflow
def relative_paths() -> str:
    Worker: value = read_value()
    return value @ Worker
""",
        encoding="utf-8",
    )
    home = tmp_path / "home"
    workspace = Workspace(root, home=home)
    workspace.initialize_project(name="relative-paths")
    workspace.select_workflow("workflow.py:relative_paths", cwd=root)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    monkeypatch.chdir(caller)
    arguments = ["run", "--project", str(root), "--llm", "mock"]
    if durable:
        arguments.extend(["--durable", "--yes"])

    assert main(arguments) == 0
    assert "from-project" in capsys.readouterr().out
    assert not (caller / "value.txt").exists()


def test_plain_run_applies_project_connector_routing(
    tmp_path, monkeypatch, capsys
):
    workflow_path = tmp_path / "connector_workflow.py"
    workflow_path.write_text(CONNECTOR_ENV_WORKFLOW_SOURCE)
    workspace = Workspace(tmp_path, home=tmp_path / "home")
    workspace.initialize_project(name="connector-run")
    workspace.select_workflow("connector_workflow.py:connector_demo")
    snapshot = {
        "requirement:mailbox": {
            "type": "service",
            "configuration": "inbox",
            "provider": "google",
        }
    }
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        "zippergen.connector_wiring.connector_runtime",
        lambda *_args, **_kwargs: (snapshot, {}),
    )

    assert main(["run", "--yes"]) == 0

    result = json.loads(capsys.readouterr().out)["result"]
    assert json.loads(result) == snapshot
    assert "ZIPPERGEN_CONNECTORS_JSON" not in os.environ


def test_plain_run_uses_configured_external_human_connector(
    tmp_path, monkeypatch, capsys
):
    from zippergen.store import complete_human_task

    workflow_path = tmp_path / "human_connector_workflow.py"
    workflow_path.write_text(HUMAN_CONNECTOR_WORKFLOW_SOURCE)
    workspace = Workspace(tmp_path, home=tmp_path / "home")
    workspace.initialize_project(name="human-connector-run")
    workspace.select_workflow(
        "human_connector_workflow.py:human_connector_demo"
    )
    snapshot = {
        "human:User.approve": {
            "type": "human",
            "target": "User.approve",
            "configuration": "approval-chat",
            "provider": "telegram",
        }
    }

    class CompletingConnector:
        def __init__(self, store_path):
            self.store_path = store_path

        def run_forever(self, *, poll_timeout, stop_event):
            while not stop_event.is_set():
                conn = open_store(self.store_path)
                try:
                    row = conn.execute(
                        "SELECT task_id FROM human_tasks "
                        "WHERE status='pending' LIMIT 1"
                    ).fetchone()
                    if row is not None:
                        complete_human_task(
                            conn, row[0], {"approved": True}
                        )
                        return
                finally:
                    conn.close()
                time.sleep(0.01)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        "zippergen.connector_wiring.connector_runtime",
        lambda *_args, **_kwargs: (snapshot, {}),
    )
    monkeypatch.setattr(
        "zippergen.connector_wiring.human_connector_factory",
        lambda _snapshot, _environment: (
            lambda store_path: CompletingConnector(store_path)
        ),
    )

    assert main(["run", "--yes", "--timeout", "3"]) == 0

    output = capsys.readouterr().out
    assert "External human connector started for this run." in output
    assert json.loads(output.splitlines()[-1]) == {"result": True}


def test_run_command_calls_setup_hook_with_options(tmp_path, monkeypatch, capsys):
    workflow_path = tmp_path / "setup_workflow.py"
    workflow_path.write_text(SETUP_WORKFLOW_SOURCE)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))
    Workspace(tmp_path, home=tmp_path / "home").initialize_project()

    rc = main([
        "run",
        "--workflow",
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
    Workspace(tmp_path, home=zippergen_home).initialize_project()

    rc = main([
        "run",
        "--workflow",
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
    assert decode_value(profile["inputs"]) == {"topic": "deploy"}
    assert script_path.exists()
    assert f"ZIPPERGEN_HOME={zippergen_home}" in script_path.read_text()
    assert service_path.exists()
    assert store_path.exists()
    connection = open_store(str(store_path))
    assert connection.execute("SELECT COUNT(*) FROM role_state").fetchone()[0] == 0
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


def test_deploy_records_external_paths_absolutely(
    tmp_path, monkeypatch, capsys
):
    workflow_path = tmp_path / "path_workflow.py"
    workflow_path.write_text(PATH_WORKFLOW_SOURCE)
    mailbox = tmp_path / "mailbox"
    mailbox.mkdir()
    home = tmp_path / "zg-home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    monkeypatch.chdir(tmp_path)

    assert _deploy_for_test([
        f"{workflow_path}:path_demo",
        "--set",
        "directory=mailbox",
    ]) == 0
    capsys.readouterr()

    profile = json.loads(_the_deployment(home).read_text())
    assert decode_value(profile["inputs"])["directory"] == str(mailbox.resolve())


def test_internal_deployment_run_does_not_conflict_with_its_own_service(
    tmp_path, monkeypatch, capsys
):
    workflow_path = tmp_path / "workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    home = tmp_path / "zg-home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    assert _deploy_for_test([f"{workflow_path}:hello"]) == 0
    capsys.readouterr()

    assert _run_prepared_deployment(home) == 0
    assert json.loads(capsys.readouterr().out) == {"result": "deploy!"}


def test_foreground_run_refuses_to_compete_with_running_deployment(
    tmp_path, monkeypatch,
):
    from zippergen.execution_lock import execution_lock, execution_lock_path

    home = tmp_path / "home"
    workspace = Workspace(tmp_path, home=home)
    workspace.initialize_project()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))

    lock_path = execution_lock_path(home, workspace.directory.name)
    with execution_lock(lock_path, owner="project deployment"):
        with pytest.raises(
            SystemExit,
            match="active project deployment.*Only one run or deployment",
        ):
            main(["run", "--yes"])


def test_deployment_start_refuses_to_compete_with_foreground_run(
    tmp_path, monkeypatch, capsys
):
    from zippergen.execution_lock import execution_lock, execution_lock_path

    workflow_path = tmp_path / "workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    home = tmp_path / "zg-home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    assert _deploy_for_test([f"{workflow_path}:hello"]) == 0
    capsys.readouterr()
    profile = json.loads(_the_deployment(home).read_text())

    lock_path = execution_lock_path(home, str(profile["name"]))
    with execution_lock(lock_path, owner="foreground run"):
        with pytest.raises(
            SystemExit,
            match="active foreground run.*Only one run or deployment",
        ):
            main(["deploy", "start"])


def test_redeploy_requires_the_existing_deployment_to_be_stopped(
    tmp_path, monkeypatch, capsys
):
    from zippergen.execution_lock import execution_lock, execution_lock_path

    workflow_path = tmp_path / "workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    home = tmp_path / "zg-home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    assert _deploy_for_test([f"{workflow_path}:hello"]) == 0
    capsys.readouterr()
    profile = json.loads(_the_deployment(home).read_text())

    lock_path = execution_lock_path(home, str(profile["name"]))
    with execution_lock(lock_path, owner="project deployment"):
        with pytest.raises(
            SystemExit,
            match="already running.*zg deploy stop.*before updating",
        ):
            main(["deploy", "--yes"])


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


def test_deploy_list_and_prune_find_a_deleted_projects_deployment(
    tmp_path, monkeypatch, capsys
):
    from zippergen import serve

    home = tmp_path / "home"
    deployments = home / "deployments"
    deployments.mkdir(parents=True)
    (deployments / "orphan.json").write_text(json.dumps({
        "name": "orphan",
        "source_cwd": str(tmp_path / "deleted-project"),
        "project_id": "old-project",
    }))
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    monkeypatch.setattr(
        "zippergen.deployment_platform.deployment_service_status",
        lambda _name: {"state": "running"},
    )
    removed = []
    monkeypatch.setattr(
        serve,
        "_remove_command",
        lambda args: removed.append((args.name, args.purge, args.yes)) or 0,
    )

    assert main(["deploy", "list"]) == 0
    output = capsys.readouterr().out
    assert "orphan" in output
    assert "project directory is missing" in output
    assert main(["deploy", "prune", "--yes"]) == 0
    assert removed == [("orphan", False, True)]


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


def test_start_refuses_a_deployment_that_fails_readiness(
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

    rc = main(["deploy", "start"])

    captured = capsys.readouterr()
    assert rc == 1
    assert "1 failure(s)" in captured.out
    assert "provider credential openai-test" in captured.out
    assert "was not started because readiness checks found failures" in (
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
    assert secrets == {
        "ZIPPERGEN_PROVIDER_MISTRAL_DASH_TEST_API_KEY": "private-key"
    }


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
    workspace.save_provider_connection(
        "google-work",
        {
            "kind": "google",
            "granted_scopes": json.dumps([GOOGLE_SHEETS_SCOPE]),
        },
    )
    workspace.save_provider_secret(
        "google-work", "authorized_user_json", credential
    )
    workspace.save_connector_configuration(
        "records",
        {
            "connection": "google-work",
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
        "ZIPPERGEN_PROVIDER_LOCAL_DASH_TEST_BASE_URL": "http://127.0.0.1:11434/v1"
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
    assert "FAIL provider credential mistral-test" in captured.out
    assert "was not started" in captured.out
    assert "found problems" in captured.out


def test_guided_deploy_persists_config_and_private_secrets(tmp_path, monkeypatch, capsys):
    workflow_path = tmp_path / "guided_workflow.py"
    workflow_path.write_text(GUIDED_WORKFLOW_SOURCE)
    zippergen_home = tmp_path / "zg-home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(zippergen_home))
    monkeypatch.setenv("DEMO_TOKEN", "top-secret")
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
    assert connection.execute("SELECT COUNT(*) FROM role_state").fetchone()[0] == 0
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
    monkeypatch.setenv("DEMO_TOKEN", "top-secret")
    workspace = Workspace(tmp_path, home=zippergen_home)
    workspace.initialize_project(name=tmp_path.name)
    workspace.select_workflow("guided_workflow.py:guided", cwd=tmp_path)
    monkeypatch.chdir(tmp_path)
    main([
        "deploy",
        "--set",
        "topic=deploy",
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


def test_deploy_does_not_warn_about_what_it_is_about_to_do(
    tmp_path, monkeypatch, capsys
):
    """`zg deploy` installs the unit and starts the service.

    Warning that neither exists, one second before creating both, reports a
    problem the same command is fixing. `zg deploy check` is different: there
    nothing is about to happen, so the same facts are real news.
    """

    from zippergen.deployment_checks import _doctor_checks

    workflow_path = tmp_path / "deploy_workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "zg-home"))
    monkeypatch.chdir(tmp_path)
    _deploy_for_test([f"{workflow_path}:hello"])
    capsys.readouterr()
    main(["deploy", "check", "--json", "--no-systemd"])
    name = json.loads(capsys.readouterr().out)["deployment"]

    standalone = {
        check["name"]: check
        for check in _doctor_checks(name, include_systemd=False)
    }
    during_deploy = {
        check["name"]: check
        for check in _doctor_checks(
            name, include_systemd=False, before_start=True
        )
    }

    assert standalone["log file"]["status"] == "warn"
    assert "log file" not in during_deploy
    installed = [key for key in standalone if key.endswith(" installed")]
    assert installed, "the standalone check must still report the unit"
    assert not [key for key in during_deploy if key.endswith(" installed")]


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
        "schema_version": DEPLOYMENT_PROFILE_SCHEMA_VERSION,
        "name": name,
        "source_cwd": str(Path.cwd()),
        "project_id": Workspace().project_manifest().get("project_id"),
        "workflow": "missing.py:hello",
        "cwd": str(tmp_path / "missing-cwd"),
        "store": str(tmp_path / "runs" / "broken.sqlite"),
        "log": str(tmp_path / "logs" / "broken.log"),
        "python": str(tmp_path / "missing-python"),
        "inputs": encode_value({}),
    }))

    rc = main(["deploy", "check", "--strict", "--json", "--no-systemd"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    failures = [check for check in payload["checks"] if check["status"] == "fail"]
    assert rc == 1
    assert any(check["name"] == "working directory" for check in failures)
    assert any(check["name"] == "run script" for check in failures)

    # Without --strict the same broken profile still reports, but reporting is
    # not itself a failure, so an interactive shell sees a clean exit.
    assert main(["deploy", "check", "--json", "--no-systemd"]) == 0
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
    assert [role["status"] for role in status["roles"]] == ["done"]
    assert status["outstanding_messages"] == []
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
        spec={"kind": "confirm", "output": "approved", "output_type": "bool"},
    )
    conn.close()

    rc = main(["deploy", "status"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "Durable state: waiting (waiting for 1 human task(s))" in captured.out
    assert "Pending human tasks: 1" in captured.out
    assert "task-1 User.approve" in captured.out


def test_status_says_whether_the_deployment_is_running(
    tmp_path, monkeypatch, capsys
):
    """"Is it running?" is the first thing status is asked.

    It used to report only the durable store, so the obvious question had to be
    answered by 'deploy check' or 'deploy list' instead.
    """

    _prepared_deployment_store(tmp_path, monkeypatch, capsys)
    monkeypatch.setattr(
        "zippergen.deployment_platform.deployment_service_status",
        lambda _name: {
            "state": "running",
            "healthy": True,
            "detail": "io.zippergen.demo is running",
        },
    )
    capsys.readouterr()

    assert main(["deploy", "status"]) == 0

    output = capsys.readouterr().out
    assert "Service: running" in output
    assert "Deployment:" in output
    # Two lines both called "State" would be the confusion, not the fix.
    assert "Durable state:" in output
    assert "\nState:" not in output


def test_status_speaks_plainly_about_a_stopped_service(
    tmp_path, monkeypatch, capsys
):
    """launchd says "not-loaded" and systemd says other things; users say stopped."""

    _prepared_deployment_store(tmp_path, monkeypatch, capsys)
    monkeypatch.setattr(
        "zippergen.deployment_platform.deployment_service_status",
        lambda _name: {
            "state": "not-loaded",
            "healthy": False,
            "detail": "not loaded",
        },
    )
    capsys.readouterr()

    assert main(["deploy", "status"]) == 0

    output = capsys.readouterr().out
    assert "Service: stopped" in output
    # A detail that only restates the state is noise.
    assert "not-loaded" not in output


def test_status_command_reports_missing_store(tmp_path, monkeypatch, capsys):
    store_path = _prepared_deployment_store(tmp_path, monkeypatch, capsys)
    store_path.unlink()

    rc = main(["deploy", "status", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 0
    assert payload["store"] == str(store_path)
    assert payload["exists"] is False
    assert payload["state"] == "missing"
    assert payload["summary"] == "store does not exist"
    # Status answers "is it running?" too, so the service is always reported.
    assert payload["service"]["state"]
    assert payload["deployment"]


def test_trace_command_reports_recent_trace_events(tmp_path, monkeypatch, capsys):
    store_path = _prepared_deployment_store(tmp_path, monkeypatch, capsys)
    monkeypatch.setattr("zippergen.store.time.time", lambda: 1_700_000_000.125)
    conn = open_store(str(store_path))
    first = record_history(
        conn,
        "Writer",
        {"type": "send", "from": "Writer", "to": "User", "channel": "main", "values": ["old"]},
    )
    second = record_history(
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

    rc = main(["deploy", "trace", "--tail", "1"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "Subject: project deployment" in captured.out
    assert f"Store: {store_path}" in captured.out
    assert "Trace (1 event)" in captured.out
    assert "Time" in captured.out
    assert "Participant" in captured.out
    assert "2023-11-" in captured.out
    assert f"#{second}" in captured.out
    assert "Writer → User [main]" in captured.out
    assert 'draft="Looks good."' in captured.out
    assert f"#{first}" not in captured.out


def test_trace_command_outputs_json_after_rowid(tmp_path, monkeypatch, capsys):
    store_path = _prepared_deployment_store(tmp_path, monkeypatch, capsys)
    monkeypatch.setattr("zippergen.store.time.time", lambda: 1_700_000_000.125)
    conn = open_store(str(store_path))
    first = record_history(
        conn,
        "Writer",
        {"type": "act_start", "action": "draft", "action_kind": "llm", "inputs": {"topic": "x"}},
    )
    second = record_history(
        conn,
        "Writer",
        {"type": "act", "action": "draft", "action_kind": "llm", "outputs": {"reply": "hello"}},
    )
    conn.close()

    rc = main(["deploy", "trace", "--after", str(first), "--json"])

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
                "recorded_at": 1_700_000_000.125,
            },
        }
    ]


def test_trace_command_renders_control_messages_without_internal_tags(
    tmp_path, monkeypatch, capsys
):
    store_path = _prepared_deployment_store(tmp_path, monkeypatch, capsys)
    conn = open_store(str(store_path))
    event_id = record_history(
        conn,
        "Mailbox",
        {
            "type": "send",
            "from": "Mailbox",
            "to": "Extractor",
            "channel": "main",
            "values": [True, "κ_ctrl_internal"],
        },
    )
    conn.close()

    assert main(["deploy", "trace", "--tail", "1"]) == 0

    output = capsys.readouterr().out
    assert f"#{event_id}" in output
    assert "control send" in output
    assert "Mailbox → Extractor [main]" in output
    assert "value=true" in output
    assert "κ_ctrl_internal" not in output


def test_trace_command_marks_legacy_events_without_a_timestamp(
    tmp_path, monkeypatch, capsys
):
    store_path = _prepared_deployment_store(tmp_path, monkeypatch, capsys)
    conn = open_store(str(store_path))
    cursor = conn.execute(
        "INSERT INTO history(role,payload) VALUES(?,?)",
        ("Writer", json.dumps({"type": "decision", "kind": "while", "value": True})),
    )
    event_id = int(cursor.lastrowid)
    conn.close()

    assert main(["deploy", "trace", "--tail", "1"]) == 0

    output = capsys.readouterr().out
    assert f"#{event_id}" in output
    assert "—" in output
    assert "while → continue" in output


def test_trace_command_shows_elapsed_action_time(tmp_path, monkeypatch, capsys):
    store_path = _prepared_deployment_store(tmp_path, monkeypatch, capsys)
    recorded_times = iter((1_700_000_000.000, 1_700_000_000.013))
    monkeypatch.setattr("zippergen.store.time.time", lambda: next(recorded_times))
    conn = open_store(str(store_path))
    record_history(
        conn,
        "Writer",
        {
            "type": "act_start",
            "action": "draft_reply",
            "action_kind": "llm",
            "inputs": {"message": "Hello"},
            "seq": 7,
        },
    )
    record_history(
        conn,
        "Writer",
        {
            "type": "act",
            "action": "draft_reply",
            "action_kind": "llm",
            "outputs": {"draft": "Hi"},
            "seq": 7,
        },
    )
    conn.close()

    assert main(["deploy", "trace", "--tail", "2"]) == 0

    output = capsys.readouterr().out
    assert "llm start" in output
    assert "llm done" in output
    assert "13ms" in output


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

    rc = main(["deploy", "tasks"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "Subject: project deployment" in captured.out
    assert f"Store: {store_path}" in captured.out
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

    rc = main(["deploy", "approve", "--task", "task-1", "--no"])

    captured = capsys.readouterr()
    assert rc == 0
    assert 'Completed human task task-1: {"approved": false}' in captured.out
    conn = open_store(str(store_path))
    try:
        assert load_human_task(conn, "task-1")["result"] == {"approved": False}
    finally:
        conn.close()


def test_approve_command_rejects_declining_acknowledgement(
    tmp_path, monkeypatch, capsys
):
    store_path = _prepared_deployment_store(tmp_path, monkeypatch, capsys)
    conn = open_store(str(store_path))
    ensure_human_task(
        conn,
        task_id="task-1",
        role="User",
        locator=[0],
        action="acknowledge",
        input_hash=None,
        inputs={},
        spec={"kind": "ack", "output": "seen", "output_type": "bool"},
    )
    conn.close()

    with pytest.raises(SystemExit, match="only be completed affirmatively"):
        main(["deploy", "approve", "--task", "task-1", "--no"])


def test_approve_command_rejects_value_outside_select_options(
    tmp_path, monkeypatch, capsys
):
    store_path = _prepared_deployment_store(tmp_path, monkeypatch, capsys)
    conn = open_store(str(store_path))
    ensure_human_task(
        conn,
        task_id="task-1",
        role="User",
        locator=[0],
        action="choose",
        input_hash=None,
        inputs={},
        spec={
            "kind": "select",
            "output": "choice",
            "output_type": "str",
            "rendered": {"prefill": "A\nB"},
        },
    )
    conn.close()

    with pytest.raises(SystemExit, match="Choose a number between 1 and 2"):
        main([
            "deploy", "approve", "--task", "task-1", "--value", "C"
        ])


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
        "deploy",
        "approve",
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
        main(["deploy", "approve", "--task", "task-1"])
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

    rc = main(["deploy", "tasks", "--tokens", "--channel", "email", "--json"])
    captured = capsys.readouterr()
    first = json.loads(captured.out)
    assert rc == 0
    assert first[0]["token"].startswith("zg_")
    assert first[0]["token_channel"] == "email"

    rc = main(["deploy", "tasks", "--tokens", "--channel", "email", "--json"])
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

    main(["deploy", "tasks", "--tokens", "--channel", "telegram", "--json"])
    token = json.loads(capsys.readouterr().out)[0]["token"]

    rc = main(["deploy", "approve", "--token", token, "--yes"])

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
    assert "zippergen deploy approve" in captured.out
    assert "--token zg_" in captured.out
    assert "--no" in captured.out


def test_notify_stdout_reports_no_pending_tasks(tmp_path, capsys):
    store_path = tmp_path / "notify-empty.sqlite"
    open_store(str(store_path)).close()

    rc = main(["notify", "stdout", "--store", str(store_path)])

    captured = capsys.readouterr()
    assert rc == 0
    assert "No pending human tasks." in captured.out


@pytest.mark.parametrize(
    ("kind", "output", "output_type", "expected", "absent"),
    [
        ("ack", "seen", "bool", "Acknowledge:", "Decline:"),
        ("input", "answer", "str", "--value '<value>'", "Approve:"),
    ],
)
def test_notify_stdout_commands_match_the_human_task_kind(
    tmp_path, capsys, kind, output, output_type, expected, absent
):
    store_path = tmp_path / f"notify-{kind}.sqlite"
    conn = open_store(str(store_path))
    ensure_human_task(
        conn,
        task_id="task-1",
        role="User",
        locator=[0],
        action=kind,
        input_hash=None,
        inputs={},
        spec={"kind": kind, "output": output, "output_type": output_type},
    )
    conn.close()

    assert main(["notify", "stdout", "--store", str(store_path)]) == 0

    output_text = capsys.readouterr().out
    assert expected in output_text
    assert absent not in output_text


def test_deployment_starts_one_telegram_bridge_for_shared_routes(
    tmp_path,
    monkeypatch,
):
    observed = []
    monkeypatch.setenv("ZIPPERGEN_CONNECTOR_TELEGRAM_TOKEN", "secret")
    monkeypatch.setattr(
        "zippergen.telegram_notify.TelegramDeploymentNotifier.run_forever",
        lambda notifier, **_kwargs: observed.append(
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
                "connection": "telegram-main",
                "token_env": "ZIPPERGEN_CONNECTOR_TELEGRAM_TOKEN",
            },
            "human:Reviewer": {
                "type": "human",
                "target": "Reviewer",
                "kind": "telegram",
                "configuration": "team-chat",
                "chat_id": "123",
                "channel": "telegram:team-chat",
                "connection": "telegram-main",
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
