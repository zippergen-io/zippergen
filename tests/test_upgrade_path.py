"""Exercise every persisted-format gate with non-current state.

The first release has no earlier released format to migrate. Older internal
prerelease records are refused with replacement instructions; records from a
newer ZipperGen are refused without being rewritten. A future format change
belongs here together with a fixture written in the then-previous release.
"""

import json
import sqlite3

import pytest

from zippergen.deployment_profiles import (
    DEPLOYMENT_PROFILE_SCHEMA_VERSION,
    _load_deployment_profile,
)
from zippergen.store import SCHEMA_VERSION, StoreSchemaError, open_store
from zippergen.value_codec import encode_value
from zippergen.workspace import (
    PROJECT_SCHEMA_VERSION,
    RUN_SCHEMA_VERSION,
    WORKSPACE_SCHEMA_VERSION,
    Workspace,
    WorkspaceError,
)


# ---------------------------------------------------------------------------
# Configuration and metadata: strict gates
# ---------------------------------------------------------------------------


def test_a_previous_deployment_profile_is_refused(tmp_path, monkeypatch):

    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))
    directory = tmp_path / "home" / "deployments"
    directory.mkdir(parents=True)
    (directory / "old.json").write_text(json.dumps({
        "schema_version": DEPLOYMENT_PROFILE_SCHEMA_VERSION - 1,
        "name": "old",
        "inputs": {"number": 3},
        "options": {"send_mode": "send"},
    }))

    with pytest.raises(SystemExit, match="No migration is available"):
        _load_deployment_profile("old")


def test_an_unreadably_old_configuration_says_what_to_do(tmp_path, monkeypatch):
    """Refusing is allowed. Refusing without an instruction is not."""

    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))
    directory = tmp_path / "home" / "deployments"
    directory.mkdir(parents=True)
    (directory / "ancient.json").write_text(
        json.dumps({"schema_version": 1, "name": "ancient", "inputs": {}})
    )

    with pytest.raises(SystemExit) as caught:
        _load_deployment_profile("ancient")

    message = str(caught.value)
    assert "No migration is available" in message
    assert "zippergen deploy" in message


def test_a_newer_configuration_says_to_upgrade(tmp_path, monkeypatch):
    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))
    directory = tmp_path / "home" / "deployments"
    directory.mkdir(parents=True)
    (directory / "future.json").write_text(json.dumps({
        "schema_version": DEPLOYMENT_PROFILE_SCHEMA_VERSION + 1,
        "name": "future",
        "inputs": encode_value({}),
    }))

    with pytest.raises(SystemExit, match="newer ZipperGen"):
        _load_deployment_profile("future")


@pytest.mark.parametrize(
    ("record_kind", "version", "message"),
    [
        ("workspace", WORKSPACE_SCHEMA_VERSION - 1, "No migration is available"),
        ("workspace", WORKSPACE_SCHEMA_VERSION + 1, "newer ZipperGen"),
        ("run", RUN_SCHEMA_VERSION - 1, "No migration is available"),
        ("run", RUN_SCHEMA_VERSION + 1, "newer ZipperGen"),
    ],
)
def test_workspace_and_run_record_mismatches_are_refused_unchanged(
    tmp_path, record_kind, version, message
):
    root = tmp_path / "project"
    root.mkdir()
    workspace = Workspace(root, home=tmp_path / "home")
    workspace.initialize_project(name="project")
    if record_kind == "workspace":
        path = workspace.state_path
        record = {
            "schema_version": version,
            "project_root": str(root),
        }
        load = workspace.load
    else:
        path = workspace.run_path("record")
        record = {
            "schema_version": version,
            "run_id": "record",
            "inputs": encode_value({}),
        }
        load = lambda: workspace.load_run("record")
    path.parent.mkdir(parents=True, exist_ok=True)
    original = json.dumps(record)
    path.write_text(original)

    with pytest.raises(WorkspaceError, match=message):
        load()

    assert path.read_text() == original


def test_an_unstamped_project_manifest_is_the_first_layout(tmp_path, monkeypatch):
    """Absence identifies the initial format; a stated mismatch is refused."""

    monkeypatch.chdir(tmp_path)
    (tmp_path / "zippergen.toml").write_text(
        "name = 'p'\nspecification_file = 'spec.md'\n"
    )

    manifest = Workspace(tmp_path).project_manifest()

    assert manifest["name"] == "p"
    assert manifest["specification_file"] == "spec.md"


def test_a_project_manifest_with_a_different_old_schema_is_refused(tmp_path):
    (tmp_path / "zippergen.toml").write_text(
        f"schema_version = {PROJECT_SCHEMA_VERSION - 1}\n"
        "name = 'old'\nspecification_file = 'spec.md'\n"
    )

    with pytest.raises(WorkspaceError, match="No migration is available"):
        Workspace(tmp_path).project_manifest()


def test_a_project_manifest_written_by_a_newer_zippergen_is_refused(tmp_path):
    path = tmp_path / "zippergen.toml"
    original = (
        f"schema_version = {PROJECT_SCHEMA_VERSION + 1}\n"
        "name = 'future'\n"
        "specification_file = 'spec.md'\n"
        "future_root = 'keep me'\n"
        "\n[future]\nmode = 'keep me too'\n"
    )
    path.write_text(original)

    with pytest.raises(WorkspaceError, match="newer ZipperGen"):
        Workspace(tmp_path).write_configuration_values({"answer": 42})

    assert path.read_text() == original, "refusing must leave future data untouched"


def test_a_manifest_project_id_is_not_a_supported_identity(tmp_path):
    home = tmp_path / "home"
    root = tmp_path / "project"
    root.mkdir()
    manifest = root / "zippergen.toml"
    current = (
        f"schema_version = {PROJECT_SCHEMA_VERSION}\n"
        "name = 'project'\nspecification_file = 'spec.md'\n"
    )
    manifest.write_text(current)
    path_keyed_directory = Workspace(root, home=home).directory
    manifest.write_text(
        current + "project_id = 'a1b2c3d4e5f60718293a4b5c6d7e8f90'\n"
    )

    workspace = Workspace(root, home=home)

    assert workspace.directory == path_keyed_directory
    workspace.require_project()
    assert workspace.project_manifest()["project_id"] is None
    assert not workspace.project_id_path.exists()


def test_a_project_that_never_had_an_identity_is_not_given_one(tmp_path):
    home = tmp_path / "home"
    root = tmp_path / "ancient"
    root.mkdir()
    (root / "zippergen.toml").write_text(
        "name = 'ancient'\nspecification_file = 'spec.md'\n"
    )
    workspace = Workspace(root, home=home)
    before = workspace.directory

    workspace.require_project()

    assert not workspace.project_id_path.exists()
    assert workspace.directory == before


# ---------------------------------------------------------------------------
# Durable store: refused, and it says which command replaces it
# ---------------------------------------------------------------------------


def test_a_previous_store_is_refused_and_names_the_command(tmp_path):
    """Control positions from an incompatible store cannot be interpreted."""

    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE store_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute(
        "INSERT INTO store_meta VALUES('schema_version',?)",
        (str(SCHEMA_VERSION - 1),),
    )
    conn.commit()
    conn.close()

    with pytest.raises(StoreSchemaError) as caught:
        open_store(str(path))

    message = str(caught.value)
    assert "not migrated" in message
    # Refusing is only acceptable because there is a way through.
    assert "reset" in message


def test_a_store_from_a_newer_zippergen_is_refused_and_names_reset(tmp_path):
    path = tmp_path / "future.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE store_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute(
        "INSERT INTO store_meta VALUES('schema_version',?)",
        (str(SCHEMA_VERSION + 1),),
    )
    conn.commit()
    conn.close()

    with pytest.raises(StoreSchemaError) as caught:
        open_store(str(path))

    message = str(caught.value)
    assert f"schema {SCHEMA_VERSION + 1}" in message
    # A newer store is upgraded to, not reset: resetting would destroy state a
    # newer ZipperGen wrote. "Reset" remains the instruction for a store this
    # version cannot read for any other reason.
    assert "upgrade this one" in message
    assert "not modified" in message


def test_a_current_store_still_opens(tmp_path):
    path = tmp_path / "now.sqlite"
    open_store(str(path)).close()
    open_store(str(path)).close()


def test_a_future_store_is_refused_without_being_touched(tmp_path):
    """Looking at a store must not change it.

    Switching to WAL is a persistent file property, so an installation that
    merely opened a store written by a newer ZipperGen left it permanently
    altered. An installation cannot acquire that restraint later; only the
    version that will one day be the old one can ship with it.
    """

    import hashlib
    import sqlite3

    from zippergen.store import SCHEMA_VERSION, StoreSchemaError, open_store

    store = tmp_path / "future.sqlite"
    open_store(str(store)).close()
    connection = sqlite3.connect(store)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute(
        "UPDATE store_meta SET value=? WHERE key='schema_version'",
        (str(SCHEMA_VERSION + 1),),
    )
    connection.commit()
    connection.close()
    for suffix in ("-wal", "-shm"):
        (tmp_path / f"future.sqlite{suffix}").unlink(missing_ok=True)

    before = hashlib.sha256(store.read_bytes()).hexdigest()

    with pytest.raises(StoreSchemaError) as caught:
        open_store(str(store))

    assert "newer ZipperGen" in str(caught.value)
    assert hashlib.sha256(store.read_bytes()).hexdigest() == before
    assert (
        sqlite3.connect(store).execute("PRAGMA journal_mode").fetchone()[0]
        == "delete"
    ), "the journal mode is a persistent property and must be left alone"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["future.sqlite"]


@pytest.mark.parametrize("version", [2.0, "2", True, None])
def test_a_schema_version_that_is_not_a_whole_number_is_refused(
    tmp_path, version
):
    """`3.0 == 3` in Python, so the type is decided before the value."""

    import json

    from zippergen.workspace import Workspace, WorkspaceError

    root = tmp_path / "project"
    root.mkdir()
    workspace = Workspace(root, home=tmp_path / "home")
    workspace.initialize_project(name="typed")
    workspace.update(anything=1)
    state = json.loads(workspace.state_path.read_text())
    state["schema_version"] = version
    workspace.state_path.write_text(json.dumps(state))

    with pytest.raises(WorkspaceError):
        Workspace(root, home=tmp_path / "home").load()


def test_a_deployment_profile_version_must_be_a_whole_number(
    tmp_path, monkeypatch
):
    import json

    from zippergen.deployment_profiles import (
        DEPLOYMENT_PROFILE_SCHEMA_VERSION,
        _load_deployment_profile,
    )

    home = tmp_path / "home"
    (home / "deployments").mkdir(parents=True)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    (home / "deployments" / "floaty.json").write_text(json.dumps({
        "schema_version": float(DEPLOYMENT_PROFILE_SCHEMA_VERSION),
        "name": "floaty",
        "store": str(home / "runs/floaty.sqlite"),
        "log": str(home / "logs/floaty.log"),
        "cwd": str(tmp_path),
    }))

    with pytest.raises(SystemExit) as caught:
        _load_deployment_profile("floaty")

    assert "does not say which schema it uses" in str(caught.value)


# One rule orders `open_store`: nothing about the file is changed until the
# store has been identified. These cover the ways a store can fail to be
# identified, because testing one of them and asserting the rule is how a
# locked store came to be converted to WAL by a reader that refused it.


def _future_store(directory):
    import sqlite3

    from zippergen.store import SCHEMA_VERSION, open_store

    path = directory / "future.sqlite"
    open_store(str(path)).close()
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute(
        "UPDATE store_meta SET value=? WHERE key='schema_version'",
        (str(SCHEMA_VERSION + 1),),
    )
    connection.commit()
    connection.close()
    for suffix in ("-wal", "-shm"):
        (directory / f"future.sqlite{suffix}").unlink(missing_ok=True)
    return path


def test_a_future_store_keeps_its_permissions(tmp_path):
    """Permissions are persistent too, so they are set after identification."""

    import stat

    from zippergen.store import StoreSchemaError, open_store

    store = _future_store(tmp_path)
    store.chmod(0o640)

    with pytest.raises(StoreSchemaError):
        open_store(str(store))

    assert stat.S_IMODE(store.stat().st_mode) == 0o640


def test_a_locked_store_is_refused_rather_than_assumed_versionless(tmp_path):
    """A locked store reads as "no version"; that is not "nothing to protect"."""

    import hashlib
    import subprocess
    import sys
    import textwrap

    from zippergen.store import StoreSchemaError, open_store

    store = _future_store(tmp_path)
    before = hashlib.sha256(store.read_bytes()).hexdigest()
    holder_source = tmp_path / "holder.py"
    holder_source.write_text(textwrap.dedent('''
        import sqlite3, sys, time
        connection = sqlite3.connect(sys.argv[1], timeout=30)
        connection.execute("BEGIN EXCLUSIVE")
        print("locked", flush=True)
        time.sleep(8)
        connection.rollback()
    '''))
    holder = subprocess.Popen(
        [sys.executable, str(holder_source), str(store)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        holder.stdout.readline()
        with pytest.raises(StoreSchemaError) as caught:
            open_store(str(store))
    finally:
        holder.kill()
        holder.wait()

    assert "was not modified" in str(caught.value)
    assert hashlib.sha256(store.read_bytes()).hexdigest() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "future.sqlite",
        "holder.py",
    ]


def test_a_corrupt_store_says_what_to_do(tmp_path):
    """A raw SQLite error tells a person nothing about their options."""

    from zippergen.store import StoreSchemaError, open_store

    store = tmp_path / "corrupt.sqlite"
    store.write_text("not a database at all")

    with pytest.raises(StoreSchemaError) as caught:
        open_store(str(store))

    assert "reset the run or deployment" in str(caught.value)
