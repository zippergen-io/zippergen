"""Running current code over state an older ZipperGen wrote.

Every other test builds its state fresh, which is why two upgrade failures
reached a real deployment: a schema-2 deployment profile that `zg deploy`
refused, and a `zg deploy` that died on a schema-2 store before `reset` could be
reached. Neither was reproducible without state from a previous version.

The rule these tests pin down:

    Configuration is carried forward. Durable recovery state is refused, with
    an error naming the command that replaces it.

A control position means something only under the program that wrote it, so the
store cannot be migrated. Nothing else here has that property.
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
# Configuration: carried forward
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


@pytest.mark.parametrize(
    "what, current",
    [
        ("workspace state", WORKSPACE_SCHEMA_VERSION),
        ("run record", RUN_SCHEMA_VERSION),
        ("deployment profile", DEPLOYMENT_PROFILE_SCHEMA_VERSION),
    ],
)
def test_configuration_schemas_start_above_one(what, current):
    """A version of 1 would leave no room to describe an older shape.

    Project configuration is deliberately absent from this list: it carries no
    schema stamp, because everything in that file is a choice a person made.
    """

    assert current >= 2, what


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
# Durable state: refused, and it says which command replaces it
# ---------------------------------------------------------------------------


def test_a_previous_store_is_refused_and_names_the_command(tmp_path):
    """The store is the one thing that must not be migrated.

    Control state is child-index paths into the projected programs, so resuming
    under changed code would silently mean something else.
    """

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


def test_a_current_store_still_opens(tmp_path):
    path = tmp_path / "now.sqlite"
    open_store(str(path)).close()
    open_store(str(path)).close()
