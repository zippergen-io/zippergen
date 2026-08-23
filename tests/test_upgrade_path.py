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
    RUN_SCHEMA_VERSION,
    WORKSPACE_SCHEMA_VERSION,
    Workspace,
    WorkspaceError,
)


# ---------------------------------------------------------------------------
# Configuration: carried forward
# ---------------------------------------------------------------------------


def test_a_previous_deployment_profile_still_loads(tmp_path, monkeypatch):
    """The failure that had no way out.

    `zg deploy` writes a current profile, and loads the existing one first, so
    refusing the old schema meant the advice to redeploy could not be followed.
    `zg deploy remove` keeps the profile, so nothing else unblocked it either.
    """

    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))
    directory = tmp_path / "home" / "deployments"
    directory.mkdir(parents=True)
    (directory / "old.json").write_text(json.dumps({
        "schema_version": DEPLOYMENT_PROFILE_SCHEMA_VERSION - 1,
        "name": "old",
        "inputs": {"number": 3},
        "options": {"send_mode": "send"},
    }))

    profile = _load_deployment_profile("old")

    assert profile["schema_version"] == DEPLOYMENT_PROFILE_SCHEMA_VERSION
    # The settings someone answered are the reason to migrate rather than start
    # again, so they must survive with their types.
    assert profile["options"] == {"send_mode": "send"}
    assert profile["inputs"] == {"number": 3}
    assert isinstance(profile["inputs"]["number"], int)


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
    assert "cannot carry forward" in message
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


def test_a_project_manifest_carrying_an_old_schema_stamp_is_still_read(
    tmp_path, monkeypatch
):
    """`zippergen.toml` holds choices a person made, and no bookkeeping.

    Project configuration carries no schema stamp: an older file simply has one
    key nobody reads. Refusing such a file would strand a colleague holding a
    version-controlled project, and unlike a deployment there is no `reset` to
    fall back on. If a breaking format change ever needs one, the stamp returns
    then, and its absence identifies this layout.
    """

    monkeypatch.chdir(tmp_path)
    (tmp_path / "zippergen.toml").write_text(
        "schema_version = 1\nname = 'p'\nspecification_file = 'spec.md'\n"
    )

    manifest = Workspace(tmp_path).project_manifest()

    assert manifest["name"] == "p"
    assert manifest["specification_file"] == "spec.md"


def test_a_project_manifest_identity_is_adopted_without_moving_the_workspace(
    tmp_path,
):
    """An existing project keeps the private state it already has.

    The workspace key hashes the identity, so adopting a manifest's value into
    local state must not change where the project's credentials live -- and a
    project that never had one must not be given one.
    """

    home = tmp_path / "home"
    root = tmp_path / "legacy"
    root.mkdir()
    (root / "zippergen.toml").write_text(
        "project_id = 'a1b2c3d4e5f60718293a4b5c6d7e8f90'\n"
        "name = 'legacy'\nspecification_file = 'spec.md'\n"
    )
    workspace = Workspace(root, home=home)
    before = workspace.directory

    workspace.require_project()

    assert workspace.project_id_path.read_text().strip() == (
        "a1b2c3d4e5f60718293a4b5c6d7e8f90"
    )
    assert workspace.directory == before


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


def test_adopting_an_identity_clears_the_bookkeeping_it_came_from(tmp_path):
    """The move finishes in one step, and preserves every real choice.

    A manifest left carrying `project_id` after the value moved would still
    look like the place that holds it, which is the confusion the move removes.
    """

    import tomllib

    home = tmp_path / "home"
    root = tmp_path / "rich"
    root.mkdir()
    (root / "zippergen.toml").write_text(
        "schema_version = 2\n"
        "project_id = 'c1c2c3c4c5c6c7c8c9cacbcccdcecfd0'\n"
        "name = 'rich'\nspecification_file = 'spec.md'\n"
        "\n[providers.connections.'google-main']\n'kind' = 'google'\n"
        "\n[connectors.configurations.'sheet']\n"
        "'connection' = 'google-main'\n'kind' = 'google-sheets'\n"
        "'spreadsheet_id' = '1Abc'\n'tab' = 'Calls'\n"
    )
    before = tomllib.loads((root / "zippergen.toml").read_text())

    workspace = Workspace(root, home=home)
    workspace.require_project()

    after = tomllib.loads((root / "zippergen.toml").read_text())
    assert sorted(set(before) - set(after)) == ["project_id", "schema_version"]
    assert after["connectors"] == before["connectors"]
    assert after["providers"] == before["providers"]
    assert workspace.project_id_path.read_text().strip() == (
        "c1c2c3c4c5c6c7c8c9cacbcccdcecfd0"
    )


def test_the_manifest_is_rewritten_once_and_not_on_every_command(tmp_path):
    home = tmp_path / "home"
    root = tmp_path / "once"
    root.mkdir()
    (root / "zippergen.toml").write_text(
        "project_id = 'd1d2d3d4d5d6d7d8d9dadbdcdddedfe0'\n"
        "name = 'once'\nspecification_file = 'spec.md'\n"
    )
    Workspace(root, home=home).require_project()
    stamp = (root / "zippergen.toml").stat().st_mtime_ns

    Workspace(root, home=home).require_project()

    assert (root / "zippergen.toml").stat().st_mtime_ns == stamp
