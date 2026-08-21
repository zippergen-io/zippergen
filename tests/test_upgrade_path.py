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
        ("project manifest", PROJECT_SCHEMA_VERSION),
        ("workspace state", WORKSPACE_SCHEMA_VERSION),
        ("run record", RUN_SCHEMA_VERSION),
        ("deployment profile", DEPLOYMENT_PROFILE_SCHEMA_VERSION),
    ],
)
def test_configuration_schemas_start_above_one(what, current):
    """A version of 1 would leave no room to describe an older shape."""

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


def test_a_previous_project_manifest_is_not_refused_bare(tmp_path, monkeypatch):
    """`zippergen.toml` is version-controlled and shared with colleagues.

    Refusing it with no instruction leaves the person holding it with nothing to
    do, and unlike a deployment there is no `reset` to fall back on.
    """

    monkeypatch.chdir(tmp_path)
    (tmp_path / "zippergen.toml").write_text(
        f"schema_version = {PROJECT_SCHEMA_VERSION - 1}\nname = 'p'\n"
    )

    with pytest.raises(WorkspaceError) as caught:
        Workspace(tmp_path).project_manifest()

    message = str(caught.value)
    assert "cannot carry forward" in message
    assert "zippergen init" in message


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
