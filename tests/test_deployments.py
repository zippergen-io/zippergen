import json
import os
import plistlib
from pathlib import Path

import pytest

from zippergen.deployment_profiles import (
    DEPLOYMENT_PROFILE_SCHEMA_VERSION,
    _load_deployment_profile,
)
from zippergen.deployments import (
    DeploymentRemovalError,
    compact_deployment_logs,
    present_deployment_artifacts,
    remove_deployment_artifacts,
    reset_deployment_store,
    unregister_deployment_service,
)
from zippergen.serve import (
    _repair_deployment_permissions,
    _write_deployment_artifacts,
)


def test_deployment_profile_preserves_typed_inputs(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    profile = {
        "schema_version": DEPLOYMENT_PROFILE_SCHEMA_VERSION,
        "name": "typed-inputs",
        "cwd": str(tmp_path),
        "store": str(home / "runs/typed-inputs.sqlite"),
        "log": str(home / "logs/typed-inputs.log"),
        "python": "/usr/bin/python3",
        "inputs": {"coordinates": (1, [2, 3])},
    }

    _write_deployment_artifacts(profile)
    raw = json.loads((home / "deployments/typed-inputs.json").read_text())
    assert "__zippergen_typed_value_v1__" in raw["inputs"]

    loaded = _load_deployment_profile("typed-inputs")
    assert loaded["inputs"] == {"coordinates": (1, [2, 3])}
    assert type(loaded["inputs"]["coordinates"]) is tuple


def test_deployment_artifacts_are_private_under_a_permissive_umask(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    profile = {
        "name": "private",
        "cwd": str(tmp_path),
        "store": str(home / "runs/private.sqlite"),
        "log": str(home / "logs/private.log"),
        "python": "/usr/bin/python3",
        "inputs": {"private": "workflow-input"},
    }

    previous = os.umask(0o022)
    try:
        _write_deployment_artifacts(profile)
    finally:
        os.umask(previous)

    for directory in (home, home / "runs", home / "logs", home / "deployments"):
        assert directory.stat().st_mode & 0o777 == 0o700
    for path in (
        home / "logs/private.log",
        home / "deployments/private.json",
        home / "deployments/zippergen-private.service",
        home / "deployments/io.zippergen.private.plist",
    ):
        assert path.stat().st_mode & 0o777 == 0o600
    assert (home / "deployments/private.sh").stat().st_mode & 0o777 == 0o700
    service = (home / "deployments/zippergen-private.service").read_text()
    assert "UMask=0077" in service
    launchd = plistlib.loads(
        (home / "deployments/io.zippergen.private.plist").read_bytes()
    )
    assert launchd["Umask"] == 0o077


def test_permission_repair_secures_existing_managed_artifacts(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    profile = {
        "name": "repair",
        "cwd": str(tmp_path),
        "store": str(home / "runs/repair.sqlite"),
        "log": str(home / "logs/repair.log"),
        "python": "/usr/bin/python3",
        "inputs": {},
    }
    _write_deployment_artifacts(profile)
    profile_path = home / "deployments/repair.json"
    log_path = home / "logs/repair.log"
    home.chmod(0o755)
    profile_path.chmod(0o644)
    log_path.chmod(0o644)

    _repair_deployment_permissions("repair", profile)

    assert home.stat().st_mode & 0o777 == 0o700
    assert profile_path.stat().st_mode & 0o777 == 0o600
    assert log_path.stat().st_mode & 0o777 == 0o600


def _deployment_fixture(tmp_path, monkeypatch, name="review-demo"):
    home = tmp_path / "home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv(
        "ZIPPERGEN_LAUNCH_AGENTS_DIR",
        str(tmp_path / "launch-agents"),
    )
    store = home / "runs" / f"{name}.sqlite"
    log = home / "logs" / f"{name}.log"
    secrets = home / "deployments" / f"{name}.secrets.json"
    profile = {
        "name": name,
        "store": str(store),
        "log": str(log),
        "secrets_file": str(secrets),
    }
    profile_path = home / "deployments" / f"{name}.json"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(json.dumps(profile))
    secrets.write_text("{}")
    store.parent.mkdir(parents=True)
    store.write_bytes(b"sqlite")
    Path(str(store) + "-wal").write_bytes(b"wal")
    log.parent.mkdir(parents=True)
    log.write_text("log")
    environment = home / "environments" / name
    environment.mkdir(parents=True)
    (environment / "python").write_text("python")
    bundles = home / "apps" / name / "revision"
    bundles.mkdir(parents=True)
    (bundles / "workflow.py").write_text("workflow")
    return home, profile, store, log


def test_remove_deployment_archives_only_what_cannot_be_rebuilt(
    tmp_path,
    monkeypatch,
):
    home, profile, store, log = _deployment_fixture(tmp_path, monkeypatch)

    result = remove_deployment_artifacts(
        "review-demo",
        profile,
        purge=False,
    )

    assert result.archive is not None
    assert result.archive.is_dir()
    assert (result.archive / "removal.json").is_file()
    # What actually happened, and what produced it, is kept.
    assert (result.archive / "state/store.sqlite").is_file()
    assert (result.archive / "state/store.sqlite-wal").is_file()
    assert (result.archive / "logs/deployment.log").is_file()
    assert (result.archive / "profile/deployment.json").is_file()
    # Secrets are never left behind, and everything else is rebuilt by
    # deploying again.
    assert not (result.archive / "profile/secrets.json").exists()
    assert not (result.archive / "runtime").exists()
    assert not (result.archive / "launch").exists()
    # The deployment itself is gone from active use.
    assert not store.exists()
    assert not log.exists()
    assert not (home / "deployments/review-demo.json").exists()


def test_remove_deployment_leaves_no_secret_anywhere_in_the_archive(
    tmp_path,
    monkeypatch,
):
    home, profile, _store, _log = _deployment_fixture(tmp_path, monkeypatch)

    result = remove_deployment_artifacts(
        "review-demo",
        profile,
        purge=False,
    )

    assert result.archive is not None
    for path in result.archive.rglob("*"):
        if path.is_file():
            assert "secret" not in path.name.casefold()
    trash = home / "trash" / "deployments"
    assert not any(
        "secret" in path.name.casefold()
        for path in trash.rglob("*")
        if path.is_file()
    )


def test_remove_preserves_external_paths_referenced_by_a_profile(
    tmp_path,
    monkeypatch,
):
    _home, profile, _store, _log = _deployment_fixture(tmp_path, monkeypatch)
    external = tmp_path / "external"
    external.mkdir()
    external_secrets = external / "secrets.json"
    external_store = external / "state.sqlite"
    external_log = external / "workflow.log"
    external_secrets.write_text("private")
    external_store.write_text("state")
    external_log.write_text("log")
    profile.update({
        "secrets_file": str(external_secrets),
        "store": str(external_store),
        "log": str(external_log),
    })

    result = remove_deployment_artifacts("review-demo", profile, purge=False)

    assert result.archive is not None
    assert external_secrets.read_text() == "private"
    assert external_store.read_text() == "state"
    assert external_log.read_text() == "log"
    archived_sources = json.loads(
        (result.archive / "removal.json").read_text()
    )["artifacts"]
    assert str(external_secrets) not in {
        item["source"] for item in archived_sources
    }


def test_reset_and_compact_refuse_external_profile_references(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    monkeypatch.setattr(
        "zippergen.deployments._deployment_service_status",
        lambda _name: {"state": "not-loaded", "detail": "not loaded"},
    )
    external_store = tmp_path / "external.sqlite"
    external_log = tmp_path / "external.log"
    external_store.write_text("keep-store")
    external_log.write_text("keep-log")

    with pytest.raises(DeploymentRemovalError, match="external store"):
        reset_deployment_store(
            "review-demo", {"store": str(external_store)}
        )
    with pytest.raises(DeploymentRemovalError, match="external log"):
        compact_deployment_logs(
            "review-demo", {"log": str(external_log)}
        )

    assert external_store.read_text() == "keep-store"
    assert external_log.read_text() == "keep-log"


def test_remove_deployment_artifacts_purge_leaves_no_archive(
    tmp_path,
    monkeypatch,
):
    home, profile, store, _log = _deployment_fixture(tmp_path, monkeypatch)

    result = remove_deployment_artifacts(
        "review-demo",
        profile,
        purge=True,
    )

    assert result.purged is True
    assert result.archive is None
    assert not store.exists()
    trash = home / "trash" / "deployments"
    assert trash.is_dir()
    assert list(trash.iterdir()) == []


def test_compact_deployment_logs_rotates_and_bounds_archives(
    tmp_path,
    monkeypatch,
):
    home, profile, _store, log = _deployment_fixture(
        tmp_path,
        monkeypatch,
    )
    log.write_bytes(b"current deployment log\n")
    archive_root = home / "trash" / "deployment-logs"
    archive_root.mkdir(parents=True)
    for index in range(4):
        archive = archive_root / f"review-demo-20260101-00000{index}.log"
        archive.write_bytes(f"old-{index}".encode())
        archive.touch()
    monkeypatch.setattr(
        "zippergen.deployments._deployment_service_status",
        lambda _name: {
            "state": "not-loaded",
            "detail": "service is stopped",
        },
    )

    result = compact_deployment_logs(
        "review-demo",
        profile,
        keep_archives=3,
    )

    assert result.archived_bytes == len(b"current deployment log\n")
    assert result.removed_archives == 2
    assert log.read_bytes() == b""
    archives = sorted(archive_root.glob("review-demo-*.log"))
    assert len(archives) == 3
    assert result.archive in archives
    stored = json.loads(
        (home / "deployments" / "review-demo.json").read_text()
    )
    assert stored["log_generation_offset"] == 0
    assert stored["log_compacted_at"]


def test_compact_deployment_logs_refuses_a_running_service(
    tmp_path,
    monkeypatch,
):
    _home, profile, _store, _log = _deployment_fixture(
        tmp_path,
        monkeypatch,
    )
    monkeypatch.setattr(
        "zippergen.deployments._deployment_service_status",
        lambda _name: {
            "state": "running",
            "detail": "service is running",
        },
    )

    with pytest.raises(DeploymentRemovalError, match="Stop deployment"):
        compact_deployment_logs("review-demo", profile)


def test_remove_refuses_a_store_shared_with_another_deployment(
    tmp_path,
    monkeypatch,
):
    home, profile, store, _log = _deployment_fixture(tmp_path, monkeypatch)
    other = {
        "name": "other",
        "store": str(store),
        "log": str(home / "logs" / "other.log"),
    }
    (home / "deployments/other.json").write_text(json.dumps(other))

    with pytest.raises(
        DeploymentRemovalError,
        match="also used by deployment other",
    ):
        present_deployment_artifacts("review-demo", profile)


def test_unregister_launchd_service_boots_out_and_removes_registration(
    tmp_path,
    monkeypatch,
):
    _home, _profile, _store, _log = _deployment_fixture(
        tmp_path,
        monkeypatch,
    )
    installed = tmp_path / "launch-agents/io.zippergen.review-demo.plist"
    installed.parent.mkdir(parents=True)
    installed.write_text("plist")
    calls = []
    monkeypatch.setattr(
        "zippergen.deployments._service_manager",
        lambda: "launchd",
    )
    monkeypatch.setattr(
        "zippergen.deployments._deployment_service_status",
        lambda _name: {"state": "running", "detail": "running"},
    )
    monkeypatch.setattr(
        "zippergen.deployments._run_launchctl",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    result = unregister_deployment_service("review-demo")

    assert "stopped and unregistered" in result
    assert calls
    assert "bootout" in calls[0][0]
    assert calls[0][1]["check"] is True
    assert not installed.exists()


def test_unregister_refuses_unknown_state_with_an_installed_service(
    tmp_path,
    monkeypatch,
):
    _home, _profile, _store, _log = _deployment_fixture(
        tmp_path,
        monkeypatch,
    )
    installed = tmp_path / "launch-agents/io.zippergen.review-demo.plist"
    installed.parent.mkdir(parents=True)
    installed.write_text("plist")
    monkeypatch.setattr(
        "zippergen.deployments._service_manager",
        lambda: "launchd",
    )
    monkeypatch.setattr(
        "zippergen.deployments._deployment_service_status",
        lambda _name: {"state": "unknown", "detail": "timed out"},
    )

    with pytest.raises(
        DeploymentRemovalError,
        match="Cannot verify",
    ):
        unregister_deployment_service("review-demo")


def test_reset_accepts_a_service_that_was_deliberately_stopped(tmp_path, monkeypatch):
    """`zg deploy reset` stopped the service, then refused because it was stopped.

    On systemd a stopped long-running unit reports "loaded": inactive, still
    installed, non-zero last exit because SIGTERM ended it. The old guard
    allowed only "not-loaded" and "completed", so reset was unreachable on every
    Linux deployment.
    """

    from zippergen import deployments

    home = tmp_path / "home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    store = home / "runs/d.sqlite"
    store.parent.mkdir(parents=True)
    store.write_text("durable state")
    profile = {"name": "d", "store": str(store)}
    monkeypatch.setattr(
        deployments,
        "_deployment_service_status",
        lambda _name: {"state": "loaded", "detail": "unit is inactive"},
    )
    result = deployments.reset_deployment_store("d", profile)

    assert not store.exists()
    assert result.archive is not None


def test_reset_still_refuses_while_the_service_is_running(tmp_path, monkeypatch):
    from zippergen import deployments

    store = tmp_path / "run.sqlite"
    store.write_text("durable state")
    monkeypatch.setattr(
        deployments,
        "_deployment_service_status",
        lambda _name: {"state": "running", "detail": "unit is running"},
    )

    with pytest.raises(deployments.DeploymentRemovalError, match="before resetting"):
        deployments.reset_deployment_store("d", {"name": "d", "store": str(store)})

    assert store.exists()


def test_a_brand_new_home_passes_its_own_readiness_checks(tmp_path, monkeypatch):
    """The first deploy on a new machine must not fail its own checks.

    The readiness checks require a private home containing a log directory.
    Creating that only when a candidate is published meant the checks demanded
    a state that nothing had created yet, so every first deploy failed.
    """

    from zippergen.serve import _prepare_managed_home

    home = tmp_path / "home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    profile = {
        "schema_version": DEPLOYMENT_PROFILE_SCHEMA_VERSION,
        "name": "first-deploy",
        "cwd": str(tmp_path),
        "store": str(home / "runs/first-deploy.sqlite"),
        "log": str(home / "logs/first-deploy.log"),
        "python": "/usr/bin/python3",
    }

    _prepare_managed_home(profile)

    assert home.is_dir()
    assert (home / "logs").is_dir(), "the log directory the checks require"
    assert (home / "runs").is_dir()
    assert (home / "deployments").is_dir()
    for directory in (home, home / "logs", home / "runs", home / "deployments"):
        assert directory.stat().st_mode & 0o077 == 0, directory


def test_preparing_the_managed_home_twice_changes_nothing(tmp_path, monkeypatch):
    from zippergen.serve import _prepare_managed_home

    home = tmp_path / "home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    profile = {
        "schema_version": DEPLOYMENT_PROFILE_SCHEMA_VERSION,
        "name": "twice",
        "cwd": str(tmp_path),
        "store": str(home / "runs/twice.sqlite"),
        "log": str(home / "logs/twice.log"),
        "python": "/usr/bin/python3",
    }

    _prepare_managed_home(profile)
    (home / "logs" / "twice.log").write_text("kept\n")
    _prepare_managed_home(profile)

    assert (home / "logs" / "twice.log").read_text() == "kept\n"


def test_a_world_readable_log_from_an_earlier_release_is_made_private(
    tmp_path, monkeypatch
):
    """The checks require a private log file, not merely a private directory.

    A log left readable by an older release failed the checks on every later
    deploy, because the only code that tightened it ran after they had already
    refused the candidate.
    """

    from zippergen.serve import _prepare_managed_home

    home = tmp_path / "home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    log_path = home / "logs" / "legacy.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("from an earlier release\n")
    log_path.chmod(0o644)

    _prepare_managed_home({
        "schema_version": DEPLOYMENT_PROFILE_SCHEMA_VERSION,
        "name": "legacy",
        "cwd": str(tmp_path),
        "store": str(home / "runs/legacy.sqlite"),
        "log": str(log_path),
        "python": "/usr/bin/python3",
    })

    assert log_path.stat().st_mode & 0o077 == 0
    assert log_path.read_text() == "from an earlier release\n", "log kept"


# A configured deployment is the result of four sources with a fixed
# precedence. Reporting the value without the source leaves an operator unable
# to tell a deliberate setting from a default nobody chose -- which is exactly
# what `deploy --yes` does silently.


def _sources_for(profile, *, overrides=None, environ=None, monkeypatch=None):
    from zippergen.deployment import DeploymentField, DeploymentSpec
    from zippergen.serve import _collect_deployment_fields

    spec = DeploymentSpec(
        description="under test",
        fields=(
            DeploymentField("recipient", "Recipient", target="option"),
            DeploymentField("mode", "Mode", target="option", default="draft"),
            DeploymentField("token", "Token", target="env"),
        ),
    )
    sources: dict[str, str] = {}
    values, _ = _collect_deployment_fields(
        spec,
        profile,
        overrides=overrides or {},
        interactive=False,
        sources=sources,
    )
    return values, sources


def test_each_field_reports_where_its_value_came_from(tmp_path, monkeypatch):
    from zippergen import serve

    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("token", "from-the-environment")
    profile = {
        "name": "sources",
        "options": {"recipient": "kept@example.com"},
    }

    values, sources = _sources_for(profile, overrides={"mode": "send"})

    assert values["recipient"] == "kept@example.com"
    assert sources["recipient"] == serve.FIELD_SOURCE_DEPLOYMENT
    assert values["mode"] == "send"
    assert sources["mode"] == serve.FIELD_SOURCE_OVERRIDE
    assert values["token"] == "from-the-environment"
    assert sources["token"] == serve.FIELD_SOURCE_ENVIRONMENT


def test_a_value_nobody_chose_is_named_a_default(tmp_path, monkeypatch):
    from zippergen import serve

    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("token", raising=False)

    values, sources = _sources_for({"name": "defaults"})

    assert values["mode"] == "draft"
    assert sources["mode"] == serve.FIELD_SOURCE_DEFAULT
    assert sources["recipient"] == serve.FIELD_SOURCE_UNSET


def test_a_reported_configuration_never_prints_a_secret(capsys):
    from zippergen.deployment import DeploymentField, DeploymentSpec
    from zippergen.serve import _print_deployment_configuration

    spec = DeploymentSpec(
        description="under test",
        fields=(
            DeploymentField("api_key", "Key", target="env", secret=True),
            DeploymentField("recipient", "Recipient", target="option"),
        ),
    )

    _print_deployment_configuration(
        spec,
        {"api_key": "sk-do-not-print-me", "recipient": "a@b.com"},
        {"api_key": "environment", "recipient": "this deployment"},
        heading="Configuration",
    )

    output = capsys.readouterr().out
    assert "sk-do-not-print-me" not in output
    assert "18 characters" in output
    assert "a@b.com" in output


def test_a_stored_configuration_is_readable_without_the_workflow(tmp_path, monkeypatch):
    """`deploy status` must answer 'where do these values live' on its own."""

    from zippergen.deployment import DeploymentField, DeploymentSpec
    from zippergen.serve import _stored_deployment_configuration

    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))
    spec = DeploymentSpec(
        description="under test",
        fields=(DeploymentField("recipient", "Recipient", target="option"),),
    )
    profile = {
        "name": "stored",
        "deployment_spec": spec.as_dict(),
        "options": {"recipient": "kept@example.com"},
    }

    found = _stored_deployment_configuration(profile)

    assert found is not None
    recovered_spec, stored = found
    assert [field.name for field in recovered_spec.fields] == ["recipient"]
    assert stored["recipient"] == "kept@example.com"


def test_a_profile_without_a_declaration_shows_no_configuration(tmp_path, monkeypatch):
    from zippergen.serve import _stored_deployment_configuration

    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))

    assert _stored_deployment_configuration({"name": "bare"}) is None


# The invariant: every non-secret answer a person gives is kept in the visible
# project file, and the deployment profile is derived from it. Two places to
# author configuration is what made "where is the value I typed?" have two
# answers depending on which code path collected it.


def _configuration_spec():
    from zippergen.deployment import DeploymentField, DeploymentSpec

    return DeploymentSpec(
        description="under test",
        fields=(
            DeploymentField("recipient", "Recipient", target="option"),
            DeploymentField("rounds", "Rounds", target="option", default=4),
            DeploymentField("token", "Token", target="env", secret=True),
        ),
    )


def _workspace_at(tmp_path, monkeypatch):
    from zippergen.workspace import Workspace

    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))
    project = tmp_path / "project"
    project.mkdir()
    workspace = Workspace(project, home=tmp_path / "home")
    workspace.initialize_project(name="configured")
    return workspace


def test_an_answer_is_written_to_the_visible_project_file(tmp_path, monkeypatch):
    from zippergen.serve import _collect_deployment_fields

    workspace = _workspace_at(tmp_path, monkeypatch)
    profile = {"name": "configured"}

    _collect_deployment_fields(
        _configuration_spec(),
        profile,
        overrides={"recipient": "alice@example.org"},
        interactive=False,
        workspace=workspace,
    )

    assert workspace.configuration_values() == {
        "recipient": "alice@example.org",
        "rounds": 4,
    }
    assert "alice@example.org" in workspace.manifest_path.read_text()


def test_the_project_file_is_read_back_as_the_source(tmp_path, monkeypatch):
    from zippergen import serve
    from zippergen.serve import _collect_deployment_fields

    workspace = _workspace_at(tmp_path, monkeypatch)
    workspace.write_configuration_values(
        {"recipient": "kept@example.org", "rounds": 9}
    )
    sources: dict[str, str] = {}

    values, _ = _collect_deployment_fields(
        _configuration_spec(),
        {"name": "configured"},
        overrides={},
        interactive=False,
        sources=sources,
        workspace=workspace,
    )

    assert values["recipient"] == "kept@example.org"
    assert values["rounds"] == 9, "a number keeps its type through TOML"
    assert sources["recipient"] == serve.FIELD_SOURCE_PROJECT


def test_a_secret_is_never_written_to_the_visible_project_file(
    tmp_path, monkeypatch
):
    from zippergen.serve import _collect_deployment_fields

    workspace = _workspace_at(tmp_path, monkeypatch)
    monkeypatch.setenv("token", "sk-do-not-commit-me")

    _collect_deployment_fields(
        _configuration_spec(),
        {"name": "configured"},
        overrides={},
        interactive=False,
        workspace=workspace,
    )

    assert "token" not in workspace.configuration_values()
    assert "sk-do-not-commit-me" not in workspace.manifest_path.read_text()


def test_a_deployment_configured_before_this_rule_migrates_itself(
    tmp_path, monkeypatch
):
    """An older deployment holds its answers only in its profile.

    Adopting them on the next deploy, and writing them into the project, is the
    whole migration: no separate command, and nothing for an operator to run.
    """

    from zippergen import serve
    from zippergen.serve import _collect_deployment_fields

    workspace = _workspace_at(tmp_path, monkeypatch)
    assert workspace.configuration_values() == {}
    legacy_profile = {
        "name": "configured",
        "options": {"recipient": "legacy@example.org", "rounds": 7},
    }
    sources: dict[str, str] = {}

    values, _ = _collect_deployment_fields(
        _configuration_spec(),
        legacy_profile,
        overrides={},
        interactive=False,
        sources=sources,
        workspace=workspace,
    )

    assert values["recipient"] == "legacy@example.org"
    assert sources["recipient"] == serve.FIELD_SOURCE_DEPLOYMENT
    assert workspace.configuration_values() == {
        "recipient": "legacy@example.org",
        "rounds": 7,
    }


def test_an_edited_project_file_wins_over_the_published_profile(
    tmp_path, monkeypatch
):
    from zippergen.serve import _collect_deployment_fields

    workspace = _workspace_at(tmp_path, monkeypatch)
    workspace.write_configuration_values({"recipient": "edited@example.org"})
    published = {
        "name": "configured",
        "options": {"recipient": "published@example.org"},
    }

    values, _ = _collect_deployment_fields(
        _configuration_spec(),
        published,
        overrides={},
        interactive=False,
        workspace=workspace,
    )

    assert values["recipient"] == "edited@example.org"


# A deployment is made of three things: the runtime, the workflow bundle, and
# the answers. Two of them already report whether the deployment runs something
# older than the project. The third did not, so a value edited in
# zippergen.toml looked applied when it was not.


def _drift_profile(tmp_path, monkeypatch, deployed):
    from zippergen.deployment import DeploymentField, DeploymentSpec
    from zippergen.workspace import Workspace

    home = tmp_path / "home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    project = tmp_path / "project"
    project.mkdir()
    workspace = Workspace(project, home=home)
    workspace.initialize_project(name="drifting")
    spec = DeploymentSpec(
        description="under test",
        fields=(
            DeploymentField("rounds", "Rounds", target="option"),
            DeploymentField("token", "Token", target="env", secret=True),
        ),
    )
    return workspace, {
        "name": "drifting",
        "source_cwd": str(project),
        "deployment_spec": spec.as_dict(),
        "options": {"rounds": deployed},
    }


def test_configuration_matching_the_project_reports_current(tmp_path, monkeypatch):
    from zippergen.deployment_checks import _configuration_freshness_check

    workspace, profile = _drift_profile(tmp_path, monkeypatch, deployed=9)
    workspace.write_configuration_values({"rounds": 9})

    check = _configuration_freshness_check(profile, workspace.root)

    assert check["status"] == "ok"
    assert check["freshness"] == "current"


def test_an_answer_edited_but_not_deployed_is_reported(tmp_path, monkeypatch):
    from zippergen.deployment_checks import _configuration_freshness_check

    workspace, profile = _drift_profile(tmp_path, monkeypatch, deployed=9)
    workspace.write_configuration_values({"rounds": 2})

    check = _configuration_freshness_check(profile, workspace.root)

    assert check["status"] == "warn"
    assert check["freshness"] == "stale"
    detail = str(check["detail"])
    assert "deployed 9" in detail and "project 2" in detail
    assert "Redeploy" in detail


def test_drift_reporting_never_names_a_secret(tmp_path, monkeypatch):
    from zippergen.deployment_checks import _stored_deployment_answers

    _workspace, profile = _drift_profile(tmp_path, monkeypatch, deployed=9)
    profile["environment"] = {"token": "sk-do-not-report-me"}

    answers = _stored_deployment_answers(profile)

    assert "token" not in answers
    assert answers == {"rounds": 9}


def test_a_deployment_with_no_declaration_reports_nothing_to_compare(
    tmp_path, monkeypatch
):
    from zippergen.deployment_checks import _configuration_freshness_check

    workspace, profile = _drift_profile(tmp_path, monkeypatch, deployed=9)
    profile.pop("deployment_spec")

    check = _configuration_freshness_check(profile, workspace.root)

    assert check["status"] == "ok"
    assert "0 answer(s)" in str(check["detail"])
