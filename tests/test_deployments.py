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
