import json
from pathlib import Path

import pytest

from zippergen.studio_deployments import (
    DeploymentRemovalError,
    compact_deployment_logs,
    present_deployment_artifacts,
    remove_deployment_artifacts,
    unregister_deployment_service,
)


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


def test_remove_deployment_artifacts_archives_the_complete_owned_unit(
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
    assert (result.archive / "profile/deployment.json").is_file()
    assert (result.archive / "profile/secrets.json").is_file()
    assert (result.archive / "state/store.sqlite").is_file()
    assert (result.archive / "state/store.sqlite-wal").is_file()
    assert (result.archive / "logs/deployment.log").is_file()
    assert (result.archive / "runtime/environment/python").is_file()
    assert (
        result.archive / "runtime/bundles/revision/workflow.py"
    ).is_file()
    assert not store.exists()
    assert not log.exists()
    assert not (home / "deployments/review-demo.json").exists()


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
        "zippergen.studio_deployments._deployment_service_status",
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
        "zippergen.studio_deployments._deployment_service_status",
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
        "zippergen.studio_deployments._service_manager",
        lambda: "launchd",
    )
    monkeypatch.setattr(
        "zippergen.studio_deployments._deployment_service_status",
        lambda _name: {"state": "running", "detail": "running"},
    )
    monkeypatch.setattr(
        "zippergen.studio_deployments._run_launchctl",
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
        "zippergen.studio_deployments._service_manager",
        lambda: "launchd",
    )
    monkeypatch.setattr(
        "zippergen.studio_deployments._deployment_service_status",
        lambda _name: {"state": "unknown", "detail": "timed out"},
    )

    with pytest.raises(
        DeploymentRemovalError,
        match="Cannot verify",
    ):
        unregister_deployment_service("review-demo")
