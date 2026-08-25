"""A project has two executions; each status names the other when it is live.

A deployed service and a selected durable run are the two places a project's
workflow can run, and only one may execute at a time. Each status command
reports the half it owns. Without this, a person who asks the wrong one sees a
stopped service or a finished run and concludes nothing is happening -- while
the other half is mid-workflow.
"""

import sqlite3
from types import SimpleNamespace

from zippergen.serve import _other_execution_line

import pytest


@pytest.fixture
def args(tmp_path):
    return SimpleNamespace(project=str(tmp_path), name=None)


def test_run_status_names_a_running_deployment(monkeypatch, args) -> None:
    monkeypatch.setattr(
        "zippergen.serve._resolved_deployment_name", lambda _: "shop-a1b2"
    )
    monkeypatch.setattr(
        "zippergen.deployment_platform.deployment_service_status",
        lambda _: {"state": "running"},
    )
    line = _other_execution_line(args, owner="run")
    assert line is not None
    assert "shop-a1b2" in line
    assert "deploy status" in line


def test_run_status_is_silent_when_the_deployment_is_stopped(
    monkeypatch, args
) -> None:
    monkeypatch.setattr(
        "zippergen.serve._resolved_deployment_name", lambda _: "shop-a1b2"
    )
    monkeypatch.setattr(
        "zippergen.deployment_platform.deployment_service_status",
        lambda _: {"state": "stopped"},
    )
    assert _other_execution_line(args, owner="run") is None


def test_run_status_is_silent_when_there_is_no_deployment(
    monkeypatch, args
) -> None:
    """A project without a deployment must not be told about one."""

    def refuse(_):
        raise SystemExit("no deployment for this project")

    monkeypatch.setattr("zippergen.serve._resolved_deployment_name", refuse)
    assert _other_execution_line(args, owner="run") is None


@pytest.mark.parametrize("status", ["running", "waiting"])
def test_deploy_status_names_an_executing_durable_run(
    monkeypatch, args, status
) -> None:
    monkeypatch.setattr(
        "zippergen.workspace.Workspace.current_run",
        lambda _: {"run_id": "shop-20260825-101010", "status": status},
    )
    line = _other_execution_line(args, owner="deploy")
    assert line is not None
    assert "shop-20260825-101010" in line
    assert status in line
    assert "run status" in line
    if status == "waiting":
        assert "waiting for a person" in line


@pytest.mark.parametrize("status", ["done", "failed", "reset", ""])
def test_deploy_status_is_silent_about_a_finished_run(
    monkeypatch, args, status
) -> None:
    monkeypatch.setattr(
        "zippergen.workspace.Workspace.current_run",
        lambda _: {"run_id": "shop-20260825-101010", "status": status},
    )
    assert _other_execution_line(args, owner="deploy") is None


def test_deploy_status_is_silent_when_no_run_is_selected(
    monkeypatch, args
) -> None:
    monkeypatch.setattr(
        "zippergen.workspace.Workspace.current_run", lambda _: None
    )
    assert _other_execution_line(args, owner="deploy") is None


# ---------------------------------------------------------------------------
# A failed observation is not an absence
# ---------------------------------------------------------------------------

def test_a_failed_service_query_warns_instead_of_going_quiet(
    monkeypatch, args
) -> None:
    """Silence here reads as "nothing else is running", which may be false.

    Catching every failure and returning None made a corrupt profile or an
    unreachable service manager look exactly like a project with no
    deployment -- so an operator could read a finished run while the
    deployment was live but unreadable.
    """

    monkeypatch.setattr(
        "zippergen.serve._resolved_deployment_name", lambda _: "shop-a1b2"
    )

    def broken(_name):
        raise OSError("launchctl is not available")

    monkeypatch.setattr(
        "zippergen.deployment_platform.deployment_service_status", broken
    )
    line = _other_execution_line(args, owner="run")
    assert line is not None
    assert line.startswith("WARN")
    assert "shop-a1b2" in line
    assert "OSError" in line


def test_an_unknown_service_state_warns_rather_than_reading_as_stopped(
    monkeypatch, args
) -> None:
    """"unknown" means the manager could not be asked -- not that it is idle."""

    monkeypatch.setattr(
        "zippergen.serve._resolved_deployment_name", lambda _: "shop-a1b2"
    )
    monkeypatch.setattr(
        "zippergen.deployment_platform.deployment_service_status",
        lambda _: {"state": "unknown", "detail": "launchctl exited 1"},
    )
    line = _other_execution_line(args, owner="run")
    assert line is not None
    assert line.startswith("WARN")
    assert "may be running" in line


def test_a_failed_run_lookup_warns(monkeypatch, args) -> None:
    def broken(_self):
        raise sqlite3.DatabaseError("file is not a database")

    monkeypatch.setattr("zippergen.workspace.Workspace.current_run", broken)
    line = _other_execution_line(args, owner="deploy")
    assert line is not None
    assert line.startswith("WARN")
    assert "DatabaseError" in line


def test_expected_absence_stays_quiet_on_both_sides(monkeypatch, args) -> None:
    """Only failures speak up; a project without the other half says nothing."""

    from zippergen.workspace import WorkspaceError

    def no_deployment(_):
        raise SystemExit("no deployment for this project")

    monkeypatch.setattr(
        "zippergen.serve._resolved_deployment_name", no_deployment
    )
    assert _other_execution_line(args, owner="run") is None

    def no_project(_self):
        raise WorkspaceError("Not a ZipperGen project")

    monkeypatch.setattr("zippergen.workspace.Workspace.current_run", no_project)
    assert _other_execution_line(args, owner="deploy") is None
