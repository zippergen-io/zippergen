"""A project has two executions; each status names the other when it is live.

A deployed service and a selected durable run are the two places a project's
workflow can run, and only one may execute at a time. Each status command
reports the half it owns. Without this, a person who asks the wrong one sees a
stopped service or a finished run and concludes nothing is happening -- while
the other half is mid-workflow.
"""

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
