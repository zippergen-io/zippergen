"""Service-manager state classification tests."""

import subprocess

from zippergen.deployment_platform import systemd_service_status


def _systemd_show(*, active: str, sub: str, status: int, restarts: int) -> str:
    return (
        "LoadState=loaded\n"
        f"ActiveState={active}\n"
        f"SubState={sub}\n"
        f"ExecMainStatus={status}\n"
        f"NRestarts={restarts}\n"
    )


def test_stopped_systemd_service_is_not_restarting_because_of_its_history(
    monkeypatch,
):
    monkeypatch.setattr(
        "zippergen.deployment_platform.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            stdout=_systemd_show(
                active="inactive",
                sub="dead",
                status=15,
                restarts=12,
            ),
            stderr="",
        ),
    )

    status = systemd_service_status("call-intake")

    assert status["state"] == "loaded"
    assert status["healthy"] is False
    assert status["active_state"] == "inactive"
    assert status["last_exit_code"] == 15
    assert status["restarts"] == 12


def test_activating_systemd_service_is_restarting(monkeypatch):
    monkeypatch.setattr(
        "zippergen.deployment_platform.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            stdout=_systemd_show(
                active="activating",
                sub="auto-restart",
                status=1,
                restarts=3,
            ),
            stderr="",
        ),
    )

    status = systemd_service_status("call-intake")

    assert status["state"] == "restarting"
    assert status["healthy"] is False
