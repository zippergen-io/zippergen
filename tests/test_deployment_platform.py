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


# ---------------------------------------------------------------------------
# When is it safe to change a deployment's durable state?
# ---------------------------------------------------------------------------


def test_a_stopped_service_is_not_live(  # noqa: D103
):
    """The state a deliberate stop actually produces on systemd.

    `systemctl stop` on a long-running unit leaves it inactive with a non-zero
    ExecMainStatus, which maps to "loaded", not "completed". Enumerating safe
    states missed it, so `zg deploy reset` stopped the service and then refused
    because it was stopped.
    """

    from zippergen.deployment_platform import service_is_live

    assert service_is_live({"state": "loaded"}) is False
    assert service_is_live({"state": "completed"}) is False
    assert service_is_live({"state": "not-loaded"}) is False


def test_a_service_that_may_still_have_a_process_is_live():
    from zippergen.deployment_platform import service_is_live

    assert service_is_live({"state": "running"}) is True
    assert service_is_live({"state": "restarting"}) is True


def test_an_unknown_service_state_counts_as_live():
    """Reset destroys state, so 'could not ask' must not read as 'stopped'."""

    from zippergen.deployment_platform import service_is_live

    assert service_is_live({"state": "unknown"}) is True
    assert service_is_live({}) is True


def test_every_state_the_managers_report_is_classified():
    """Both managers share one vocabulary; none of it may fall through."""

    from zippergen.deployment_platform import LIVE_SERVICE_STATES

    reported = {"running", "restarting", "completed", "loaded", "not-loaded", "unknown"}
    assert LIVE_SERVICE_STATES <= reported
