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
# Two questions about a service, which answer "unknown" differently
# ---------------------------------------------------------------------------


ALL_REPORTED_STATES = (
    "running",
    "restarting",
    "completed",
    "loaded",
    "not-loaded",
    "unknown",
)


def test_a_stopped_service_is_not_running_and_holds_nothing():
    """The state a deliberate stop actually produces on systemd.

    `systemctl stop` on a long-running unit leaves it inactive with a non-zero
    ExecMainStatus, which maps to "loaded", not "completed". Enumerating safe
    states missed it, so `zg deploy reset` stopped the service and then refused
    because it was stopped.
    """

    from zippergen.deployment_platform import (
        service_is_running,
        service_may_be_attached,
    )

    for state in ("loaded", "completed", "not-loaded"):
        assert service_is_running({"state": state}) is False, state
        assert service_may_be_attached({"state": state}) is False, state


def test_a_live_service_answers_yes_to_both():
    from zippergen.deployment_platform import (
        service_is_running,
        service_may_be_attached,
    )

    for state in ("running", "restarting"):
        assert service_is_running({"state": state}) is True, state
        assert service_may_be_attached({"state": state}) is True, state


def test_an_unknown_state_blocks_destruction_but_does_not_claim_it_is_running():
    """The one state where the two questions must disagree.

    Reset destroys durable state, so "could not ask the service manager" must
    not read as "stopped". But it must not read as "running" either: that is
    what made `deploy start` report a service as already running when nothing
    could reach it, and skip the start entirely.
    """

    from zippergen.deployment_platform import (
        service_is_running,
        service_may_be_attached,
    )

    assert service_is_running({"state": "unknown"}) is False
    assert service_may_be_attached({"state": "unknown"}) is True
    # A status with no state at all is unknown, not stopped.
    assert service_is_running({}) is False
    assert service_may_be_attached({}) is True


def test_every_state_the_managers_report_is_answered_by_both():
    """Neither question may fall through on anything either manager emits."""

    from zippergen.deployment_platform import (
        service_is_running,
        service_may_be_attached,
    )

    for state in ALL_REPORTED_STATES:
        assert isinstance(service_is_running({"state": state}), bool), state
        assert isinstance(service_may_be_attached({"state": state}), bool), state
    # Anything that may hold the store is either running or unaskable.
    assert {s for s in ALL_REPORTED_STATES if service_may_be_attached({"state": s})} == {
        "running",
        "restarting",
        "unknown",
    }
