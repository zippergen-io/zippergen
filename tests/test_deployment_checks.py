import subprocess

from zippergen.deployment_checks import (
    _systemd_active_check,
    _systemd_enabled_check,
    _systemd_linger_check,
)


def _completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr="")


def test_systemd_linger_check_reports_logout_and_boot_readiness(monkeypatch):
    monkeypatch.setattr(
        "zippergen.deployment_checks.subprocess.run",
        lambda *_args, **_kwargs: _completed("yes\n"),
    )

    check = _systemd_linger_check()

    assert check["status"] == "ok"
    assert check["name"] == "systemd linger"
    assert check["enabled"] is True
    assert "without a login session" in check["detail"]


def test_systemd_linger_check_gives_an_action_when_disabled(monkeypatch):
    monkeypatch.setenv("USER", "workflow-user")
    monkeypatch.setattr(
        "zippergen.deployment_checks.subprocess.run",
        lambda *_args, **_kwargs: _completed("no\n", returncode=1),
    )

    check = _systemd_linger_check()

    assert check["status"] == "warn"
    assert check["enabled"] is False
    assert "loginctl enable-linger workflow-user" in check["detail"]


def test_systemd_enabled_check_gives_the_exact_enable_sequence(monkeypatch):
    monkeypatch.setattr(
        "zippergen.deployment_checks.subprocess.run",
        lambda *_args, **_kwargs: _completed("disabled\n", returncode=1),
    )

    check = _systemd_enabled_check("mailbox-poller")

    assert check["status"] == "warn"
    assert check["state"] == "disabled"
    assert "zippergen-mailbox-poller.service" in check["detail"]
    assert "zippergen deploy start --enable" in check["detail"]


def test_systemd_failed_service_is_a_failure_with_diagnostics(monkeypatch):
    monkeypatch.setattr(
        "zippergen.deployment_checks._systemd_service_status",
        lambda _name: {
            "manager": "systemd",
            "state": "loaded",
            "healthy": False,
            "active_state": "failed",
            "detail": (
                "zippergen-demo.service failed; last exit code 1. "
                "Inspect 'zippergen deploy logs'"
            ),
        },
    )

    check = _systemd_active_check("demo")

    assert check["status"] == "fail"
    assert check["active_state"] == "failed"
    assert "deploy logs" in check["detail"]
