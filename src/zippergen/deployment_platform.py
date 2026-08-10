"""Host paths and launchd/systemd operations for local deployments."""

from __future__ import annotations

import os
import platform
import re
import shlex
import subprocess
from pathlib import Path


def slug(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text.strip()).strip("-._")
    return text or "workflow"


def zippergen_home() -> Path:
    return Path(os.environ.get("ZIPPERGEN_HOME", str(Path.home() / ".zippergen"))).expanduser()


def deployments_dir() -> Path:
    return zippergen_home() / "deployments"


def deployment_profile_path(name: str) -> Path:
    return deployments_dir() / f"{slug(name)}.json"


def deployment_script_path(name: str) -> Path:
    return deployments_dir() / f"{slug(name)}.sh"


def deployment_service_path(name: str) -> Path:
    return deployments_dir() / f"zippergen-{slug(name)}.service"


def deployment_launchd_path(name: str) -> Path:
    return deployments_dir() / f"io.zippergen.{slug(name)}.plist"


def deployment_secrets_path(name: str) -> Path:
    return deployments_dir() / f"{slug(name)}.secrets.json"


def deployment_environment_dir(name: str) -> Path:
    return zippergen_home() / "environments" / slug(name)


def deployment_bundles_dir(name: str) -> Path:
    return zippergen_home() / "apps" / slug(name)


def systemd_user_dir() -> Path:
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))).expanduser()
    return config_home / "systemd" / "user"


def systemd_unit_name(name: str) -> str:
    return f"zippergen-{slug(name)}.service"


def installed_systemd_service_path(name: str) -> Path:
    return systemd_user_dir() / systemd_unit_name(name)


def launchd_label(name: str) -> str:
    return f"io.zippergen.{slug(name)}"


def launch_agents_dir() -> Path:
    configured = os.environ.get("ZIPPERGEN_LAUNCH_AGENTS_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / "Library" / "LaunchAgents"


def installed_launchd_path(name: str) -> Path:
    return launch_agents_dir() / f"{launchd_label(name)}.plist"


def systemctl_command(*args: str) -> list[str]:
    systemctl = os.environ.get("ZIPPERGEN_SYSTEMCTL", "systemctl")
    return [systemctl, "--user", *args]


def run_systemctl(args: list[str], *, dry_run: bool = False) -> None:
    if dry_run:
        print(shlex.join(args))
        return
    try:
        subprocess.run(args, check=True)
    except FileNotFoundError as exc:
        raise SystemExit("systemctl was not found. Use `zippergen deploy run` directly or install systemd user services.") from exc
    except subprocess.CalledProcessError as exc:
        command = shlex.join(args)
        raise SystemExit(f"Command failed with exit code {exc.returncode}: {command}") from exc


def service_manager() -> str:
    configured = os.environ.get("ZIPPERGEN_SERVICE_MANAGER", "").strip().lower()
    if configured:
        if configured not in {"systemd", "launchd"}:
            raise SystemExit("ZIPPERGEN_SERVICE_MANAGER must be systemd or launchd.")
        return configured
    system = platform.system()
    if system == "Darwin":
        return "launchd"
    if system == "Linux":
        return "systemd"
    raise SystemExit(
        f"No supported deployment service manager for {system or 'this platform'}. "
        "Use `zippergen deploy run NAME` directly."
    )


def launchctl_domain() -> str:
    return f"gui/{os.getuid()}"


def launchctl_command(*args: str) -> list[str]:
    launchctl = os.environ.get("ZIPPERGEN_LAUNCHCTL", "launchctl")
    return [launchctl, *args]


def run_launchctl(
    args: list[str],
    *,
    dry_run: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess | None:
    if dry_run:
        print(shlex.join(args))
        return None
    try:
        return subprocess.run(args, check=check, capture_output=not check, text=True)
    except FileNotFoundError as exc:
        raise SystemExit(
            "launchctl was not found. Use `zippergen deploy run` directly or run on macOS."
        ) from exc
    except subprocess.CalledProcessError as exc:
        command = shlex.join(args)
        raise SystemExit(f"Command failed with exit code {exc.returncode}: {command}") from exc


def systemd_service_status(name: str) -> dict[str, object]:
    unit = systemd_unit_name(name)
    try:
        result = subprocess.run(
            systemctl_command(
                "show",
                unit,
                "--property=LoadState,ActiveState,SubState,ExecMainStatus,NRestarts",
            ),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        return {
            "manager": "systemd",
            "service": unit,
            "state": "unknown",
            "healthy": False,
            "detail": "systemctl was not found",
        }
    except subprocess.TimeoutExpired:
        return {
            "manager": "systemd",
            "service": unit,
            "state": "unknown",
            "healthy": False,
            "detail": "systemctl timed out",
        }
    values = {}
    for raw in (result.stdout or "").splitlines():
        key, separator, value = raw.partition("=")
        if separator:
            values[key] = value
    active = values.get("ActiveState", "")
    sub = values.get("SubState", "")
    try:
        exit_code = int(values.get("ExecMainStatus", "0"))
    except ValueError:
        exit_code = None
    try:
        restarts = int(values.get("NRestarts", "0"))
    except ValueError:
        restarts = 0
    if result.returncode == 0 and active == "active" and sub == "running":
        state = "running"
        healthy = True
        detail = f"{unit} is running"
    elif active == "activating" or restarts and exit_code not in {None, 0}:
        state = "restarting"
        healthy = False
        detail = (
            f"{unit} is not healthy; {active or 'unknown'}/{sub or 'unknown'}, "
            f"last exit code {exit_code}, {restarts} restart(s)"
        )
    elif (
        result.returncode == 0
        and active == "inactive"
        and exit_code == 0
    ):
        state = "completed"
        healthy = True
        detail = f"{unit} completed successfully"
    elif values.get("LoadState") == "not-found":
        state = "not-loaded"
        healthy = False
        detail = f"{unit} is not installed"
    else:
        state = "loaded" if result.returncode == 0 else "not-loaded"
        healthy = False
        detail = f"{unit} is {active or sub or 'not active'}"
    return {
        "manager": "systemd",
        "service": unit,
        "state": state,
        "healthy": healthy,
        "detail": detail,
        "last_exit_code": exit_code,
        "restarts": restarts,
        "active_state": active,
        "sub_state": sub,
    }


def launchd_service_status(name: str) -> dict[str, object]:
    service = f"{launchctl_domain()}/{launchd_label(name)}"
    try:
        result = subprocess.run(
            launchctl_command("print", service),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        return {
            "manager": "launchd",
            "service": service,
            "state": "unknown",
            "healthy": False,
            "detail": "launchctl was not found",
        }
    except subprocess.TimeoutExpired:
        return {
            "manager": "launchd",
            "service": service,
            "state": "unknown",
            "healthy": False,
            "detail": "launchctl timed out",
        }
    if result.returncode != 0:
        detail = (
            result.stderr or result.stdout or "not loaded"
        ).strip().splitlines()[0]
        return {
            "manager": "launchd",
            "service": service,
            "state": "not-loaded",
            "healthy": False,
            "detail": f"{service} is not loaded: {detail}",
        }

    output = result.stdout or ""
    values: dict[str, str] = {}
    for raw in output.splitlines():
        if "=" not in raw:
            continue
        key, value = raw.strip().split("=", 1)
        # `launchctl print` contains nested coalition blocks with repeated
        # `state` and `active count` keys. The service-level values appear
        # first and must not be overwritten by those nested records.
        values.setdefault(key.strip(), value.strip())
    state = values.get("state", "loaded")
    try:
        active_count = int(values.get("active count", "0"))
    except ValueError:
        active_count = 0
    try:
        runs = int(values.get("runs", "0"))
    except ValueError:
        runs = 0
    try:
        last_exit = int(values["last exit code"])
    except (KeyError, ValueError):
        last_exit = None

    if state == "running" or active_count > 0:
        health = "running"
        healthy = True
        detail = f"{service} is running"
    elif last_exit not in {None, 0}:
        health = "restarting"
        healthy = False
        detail = (
            f"{service} is loaded but not running; last exit code "
            f"{last_exit} after {runs} launch(es)"
        )
    elif last_exit == 0 and runs > 0:
        health = "completed"
        healthy = True
        detail = f"{service} completed successfully"
    else:
        health = "loaded"
        healthy = False
        detail = f"{service} is loaded but has no active process"
    return {
        "manager": "launchd",
        "service": service,
        "state": health,
        "healthy": healthy,
        "detail": detail,
        "active_count": active_count,
        "runs": runs,
        "last_exit_code": last_exit,
        "raw_state": state,
    }


def deployment_service_status(name: str) -> dict[str, object]:
    """Describe the supervised process, not merely service installation."""

    try:
        manager = service_manager()
    except SystemExit as exc:
        return {
            "manager": "unsupported",
            "service": name,
            "state": "unknown",
            "healthy": False,
            "detail": str(exc),
        }
    if manager == "launchd":
        return launchd_service_status(name)
    return systemd_service_status(name)
