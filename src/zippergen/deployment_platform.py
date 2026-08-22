"""Host paths and launchd/systemd operations for local deployments."""

from __future__ import annotations

import os
import platform
import re
import shlex
import subprocess
from collections.abc import Mapping
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
    """Legacy single-generation environment path."""

    return zippergen_home() / "environments" / slug(name)


def deployment_environment_releases_dir(name: str) -> Path:
    """Directory of immutable environment generations for a deployment."""

    return zippergen_home() / "environments" / ".releases" / slug(name)


def deployment_bundles_dir(name: str) -> Path:
    return zippergen_home() / "apps" / slug(name)


def deployment_secrets_dir(name: str) -> Path:
    """Directory of immutable secret generations for a deployment."""

    return deployments_dir() / ".secrets" / slug(name)


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
        raise SystemExit(
            "systemctl was not found. Install systemd user services, or use "
            "`zippergen run --durable` for foreground execution."
        ) from exc
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
        "Use `zippergen run --durable` for foreground execution."
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
            "launchctl was not found. Use `zippergen run --durable` for "
            "foreground execution, or run on macOS."
        ) from exc
    except subprocess.CalledProcessError as exc:
        command = shlex.join(args)
        raise SystemExit(f"Command failed with exit code {exc.returncode}: {command}") from exc


# Both service managers report the same small vocabulary: running, restarting,
# completed, loaded, not-loaded, unknown.
#
# Two different questions get asked about it, and they answer "unknown"
# differently. Giving them one name is how "already running" started being said
# about a service nobody could reach.
RUNNING_SERVICE_STATES = frozenset({"running", "restarting"})


def service_is_running(status: Mapping[str, object]) -> bool:
    """Is a process running now? Used to decide whether to act.

    "unknown" is not a yes: it means the service manager could not be asked, and
    reporting a service as already running on that basis stops a start that
    should have been attempted.
    """

    return str(status.get("state") or "unknown") in RUNNING_SERVICE_STATES


def service_may_be_attached(status: Mapping[str, object]) -> bool:
    """Might a process still hold this deployment's store? Used before destroying it.

    "unknown" is a yes here, for the same reason it is a no above: it means the
    service manager could not be asked, so a stopped service cannot be
    confirmed, and reset and compact destroy state.

    Enumerating the safe states instead is what made a deliberately stopped
    systemd unit -- which reports "loaded", not "completed" -- look unsafe, so
    reset refused the very state it had just created.
    """

    return (
        service_is_running(status)
        or str(status.get("state") or "unknown") == "unknown"
    )


class ServiceIsLiveError(Exception):
    """A command that changes durable state was refused because a service may hold it."""


# What every deploy verb needs from the service before it may run. ``None`` is
# the bare `zg deploy`.
#
# The point of a table is that a verb cannot be added without answering the
# question. Four separate hand-written guards is how four different rules ended
# up in four files, all of them wrong about a service that had just been
# stopped. A completeness test keeps this in step with the parser.
#
#   "any"      -- reads, reports, or manages the service itself
#   "stopped"  -- replaces or destroys durable state, so nothing may hold it
DEPLOY_SERVICE_REQUIREMENT: dict[str | None, str] = {
    None: "stopped",    # bare deploy replaces the active immutable release
    "start": "any",     # manages the service
    "stop": "any",      # manages the service
    "list": "any",
    "prune": "any",     # spans deployments; decides per deployment inside
    "status": "any",
    "logs": "any",
    "check": "any",
    "inspect": "any",
    "trace": "any",
    "tasks": "any",
    "approve": "any",
    "reset": "any",     # stops the service itself, then guards before archiving
    "compact": "stopped",
    "remove": "stopped",
}

# How the refusal reads, per verb, completing "Stop deployment NAME before ...".
DEPLOY_REQUIREMENT_VERB = {
    None: "updating it",
    "compact": "compacting it",
    "remove": "removing it",
}


def require_service_stopped(name: str, verb: str) -> None:
    """Refuse a command that would change durable state under a live service.

    Every such command asks the same question and refuses in the same words.
    Asking it here, once, is what keeps the answer from drifting.
    """

    status = deployment_service_status(name)
    if service_may_be_attached(status):
        raise ServiceIsLiveError(
            f"Stop deployment {name} before {verb}. "
            f"Current service state: {status.get('detail') or status.get('state')}"
        )


def enforce_deploy_requirement(action: str | None, name: str) -> None:
    """Apply the declared requirement for one deploy verb.

    An unlisted verb is a programming error, not a user error: the parser
    accepted something this table has never been asked about.
    """

    requirement = DEPLOY_SERVICE_REQUIREMENT.get(action, "__missing__")
    if requirement == "__missing__":
        raise AssertionError(
            f"deploy verb {action!r} has no entry in DEPLOY_SERVICE_REQUIREMENT"
        )
    if requirement == "stopped":
        require_service_stopped(name, DEPLOY_REQUIREMENT_VERB[action])


def unreachable_service_status(
    manager: str, service: str, detail: str
) -> dict[str, object]:
    """The service manager could not be asked, so nothing is known.

    "unknown" is not a guess about the process; it is the honest answer to a
    question nobody could put. Answering "not-loaded" instead is what let
    compact and remove destroy a store while a service still held it.
    """

    return {
        "manager": manager,
        "service": service,
        "state": "unknown",
        "healthy": False,
        "detail": detail,
    }


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
        return unreachable_service_status(
            "systemd", unit, "systemctl was not found"
        )
    except subprocess.TimeoutExpired:
        return unreachable_service_status("systemd", unit, "systemctl timed out")
    values = {}
    for raw in (result.stdout or "").splitlines():
        key, separator, value = raw.partition("=")
        if separator:
            values[key] = value
    # A unit that does not exist is still an answer: `systemctl show` succeeds
    # and reports LoadState=not-found. So a non-zero exit never means "no such
    # service" -- it means systemctl could not reach the user manager at all,
    # which is what an ssh session without a lingering login looks like.
    if result.returncode != 0 or not values:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        return unreachable_service_status(
            "systemd",
            unit,
            f"systemctl could not be asked: {detail[0] if detail else 'no output'}",
        )
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
    # Whether the unit exists at all is settled before anything is read into
    # how it ran. An uninstalled unit still reports inactive with exit status
    # 0, which is indistinguishable from a workflow that finished.
    if values.get("LoadState") == "not-found":
        state = "not-loaded"
        healthy = False
        detail = f"{unit} is not installed"
    elif active == "active" and sub == "running":
        state = "running"
        healthy = True
        # A process that keeps dying and coming back is running every time it
        # is looked at. The restart count is the only thing that distinguishes
        # it from one that has been up since it started, so it is always said.
        detail = f"{unit} is running"
        if restarts:
            detail += f" after {restarts} restart(s)"
    # Current systemd state determines whether the service is restarting.
    # NRestarts and ExecMainStatus are historical diagnostics: they remain
    # non-zero after a deliberate stop and must not make an inactive unit look
    # live forever.
    elif active == "activating":
        state = "restarting"
        healthy = False
        detail = (
            f"{unit} is not healthy; {active or 'unknown'}/{sub or 'unknown'}, "
            f"last exit code {exit_code}, {restarts} restart(s)"
        )
    elif active == "inactive" and exit_code == 0:
        state = "completed"
        healthy = True
        detail = f"{unit} completed successfully"
    elif active == "failed" or sub == "failed":
        state = "loaded"
        healthy = False
        detail = (
            f"{unit} failed; last exit code {exit_code}, "
            f"{restarts} restart(s). Inspect 'zippergen deploy logs'"
        )
    else:
        state = "loaded"
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


def launchctl_print(target: str) -> subprocess.CompletedProcess | None:
    """Ask launchctl about one target, or report that it could not be asked.

    ``None`` means launchctl itself did not run. A returned process may still
    carry a non-zero code: that is launchctl answering, not launchctl failing.
    """

    try:
        return subprocess.run(
            launchctl_command("print", target),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def launchd_service_status(name: str) -> dict[str, object]:
    service = f"{launchctl_domain()}/{launchd_label(name)}"
    result = launchctl_print(service)
    if result is None:
        return unreachable_service_status(
            "launchd", service, "launchctl could not be asked"
        )
    if result.returncode != 0:
        # `launchctl print` fails the same way for "there is no such service"
        # and for "I cannot reach that domain", and those two must not be
        # confused: only the first one makes it safe to destroy the store.
        # Printing the domain separates them -- it succeeds exactly when
        # launchctl could have found the service had it been loaded.
        domain = launchctl_print(launchctl_domain())
        detail = (
            result.stderr or result.stdout or "not loaded"
        ).strip().splitlines()[0]
        if domain is None or domain.returncode != 0:
            return unreachable_service_status(
                "launchd", service, f"launchctl could not be asked: {detail}"
            )
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
        # A process that keeps dying and coming back is running every time it
        # is looked at. The launch count is the only thing that distinguishes
        # it from one that has been up since it started, so it is always said.
        detail = f"{service} is running"
        if runs > 1:
            detail += f" after {runs} launch(es)"
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
