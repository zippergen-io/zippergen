"""Command-line execution, inspection, and local deployment entry point.

The per-role durable runtime lives in :mod:`zippergen.role_runner`; this module
parses ordinary CLI commands and coordinates the supporting subsystems.
"""
from __future__ import annotations

from zippergen.role_runner import (
    run_role,
)


# ---------------------------------------------------------------------------
# CLI:
#   `zippergen run MODULE_OR_PATH:WORKFLOW [--llm SPEC] [--store PATH] [--input k=v]`
#   `zippergen serve --workflow PATH --role NAME --store PATH [--input k=v]`
#       Legacy low-level per-role entry point; prefer `zippergen run`.
# ---------------------------------------------------------------------------
import argparse
import getpass
import hashlib
import json
import math
import os
import plistlib
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import venv
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from zippergen.deployment import (
    DeploymentField,
    DeploymentSetup,
    DeploymentSpec,
    deployment_spec_from_module,
)
from zippergen.deployment_platform import (
    deployment_bundles_dir as _deployment_bundles_dir,
    deployment_environment_dir as _deployment_environment_dir,
    deployment_launchd_path as _deployment_launchd_path,
    deployment_profile_path as _deployment_profile_path,
    deployment_script_path as _deployment_script_path,
    deployment_secrets_path as _deployment_secrets_path,
    deployment_service_path as _deployment_service_path,
    deployments_dir as _deployments_dir,
    installed_launchd_path as _installed_launchd_path,
    installed_systemd_service_path as _installed_systemd_service_path,
    launchctl_command as _launchctl_command,
    launchctl_domain as _launchctl_domain,
    launchd_label as _launchd_label,
    launchd_service_status as _launchd_service_status,
    run_launchctl as _run_launchctl,
    run_systemctl as _run_systemctl,
    service_manager as _service_manager,
    slug as _slug,
    systemctl_command as _systemctl_command,
    systemd_unit_name as _systemd_unit_name,
    zippergen_home as _zippergen_home,
)
from zippergen.connectors import connector_requirements_from_module
from zippergen.models import (
    apply_model_overrides,
    effective_llm_routes,
    normalize_llm_overrides,
    project_model_routing,
    selected_llm_specs,
)
from zippergen.assistant_configuration import (
    apply_assistant_overrides,
    normalize_assistant_overrides,
    project_assistant_routing,
)
from zippergen.view import DETAILS, ViewOptions, workflow_view_data
from zippergen.workflow_io import (
    RunConfig,
    _call_setup_hook,
    _looks_like_path,
    _workflow_lifelines,
    load_workflow,
    load_workflow_spec,
)
from zippergen.validation import (
    assistant_actions as _assistant_actions,
    validate_workflow as _validate_workflow,
)
from zippergen.semantic import (
    read_semantic_snapshot,
    render_semantic_diff,
    semantic_diff_models,
    semantic_snapshot,
    workflow_semantics,
)
from zippergen.syntax import Workflow
from zippergen.projection import project
from zippergen.store import (
    RECOVERY_COMPACTION_VERSION,
    TRACE_RETENTION_VERSION,
    complete_human_task,
    ensure_human_task_token,
    list_workflow_results,
    load_human_task,
    load_human_task_token,
    mark_human_task_token_used,
    open_store,
)

@dataclass(frozen=True)
class DoctorConfig:
    """Context passed to an optional module-level ``zippergen_doctor`` hook."""

    deployment_name: str
    profile: dict[str, object]
    workflow: Workflow
    module: ModuleType
    store_path: str
    log_path: str
    options: dict[str, object]
    services: str | None

    def option(self, name: str, default: object = None) -> object:
        return self.options.get(name, default)














def seed_env(conn, role: str, wf: Workflow, inputs: dict) -> dict:
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT payload FROM events WHERE kind='seed' AND sender=? ORDER BY rowid LIMIT 1",
            (role,),
        ).fetchone()
        if row is not None:
            conn.execute("ROLLBACK")
            return json.loads(row[0])
        conn.execute(
            "INSERT INTO events(sender,receiver,channel,kind,payload,causal_stamp) "
            "VALUES(?,?,?,?,?,?)",
            (role, None, None, "seed", json.dumps(inputs), None),
        )
        conn.execute("COMMIT")
        return dict(inputs)
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def _parse_inputs(pairs: list[str]) -> dict:
    out: dict = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        if not k or not _:
            raise SystemExit(f"Invalid --input {p!r}; expected name=value.")
        try:
            out[k] = json.loads(v)      # 7 -> int, "true" via JSON, '"s"' -> str
        except json.JSONDecodeError:
            out[k] = v                  # bare string fallback
    return out


def _parse_secret_inputs(
    pairs: list[str],
    *,
    option: str,
) -> dict[str, str]:
    """Split secret arguments without interpreting or rewriting their values."""

    values: dict[str, str] = {}
    for pair in pairs or []:
        name, separator, value = pair.partition("=")
        if not name or not separator:
            raise SystemExit(
                f"Invalid {option} {pair!r}; expected name=value."
            )
        values[name] = value
    return values


def _parse_input_json(text: str | None) -> dict:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--input-json must be valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise SystemExit("--input-json must be a JSON object.")
    return value


def _parse_options(pairs: list[str], *, services: str | None = None) -> dict:
    options = _parse_inputs(pairs)
    if services is not None:
        existing = options.get("services")
        if existing is not None and existing != services:
            raise SystemExit("Use either --services or --option services=..., not both.")
        options["services"] = services
    return options


def _parse_llm_idle_timeouts(pairs: list[str]) -> dict[str, float]:
    values = _parse_inputs(pairs)
    timeouts: dict[str, float] = {}
    for target, value in values.items():
        try:
            seconds = float(value)
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                "--llm-idle-timeout-for requires "
                "PARTICIPANT_OR_ACTION=SECONDS."
            ) from exc
        if not math.isfinite(seconds) or seconds < 0:
            raise SystemExit("LLM idle release seconds must be non-negative.")
        timeouts[str(target)] = seconds
    return timeouts


def _parse_llm_idle_timeouts_json(text: str | None) -> dict[str, float] | None:
    if text is None:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            "--llm-idle-timeouts-json must be a JSON object."
        ) from exc
    if not isinstance(value, dict):
        raise SystemExit("--llm-idle-timeouts-json must be a JSON object.")
    return _parse_llm_idle_timeouts(_jsonable_kv_pairs(value))


def _seed_inputs(wf: Workflow, inputs: dict) -> dict:
    """Var defaults from the workflow namespace, overlaid by caller inputs —
    parity with the in-process run() seeding (runtime.py:1014)."""
    from zippergen.syntax import Var
    env = {k: v.default for k, v in wf.ns.items() if isinstance(v, Var)}
    env.update(inputs)
    return env




def _default_store_path(workflow_spec: str, wf: Workflow) -> str:
    base = workflow_spec.split(":", 1)[0]
    if _looks_like_path(base):
        label = f"{Path(base).stem}.{wf.name}"
    else:
        label = f"{base}.{wf.name}"
    return str(Path.home() / ".zippergen" / "runs" / f"{_slug(label)}.sqlite")


def _ensure_store_parent(path: str) -> str:
    expanded = Path(path).expanduser()
    expanded.parent.mkdir(parents=True, exist_ok=True)
    return str(expanded)
































def _default_deployment_store_path(name: str) -> str:
    return str(_zippergen_home() / "runs" / f"{_slug(name)}.sqlite")


def _default_deployment_log_path(name: str) -> str:
    return str(_zippergen_home() / "logs" / f"{_slug(name)}.log")


def _load_deployment_profile(name: str) -> dict[str, object]:
    path = _deployment_profile_path(name)
    if not path.exists():
        raise SystemExit(f"Deployment profile not found: {name}")
    try:
        profile = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Deployment profile is not valid JSON: {path}") from exc
    if not isinstance(profile, dict):
        raise SystemExit(f"Deployment profile is not an object: {path}")
    return profile


def _load_deployment_secrets(profile: dict[str, object]) -> dict[str, str]:
    raw_path = profile.get("secrets_file")
    if not raw_path:
        return {}
    path = Path(str(raw_path)).expanduser()
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Deployment secrets file is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"Deployment secrets file is not an object: {path}")
    return {str(key): str(item) for key, item in value.items()}


def _write_deployment_secrets(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(values, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
    path.chmod(0o600)


def _deployment_environment(profile: dict[str, object]) -> dict[str, str]:
    raw = profile.get("environment") or {}
    if not isinstance(raw, dict):
        raise SystemExit("Deployment profile environment must be an object.")
    values = {str(key): str(value) for key, value in raw.items()}
    values.update(_load_deployment_secrets(profile))
    connectors = profile.get("connectors")
    if isinstance(connectors, dict):
        values["ZIPPERGEN_CONNECTORS_JSON"] = json.dumps(
            connectors,
            sort_keys=True,
        )
    return values


@contextmanager
def _profile_environment(profile: dict[str, object]):
    values = _deployment_environment(profile)
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield values
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _deployment_name_from_workflow(workflow_spec: str, wf: Workflow) -> str:
    base = workflow_spec.split(":", 1)[0]
    stem = Path(base).stem if _looks_like_path(base) else base.rsplit(".", 1)[-1]
    return _slug(f"{stem}-{wf.name}")


def _jsonable_kv_pairs(values: Mapping[str, object]) -> list[str]:
    return [f"{key}={json.dumps(value, default=str)}" for key, value in sorted(values.items())]


def _run_args_from_deployment(profile: dict[str, object]):
    timeout_raw = profile.get("timeout", 0.0)
    timeout = float(timeout_raw) if isinstance(timeout_raw, (int, float, str)) else 0.0
    return argparse.Namespace(
        workflow=str(profile["workflow"]),
        llm=profile.get("llm") or None,
        llm_for=_jsonable_kv_pairs(normalize_llm_overrides(profile.get("llms"))),
        llm_idle_timeout=profile.get("llm_idle_timeout"),
        llm_idle_timeout_for=_jsonable_kv_pairs(
            profile.get("llm_idle_timeouts") or {}  # type: ignore[arg-type]
        ),
        assistant=profile.get("assistant") or None,
        assistants=normalize_assistant_overrides(profile.get("assistants")),
        store=str(profile["store"]),
        input=[],
        input_json=json.dumps(profile.get("inputs") or {}, default=str),
        option=_jsonable_kv_pairs(profile.get("options") or {}),  # type: ignore[arg-type]
        services=profile.get("services") or None,
        timeout=timeout,
        execution=str(profile.get("execution", "sqlite")),
    )


def _deployment_command(name: str, *, python_executable: str | None = None) -> str:
    python = python_executable or sys.executable
    return f"{shlex.quote(python)} -m zippergen.serve deploy run {shlex.quote(_slug(name))}"


def _write_deployment_artifacts(profile: dict[str, object]) -> None:
    name = str(profile["name"])
    profile_path = _deployment_profile_path(name)
    script_path = _deployment_script_path(name)
    service_path = _deployment_service_path(name)
    launchd_path = _deployment_launchd_path(name)
    Path(str(profile["store"])).expanduser().parent.mkdir(parents=True, exist_ok=True)
    Path(str(profile["log"])).expanduser().parent.mkdir(parents=True, exist_ok=True)
    _deployments_dir().mkdir(parents=True, exist_ok=True)

    profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    script_path.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        f"cd {shlex.quote(str(profile['cwd']))}\n"
        f"exec env ZIPPERGEN_HOME={shlex.quote(str(_zippergen_home()))} "
        f"{_deployment_command(name, python_executable=str(profile.get('python') or sys.executable))}\n"
    )
    script_path.chmod(0o755)
    service_path.write_text(
        "[Unit]\n"
        f"Description=ZipperGen deployment {name}\n"
        "After=network-online.target\n\n"
        "[Service]\n"
        "Type=simple\n"
        f"WorkingDirectory={profile['cwd']}\n"
        f"ExecStart={script_path}\n"
        "Restart=on-failure\n"
        "RestartSec=10\n"
        f"StandardOutput=append:{profile['log']}\n"
        f"StandardError=append:{profile['log']}\n\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )
    launchd = {
        "Label": _launchd_label(name),
        "ProgramArguments": [str(script_path)],
        "WorkingDirectory": str(profile["cwd"]),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 10,
        "StandardOutPath": str(profile["log"]),
        "StandardErrorPath": str(profile["log"]),
        "ProcessType": "Background",
    }
    launchd_path.write_bytes(plistlib.dumps(launchd, sort_keys=True))


def _initialize_deployment_store(profile: dict[str, object]) -> bool:
    """Allocate one valid durable store for a deployment if it has none."""

    path = Path(str(profile["store"])).expanduser()
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        connection = open_store(str(path))
    except Exception as exc:
        raise SystemExit(
            f"Could not initialize deployment store {path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    connection.close()
    return True


def _install_systemd_unit(profile: dict[str, object], *, dry_run: bool = False) -> Path:
    name = str(profile["name"])
    _write_deployment_artifacts(profile)
    source = _deployment_service_path(name)
    target = _installed_systemd_service_path(name)
    if dry_run:
        print(f"Install systemd unit: {source} -> {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source.read_text())
    return target


def _install_launchd_agent(profile: dict[str, object], *, dry_run: bool = False) -> Path:
    name = str(profile["name"])
    _write_deployment_artifacts(profile)
    source = _deployment_launchd_path(name)
    target = _installed_launchd_path(name)
    if dry_run:
        print(f"Install launchd agent: {source} -> {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    return target














def _deployment_lifecycle_command(args, action: str) -> int:
    profile = _load_deployment_profile(args.name)
    name = str(profile["name"])
    if action in {"start", "restart"} and not args.dry_run:
        _initialize_deployment_store(profile)
    if (
        action in {"start", "restart"}
        and not args.dry_run
        and not getattr(args, "skip_readiness", False)
    ):
        checks = _doctor_checks(name, include_systemd=False)
        failures = [
            check for check in checks if check.get("status") == "fail"
        ]
        if failures:
            _print_doctor_summary(name, checks)
            print(
                f"Deployment {name} was not {action}ed because readiness "
                "checks found failures."
            )
            return 1
    manager = _service_manager()
    if manager == "systemd":
        unit = _systemd_unit_name(name)
        if action in {"start", "restart"}:
            target = _install_systemd_unit(profile, dry_run=args.dry_run)
            if not args.dry_run:
                print(f"Installed systemd unit: {target}")
            _run_systemctl(_systemctl_command("daemon-reload"), dry_run=args.dry_run)
            if action == "start" and args.enable:
                _run_systemctl(_systemctl_command("enable", unit), dry_run=args.dry_run)
        _run_systemctl(_systemctl_command(action, unit), dry_run=args.dry_run)
        service = unit
    else:
        label = _launchd_label(name)
        domain = _launchctl_domain()
        service = f"{domain}/{label}"
        if action in {"start", "restart"}:
            target = _install_launchd_agent(profile, dry_run=args.dry_run)
            if not args.dry_run:
                print(f"Installed launchd agent: {target}")
            # bootout makes both start and restart idempotent when the agent was
            # already loaded.  A missing prior agent is expected.
            _run_launchctl(
                _launchctl_command("bootout", service),
                dry_run=args.dry_run,
                check=False,
            )
            _run_launchctl(
                _launchctl_command("bootstrap", domain, str(target)),
                dry_run=args.dry_run,
            )
        else:
            _run_launchctl(_launchctl_command("bootout", service), dry_run=args.dry_run)
    if args.dry_run:
        return 0
    done = {"start": "Started", "stop": "Stopped", "restart": "Restarted"}[action]
    print(f"{done} deployment {name} ({service}).")
    return 0


def _logs_command(args) -> int:
    if args.tail <= 0:
        raise SystemExit("--tail must be greater than 0.")
    profile = _load_deployment_profile(args.name)
    log_path = Path(str(profile.get("log") or _default_deployment_log_path(args.name))).expanduser()
    if not log_path.exists():
        print(f"Log does not exist yet: {log_path}")
        return 0

    raw_offset = profile.get("log_generation_offset")
    initial_offset = (
        raw_offset
        if isinstance(raw_offset, int)
        and 0 <= raw_offset <= log_path.stat().st_size
        else 0
    )

    def visible_lines() -> list[str]:
        content = log_path.read_bytes()
        offset = initial_offset if initial_offset <= len(content) else 0
        return content[offset:].decode(errors="replace").splitlines()

    def print_tail() -> int:
        lines = visible_lines()
        for line in lines[-args.tail:]:
            print(line)
        return len(lines)

    seen = print_tail()
    if not args.follow:
        return 0
    while True:
        time.sleep(args.interval)
        lines = visible_lines()
        for line in lines[seen:]:
            print(line)
        seen = len(lines)


def _doctor_check(status: str, name: str, detail: str, **extra: object) -> dict[str, object]:
    return {"status": status, "name": name, "detail": detail, **extra}


_MODEL_PROVIDER_SECRETS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}


def _model_provider(value: object) -> str:
    provider = str(value or "mock").partition(":")[0].strip().lower()
    return {"claude": "anthropic", "ollama": "local"}.get(provider, provider)


def _required_model_provider_secrets(profile: dict[str, object]) -> set[str]:
    return {
        secret
        for model in selected_llm_specs(profile.get("llm"), profile.get("llms"))
        if (secret := _MODEL_PROVIDER_SECRETS.get(_model_provider(model)))
    }


def _path_parent_check(label: str, path: Path) -> dict[str, object]:
    parent = path.expanduser().parent
    if not parent.exists():
        return _doctor_check("fail", label, f"parent directory does not exist: {parent}")
    if not parent.is_dir():
        return _doctor_check("fail", label, f"parent path is not a directory: {parent}")
    if not os.access(parent, os.W_OK):
        return _doctor_check("fail", label, f"parent directory is not writable: {parent}")
    return _doctor_check("ok", label, f"parent directory is writable: {parent}")


def _profile_options(profile: dict[str, object]) -> dict[str, object]:
    return _profile_mapping(profile, "options")


def _profile_mapping(profile: dict[str, object], key: str) -> dict[str, object]:
    raw = profile.get(key)
    if not isinstance(raw, dict):
        return {}
    return {str(name): value for name, value in raw.items()}


def _systemd_active_check(name: str) -> dict[str, object]:
    unit = _systemd_unit_name(name)
    try:
        result = subprocess.run(
            _systemctl_command("is-active", unit),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        return _doctor_check("warn", "systemd active", "systemctl was not found")
    except subprocess.TimeoutExpired:
        return _doctor_check("warn", "systemd active", "systemctl timed out")

    state = (result.stdout or result.stderr or "").strip() or f"exit {result.returncode}"
    if result.returncode == 0:
        return _doctor_check("ok", "systemd active", f"{unit} is active", state=state)
    return _doctor_check("warn", "systemd active", f"{unit} is not active: {state}", state=state)






def _launchd_active_check(name: str) -> dict[str, object]:
    status = _launchd_service_status(name)
    if status["state"] in {"running", "completed"}:
        kind = "ok"
    elif status["state"] == "restarting":
        kind = "fail"
    else:
        kind = "warn"
    return _doctor_check(
        kind,
        "launchd process",
        str(status["detail"]),
        **{
            key: value
            for key, value in status.items()
            if key not in {"detail"}
        },
    )










def _call_doctor_hook(
    module: ModuleType,
    config: DoctorConfig,
) -> list[dict[str, object]]:
    hook = getattr(module, "zippergen_doctor", None)
    if hook is None:
        return []
    if not callable(hook):
        return [_doctor_check("fail", "workflow doctor hook", "zippergen_doctor exists but is not callable")]
    try:
        result = hook(config)
    except Exception as exc:
        return [_doctor_check("fail", "workflow doctor hook", f"{type(exc).__name__}: {exc}")]
    if result is None:
        return []
    if not isinstance(result, list):
        return [_doctor_check("fail", "workflow doctor hook", "zippergen_doctor must return a list or None")]
    checks: list[dict[str, object]] = []
    for item in result:
        if not isinstance(item, dict):
            checks.append(_doctor_check("fail", "workflow doctor hook", f"invalid check item: {item!r}"))
            continue
        status = str(item.get("status", "warn"))
        if status not in {"ok", "warn", "fail"}:
            status = "warn"
        checks.append({
            "status": status,
            "name": str(item.get("name", "workflow hook")),
            "detail": str(item.get("detail", "")),
            **{k: v for k, v in item.items() if k not in {"status", "name", "detail"}},
        })
    return checks


def _doctor_checks(
    name: str,
    *,
    include_systemd: bool = True,
    live_connectors: bool = True,
    check_store_integrity: bool = False,
) -> list[dict[str, object]]:
    profile_path = _deployment_profile_path(name)
    checks: list[dict[str, object]] = []
    profile = _load_deployment_profile(name)
    profile_name = str(profile.get("name") or name)
    checks.append(_doctor_check("ok", "profile", f"loaded {profile_path}", path=str(profile_path)))

    for field in ["workflow", "cwd", "store", "log"]:
        if profile.get(field):
            checks.append(_doctor_check("ok", f"profile.{field}", str(profile[field])))
        else:
            checks.append(_doctor_check("fail", f"profile.{field}", "required field is missing"))

    cwd = Path(str(profile.get("cwd") or ".")).expanduser()
    if cwd.exists() and cwd.is_dir():
        checks.append(_doctor_check("ok", "working directory", str(cwd)))
    else:
        checks.append(_doctor_check("fail", "working directory", f"directory does not exist: {cwd}"))

    store_path = Path(str(profile.get("store") or _default_deployment_store_path(profile_name))).expanduser()
    log_path = Path(str(profile.get("log") or _default_deployment_log_path(profile_name))).expanduser()
    checks.append(_path_parent_check("store path", store_path))
    checks.append(_path_parent_check("log path", log_path))

    if store_path.exists():
        try:
            status = _store_status(str(store_path))
        except Exception as exc:
            checks.append(_doctor_check("fail", "sqlite store", f"{type(exc).__name__}: {exc}"))
        else:
            checks.append(_doctor_check("ok", "sqlite store", str(status["summary"]), state=status["state"]))
            if check_store_integrity:
                from zippergen.storage_maintenance import (
                    check_store_integrity as inspect_integrity,
                )

                integrity = inspect_integrity(store_path)
                checks.append(
                    _doctor_check(
                        "ok" if integrity.ok else "fail",
                        "sqlite integrity",
                        integrity.detail,
                    )
                )
    else:
        checks.append(_doctor_check("warn", "sqlite store", f"store does not exist yet: {store_path}"))

    if log_path.exists():
        checks.append(_doctor_check("ok", "log file", str(log_path)))
    else:
        checks.append(_doctor_check("warn", "log file", f"log does not exist yet: {log_path}"))

    script_path = _deployment_script_path(profile_name)
    if script_path.exists() and os.access(script_path, os.X_OK):
        checks.append(_doctor_check("ok", "run script", str(script_path)))
    elif script_path.exists():
        checks.append(_doctor_check("fail", "run script", f"script is not executable: {script_path}"))
    else:
        checks.append(_doctor_check("fail", "run script", f"script does not exist: {script_path}"))

    template_path = _deployment_service_path(profile_name)
    if template_path.exists():
        checks.append(_doctor_check("ok", "systemd template", str(template_path)))
    else:
        checks.append(_doctor_check("warn", "systemd template", f"template does not exist: {template_path}"))

    launchd_template = _deployment_launchd_path(profile_name)
    if launchd_template.exists():
        checks.append(_doctor_check("ok", "launchd template", str(launchd_template)))
    else:
        checks.append(_doctor_check("warn", "launchd template", f"template does not exist: {launchd_template}"))

    try:
        manager = _service_manager()
    except SystemExit as exc:
        manager = ""
        checks.append(_doctor_check("warn", "service manager", str(exc)))
    else:
        checks.append(_doctor_check("ok", "service manager", manager))

    installed_path = (
        _installed_launchd_path(profile_name)
        if manager == "launchd"
        else _installed_systemd_service_path(profile_name)
    )
    if installed_path.exists():
        checks.append(_doctor_check("ok", f"{manager or 'service'} installed", str(installed_path)))
    else:
        checks.append(_doctor_check(
            "warn",
            f"{manager or 'service'} installed",
            f"service is not installed: {installed_path}",
        ))

    secrets_path = profile.get("secrets_file")
    raw_secret_names = profile.get("secret_names")
    secret_count = len(raw_secret_names) if isinstance(raw_secret_names, (list, tuple, set)) else 0
    if secrets_path:
        secret_file = Path(str(secrets_path)).expanduser()
        if not secret_file.exists():
            checks.append(_doctor_check("fail", "secrets file", f"file does not exist: {secret_file}"))
        elif secret_file.stat().st_mode & 0o077:
            checks.append(_doctor_check("fail", "secrets file", f"permissions are not private: {secret_file}"))
        else:
            checks.append(_doctor_check(
                "ok",
                "secrets file",
                f"{secret_count} secret(s) stored with private permissions",
            ))

    workflow = None
    module = None
    if cwd.exists() and cwd.is_dir() and profile.get("workflow"):
        old_cwd = Path.cwd()
        try:
            os.chdir(cwd)
            with _profile_environment(profile):
                workflow, module = load_workflow_spec(str(profile["workflow"]))
        except SystemExit as exc:
            checks.append(_doctor_check("fail", "workflow import", str(exc)))
        except Exception as exc:
            checks.append(_doctor_check("fail", "workflow import", f"{type(exc).__name__}: {exc}"))
        else:
            checks.append(_doctor_check("ok", "workflow import", f"{profile['workflow']} -> {workflow.name}"))
        finally:
            os.chdir(old_cwd)

    python_path = Path(str(profile.get("python") or sys.executable)).expanduser()
    if python_path.exists():
        checks.append(_doctor_check("ok", "python", str(python_path)))
    else:
        checks.append(_doctor_check("warn", "python", f"recorded Python does not exist: {python_path}"))

    deployment_spec = DeploymentSpec()
    if module is not None:
        try:
            deployment_spec = deployment_spec_from_module(module)
        except Exception as exc:
            checks.append(_doctor_check(
                "fail",
                "deployment declaration",
                f"{type(exc).__name__}: {exc}",
            ))
        else:
            checks.append(_doctor_check(
                "ok",
                "deployment declaration",
                f"{len(deployment_spec.fields)} field(s), "
                f"{len(deployment_spec.packages)} package(s), "
                f"{len(deployment_spec.setup)} setup step(s)",
            ))

    if workflow is not None:
        from zippergen.assistant_backends import check_cli_assistant
        from zippergen.assistant_configuration import (
            effective_assistant_routes,
            resolved_assistant_actions,
        )

        assistant_overrides = normalize_assistant_overrides(
            profile.get("assistants")
        )
        assistant_routing = effective_assistant_routes(
            workflow,
            str(profile["assistant"]) if profile.get("assistant") else None,
            assistant_overrides,
            module=module,
        )
        profile_assignments = {
            "default": "deployment" if assistant_routing.default_backend else "",
            "lifelines": {
                target: "deployment"
                for target in assistant_overrides
                if "." not in target
            },
            "actions": {
                target: "deployment"
                for target in assistant_overrides
                if "." in target
            },
        }
        resolved = resolved_assistant_actions(
            workflow,
            assistant_routing,
            module=module,
            assignments=profile_assignments,
        )
        missing_selection = [
            item.target for item in resolved if item.backend is None
        ]
        if missing_selection:
            checks.append(_doctor_check(
                "fail",
                "assistant backend",
                "no backend selected for: " + ", ".join(missing_selection),
            ))
        selected_assistants = {
            str(item.backend) for item in resolved if item.backend
        }
        for selected in sorted(selected_assistants):
            result = check_cli_assistant(selected)
            if result.supported:
                checks.append(_doctor_check(
                    "ok",
                    f"assistant {selected}",
                    result.detail,
                ))
            else:
                checks.append(_doctor_check(
                    "fail",
                    f"assistant {selected}",
                    result.detail,
                ))

    environment = _deployment_environment(profile)
    connector_bindings = profile.get("connectors") or {}
    if not isinstance(connector_bindings, dict):
        checks.append(_doctor_check(
            "fail",
            "connector bindings",
            "deployment connector bindings must be an object",
        ))
        connector_bindings = {}
    if module is not None:
        try:
            connector_requirements = connector_requirements_from_module(module)
        except (TypeError, ValueError) as exc:
            checks.append(_doctor_check(
                "fail",
                "connector requirements",
                str(exc),
            ))
            connector_requirements = ()
        for requirement in connector_requirements:
            raw_binding = connector_bindings.get(
                f"requirement:{requirement.name}"
            ) or connector_bindings.get(requirement.name)
            if raw_binding is None:
                checks.append(_doctor_check(
                    "fail" if requirement.required else "warn",
                    f"connector {requirement.name}",
                    "required binding is missing"
                    if requirement.required
                    else "optional binding is not configured",
                ))
                continue
            if not isinstance(raw_binding, dict):
                checks.append(_doctor_check(
                    "fail",
                    f"connector {requirement.name}",
                    "binding must be an object",
                ))
                continue
            kind = str(raw_binding.get("kind") or "")
            if kind != requirement.kind:
                checks.append(_doctor_check(
                    "fail",
                    f"connector {requirement.name}",
                    f"requires {requirement.kind}; deployment has {kind or 'none'}",
                ))
                continue
            if kind == "telegram":
                token_env = str(raw_binding.get("token_env") or "")
                token = environment.get(token_env)
                chat_id = str(raw_binding.get("chat_id") or "")
                if not token or not chat_id:
                    checks.append(_doctor_check(
                        "fail",
                        f"connector {requirement.name}",
                        "Telegram token or chat id is missing",
                    ))
                    continue
                if not live_connectors:
                    checks.append(_doctor_check(
                        "ok",
                        f"connector {requirement.name}",
                        (
                            f"Telegram chat {chat_id} is configured; live "
                            "availability was not checked"
                        ),
                    ))
                    continue
                try:
                    from zippergen.telegram_notify import TelegramBotClient

                    client = TelegramBotClient(token, timeout=5)
                    client.request("getMe")
                    client.request("getChat", chat_id=chat_id)
                except Exception as exc:
                    checks.append(_doctor_check(
                        "fail",
                        f"connector {requirement.name}",
                        f"Telegram is unavailable: {exc}",
                    ))
                else:
                    checks.append(_doctor_check(
                        "ok",
                        f"connector {requirement.name}",
                        f"Telegram chat {chat_id} is reachable",
                    ))
            elif kind == "google-sheets":
                credential_env = str(
                    raw_binding.get("credential_env") or ""
                )
                credential = environment.get(credential_env)
                spreadsheet_id = str(
                    raw_binding.get("spreadsheet_id") or ""
                )
                tab = str(raw_binding.get("tab") or "")
                if not credential or not spreadsheet_id or not tab:
                    checks.append(_doctor_check(
                        "fail",
                        f"connector {requirement.name}",
                        "Google credential, spreadsheet ID, or tab is missing",
                    ))
                    continue
                if not live_connectors:
                    checks.append(_doctor_check(
                        "ok",
                        f"connector {requirement.name}",
                        f"Google spreadsheet {spreadsheet_id}, tab {tab} is configured",
                    ))
                    continue
                try:
                    from zippergen.google_sheets import GoogleSheetsTable

                    info = GoogleSheetsTable(
                        requirement=requirement.name,
                        spreadsheet_id=spreadsheet_id,
                        tab=tab,
                        credential_json=credential,
                        access=requirement.access,
                    ).inspect()
                except Exception as exc:
                    checks.append(_doctor_check(
                        "fail",
                        f"connector {requirement.name}",
                        f"Google Sheets is unavailable: {exc}",
                    ))
                else:
                    checks.append(_doctor_check(
                        "ok",
                        f"connector {requirement.name}",
                        f"{info['title']}, tab {info['tab']} is reachable",
                    ))
            elif kind == "gmail":
                credential_env = str(
                    raw_binding.get("credential_env") or ""
                )
                credential = environment.get(credential_env)
                account = str(raw_binding.get("account") or "me")
                query = str(
                    raw_binding.get("query") or "is:unread in:inbox"
                )
                if not credential:
                    checks.append(_doctor_check(
                        "fail",
                        f"connector {requirement.name}",
                        "Google credential is missing",
                    ))
                    continue
                if not live_connectors:
                    checks.append(_doctor_check(
                        "ok",
                        f"connector {requirement.name}",
                        f"Gmail account {account}, query {query!r} is configured",
                    ))
                    continue
                try:
                    from zippergen.google_gmail import GmailMailbox

                    info = GmailMailbox(
                        requirement=requirement.name,
                        account=account,
                        query=query,
                        credential_json=credential,
                        access=requirement.access,
                    ).inspect()
                except Exception as exc:
                    checks.append(_doctor_check(
                        "fail",
                        f"connector {requirement.name}",
                        f"Gmail is unavailable: {exc}",
                    ))
                else:
                    checks.append(_doctor_check(
                        "ok",
                        f"connector {requirement.name}",
                        f"Gmail account {info['email']} is reachable",
                    ))
            else:
                checks.append(_doctor_check(
                    "warn",
                    f"connector {requirement.name}",
                    f"{kind} is bound but has no live readiness adapter yet",
                ))
    checked_human_configurations: set[str] = set()
    for route_name, raw_binding in connector_bindings.items():
        if (
            not isinstance(raw_binding, dict)
            or raw_binding.get("type") != "human"
        ):
            continue
        configuration = str(
            raw_binding.get("configuration") or route_name
        )
        if configuration in checked_human_configurations:
            continue
        checked_human_configurations.add(configuration)
        kind = str(raw_binding.get("kind") or "")
        if kind != "telegram":
            checks.append(_doctor_check(
                "fail",
                f"human connector {configuration}",
                f"unsupported human connector provider: {kind or 'none'}",
            ))
            continue
        token_env = str(raw_binding.get("token_env") or "")
        token = environment.get(token_env)
        chat_id = str(raw_binding.get("chat_id") or "")
        if not token or not chat_id:
            checks.append(_doctor_check(
                "fail",
                f"human connector {configuration}",
                "Telegram token or chat id is missing",
            ))
            continue
        if not live_connectors:
            checks.append(_doctor_check(
                "ok",
                f"human connector {configuration}",
                f"Telegram chat {chat_id} is configured",
            ))
            continue
        try:
            from zippergen.telegram_notify import TelegramBotClient

            client = TelegramBotClient(token, timeout=5)
            client.request("getMe")
            client.request("getChat", chat_id=chat_id)
        except Exception as exc:
            checks.append(_doctor_check(
                "fail",
                f"human connector {configuration}",
                f"Telegram is unavailable: {exc}",
            ))
        else:
            checks.append(_doctor_check(
                "ok",
                f"human connector {configuration}",
                f"Telegram chat {chat_id} is reachable",
            ))
    declared_values = {
        field.name: _profile_field_value(profile, field, environment)
        for field in deployment_spec.fields
    }
    declared_values["__llm_specs__"] = selected_llm_specs(
        profile.get("llm"),
        profile.get("llms"),
    )
    declared_values["__llm_field_names__"] = tuple(
        field.name for field in deployment_spec.fields if field.target == "llm"
    )
    for field in deployment_spec.fields:
        if (
            field.secret
            and field.required
            and _field_enabled(field, declared_values)
            and not environment.get(field.target_name)
        ):
            checks.append(_doctor_check(
                "fail",
                f"secret {field.target_name}",
                "required secret is not configured",
            ))

    declared_secret_names = {
        field.target_name for field in deployment_spec.fields if field.secret
    }
    for secret_name in sorted(_required_model_provider_secrets(profile)):
        if environment.get(secret_name):
            if secret_name not in declared_secret_names:
                checks.append(
                    _doctor_check(
                        "ok",
                        f"model credential {secret_name}",
                        "configured in private deployment storage",
                    )
                )
        else:
            checks.append(
                _doctor_check(
                    "fail",
                    f"model credential {secret_name}",
                    "required by a selected model but not configured",
                )
            )

    if python_path.exists():
        for package in deployment_spec.packages:
            if not package.import_name:
                continue
            result = subprocess.run(
                [
                    str(python_path),
                    "-c",
                    "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)",
                    package.import_name,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                checks.append(_doctor_check(
                    "ok",
                    f"package {package.requirement}",
                    f"import {package.import_name} is available",
                ))
            else:
                checks.append(_doctor_check(
                    "fail",
                    f"package {package.requirement}",
                    f"import {package.import_name} is not available in {python_path}",
                ))

    if include_systemd and installed_path.exists():
        if manager == "launchd":
            checks.append(_launchd_active_check(profile_name))
        elif manager == "systemd":
            checks.append(_systemd_active_check(profile_name))

    if workflow is not None and module is not None:
        config = DoctorConfig(
            deployment_name=profile_name,
            profile=profile,
            workflow=workflow,
            module=module,
            store_path=str(store_path),
            log_path=str(log_path),
            options=_profile_options(profile),
            services=str(profile.get("services") or "") or None,
        )
        with _profile_environment(profile):
            checks.extend(_call_doctor_hook(module, config))

    return checks


def _print_doctor(name: str, checks: list[dict[str, object]]) -> None:
    print(f"Doctor: {name}")
    for check in checks:
        status = str(check.get("status", "warn")).upper()
        print(f"{status:4} {check.get('name')}: {check.get('detail')}")
    counts = {
        status: sum(1 for check in checks if check.get("status") == status)
        for status in ("ok", "warn", "fail")
    }
    print(f"Summary: {counts['ok']} ok, {counts['warn']} warn, {counts['fail']} fail")


def _print_doctor_summary(
    name: str,
    checks: list[dict[str, object]],
) -> None:
    counts = {
        status: sum(1 for check in checks if check.get("status") == status)
        for status in ("ok", "warn", "fail")
    }
    print(
        f"Readiness {name}: {counts['ok']} ok, "
        f"{counts['warn']} warning(s), {counts['fail']} failure(s)"
    )
    for check in checks:
        if check.get("status") == "fail":
            print(f"FAIL {check.get('name')}: {check.get('detail')}")


def _doctor_command(args) -> int:
    checks = _doctor_checks(
        args.name,
        include_systemd=not args.no_systemd,
        check_store_integrity=True,
    )
    if args.json:
        print(json.dumps({"deployment": args.name, "checks": checks}, default=str, sort_keys=True))
    else:
        _print_doctor(args.name, checks)
    return 1 if any(check.get("status") == "fail" for check in checks) else 0


def _resolve_store_arg(args) -> str:
    deployment = getattr(args, "deployment", None)
    store = getattr(args, "store", None)
    if deployment and store:
        raise SystemExit("Use either a deployment name or --store, not both.")
    if deployment:
        profile = _load_deployment_profile(deployment)
        return str(profile["store"])
    if store:
        return str(store)
    raise SystemExit("Provide a deployment name or --store.")




def _safe_json_loads(value):
    if value is None:
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def _fmt_time(ts: float | None) -> str:
    if ts is None:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _store_status(store_path: str) -> dict[str, object]:
    path = Path(store_path).expanduser()
    if not path.exists():
        return {
            "store": str(path),
            "exists": False,
            "state": "missing",
            "summary": "store does not exist",
        }

    conn = open_store(str(path))
    try:
        event_count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        last_event = conn.execute(
            "SELECT rowid, sender, receiver, channel, kind, payload "
            "FROM events ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        human_rows = conn.execute(
            "SELECT task_id, role, action, status, created_at, updated_at "
            "FROM human_tasks ORDER BY updated_at DESC"
        ).fetchall()
        pending_tasks = [
            {
                "task_id": row[0],
                "role": row[1],
                "action": row[2],
                "created_at": row[4],
                "updated_at": row[5],
            }
            for row in human_rows
            if row[3] == "pending"
        ]
        done_task_count = sum(1 for row in human_rows if row[3] == "done")
        results = list_workflow_results(conn)
    finally:
        conn.close()

    if pending_tasks:
        state = "waiting"
        summary = f"waiting for {len(pending_tasks)} human task(s)"
    elif results:
        state = "done"
        summary = f"{len(results)} workflow result(s)"
    elif event_count:
        state = "active"
        summary = "events recorded; no result yet"
    else:
        state = "empty"
        summary = "store is initialized but empty"

    last_event_dict = None
    if last_event is not None:
        last_event_dict = {
            "rowid": last_event[0],
            "sender": last_event[1],
            "receiver": last_event[2],
            "channel": last_event[3],
            "kind": last_event[4],
            "payload": _safe_json_loads(last_event[5]),
        }

    return {
        "store": str(path),
        "exists": True,
        "state": state,
        "summary": summary,
        "event_count": event_count,
        "last_event": last_event_dict,
        "pending_human_tasks": pending_tasks,
        "done_human_task_count": done_task_count,
        "workflow_results": results,
    }


def _load_trace_events(
    store_path: str,
    *,
    after_rowid: int = 0,
    limit: int = 50,
) -> list[dict]:
    if limit <= 0:
        raise SystemExit("--tail must be greater than 0.")
    path = Path(store_path).expanduser()
    if not path.exists():
        raise SystemExit(f"Store does not exist: {store_path}")
    conn = open_store(str(path))
    try:
        rows = conn.execute(
            "SELECT rowid, sender, payload FROM events "
            "WHERE kind='trace' AND rowid>? "
            "ORDER BY rowid DESC LIMIT ?",
            (after_rowid, limit),
        ).fetchall()
    finally:
        conn.close()
    rows = list(reversed(rows))
    return [
        {
            "rowid": row[0],
            "role": row[1],
            "event": _safe_json_loads(row[2]),
        }
        for row in rows
    ]


def _load_human_tasks(
    store_path: str,
    *,
    status: str | None = "pending",
    limit: int | None = None,
    with_tokens: bool = False,
    token_channel: str = "cli",
) -> list[dict]:
    conn = open_store(str(Path(store_path).expanduser()))
    try:
        query = (
            "SELECT task_id FROM human_tasks "
            + ("WHERE status=? " if status is not None else "")
            + "ORDER BY updated_at DESC, task_id"
        )
        params: tuple[object, ...] = (status,) if status is not None else ()
        if limit is not None:
            query += " LIMIT ?"
            params = (*params, limit)
        rows = conn.execute(query, params).fetchall()
        tasks = []
        for row in rows:
            task = load_human_task(conn, row[0])
            if task is not None:
                if with_tokens:
                    record = ensure_human_task_token(conn, task["task_id"], channel=token_channel)
                    task["token"] = record["token"]
                    task["token_channel"] = record["channel"]
                tasks.append(task)
        return tasks
    finally:
        conn.close()


def _short_text(value: object, *, limit: int = 120) -> str:
    text = "" if value is None else str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "..."


def _short_json(value: object, *, limit: int = 160) -> str:
    text = json.dumps(value, default=str, sort_keys=True)
    return text if len(text) <= limit else text[: limit - 1] + "..."


def _trace_summary(role: str, event: object) -> str:
    if not isinstance(event, dict):
        return f"{role} {_short_json(event)}"

    event_type = event.get("type", "event")
    if event_type == "send":
        source = event.get("from", role)
        target = event.get("to", "?")
        channel = event.get("channel") or "-"
        return f"{role} send {source}->{target} {channel} values={_short_json(event.get('values') or [])}"
    if event_type == "recv":
        source = event.get("from", "?")
        target = event.get("to", role)
        channel = event.get("channel") or "-"
        return f"{role} recv {source}->{target} {channel} bindings={_short_json(event.get('bindings') or {})}"
    if event_type in {"act_start", "act"}:
        action = event.get("action", "?")
        kind = event.get("action_kind") or "action"
        payload_name = "outputs" if event_type == "act" else "inputs"
        payload = event.get(payload_name) or {}
        seq = event.get("seq")
        seq_text = f" seq={seq}" if seq is not None else ""
        return f"{role} {event_type} {kind} {action}{seq_text} {payload_name}={_short_json(payload)}"
    if event_type == "decision":
        kind = event.get("kind", "if")
        return f"{role} decision {kind} value={_short_json(event.get('value'))}"
    return f"{role} {event_type} {_short_json(event)}"


def _print_trace_events(events: list[dict]) -> None:
    print(f"Trace events: {len(events)}")
    for item in events:
        print(f"#{item['rowid']} {_trace_summary(item.get('role') or '-', item.get('event'))}")


def _print_tasks(tasks: list[dict], *, heading: str) -> None:
    print(f"{heading}: {len(tasks)}")
    for task in tasks:
        spec = task.get("spec") or {}
        rendered = spec.get("rendered") or {}
        output = spec.get("output")
        output_type = spec.get("output_type")
        print(
            f"{task['task_id']} {task['role']}.{task['action']} "
            f"{spec.get('kind', 'human')} -> {output}: {output_type} "
            f"status={task['status']} updated={_fmt_time(task['updated_at'])}"
        )
        if task.get("token"):
            print(f"  token[{task.get('token_channel', 'default')}]: {task['token']}")
        instruction = rendered.get("instruction")
        context = rendered.get("context")
        prefill = rendered.get("prefill")
        if instruction:
            print(f"  instruction: {_short_text(instruction)}")
        if context:
            print(f"  context: {_short_text(context)}")
        if prefill:
            print(f"  prefill: {_short_text(prefill)}")


def _notify_stdout_task(task: dict, *, store_path: str) -> None:
    spec = task.get("spec") or {}
    rendered = spec.get("rendered") or {}
    token = task.get("token")
    print("=" * 72)
    print(f"Human task: {task['task_id']}")
    if token:
        print(f"Token: {token}")
    print(f"Action: {task['role']}.{task['action']} ({spec.get('kind', 'human')})")
    instruction = rendered.get("instruction")
    context = rendered.get("context")
    prefill = rendered.get("prefill")
    if instruction:
        print("\nInstruction:")
        print(instruction)
    if context:
        print("\nContext:")
        print(context)
    if prefill:
        print("\nPrefill:")
        print(prefill)
    if token:
        print("\nApprove:")
        print(f"  zippergen approve --store {store_path} --token {token}")
        if spec.get("output_type") == "bool":
            print("Decline:")
            print(f"  zippergen approve --store {store_path} --token {token} --no")
        else:
            print("Respond:")
            print(f"  zippergen approve --store {store_path} --token {token} --value '<value>'")


def _print_status(status: dict[str, object]) -> None:
    print(f"Store: {status['store']}")
    print(f"State: {status['state']} ({status['summary']})")
    if not status.get("exists"):
        return

    print(f"Events: {status['event_count']}")
    last_event = status.get("last_event")
    if isinstance(last_event, dict):
        sender = last_event.get("sender")
        receiver = last_event.get("receiver") or "-"
        kind = last_event.get("kind")
        rowid = last_event.get("rowid")
        print(f"Last event: #{rowid} {kind} {sender}->{receiver}")

    tasks = status.get("pending_human_tasks")
    if isinstance(tasks, list):
        print(f"Pending human tasks: {len(tasks)}")
        for task in tasks[:10]:
            print(
                f"  {task['task_id']} {task['role']}.{task['action']} "
                f"updated {_fmt_time(task['updated_at'])}"
            )

    results = status.get("workflow_results")
    if isinstance(results, list):
        print(f"Workflow results: {len(results)}")
        for result in results[:10]:
            print(
                f"  {result['workflow']} = {json.dumps(result['value'], default=str)} "
                f"updated {_fmt_time(result['updated_at'])}"
            )


def _resolved_workflow_spec(args) -> str:
    """Use the project's workflow when the command line does not name one.

    Convention for the simple case, explicit configuration for the ambiguous
    one, in that order: an explicit argument, then `workflow_entry`, then the
    single workflow in the project if there is exactly one. `zg init` runs
    before any workflow exists, so a beginner should not have to record an
    entry by hand before the first `zg validate`.

    Inference is a convenience only. Nothing here writes to the manifest.
    """

    named = getattr(args, "workflow", None) or getattr(args, "target", None)
    if named:
        return str(named)
    from zippergen.workspace import Workspace, WorkspaceError

    workspace = Workspace(getattr(args, "project", None))
    try:
        resolved = workspace.resolve_workflow(str(named) if named else None)
    except WorkspaceError as exc:
        raise SystemExit(str(exc)) from exc
    return str(workspace.absolute_spec(resolved))


def _run_workflow_command(args) -> int:
    args.workflow = _resolved_workflow_spec(args)
    wf, module = load_workflow_spec(args.workflow)
    from zippergen.durable_runs import default_llm_spec
    from zippergen.workspace import Workspace

    workspace = Workspace(getattr(args, "project", None))
    inputs = _parse_input_json(args.input_json)
    inputs.update(_parse_inputs(args.input))
    # Apply declared defaults in every mode. Ask for a truly missing value only
    # in an interactive terminal, and otherwise fail before starting threads.
    from zippergen.durable_runs import collect_workflow_inputs

    inputs = collect_workflow_inputs(
        wf,
        module,
        inputs,
        interactive=(
            sys.stdin.isatty() and not getattr(args, "yes", False)
        ),
        input_func=input,
        output_func=print,
    )
    options = _parse_options(args.option, services=args.services)
    routing = project_model_routing(
        workspace,
        workspace.canonical_spec(args.workflow, cwd=workspace.root),
        wf,
        fallback_default=default_llm_spec(module),
    )
    routing = apply_model_overrides(
        routing,
        default_spec=args.llm,
        overrides=_parse_inputs(args.llm_for),
        idle_timeouts=_parse_llm_idle_timeouts(args.llm_idle_timeout_for),
    )
    selected_llm = routing.default_spec
    llms = routing.overrides
    llm_idle_timeouts = routing.idle_timeouts
    assistant_routing = project_assistant_routing(
        workspace,
        workspace.canonical_spec(args.workflow, cwd=workspace.root),
        wf,
        module=module,
    )
    assistant_routing = apply_assistant_overrides(
        assistant_routing,
        default_backend=args.assistant,
        overrides=normalize_assistant_overrides(
            getattr(args, "assistants", None)
        ),
        workflow=wf,
        module=module,
    )

    # Naming a store means wanting one; otherwise a plain run leaves nothing
    # behind, and --durable is the way to ask for a run you can come back to.
    execution = args.execution or ("sqlite" if args.store else "memory")
    store_path = args.store
    if execution == "sqlite":
        store_path = _ensure_store_parent(store_path or _default_store_path(args.workflow, wf))
        print(f"Store: {store_path}", file=sys.stderr)
    elif store_path:
        print(
            "--store is ignored because --execution memory was requested.",
            file=sys.stderr,
        )

    config = RunConfig(
        workflow_spec=args.workflow,
        workflow=wf,
        module=module,
        llm=selected_llm,
        llms=llms,
        assistant=assistant_routing.default_backend,
        assistants=assistant_routing.overrides,
        llm_idle_timeout=args.llm_idle_timeout,
        llm_idle_timeouts=llm_idle_timeouts,
        store_path=store_path,
        inputs=inputs,
        options=options,
        timeout=args.timeout,
        execution=execution,
    )
    _call_setup_hook(module, config)

    configure_kwargs = {
        "timeout": args.timeout,
        "llm_idle_timeout": args.llm_idle_timeout,
        "llm_idle_timeouts": llm_idle_timeouts,
        "execution": execution,
        "store_path": store_path,
        "assistant_root": str(Path.cwd()),
    }
    from zippergen.assistant_backends import make_cli_assistant_backend

    configure_kwargs["assistant_backend"] = (
        make_cli_assistant_backend(
            assistant_routing.default_backend,
            project_root=Path.cwd(),
            routes=assistant_routing.overrides,
        )
        if assistant_routing.overrides
        else make_cli_assistant_backend(
            assistant_routing.default_backend,
            project_root=Path.cwd(),
        )
    )
    if llms:
        wf.configure(
            effective_llm_routes(wf, selected_llm, llms),
            **configure_kwargs,
        )
    else:
        wf.configure(selected_llm, **configure_kwargs)

    result = wf(**inputs)
    print(json.dumps({"result": result}, default=str))
    return 0


def _durable_run_command(args) -> int:
    """Run with a recorded, resumable run.

    Same execution as a plain run; what differs is the bookkeeping. The run is
    registered in the project so `--resume` has something to continue, and any
    inputs the workflow needs but the command line did not supply are asked
    for.
    """

    from zippergen.durable_runs import run_durable
    from zippergen.workspace import Workspace

    inputs = _parse_input_json(args.input_json)
    inputs.update(_parse_inputs(args.input))
    run_durable(
        Workspace(getattr(args, "project", None)),
        workflow_spec=args.workflow,
        resume=args.resume,
        run_id=getattr(args, "run_id", None),
        provided_inputs=inputs,
        llm=args.llm,
        llms=normalize_llm_overrides(_parse_inputs(args.llm_for)),
        llm_idle_timeout=args.llm_idle_timeout,
        llm_idle_timeouts=_parse_llm_idle_timeouts(
            args.llm_idle_timeout_for
        ),
        assistant=args.assistant,
        assistants=normalize_assistant_overrides(
            getattr(args, "assistants", None)
        ),
        options=_parse_options(args.option, services=args.services),
        services=args.services,
        timeout=args.timeout,
        interactive=not args.yes and sys.stdin.isatty(),
        input_func=input,
        output_func=print,
        store_path=(
            _ensure_store_parent(args.store)
            if getattr(args, "store", None)
            else None
        ),
    )
    return 0


def _view_options_from_args(args) -> ViewOptions:
    agents = tuple(
        name.strip()
        for name in str(args.agents or "").split(",")
        if name.strip()
    )
    return ViewOptions(
        detail=args.detail,
        communications_only=args.communications,
        agent=args.agent,
        agents=agents,
    )


def _show_command(args) -> int:
    args.workflow = _resolved_workflow_spec(args)
    workflow, module = load_workflow_spec(args.workflow)
    options = _view_options_from_args(args)
    try:
        data = workflow_view_data(workflow, module, options=options)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.format == "json":
        print(json.dumps(data, indent=2, default=str))
    else:
        print(str(data["code"]))
    return 0




def _validate_command(args) -> int:
    args.workflow = _resolved_workflow_spec(args)
    workflow, module = load_workflow_spec(args.workflow)
    result = _validate_workflow(workflow, module)
    from zippergen.project_configuration import configuration_report
    from zippergen.workspace import Workspace, WorkspaceError

    workspace = Workspace(getattr(args, "project", None))
    try:
        project_workflow = workspace.resolve_workflow()
    except WorkspaceError:
        project_workflow = None
    validated_workflow = workspace.canonical_spec(args.workflow, cwd=Path.cwd())
    if project_workflow == validated_workflow:
        report = configuration_report(workspace, include_site_checks=False)
        raw_configuration_checks = report["checks"]
        assert isinstance(raw_configuration_checks, list)
        configuration_checks = [
            item
            for item in raw_configuration_checks
            if isinstance(item, dict)
            and str(item.get("name") or "") != "workflow"
        ]
        checks = result["checks"]
        assert isinstance(checks, list)
        checks.extend(configuration_checks)
        result["valid"] = not any(
            isinstance(item, dict) and item.get("status") == "fail"
            for item in checks
        )
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        verdict = "valid" if result["valid"] else "invalid"
        print(f"Workflow {workflow.name}: {verdict}")
        for check in result["checks"]:  # type: ignore[union-attr]
            print(f"{str(check['status']).upper():4} {check['name']}: {check['detail']}")
    return 0 if result["valid"] else 1


def _configuration_command(args) -> int:
    from zippergen.project_configuration import (
        configuration_report,
        render_configuration,
    )
    from zippergen.rendering import TerminalRenderer
    from zippergen.workspace import Workspace, WorkspaceError

    check = getattr(args, "config_action", None) == "check"
    try:
        report = configuration_report(
            Workspace(getattr(args, "project", None)),
            live=bool(getattr(args, "live", False)),
        )
    except WorkspaceError as exc:
        raise SystemExit(str(exc)) from exc
    if getattr(args, "json", False):
        print(json.dumps(report, indent=2, default=str))
    else:
        render_configuration(report, TerminalRenderer())
    return 1 if check and not report["valid"] else 0


def _guided_required_value(
    value: object,
    *,
    label: str,
    command: str,
    choices: tuple[str, ...] = (),
) -> str:
    """Return a required CLI value, prompting only in a human terminal."""

    entered = str(value or "").strip()
    if entered:
        return entered
    if not sys.stdin.isatty():
        raise SystemExit(
            f"{label} is required. Pass it explicitly with: {command}"
        )
    if choices:
        print(f"Available {label.casefold()}s: {', '.join(choices)}")
    try:
        entered = input(f"{label}: ").strip()
    except (EOFError, KeyboardInterrupt):
        raise SystemExit("Cancelled. Nothing was saved.") from None
    if not entered:
        raise SystemExit(f"{label} is required. Nothing was saved.")
    if choices and entered not in choices:
        raise SystemExit(
            f"Unknown {label.casefold()} {entered!r}. Available: "
            + ", ".join(choices)
        )
    return entered


def _project_choices(kind: str, project: str | None) -> tuple[str, ...]:
    """Return live project choices used by both completion and prompts."""

    from zippergen.completion import completion_candidates

    return tuple(completion_candidates(kind, project))


def _guided_model_spec(value: object) -> str:
    """Collect a compact model spec through clear provider-specific prompts."""

    entered = str(value or "").strip()
    if entered:
        return entered
    provider = _guided_required_value(
        None,
        label="Model provider",
        command="zg model configure NAME PROVIDER:MODEL",
        choices=("openai", "anthropic", "mistral", "local", "scripted"),
    )
    if provider == "scripted":
        path = _guided_required_value(
            None,
            label="Scripted response file",
            command="zg model configure NAME scripted:PATH",
        )
        return f"scripted:{path}"
    model = _guided_required_value(
        None,
        label="Model name",
        command=f"zg model configure NAME {provider}:MODEL",
    )
    return f"{provider}:{model}"


def _collect_model_credential(workspace, spec: str) -> None:
    """Offer to save a model API key in private site storage."""

    from zippergen.project_configuration import model_credential_name

    credential = model_credential_name(spec)
    if credential is None:
        return
    if workspace.development_credential(credential):
        print(f"Using {credential} already saved on this computer.")
        return
    if os.environ.get(credential):
        print(f"Found {credential} in the environment.")
        return
    if not sys.stdin.isatty():
        print(
            f"{credential} is not configured on this computer. Set it in the "
            "environment or run this command interactively to save it privately."
        )
        return
    try:
        value = getpass.getpass(
            f"{credential} (input hidden. Press Enter to configure later): "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        raise SystemExit(
            "Cancelled. The model configuration was saved without a credential."
        ) from None
    if not value:
        print(f"Skipped {credential}. Configure it before using this model.")
        return
    workspace.save_development_credential(credential, value)
    print(f"Saved {credential} in private storage on this computer.")


def _model_command(args) -> int:
    from zippergen.project_configuration import (
        assign_model,
        configuration_report,
        configuration_scope_valid,
        configure_model,
        render_model_configuration,
    )
    from zippergen.rendering import TerminalRenderer
    from zippergen.workspace import Workspace, WorkspaceError

    workspace = Workspace(getattr(args, "project", None))
    action = getattr(args, "model_action", None)
    try:
        if action == "configure":
            name = _guided_required_value(
                args.name,
                label="Model configuration name",
                command="zg model configure NAME PROVIDER:MODEL",
            )
            spec = _guided_model_spec(args.spec)
            value = configure_model(
                workspace,
                name,
                spec,
                idle_timeout=args.idle_timeout,
                base_url=args.base_url,
            )
            print(f"Saved model configuration {name}: {value['spec']}")
            _collect_model_credential(workspace, str(value["spec"]))
            return 0
        if action in {"assign", "unassign"}:
            target = _guided_required_value(
                args.target,
                label="Model assignment target",
                command=f"zg model {action} TARGET"
                + (" CONFIGURATION" if action == "assign" else ""),
                choices=_project_choices("model-targets", args.project),
            )
            configuration = None
            if action == "assign":
                configuration = _guided_required_value(
                    args.configuration,
                    label="Model configuration",
                    command="zg model assign TARGET CONFIGURATION",
                    choices=_project_choices(
                        "model-configurations", args.project
                    ),
                )
            assign_model(workspace, target, configuration)
            if configuration is None:
                print(f"Removed model assignment for {target}.")
            else:
                print(f"Assigned {target} to model configuration {configuration}.")
            return 0
        if action == "remove":
            name = _guided_required_value(
                args.name,
                label="Model configuration",
                command="zg model remove NAME",
                choices=_project_choices("model-configurations", args.project),
            )
            workspace.remove_model_configuration(name)
            print(f"Removed model configuration {name}.")
            return 0
        if action == "check" and args.name:
            if args.name not in workspace.model_configurations():
                raise WorkspaceError(
                    f"Model configuration does not exist: {args.name}."
                )
        report = configuration_report(
            workspace,
            live=bool(action == "check" and args.live),
            model_names=(args.name,) if action == "check" and args.name else (),
        )
    except WorkspaceError as exc:
        raise SystemExit(str(exc)) from exc
    render_model_configuration(report, TerminalRenderer())
    return (
        1
        if action == "check" and not configuration_scope_valid(report, "model")
        else 0
    )


def _assistant_command(args) -> int:
    """Show and manage named coding-assistant backend assignments."""

    from zippergen.project_configuration import (
        assign_assistant,
        configuration_report,
        configuration_scope_valid,
        configure_assistant,
        render_assistant_configuration,
    )
    from zippergen.rendering import TerminalRenderer
    from zippergen.workspace import Workspace, WorkspaceError

    workspace = Workspace(getattr(args, "project", None))
    action = getattr(args, "assistant_action", None)
    try:
        if action == "configure":
            name = _guided_required_value(
                args.name,
                label="Assistant configuration name",
                command="zg assistant configure NAME BACKEND",
            )
            backend = _guided_required_value(
                args.backend,
                label="Assistant backend",
                command="zg assistant configure NAME BACKEND",
                choices=("codex", "claude"),
            )
            value = configure_assistant(workspace, name, backend)
            print(
                f"Saved assistant configuration {name}: "
                f"{value['backend']}"
            )
            owner = (
                "Codex CLI"
                if value["backend"] == "codex"
                else "Claude Code"
            )
            print(
                f"Authentication remains managed by {owner}."
            )
            return 0
        if action in {"assign", "unassign"}:
            target = _guided_required_value(
                args.target,
                label="Assistant assignment target",
                command=f"zg assistant {action} TARGET"
                + (" CONFIGURATION" if action == "assign" else ""),
                choices=_project_choices("assistant-targets", args.project),
            )
            configuration = None
            if action == "assign":
                configuration = _guided_required_value(
                    args.configuration,
                    label="Assistant configuration",
                    command="zg assistant assign TARGET CONFIGURATION",
                    choices=_project_choices(
                        "assistant-configurations", args.project
                    ),
                )
            assign_assistant(workspace, target, configuration)
            if configuration is None:
                print(f"Removed assistant assignment for {target}.")
            else:
                print(
                    f"Assigned {target} to assistant configuration "
                    f"{configuration}."
                )
            return 0
        if action == "remove":
            name = _guided_required_value(
                args.name,
                label="Assistant configuration",
                command="zg assistant remove NAME",
                choices=_project_choices(
                    "assistant-configurations", args.project
                ),
            )
            workspace.remove_assistant_configuration(name)
            print(f"Removed assistant configuration {name}.")
            return 0
        if action == "check" and args.name:
            if args.name not in workspace.assistant_configurations():
                raise WorkspaceError(
                    f"Assistant configuration does not exist: {args.name}."
                )
        report = configuration_report(
            workspace,
            assistant_names=(
                (args.name,)
                if action == "check" and args.name
                else ()
            ),
        )
    except WorkspaceError as exc:
        raise SystemExit(str(exc)) from exc
    render_assistant_configuration(report, TerminalRenderer())
    return (
        1
        if action == "check"
        and not configuration_scope_valid(report, "assistant")
        else 0
    )


def _connector_management_command(args) -> int:
    from zippergen.project_configuration import (
        assign_connector,
        bind_connector,
        configuration_report,
        configuration_scope_valid,
        render_connector_configuration,
        unbind_connector,
    )
    from zippergen.rendering import TerminalRenderer
    from zippergen.workspace import Workspace, WorkspaceError

    workspace = Workspace(getattr(args, "project", None))
    action = getattr(args, "connector_action", None)
    try:
        if action == "bind":
            requirement = _guided_required_value(
                args.requirement,
                label="Connector requirement",
                command="zg connector bind REQUIREMENT CONFIGURATION",
                choices=_project_choices(
                    "connector-requirements", args.project
                ),
            )
            configuration = _guided_required_value(
                args.configuration,
                label="Connector configuration",
                command="zg connector bind REQUIREMENT CONFIGURATION",
                choices=_project_choices(
                    "connector-configurations", args.project
                ),
            )
            bind_connector(workspace, requirement, configuration)
            print(
                f"Bound connector requirement {requirement} to "
                f"configuration {configuration}."
            )
            return 0
        if action == "unassign":
            target = _guided_required_value(
                args.target,
                label="Human-action target",
                command="zg connector unassign TARGET",
                choices=_project_choices("connector-targets", args.project),
            )
            assign_connector(workspace, target, None)
            print(f"Removed connector assignment for {target}.")
            return 0
        if action == "unbind":
            requirement = _guided_required_value(
                args.requirement,
                label="Connector requirement",
                command="zg connector unbind REQUIREMENT",
                choices=_project_choices(
                    "connector-requirements", args.project
                ),
            )
            unbind_connector(workspace, requirement)
            print(f"Removed connector binding for {requirement}.")
            return 0
        if action == "remove":
            name = _guided_required_value(
                args.name,
                label="Connector configuration",
                command="zg connector remove NAME",
                choices=_project_choices(
                    "connector-configurations", args.project
                ),
            )
            workspace.remove_connector_configuration(name)
            print(f"Removed connector configuration {name}.")
            return 0
        if action == "check" and args.name:
            if args.name not in workspace.connector_configurations():
                raise WorkspaceError(
                    f"Connector configuration does not exist: {args.name}."
                )
        report = configuration_report(
            workspace,
            live=bool(action == "check" and args.live),
            connector_names=(args.name,)
            if action == "check" and args.name
            else (),
        )
    except WorkspaceError as exc:
        raise SystemExit(str(exc)) from exc
    render_connector_configuration(report, TerminalRenderer())
    return (
        1
        if action == "check"
        and not configuration_scope_valid(report, "connector")
        else 0
    )


SPECIFICATION_TEMPLATE = """# {name}

What should this workflow do? Describe the participants, what they exchange,
and in what order. Plain prose is fine — this is the statement of intent, not
a formal document.
"""


def _remove_command(args) -> int:
    """Delete a deployment. Its durable store survives unless --purge."""

    from zippergen.deployments import (
        DeploymentRemovalError,
        present_deployment_artifacts,
        remove_deployment_artifacts,
        unregister_deployment_service,
    )

    profile = _load_deployment_profile(args.name)
    try:
        artifacts = present_deployment_artifacts(args.name, profile)
    except DeploymentRemovalError as exc:
        raise SystemExit(str(exc)) from exc

    kept = [item.label for item in artifacts if item.retain and not args.purge]
    print(f"Deployment {args.name}: {len(artifacts)} artifact(s) to remove.")
    if args.purge:
        print("  --purge: nothing is kept, including the durable store.")
    elif kept:
        print(f"  Kept in the archive: {', '.join(kept)}.")

    if not args.yes:
        if not sys.stdin.isatty():
            raise SystemExit(
                f"Removing {args.name} is permanent. Re-run with --yes."
            )
        typed = input(f"Type {args.name} to remove this deployment: ").strip()
        if typed != args.name:
            print("The name did not match; nothing was changed.")
            return 1

    try:
        service = unregister_deployment_service(args.name)
        result = remove_deployment_artifacts(args.name, profile, purge=args.purge)
    except DeploymentRemovalError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Removed {result.name}: {result.artifact_count} artifact(s). {service}")
    if result.archive is not None:
        print(f"  Archive: {result.archive}")
    return 0


def _compact_command(args) -> int:
    """Reclaim space in a stopped deployment's durable store and logs."""

    from zippergen.deployments import DeploymentRemovalError, compact_deployment_logs
    from zippergen.storage_maintenance import compact_store

    profile = _load_deployment_profile(args.name)
    store = profile.get("store")
    if store:
        outcome = compact_store(str(store))
        print(f"Store {store}")
        print(f"  removed events: {outcome.deleted_total}")
        print(
            "  reclaimed bytes: "
            f"{max(0, outcome.before_bytes - outcome.after_bytes)}"
        )

    try:
        logs = compact_deployment_logs(
            args.name,
            profile,
            keep_archives=args.keep_archives,
        )
    except DeploymentRemovalError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Logs: rotated, keeping {args.keep_archives} archive(s).")
    if logs.removed_archives:
        print(
            f"  removed archives: {logs.removed_archives} "
            f"({logs.removed_archive_bytes} bytes)"
        )
    return 0


def _telegram_bot_token(workspace) -> None:
    """Read the bot token once, without echo, and keep it off this machine's argv."""

    import getpass

    if workspace.connector_provider_secret("telegram", "bot_token"):
        print("Using the Telegram bot token already saved on this computer.")
        return
    try:
        token = getpass.getpass("Telegram bot token (input hidden): ").strip()
    except (EOFError, KeyboardInterrupt):
        raise SystemExit("Cancelled; nothing was saved.") from None
    if not token:
        raise SystemExit("A bot token is required.")
    workspace.save_connector_provider_secret("telegram", "bot_token", token)
    print("Saved the Telegram bot token for this computer.")


def _telegram_chat_id(value: object) -> str:
    """Collect the portable chat identifier in the user's terminal."""

    return _guided_required_value(
        value,
        label="Telegram chat id",
        command="zg connector configure NAME telegram --chat-id CHAT_ID",
    )


def _project_connector_runtime(
    args,
    deployed_workflow: str | None = None,
    deployed_project: str | None = None,
) -> tuple[dict, dict[str, str]]:
    """Build deployment connector routing from the project's configuration.

    `deploy` reads the project directly, so connector configurations and
    assignments reach the deployment without an intermediate UI layer.

    `deployed_workflow` and `deployed_project` are what an existing deployment
    already records about itself. Reconfiguring by name must wire that
    workflow, from that project — not from whatever directory the shell
    happens to be standing in, which may be an unrelated project or none.
    """

    from zippergen.connector_wiring import connector_runtime
    from zippergen.workspace import Workspace, WorkspaceError

    workspace = Workspace(getattr(args, "project", None) or deployed_project)
    # Follow the workflow actually being deployed. Falling back to the
    # project's configured entry silently skipped connectors whenever a
    # workflow was named explicitly.
    target = getattr(args, "target", None)
    entry = target if target and (":" in target or _looks_like_path(target)) else None
    entry = entry or deployed_workflow
    try:
        canonical = workspace.resolve_workflow(entry)
    except WorkspaceError:
        return {}, {}
    workflow, module = load_workflow_spec(str(workspace.absolute_spec(canonical)))
    return connector_runtime(workspace, canonical, workflow, module)


def _connector_accept_google_command(args) -> int:
    """Save a Google authorization produced on a computer with a browser.

    `connector authorize google` runs the browser flow and prints an encoded
    result. This is the other half: it stores the credential and the scopes
    Google actually granted, so a machine with no browser — a server — can be
    authorized from your laptop.
    """

    import getpass

    from zippergen.google_auth import (
        GoogleConnectorError,
        decode_google_authorization,
        google_authorization_summary,
    )
    from zippergen.workspace import Workspace

    encoded = args.result
    if not encoded:
        try:
            encoded = getpass.getpass(
                "Paste the zg-google-v1... result (input hidden): "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("Cancelled; nothing was saved.") from None
    if not encoded:
        raise SystemExit("An encoded authorization result is required.")

    try:
        result = decode_google_authorization(encoded)
    except (GoogleConnectorError, ValueError) as exc:
        raise SystemExit(f"That is not a valid authorization result: {exc}") from exc

    granted, client, expiry = google_authorization_summary(result)
    workspace = Workspace(getattr(args, "project", None))
    workspace.save_connector_provider_secret(
        "google", "authorized_user_json", result.authorized_user_json
    )
    profile = workspace.connector_provider_profiles().get("google") or {}
    workspace.save_connector_provider_profile(
        "google",
        {
            **profile,
            "kind": "google",
            "granted_scopes": json.dumps(list(result.granted_scopes)),
            "client_id": client,
            "credential_expiry": expiry,
        },
    )
    print("Google authorization saved for this computer.")
    print(f"  Granted: {granted}")
    print(f"  Expiry:  {expiry}")
    return 0


def _connector_assign_command(args) -> int:
    """Route a participant's human actions to a saved connector.

    A `@human` action asks a person something. This says where that question
    is delivered. Without it the question appears in whichever terminal is
    running the workflow, which is right for development and wrong for a
    deployment nobody is watching.
    """

    from zippergen.project_configuration import assign_connector
    from zippergen.workspace import Workspace, WorkspaceError

    workspace = Workspace(getattr(args, "project", None))
    try:
        target = _guided_required_value(
            args.target,
            label="Human-action target",
            command="zg connector assign TARGET CONFIGURATION",
            choices=_project_choices("connector-targets", args.project),
        )
        configuration = _guided_required_value(
            args.configuration,
            label="Connector configuration",
            command="zg connector assign TARGET CONFIGURATION",
            choices=_project_choices(
                "connector-configurations", args.project
            ),
        )
        assign_connector(workspace, target, configuration)
    except WorkspaceError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"{target} will be asked through {configuration}.")
    return 0


def _connector_configure_command(args) -> int:
    """Save one named connector configuration for this project.

    Portable fields — which chat, which spreadsheet, which mailbox query — go
    in `zippergen.toml` and are committed. Credentials never do: the Telegram
    token is typed here, and Google uses `connector authorize`.
    """

    from zippergen.workspace import Workspace, WorkspaceError

    workspace = Workspace(args.project)
    name = _guided_required_value(
        args.name,
        label="Connector configuration name",
        command="zg connector configure NAME PROVIDER",
    )
    provider = _guided_required_value(
        args.connector_provider,
        label="Connector provider",
        command="zg connector configure NAME PROVIDER",
        choices=("telegram", "gmail", "google-sheets"),
    )

    if provider == "telegram":
        if args.spreadsheet_id or args.tab or args.account or args.query:
            raise SystemExit(
                "Telegram configuration uses --chat-id only."
            )
        chat_id = _telegram_chat_id(args.chat_id)
        _telegram_bot_token(workspace)
        values = {
            "provider": "telegram",
            "kind": "telegram",
            "chat_id": chat_id,
        }
        described = f"chat {chat_id}"
    elif provider == "google-sheets":
        if args.chat_id or args.account or args.query:
            raise SystemExit(
                "Google Sheets configuration uses --spreadsheet-id and --tab."
            )
        spreadsheet_id = _guided_required_value(
            args.spreadsheet_id,
            label="Google spreadsheet id",
            command=(
                "zg connector configure NAME google-sheets "
                "--spreadsheet-id ID --tab TAB"
            ),
        )
        tab = _guided_required_value(
            args.tab,
            label="Google Sheets tab",
            command=(
                "zg connector configure NAME google-sheets "
                "--spreadsheet-id ID --tab TAB"
            ),
        )
        values = {
            "provider": "google",
            "kind": "google-sheets",
            "spreadsheet_id": spreadsheet_id,
            "tab": tab,
        }
        described = f"tab {tab}"
    elif provider == "gmail":
        if args.chat_id or args.spreadsheet_id or args.tab:
            raise SystemExit(
                "Gmail configuration does not use --chat-id, "
                "--spreadsheet-id, or --tab."
            )
        values = {
            "provider": "google",
            "kind": "gmail",
            "account": str(args.account or "me"),
            "query": str(args.query or "is:unread in:inbox"),
        }
        described = f"query {values['query']!r}"
    else:  # pragma: no cover - argparse restricts the choices
        raise SystemExit(f"Unsupported connector provider {provider!r}.")

    try:
        workspace.save_connector_configuration(name, values)
    except WorkspaceError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Saved connector configuration {name} ({described}).")

    if provider in {"google-sheets", "gmail"} and not workspace.connector_provider_secret(
        "google", "authorized_user_json"
    ):
        print()
        print("Google is not authorized on this computer yet. Run:")
        print("    zippergen connector authorize google --scopes <scopes>")

    return 0


def _init_command(args) -> int:
    """Create a project and stop. No questionnaire, no configuration.

    Everything else — models, connectors, the workflow itself — is either an
    edit to `zippergen.toml` or something a coding agent writes once you say
    what you want.
    """

    from zippergen.skill import agents_md, claude_md
    from zippergen.workspace import Workspace

    root = Path(args.directory).expanduser().resolve() if args.directory else Path.cwd()
    root.mkdir(parents=True, exist_ok=True)
    workspace = Workspace(root)
    name = args.name or root.name

    created = workspace.manifest_path.exists()
    manifest = workspace.initialize_project(name=name)
    rows = [("zippergen.toml", "exists" if created else "created")]

    specification = root / str(manifest["specification_file"])
    if specification.exists():
        rows.append((specification.name, "exists"))
    else:
        specification.write_text(
            SPECIFICATION_TEMPLATE.format(name=manifest["name"]),
            encoding="utf-8",
        )
        rows.append((specification.name, "created"))

    agents = root / "AGENTS.md"
    if agents.exists():
        rows.append(("AGENTS.md", "exists; left alone"))
    else:
        agents.write_text(agents_md(str(manifest["name"])), encoding="utf-8")
        rows.append(("AGENTS.md", "created"))

    claude = root / "CLAUDE.md"
    if claude.exists():
        rows.append(("CLAUDE.md", "exists; left alone"))
    else:
        claude.write_text(claude_md(), encoding="utf-8")
        rows.append(("CLAUDE.md", "created"))

    print(f"ZipperGen project: {manifest['name']}")
    for filename, state in rows:
        print(f"  {filename:<18} {state}")
    if agents.exists() and "zippergen skill" not in agents.read_text(encoding="utf-8"):
        print()
        print("AGENTS.md was already here and was not changed. Add this so a")
        print("coding agent finds the ZipperGen instructions:")
        print()
        print("    Before editing workflow code, run `zippergen skill`")
        print("    and follow it completely.")
    if claude.exists() and "@AGENTS.md" not in claude.read_text(encoding="utf-8"):
        print()
        print("CLAUDE.md was already here and was not changed. Add this so")
        print("Claude Code reads the shared ZipperGen instructions:")
        print()
        print("    @AGENTS.md")
    return 0


def _skill_command(args) -> int:
    from zippergen.skill import SkillNotFound, agents_md, load_skill

    if args.agents_md:
        project = args.project or Path.cwd().name
        print(agents_md(project), end="")
        return 0
    try:
        skill = load_skill()
    except SkillNotFound as exc:
        raise SystemExit(str(exc)) from exc
    print(skill.render(include_references=not args.no_references), end="")
    return 0


def _semantic_input(spec: str) -> dict[str, object]:
    candidate = Path(spec).expanduser()
    if candidate.is_file() and candidate.suffix.lower() == ".json":
        try:
            return read_semantic_snapshot(json.loads(candidate.read_text()))
        except (json.JSONDecodeError, ValueError) as exc:
            raise SystemExit(f"Invalid semantic snapshot {candidate}: {exc}") from exc
    workflow, module = load_workflow_spec(spec)
    return workflow_semantics(workflow, module)


def _diff_command(args) -> int:
    # Saving a baseline and comparing against one are the same conversation,
    # so they are one command: --save writes, everything else compares.
    if args.save:
        spec = args.before or _resolved_workflow_spec(args)
        workflow, module = load_workflow_spec(spec)
        payload = json.dumps(
            semantic_snapshot(workflow, module), indent=2, default=str
        )
        if args.save == "-":
            print(payload)
            return 0
        output_path = Path(args.save).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n")
        print(f"Wrote semantic baseline to {output_path}")
        return 0

    if not args.before:
        raise SystemExit(
            "Give a baseline to compare against, or save one first with "
            "'zippergen diff --save PATH'."
        )
    # With one argument, compare a saved baseline against the project workflow.
    after = args.after or _resolved_workflow_spec(args)
    result = semantic_diff_models(
        _semantic_input(args.before),
        _semantic_input(after),
    )
    if args.format == "json":
        print(json.dumps(result, indent=2, default=str))
    else:
        print(render_semantic_diff(result))
    return 0


def _field_enabled(field: DeploymentField, values: dict[str, object]) -> bool:
    if not field.when:
        return True
    candidates = [values.get(field.when)]
    llm_field_names = values.get("__llm_field_names__")
    if (
        field.when == "llm"
        or (
            isinstance(llm_field_names, (list, tuple, set))
            and field.when in llm_field_names
        )
    ):
        configured = values.get("__llm_specs__")
        if isinstance(configured, (list, tuple, set)):
            candidates.extend(configured)
    if not field.when_values:
        return any(bool(current) for current in candidates)
    return any(
        str(current).startswith(expected[:-1])
        if expected.endswith("*")
        else str(current) == expected
        for current in candidates
        for expected in field.when_values
    )


def _profile_field_value(
    profile: dict[str, object],
    field: DeploymentField,
    secrets: dict[str, str],
) -> object:
    if field.target == "llm":
        return profile.get("llm")
    if field.target == "services":
        return profile.get("services")
    if field.target == "input":
        values = profile.get("inputs") or {}
        return values.get(field.target_name) if isinstance(values, dict) else None
    if field.target == "option":
        values = profile.get("options") or {}
        return values.get(field.target_name) if isinstance(values, dict) else None
    if field.secret:
        return secrets.get(field.target_name)
    values = profile.get("environment") or {}
    return values.get(field.target_name) if isinstance(values, dict) else None


def _parse_guided_value(raw: str, default: object) -> object:
    text = raw.strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


class _FormatValues(dict[str, object]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _resolve_field_default(
    field: DeploymentField,
    current: object,
    values: dict[str, object],
) -> object:
    if current != field.default or not isinstance(current, str) or "{" not in current:
        return current
    try:
        return current.format_map(_FormatValues(values))
    except (KeyError, ValueError):
        return current


def _display_default(value: object, *, secret: bool) -> str:
    if value is None or value == "":
        return ""
    if secret:
        return " [already set]"
    if isinstance(value, (dict, list, tuple, bool, int, float)):
        return f" [{json.dumps(value, default=str)}]"
    return f" [{value}]"


def _collect_deployment_fields(
    spec: DeploymentSpec,
    profile: dict[str, object],
    *,
    overrides: dict[str, object],
    interactive: bool,
) -> tuple[dict[str, object], dict[str, str]]:
    existing_secrets = _load_deployment_secrets(profile)
    values: dict[str, object] = {}
    secrets: dict[str, str] = dict(existing_secrets)

    for field in spec.fields:
        current = _profile_field_value(profile, field, existing_secrets)
        if current is None and field.target == "env":
            current = os.environ.get(field.target_name)
        if current is None:
            current = field.default
        if field.name in overrides:
            current = overrides[field.name]
        values[field.name] = current
    global_llm = next(
        (
            values.get(field.name)
            for field in spec.fields
            if field.target == "llm"
        ),
        profile.get("llm"),
    )
    values["__llm_specs__"] = selected_llm_specs(
        global_llm,
        profile.get("llms"),
    )
    values["__llm_field_names__"] = tuple(
        field.name for field in spec.fields if field.target == "llm"
    )

    for field in spec.fields:
        if not _field_enabled(field, values):
            continue
        current = _resolve_field_default(field, values.get(field.name), values)
        values[field.name] = current
        if interactive and field.name not in overrides:
            choices = f" ({'/'.join(field.choices)})" if field.choices else ""
            label = field.prompt + choices + _display_default(current, secret=field.secret) + ": "
            if field.secret:
                entered = getpass.getpass(label)
            else:
                entered = input(label)
            values[field.name] = _parse_guided_value(entered, current)
        value = values.get(field.name)
        if field.required and (value is None or str(value).strip() == ""):
            raise SystemExit(
                f"Deployment field {field.name!r} is required. "
                f"Use --set {field.name}=VALUE or run interactively."
            )
        if value is not None and field.choices and str(value) not in field.choices:
            raise SystemExit(
                f"Deployment field {field.name!r} must be one of "
                f"{', '.join(field.choices)}; got {value!r}."
            )
        if value is not None and value != "" and field.path_exists:
            path = Path(str(value)).expanduser()
            if not path.exists():
                raise SystemExit(f"Deployment field {field.name!r} points to a missing path: {path}")

    options: dict[str, object] = _profile_mapping(profile, "options")
    inputs: dict[str, object] = _profile_mapping(profile, "inputs")
    environment = {
        key: str(value)
        for key, value in _profile_mapping(profile, "environment").items()
    }
    for field in spec.fields:
        if not _field_enabled(field, values):
            continue
        value = values.get(field.name)
        if value is None:
            continue
        if field.target == "llm":
            profile["llm"] = value
        elif field.target == "services":
            profile["services"] = value
        elif field.target == "input":
            inputs[field.target_name] = value
        elif field.target == "option":
            options[field.target_name] = value
        elif field.secret:
            if str(value):
                secrets[field.target_name] = str(value)
        else:
            environment[field.target_name] = str(value)
    profile["options"] = options
    profile["inputs"] = inputs
    profile["environment"] = environment
    return values, secrets


def _deployment_python_path(environment_dir: Path) -> Path:
    if os.name == "nt":
        return environment_dir / "Scripts" / "python.exe"
    return environment_dir / "bin" / "python"


def _bundle_relative_path(source: Path, source_root: Path) -> Path:
    try:
        return source.relative_to(source_root)
    except ValueError:
        digest = hashlib.sha1(str(source).encode()).hexdigest()[:8]
        return Path("external") / f"{digest}-{source.name}"


def _copy_deployment_source(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(
            source,
            target,
            ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__", "*.pyc"),
        )
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)








def _bundle_deployment(
    profile: dict[str, object],
    spec: DeploymentSpec,
    workflow: Workflow,
) -> None:
    source_cwd = Path(str(profile.get("source_cwd") or profile["cwd"])).expanduser().resolve()
    source_workflow = str(profile.get("source_workflow") or profile["workflow"])
    module_ref, separator, workflow_name = source_workflow.partition(":")
    module_path = Path(module_ref).expanduser()
    if not module_path.is_absolute():
        module_path = source_cwd / module_path
    if not module_path.exists():
        # Importable modules are already versioned Python artifacts.  A later
        # packaging layer can snapshot their entire distribution; path-based
        # workflows get a concrete source bundle today.
        profile.setdefault("source_cwd", str(source_cwd))
        profile.setdefault("source_workflow", source_workflow)
        return

    version = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1_000_000_000:09d}"
    bundle_root = _deployment_bundles_dir(str(profile["name"])) / version
    bundle_root.mkdir(parents=True, exist_ok=False)

    sources = [module_path.resolve()]
    for declared in spec.files:
        path = Path(declared).expanduser()
        if not path.is_absolute():
            path = source_cwd / path
        path = path.resolve()
        if not path.exists():
            raise SystemExit(f"Declared deployment file does not exist: {path}")
        if path not in sources:
            sources.append(path)
    for action in _assistant_actions(workflow):
        if action.instructions_path is None:
            continue
        path = Path(action.instructions_path).resolve()
        try:
            path.relative_to(source_cwd)
        except ValueError as exc:
            raise SystemExit(
                f"Assistant instruction file for {action.name!r} is outside "
                f"the project root and cannot be bundled portably: {path}"
            ) from exc
        if path not in sources:
            sources.append(path)

    copied: dict[Path, Path] = {}
    for source in sources:
        relative = _bundle_relative_path(source, source_cwd)
        _copy_deployment_source(source, bundle_root / relative)
        copied[source] = relative

    workflow_relative = copied[module_path.resolve()]
    profile["source_cwd"] = str(source_cwd)
    profile["source_workflow"] = source_workflow
    profile["cwd"] = str(bundle_root)
    profile["workflow"] = str(workflow_relative) + (f":{workflow_name}" if separator else "")
    profile["bundle"] = str(bundle_root)
    profile["bundled_files"] = [str(path) for path in copied.values()]


def _zippergen_install_requirement(
    *,
    extras: tuple[str, ...] = (),
) -> str:
    project_root = Path(__file__).resolve().parents[2]
    if (project_root / "pyproject.toml").exists():
        requirement = str(project_root)
    else:
        try:
            from importlib.metadata import version

            requirement = f"zippergen=={version('zippergen')}"
        except Exception:
            requirement = "zippergen"
    if extras:
        name, separator, version_spec = requirement.partition("==")
        suffix = ",".join(sorted(set(extras)))
        return (
            f"{name}[{suffix}]=={version_spec}"
            if separator
            else f"{requirement}[{suffix}]"
        )
    return requirement


def _checkout_revision(project_root: Path) -> str | None:
    """Read a checkout revision without invoking Git or contacting a remote."""

    git_marker = project_root / ".git"
    try:
        if git_marker.is_file():
            marker = git_marker.read_text().strip()
            if not marker.startswith("gitdir:"):
                return None
            git_dir = Path(marker.partition(":")[2].strip())
            if not git_dir.is_absolute():
                git_dir = (project_root / git_dir).resolve()
        elif git_marker.is_dir():
            git_dir = git_marker
        else:
            return None
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head if len(head) >= 12 else None
        reference = head.partition(":")[2].strip()
        loose = git_dir / reference
        if loose.is_file():
            return loose.read_text().strip()
        packed = git_dir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text().splitlines():
                if not line or line.startswith(("#", "^")):
                    continue
                revision, _, name = line.partition(" ")
                if name == reference:
                    return revision
    except OSError:
        return None
    return None


def _zippergen_runtime_provenance() -> dict[str, str]:
    """Describe the ZipperGen source selected for a deployment environment."""

    from importlib.metadata import PackageNotFoundError, version

    project_root = Path(__file__).resolve().parents[2]
    try:
        installed_version = version("zippergen")
    except PackageNotFoundError:
        installed_version = "unknown"
    if (project_root / "pyproject.toml").is_file():
        provenance = {
            "kind": "source-checkout",
            "version": installed_version,
            "source": str(project_root),
        }
        digest = hashlib.sha256()
        package_root = project_root / "src" / "zippergen"
        source_files = (
            [
                path
                for path in package_root.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix != ".pyc"
            ]
            if package_root.is_dir()
            else []
        )
        for path in [project_root / "pyproject.toml", *sorted(source_files)]:
            try:
                relative = path.relative_to(project_root)
                digest.update(str(relative).encode())
                digest.update(b"\0")
                digest.update(path.read_bytes())
                digest.update(b"\0")
            except OSError:
                continue
        provenance["source_sha256"] = digest.hexdigest()
        revision = _checkout_revision(project_root)
        if revision:
            provenance["revision"] = revision
        return provenance
    return {
        "kind": "package",
        "version": installed_version,
        "source": "installed package",
    }


def _deployment_zippergen_extras(
    profile: dict[str, object],
) -> tuple[str, ...]:
    raw = profile.get("connectors") or {}
    bindings = raw if isinstance(raw, dict) else {}
    if any(
        isinstance(value, dict)
        and value.get("kind") in {"gmail", "google-sheets"}
        for value in bindings.values()
    ):
        return ("google",)
    return ()


def _prepare_deployment_environment(
    profile: dict[str, object],
    spec: DeploymentSpec,
    *,
    skip_install: bool,
) -> None:
    requirements = [package.requirement for package in spec.packages]
    profile["packages"] = requirements
    zippergen_extras = _deployment_zippergen_extras(profile)
    profile["zippergen_extras"] = list(zippergen_extras)
    profile["zippergen_runtime"] = _zippergen_runtime_provenance()
    profile["recovery_compaction_version"] = RECOVERY_COMPACTION_VERSION
    profile["trace_retention_version"] = TRACE_RETENTION_VERSION
    if skip_install:
        profile["python"] = str(profile.get("python") or sys.executable)
        return

    name = str(profile["name"])
    environment_dir = _deployment_environment_dir(name)
    environment_dir.parent.mkdir(parents=True, exist_ok=True)
    build_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{_slug(name)}-building-",
            dir=environment_dir.parent,
        )
    )
    build_python = _deployment_python_path(build_dir)
    uv = shutil.which("uv")
    phase = "creating the environment"
    print(f"Creating managed Python environment for {name}...")
    try:
        if uv is not None:
            subprocess.run(
                [
                    uv,
                    "venv",
                    "--python",
                    sys.executable,
                    str(build_dir),
                ],
                check=True,
            )
            install = [
                uv,
                "pip",
                "install",
                "--refresh-package",
                "zippergen",
                "--python",
                str(build_python),
                _zippergen_install_requirement(extras=zippergen_extras),
                *requirements,
            ]
        else:
            venv.EnvBuilder(with_pip=True).create(build_dir)
            install = [
                str(build_python),
                "-m",
                "pip",
                "install",
                _zippergen_install_requirement(extras=zippergen_extras),
                *requirements,
            ]
        phase = "installing deployment dependencies"
        print("Installing deployment dependencies...")
        subprocess.run(install, check=True)
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(build_dir, ignore_errors=True)
        outcome = (
            f"signal {-exc.returncode}"
            if exc.returncode < 0
            else f"exit code {exc.returncode}"
        )
        guidance = (
            " ZipperGen found uv and used it instead of ensurepip."
            if uv is not None
            else " Install uv and retry to avoid the standard-library "
            "ensurepip bootstrap."
        )
        raise SystemExit(
            f"Managed environment failed while {phase} ({outcome})."
            f"{guidance} The previous deployment environment, if any, was "
            "left unchanged."
        ) from None
    except (OSError, subprocess.SubprocessError) as exc:
        shutil.rmtree(build_dir, ignore_errors=True)
        raise SystemExit(
            f"Managed environment failed while {phase}: {exc}. The previous "
            "deployment environment, if any, was left unchanged."
        ) from None
    except KeyboardInterrupt:
        shutil.rmtree(build_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(build_dir, ignore_errors=True)
        raise SystemExit(
            f"Managed environment failed while {phase}: {exc}. The previous "
            "deployment environment, if any, was left unchanged."
        ) from None

    replaced: Path | None = None
    try:
        if environment_dir.exists() or environment_dir.is_symlink():
            replaced = environment_dir.with_name(
                f".{environment_dir.name}-replaced-"
                f"{time.strftime('%Y%m%d-%H%M%S')}-"
                f"{time.time_ns() % 1_000_000_000:09d}"
            )
            os.replace(environment_dir, replaced)
        os.replace(build_dir, environment_dir)
    except OSError as exc:
        if replaced is not None and replaced.exists():
            os.replace(replaced, environment_dir)
        shutil.rmtree(build_dir, ignore_errors=True)
        raise SystemExit(
            f"Managed environment was built but could not replace "
            f"{environment_dir}: {exc}. The previous environment was "
            "restored."
        ) from None
    if replaced is not None:
        shutil.rmtree(replaced, ignore_errors=True)

    python = _deployment_python_path(environment_dir)
    profile["python"] = str(python)
    profile["environment_dir"] = str(environment_dir)


def _setup_enabled(step: DeploymentSetup, values: dict[str, object]) -> bool:
    if not step.when:
        return True
    current = values.get(step.when)
    if not step.when_values:
        return bool(current)
    text = str(current)
    return any(
        text.startswith(expected[:-1]) if expected.endswith("*") else text == expected
        for expected in step.when_values
    )


def _run_deployment_setup(
    profile: dict[str, object],
    spec: DeploymentSpec,
    values: dict[str, object],
    *,
    skip_setup: bool,
) -> None:
    if skip_setup:
        return
    environment = {**os.environ, **_deployment_environment(profile)}
    replacements = {
        "python": str(profile.get("python") or sys.executable),
        "cwd": str(profile["cwd"]),
        "deployment": str(profile["name"]),
    }
    for step in spec.setup:
        if not _setup_enabled(step, values):
            continue
        if step.creates_env:
            created_path = environment.get(step.creates_env, "")
            if created_path and Path(created_path).expanduser().exists():
                print(f"Setup already complete: {step.description}")
                continue
        command = [part.format(**replacements) for part in step.command]
        print(f"Setup: {step.description}")
        try:
            subprocess.run(
                command,
                cwd=str(profile["cwd"]),
                env=environment,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                f"Deployment setup {step.name!r} failed with exit code {exc.returncode}."
            ) from exc


def _deployment_context(
    name: str,
    *,
    source: bool = False,
) -> tuple[dict[str, object], Workflow, ModuleType, DeploymentSpec]:
    profile = _load_deployment_profile(name)
    cwd_key = "source_cwd" if source and profile.get("source_cwd") else "cwd"
    workflow_key = "source_workflow" if source and profile.get("source_workflow") else "workflow"
    cwd = Path(str(profile.get(cwd_key) or ".")).expanduser()
    old_cwd = Path.cwd()
    try:
        os.chdir(cwd)
        with _profile_environment(profile):
            workflow, module = load_workflow_spec(str(profile[workflow_key]))
    finally:
        os.chdir(old_cwd)
    return profile, workflow, module, deployment_spec_from_module(module)


def _apply_deploy_arguments(
    profile: dict[str, object],
    args,
    spec: DeploymentSpec,
    workflow: Workflow,
) -> tuple[dict[str, object], dict[str, str]]:
    # Remove fields written by the retired browser viewer.
    profile.pop("ui", None)
    profile.pop("show_decisions", None)
    if args.llm is not None:
        profile["llm"] = args.llm
        # A global command-line model means every LLM action.  Keeping project
        # assignments here would make `--llm mock` unexpectedly call paid
        # providers for assigned participants.
        profile["llms"] = {}
        profile["llm_idle_timeouts"] = {}
    llms = normalize_llm_overrides(profile.get("llms"))
    for lifeline, model in normalize_llm_overrides(
        _parse_inputs(args.llm_for)
    ).items():
        existing_idle_timeouts = _profile_mapping(
            profile, "llm_idle_timeouts"
        )
        existing_idle_timeouts.pop(lifeline, None)
        profile["llm_idle_timeouts"] = existing_idle_timeouts
        if model.lower() in {"inherit", "default"}:
            llms.pop(lifeline, None)
        else:
            llms[lifeline] = model
    effective_llm_routes(workflow, str(profile.get("llm") or "mock"), llms)
    profile["llms"] = llms
    if args.llm_idle_timeout is not None:
        profile["llm_idle_timeout"] = args.llm_idle_timeout
    supplied_idle_timeouts = _parse_llm_idle_timeouts_json(
        args.llm_idle_timeouts_json
    )
    if supplied_idle_timeouts is None:
        repeated_idle_timeouts = _parse_llm_idle_timeouts(
            args.llm_idle_timeout_for
        )
        supplied_idle_timeouts = repeated_idle_timeouts or None
    if supplied_idle_timeouts is not None:
        profile["llm_idle_timeouts"] = supplied_idle_timeouts
    if args.assistant is not None:
        profile["assistant"] = args.assistant
    assistants = normalize_assistant_overrides(profile.get("assistants"))
    profile["assistants"] = assistants
    if args.services is not None:
        profile["services"] = args.services
    if args.timeout is not None:
        profile["timeout"] = args.timeout
    if args.store is not None:
        profile["store"] = _ensure_store_parent(args.store)
    if args.log is not None:
        profile["log"] = str(Path(args.log).expanduser())
    project_root = getattr(args, "project_root", None)
    if project_root:
        profile["project_root"] = str(Path(project_root).expanduser().resolve())
    project_alignment_json = getattr(args, "project_alignment_json", None)
    if project_alignment_json is not None:
        try:
            project_alignment = json.loads(project_alignment_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"Project alignment metadata is not valid JSON: {exc}"
            ) from exc
        if not isinstance(project_alignment, dict):
            raise SystemExit(
                "Project alignment metadata must be a JSON object."
            )
        profile["project_alignment"] = project_alignment
    provider_environment = _parse_inputs(getattr(args, "provider_env", []))
    unsupported_provider_environment = sorted(
        set(provider_environment) - {"OLLAMA_BASE_URL"}
    )
    if unsupported_provider_environment:
        raise SystemExit(
            "Unsupported model-provider environment setting: "
            + ", ".join(unsupported_provider_environment)
        )
    environment_values = _profile_mapping(profile, "environment")
    environment_values.update(
        {
            name: str(value)
            for name, value in provider_environment.items()
            if value is not None and str(value)
        }
    )
    profile["environment"] = environment_values
    # Wire the project's connectors unless the caller passed a snapshot.
    if getattr(args, "connectors_json", None) is None:
        try:
            snapshot, connector_environment = _project_connector_runtime(
                args,
                deployed_workflow=str(
                    profile.get("source_workflow") or profile.get("workflow") or ""
                )
                or None,
                deployed_project=str(profile.get("source_cwd") or "") or None,
            )
        except Exception as exc:  # surfaced as a clear refusal below
            from zippergen.connector_wiring import ConnectorWiringError

            if isinstance(exc, ConnectorWiringError):
                raise SystemExit(str(exc)) from exc
            raise
        if snapshot:
            profile["connectors"] = snapshot
            environment_values.update(connector_environment)
            profile["environment"] = environment_values

    connectors_json = getattr(args, "connectors_json", None)
    if connectors_json is not None:
        try:
            connector_bindings = json.loads(connectors_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"Connector bindings are not valid JSON: {exc}"
            ) from exc
        if not isinstance(connector_bindings, dict):
            raise SystemExit("Connector bindings must be a JSON object.")
        profile["connectors"] = connector_bindings

    input_arguments = _parse_input_json(args.input_json)
    input_arguments.update(_parse_inputs(args.input))
    inputs: dict[str, object] = _profile_mapping(profile, "inputs")
    inputs.update(input_arguments)
    profile["inputs"] = inputs
    option_arguments = _parse_inputs(args.option)
    options: dict[str, object] = _profile_mapping(profile, "options")
    options.update(option_arguments)
    profile["options"] = options

    overrides = _parse_inputs(args.set)
    for field in spec.fields:
        if field.target == "llm" and args.llm is not None:
            overrides[field.name] = args.llm
        elif field.target == "services" and args.services is not None:
            overrides[field.name] = args.services
        elif field.target == "option" and field.target_name in option_arguments:
            overrides.setdefault(field.name, option_arguments[field.target_name])
        elif field.target == "input" and field.target_name in input_arguments:
            overrides.setdefault(field.name, input_arguments[field.target_name])

    interactive = not args.yes and sys.stdin.isatty()
    values, secrets = _collect_deployment_fields(
        spec,
        profile,
        overrides=overrides,
        interactive=interactive,
    )
    provider_secrets = _parse_secret_inputs(
        getattr(args, "provider_secret", []),
        option="--provider-secret",
    )
    unsupported = sorted(
        set(provider_secrets) - set(_MODEL_PROVIDER_SECRETS.values())
    )
    if unsupported:
        raise SystemExit(
            "Unsupported model-provider secret: " + ", ".join(unsupported)
        )
    secrets.update(
        {
            name: str(value)
            for name, value in provider_secrets.items()
            if value is not None and str(value)
        }
    )
    connector_secrets = _parse_secret_inputs(
        getattr(args, "connector_secret", []),
        option="--connector-secret",
    )
    unsupported_connector_secrets = sorted(
        name
        for name in connector_secrets
        if not name.startswith("ZIPPERGEN_CONNECTOR_")
    )
    if unsupported_connector_secrets:
        raise SystemExit(
            "Unsupported connector secret: "
            + ", ".join(unsupported_connector_secrets)
        )
    secrets.update(
        {
            name: str(value)
            for name, value in connector_secrets.items()
            if value is not None and str(value)
        }
    )
    return values, secrets


def _finalize_guided_deployment(
    profile: dict[str, object],
    spec: DeploymentSpec,
    workflow: Workflow,
    values: dict[str, object],
    secrets: dict[str, str],
    args,
) -> int:
    name = str(profile["name"])
    secret_fields = [field for field in spec.fields if field.secret]
    if secret_fields or secrets:
        secrets_path = _deployment_secrets_path(name)
        _write_deployment_secrets(secrets_path, secrets)
        profile["secrets_file"] = str(secrets_path)
        profile["secret_names"] = sorted(secrets)
    profile["deployment_spec"] = spec.as_dict()
    profile["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    log_path = Path(
        str(profile.get("log") or _default_deployment_log_path(name))
    ).expanduser()
    try:
        profile["log_generation_offset"] = (
            log_path.stat().st_size if log_path.is_file() else 0
        )
    except OSError:
        profile["log_generation_offset"] = 0
    profile["deployment_generation_at"] = profile["updated_at"]

    if not args.no_bundle:
        _bundle_deployment(profile, spec, workflow)
    # Persist enough state to resume configuration even if dependency install
    # or an interactive OAuth step fails.
    _write_deployment_artifacts(profile)
    _initialize_deployment_store(profile)
    _prepare_deployment_environment(profile, spec, skip_install=args.no_install)
    _write_deployment_artifacts(profile)
    _run_deployment_setup(profile, spec, values, skip_setup=args.no_setup)
    _write_deployment_artifacts(profile)

    if not args.no_doctor:
        checks = _doctor_checks(name, include_systemd=False)
        if getattr(args, "concise", False):
            _print_doctor_summary(name, checks)
        else:
            _print_doctor(name, checks)
        if any(check.get("status") == "fail" for check in checks):
            print(f"Deployment {name} was configured but not started because doctor found failures.")
            return 1

    if not args.no_start:
        lifecycle_args = argparse.Namespace(
            name=name,
            enable=True,
            dry_run=False,
            skip_readiness=True,
        )
        _deployment_lifecycle_command(lifecycle_args, "start")

    print(f"Deployment: {name}")
    if not getattr(args, "concise", False):
        print(f"Status: zippergen deploy status {name}")
        print(f"Logs: zippergen deploy logs {name} --follow")
        print(f"Restart: zippergen deploy restart {name}")
    return 0


def _deployment_overview_command(args) -> int:
    """Show every deployment. `zg deploy` with no action, like `zg model`."""

    from zippergen.rendering import TerminalRenderer

    directory = _deployments_dir()
    profiles = sorted(directory.glob("*.json")) if directory.exists() else []
    if not profiles:
        print("No deployments yet. Create one with 'zippergen deploy create'.")
        return 0

    rows: list[tuple[object, ...]] = []
    for path in profiles:
        try:
            profile = _load_deployment_profile(path.stem)
        except SystemExit:
            rows.append((path.stem, "unreadable", "-", "-"))
            continue
        store = Path(str(profile.get("store") or ""))
        rows.append((
            path.stem,
            str(profile.get("source_workflow") or profile.get("workflow") or "-"),
            str(profile.get("llm") or "-"),
            "present" if store.exists() else "not created",
        ))
    TerminalRenderer().columns(
        "Deployments",
        ("Name", "Workflow", "Model", "Store"),
        rows,
    )
    print()
    print("Act on one with: zippergen deploy start|stop|logs|check NAME")
    return 0


def _deploy_command(args) -> int:
    if not args.target:
        args.target = _resolved_workflow_spec(args)
    existing_path = _deployment_profile_path(args.target)
    if existing_path.exists() and not _looks_like_path(args.target) and ":" not in args.target:
        profile, workflow, module, spec = _deployment_context(args.target, source=True)
        if args.name and _slug(args.name) != str(profile["name"]):
            raise SystemExit("--name cannot rename an existing deployment.")
    else:
        # A bare word that is neither a workflow nor a known deployment is
        # almost always a deployment name typed before it existed.
        if (
            not _looks_like_path(args.target)
            and ":" not in args.target
            and not args.target.isidentifier()
        ) or (
            not _looks_like_path(args.target)
            and ":" not in args.target
            and not Path(f"{args.target}.py").exists()
            and args.target not in sys.modules
        ):
            known = sorted(
                path.stem for path in _deployments_dir().glob("*.json")
            ) if _deployments_dir().exists() else []
            raise SystemExit(
                f"There is no deployment named {args.target!r}, and it is not "
                "a workflow spec. To create it, name the workflow and pass "
                f"--name {args.target}. "
                + (f"Existing deployments: {', '.join(known)}." if known
                   else "No deployments exist yet.")
            )
        workflow, module = load_workflow_spec(args.target)
        spec = deployment_spec_from_module(module)
        name = _slug(args.name or spec.name or _deployment_name_from_workflow(args.target, workflow))
        if _deployment_profile_path(name).exists():
            profile = _load_deployment_profile(name)
            # An explicit workflow target is a redeployment request. Preserve
            # the named deployment's operational configuration, but bundle
            # and validate the newly supplied source rather than silently
            # falling back to the previous bundle.
            profile["source_workflow"] = args.target
            profile["source_cwd"] = str(Path.cwd())
            profile["workflow"] = args.target
            profile["cwd"] = str(Path.cwd())
        else:
            profile = {
                "schema_version": 2,
                "name": name,
                "workflow": args.target,
                "cwd": str(Path.cwd()),
                "source_workflow": args.target,
                "source_cwd": str(Path.cwd()),
                "store": _default_deployment_store_path(name),
                "log": _default_deployment_log_path(name),
                "llm": None,
                "llms": {},
                "llm_idle_timeout": None,
                "llm_idle_timeouts": {},
                "assistant": None,
                "assistants": {},
                "services": None,
                "options": {},
                "inputs": {},
                "environment": {},
                "timeout": 0.0,
                "execution": "sqlite",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "python": sys.executable,
            }

    profile["schema_version"] = 2
    # A deployment snapshots the project's current routing.  Re-deploying is
    # what applies later edits to zippergen.toml; configuring an already
    # prepared deployment remains an explicit profile-only operation.
    from zippergen.durable_runs import default_llm_spec
    from zippergen.workspace import Workspace, WorkspaceError

    source_project = str(profile.get("source_cwd") or profile.get("cwd") or "")
    model_workspace = Workspace(
        getattr(args, "project", None) or source_project or None
    )
    source_workflow = str(
        profile.get("source_workflow") or profile.get("workflow") or args.target
    )
    try:
        model_workflow_spec = model_workspace.resolve_workflow(source_workflow)
    except WorkspaceError:
        model_workflow_spec = model_workspace.canonical_spec(
            source_workflow,
            cwd=model_workspace.root,
        )
    model_routing = project_model_routing(
        model_workspace,
        model_workflow_spec,
        workflow,
        fallback_default=default_llm_spec(module),
    )
    if model_workspace.has_model_assignment_profile(model_workflow_spec):
        profile["llm"] = model_routing.default_spec
        profile["llms"] = model_routing.overrides
        profile["llm_idle_timeout"] = None
        profile["llm_idle_timeouts"] = model_routing.idle_timeouts
    elif profile.get("llm") is None and not profile.get("llms"):
        # A new deployment without project routing uses the workflow default.
        # An existing deployment configured directly keeps its snapshot.
        profile["llm"] = model_routing.default_spec
    assistant_routing = project_assistant_routing(
        model_workspace,
        model_workflow_spec,
        workflow,
        module=module,
    )
    profile["assistant"] = assistant_routing.default_backend
    profile["assistants"] = assistant_routing.overrides
    if not args.yes and sys.stdin.isatty() and spec.description:
        print(spec.description)
        print()
    values, secrets = _apply_deploy_arguments(profile, args, spec, workflow)
    return _finalize_guided_deployment(
        profile, spec, workflow, values, secrets, args
    )


def _configure_deployment_command(args) -> int:
    profile, _workflow, _module, spec = _deployment_context(args.name)
    values, secrets = _apply_deploy_arguments(profile, args, spec, _workflow)
    rc = _finalize_guided_deployment(
        profile, spec, _workflow, values, secrets, args
    )
    if rc == 0 and args.restart and args.no_start:
        lifecycle_args = argparse.Namespace(
            name=args.name,
            enable=False,
            dry_run=False,
            skip_readiness=True,
        )
        return _deployment_lifecycle_command(lifecycle_args, "restart")
    return rc


def _run_deployment_command(args) -> int:
    profile = _load_deployment_profile(args.name)
    cwd = Path(str(profile.get("cwd") or ".")).expanduser()
    old_cwd = Path.cwd()
    try:
        os.chdir(cwd)
        with _profile_environment(profile):
            _start_deployment_connector_workers(profile)
            return _run_workflow_command(_run_args_from_deployment(profile))
    finally:
        os.chdir(old_cwd)


def _start_deployment_connector_workers(
    profile: dict[str, object],
) -> tuple[threading.Thread, ...]:
    """Start best-effort connector bridges owned by this service process."""

    raw = profile.get("connectors") or {}
    if not isinstance(raw, dict):
        return ()
    human_routes = [
        value
        for value in raw.values()
        if isinstance(value, dict) and value.get("type") == "human"
    ]
    telegram_routes = [
        value for value in human_routes
        if value.get("kind") == "telegram"
    ]
    if not telegram_routes:
        return ()

    grouped: dict[str, list[dict[str, object]]] = {}
    for route in telegram_routes:
        token_env = str(route.get("token_env") or "")
        if token_env:
            grouped.setdefault(token_env, []).append(route)

    threads: list[threading.Thread] = []
    for token_env, records in grouped.items():
        token = os.environ.get(token_env, "")
        if not token:
            raise SystemExit(
                f"Telegram connector credential is missing: {token_env}."
            )
        routes: dict[str, dict[str, object]] = {}
        assignments: dict[str, str] = {}
        for route in records:
            configuration = str(route.get("configuration") or "")
            target = str(route.get("target") or "")
            if not configuration or not target:
                continue
            routes[configuration] = route
            assignments[target] = configuration
        if not routes:
            continue
        from zippergen.telegram_notify import (
            TelegramBotClient,
            TelegramDeploymentNotifier,
        )

        notifier = TelegramDeploymentNotifier(
            store_path=str(profile["store"]),
            client=TelegramBotClient(token),
            routes=routes,
            assignments=assignments,
        )
        thread = threading.Thread(
            target=notifier.run_forever,
            name=f"connector-telegram-{len(threads) + 1}",
            daemon=True,
        )
        thread.start()
        threads.append(thread)
        print(
            "Telegram connector started for "
            + ", ".join(sorted(assignments)),
            file=sys.stderr,
            flush=True,
        )
    return tuple(threads)


def _status_command(args) -> int:
    status = _store_status(_resolve_store_arg(args))
    if args.json:
        print(json.dumps(status, default=str))
    else:
        _print_status(status)
    return 0


def _execution_age(updated_at: float | None) -> str:
    if updated_at is None:
        return "-"
    seconds = max(0, int(time.time() - updated_at))
    if seconds < 2:
        return "just now"
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    return f"{hours}h" if hours < 48 else f"{hours // 24}d"


def _inspection_context(args) -> tuple[Workflow, str, str]:
    """Resolve one deployment or durable run without mutating its state."""

    from zippergen.workspace import Workspace, WorkspaceError

    if args.deployment and args.store:
        raise SystemExit("Use either a deployment name or --store, not both.")
    if args.deployment:
        profile = _load_deployment_profile(args.deployment)
        workflow_spec = str(profile["workflow"])
        cwd = Path(str(profile.get("cwd") or ".")).expanduser()
        old_cwd = Path.cwd()
        try:
            os.chdir(cwd)
            workflow, _module = load_workflow_spec(workflow_spec)
        finally:
            os.chdir(old_cwd)
        return workflow, str(profile["store"]), f"deployment {args.deployment}"

    workspace = Workspace(args.project)
    if args.store:
        try:
            workflow_spec = workspace.resolve_workflow(args.workflow)
        except WorkspaceError as exc:
            raise SystemExit(str(exc)) from exc
        workflow, _module = load_workflow_spec(
            str(workspace.absolute_spec(workflow_spec))
        )
        return workflow, str(args.store), "explicit store"

    record = workspace.current_run()
    if record is None:
        raise SystemExit(
            "There is no current durable run. Name a deployment or use --store."
        )
    workflow_spec = str(record["workflow_spec"])
    workflow, _module = load_workflow_spec(
        str(workspace.absolute_spec(workflow_spec))
    )
    return workflow, str(record["store"]), f"run {record['run_id']}"


def _inspection_snapshot(workflow: Workflow, store: str, agent: str | None):
    from zippergen.execution_inspection import (
        default_focus,
        participant_positions,
        read_execution_states,
    )

    observed = read_execution_states(store)
    positions = participant_positions(workflow, observed)
    names = [position.participant for position in positions]
    if agent:
        focus = next(
            (name for name in names if name.casefold() == agent.casefold()),
            None,
        )
        if focus is None:
            raise SystemExit(
                f"Unknown participant {agent!r}. Available: "
                f"{', '.join(names) or 'none'}."
            )
    else:
        focus = default_focus(positions)
    return observed, positions, focus


def _render_inspection(
    workflow: Workflow,
    store: str,
    subject: str,
    observed,
    positions,
    focus: str | None,
    renderer,
    *,
    projection_indent: int = 0,
) -> None:
    from zippergen.execution_inspection import state_label
    from zippergen.view import render_local_projection_with_pointers

    renderer.section("Execution positions")
    renderer.emit(f"Subject: {subject}")
    renderer.emit(f"Store: {store}")
    if not observed:
        renderer.status(
            "warning",
            "No position data. The run may not have started yet.",
        )
    renderer.emit()
    renderer.columns(
        "Participants",
        ("Focus", "Participant", "State", "Current position", "Elapsed"),
        [
            (
                "▶" if position.participant == focus else "",
                position.participant,
                state_label(position.state),
                position.location,
                _execution_age(position.updated_at),
            )
            for position in positions
        ],
    )
    if focus is not None:
        selected = next(
            position for position in positions
            if position.participant == focus
        )
        renderer.emit()
        renderer.section(f"{focus} local projection")
        renderer.emit(
            render_local_projection_with_pointers(
                workflow,
                focus,
                selected.locators,
                indent=projection_indent,
            )
        )


def _inspect_command(args) -> int:
    from zippergen.execution_inspection import state_label
    from zippergen.rendering import TerminalRenderer

    if args.watch and args.json:
        raise SystemExit("Use either --watch or --json, not both.")
    if args.interval is not None and not args.watch:
        raise SystemExit("--interval requires --watch.")
    interval = 1.0 if args.interval is None else args.interval
    if not math.isfinite(interval) or interval <= 0:
        raise SystemExit("--interval must be a positive number.")

    workflow, store, subject = _inspection_context(args)
    observed, positions, focus = _inspection_snapshot(
        workflow,
        store,
        args.agent,
    )

    if args.json:
        print(json.dumps({
            "subject": subject,
            "store": store,
            "focus": focus,
            "positions": [
                {
                    "participant": position.participant,
                    "state": position.state,
                    "state_label": state_label(position.state),
                    "locators": [list(path) for path in position.locators],
                    "location": position.location,
                    "updated_at": position.updated_at,
                    "detail": position.detail,
                }
                for position in positions
            ],
        }, indent=2, default=str))
        return 0

    if args.watch:
        from zippergen.live_display import (
            live_display_available,
            watch_frames,
        )

        if not live_display_available():
            raise SystemExit("--watch requires an interactive terminal.")

        def frame(columns: int) -> str:
            live_observed, live_positions, live_focus = _inspection_snapshot(
                workflow,
                store,
                args.agent,
            )
            lines: list[str] = []
            live_renderer = TerminalRenderer(
                output=lines.append,
                color=False,
                columns=lambda: columns,
            )
            _render_inspection(
                workflow,
                store,
                subject,
                live_observed,
                live_positions,
                live_focus,
                live_renderer,
                projection_indent=1,
            )
            return "\n".join(lines)

        watch_frames(frame, interval=interval)
        print(f"Stopped watching {subject}. The execution was not interrupted.")
        return 0

    renderer = TerminalRenderer()
    _render_inspection(
        workflow,
        store,
        subject,
        observed,
        positions,
        focus,
        renderer,
    )
    return 0


def _trace_command(args) -> int:
    events = _load_trace_events(_resolve_store_arg(args), after_rowid=args.after, limit=args.tail)
    if args.json:
        print(json.dumps(events, default=str))
    else:
        _print_trace_events(events)
    return 0


def _tasks_command(args) -> int:
    if not Path(args.store).expanduser().exists():
        raise SystemExit(f"Store does not exist: {args.store}")
    status = None if args.all else "pending"
    tasks = _load_human_tasks(
        args.store,
        status=status,
        limit=args.limit,
        with_tokens=args.tokens,
        token_channel=args.channel,
    )
    if args.json:
        print(json.dumps(tasks, default=str))
    else:
        _print_tasks(tasks, heading="Human tasks" if args.all else "Pending human tasks")
    return 0


def _parse_bool_value(raw: object) -> bool:
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in {"true", "yes", "1", "y", "approve", "approved", "ack"}:
        return True
    if text in {"false", "no", "0", "n", "decline", "declined", "reject", "rejected"}:
        return False
    raise SystemExit(f"Cannot parse boolean human response: {raw!r}")


def _approve_result_from_args(task: dict, args) -> dict:
    spec = task.get("spec") or {}
    output = spec.get("output")
    if not output:
        raise SystemExit(f"Task {task['task_id']} has no output field in its spec.")
    output_type = spec.get("output_type", "str")

    if args.result_json is not None:
        try:
            result = json.loads(args.result_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--result-json must be valid JSON: {exc.msg}") from exc
        if not isinstance(result, dict):
            raise SystemExit("--result-json must be a JSON object.")
        if output not in result:
            raise SystemExit(f"--result-json must include output key {output!r}.")
        result[output] = _parse_bool_value(result[output]) if output_type == "bool" else str(result[output])
        return result

    if args.yes and args.no:
        raise SystemExit("Use only one of --yes or --no.")
    if args.value is not None and (args.yes or args.no):
        raise SystemExit("Use either --value or --yes/--no, not both.")

    if output_type == "bool":
        if args.no:
            value = False
        elif args.value is not None:
            value = _parse_bool_value(args.value)
        else:
            value = True
    else:
        if args.yes or args.no:
            raise SystemExit("--yes/--no can only be used for boolean human tasks.")
        if args.value is None:
            raise SystemExit(f"Task {task['task_id']} requires --value for output {output!r}.")
        value = args.value
    return {output: value}


def _approve_command(args) -> int:
    store_path = str(Path(args.store).expanduser())
    if not Path(store_path).exists():
        raise SystemExit(f"Store does not exist: {args.store}")
    conn = open_store(store_path)
    try:
        token_record = None
        task_id = args.task
        if args.token is not None:
            token_record = load_human_task_token(conn, args.token)
            if token_record is None:
                raise SystemExit(f"Human task token not found: {args.token}")
            task_id = token_record["task_id"]
        task = load_human_task(conn, task_id)
        if task is None:
            raise SystemExit(f"Human task not found: {task_id}")
        if task["status"] != "pending":
            raise SystemExit(f"Human task {task_id} is already {task['status']}.")
        result = _approve_result_from_args(task, args)
        conn.execute("BEGIN IMMEDIATE")
        try:
            task = complete_human_task(conn, task_id, result)
            if token_record is not None:
                mark_human_task_token_used(conn, token_record["token"])
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()

    if args.json:
        print(json.dumps(task, default=str))
    else:
        print(f"Completed human task {task['task_id']}: {json.dumps(task['result'], default=str)}")
    return 0


def _notify_stdout_command(args) -> int:
    store_path = str(Path(args.store).expanduser())
    if not Path(store_path).exists():
        raise SystemExit(f"Store does not exist: {args.store}")
    seen: set[str] = set()
    while True:
        tasks = _load_human_tasks(
            store_path,
            status="pending",
            limit=args.limit,
            with_tokens=True,
            token_channel=args.channel,
        )
        emitted = 0
        for task in tasks:
            token = task.get("token") or task["task_id"]
            if token in seen:
                continue
            _notify_stdout_task(task, store_path=store_path)
            seen.add(token)
            emitted += 1
        if not args.watch:
            if emitted == 0 and not args.quiet:
                print("No pending human tasks.")
            return 0
        time.sleep(args.interval)


def _notify_telegram_command(args) -> int:
    from zippergen.telegram_notify import (
        TelegramBotClient,
        TelegramNotifier,
        load_telegram_chat_id,
        load_telegram_token,
    )

    store_path = str(Path(args.store).expanduser())
    if not Path(store_path).exists():
        raise SystemExit(f"Store does not exist: {args.store}")
    token = load_telegram_token(args.bot_token)
    chat_id = load_telegram_chat_id(args.chat_id)
    if not chat_id:
        raise SystemExit("Telegram chat id is required. Set ZIPPERGEN_TELEGRAM_CHAT_ID or pass --chat-id.")
    client = TelegramBotClient(token)
    notifier = TelegramNotifier(
        store_path=store_path,
        client=client,
        chat_id=chat_id,
        channel=args.channel,
        limit=args.limit,
    )

    if not args.watch:
        sent = notifier.send_pending_once(resend=args.resend)
        processed = notifier.poll_updates_once(timeout=0)
        if not args.quiet:
            print(f"Telegram: sent {sent} task notification(s), processed {processed} update(s).")
        return 0

    if not args.quiet:
        print(f"Watching Telegram chat {chat_id} for store {store_path}.")
    while True:
        sent = notifier.send_pending_once(resend=args.resend)
        processed = notifier.poll_updates_once(timeout=args.poll_timeout)
        if not args.quiet and (sent or processed):
            print(f"Telegram: sent {sent} task notification(s), processed {processed} update(s).")
        time.sleep(args.interval)


def _add_guided_deployment_arguments(
    parser: argparse.ArgumentParser,
    *,
    configure: bool = False,
) -> None:
    parser.add_argument("--llm", metavar="SPEC", help="LLM spec stored in the deployment profile.")
    parser.add_argument(
        "--llm-for",
        action="append",
        default=[],
        metavar="PARTICIPANT_OR_ACTION=SPEC",
        help=(
            "Override the LLM for one participant or exact action; repeat as "
            "needed. Use PARTICIPANT=inherit to remove an existing override."
        ),
    )
    parser.add_argument("--llm-idle-timeout", type=float, help="Release a managed local LLM after this idle time.")
    parser.add_argument(
        "--llm-idle-timeout-for",
        action="append",
        default=[],
        metavar="PARTICIPANT_OR_ACTION=SECONDS",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--llm-idle-timeouts-json",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--assistant",
        choices=("codex", "claude"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--store", help="SQLite store path.")
    parser.add_argument("--log", help="Deployment log path.")
    parser.add_argument("--input", action="append", default=[], metavar="name=value", help="Workflow input value.")
    parser.add_argument("--input-json", help="Workflow inputs as a JSON object.")
    parser.add_argument("--option", action="append", default=[], metavar="name=value", help="Workflow setup option.")
    parser.add_argument("--set", action="append", default=[], metavar="field=value", help="Declared deployment field value.")
    parser.add_argument(
        "--provider-secret",
        action="append",
        default=[],
        metavar="ENV=value",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--project-root", help=argparse.SUPPRESS)
    parser.add_argument("--project-alignment-json", help=argparse.SUPPRESS)
    parser.add_argument("--connectors-json", help=argparse.SUPPRESS)
    parser.add_argument(
        "--connector-secret",
        action="append",
        default=[],
        metavar="ENV=value",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--provider-env",
        action="append",
        default=[],
        metavar="ENV=value",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--concise", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--services", choices=("fake", "live"), help="Workflow service mode.")
    parser.add_argument("--timeout", type=float, help="Workflow timeout; defaults to 0 (no deadline).")
    parser.add_argument("--yes", action="store_true", help="Accept defaults and existing environment values without prompting.")
    if configure:
        parser.add_argument("--install", dest="no_install", action="store_false", help="Update the managed Python environment.")
        parser.add_argument("--setup", dest="no_setup", action="store_false", help="Run declared one-time setup commands.")
    else:
        parser.add_argument("--no-install", action="store_true", help="Do not create/update the managed Python environment.")
        parser.add_argument("--no-setup", action="store_true", help="Skip declared one-time setup commands.")
    parser.add_argument("--no-doctor", action="store_true", help="Skip readiness checks.")


def _connector_authorize_google_command(args) -> int:
    """Authorize Google on the computer that owns the browser."""

    from zippergen.google_auth import (
        GoogleConnectorError,
        authorize_google_client_result,
        encode_google_authorization,
        google_authorization_summary,
        google_scope_names,
        google_scopes_cover,
        normalize_google_client_json,
        parse_google_scopes,
    )

    try:
        scopes = parse_google_scopes(args.scopes)
        entered = str(args.client or "").strip()
        if not entered:
            entered = input("Google OAuth Desktop app JSON path: ").strip()
        if not entered:
            raise GoogleConnectorError(
                "Select the OAuth Desktop app JSON downloaded from Google "
                "Cloud."
            )
        path = Path(entered).expanduser().resolve()
        if not path.is_file():
            raise GoogleConnectorError(
                f"Google OAuth desktop client JSON does not exist: {path}"
            )
        client_json = normalize_google_client_json(path.read_text())
        result = authorize_google_client_result(
            client_json,
            scopes=scopes,
        )
        if not google_scopes_cover(result.granted_scopes, scopes):
            missing = [
                name
                for scope, name in zip(
                    scopes, google_scope_names(scopes), strict=True
                )
                if not google_scopes_cover(
                    result.granted_scopes, (scope,)
                )
            ]
            raise GoogleConnectorError(
                "Google authorization did not grant: "
                + ", ".join(missing)
                + ". Run the command again and leave those permissions "
                "selected on Google's consent screen."
            )
        granted, client, expiry = google_authorization_summary(result)
        print("Google authorization completed.")
        print(f"Granted scopes: {granted}")
        print(f"OAuth client: {client}")
        print(f"Credential expiry: {expiry}")
        print(
            "On the other computer, run 'zippergen connector accept google' "
            "and paste the private result below. It contains a refresh token, "
            "so do not share it or save it in shell history."
        )
        print(encode_google_authorization(result))
        return 0
    except (OSError, GoogleConnectorError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


# Registered so they keep working, kept out of the help list. `__complete`
# backs shell completion, `serve` is the legacy per-role runner, and `notify`
# is an adapter a deployment runs for itself rather than something you type.
HIDDEN_COMMANDS = frozenset({"__complete", "serve", "notify"})


def _parse_cli_args(
    argv=None,
) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    """Build the real CLI parser and parse arguments without dispatching."""

    ap = argparse.ArgumentParser(prog="zippergen")
    sub = ap.add_subparsers(dest="cmd")

    config = sub.add_parser(
        "config",
        help="show or check the effective project configuration",
    )
    config.add_argument("config_action", nargs="?", choices=("check",))
    config.add_argument(
        "--live",
        action="store_true",
        help="Contact configured providers; only meaningful with 'check'.",
    )
    config.add_argument("--json", action="store_true", help="Print JSON.")
    config.add_argument("--project", help="Project root.")

    model = sub.add_parser(
        "model",
        help="show and manage model configurations and assignments",
        description=(
            "One pattern: configure NAME SPEC, then assign TARGET NAME. "
            "Run without an action to show everything. In a terminal, omit "
            "required values to be guided."
        ),
    )
    model_sub = model.add_subparsers(dest="model_action")
    model_configure = model_sub.add_parser(
        "configure",
        help="save one named model configuration",
    )
    model_configure.add_argument("name", nargs="?")
    model_configure.add_argument(
        "spec",
        nargs="?",
        help=(
            "Compact model spec such as openai:gpt-4o-mini or "
            "local:qwen2.5:14b. When omitted, ask for provider and model."
        ),
    )
    model_configure.add_argument("--base-url", help="Local provider base URL.")
    model_configure.add_argument(
        "--idle-timeout",
        type=float,
        help="Release a local model after this many idle seconds.",
    )
    model_configure.add_argument("--project", help="Project root.")
    model_assign = model_sub.add_parser(
        "assign",
        help="assign a named configuration to a participant or action",
    )
    model_assign.add_argument("target", nargs="?", help="default, Participant, or Participant.action.")
    model_assign.add_argument("configuration", nargs="?")
    model_assign.add_argument("--project", help="Project root.")
    model_unassign = model_sub.add_parser("unassign", help="remove one assignment")
    model_unassign.add_argument("target", nargs="?")
    model_unassign.add_argument("--project", help="Project root.")
    model_check = model_sub.add_parser(
        "check",
        help="check model configuration and credentials",
    )
    model_check.add_argument("name", nargs="?")
    model_check.add_argument("--live", action="store_true", help="Contact providers.")
    model_check.add_argument("--project", help="Project root.")
    model_remove = model_sub.add_parser("remove", help="remove an unused configuration")
    model_remove.add_argument("name", nargs="?")
    model_remove.add_argument("--project", help="Project root.")

    assistant = sub.add_parser(
        "assistant",
        help="show and manage coding-assistant configurations and assignments",
        description=(
            "One pattern: configure NAME BACKEND, then assign TARGET NAME. "
            "Run without an action to show everything. In a terminal, omit "
            "required values to be guided."
        ),
    )
    assistant_sub = assistant.add_subparsers(dest="assistant_action")
    assistant_configure = assistant_sub.add_parser(
        "configure",
        help="save one named Codex or Claude configuration",
    )
    assistant_configure.add_argument(
        "name",
        nargs="?",
        help="User-defined configuration name, such as coding-agent.",
    )
    assistant_configure.add_argument(
        "backend",
        nargs="?",
        choices=("codex", "claude"),
        help="Coding-assistant CLI selected by this configuration.",
    )
    assistant_configure.add_argument("--project", help="Project root.")
    assistant_assign = assistant_sub.add_parser(
        "assign",
        help="assign a configuration to a participant or exact action",
    )
    assistant_assign.add_argument(
        "target",
        nargs="?",
        help="default, Participant, or Participant.action.",
    )
    assistant_assign.add_argument("configuration", nargs="?")
    assistant_assign.add_argument("--project", help="Project root.")
    assistant_unassign = assistant_sub.add_parser(
        "unassign",
        help="remove one assistant assignment",
    )
    assistant_unassign.add_argument("target", nargs="?")
    assistant_unassign.add_argument("--project", help="Project root.")
    assistant_check = assistant_sub.add_parser(
        "check",
        help="check CLI availability and required safety options",
    )
    assistant_check.add_argument("name", nargs="?")
    assistant_check.add_argument("--project", help="Project root.")
    assistant_remove = assistant_sub.add_parser(
        "remove",
        help="remove an unused assistant configuration",
    )
    assistant_remove.add_argument("name", nargs="?")
    assistant_remove.add_argument("--project", help="Project root.")

    completion = sub.add_parser(
        "completion",
        help="print shell completion for zsh, bash, or fish",
    )
    completion.add_argument("shell", choices=("zsh", "bash", "fish"))
    internal_completion = sub.add_parser("__complete")
    internal_completion.add_argument("kind")
    internal_completion.add_argument("path", nargs="*")
    internal_completion.add_argument("--project")

    connector = sub.add_parser(
        "connector",
        help="show and manage connector configurations and assignments",
        description=(
            "One pattern: configure NAME PROVIDER, then assign TARGET NAME "
            "or bind REQUIREMENT NAME. Run without an action to show "
            "everything. In a terminal, omit required values to be guided."
        ),
    )
    connector_sub = connector.add_subparsers(
        dest="connector_action",
    )
    connector_configure = connector_sub.add_parser(
        "configure",
        help="save one named connector configuration",
    )
    connector_configure.add_argument(
        "name",
        nargs="?",
        help="Configuration name, such as approval-chat, inbox, or records.",
    )
    connector_configure.add_argument(
        "connector_provider",
        nargs="?",
        choices=("telegram", "gmail", "google-sheets"),
        help="Connector provider or service kind.",
    )
    connector_configure.add_argument(
        "--chat-id",
        help=(
            "Telegram chat id that receives approval messages. When omitted, "
            "ask in the terminal."
        ),
    )
    connector_configure.add_argument(
        "--project",
        help="Project root; defaults to discovery from the current directory.",
    )
    connector_configure.add_argument(
        "--spreadsheet-id",
        help="Spreadsheet id, the long value in its URL.",
    )
    connector_configure.add_argument(
        "--tab",
        help="Sheet tab name.",
    )
    connector_configure.add_argument(
        "--account",
        help="Mailbox to read. Default 'me', the authorized account.",
    )
    connector_configure.add_argument(
        "--query",
        help="Gmail search that selects the messages to handle.",
    )

    connector_accept = connector_sub.add_parser(
        "accept",
        help="save an authorization produced on another computer",
    )
    accept_sub = connector_accept.add_subparsers(
        dest="connector_provider",
        required=True,
    )
    accept_google = accept_sub.add_parser(
        "google",
        help="save the zg-google-v1... result printed by 'connector authorize'",
    )
    accept_google.add_argument(
        "result",
        nargs="?",
        help="The encoded result. Omit to be prompted without echo.",
    )
    accept_google.add_argument("--project", help="Project root.")

    connector_assign = connector_sub.add_parser(
        "assign",
        help="route a participant's human actions to a saved connector",
    )
    connector_assign.add_argument(
        "target",
        nargs="?",
        help="Participant, or Participant.action for a single action.",
    )
    connector_assign.add_argument(
        "configuration",
        nargs="?",
        help="Name of a saved connector configuration.",
    )
    connector_assign.add_argument("--project", help="Project root.")

    connector_unassign = connector_sub.add_parser(
        "unassign",
        help="remove one human-action assignment",
    )
    connector_unassign.add_argument("target", nargs="?")
    connector_unassign.add_argument("--project", help="Project root.")

    connector_bind = connector_sub.add_parser(
        "bind",
        help="bind a workflow service requirement to a named configuration",
    )
    connector_bind.add_argument("requirement", nargs="?")
    connector_bind.add_argument("configuration", nargs="?")
    connector_bind.add_argument("--project", help="Project root.")

    connector_unbind = connector_sub.add_parser(
        "unbind",
        help="remove one workflow requirement binding",
    )
    connector_unbind.add_argument("requirement", nargs="?")
    connector_unbind.add_argument("--project", help="Project root.")

    connector_check = connector_sub.add_parser(
        "check",
        help="check connector configuration and credentials",
    )
    connector_check.add_argument("name", nargs="?")
    connector_check.add_argument("--live", action="store_true", help="Contact providers.")
    connector_check.add_argument("--project", help="Project root.")

    connector_remove = connector_sub.add_parser(
        "remove",
        help="remove an unused connector configuration",
    )
    connector_remove.add_argument("name", nargs="?")
    connector_remove.add_argument("--project", help="Project root.")

    connector_authorize = connector_sub.add_parser(
        "authorize",
        help="create a private authorization handoff",
    )
    authorize_sub = connector_authorize.add_subparsers(
        dest="connector_provider",
        required=True,
    )
    authorize_google = authorize_sub.add_parser(
        "google",
        help="authorize Google with this computer's browser",
    )
    authorize_google.add_argument(
        "--scopes",
        required=True,
        help=(
            "Comma-separated scopes: gmail.readonly, gmail.modify, "
            "spreadsheets.readonly, spreadsheets"
        ),
    )
    authorize_google.add_argument(
        "--client",
        help="OAuth Desktop app JSON path; prompts when omitted.",
    )

    rn = sub.add_parser(
        "run",
        help="run a workflow; --durable records it so it can be resumed",
    )
    rn.add_argument(
        "workflow",
        nargs="?",
        help=(
            "Workflow spec: module:workflow or path.py:workflow. Optional with "
            "--resume, which continues a recorded run."
        ),
    )
    rn.add_argument("--llm", metavar="SPEC", help="LLM spec: mock, openai:gpt-4o, ollama:qwen2.5:7b, ...")
    rn.add_argument(
        "--llm-for",
        action="append",
        default=[],
        metavar="PARTICIPANT_OR_ACTION=SPEC",
        help="Override the LLM for one participant or exact action; repeat as needed.",
    )
    rn.add_argument("--llm-idle-timeout", type=float, help="Release a managed local LLM after this many idle seconds.")
    rn.add_argument(
        "--llm-idle-timeout-for",
        action="append",
        default=[],
        metavar="PARTICIPANT_OR_ACTION=SECONDS",
        help=argparse.SUPPRESS,
    )
    rn.add_argument(
        "--assistant",
        choices=("codex", "claude"),
        help=argparse.SUPPRESS,
    )
    rn.add_argument(
        "--store",
        help=(
            "SQLite path for a recorded, resumable run. Naming one implies "
            "--durable."
        ),
    )
    rn.add_argument(
        "--durable",
        action="store_true",
        help=(
            "Record the run so it can be resumed, and collect any missing "
            "inputs interactively."
        ),
    )
    rn.add_argument(
        "--resume",
        action="store_true",
        help="Continue the project's most recent unfinished run.",
    )
    rn.add_argument(
        "--run-id",
        help="Which recorded run to resume; requires --resume.",
    )
    rn.add_argument(
        "--project",
        help="Project root for a durable run; defaults to discovery.",
    )
    rn.add_argument(
        "--yes",
        action="store_true",
        help="Do not prompt for missing inputs during a durable run.",
    )
    rn.add_argument("--input", action="append", default=[], metavar="name=value", help="Workflow input value.")
    rn.add_argument("--input-json", help="Workflow inputs as a JSON object.")
    rn.add_argument("--option", action="append", default=[], metavar="name=value", help="Option passed to zippergen_setup(config).")
    rn.add_argument("--services", choices=("fake", "live"), help="Shortcut for --option services=<value>.")
    rn.add_argument("--timeout", type=float, default=60.0, help="Workflow timeout in seconds; use 0 for no deadline.")
    rn.add_argument(
        "--execution",
        choices=("sqlite", "memory"),
        default=None,
        help=(
            "Where the run keeps its state. Defaults to 'memory', which leaves "
            "nothing behind. 'sqlite' uses an unregistered SQLite store; "
            "use --durable or --store for a recorded, resumable run."
        ),
    )

    show = sub.add_parser("show", help="render a workflow as a code-first semantic view")
    show.add_argument("workflow", nargs="?", help="Workflow spec: module:workflow or path.py:workflow. Defaults to this project's workflow.")
    show.add_argument("--detail", choices=DETAILS, default="protocol", help="Amount of implementation detail to include.")
    show.add_argument("--communications", action="store_true", help="Show communication and control flow only.")
    focus = show.add_mutually_exclusive_group()
    focus.add_argument("--agent", help="Show the exact local projection for one agent.")
    focus.add_argument("--agents", help="Comma-separated agents to retain in a boundary-aware focus view.")
    show.add_argument("--format", choices=("code", "json"), default="code", help="Output format.")

    validate = sub.add_parser("validate", help="validate loading, projection, rendering, and deployment metadata")
    validate.add_argument("workflow", nargs="?", help="Workflow spec: module:workflow or path.py:workflow. Defaults to this project's workflow.")
    validate.add_argument("--json", action="store_true", help="Print machine-readable validation results.")

    init_parser = sub.add_parser(
        "init",
        help="create a ZipperGen project in this directory",
    )
    init_parser.add_argument(
        "name",
        nargs="?",
        help="Project name; defaults to the directory name.",
    )
    init_parser.add_argument(
        "--directory",
        help="Create the project here instead of the current directory.",
    )

    skill_parser = sub.add_parser(
        "skill",
        help="print the coding-agent skill shipped with this package",
    )
    skill_parser.add_argument(
        "--agents-md",
        action="store_true",
        help="Print an AGENTS.md that points a coding agent at the skill.",
    )
    skill_parser.add_argument(
        "--project",
        help="Project name for --agents-md; defaults to the directory name.",
    )
    skill_parser.add_argument(
        "--no-references",
        action="store_true",
        help="Print SKILL.md alone, without the reference files it links.",
    )

    semantic_diff_parser = sub.add_parser(
        "diff",
        help="save a semantic baseline, or compare against one",
        description=(
            "Save a baseline with --save, then run 'zg diff BASELINE' after "
            "editing to see exactly what changed in the protocol."
        ),
    )
    semantic_diff_parser.add_argument(
        "before",
        nargs="?",
        help="Original workflow spec or saved baseline JSON.",
    )
    semantic_diff_parser.add_argument(
        "after",
        nargs="?",
        help=(
            "Modified workflow spec or semantic snapshot JSON. Defaults to "
            "this project's workflow, so a saved baseline can be compared "
            "against the working tree with one argument."
        ),
    )
    semantic_diff_parser.add_argument("--format", choices=("code", "json"), default="code", help="Output format.")
    semantic_diff_parser.add_argument(
        "--save",
        metavar="PATH",
        help=(
            "Write a semantic baseline of this project's workflow to PATH "
            "instead of comparing. Use '-' for standard output."
        ),
    )

    # One family, the same shape as model/assistant/connector: run it bare to
    # see everything, then one verb per thing you can do to a deployment.
    deploy = sub.add_parser(
        "deploy",
        help="show and manage deployments",
        description=(
            "One pattern: create a deployment, then act on it by name. "
            "Run without an action to show every deployment."
        ),
    )
    deploy_sub = deploy.add_subparsers(dest="deploy_action")

    deploy_create = deploy_sub.add_parser(
        "create",
        help="configure, validate, and start a workflow deployment",
    )
    deploy_create.add_argument(
        "target",
        nargs="?",
        help=(
            "Workflow spec, or the name of a deployment that already exists. "
            "Defaults to this project's workflow; name a new deployment with "
            "--name."
        ),
    )
    deploy_create.add_argument("--name", help="Deployment name; defaults to the workflow declaration or workflow name.")
    _add_guided_deployment_arguments(deploy_create)
    deploy_create.add_argument("--no-bundle", action="store_true", help="Run from source instead of snapshotting declared files.")
    deploy_create.add_argument("--no-start", action="store_true", help="Configure the deployment without starting its service.")

    deploy_configure = deploy_sub.add_parser(
        "configure",
        help="update a deployment's persistent configuration",
    )
    deploy_configure.add_argument("name", help="Deployment name.")
    _add_guided_deployment_arguments(deploy_configure, configure=True)
    deploy_configure.add_argument("--restart", action="store_true", help="Restart the service after configuration succeeds.")
    deploy_configure.set_defaults(no_start=True, no_bundle=True, no_install=True, no_setup=True)

    deploy_run = deploy_sub.add_parser("run")
    deploy_run.add_argument("name", help="Deployment name.")

    deploy_start = deploy_sub.add_parser("start", help="start a deployment as a supervised user service")
    deploy_start.add_argument("name", help="Deployment name.")
    deploy_start.add_argument("--enable", action="store_true", help="Enable the service to start automatically for this user.")
    deploy_start.add_argument("--dry-run", action="store_true", help="Print service-manager commands without running them.")

    deploy_stop = deploy_sub.add_parser("stop", help="stop a supervised deployment")
    deploy_stop.add_argument("name", help="Deployment name.")
    deploy_stop.add_argument("--dry-run", action="store_true", help="Print the service-manager command without running it.")

    deploy_restart = deploy_sub.add_parser("restart", help="restart a supervised deployment")
    deploy_restart.add_argument("name", help="Deployment name.")
    deploy_restart.add_argument("--dry-run", action="store_true", help="Print service-manager commands without running them.")

    deploy_remove = deploy_sub.add_parser(
        "remove",
        help="delete a deployment, keeping its durable store unless purged",
    )
    deploy_remove.add_argument("name", help="Deployment name.")
    deploy_remove.add_argument("--purge", action="store_true", help="Delete the durable store and log too, leaving nothing.")
    deploy_remove.add_argument("--yes", action="store_true", help="Do not ask for confirmation.")

    deploy_compact = deploy_sub.add_parser(
        "compact",
        help="reclaim space in a stopped deployment's store and logs",
    )
    deploy_compact.add_argument("name", help="Deployment name.")
    deploy_compact.add_argument("--keep-archives", type=int, default=3, help="How many rotated log archives to retain. Default 3.")

    deploy_logs = deploy_sub.add_parser("logs", help="show logs for a deployment")
    deploy_logs.add_argument("name", help="Deployment name.")
    deploy_logs.add_argument("--tail", type=int, default=80, help="Number of log lines to show.")
    deploy_logs.add_argument("--follow", action="store_true", help="Keep watching the log file.")
    deploy_logs.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds for --follow.")

    deploy_check = deploy_sub.add_parser("check", help="check a deployment for common problems")
    deploy_check.add_argument("name", help="Deployment name.")
    deploy_check.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    deploy_check.add_argument(
        "--no-service-check",
        "--no-systemd",
        dest="no_systemd",
        action="store_true",
        help="Skip live launchd/systemd active-state checks.",
    )

    deploy_status = deploy_sub.add_parser("status", help="show durable store status for a deployment")
    deploy_status.add_argument("deployment", nargs="?", help="Deployment name.")
    deploy_status.add_argument("--store", help="SQLite store path.")
    deploy_status.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    inspect_parser = sub.add_parser(
        "inspect",
        help="show the current durable program position for each participant",
    )
    inspect_parser.add_argument("deployment", nargs="?", help="Deployment name.")
    inspect_parser.add_argument("--store", help="SQLite store path.")
    inspect_parser.add_argument(
        "--workflow",
        help="Workflow spec for --store; defaults to this project's workflow.",
    )
    inspect_parser.add_argument("--project", help="Project root for a durable run.")
    inspect_parser.add_argument("--agent", help="Participant whose local projection to show.")
    inspect_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    inspect_parser.add_argument(
        "--watch",
        action="store_true",
        help="Refresh the inspection in place until Ctrl-C.",
    )
    inspect_parser.add_argument(
        "--interval",
        type=float,
        help="Refresh interval in seconds for --watch. Default 1.",
    )

    tr = sub.add_parser("trace", help="show recent trace events from a local SQLite store")
    tr.add_argument("deployment", nargs="?", help="Deployment name.")
    tr.add_argument("--store", help="SQLite store path.")
    tr.add_argument("--tail", type=int, default=50, help="Maximum number of trace events to show.")
    tr.add_argument("--after", type=int, default=0, help="Only show trace events after this event rowid.")
    tr.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    tk = sub.add_parser("tasks", help="list human tasks in a local SQLite store")
    tk.add_argument("--store", required=True, help="SQLite store path.")
    tk.add_argument("--all", action="store_true", help="Include completed tasks.")
    tk.add_argument("--limit", type=int, help="Maximum number of tasks to show.")
    tk.add_argument("--tokens", action="store_true", help="Generate/show durable approval tokens.")
    tk.add_argument("--channel", default="cli", help="Token channel name used with --tokens.")
    tk.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    apv = sub.add_parser("approve", help="complete a pending human task")
    apv.add_argument("--store", required=True, help="SQLite store path.")
    target = apv.add_mutually_exclusive_group(required=True)
    target.add_argument("--task", help="Human task id.")
    target.add_argument("--token", help="Durable approval token.")
    apv.add_argument("--yes", action="store_true", help="Complete a boolean task with true.")
    apv.add_argument("--no", action="store_true", help="Complete a boolean task with false.")
    apv.add_argument("--value", help="Value for string tasks, or explicit true/false for boolean tasks.")
    apv.add_argument("--result-json", help="Complete with an explicit JSON object result.")
    apv.add_argument("--json", action="store_true", help="Print the completed task as JSON.")

    nt = sub.add_parser("notify", description="Notification adapter a deployment runs for itself.")
    notify_sub = nt.add_subparsers(dest="adapter", required=True)
    out = notify_sub.add_parser("stdout", help="print pending human tasks with approval tokens")
    out.add_argument("--store", required=True, help="SQLite store path.")
    out.add_argument("--channel", default="stdout", help="Approval token channel name.")
    out.add_argument("--watch", action="store_true", help="Keep polling for new pending tasks.")
    out.add_argument("--interval", type=float, default=2.0, help="Polling interval in seconds for --watch.")
    out.add_argument("--limit", type=int, help="Maximum number of tasks to notify per poll.")
    out.add_argument("--quiet", action="store_true", help="Suppress the no-pending-tasks message in one-shot mode.")

    tg = notify_sub.add_parser("telegram", help="send and receive human task approvals through Telegram")
    tg.add_argument("--store", required=True, help="SQLite store path.")
    tg.add_argument("--bot-token", help="Telegram bot token. Defaults to ZIPPERGEN_TELEGRAM_TOKEN.")
    tg.add_argument("--chat-id", help="Telegram chat id. Defaults to ZIPPERGEN_TELEGRAM_CHAT_ID.")
    tg.add_argument("--channel", default="telegram", help="Approval token channel name.")
    tg.add_argument("--watch", action="store_true", help="Keep polling Telegram and the local store.")
    tg.add_argument("--interval", type=float, default=2.0, help="Delay between store scans in --watch mode.")
    tg.add_argument("--poll-timeout", type=float, default=20.0, help="Telegram long-poll timeout in seconds.")
    tg.add_argument("--limit", type=int, help="Maximum number of tasks to notify per poll.")
    tg.add_argument("--resend", action="store_true", help="Resend already-notified pending tasks.")
    tg.add_argument("--quiet", action="store_true", help="Suppress progress messages.")

    sv = sub.add_parser(
        "serve",
        description="Legacy low-level per-role runner. Prefer `zippergen run` for local deployment.",
    )
    sv.add_argument("--workflow", required=True)
    sv.add_argument("--role", required=True)
    sv.add_argument("--store", required=True)
    sv.add_argument("--input", action="append", default=[], metavar="k=v")

    # Derive the visible command list from what is actually registered. A
    # hand-written list drifts the moment a command is added or renamed, and
    # this one had drifted.
    sub.metavar = "{" + ",".join(
        name for name in sub.choices if name not in HIDDEN_COMMANDS
    ) + "}"

    args = ap.parse_args(argv)
    return ap, args


def main(argv=None) -> int:
    ap, args = _parse_cli_args(argv)

    if args.cmd is None:
        ap.print_help()
        return 0
    if args.cmd == "config":
        if args.live and args.config_action != "check":
            raise SystemExit("--live requires 'config check'.")
        return _configuration_command(args)
    if args.cmd == "model":
        return _model_command(args)
    if args.cmd == "assistant":
        return _assistant_command(args)
    if args.cmd == "completion":
        from zippergen.completion import render_completion

        print(render_completion(args.shell))
        return 0
    if args.cmd == "__complete":
        from zippergen.completion import completion_candidates

        print(
            "\n".join(
                completion_candidates(args.kind, args.project, tuple(args.path))
            )
        )
        return 0
    if (
        args.cmd == "connector"
        and args.connector_action == "authorize"
        and args.connector_provider == "google"
    ):
        return _connector_authorize_google_command(args)
    if args.cmd == "run":
        if getattr(args, "run_id", None) and not args.resume:
            raise SystemExit("--run-id requires --resume.")
        if (
            args.execution == "memory"
            and (
                getattr(args, "durable", False)
                or getattr(args, "resume", False)
                or getattr(args, "store", None)
            )
        ):
            raise SystemExit(
                "A durable or stored run uses SQLite; remove '--execution memory'."
            )
        if (
            getattr(args, "durable", False)
            or getattr(args, "resume", False)
            or getattr(args, "store", None)
        ):
            return _durable_run_command(args)
        return _run_workflow_command(args)
    if args.cmd == "show":
        return _show_command(args)
    if args.cmd == "validate":
        return _validate_command(args)
    if args.cmd == "connector" and args.connector_action == "configure":
        return _connector_configure_command(args)
    if args.cmd == "connector" and args.connector_action == "assign":
        return _connector_assign_command(args)
    if args.cmd == "connector" and args.connector_action in {
        None,
        "bind",
        "unassign",
        "unbind",
        "check",
        "remove",
    }:
        return _connector_management_command(args)
    if (
        args.cmd == "connector"
        and args.connector_action == "accept"
        and args.connector_provider == "google"
    ):
        return _connector_accept_google_command(args)
    if args.cmd == "init":
        return _init_command(args)
    if args.cmd == "skill":
        return _skill_command(args)
    if args.cmd == "diff":
        return _diff_command(args)
    if args.cmd == "deploy":
        action = getattr(args, "deploy_action", None)
        if action is None:
            return _deployment_overview_command(args)
        if action == "create":
            return _deploy_command(args)
        if action == "configure":
            return _configure_deployment_command(args)
        if action == "run":
            return _run_deployment_command(args)
        if action in {"start", "stop", "restart"}:
            return _deployment_lifecycle_command(args, action)
        if action == "remove":
            return _remove_command(args)
        if action == "compact":
            return _compact_command(args)
        if action == "logs":
            return _logs_command(args)
        if action == "check":
            return _doctor_command(args)
        if action == "status":
            return _status_command(args)
    if args.cmd == "inspect":
        return _inspect_command(args)
    if args.cmd == "trace":
        return _trace_command(args)
    if args.cmd == "tasks":
        return _tasks_command(args)
    if args.cmd == "approve":
        return _approve_command(args)
    if args.cmd == "notify" and args.adapter == "stdout":
        return _notify_stdout_command(args)
    if args.cmd == "notify" and args.adapter == "telegram":
        return _notify_telegram_command(args)

    print("Warning: `zippergen serve` is a legacy low-level command; prefer `zippergen run`.", file=sys.stderr)
    wf, role_ll = load_workflow(args.workflow, args.role)
    lifelines = _workflow_lifelines(wf)
    from zippergen.runtime import _build_formula_monitors
    monitors, formula_conditions = _build_formula_monitors(wf, lifelines)
    conn = open_store(args.store)
    env = seed_env(conn, args.role, wf, _seed_inputs(wf, _parse_inputs(args.input)))
    local = project(wf, role_ll)
    final = run_role(
        conn,
        args.role,
        local,
        env,
        wf.ns,
        monitor=monitors.get(args.role),
        formula_conditions=formula_conditions,
    )
    print(json.dumps({k: v for k, v in final.items()
                      if isinstance(v, (bool, int, float, str, type(None)))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
