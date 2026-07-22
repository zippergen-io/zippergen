"""Per-role durable runtime: project one role, replay its committed history,
then run live, persisting each step atomically.

Non-deterministic results are journaled so replay reconstructs them instead of
re-executing: external acts (LLM/Human/Planner/Effect) and owner if/while
decisions are recorded as ``kind='act'``/``'decision'`` rows and consumed on
replay; a ``@pure`` act is deterministic by contract and recomputed. A blocking external
call runs OUTSIDE the SQLite write transaction (the lock is released first), then
its result is journaled and committed before env/residual advance.

Limitations (v1):
- External effects are at-least-once: a crash between an external call returning
  and its journal row committing re-runs the act on restart. Harmless for an LLM
  (paid twice); visible human actions are first materialized as durable
  ``human_tasks`` rows and can be answered out-of-band. Irreversible effects
  (e.g. sending mail) are NOT made exactly-once here — that needs effect-level
  idempotency keys.
- No snapshot fires inside a parallel region (the rebuilt residual changes
  identity each step), so a crash replays the region from the enclosing loop
  boundary; replay-length bounding inside parallel is a later refinement.
- CPL Formula guards use full replay for now because role snapshots do not yet
  persist monitor state.
- A blocking external act in one parallel branch stalls that role's other
  branches until it returns (no intra-role concurrency of external calls).

See docs/superpowers/specs/2026-07-04-durable-deploy-hardening-design.md."""
from __future__ import annotations

from zippergen.role_runner import (
    JournalContext,
    RoleRunner,
    _floor_coherent,
    _maybe_snapshot,
    _try_resume,
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
import importlib
import importlib.util
import json
import os
import platform
import plistlib
import re
import shlex
import shutil
import subprocess
import sys
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
from zippergen.models import (
    effective_llm_routes,
    normalize_llm_overrides,
    selected_llm_specs,
)
from zippergen.view import DETAILS, ViewOptions, render_workflow, workflow_view_data
from zippergen.semantic import (
    read_semantic_snapshot,
    render_semantic_diff,
    semantic_diff_models,
    semantic_snapshot,
    workflow_semantics,
)
from zippergen.syntax import Workflow, Lifeline
from zippergen.projection import project
from zippergen.store import (
    complete_human_task,
    ensure_human_task_token,
    list_workflow_results,
    load_human_task,
    load_human_task_token,
    mark_human_task_token_used,
    open_store,
)


@dataclass(frozen=True)
class RunConfig:
    """Configuration passed to an optional module-level ``zippergen_setup`` hook."""

    workflow_spec: str
    workflow: Workflow
    module: ModuleType
    llm: str | None
    llms: dict[str, str]
    llm_idle_timeout: float | None
    store_path: str | None
    inputs: dict[str, object]
    options: dict[str, object]
    ui: bool
    timeout: float
    execution: str
    show_decisions: bool

    def option(self, name: str, default: object = None) -> object:
        return self.options.get(name, default)


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


def _import_module_path(module_path: str) -> ModuleType:
    path = Path(module_path).expanduser()
    spec = importlib.util.spec_from_file_location(
        f"_zippergen_wf_{hashlib.sha1(str(path).encode()).hexdigest()[:12]}",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(path.parent))
        except ValueError:
            pass
    return module


def _looks_like_path(name: str) -> bool:
    return name.endswith(".py") or "/" in name or "\\" in name or Path(name).exists()


def _import_workflow_module(module_ref: str) -> ModuleType:
    if _looks_like_path(module_ref):
        return _import_module_path(module_ref)
    return importlib.import_module(module_ref)


def load_workflow_spec(spec_text: str) -> tuple[Workflow, ModuleType]:
    """Load ``module:workflow`` or ``path.py:workflow``.

    If no workflow name is supplied, the module must define exactly one
    ``Workflow`` object.
    """

    module_ref, sep, workflow_name = spec_text.partition(":")
    if not module_ref:
        raise SystemExit("Workflow spec must be MODULE:WORKFLOW or PATH.py:WORKFLOW.")
    module = _import_workflow_module(module_ref)
    if sep:
        try:
            value = getattr(module, workflow_name)
        except AttributeError as exc:
            raise SystemExit(f"Workflow {workflow_name!r} not found in {module_ref!r}.") from exc
        if not isinstance(value, Workflow):
            raise SystemExit(f"{module_ref}:{workflow_name} is not a ZipperGen Workflow.")
        return value, module

    workflows = {
        name: value
        for name, value in vars(module).items()
        if isinstance(value, Workflow)
    }
    if len(workflows) == 1:
        return next(iter(workflows.values())), module
    if not workflows:
        raise SystemExit(f"No ZipperGen Workflow found in {module_ref!r}.")
    names = ", ".join(sorted(workflows))
    raise SystemExit(f"Multiple workflows found in {module_ref!r}: {names}. Use MODULE:WORKFLOW.")


def load_workflow(module_path: str, role_name: str) -> tuple[Workflow, Lifeline]:
    module = _import_module_path(module_path)
    workflows = [value for value in vars(module).values() if isinstance(value, Workflow)]
    if not workflows:
        raise SystemExit(f"No ZipperGen Workflow found in {module_path!r}.")
    wf = workflows[0]
    lifelines = {ll.name: ll for ll in _workflow_lifelines(wf)}
    if role_name not in lifelines:
        raise SystemExit(f"role {role_name!r} not in workflow lifelines {sorted(lifelines)}")
    return wf, lifelines[role_name]


def _workflow_lifelines(wf: Workflow) -> tuple[Lifeline, ...]:
    from zippergen.syntax import _ordered_workflow_lifelines
    return _ordered_workflow_lifelines(wf)


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


def _seed_inputs(wf: Workflow, inputs: dict) -> dict:
    """Var defaults from the workflow namespace, overlaid by caller inputs —
    parity with the in-process run() seeding (runtime.py:1014)."""
    from zippergen.syntax import Var
    env = {k: v.default for k, v in wf.ns.items() if isinstance(v, Var)}
    env.update(inputs)
    return env


def _slug(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text.strip()).strip("-._")
    return text or "workflow"


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


def _zippergen_home() -> Path:
    return Path(os.environ.get("ZIPPERGEN_HOME", str(Path.home() / ".zippergen"))).expanduser()


def _deployments_dir() -> Path:
    return _zippergen_home() / "deployments"


def _deployment_profile_path(name: str) -> Path:
    return _deployments_dir() / f"{_slug(name)}.json"


def _deployment_script_path(name: str) -> Path:
    return _deployments_dir() / f"{_slug(name)}.sh"


def _deployment_service_path(name: str) -> Path:
    return _deployments_dir() / f"zippergen-{_slug(name)}.service"


def _deployment_launchd_path(name: str) -> Path:
    return _deployments_dir() / f"io.zippergen.{_slug(name)}.plist"


def _deployment_secrets_path(name: str) -> Path:
    return _deployments_dir() / f"{_slug(name)}.secrets.json"


def _deployment_environment_dir(name: str) -> Path:
    return _zippergen_home() / "environments" / _slug(name)


def _deployment_bundles_dir(name: str) -> Path:
    return _zippergen_home() / "apps" / _slug(name)


def _systemd_user_dir() -> Path:
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))).expanduser()
    return config_home / "systemd" / "user"


def _systemd_unit_name(name: str) -> str:
    return f"zippergen-{_slug(name)}.service"


def _installed_systemd_service_path(name: str) -> Path:
    return _systemd_user_dir() / _systemd_unit_name(name)


def _launchd_label(name: str) -> str:
    return f"io.zippergen.{_slug(name)}"


def _launch_agents_dir() -> Path:
    configured = os.environ.get("ZIPPERGEN_LAUNCH_AGENTS_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / "Library" / "LaunchAgents"


def _installed_launchd_path(name: str) -> Path:
    return _launch_agents_dir() / f"{_launchd_label(name)}.plist"


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
        store=str(profile["store"]),
        input=[],
        input_json=json.dumps(profile.get("inputs") or {}, default=str),
        option=_jsonable_kv_pairs(profile.get("options") or {}),  # type: ignore[arg-type]
        services=profile.get("services") or None,
        ui=bool(profile.get("ui", False)),
        timeout=timeout,
        execution=str(profile.get("execution", "sqlite")),
        show_decisions=bool(profile.get("show_decisions", False)),
    )


def _deployment_command(name: str, *, python_executable: str | None = None) -> str:
    python = python_executable or sys.executable
    return f"{shlex.quote(python)} -m zippergen.serve run-deployment {shlex.quote(_slug(name))}"


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
        "Restart=always\n"
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
        "KeepAlive": True,
        "ThrottleInterval": 10,
        "StandardOutPath": str(profile["log"]),
        "StandardErrorPath": str(profile["log"]),
        "ProcessType": "Background",
    }
    launchd_path.write_bytes(plistlib.dumps(launchd, sort_keys=True))


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


def _systemctl_command(*args: str) -> list[str]:
    systemctl = os.environ.get("ZIPPERGEN_SYSTEMCTL", "systemctl")
    return [systemctl, "--user", *args]


def _run_systemctl(args: list[str], *, dry_run: bool = False) -> None:
    if dry_run:
        print(shlex.join(args))
        return
    try:
        subprocess.run(args, check=True)
    except FileNotFoundError as exc:
        raise SystemExit("systemctl was not found. Use `run-deployment` directly or install systemd user services.") from exc
    except subprocess.CalledProcessError as exc:
        command = shlex.join(args)
        raise SystemExit(f"Command failed with exit code {exc.returncode}: {command}") from exc


def _service_manager() -> str:
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
        "Use `zippergen run-deployment NAME` directly."
    )


def _launchctl_domain() -> str:
    return f"gui/{os.getuid()}"


def _launchctl_command(*args: str) -> list[str]:
    launchctl = os.environ.get("ZIPPERGEN_LAUNCHCTL", "launchctl")
    return [launchctl, *args]


def _run_launchctl(
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
            "launchctl was not found. Use `run-deployment` directly or run on macOS."
        ) from exc
    except subprocess.CalledProcessError as exc:
        command = shlex.join(args)
        raise SystemExit(f"Command failed with exit code {exc.returncode}: {command}") from exc


def _deployment_lifecycle_command(args, action: str) -> int:
    profile = _load_deployment_profile(args.name)
    name = str(profile["name"])
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

    def print_tail() -> int:
        lines = log_path.read_text(errors="replace").splitlines()
        for line in lines[-args.tail:]:
            print(line)
        return len(lines)

    seen = print_tail()
    if not args.follow:
        return 0
    while True:
        time.sleep(args.interval)
        lines = log_path.read_text(errors="replace").splitlines()
        for line in lines[seen:]:
            print(line)
        seen = len(lines)


def _doctor_check(status: str, name: str, detail: str, **extra: object) -> dict[str, object]:
    return {"status": status, "name": name, "detail": detail, **extra}


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
    service = f"{_launchctl_domain()}/{_launchd_label(name)}"
    try:
        result = subprocess.run(
            _launchctl_command("print", service),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        return _doctor_check("warn", "launchd active", "launchctl was not found")
    except subprocess.TimeoutExpired:
        return _doctor_check("warn", "launchd active", "launchctl timed out")
    if result.returncode == 0:
        return _doctor_check("ok", "launchd active", f"{service} is loaded")
    detail = (result.stderr or result.stdout or "not loaded").strip().splitlines()[0]
    return _doctor_check("warn", "launchd active", f"{service} is not loaded: {detail}")


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


def _doctor_checks(name: str, *, include_systemd: bool = True) -> list[dict[str, object]]:
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

    environment = _deployment_environment(profile)
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


def _doctor_command(args) -> int:
    checks = _doctor_checks(args.name, include_systemd=not args.no_systemd)
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


def _call_setup_hook(module: ModuleType, config: RunConfig) -> None:
    setup = getattr(module, "zippergen_setup", None)
    if setup is None:
        return
    if not callable(setup):
        raise SystemExit("zippergen_setup exists but is not callable.")
    setup(config)


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


def _run_workflow_command(args) -> int:
    wf, module = load_workflow_spec(args.workflow)
    inputs = _parse_input_json(args.input_json)
    inputs.update(_parse_inputs(args.input))
    options = _parse_options(args.option, services=args.services)
    llms = normalize_llm_overrides(_parse_inputs(args.llm_for))

    store_path = args.store
    if args.execution == "sqlite":
        store_path = _ensure_store_parent(store_path or _default_store_path(args.workflow, wf))
        print(f"Store: {store_path}", file=sys.stderr)
    elif store_path:
        print("--store is ignored when --execution memory is used.", file=sys.stderr)

    config = RunConfig(
        workflow_spec=args.workflow,
        workflow=wf,
        module=module,
        llm=args.llm,
        llms=llms,
        llm_idle_timeout=args.llm_idle_timeout,
        store_path=store_path,
        inputs=inputs,
        options=options,
        ui=args.ui,
        timeout=args.timeout,
        execution=args.execution,
        show_decisions=args.show_decisions,
    )
    _call_setup_hook(module, config)

    configure_kwargs = {
        "ui": args.ui,
        "timeout": args.timeout,
        "llm_idle_timeout": args.llm_idle_timeout,
        "execution": args.execution,
        "store_path": store_path,
        "show_decisions": args.show_decisions,
    }
    if llms:
        wf.configure(
            effective_llm_routes(wf, args.llm or "mock", llms),
            **configure_kwargs,
        )
    elif args.llm:
        wf.configure(args.llm, **configure_kwargs)
    else:
        wf.configure(**configure_kwargs)

    result = wf(**inputs)
    print(json.dumps({"result": result}, default=str))
    if args.ui and sys.stdin.isatty():
        input("ZipperChat running at http://localhost:8765. Press Enter to exit. ")
    return 0


def _dev_command(args) -> int:
    from zippergen.dev import run_dev
    from zippergen.workspace import Workspace

    inputs = _parse_input_json(args.input_json)
    inputs.update(_parse_inputs(args.input))
    options = _parse_options(args.option, services=args.services)
    workspace = Workspace(args.project)
    run_dev(
        workspace,
        workflow_spec=args.workflow,
        resume=args.resume,
        run_id=args.run_id,
        provided_inputs=inputs,
        llm=args.llm,
        llms=normalize_llm_overrides(_parse_inputs(args.llm_for)),
        options=options,
        services=args.services,
        timeout=args.timeout,
        interactive=not args.yes and sys.stdin.isatty(),
        input_func=input,
        output_func=print,
    )
    return 0


def _studio_command(args) -> int:
    from zippergen.studio import Studio
    from zippergen.workspace import Workspace

    workspace = Workspace(args.project)
    if args.workflow:
        canonical = workspace.canonical_spec(args.workflow)
        load_workflow_spec(workspace.absolute_spec(canonical))
        workspace.select_workflow(canonical, cwd=workspace.root)
    studio = Studio(workspace, input_func=input, output_func=print)
    if args.command:
        studio.welcome()
        for command in args.command:
            if not studio.execute(command, show_boundary=True):
                break
        return 0
    return studio.run()


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


def _validate_workflow(workflow: Workflow, module: ModuleType) -> dict[str, object]:
    lifelines = _workflow_lifelines(workflow)
    checks: list[dict[str, object]] = []
    names = [lifeline.name for lifeline in lifelines]
    if len(names) != len(set(names)):
        checks.append({
            "status": "fail",
            "name": "lifelines",
            "detail": "lifeline names are not unique",
        })
    else:
        checks.append({
            "status": "ok",
            "name": "lifelines",
            "detail": f"{len(lifelines)} unique lifeline(s): {', '.join(names)}",
        })

    projections: dict[str, str] = {}
    for lifeline in lifelines:
        try:
            local = project(workflow, lifeline)
            code = render_workflow(
                workflow,
                module,
                options=ViewOptions(agent=lifeline.name),
            )
        except Exception as exc:
            checks.append({
                "status": "fail",
                "name": f"projection {lifeline.name}",
                "detail": f"{type(exc).__name__}: {exc}",
            })
        else:
            projections[lifeline.name] = code
            checks.append({
                "status": "ok",
                "name": f"projection {lifeline.name}",
                "detail": type(local).__name__,
            })

    try:
        declaration = deployment_spec_from_module(module)
    except Exception as exc:
        checks.append({
            "status": "fail",
            "name": "deployment declaration",
            "detail": f"{type(exc).__name__}: {exc}",
        })
        deployment: dict[str, object] | None = None
    else:
        deployment = declaration.as_dict()
        checks.append({
            "status": "ok",
            "name": "deployment declaration",
            "detail": (
                f"{len(declaration.fields)} field(s), "
                f"{len(declaration.packages)} package(s), "
                f"{len(declaration.setup)} setup step(s)"
            ),
        })

    try:
        render_workflow(workflow, module, options=ViewOptions(detail="full"))
    except Exception as exc:
        checks.append({
            "status": "fail",
            "name": "canonical rendering",
            "detail": f"{type(exc).__name__}: {exc}",
        })
    else:
        checks.append({
            "status": "ok",
            "name": "canonical rendering",
            "detail": "global and local code views rendered successfully",
        })

    return {
        "workflow": workflow.name,
        "valid": not any(check["status"] == "fail" for check in checks),
        "lifelines": names,
        "inputs": [
            {
                "name": name,
                "type": getattr(value_type, "__name__", str(value_type)),
                "lifeline": lifeline.name if lifeline else None,
            }
            for name, value_type, lifeline in workflow.inputs
        ],
        "outputs": [
            {
                "name": value.name,
                "type": getattr(value.type, "__name__", str(value.type)),
                "lifeline": lifeline.name,
            }
            for value, lifeline in workflow.outputs
        ],
        "deployment": deployment,
        "checks": checks,
        "projections": projections,
    }


def _validate_command(args) -> int:
    workflow, module = load_workflow_spec(args.workflow)
    result = _validate_workflow(workflow, module)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        verdict = "valid" if result["valid"] else "invalid"
        print(f"Workflow {workflow.name}: {verdict}")
        for check in result["checks"]:  # type: ignore[union-attr]
            print(f"{str(check['status']).upper():4} {check['name']}: {check['detail']}")
    return 0 if result["valid"] else 1


def _snapshot_command(args) -> int:
    workflow, module = load_workflow_spec(args.workflow)
    payload = json.dumps(semantic_snapshot(workflow, module), indent=2, default=str)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n")
        print(f"Wrote semantic snapshot to {output_path}")
    else:
        print(payload)
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
    result = semantic_diff_models(
        _semantic_input(args.before),
        _semantic_input(args.after),
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


def _bundle_deployment(profile: dict[str, object], spec: DeploymentSpec) -> None:
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


def _zippergen_install_requirement() -> str:
    project_root = Path(__file__).resolve().parents[2]
    if (project_root / "pyproject.toml").exists():
        return str(project_root)
    try:
        from importlib.metadata import version

        return f"zippergen=={version('zippergen')}"
    except Exception:
        return "zippergen"


def _prepare_deployment_environment(
    profile: dict[str, object],
    spec: DeploymentSpec,
    *,
    skip_install: bool,
) -> None:
    requirements = [package.requirement for package in spec.packages]
    profile["packages"] = requirements
    if skip_install:
        profile["python"] = str(profile.get("python") or sys.executable)
        return

    name = str(profile["name"])
    environment_dir = _deployment_environment_dir(name)
    python = _deployment_python_path(environment_dir)
    if not python.exists():
        print(f"Creating managed Python environment for {name}...")
        venv.EnvBuilder(with_pip=True).create(environment_dir)

    install = [str(python), "-m", "pip", "install", _zippergen_install_requirement(), *requirements]
    print("Installing deployment dependencies...")
    try:
        subprocess.run(install, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Dependency installation failed with exit code {exc.returncode}.") from exc
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
    if args.llm is not None:
        profile["llm"] = args.llm
    llms = normalize_llm_overrides(profile.get("llms"))
    for lifeline, model in normalize_llm_overrides(
        _parse_inputs(args.llm_for)
    ).items():
        if model.lower() in {"inherit", "default"}:
            llms.pop(lifeline, None)
        else:
            llms[lifeline] = model
    effective_llm_routes(workflow, str(profile.get("llm") or "mock"), llms)
    profile["llms"] = llms
    if args.llm_idle_timeout is not None:
        profile["llm_idle_timeout"] = args.llm_idle_timeout
    if args.services is not None:
        profile["services"] = args.services
    if args.timeout is not None:
        profile["timeout"] = args.timeout
    if args.store is not None:
        profile["store"] = _ensure_store_parent(args.store)
    if args.log is not None:
        profile["log"] = str(Path(args.log).expanduser())
    if args.ui is not None:
        profile["ui"] = args.ui
    if args.show_decisions is not None:
        profile["show_decisions"] = args.show_decisions

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
    return _collect_deployment_fields(
        spec,
        profile,
        overrides=overrides,
        interactive=interactive,
    )


def _finalize_guided_deployment(
    profile: dict[str, object],
    spec: DeploymentSpec,
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

    if not args.no_bundle:
        _bundle_deployment(profile, spec)
    # Persist enough state to resume configuration even if dependency install
    # or an interactive OAuth step fails.
    _write_deployment_artifacts(profile)
    _prepare_deployment_environment(profile, spec, skip_install=args.no_install)
    _write_deployment_artifacts(profile)
    _run_deployment_setup(profile, spec, values, skip_setup=args.no_setup)
    _write_deployment_artifacts(profile)

    if not args.no_doctor:
        checks = _doctor_checks(name, include_systemd=False)
        _print_doctor(name, checks)
        if any(check.get("status") == "fail" for check in checks):
            print(f"Deployment {name} was configured but not started because doctor found failures.")
            return 1

    if not args.no_start:
        lifecycle_args = argparse.Namespace(name=name, enable=True, dry_run=False)
        _deployment_lifecycle_command(lifecycle_args, "start")

    print(f"Deployment: {name}")
    print(f"Status: zippergen status {name}")
    print(f"Logs: zippergen logs {name} --follow")
    print(f"Restart: zippergen restart {name}")
    return 0


def _deploy_command(args) -> int:
    existing_path = _deployment_profile_path(args.target)
    if existing_path.exists() and not _looks_like_path(args.target) and ":" not in args.target:
        profile, workflow, module, spec = _deployment_context(args.target, source=True)
        if args.name and _slug(args.name) != str(profile["name"]):
            raise SystemExit("--name cannot rename an existing deployment.")
    else:
        workflow, module = load_workflow_spec(args.target)
        spec = deployment_spec_from_module(module)
        name = _slug(args.name or spec.name or _deployment_name_from_workflow(args.target, workflow))
        if _deployment_profile_path(name).exists():
            profile, workflow, module, spec = _deployment_context(name, source=True)
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
                "services": None,
                "options": {},
                "inputs": {},
                "environment": {},
                "timeout": 0.0,
                "execution": "sqlite",
                "ui": False,
                "show_decisions": False,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "python": sys.executable,
            }

    profile["schema_version"] = 2
    if not args.yes and sys.stdin.isatty() and spec.description:
        print(spec.description)
        print()
    values, secrets = _apply_deploy_arguments(profile, args, spec, workflow)
    return _finalize_guided_deployment(profile, spec, values, secrets, args)


def _configure_deployment_command(args) -> int:
    profile, _workflow, _module, spec = _deployment_context(args.name)
    values, secrets = _apply_deploy_arguments(profile, args, spec, _workflow)
    rc = _finalize_guided_deployment(profile, spec, values, secrets, args)
    if rc == 0 and args.restart and args.no_start:
        lifecycle_args = argparse.Namespace(name=args.name, enable=False, dry_run=False)
        return _deployment_lifecycle_command(lifecycle_args, "restart")
    return rc


def _deploy_local_command(args) -> int:
    wf, _module = load_workflow_spec(args.workflow)
    name = _slug(args.name or _deployment_name_from_workflow(args.workflow, wf))
    profile_path = _deployment_profile_path(name)
    if profile_path.exists() and not args.force:
        raise SystemExit(f"Deployment profile already exists: {name}. Use --force to overwrite.")

    inputs = _parse_input_json(args.input_json)
    inputs.update(_parse_inputs(args.input))
    options = _parse_options(args.option)
    llms = normalize_llm_overrides(_parse_inputs(args.llm_for))
    effective_llm_routes(wf, args.llm or "mock", llms)
    store_path = _ensure_store_parent(args.store or _default_deployment_store_path(name))
    log_path = str(Path(args.log or _default_deployment_log_path(name)).expanduser())
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "schema_version": 1,
        "name": name,
        "workflow": args.workflow,
        "cwd": str(Path.cwd()),
        "store": store_path,
        "log": log_path,
        "llm": args.llm,
        "llms": llms,
        "llm_idle_timeout": args.llm_idle_timeout,
        "services": args.services,
        "options": options,
        "inputs": inputs,
        "timeout": args.timeout,
        "execution": "sqlite",
        "ui": args.ui,
        "show_decisions": args.show_decisions,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.executable,
    }
    _write_deployment_artifacts(profile)

    if args.json:
        print(json.dumps({
            **profile,
            "profile": str(profile_path),
            "script": str(_deployment_script_path(name)),
            "systemd_unit": str(_deployment_service_path(name)),
        }, default=str, sort_keys=True))
        return 0

    print(f"Deployment: {name}")
    print(f"Profile: {profile_path}")
    print(f"Store: {store_path}")
    print(f"Log: {log_path}")
    print(f"Run: zippergen run-deployment {name}")
    print(f"Status: zippergen status {name}")
    print(f"Trace: zippergen trace {name}")
    print(f"Systemd unit template: {_deployment_service_path(name)}")
    print("Install later with: mkdir -p ~/.config/systemd/user && cp "
          f"{_deployment_service_path(name)} ~/.config/systemd/user/")
    return 0


def _run_deployment_command(args) -> int:
    profile = _load_deployment_profile(args.name)
    cwd = Path(str(profile.get("cwd") or ".")).expanduser()
    old_cwd = Path.cwd()
    try:
        os.chdir(cwd)
        with _profile_environment(profile):
            return _run_workflow_command(_run_args_from_deployment(profile))
    finally:
        os.chdir(old_cwd)


def _status_command(args) -> int:
    status = _store_status(_resolve_store_arg(args))
    if args.json:
        print(json.dumps(status, default=str))
    else:
        _print_status(status)
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
        metavar="LIFELINE=SPEC",
        help=(
            "Override the LLM for one lifeline; repeat as needed. Use "
            "LIFELINE=inherit to remove an existing override."
        ),
    )
    parser.add_argument("--llm-idle-timeout", type=float, help="Release a managed local LLM after this idle time.")
    parser.add_argument("--store", help="SQLite store path.")
    parser.add_argument("--log", help="Deployment log path.")
    parser.add_argument("--input", action="append", default=[], metavar="name=value", help="Workflow input value.")
    parser.add_argument("--input-json", help="Workflow inputs as a JSON object.")
    parser.add_argument("--option", action="append", default=[], metavar="name=value", help="Workflow setup option.")
    parser.add_argument("--set", action="append", default=[], metavar="field=value", help="Declared deployment field value.")
    parser.add_argument("--services", choices=("fake", "live"), help="Workflow service mode.")
    parser.add_argument("--timeout", type=float, help="Workflow timeout; defaults to 0 (no deadline).")
    parser.add_argument("--ui", action="store_true", default=None, help="Start legacy ZipperChat visualization.")
    parser.add_argument("--show-decisions", action="store_true", default=None, help="Show control events in ZipperChat.")
    parser.add_argument("--yes", action="store_true", help="Accept defaults and existing environment values without prompting.")
    if configure:
        parser.add_argument("--install", dest="no_install", action="store_false", help="Update the managed Python environment.")
        parser.add_argument("--setup", dest="no_setup", action="store_false", help="Run declared one-time setup commands.")
    else:
        parser.add_argument("--no-install", action="store_true", help="Do not create/update the managed Python environment.")
        parser.add_argument("--no-setup", action="store_true", help="Skip declared one-time setup commands.")
    parser.add_argument("--no-doctor", action="store_true", help="Skip readiness checks.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="zippergen")
    sub = ap.add_subparsers(dest="cmd")
    studio = sub.add_parser("studio", help="open the project-aware interactive development workspace")
    studio.add_argument("workflow", nargs="?", help="Initial workflow spec: module:workflow or path.py:workflow")
    studio.add_argument("--project", help="Project root; defaults to discovery from the current directory.")
    studio.add_argument(
        "--command",
        action="append",
        default=[],
        help="Execute one Studio command and exit; repeat for several commands.",
    )

    dev = sub.add_parser("dev", help="run a workflow durably with guided inputs and inline human tasks")
    dev.add_argument("workflow", nargs="?", help="Workflow spec; defaults to the current Studio workflow.")
    dev.add_argument("--resume", action="store_true", help="Resume the current incomplete managed run.")
    dev.add_argument("--run-id", help="Managed run id to resume; requires --resume.")
    dev.add_argument("--project", help="Project root; defaults to discovery from the current directory.")
    dev.add_argument("--llm", metavar="SPEC", help="LLM spec; defaults to the workflow declaration or mock.")
    dev.add_argument(
        "--llm-for",
        action="append",
        default=[],
        metavar="LIFELINE=SPEC",
        help="Override the LLM for one lifeline; repeat as needed.",
    )
    dev.add_argument("--input", action="append", default=[], metavar="name=value", help="Workflow input value.")
    dev.add_argument("--input-json", help="Workflow inputs as a JSON object.")
    dev.add_argument("--option", action="append", default=[], metavar="name=value", help="Workflow setup option.")
    dev.add_argument("--services", choices=("fake", "live"), help="Workflow service mode.")
    dev.add_argument("--timeout", type=float, default=0.0, help="Execution deadline; default 0 means no deadline.")
    dev.add_argument("--yes", action="store_true", help="Use declared input defaults without guided questions.")

    rn = sub.add_parser("run", help="run a workflow locally through SQLite")
    rn.add_argument("workflow", help="Workflow spec: module:workflow or path.py:workflow")
    rn.add_argument("--llm", metavar="SPEC", help="LLM spec: mock, openai:gpt-4o, ollama:qwen2.5:7b, ...")
    rn.add_argument(
        "--llm-for",
        action="append",
        default=[],
        metavar="LIFELINE=SPEC",
        help="Override the LLM for one lifeline; repeat as needed.",
    )
    rn.add_argument("--llm-idle-timeout", type=float, help="Release a managed local LLM after this many idle seconds.")
    rn.add_argument("--store", help="SQLite store path. Defaults to ~/.zippergen/runs/<workflow>.sqlite")
    rn.add_argument("--input", action="append", default=[], metavar="name=value", help="Workflow input value.")
    rn.add_argument("--input-json", help="Workflow inputs as a JSON object.")
    rn.add_argument("--option", action="append", default=[], metavar="name=value", help="Option passed to zippergen_setup(config).")
    rn.add_argument("--services", choices=("fake", "live"), help="Shortcut for --option services=<value>.")
    rn.add_argument("--ui", action="store_true", help="Start legacy ZipperChat visualization; approvals still live in SQLite.")
    rn.add_argument("--timeout", type=float, default=60.0, help="Workflow timeout in seconds; use 0 for no deadline.")
    rn.add_argument("--execution", choices=("sqlite", "memory"), default="sqlite", help="Execution backend.")
    rn.add_argument("--show-decisions", action="store_true", help="Show branch/control events in ZipperChat.")

    show = sub.add_parser("show", help="render a workflow as a code-first semantic view")
    show.add_argument("workflow", help="Workflow spec: module:workflow or path.py:workflow")
    show.add_argument("--detail", choices=DETAILS, default="protocol", help="Amount of implementation detail to include.")
    show.add_argument("--communications", action="store_true", help="Show communication and control flow only.")
    focus = show.add_mutually_exclusive_group()
    focus.add_argument("--agent", help="Show the exact local projection for one agent.")
    focus.add_argument("--agents", help="Comma-separated agents to retain in a boundary-aware focus view.")
    show.add_argument("--format", choices=("code", "json"), default="code", help="Output format.")

    validate = sub.add_parser("validate", help="validate loading, projection, rendering, and deployment metadata")
    validate.add_argument("workflow", help="Workflow spec: module:workflow or path.py:workflow")
    validate.add_argument("--json", action="store_true", help="Print machine-readable validation results.")

    snapshot = sub.add_parser("snapshot", help="save a stable semantic workflow baseline")
    snapshot.add_argument("workflow", help="Workflow spec: module:workflow or path.py:workflow")
    snapshot.add_argument("--output", "-o", help="Write JSON to this path instead of standard output.")

    semantic_diff_parser = sub.add_parser("diff", help="compare two workflows by semantic IR changes")
    semantic_diff_parser.add_argument("before", help="Original workflow spec or semantic snapshot JSON.")
    semantic_diff_parser.add_argument("after", help="Modified workflow spec or semantic snapshot JSON.")
    semantic_diff_parser.add_argument("--format", choices=("code", "json"), default="code", help="Output format.")

    deploy = sub.add_parser("deploy", help="configure, validate, and start a workflow deployment")
    deploy.add_argument("target", help="Workflow spec or existing deployment name.")
    deploy.add_argument("--name", help="Deployment name; defaults to the workflow declaration or workflow name.")
    _add_guided_deployment_arguments(deploy)
    deploy.add_argument("--no-bundle", action="store_true", help="Run from source instead of snapshotting declared files.")
    deploy.add_argument("--no-start", action="store_true", help="Configure the deployment without starting its service.")

    configure = sub.add_parser("configure", help="update a named deployment's persistent configuration")
    configure.add_argument("name", help="Deployment name.")
    _add_guided_deployment_arguments(configure, configure=True)
    configure.add_argument("--restart", action="store_true", help="Restart the service after configuration succeeds.")
    configure.set_defaults(no_start=True, no_bundle=True, no_install=True, no_setup=True)

    dl = sub.add_parser("deploy-local", help="create a named local deployment profile")
    dl.add_argument("workflow", help="Workflow spec: module:workflow or path.py:workflow")
    dl.add_argument("--name", help="Deployment name. Defaults to a slug derived from the workflow.")
    dl.add_argument("--llm", metavar="SPEC", help="LLM spec stored in the deployment profile.")
    dl.add_argument(
        "--llm-for",
        action="append",
        default=[],
        metavar="LIFELINE=SPEC",
        help="Override the LLM for one lifeline; repeat as needed.",
    )
    dl.add_argument("--llm-idle-timeout", type=float, help="Release a managed local LLM after this many idle seconds.")
    dl.add_argument("--store", help="SQLite store path. Defaults to $ZIPPERGEN_HOME/runs/<name>.sqlite")
    dl.add_argument("--log", help="Log path. Defaults to $ZIPPERGEN_HOME/logs/<name>.log")
    dl.add_argument("--input", action="append", default=[], metavar="name=value", help="Workflow input value.")
    dl.add_argument("--input-json", help="Workflow inputs as a JSON object.")
    dl.add_argument("--option", action="append", default=[], metavar="name=value", help="Option passed to zippergen_setup(config).")
    dl.add_argument("--services", choices=("fake", "live"), help="Shortcut stored as services=<value>.")
    dl.add_argument("--ui", action="store_true", help="Start legacy ZipperChat visualization when the deployment runs.")
    dl.add_argument("--timeout", type=float, default=0.0, help="Workflow timeout in seconds; default 0 for no deadline.")
    dl.add_argument("--show-decisions", action="store_true", help="Show branch/control events in ZipperChat.")
    dl.add_argument("--force", action="store_true", help="Overwrite an existing deployment profile.")
    dl.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    rd = sub.add_parser("run-deployment", help="run a named local deployment profile")
    rd.add_argument("name", help="Deployment name.")

    start = sub.add_parser("start", help="start a named deployment as a supervised user service")
    start.add_argument("name", help="Deployment name.")
    start.add_argument("--enable", action="store_true", help="Enable the service to start automatically for this user.")
    start.add_argument("--dry-run", action="store_true", help="Print service-manager commands without running them.")

    stop = sub.add_parser("stop", help="stop a named supervised deployment")
    stop.add_argument("name", help="Deployment name.")
    stop.add_argument("--dry-run", action="store_true", help="Print the service-manager command without running it.")

    restart = sub.add_parser("restart", help="restart a named supervised deployment")
    restart.add_argument("name", help="Deployment name.")
    restart.add_argument("--dry-run", action="store_true", help="Print service-manager commands without running them.")

    logs = sub.add_parser("logs", help="show logs for a named deployment")
    logs.add_argument("name", help="Deployment name.")
    logs.add_argument("--tail", type=int, default=80, help="Number of log lines to show.")
    logs.add_argument("--follow", action="store_true", help="Keep watching the log file.")
    logs.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds for --follow.")

    doctor = sub.add_parser("doctor", help="check a named local deployment for common problems")
    doctor.add_argument("name", help="Deployment name.")
    doctor.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    doctor.add_argument(
        "--no-service-check",
        "--no-systemd",
        dest="no_systemd",
        action="store_true",
        help="Skip live launchd/systemd active-state checks.",
    )

    st = sub.add_parser("status", help="show local SQLite deployment status")
    st.add_argument("deployment", nargs="?", help="Deployment name.")
    st.add_argument("--store", help="SQLite store path.")
    st.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

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

    nt = sub.add_parser("notify", help="run a notification adapter")
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
        help="legacy: run one role as a durable process",
        description="Legacy low-level per-role runner. Prefer `zippergen run` for local deployment.",
    )
    sv.add_argument("--workflow", required=True)
    sv.add_argument("--role", required=True)
    sv.add_argument("--store", required=True)
    sv.add_argument("--input", action="append", default=[], metavar="k=v")
    args = ap.parse_args(argv)

    if args.cmd is None:
        return _studio_command(argparse.Namespace(project=None, workflow=None, command=[]))
    if args.cmd == "studio":
        return _studio_command(args)
    if args.cmd == "dev":
        if args.run_id and not args.resume:
            raise SystemExit("--run-id requires --resume.")
        return _dev_command(args)
    if args.cmd == "run":
        return _run_workflow_command(args)
    if args.cmd == "show":
        return _show_command(args)
    if args.cmd == "validate":
        return _validate_command(args)
    if args.cmd == "snapshot":
        return _snapshot_command(args)
    if args.cmd == "diff":
        return _diff_command(args)
    if args.cmd == "deploy":
        return _deploy_command(args)
    if args.cmd == "configure":
        return _configure_deployment_command(args)
    if args.cmd == "deploy-local":
        return _deploy_local_command(args)
    if args.cmd == "run-deployment":
        return _run_deployment_command(args)
    if args.cmd == "start":
        return _deployment_lifecycle_command(args, "start")
    if args.cmd == "stop":
        return _deployment_lifecycle_command(args, "stop")
    if args.cmd == "restart":
        return _deployment_lifecycle_command(args, "restart")
    if args.cmd == "logs":
        return _logs_command(args)
    if args.cmd == "doctor":
        return _doctor_command(args)
    if args.cmd == "status":
        return _status_command(args)
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
