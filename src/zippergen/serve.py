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
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

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


def _systemd_user_dir() -> Path:
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))).expanduser()
    return config_home / "systemd" / "user"


def _systemd_unit_name(name: str) -> str:
    return f"zippergen-{_slug(name)}.service"


def _installed_systemd_service_path(name: str) -> Path:
    return _systemd_user_dir() / _systemd_unit_name(name)


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


def _deployment_name_from_workflow(workflow_spec: str, wf: Workflow) -> str:
    base = workflow_spec.split(":", 1)[0]
    stem = Path(base).stem if _looks_like_path(base) else base.rsplit(".", 1)[-1]
    return _slug(f"{stem}-{wf.name}")


def _jsonable_kv_pairs(values: dict[str, object]) -> list[str]:
    return [f"{key}={json.dumps(value, default=str)}" for key, value in sorted(values.items())]


def _run_args_from_deployment(profile: dict[str, object]):
    timeout_raw = profile.get("timeout", 0.0)
    timeout = float(timeout_raw) if isinstance(timeout_raw, (int, float, str)) else 0.0
    return argparse.Namespace(
        workflow=str(profile["workflow"]),
        llm=profile.get("llm") or None,
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
    Path(str(profile["store"])).expanduser().parent.mkdir(parents=True, exist_ok=True)
    Path(str(profile["log"])).expanduser().parent.mkdir(parents=True, exist_ok=True)
    _deployments_dir().mkdir(parents=True, exist_ok=True)

    profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    script_path.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        f"cd {shlex.quote(str(profile['cwd']))}\n"
        f"exec {_deployment_command(name, python_executable=str(profile.get('python') or sys.executable))}\n"
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


def _deployment_lifecycle_command(args, action: str) -> int:
    profile = _load_deployment_profile(args.name)
    name = str(profile["name"])
    unit = _systemd_unit_name(name)
    if action in {"start", "restart"}:
        target = _install_systemd_unit(profile, dry_run=args.dry_run)
        if not args.dry_run:
            print(f"Installed systemd unit: {target}")
        _run_systemctl(_systemctl_command("daemon-reload"), dry_run=args.dry_run)
        if action == "start" and args.enable:
            _run_systemctl(_systemctl_command("enable", unit), dry_run=args.dry_run)
    _run_systemctl(_systemctl_command(action, unit), dry_run=args.dry_run)
    if args.dry_run:
        return 0
    done = {"start": "Started", "stop": "Stopped", "restart": "Restarted"}[action]
    print(f"{done} deployment {name} ({unit}).")
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
    raw = profile.get("options") or {}
    return raw if isinstance(raw, dict) else {}


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

    installed_path = _installed_systemd_service_path(profile_name)
    if installed_path.exists():
        checks.append(_doctor_check("ok", "systemd installed", str(installed_path)))
    else:
        checks.append(_doctor_check("warn", "systemd installed", f"service is not installed: {installed_path}"))

    workflow = None
    module = None
    if cwd.exists() and cwd.is_dir() and profile.get("workflow"):
        old_cwd = Path.cwd()
        try:
            os.chdir(cwd)
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

    if include_systemd and installed_path.exists():
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
    if args.llm:
        wf.configure(args.llm, **configure_kwargs)
    else:
        wf.configure(**configure_kwargs)

    result = wf(**inputs)
    print(json.dumps({"result": result}, default=str))
    if args.ui and sys.stdin.isatty():
        input("ZipperChat running at http://localhost:8765. Press Enter to exit. ")
    return 0


def _deploy_local_command(args) -> int:
    wf, _module = load_workflow_spec(args.workflow)
    name = _slug(args.name or _deployment_name_from_workflow(args.workflow, wf))
    profile_path = _deployment_profile_path(name)
    if profile_path.exists() and not args.force:
        raise SystemExit(f"Deployment profile already exists: {name}. Use --force to overwrite.")

    inputs = _parse_input_json(args.input_json)
    inputs.update(_parse_inputs(args.input))
    options = _parse_options(args.option)
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="zippergen")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rn = sub.add_parser("run", help="run a workflow locally through SQLite")
    rn.add_argument("workflow", help="Workflow spec: module:workflow or path.py:workflow")
    rn.add_argument("--llm", metavar="SPEC", help="LLM spec: mock, openai:gpt-4o, ollama:qwen2.5:7b, ...")
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

    dl = sub.add_parser("deploy-local", help="create a named local deployment profile")
    dl.add_argument("workflow", help="Workflow spec: module:workflow or path.py:workflow")
    dl.add_argument("--name", help="Deployment name. Defaults to a slug derived from the workflow.")
    dl.add_argument("--llm", metavar="SPEC", help="LLM spec stored in the deployment profile.")
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

    start = sub.add_parser("start", help="start a named deployment as a systemd user service")
    start.add_argument("name", help="Deployment name.")
    start.add_argument("--enable", action="store_true", help="Enable the service to start automatically for this user.")
    start.add_argument("--dry-run", action="store_true", help="Print the systemd commands without running them.")

    stop = sub.add_parser("stop", help="stop a named deployment systemd user service")
    stop.add_argument("name", help="Deployment name.")
    stop.add_argument("--dry-run", action="store_true", help="Print the systemd command without running it.")

    restart = sub.add_parser("restart", help="restart a named deployment systemd user service")
    restart.add_argument("name", help="Deployment name.")
    restart.add_argument("--dry-run", action="store_true", help="Print the systemd commands without running them.")

    logs = sub.add_parser("logs", help="show logs for a named deployment")
    logs.add_argument("name", help="Deployment name.")
    logs.add_argument("--tail", type=int, default=80, help="Number of log lines to show.")
    logs.add_argument("--follow", action="store_true", help="Keep watching the log file.")
    logs.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds for --follow.")

    doctor = sub.add_parser("doctor", help="check a named local deployment for common problems")
    doctor.add_argument("name", help="Deployment name.")
    doctor.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    doctor.add_argument("--no-systemd", action="store_true", help="Skip live systemd active-state checks.")

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

    if args.cmd == "run":
        return _run_workflow_command(args)
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
