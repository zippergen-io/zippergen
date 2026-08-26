"""Readiness checks for a project's deployment.

`zippergen deploy check` answers one question: would this deployment start, and
if not, why. Every check reports a status, a name and a detail, so the same
data renders as a table or as JSON.

Extracted from the CLI dispatcher, which is not where domain logic belongs.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from types import ModuleType

from zippergen.deployment import DeploymentSpec, deployment_spec_from_module
from zippergen.connectors import connector_kind_spec, connector_requirements_from_module
from zippergen.assistant_configuration import normalize_assistant_overrides
from zippergen.models import selected_llm_specs
from zippergen.workflow_io import load_workflow_spec
from zippergen.syntax import Workflow
from zippergen.store import (
    list_connector_health,
    list_outstanding_messages,
    list_role_states,
    list_workflow_results,
    open_store,
    open_store_readonly,
    read_history_keep,
    read_last_failure,
    StoreSchemaError,
)
from zippergen.deployment_platform import (
    deployment_launchd_path as _deployment_launchd_path,
    deployment_profile_path as _deployment_profile_path,
    deployment_script_path as _deployment_script_path,
    deployment_service_path as _deployment_service_path,
    installed_launchd_path as _installed_launchd_path,
    installed_systemd_service_path as _installed_systemd_service_path,
    launchd_service_status as _launchd_service_status,
    service_manager as _service_manager,
    slug as _slug,
    systemd_service_status as _systemd_service_status,
    systemctl_command as _systemctl_command,
    systemd_unit_name as _systemd_unit_name,
    zippergen_home as _zippergen_home,
)
from zippergen.deployment_profiles import (
    _default_deployment_log_path,
    _default_deployment_store_path,
    _deployment_environment,
    _field_enabled,
    _load_deployment_profile,
    _profile_environment,
    _profile_field_value,
    _profile_options,
)


def _provenance_check(
    label: str,
    recorded: object,
    current: object,
) -> dict[str, object]:
    """Compare two honest provenance signals without guessing through gaps."""

    before = recorded if isinstance(recorded, dict) else {}
    now = current if isinstance(current, dict) else {}
    before_hash = str(before.get("source_sha256") or "")
    now_hash = str(now.get("source_sha256") or "")
    signal = "source hash"
    if before_hash or now_hash:
        if not before_hash or not now_hash:
            return _doctor_check(
                "warn",
                label,
                "freshness cannot be compared because only one source hash is available",
                freshness="unavailable",
            )
        matches = before_hash == now_hash
    else:
        before_version = str(before.get("version") or "")
        now_version = str(now.get("version") or "")
        if before_version and now_version and "unknown" not in {
            before_version, now_version
        }:
            matches = before_version == now_version
            signal = "package version"
        else:
            return _doctor_check(
                "warn",
                label,
                "freshness cannot be compared from the available provenance",
                freshness="unavailable",
            )
    before_revision = str(before.get("revision") or "")
    now_revision = str(now.get("revision") or "")
    revisions = (
        f"; deployed {before_revision[:12]}, current {now_revision[:12]}"
        if before_revision and now_revision
        else ""
    )
    if label == "ZipperGen runtime":
        stale_detail = (
            f"deployment is running older runtime code ({signal}){revisions}; "
            "fixes in the current checkout are not active. The provenance "
            "difference cannot show its severity; review the diff, then "
            "redeploy to apply it"
        )
    else:
        stale_detail = (
            f"deployment is running its older immutable workflow bundle "
            f"({signal}){revisions}; current source edits are not active. "
            "Review the diff, then redeploy to apply it"
        )
    return _doctor_check(
        "ok" if matches else "warn",
        label,
        (
            f"current ({signal}){revisions}"
            if matches
            else stale_detail
        ),
        freshness="current" if matches else "stale",
        signal=signal,
    )


def deployment_freshness_checks(
    profile: dict[str, object],
) -> list[dict[str, object]]:
    """Compare deployed runtime and workflow snapshots with current sources."""

    from zippergen.deployment_environment import (
        deployment_source_provenance,
        zippergen_runtime_provenance,
    )

    checks = [
        _provenance_check(
            "ZipperGen runtime",
            profile.get("zippergen_runtime"),
            zippergen_runtime_provenance(),
        )
    ]
    source_cwd = Path(str(profile.get("source_cwd") or "")).expanduser()
    source_workflow = str(profile.get("source_workflow") or "")
    if not source_workflow or not source_cwd.is_dir():
        checks.append(
            _doctor_check(
                "warn",
                "workflow source",
                "freshness cannot be compared because the source project is unavailable",
                freshness="unavailable",
            )
        )
        return checks
    previous = Path.cwd()
    try:
        os.chdir(source_cwd)
        workflow, module = load_workflow_spec(source_workflow)
        spec = deployment_spec_from_module(module)
        current = deployment_source_provenance(profile, spec, workflow)
    except (SystemExit, Exception) as exc:
        checks.append(
            _doctor_check(
                "warn",
                "workflow source",
                f"freshness cannot be compared: {type(exc).__name__}: {exc}",
                freshness="unavailable",
            )
        )
    else:
        checks.append(
            _provenance_check(
                "workflow source", profile.get("workflow_source"), current
            )
        )
    finally:
        os.chdir(previous)
    checks.append(_configuration_freshness_check(profile, source_cwd))
    return checks


def _assistant_workspace_checks(
    workflow: Workflow,
    deployment_cwd: Path,
) -> list[dict[str, object]]:
    """Check each assistant workspace from where the service will actually run.

    A workspace is resolved against the root the workflow runs from, and a
    deployment runs from an immutable bundle rather than from the project
    directory. So a relative path names one directory during development and a
    different, absent one once deployed. Validating from the project cannot see
    that; this is the first place that can, and it is still before anything
    starts.
    """

    from zippergen.validation import assistant_actions

    checks: list[dict[str, object]] = []
    for action in assistant_actions(workflow):
        declared = action.workspace
        if not declared:
            continue
        requested = Path(declared).expanduser()
        resolved = (
            requested.resolve()
            if requested.is_absolute()
            else (deployment_cwd / requested).resolve()
        )
        name = f"assistant workspace {action.name}"
        if resolved.is_dir():
            checks.append(_doctor_check("ok", name, str(resolved)))
            continue
        if requested.is_absolute():
            detail = f"does not exist: {resolved}"
        else:
            detail = (
                f"{declared!r} resolves to {resolved}, which does not exist. "
                "A deployment runs from its immutable bundle, not from the "
                "project directory, so a relative workspace means something "
                "different once deployed. Declare an absolute path"
            )
        checks.append(_doctor_check("fail", name, detail))
    return checks


def _configuration_freshness_check(
    profile: Mapping[str, object],
    source_cwd: Path,
) -> dict[str, object]:
    """Report answers edited in the project but not yet deployed.

    The project file is where answers are authored and the profile is what is
    running, so the two can differ the moment somebody edits one. Reporting it
    beside runtime and workflow freshness answers the same question about the
    third thing a deployment is made of.
    """

    from zippergen.workspace import Workspace, WorkspaceError

    stored = _stored_deployment_answers(profile)
    try:
        authored = Workspace(source_cwd).configuration_values()
    except (WorkspaceError, OSError) as exc:
        return _doctor_check(
            "warn",
            "configuration",
            f"freshness cannot be compared: {exc}",
            freshness="unavailable",
        )
    differing = sorted(
        name
        for name in set(stored) | set(authored)
        if stored.get(name) != authored.get(name)
    )
    if not differing:
        return _doctor_check(
            "ok",
            "configuration",
            f"current; {len(stored)} answer(s) match zippergen.toml",
            freshness="current",
        )
    detail = "; ".join(
        f"{name}: deployed {stored.get(name)!r}, project {authored.get(name)!r}"
        for name in differing
    )
    return _doctor_check(
        "warn",
        "configuration",
        f"{detail}. Redeploy to apply the project's answers",
        freshness="stale",
    )


def _stored_deployment_answers(
    profile: Mapping[str, object],
) -> dict[str, object]:
    """The answers this deployment is actually running."""

    raw = profile.get("deployment_spec")
    if not isinstance(raw, Mapping) or not raw.get("fields"):
        return {}
    from zippergen.deployment import normalize_deployment_spec

    spec = normalize_deployment_spec(dict(raw))
    answers: dict[str, object] = {}
    for field in spec.fields:
        if field.secret:
            continue
        # A field is delivered to whichever section its target names, so all
        # three are read. Leaving out "environment" made every non-secret
        # env field report drift forever: the project had an answer and the
        # deployment appeared to have none.
        for section in ("options", "inputs", "environment"):
            values = profile.get(section)
            if isinstance(values, Mapping) and field.target_name in values:
                answers[field.name] = values[field.target_name]
    return answers


def _safe_json_loads(value):
    if value is None:
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def _store_status(store_path: str) -> dict[str, object]:
    path = Path(store_path).expanduser()
    if not path.exists():
        return {
            "store": str(path),
            "exists": False,
            "state": "missing",
            "summary": "store does not exist",
        }

    # Opening a store and reading it answer one question -- can this durable
    # state be inspected? -- so they share one answer. A file whose header is
    # intact but whose pages are damaged opens and then fails on the first
    # SELECT, and splitting the answer is what made `deploy status` traceback
    # on a store that `deploy check` reported cleanly.
    try:
        conn = open_store_readonly(path)
        try:
            roles = list_role_states(conn)
            outstanding = list_outstanding_messages(conn)
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
            connectors = list_connector_health(conn)
            history = {
                "rows": int(
                    conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
                ),
                "keep": read_history_keep(conn),
            }
            last_failure = read_last_failure(conn)
        finally:
            conn.close()
    except (OSError, StoreSchemaError, sqlite3.Error) as exc:
        return {
            "store": str(path),
            "exists": True,
            "state": "incompatible",
            "summary": f"cannot inspect durable state: {exc}",
        }

    if pending_tasks:
        state = "waiting"
        summary = f"waiting for {len(pending_tasks)} human task(s)"
    elif results:
        state = "done"
        summary = f"{len(results)} workflow result(s)"
    elif roles:
        state = "active"
        summary = f"{len(roles)} role(s) in progress; no result yet"
    else:
        state = "empty"
        summary = "store is initialized but empty"

    # `last_failure` is deliberately independent of the bounded trace, but it
    # is not necessarily the current outcome.  A later committed workflow
    # result proves that execution recovered.  Preserve the diagnostic while
    # making that relationship explicit for every status consumer.
    if last_failure is not None:
        failure_at = last_failure.get("recorded_at")
        recovered_at = max(
            (
                result["updated_at"]
                for result in results
                if isinstance(result.get("updated_at"), (int, float))
                and not isinstance(result.get("updated_at"), bool)
            ),
            default=None,
        )
        historical = (
            isinstance(failure_at, (int, float))
            and not isinstance(failure_at, bool)
            and recovered_at is not None
            and recovered_at > failure_at
        )
        last_failure = {**last_failure, "historical": historical}
        if historical:
            last_failure["recovered_at"] = recovered_at

    return {
        "store": str(path),
        "exists": True,
        "state": state,
        "summary": summary,
        "roles": [
            {
                "role": row["role"],
                "status": row["status"],
                "detail": row["detail"],
                "steps": row["steps"],
                "updated_at": row["updated_at"],
            }
            for row in roles
        ],
        "outstanding_messages": outstanding,
        "connectors": connectors,
        "pending_human_tasks": pending_tasks,
        "done_human_task_count": done_task_count,
        "workflow_results": results,
        "history": history,
        "last_failure": last_failure,
    }


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

    def option(self, name: str, default: object = None) -> object:
        return self.options.get(name, default)


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


def _required_model_provider_secrets(profile: dict[str, object]) -> dict[str, str]:
    from zippergen.provider_connections import (
        provider_credential_field,
        provider_environment_name,
        provider_standard_environment,
        split_model_spec,
    )

    required: dict[str, str] = {}
    for spec in selected_llm_specs(profile.get("llm"), profile.get("llms")):
        try:
            kind, connection, _model = split_model_spec(spec)
        except ValueError:
            continue
        field = provider_credential_field(kind)
        if field is None:
            continue
        if connection:
            required[provider_environment_name(connection, field)] = connection
        elif standard := provider_standard_environment(kind):
            required[standard] = standard
    return required


def _systemd_active_check(name: str) -> dict[str, object]:
    status = _systemd_service_status(name)
    state = str(status.get("state") or "unknown")
    extra = {key: value for key, value in status.items() if key != "detail"}
    if state == "running" and bool(status.get("healthy")):
        return _doctor_check(
            "ok",
            "service",
            str(status.get("detail") or f"{_systemd_unit_name(name)} is running"),
            **extra,
        )
    if state in {"restarting"} or status.get("active_state") == "failed":
        return _doctor_check(
            "fail",
            "service",
            str(status.get("detail") or "the service is unhealthy"),
            **extra,
        )
    return _doctor_check(
        "warn",
        "service",
        (
            str(status.get("detail"))
            if state == "unknown"
            else "the service is not running; start it with 'zippergen deploy start'"
        ),
        **extra,
    )


def _systemd_enabled_check(name: str) -> dict[str, object]:
    """Say whether the user unit will be started with its user manager."""

    unit = _systemd_unit_name(name)
    try:
        result = subprocess.run(
            _systemctl_command("is-enabled", unit),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        return _doctor_check("warn", "systemd autostart", "systemctl was not found")
    except subprocess.TimeoutExpired:
        return _doctor_check("warn", "systemd autostart", "systemctl timed out")

    state = (result.stdout or result.stderr or "").strip() or f"exit {result.returncode}"
    if result.returncode == 0:
        return _doctor_check(
            "ok",
            "systemd autostart",
            f"{unit} is {state}",
            state=state,
        )
    return _doctor_check(
        "warn",
        "systemd autostart",
        (
            f"{unit} is {state}; stop it, then run "
            "'zippergen deploy start --enable'"
        ),
        state=state,
    )


def _systemd_linger_check() -> dict[str, object]:
    """Check whether user services survive logout and start during boot."""

    user = os.environ.get("USER") or str(os.getuid())
    try:
        result = subprocess.run(
            [
                "loginctl",
                "show-user",
                str(os.getuid()),
                "--property=Linger",
                "--value",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        return _doctor_check(
            "warn",
            "systemd linger",
            "loginctl was not found; verify that this user manager survives logout",
        )
    except subprocess.TimeoutExpired:
        return _doctor_check(
            "warn",
            "systemd linger",
            "loginctl timed out; verify that this user manager survives logout",
        )

    value = (result.stdout or result.stderr or "").strip().casefold()
    if result.returncode == 0 and value == "yes":
        return _doctor_check(
            "ok",
            "systemd linger",
            "enabled; user services can remain available without a login session",
            enabled=True,
        )
    return _doctor_check(
        "warn",
        "systemd linger",
        (
            "disabled or unavailable; the deployment may stop after logout or "
            f"fail to start at boot. Enable it with 'loginctl enable-linger {user}' "
            "if permitted by this server's policy"
        ),
        enabled=False,
    )


def _launchd_active_check(name: str) -> dict[str, object]:
    status = _launchd_service_status(name)
    if status["state"] in {"running", "completed"}:
        kind = "ok"
    elif status["state"] == "restarting":
        kind = "fail"
    else:
        kind = "warn"
        status = {
            **status,
            "detail": (
                "the service is not running; start it with "
                "'zippergen deploy start'"
            ),
        }
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
    before_start: bool = False,
    profile_override: dict[str, object] | None = None,
    check_artifacts: bool = True,
) -> list[dict[str, object]]:
    """Report on one deployment.

    ``before_start`` is set by the commands that are themselves about to
    install and start the service. Two checks -- the log file and the
    installed unit -- describe state that only exists afterwards, so warning
    about them there would be reporting a problem the same command fixes a
    second later.
    """

    profile_path = _deployment_profile_path(name)
    checks: list[dict[str, object]] = []
    profile = (
        dict(profile_override)
        if profile_override is not None
        else _load_deployment_profile(name)
    )
    profile_name = str(profile.get("name") or name)
    checks.append(_doctor_check(
        "ok",
        "profile",
        (
            "candidate configuration is complete"
            if profile_override is not None
            else f"loaded {profile_path}"
        ),
        path=str(profile_path),
    ))
    home = _zippergen_home()
    if home.is_symlink():
        checks.append(_doctor_check(
            "fail", "deployment home permissions", f"directory is a symlink: {home}"
        ))
    elif home.exists() and home.stat().st_mode & 0o077:
        checks.append(_doctor_check(
            "fail",
            "deployment home permissions",
            f"permissions are not private: {home}; run 'zg deploy check --repair-permissions'",
        ))
    elif home.exists():
        checks.append(_doctor_check(
            "ok", "deployment home permissions", f"owner-only directory: {home}"
        ))
    if profile_override is None:
        if profile_path.is_symlink():
            checks.append(_doctor_check(
                "fail", "profile permissions", f"profile is a symlink: {profile_path}"
            ))
        elif profile_path.stat().st_mode & 0o077:
            checks.append(_doctor_check(
                "fail",
                "profile permissions",
                f"permissions are not private: {profile_path}; run 'zg deploy check --repair-permissions'",
            ))
        else:
            checks.append(_doctor_check(
                "ok", "profile permissions", f"owner-only file: {profile_path}"
            ))
    checks.extend(deployment_freshness_checks(profile))

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
        status = _store_status(str(store_path))
        store_state = str(status["state"])
        checks.append(_doctor_check(
            "fail" if store_state == "incompatible" else "ok",
            "sqlite store",
            str(status["summary"]),
            state=store_state,
        ))
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

    if log_path.is_symlink():
        checks.append(_doctor_check(
            "fail", "log file permissions", f"log is a symlink: {log_path}"
        ))
    elif log_path.exists() and log_path.stat().st_mode & 0o077:
        checks.append(_doctor_check(
            "fail",
            "log file permissions",
            f"permissions are not private: {log_path}; run 'zg deploy check --repair-permissions'",
        ))
    elif log_path.exists():
        checks.append(_doctor_check("ok", "log file", str(log_path)))
    elif not before_start:
        checks.append(_doctor_check("warn", "log file", f"log does not exist yet: {log_path}"))

    if check_artifacts:
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
    if check_artifacts and installed_path.exists():
        checks.append(_doctor_check("ok", f"{manager or 'service'} installed", str(installed_path)))
    elif check_artifacts and not before_start:
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
        if secret_file.is_symlink():
            checks.append(_doctor_check("fail", "secrets file", f"file is a symlink: {secret_file}"))
        elif not secret_file.exists():
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
            checks.extend(_assistant_workspace_checks(workflow, cwd))
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
        # Use the deployment's environment, not the shell running `zg deploy`.
        # In particular, this makes executable discovery identical here and
        # inside a supervised service.
        with _profile_environment(profile):
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
            "connector assignments",
            "deployment connector assignments must be an object",
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
                    "required, not assigned yet"
                    if requirement.required
                    else "optional, not assigned",
                ))
                continue
            if not isinstance(raw_binding, dict):
                checks.append(_doctor_check(
                    "fail",
                    f"connector {requirement.name}",
                    "assignment must be an object",
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
            spec = connector_kind_spec(kind)
            if spec is None:
                checks.append(_doctor_check(
                    "warn",
                    f"connector {requirement.name}",
                    f"{kind} is bound but has no live readiness adapter yet",
                ))
                continue
            readiness = spec.readiness(
                requirement, raw_binding, environment, live_connectors
            )
            checks.append(_doctor_check(
                readiness.status,
                f"connector {requirement.name}",
                readiness.detail,
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
    declared_values["__llm_field_names__"] = ()
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
    for secret_name, connection in sorted(
        _required_model_provider_secrets(profile).items()
    ):
        if environment.get(secret_name):
            if secret_name not in declared_secret_names:
                checks.append(
                    _doctor_check(
                        "ok",
                        f"provider credential {connection}",
                        "configured in private deployment storage",
                    )
                )
        else:
            checks.append(
                _doctor_check(
                    "fail",
                    f"provider credential {connection}",
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

    if include_systemd and manager == "systemd":
        checks.append(_systemd_linger_check())
    if include_systemd and installed_path.exists():
        if manager == "launchd":
            checks.append(_launchd_active_check(profile_name))
        elif manager == "systemd":
            checks.append(_systemd_enabled_check(profile_name))
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
        )
        with _profile_environment(profile):
            checks.extend(_call_doctor_hook(module, config))

    return checks
