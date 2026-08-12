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
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from types import ModuleType

from zippergen.deployment import DeploymentSpec, deployment_spec_from_module
from zippergen.connectors import connector_requirements_from_module
from zippergen.assistant_configuration import normalize_assistant_overrides
from zippergen.models import selected_llm_specs
from zippergen.workflow_io import load_workflow_spec
from zippergen.syntax import Workflow
from zippergen.store import (
    list_outstanding_messages,
    list_role_states,
    list_workflow_results,
    open_store,
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
    systemctl_command as _systemctl_command,
    systemd_unit_name as _systemd_unit_name,
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


def _safe_json_loads(value):
    if value is None:
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


_MODEL_PROVIDER_SECRETS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}


def _model_provider(value: object) -> str:
    provider = str(value or "mock").partition(":")[0].strip().lower()
    return {"claude": "anthropic", "ollama": "local"}.get(provider, provider)


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
    finally:
        conn.close()

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

    return {
        "store": str(path),
        "exists": True,
        "state": state,
        "summary": summary,
        "roles": [
            {
                "role": row["role"],
                "status": row["status"],
                "steps": row["steps"],
                "updated_at": row["updated_at"],
            }
            for row in roles
        ],
        "outstanding_messages": outstanding,
        "pending_human_tasks": pending_tasks,
        "done_human_task_count": done_task_count,
        "workflow_results": results,
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


def _required_model_provider_secrets(profile: dict[str, object]) -> set[str]:
    return {
        secret
        for model in selected_llm_specs(profile.get("llm"), profile.get("llms"))
        if (secret := _MODEL_PROVIDER_SECRETS.get(_model_provider(model)))
    }


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
        )
        with _profile_environment(profile):
            checks.extend(_call_doctor_hook(module, config))

    return checks
