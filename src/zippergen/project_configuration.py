"""Inspect and manage one project's portable and site configuration."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

from zippergen.assistant_configuration import (
    assistant_targets,
    project_assistant_routing,
    resolved_assistant_actions,
)
from zippergen.connector_wiring import human_action_sites
from zippergen.connectors import connector_requirements_from_module
from zippergen.models import project_model_routing, selected_llm_specs
from zippergen.rendering import TerminalRenderer
from zippergen.semantic import workflow_semantics
from zippergen.syntax import LLMAction, Workflow
from zippergen.workspace import Workspace, WorkspaceError


Check = dict[str, object]


def _check(status: str, name: str, detail: str) -> Check:
    return {"status": status, "name": name, "detail": detail}


def _provider(spec: str) -> str:
    value = spec.partition(":")[0].strip().casefold()
    return {"claude": "anthropic", "ollama": "local"}.get(value, value)


def model_credential_name(spec: str) -> str | None:
    """Return the private credential name used by one model specification."""

    provider = _provider(spec)
    return {
        "openai": "OPENAI_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }.get(provider)


def _project_model_source(
    project: dict[str, object],
    name: str,
    merged: dict[str, str],
) -> str:
    raw_models = project.get("models") or {}
    configurations = (
        raw_models.get("configurations") or {}
        if isinstance(raw_models, dict)
        else {}
    )
    raw = configurations.get(name) if isinstance(configurations, dict) else None
    project_keys = set(raw) if isinstance(raw, dict) else set()
    site_keys = set(merged) - project_keys
    if project_keys and site_keys:
        return "project + site"
    if project_keys:
        return "project"
    if name == "mock":
        return "built-in"
    return "site"


def _model_targets(workflow: Workflow, module: ModuleType) -> list[str]:
    sites = workflow_semantics(workflow, module).get("action_sites") or []
    targets: list[str] = []
    for site in sites if isinstance(sites, list) else []:
        if not isinstance(site, dict) or site.get("kind") != "llm":
            continue
        participant = str(site.get("lifeline") or "")
        action = str(site.get("action") or "")
        for value in (participant, f"{participant}.{action}"):
            if value and value not in targets:
                targets.append(value)
    return targets


def _used_connector_names(
    assignments: dict[str, dict[str, str]],
    bindings: dict[str, str],
) -> set[str]:
    return {
        *bindings.values(),
        *assignments.get("lifelines", {}).values(),
        *assignments.get("actions", {}).values(),
    }


def _static_connector_checks(
    workflow: Workflow,
    module: ModuleType,
    assignments: dict[str, dict[str, str]],
    bindings: dict[str, str],
    configurations: dict[str, dict[str, str]],
) -> list[Check]:
    checks: list[Check] = []
    human_sites = human_action_sites(workflow, module)
    human_targets = {
        *human_sites,
        *(
            f"{participant}.{action}"
            for participant, actions in human_sites.items()
            for action in actions
        ),
    }
    for group in ("lifelines", "actions"):
        for target, configuration in assignments[group].items():
            if target not in human_targets:
                checks.append(
                    _check(
                        "fail",
                        f"connector assignment {target}",
                        "target has no human action",
                    )
                )
            elif configuration not in configurations:
                checks.append(
                    _check(
                        "fail",
                        f"connector assignment {target}",
                        f"configuration {configuration!r} does not exist",
                    )
                )
            elif configurations[configuration].get("provider") != "telegram":
                checks.append(
                    _check(
                        "fail",
                        f"connector assignment {target}",
                        f"{configuration!r} cannot deliver a human action",
                    )
                )
            else:
                checks.append(
                    _check(
                        "ok",
                        f"connector assignment {target}",
                        configuration,
                    )
                )

    for requirement in connector_requirements_from_module(module):
        configuration = bindings.get(requirement.name)
        if configuration is None:
            checks.append(
                _check(
                    "fail" if requirement.required else "warn",
                    f"connector binding {requirement.name}",
                    "required binding is missing"
                    if requirement.required
                    else "optional binding is missing",
                )
            )
            continue
        selected = configurations.get(configuration)
        if selected is None:
            checks.append(
                _check(
                    "fail",
                    f"connector binding {requirement.name}",
                    f"configuration {configuration!r} does not exist",
                )
            )
        elif selected.get("kind") != requirement.kind:
            checks.append(
                _check(
                    "fail",
                    f"connector binding {requirement.name}",
                    f"requires {requirement.kind}; {configuration!r} is "
                    f"{selected.get('kind') or 'untyped'}",
                )
            )
        else:
            checks.append(
                _check(
                    "ok",
                    f"connector binding {requirement.name}",
                    configuration,
                )
            )
    return checks


@contextmanager
def _temporary_environment(values: dict[str, str]):
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _live_model_check(spec: str, environment: dict[str, str]) -> None:
    from zippergen.backends import backend_from_spec, load_scripted_script

    provider = _provider(spec)
    if provider == "mock":
        return
    if provider == "scripted":
        path = spec.partition(":")[2]
        load_scripted_script(path)
        return
    action = LLMAction(
        name="zippergen_readiness",
        inputs=(),
        outputs=(("reply", str),),
        system_prompt="This is a connectivity check.",
        user_prompt="Reply with OK.",
        parse_format="text",
    )
    with _temporary_environment(environment):
        backend, _label = backend_from_spec(
            spec,
            fallback=lambda _action, _inputs: {"reply": "OK"},
        )
        backend(action, {})


def configuration_report(
    workspace: Workspace,
    *,
    live: bool = False,
    include_site_checks: bool = True,
    model_names: tuple[str, ...] = (),
    assistant_names: tuple[str, ...] = (),
    connector_names: tuple[str, ...] = (),
) -> dict[str, object]:
    """Return the effective, secret-free project configuration and checks."""

    manifest = workspace.project_manifest()
    try:
        workflow_spec = workspace.resolve_workflow()
        workflow, module = _load_project_workflow(workspace, workflow_spec)
        workflow_error = None
    except (SystemExit, Exception) as exc:
        workflow_spec = None
        workflow = None
        module = None
        workflow_error = f"{type(exc).__name__}: {exc}"

    model_configurations = workspace.model_configurations()
    model_rows = [
        {
            "name": name,
            "spec": values.get("spec", ""),
            "idle_timeout": values.get("idle_timeout"),
            "source": _project_model_source(manifest, name, values),
        }
        for name, values in sorted(model_configurations.items())
    ]
    model_profile: dict[str, Any] = {
        "default": "mock",
        "lifelines": {},
        "actions": {},
    }
    resolved_models: dict[str, object] = {
        "default": "mock",
        "overrides": {},
        "idle_timeouts": {},
    }
    checks: list[Check] = []
    if workflow_error:
        checks.append(_check("fail", "workflow", workflow_error))
    elif workflow_spec is not None and workflow is not None and module is not None:
        try:
            has_model_profile = workspace.has_model_assignment_profile(workflow_spec)
            if has_model_profile:
                model_profile = workspace.model_assignment_profile(workflow_spec)
            from zippergen.durable_runs import default_llm_spec

            routing = project_model_routing(
                workspace,
                workflow_spec,
                workflow,
                fallback_default=default_llm_spec(module),
            )
        except (WorkspaceError, SystemExit, ValueError) as exc:
            checks.append(_check("fail", "model assignments", str(exc)))
        else:
            if not has_model_profile:
                model_profile = {
                    "default": f"workflow default ({routing.default_spec})",
                    "lifelines": {},
                    "actions": {},
                }
            resolved_models = {
                "default": routing.default_spec,
                "overrides": routing.overrides,
                "idle_timeouts": routing.idle_timeouts,
            }
            checks.append(
                _check(
                    "ok",
                    "model assignments",
                    f"{len(routing.overrides)} explicit assignment(s)",
                )
            )

    assistant_configurations = workspace.assistant_configurations()
    assistant_profile: dict[str, object] = {
        "default": "",
        "lifelines": {},
        "actions": {},
    }
    resolved_assistants: list[dict[str, object]] = []
    if workflow_spec is not None and workflow is not None and module is not None:
        try:
            assistant_profile = workspace.assistant_assignment_profile(
                workflow_spec
            )
            assistant_routing = project_assistant_routing(
                workspace,
                workflow_spec,
                workflow,
                module=module,
            )
            effective = resolved_assistant_actions(
                workflow,
                assistant_routing,
                module=module,
                assignments=assistant_profile,
            )
        except (WorkspaceError, SystemExit, ValueError) as exc:
            checks.append(_check("fail", "assistant assignments", str(exc)))
        else:
            resolved_assistants = [
                {
                    "target": item.target,
                    "backend": item.backend,
                    "configuration": item.configuration,
                    "source": item.source,
                    "access": item.access,
                    "external_tools": item.external_tools,
                    "shell": item.shell,
                }
                for item in effective
            ]
            missing = [item.target for item in effective if item.backend is None]
            if missing:
                checks.append(
                    _check(
                        "fail",
                        "assistant assignments",
                        "no backend selected for: " + ", ".join(missing),
                    )
                )
            else:
                checks.append(
                    _check(
                        "ok",
                        "assistant assignments",
                        f"{len(effective)} assistant action(s) resolved",
                    )
                )

    connector_configurations = workspace.connector_configurations()
    connector_assignments = (
        workspace.connector_assignment_profile(workflow_spec)
        if workflow_spec is not None
        else {"lifelines": {}, "actions": {}}
    )
    connector_bindings = (
        workspace.connector_binding_profile(workflow_spec)
        if workflow_spec is not None
        else {}
    )
    if workflow is not None and module is not None:
        checks.extend(
            _static_connector_checks(
                workflow,
                module,
                connector_assignments,
                connector_bindings,
                connector_configurations,
            )
        )

    site_facts: list[dict[str, object]] = []
    if include_site_checks:
        specs = selected_llm_specs(
            resolved_models["default"],
            resolved_models["overrides"],
        )
        specs = tuple(dict.fromkeys([
            *specs,
            *(
                model_configurations[name]["spec"]
                for name in model_names
                if name in model_configurations
            ),
        ]))
        environment = workspace.development_provider_environment(specs)
        for spec in specs:
            provider = _provider(spec)
            secret = model_credential_name(spec)
            if secret:
                available = bool(environment.get(secret) or os.environ.get(secret))
                site_facts.append(
                    {
                        "kind": "model credential",
                        "name": secret,
                        "available": available,
                    }
                )
                checks.append(
                    _check(
                        "ok" if available else "fail",
                        f"model credential {secret}",
                        "available" if available else "missing on this computer",
                    )
                )
            if provider == "scripted":
                path = Path(spec.partition(":")[2]).expanduser()
                checks.append(
                    _check(
                        "ok" if path.is_file() else "fail",
                        f"scripted model {path}",
                        "response file exists" if path.is_file() else "file is missing",
                    )
                )
        used_connectors = _used_connector_names(
            connector_assignments,
            connector_bindings,
        )
        used_connectors.update(connector_names)
        used_providers = {
            str(connector_configurations.get(name, {}).get("provider") or "")
            for name in used_connectors
        }
        for provider, field, label in (
            ("telegram", "bot_token", "Telegram bot token"),
            ("google", "authorized_user_json", "Google authorization"),
        ):
            if provider not in used_providers:
                continue
            available = bool(workspace.connector_provider_secret(provider, field))
            site_facts.append(
                {"kind": "connector credential", "name": label, "available": available}
            )
            checks.append(
                _check(
                    "ok" if available else "fail",
                    f"connector credential {label}",
                    "available" if available else "missing on this computer",
                )
            )

        selected_assistant_backends = {
            str(item.get("backend"))
            for item in resolved_assistants
            if item.get("backend") in {"codex", "claude"}
        }
        selected_assistant_backends.update(
            assistant_configurations[name]["backend"]
            for name in assistant_names
            if name in assistant_configurations
        )
        if selected_assistant_backends:
            from zippergen.assistant_backends import check_cli_assistant

            for backend in sorted(selected_assistant_backends):
                result = check_cli_assistant(backend)
                site_facts.append(
                    {
                        "kind": "assistant CLI",
                        "name": backend,
                        "available": result.supported,
                    }
                )
                checks.append(
                    _check(
                        "ok" if result.supported else "fail",
                        f"assistant CLI {backend}",
                        result.detail,
                    )
                )

        if live:
            unique_specs = dict.fromkeys(specs)
            for spec in unique_specs:
                try:
                    _live_model_check(spec, environment)
                except Exception as exc:
                    checks.append(
                        _check("fail", f"live model {spec}", f"{type(exc).__name__}: {exc}")
                    )
                else:
                    checks.append(_check("ok", f"live model {spec}", "reachable"))
            _append_live_connector_checks(
                checks,
                workspace,
                workflow,
                module,
                connector_configurations,
                connector_assignments,
                connector_bindings,
                extra_names=connector_names,
            )

    return {
        "site_root": str(workspace.directory),
        "project": {
            "name": manifest.get("name"),
            "root": str(workspace.root),
            "manifest": str(workspace.manifest_path),
            "workflow": workflow_spec,
            "specification": str(workspace.specification_path),
        },
        "models": {
            "configurations": model_rows,
            "assignments": model_profile,
            "resolved": resolved_models,
        },
        "assistants": {
            "configurations": [
                {"name": name, **values}
                for name, values in sorted(assistant_configurations.items())
            ],
            "assignments": assistant_profile,
            "resolved": resolved_assistants,
        },
        "connectors": {
            "configurations": [
                {"name": name, **values}
                for name, values in sorted(connector_configurations.items())
            ],
            "bindings": connector_bindings,
            "assignments": connector_assignments,
        },
        "site_facts": site_facts,
        "checks": checks,
        "valid": not any(item["status"] == "fail" for item in checks),
    }


def _load_project_workflow(workspace: Workspace, workflow_spec: str):
    from zippergen.workflow_io import load_workflow_spec

    return load_workflow_spec(workspace.absolute_spec(workflow_spec))


def _append_live_connector_checks(
    checks: list[Check],
    workspace: Workspace,
    workflow: Workflow | None,
    module: ModuleType | None,
    configurations: dict[str, dict[str, str]],
    assignments: dict[str, dict[str, str]],
    bindings: dict[str, str],
    *,
    extra_names: tuple[str, ...] = (),
) -> None:
    used = _used_connector_names(assignments, bindings)
    used.update(extra_names)
    telegram = workspace.connector_provider_secret("telegram", "bot_token")
    for name in sorted(used):
        configuration = configurations.get(name) or {}
        provider = configuration.get("provider")
        if provider == "telegram" and telegram:
            try:
                from zippergen.telegram_notify import TelegramBotClient

                client = TelegramBotClient(telegram, timeout=5)
                client.request("getMe")
                client.request("getChat", chat_id=configuration.get("chat_id"))
            except Exception as exc:
                checks.append(
                    _check("fail", f"live connector {name}", f"{type(exc).__name__}: {exc}")
                )
            else:
                checks.append(_check("ok", f"live connector {name}", "reachable"))
    if workflow is None or module is None:
        return
    requirements = connector_requirements_from_module(module)
    google_pairs = [
        (item.kind, item.access)
        for item in requirements
        if bindings.get(item.name) in used
        and item.kind in {"gmail", "google-sheets"}
    ]
    for name in extra_names:
        configuration = configurations.get(name) or {}
        kind = configuration.get("kind")
        pair = (str(kind), "read-only")
        if kind in {"gmail", "google-sheets"} and pair not in google_pairs:
            google_pairs.append(pair)
    credential = workspace.connector_provider_secret("google", "authorized_user_json")
    if google_pairs and credential:
        try:
            from zippergen.google_auth import (
                check_google_authorization,
                google_scopes_for_access,
            )

            refreshed = check_google_authorization(
                credential,
                scopes=google_scopes_for_access(google_pairs),
            )
            workspace.save_connector_provider_secret(
                "google", "authorized_user_json", refreshed
            )
        except Exception as exc:
            checks.append(
                _check("fail", "live connector Google", f"{type(exc).__name__}: {exc}")
            )
        else:
            checks.append(_check("ok", "live connector Google", "authorization refreshed"))


def _nested_assignment_rows(
    assignments: dict[str, object],
) -> list[tuple[object, object, object]]:
    """Render participant routes with exact-action overrides directly below."""

    rows: list[tuple[object, object, object]] = []
    default = assignments.get("default")
    if default:
        rows.append(("default", default, "default"))
    raw_lifelines = assignments.get("lifelines")
    raw_actions = assignments.get("actions")
    lifelines = (
        {
            str(target): str(configuration)
            for target, configuration in raw_lifelines.items()
        }
        if isinstance(raw_lifelines, Mapping)
        else {}
    )
    actions = (
        {
            str(target): str(configuration)
            for target, configuration in raw_actions.items()
        }
        if isinstance(raw_actions, Mapping)
        else {}
    )
    participants = sorted(
        {*lifelines, *(target.partition(".")[0] for target in actions)}
    )
    for participant in participants:
        rows.append(
            (
                participant,
                lifelines.get(participant) or "inherits default",
                "participant" if participant in lifelines else "inherited",
            )
        )
        for target, configuration in sorted(actions.items()):
            owner, separator, action = target.partition(".")
            if separator and owner == participant:
                rows.append((f"  {action}", configuration, "action override"))
    return rows


def _render_columns_or_empty(
    renderer: TerminalRenderer,
    title: str,
    headers: tuple[str, ...],
    rows: Sequence[tuple[object, ...]],
    *,
    empty: str,
) -> None:
    """Render a configuration subsection without an empty table shell."""

    if rows:
        renderer.columns(title, headers, list(rows))
    else:
        renderer.empty(title, empty)


def render_configuration(
    report: dict[str, object],
    renderer: TerminalRenderer,
    *,
    show_checks: bool = False,
) -> None:
    project = report["project"]
    assert isinstance(project, dict)
    renderer.framed_section("Project")
    renderer.table(
        "Details",
        [
            ("Project", project.get("name"), None),
            ("Root", project.get("root"), None),
            ("Workflow", project.get("workflow") or "not resolved", None),
            ("Specification", project.get("specification"), None),
            ("Manifest", project.get("manifest"), None),
        ],
    )
    renderer.framed_section("Models")
    models = report["models"]
    assert isinstance(models, dict)
    configurations = models.get("configurations") or []
    model_configuration_rows = [
        (
            item.get("name"),
            item.get("spec") or "-",
            item.get("idle_timeout") or "-",
            item.get("source"),
        )
        for item in configurations
        if isinstance(item, dict)
    ]
    _render_columns_or_empty(
        renderer,
        "Configurations",
        ("Name", "Spec", "Idle", "Source"),
        model_configuration_rows,
        empty="No configurations.",
    )
    assignments = models.get("assignments") or {}
    assert isinstance(assignments, dict)
    _render_columns_or_empty(
        renderer,
        "Assignments",
        ("Target", "Configuration", "Scope"),
        _nested_assignment_rows(assignments),
        empty="No assignments.",
    )
    renderer.framed_section("Assistants")
    _render_assistant_tables(report, renderer, compact_titles=True)
    renderer.framed_section("Connectors")
    connectors = report["connectors"]
    assert isinstance(connectors, dict)
    connector_configuration_rows = [
        (
            item.get("name"),
            item.get("kind") or item.get("provider"),
            item.get("chat_id")
            or item.get("spreadsheet_id")
            or item.get("query")
            or "-",
        )
        for item in connectors.get("configurations") or []
        if isinstance(item, dict)
    ]
    _render_columns_or_empty(
        renderer,
        "Configurations",
        ("Name", "Kind", "Resource"),
        connector_configuration_rows,
        empty="No configurations.",
    )
    connector_rows: list[tuple[object, object, object]] = []
    for requirement, configuration in dict(connectors.get("bindings") or {}).items():
        connector_rows.append((requirement, configuration, "requirement"))
    connector_assignments = connectors.get("assignments") or {}
    if isinstance(connector_assignments, dict):
        connector_rows.extend(_nested_assignment_rows(connector_assignments))
    _render_columns_or_empty(
        renderer,
        "Assignments and bindings",
        ("Target", "Configuration", "Purpose"),
        connector_rows,
        empty="No assignments or bindings.",
    )
    renderer.framed_section("Site")
    renderer.table(
        "Private state",
        [
            (
                "Location",
                report.get("site_root") or "not available",
                None,
            )
        ],
    )
    if not show_checks:
        return
    raw_checks = report.get("checks") or []
    checks = raw_checks if isinstance(raw_checks, list) else []
    renderer.framed_section("Readiness")
    _render_columns_or_empty(
        renderer,
        "Checks",
        ("Status", "Check", "Detail"),
        [
            (
                renderer.status_mark(
                    "success" if item.get("status") == "ok" else (
                        "warning" if item.get("status") == "warn" else "error"
                    )
                ),
                item.get("name"),
                item.get("detail"),
            )
            for item in checks
            if isinstance(item, dict)
        ],
        empty="No checks.",
    )


def render_model_configuration(
    report: dict[str, object],
    renderer: TerminalRenderer,
    *,
    show_checks: bool = True,
) -> None:
    """Render only the project's model configurations and effective routing."""

    models = report["models"]
    assert isinstance(models, dict)
    renderer.framed_section("Models")
    configuration_rows = [
        (
            item.get("name"),
            item.get("spec") or "-",
            item.get("idle_timeout") or "-",
            item.get("source"),
        )
        for item in models.get("configurations") or []
        if isinstance(item, dict)
    ]
    _render_columns_or_empty(
        renderer,
        "Configurations",
        ("Name", "Spec", "Idle", "Source"),
        configuration_rows,
        empty="No configurations.",
    )
    assignments = models.get("assignments") or {}
    assert isinstance(assignments, dict)
    _render_columns_or_empty(
        renderer,
        "Assignments",
        ("Target", "Configuration", "Scope"),
        _nested_assignment_rows(assignments),
        empty="No assignments.",
    )
    if show_checks:
        _render_selected_checks(report, renderer, "model")


def _render_assistant_tables(
    report: dict[str, object],
    renderer: TerminalRenderer,
    *,
    compact_titles: bool = False,
) -> None:
    assistants = report["assistants"]
    assert isinstance(assistants, dict)
    configuration_rows = [
        (item.get("name"), item.get("backend"))
        for item in assistants.get("configurations") or []
        if isinstance(item, dict)
    ]
    _render_columns_or_empty(
        renderer,
        "Configurations" if compact_titles else "Assistant configurations",
        ("Name", "Backend"),
        configuration_rows,
        empty="No configurations.",
    )
    assignments = assistants.get("assignments") or {}
    assert isinstance(assignments, dict)
    _render_columns_or_empty(
        renderer,
        "Assignments" if compact_titles else "Assistant assignments",
        ("Target", "Configuration", "Scope"),
        _nested_assignment_rows(assignments),
        empty="No assignments.",
    )
    resolved_rows = [
        (
            item.get("target"),
            item.get("backend") or "missing",
            item.get("source"),
            item.get("access"),
            item.get("external_tools"),
            item.get("shell"),
        )
        for item in assistants.get("resolved") or []
        if isinstance(item, dict)
    ]
    _render_columns_or_empty(
        renderer,
        "Effective routing" if compact_titles else "Effective assistant routing",
        ("Target", "Backend", "Source", "Access", "Tools", "Shell"),
        resolved_rows,
        empty="No assistant actions.",
    )


def render_assistant_configuration(
    report: dict[str, object],
    renderer: TerminalRenderer,
    *,
    show_checks: bool = True,
) -> None:
    """Render coding-assistant configurations and effective routing."""

    renderer.framed_section("Assistants")
    _render_assistant_tables(report, renderer, compact_titles=True)
    if show_checks:
        _render_selected_checks(report, renderer, "assistant")


def render_connector_configuration(
    report: dict[str, object],
    renderer: TerminalRenderer,
    *,
    show_checks: bool = True,
) -> None:
    """Render only the project's connector configurations and routing."""

    connectors = report["connectors"]
    assert isinstance(connectors, dict)
    renderer.framed_section("Connectors")
    configuration_rows = [
        (
            item.get("name"),
            item.get("kind") or item.get("provider"),
            item.get("chat_id")
            or item.get("spreadsheet_id")
            or item.get("query")
            or "-",
        )
        for item in connectors.get("configurations") or []
        if isinstance(item, dict)
    ]
    _render_columns_or_empty(
        renderer,
        "Configurations",
        ("Name", "Kind", "Resource"),
        configuration_rows,
        empty="No configurations.",
    )
    rows: list[tuple[object, object, object]] = []
    for requirement, configuration in dict(connectors.get("bindings") or {}).items():
        rows.append((requirement, configuration, "requirement"))
    assignments = connectors.get("assignments") or {}
    if isinstance(assignments, dict):
        rows.extend(_nested_assignment_rows(assignments))
    _render_columns_or_empty(
        renderer,
        "Assignments and bindings",
        ("Target", "Configuration", "Purpose"),
        rows,
        empty="No assignments or bindings.",
    )
    if show_checks:
        _render_selected_checks(report, renderer, "connector")


def _selected_checks(report: dict[str, object], scope: str) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    raw_checks = report.get("checks") or []
    for item in raw_checks if isinstance(raw_checks, list) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").casefold()
        if name == "workflow" or scope in name:
            selected.append(item)
    return selected


def configuration_scope_valid(report: dict[str, object], scope: str) -> bool:
    """Return whether one configuration domain has no failing check."""

    return not any(
        item.get("status") == "fail" for item in _selected_checks(report, scope)
    )


def _render_selected_checks(
    report: dict[str, object],
    renderer: TerminalRenderer,
    scope: str,
) -> None:
    checks = _selected_checks(report, scope)
    _render_columns_or_empty(
        renderer,
        "Checks",
        ("Status", "Check", "Detail"),
        [
            (
                renderer.status_mark(
                    "success" if item.get("status") == "ok" else (
                        "warning" if item.get("status") == "warn" else "error"
                    )
                ),
                item.get("name"),
                item.get("detail"),
            )
            for item in checks
        ],
        empty="No checks.",
    )


def configure_model(
    workspace: Workspace,
    name: str,
    spec: str,
    *,
    idle_timeout: float | None = None,
    base_url: str | None = None,
) -> dict[str, str]:
    provider, separator, model = spec.strip().partition(":")
    canonical_provider = {
        "claude": "anthropic",
        "ollama": "local",
    }.get(provider.casefold(), provider.casefold())
    supported = {"mock", "scripted", "openai", "mistral", "anthropic", "local"}
    if canonical_provider not in supported:
        raise WorkspaceError(
            f"Unsupported model provider {provider!r}. Supported: "
            + ", ".join(sorted(supported))
        )
    if canonical_provider == "mock":
        raise WorkspaceError("Use the built-in configuration named 'mock'.")
    canonical_spec = f"{canonical_provider}:{model}" if separator else canonical_provider
    values = {
        "provider": canonical_provider,
        "model": model,
        "spec": canonical_spec,
    }
    if idle_timeout is not None:
        if not math.isfinite(idle_timeout) or idle_timeout < 0:
            raise WorkspaceError("Idle timeout must be a non-negative finite number.")
        values["idle_timeout"] = str(idle_timeout)
    if base_url is not None:
        if canonical_provider != "local":
            raise WorkspaceError("--base-url is currently supported for local models only.")
        workspace.save_provider_profile("local", {"base_url": base_url})
    return workspace.save_model_configuration(name, values)


def assign_model(
    workspace: Workspace,
    target: str,
    configuration: str | None,
) -> dict[str, object]:
    workflow = workspace.resolve_workflow()
    loaded, module = _load_project_workflow(workspace, workflow)
    known = {"default", *_model_targets(loaded, module)}
    if target not in known:
        raise WorkspaceError(
            f"Unknown model assignment target {target!r}. Available: "
            + ", ".join(sorted(known))
        )
    if configuration is not None and configuration not in workspace.model_configurations():
        raise WorkspaceError(
            f"Model configuration does not exist: {configuration}."
        )
    profile = workspace.model_assignment_profile(workflow, include_site=False)
    default = str(profile.get("default") or "mock")
    raw_lifelines = profile.get("lifelines") or {}
    raw_actions = profile.get("actions") or {}
    if not isinstance(raw_lifelines, dict) or not isinstance(raw_actions, dict):
        raise WorkspaceError("Project model assignments are malformed.")
    lifelines = {
        str(target): str(name) for target, name in raw_lifelines.items()
    }
    actions = {
        str(target): str(name) for target, name in raw_actions.items()
    }
    if target == "default":
        default = configuration or "mock"
    else:
        selected = actions if "." in target else lifelines
        if configuration is None:
            selected.pop(target, None)
        else:
            selected[target] = configuration
    saved = workspace.save_model_assignment_profile(
        workflow,
        default=default,
        lifelines=lifelines,
        actions=actions,
    )
    _clear_site_assignment(
        workspace,
        state_key="model_site_profiles",
        workflow=workflow,
        target=target,
    )
    return saved


def configure_assistant(
    workspace: Workspace,
    name: str,
    backend: str,
) -> dict[str, str]:
    """Save one named Codex or Claude configuration."""

    return workspace.save_assistant_configuration(name, backend)


def assign_assistant(
    workspace: Workspace,
    target: str,
    configuration: str | None,
) -> dict[str, object]:
    """Assign a named coding-assistant configuration to one target."""

    workflow = workspace.resolve_workflow()
    loaded, module = _load_project_workflow(workspace, workflow)
    known = set(assistant_targets(loaded, module))
    if target not in known:
        raise WorkspaceError(
            f"Unknown assistant assignment target {target!r}. Available: "
            + (", ".join(sorted(known)) or "none")
        )
    if (
        configuration is not None
        and configuration not in workspace.assistant_configurations()
    ):
        raise WorkspaceError(
            f"Assistant configuration does not exist: {configuration}."
        )
    profile = workspace.assistant_assignment_profile(workflow)
    default = str(profile.get("default") or "")
    raw_lifelines = profile.get("lifelines")
    lifelines = (
        {str(key): str(value) for key, value in raw_lifelines.items()}
        if isinstance(raw_lifelines, dict)
        else {}
    )
    raw_actions = profile.get("actions")
    actions = (
        {str(key): str(value) for key, value in raw_actions.items()}
        if isinstance(raw_actions, dict)
        else {}
    )
    if target == "default":
        default = configuration or ""
    else:
        selected = actions if "." in target else lifelines
        if configuration is None:
            selected.pop(target, None)
        else:
            selected[target] = configuration
    return workspace.save_assistant_assignment_profile(
        workflow,
        default=default,
        lifelines=lifelines,
        actions=actions,
    )


def assign_connector(
    workspace: Workspace,
    target: str,
    configuration: str | None,
) -> dict[str, dict[str, str]]:
    workflow = workspace.resolve_workflow()
    loaded, module = _load_project_workflow(workspace, workflow)
    sites = human_action_sites(loaded, module)
    known = {
        *sites,
        *(
            f"{participant}.{action}"
            for participant, actions in sites.items()
            for action in actions
        ),
    }
    if target not in known:
        raise WorkspaceError(
            f"Unknown human-action target {target!r}. Available: "
            + (", ".join(sorted(known)) or "none")
        )
    configurations = workspace.connector_configurations()
    if configuration is not None:
        selected_configuration = configurations.get(configuration)
        if selected_configuration is None:
            raise WorkspaceError(
                f"Connector configuration does not exist: {configuration}."
            )
        if selected_configuration.get("kind") != "telegram":
            raise WorkspaceError(
                f"Human actions need a Telegram configuration, but "
                f"{configuration!r} is "
                f"{selected_configuration.get('kind') or 'untyped'}."
            )
    manifest_connectors = workspace.project_manifest().get("connectors") or {}
    assert isinstance(manifest_connectors, dict)
    profile = manifest_connectors.get("assignments") or {}
    assert isinstance(profile, dict)
    lifelines = dict(profile.get("lifelines") or {})
    actions = dict(profile.get("actions") or {})
    selected = actions if "." in target else lifelines
    if configuration is None:
        selected.pop(target, None)
    else:
        selected[target] = configuration
    saved = workspace.save_connector_assignment_profile(
        workflow,
        lifelines=lifelines,
        actions=actions,
    )
    _clear_site_assignment(
        workspace,
        state_key="connector_site_assignments",
        workflow=workflow,
        target=target,
    )
    return saved


def bind_connector(
    workspace: Workspace,
    requirement: str,
    configuration: str,
) -> dict[str, str]:
    """Bind one declared service requirement to a named configuration."""

    workflow = workspace.resolve_workflow()
    _loaded, module = _load_project_workflow(workspace, workflow)
    requirements = {
        item.name: item for item in connector_requirements_from_module(module)
    }
    selected_requirement = requirements.get(requirement)
    if selected_requirement is None:
        raise WorkspaceError(
            f"Unknown connector requirement {requirement!r}. Available: "
            + (", ".join(sorted(requirements)) or "none")
        )
    configurations = workspace.connector_configurations()
    selected_configuration = configurations.get(configuration)
    if selected_configuration is None:
        raise WorkspaceError(
            f"Connector configuration does not exist: {configuration}."
        )
    configured_kind = str(selected_configuration.get("kind") or "")
    if configured_kind != selected_requirement.kind:
        raise WorkspaceError(
            f"Connector requirement {requirement!r} needs "
            f"{selected_requirement.kind}, but configuration "
            f"{configuration!r} is {configured_kind or 'untyped'}."
        )
    return workspace.bind_connector(workflow, requirement, configuration)


def _clear_site_assignment(
    workspace: Workspace,
    *,
    state_key: str,
    workflow: str,
    target: str,
) -> None:
    """Make an explicit project assignment replace an older site override."""

    state = workspace.load()
    raw_profiles = state.get(state_key) or {}
    if not isinstance(raw_profiles, dict):
        return
    canonical = workspace.canonical_spec(workflow, cwd=workspace.root)
    profiles = dict(raw_profiles)
    raw_profile = profiles.get(canonical)
    if not isinstance(raw_profile, dict):
        return
    profile = dict(raw_profile)
    if target == "default":
        profile.pop("default", None)
    else:
        group = "actions" if "." in target else "lifelines"
        values = dict(profile.get(group) or {})
        values.pop(target, None)
        profile[group] = values
    if profile.get("default") or profile.get("lifelines") or profile.get("actions"):
        profiles[canonical] = profile
    else:
        profiles.pop(canonical, None)
    workspace.update(**{state_key: profiles})


def unbind_connector(workspace: Workspace, requirement: str) -> dict[str, str]:
    workflow = workspace.resolve_workflow()
    return workspace.unbind_connector(workflow, requirement)
