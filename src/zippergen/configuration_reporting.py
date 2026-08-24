"""Build the secret-free configuration and readiness report."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from zippergen.assistant_configuration import (
    project_assistant_routing,
    resolved_assistant_actions,
)
from zippergen.backends import model_accepts_temperature
from zippergen.configuration_checks import (
    Check,
    _check,
    _site_checks,
    _static_connector_checks,
)
from zippergen.configuration_inventory import (
    _ASSISTANT_SOURCE_WORDS,
    _connector_slots,
    _load_project_workflow,
    _project_model_source,
)
from zippergen.connectors import connector_requirements_from_module
from zippergen.models import project_model_routing
from zippergen.provider_connections import (
    provider_credential_field,
    provider_standard_environment,
    split_model_spec,
)
from zippergen.semantic import workflow_semantics
from zippergen.syntax import Workflow
from zippergen.workspace import Workspace, WorkspaceError


def _effective_routing(
    workspace: Workspace,
    *,
    workflow: Workflow | None,
    module: ModuleType | None,
    model_profile: dict[str, Any],
    resolved_models: dict[str, object],
    resolved_assistants: list[dict[str, object]],
    connector_configurations: dict[str, dict[str, str]],
    connector_assignments: dict[str, Any],
    connector_bindings: dict[str, str],
    site_facts: list[dict[str, object]],
    checks: list[Check],
    live: bool,
) -> list[dict[str, object]]:
    """Resolve every action and connector slot to its effective destination."""

    effective_routing: list[dict[str, object]] = []
    if workflow is not None and module is not None:
        def check_passes(name: str) -> bool:
            matching = [item for item in checks if item.get("name") == name]
            return not matching or all(item.get("status") == "ok" for item in matching)

        def connection_available(connection: str, provider: str) -> bool:
            """Return whether a named route has every required site credential."""

            field = provider_credential_field(provider)
            if field is None:
                return True
            return any(
                item.get("kind") == "provider credential"
                and item.get("name") == connection
                and item.get("available")
                for item in site_facts
            )

        semantics = workflow_semantics(workflow, module=module)
        sites = semantics.get("action_sites") or []
        definitions = semantics.get("action_definitions") or {}
        action_definitions = definitions if isinstance(definitions, Mapping) else {}
        seen_sites: set[tuple[str, str, str]] = set()
        raw_model_overrides = resolved_models.get("overrides") or {}
        model_overrides = (
            {str(key): value for key, value in raw_model_overrides.items()}
            if isinstance(raw_model_overrides, Mapping)
            else {}
        )
        raw_model_settings = resolved_models.get("settings") or {}
        model_settings = (
            {
                str(key): dict(value)
                for key, value in raw_model_settings.items()
                if isinstance(value, Mapping)
            }
            if isinstance(raw_model_settings, Mapping)
            else {}
        )
        raw_model_actions = model_profile.get("actions") or {}
        model_actions = (
            {str(key): value for key, value in raw_model_actions.items()}
            if isinstance(raw_model_actions, Mapping)
            else {}
        )
        raw_model_lifelines = model_profile.get("lifelines") or {}
        model_lifelines = (
            {str(key): value for key, value in raw_model_lifelines.items()}
            if isinstance(raw_model_lifelines, Mapping)
            else {}
        )
        assistant_by_target = {
            str(item.get("target")): item for item in resolved_assistants
        }
        connector_actions = dict(connector_assignments.get("actions") or {})
        connector_lifelines = dict(connector_assignments.get("lifelines") or {})
        for raw in sites if isinstance(sites, list) else []:
            if not isinstance(raw, dict):
                continue
            participant = str(raw.get("lifeline") or "")
            action = str(raw.get("action") or "")
            kind = str(raw.get("kind") or "")
            if kind not in {"llm", "assistant", "human"}:
                continue
            key = (participant, action, kind)
            if key in seen_sites:
                continue
            seen_sites.add(key)
            target = f"{participant}.{action}"
            if kind == "llm":
                selected = str(
                    model_overrides.get(target)
                    or model_overrides.get(participant)
                    or resolved_models.get("default")
                    or "mock"
                )
                configuration = str(
                    model_actions.get(target)
                    or model_lifelines.get(participant)
                    or model_profile.get("default")
                    or "workflow default"
                )
                definition = action_definitions.get(action) or {}
                action_temperature = (
                    definition.get("temperature")
                    if isinstance(definition, Mapping)
                    else None
                )
                configured_settings = model_settings.get(
                    target, model_settings.get(participant, {})
                )
                raw_configured_temperature = configured_settings.get("temperature")
                configured_temperature = (
                    float(raw_configured_temperature)
                    if raw_configured_temperature is not None
                    else None
                )
                accepts_temperature = model_accepts_temperature(selected)
                effective_temperature = (
                    float(action_temperature)
                    if action_temperature is not None
                    else configured_temperature
                    if configured_temperature is not None
                    else 0.2
                    if accepts_temperature
                    else None
                )
                # Which level won is the answer to "where do I change this?",
                # and every value below names the target you would type after
                # `assign`.
                source = (
                    "action"
                    if target in model_actions
                    else "participant"
                    if participant in model_lifelines
                    else "default"
                )
                try:
                    selected_kind, connection, _model = split_model_spec(selected)
                except ValueError:
                    selected_kind = ""
                    connection = None
                if connection is not None:
                    available = connection_available(connection, selected_kind)
                else:
                    direct_environment = provider_standard_environment(selected_kind)
                    available = direct_environment is None or any(
                        item.get("kind") == "provider credential"
                        and item.get("name") == direct_environment
                        and item.get("available")
                        for item in site_facts
                    )
                if selected_kind == "scripted":
                    try:
                        _kind, _connection, model = split_model_spec(selected)
                    except ValueError:
                        model = None
                    path = Path(model or "").expanduser()
                    if not path.is_absolute():
                        path = workspace.root / path
                    available = available and check_passes(
                        f"scripted model {path}"
                    )
                if live:
                    available = available and check_passes(f"live model {selected}")
                if action_temperature is not None:
                    if not accepts_temperature:
                        available = False
                        checks.append(
                            _check(
                                "fail",
                                f"model temperature {target}",
                                f"{selected} does not support an explicit temperature",
                                scopes=("model",),
                            )
                        )
                effective_settings = dict(configured_settings)
                if effective_temperature is not None:
                    effective_settings["temperature"] = effective_temperature
                else:
                    effective_settings.pop("temperature", None)
                effective_routing.append(
                    {
                        "participant": participant,
                        "action": action,
                        "kind": "model",
                        "configuration": configuration,
                        "effective": selected,
                        "settings": effective_settings,
                        "temperature": effective_temperature,
                        "temperature_source": (
                            "action"
                            if action_temperature is not None
                            else "model configuration"
                            if configured_temperature is not None
                            else "ZipperGen default"
                            if accepts_temperature
                            else "provider default"
                        ),
                        "available": available,
                        "source": source,
                        "verified": bool(live),
                    }
                )
            elif kind == "assistant":
                item = assistant_by_target.get(target) or {}
                backend = item.get("backend")
                available = bool(backend) and any(
                    fact.get("kind") == "assistant CLI"
                    and fact.get("name") == backend
                    and fact.get("available")
                    for fact in site_facts
                )
                effective_routing.append(
                    {
                        "participant": participant,
                        "action": action,
                        "kind": "assistant",
                        "configuration": item.get("configuration") or "missing",
                        "effective": backend or "missing",
                        "available": available,
                        "source": _ASSISTANT_SOURCE_WORDS.get(
                            str(item.get("source") or ""), "default"
                        ),
                        "verified": True,
                    }
                )
            else:
                configuration = str(
                    connector_actions.get(target)
                    or connector_lifelines.get(participant)
                    or connector_assignments.get("default")
                    or "terminal"
                )
                source = (
                    "action"
                    if target in connector_actions
                    else "participant"
                    if participant in connector_lifelines
                    else "default"
                    if connector_assignments.get("default")
                    else "terminal"
                )
                available = True
                if configuration != "terminal":
                    provider = str(
                        connector_configurations.get(configuration, {}).get("provider")
                        or ""
                    )
                    connection = str(
                        connector_configurations.get(configuration, {}).get(
                            "connection"
                        )
                        or ""
                    )
                    if connection:
                        available = connection_available(connection, provider)
                    if live:
                        live_name = (
                            f"live provider {connection}"
                            if provider == "google"
                            else f"live connector {configuration}"
                        )
                        available = available and check_passes(live_name)
                effective_routing.append(
                    {
                        "participant": participant,
                        "action": action,
                        "kind": "human",
                        "configuration": configuration,
                        "effective": configuration,
                        "available": available,
                        "source": source,
                        "verified": bool(live),
                    }
                )
        for requirement in connector_requirements_from_module(module):
            configuration = connector_bindings.get(requirement.name)
            available = bool(configuration)
            if configuration:
                provider = str(
                    connector_configurations.get(configuration, {}).get("provider")
                    or ""
                )
                connection = str(
                    connector_configurations.get(configuration, {}).get(
                        "connection"
                    )
                    or ""
                )
                if connection:
                    available = connection_available(connection, provider)
                if live:
                    live_name = (
                        f"live provider {connection}"
                        if provider == "google"
                        else f"live connector {configuration}"
                    )
                    available = available and check_passes(live_name)
            effective_routing.append(
                {
                    "participant": requirement.participant,
                    "action": requirement.name,
                    "kind": requirement.kind,
                    "configuration": configuration or "missing",
                    "effective": configuration or "missing",
                    "available": available,
                    "source": "requirement",
                    "verified": bool(live),
                }
            )
    return effective_routing


def _configuration_rows(
    workspace: Workspace,
    module: ModuleType | None,
) -> list[dict[str, object]]:
    """The project's answers to its workflow's declared questions.

    These live in `zippergen.toml` beside every other project choice, so the
    command that shows a project's configuration must show them. Leaving them
    out is what made a value seem to belong to the deployment rather than to
    the project that authored it.
    """

    from zippergen.deployment import deployment_spec_from_module

    stored = workspace.configuration_values()
    if module is None:
        return [
            {"name": name, "value": value, "declared": False}
            for name, value in sorted(stored.items())
        ]
    try:
        spec = deployment_spec_from_module(module)
    except Exception:
        return []
    rows: list[dict[str, object]] = []
    for field in spec.fields:
        rows.append({
            "name": field.name,
            "prompt": field.prompt,
            "value": stored.get(field.name),
            "answered": field.name in stored,
            "secret": field.secret,
            "default": field.default,
            "choices": list(field.choices),
        })
    return rows


def configuration_report(
    workspace: Workspace,
    *,
    live: bool = False,
    include_site_checks: bool = True,
    provider_names: tuple[str, ...] = (),
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

    provider_connections = workspace.provider_connections()
    provider_rows = [
        {
            "name": name,
            "kind": values.get("kind", ""),
            "base_url": values.get("base_url"),
        }
        for name, values in sorted(provider_connections.items())
    ]
    model_configurations = workspace.model_configurations()
    model_rows = [
        {
            "name": name,
            "connection": values.get("connection", ""),
            "model": values.get("model", ""),
            "spec": values.get("spec", ""),
            "idle_timeout": values.get("idle_timeout"),
            "temperature": values.get("temperature"),
            "max_tokens": values.get("max_tokens"),
            "timeout": values.get("timeout"),
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
        "settings": {},
    }
    checks: list[Check] = []
    for item in model_rows:
        if item.get("temperature") is not None and not model_accepts_temperature(
            str(item.get("spec") or "")
        ):
            checks.append(
                _check(
                    "fail",
                    f"model temperature {item['name']}",
                    f"{item['spec']} does not support an explicit temperature",
                    scopes=("model",),
                )
            )
    if workflow_error:
        checks.append(
            _check("fail", "workflow", workflow_error, scopes=("all",))
        )
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
            checks.append(
                _check(
                    "fail",
                    "model assignments",
                    str(exc),
                    scopes=("model",),
                )
            )
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
                "settings": {
                    target: chosen.as_dict()
                    for target, chosen in routing.settings.items()
                },
            }
            checks.append(
                _check(
                    "ok",
                    "model assignments",
                    f"{len(routing.overrides)} explicit assignment(s)",
                    scopes=("model",),
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
            checks.append(
                _check(
                    "fail",
                    "assistant assignments",
                    str(exc),
                    scopes=("assistant",),
                )
            )
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
                        scopes=("assistant",),
                    )
                )
            else:
                checks.append(
                    _check(
                        "ok",
                        "assistant assignments",
                        f"{len(effective)} assistant action(s) resolved",
                        scopes=("assistant",),
                    )
                )

    connector_configurations = workspace.connector_configurations()
    connector_assignments = (
        workspace.connector_assignment_profile(workflow_spec)
        if workflow_spec is not None
        else {"default": "", "lifelines": {}, "actions": {}}
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

    site_facts = _site_checks(
        workspace,
        include_site_checks=include_site_checks,
        live=live,
        provider_names=provider_names,
        model_names=model_names,
        assistant_names=assistant_names,
        connector_names=connector_names,
        resolved_models=resolved_models,
        model_configurations=model_configurations,
        assistant_configurations=assistant_configurations,
        resolved_assistants=resolved_assistants,
        connector_configurations=connector_configurations,
        connector_assignments=connector_assignments,
        connector_bindings=connector_bindings,
        provider_connections=provider_connections,
        workflow=workflow,
        module=module,
        checks=checks,
    )
    effective_routing = _effective_routing(
        workspace,
        workflow=workflow,
        module=module,
        model_profile=model_profile,
        resolved_models=resolved_models,
        resolved_assistants=resolved_assistants,
        connector_configurations=connector_configurations,
        connector_assignments=connector_assignments,
        connector_bindings=connector_bindings,
        site_facts=site_facts,
        checks=checks,
        live=live,
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
        "configuration": _configuration_rows(workspace, module),
        "providers": {"connections": provider_rows},
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
            # Every slot the workflow offers, filled or not. The two kinds of
            # slot are keyed differently, and nothing else on screen says so.
            "slots": _connector_slots(
                workflow, module, connector_assignments, connector_bindings
            ),
        },
        "site_facts": site_facts,
        "effective_routing": effective_routing,
        "checks": checks,
        "valid": not any(item["status"] == "fail" for item in checks),
    }


def _selected_checks(report: dict[str, object], scope: str) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    raw_checks = report.get("checks") or []
    for item in raw_checks if isinstance(raw_checks, list) else []:
        if not isinstance(item, dict):
            continue
        raw_scopes = item.get("scopes") or ()
        values = raw_scopes if isinstance(raw_scopes, (list, tuple, set)) else ()
        scopes = {str(value) for value in values}
        if "all" in scopes or scope in scopes:
            selected.append(item)
    return selected


def configuration_scope_valid(report: dict[str, object], scope: str) -> bool:
    """Return whether one configuration domain has no failing check."""

    return not any(
        item.get("status") == "fail" for item in _selected_checks(report, scope)
    )
