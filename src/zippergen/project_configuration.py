"""Inspect and manage one project's portable and site configuration."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
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
from zippergen.google_auth import google_support_installed
from zippergen.connectors import connector_requirements_from_module
from zippergen.models import project_model_routing, selected_llm_specs
from zippergen.provider_connections import (
    provider_credential_field,
    provider_credential_label,
    provider_standard_environment,
    split_model_spec,
)
from zippergen.rendering import TerminalRenderer
from zippergen.semantic import workflow_semantics
from zippergen.syntax import LLMAction, Workflow
from zippergen.workspace import Workspace, WorkspaceError


Check = dict[str, object]


def _check(
    status: str,
    name: str,
    detail: str,
    *,
    scopes: Sequence[str] = (),
) -> Check:
    """Describe one check and the configuration domains that depend on it."""

    return {
        "status": status,
        "name": name,
        "detail": detail,
        "scopes": tuple(scopes),
    }


def _provider(spec: str) -> str:
    from zippergen.provider_connections import split_model_spec

    try:
        provider, _connection, _model = split_model_spec(spec)
    except ValueError:
        return spec.partition(":")[0].partition("@")[0].strip().casefold()
    return provider


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
    site_keys = set(merged) - project_keys - {"provider", "spec"}
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


def _connector_slots(
    workflow: Workflow | None,
    module: ModuleType | None,
    assignments: dict[str, object],
    bindings: dict[str, str],
) -> list[dict[str, str]]:
    """List every connector slot with the exact name you would type for it.

    A workflow offers two kinds, and they are keyed differently: a declared
    requirement by its own name, a human action by the participant who runs
    it. That is learnable, but only if something shows it, so this is what
    ``zg connector`` prints instead of leaving people to infer it from an
    error message.
    """

    if workflow is None or module is None:
        return []
    slots: list[dict[str, str]] = []
    for requirement in connector_requirements_from_module(module):
        slots.append({
            "target": requirement.name,
            "meaning": f"{requirement.kind} for {requirement.participant}",
            "configuration": bindings.get(requirement.name) or "not assigned",
        })
    sites = human_action_sites(workflow, module)
    default = str(assignments.get("default") or "")
    raw_lifelines = assignments.get("lifelines")
    lifelines = dict(raw_lifelines) if isinstance(raw_lifelines, Mapping) else {}
    raw_actions = assignments.get("actions")
    actions = dict(raw_actions) if isinstance(raw_actions, Mapping) else {}
    if sites:
        slots.append({
            "target": "default",
            "meaning": "every human action not named below",
            "configuration": default or "not assigned",
        })
    for participant, participant_actions in sites.items():
        slots.append({
            "target": participant,
            "meaning": "this participant's human actions",
            "configuration": str(
                lifelines.get(participant) or default or "not assigned"
            ),
        })
        for action in participant_actions:
            target = f"{participant}.{action}"
            slots.append({
                "target": target,
                "meaning": "this one human action",
                "configuration": str(
                    actions.get(target)
                    or lifelines.get(participant)
                    or default
                    or "not assigned"
                ),
            })
    return slots


#: The four families name resolution levels differently in their own code.
#: One column can only mean one thing, so assistants are translated here into
#: the words every other family uses: the key you would type after `assign`.
_ASSISTANT_SOURCE_WORDS = {
    "action assignment": "action",
    "participant assignment": "participant",
    "default assignment": "default",
    "runtime default": "default",
    "missing": "unassigned",
}


def _routing_status(renderer: TerminalRenderer, item: Mapping[str, object]) -> str:
    """Three states, because "configured" and "reached" are different news.

    A command that never contacts a provider cannot honestly claim a route
    works; it can only say nothing contradicts it. `zg config` is offline and
    `zg check` is live, so the depth of the answer belongs on the row rather
    than in the reader's memory of which command they typed.
    """

    if not item.get("available"):
        return renderer.status_mark("error")
    return renderer.status_mark("success" if item.get("verified") else "info")


def _render_effective_routing(
    renderer: TerminalRenderer,
    report: Mapping[str, object],
    kinds: tuple[str, ...],
    *,
    subject: str,
    resolved_header: str,
    resolved: Callable[[Mapping[str, object]], object],
    empty: str,
) -> None:
    """Answer "what will this use, and where do I change it?" for one family.

    The participant is printed once per group rather than on every row, so the
    shape of the answer is visible before any of it is read.
    """

    routes = report.get("effective_routing") or []
    rows: list[tuple[object, ...]] = []
    previous = None
    for item in routes if isinstance(routes, list) else []:
        if not isinstance(item, dict) or item.get("kind") not in kinds:
            continue
        participant = item.get("participant")
        rows.append((
            _routing_status(renderer, item),
            "" if participant == previous else participant,
            item.get("action"),
            item.get("configuration"),
            resolved(item),
            item.get("source"),
        ))
        previous = participant
    _render_columns_or_empty(
        renderer,
        "Effective routing",
        ("", "Participant", subject, "Configuration", resolved_header, "From"),
        rows,
        empty=empty,
    )


def _default_connector_check(
    default: str,
    configurations: dict[str, dict[str, str]],
    human_sites: dict[str, list[str]],
) -> Check:
    """Check the one configuration that catches everything not named."""

    selected = configurations.get(default)
    if selected is None:
        return _check(
            "fail",
            "connector assignment default",
            f"configuration {default!r} does not exist",
            scopes=("connector",),
        )
    if selected.get("provider") != "telegram":
        return _check(
            "fail",
            "connector assignment default",
            f"{default!r} cannot deliver a human action",
            scopes=("connector",),
        )
    if not human_sites:
        return _check(
            "warn",
            "connector assignment default",
            "no participant asks a human, so nothing uses it",
            scopes=("connector",),
        )
    return _check(
        "ok",
        "connector assignment default",
        f"{default} for {len(human_sites)} participant(s)",
        scopes=("connector",),
    )


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
    default = str(assignments.get("default") or "")
    if default:
        checks.append(_default_connector_check(default, configurations, human_sites))
    for group in ("lifelines", "actions"):
        for target, configuration in assignments[group].items():
            if target not in human_targets:
                checks.append(
                    _check(
                        "fail",
                        f"connector assignment {target}",
                        "target has no human action",
                        scopes=("connector",),
                    )
                )
            elif configuration not in configurations:
                checks.append(
                    _check(
                        "fail",
                        f"connector assignment {target}",
                        f"configuration {configuration!r} does not exist",
                        scopes=("connector",),
                    )
                )
            elif configurations[configuration].get("provider") != "telegram":
                checks.append(
                    _check(
                        "fail",
                        f"connector assignment {target}",
                        f"{configuration!r} cannot deliver a human action",
                        scopes=("connector",),
                    )
                )
            else:
                checks.append(
                    _check(
                        "ok",
                        f"connector assignment {target}",
                        configuration,
                        scopes=("connector",),
                    )
                )

    for requirement in connector_requirements_from_module(module):
        configuration = bindings.get(requirement.name)
        if configuration is None:
            checks.append(
                _check(
                    "fail" if requirement.required else "warn",
                    f"connector requirement {requirement.name}",
                    "required, not assigned yet"
                    if requirement.required
                    else "optional, not assigned",
                    scopes=("connector",),
                )
            )
            continue
        selected = configurations.get(configuration)
        if selected is None:
            checks.append(
                _check(
                    "fail",
                    f"connector requirement {requirement.name}",
                    f"configuration {configuration!r} does not exist",
                    scopes=("connector",),
                )
            )
        elif selected.get("kind") != requirement.kind:
            checks.append(
                _check(
                    "fail",
                    f"connector requirement {requirement.name}",
                    f"requires {requirement.kind}; {configuration!r} is "
                    f"{selected.get('kind') or 'untyped'}",
                    scopes=("connector",),
                )
            )
        else:
            checks.append(
                _check(
                    "ok",
                    f"connector requirement {requirement.name}",
                    configuration,
                    scopes=("connector",),
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


def _live_model_check(
    spec: str,
    environment: dict[str, str],
    *,
    project_root: Path,
) -> None:
    from zippergen.backends import backend_from_spec, load_scripted_script

    provider = _provider(spec)
    if provider == "mock":
        return
    if provider == "scripted":
        path = spec.partition(":")[2]
        selected = Path(path).expanduser()
        if not selected.is_absolute():
            selected = project_root / selected
        load_scripted_script(selected)
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
                "idle_timeouts": routing.idle_timeouts,
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

    site_facts: list[dict[str, object]] = []
    google_support_reported = False
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
        model_connections: set[str] = set()
        for spec in specs:
            try:
                _kind, connection, _model = split_model_spec(spec)
            except ValueError:
                continue
            if connection:
                model_connections.add(connection)
        used_connectors = _used_connector_names(
            connector_assignments,
            connector_bindings,
        )
        used_connectors.update(connector_names)
        connector_connections = {
            str(connector_configurations.get(name, {}).get("connection") or "")
            for name in used_connectors
        }
        used_connections = {
            *provider_names,
            *model_connections,
            *connector_connections,
        }
        used_connections.discard("")
        for connection in sorted(used_connections):
            connection_scopes = ["provider"]
            if connection in model_connections:
                connection_scopes.append("model")
            if connection in connector_connections:
                connection_scopes.append("connector")
            profile = provider_connections.get(connection)
            if profile is None:
                checks.append(
                    _check(
                        "fail",
                        f"provider connection {connection}",
                        "configuration does not exist",
                        scopes=connection_scopes,
                    )
                )
                continue
            kind = str(profile.get("kind") or "")
            field = provider_credential_field(kind)
            if field is None:
                checks.append(
                    _check(
                        "ok",
                        f"provider connection {connection}",
                        "no credential required",
                        scopes=connection_scopes,
                    )
                )
                continue
            standard = provider_standard_environment(kind)
            available = bool(
                workspace.provider_secret(connection, field)
                or (os.environ.get(standard) if standard else None)
            )
            label = provider_credential_label(kind) or field
            site_facts.append(
                {
                    "kind": "provider credential",
                    "name": connection,
                    "detail": label,
                    "available": available,
                }
            )
            checks.append(
                _check(
                    "ok" if available else "fail",
                    f"provider credential {connection}",
                    f"{label} available"
                    if available
                    else f"{label} missing on this computer",
                    scopes=connection_scopes,
                )
            )
            if kind == "google" and not google_support_reported:
                # Google is the one provider needing a library that the core
                # install leaves out. Without this the gap only shows up at
                # the moment somebody authorizes, which is the worst time.
                google_support_reported = True
                supported = google_support_installed()
                site_facts.append(
                    {
                        "kind": "python package",
                        "name": "zippergen[google]",
                        "available": supported,
                    }
                )
                checks.append(
                    _check(
                        "ok" if supported else "fail",
                        "google support installed",
                        "google-auth is importable"
                        if supported
                        else "not installed here; add the 'google' extra to "
                        "the environment running zippergen",
                        scopes=connection_scopes,
                    )
                )
        direct_kinds: set[str] = set()
        for spec in specs:
            try:
                kind, connection, _model = split_model_spec(spec)
            except ValueError:
                continue
            if connection is None and provider_standard_environment(kind):
                direct_kinds.add(kind)
        for kind in sorted(direct_kinds):
            environment_name = provider_standard_environment(kind)
            assert environment_name is not None
            available = bool(os.environ.get(environment_name))
            label = provider_credential_label(kind) or environment_name
            site_facts.append(
                {
                    "kind": "provider credential",
                    "name": environment_name,
                    "detail": label,
                    "available": available,
                }
            )
            checks.append(
                _check(
                    "ok" if available else "fail",
                    f"provider credential {environment_name}",
                    f"{label} available"
                    if available
                    else f"{environment_name} missing from the environment",
                    scopes=("model",),
                )
            )
        for spec in specs:
            provider = _provider(spec)
            if provider == "scripted":
                try:
                    _kind, _connection, model = split_model_spec(spec)
                except ValueError:
                    model = None
                path = Path(model or "").expanduser()
                if not path.is_absolute():
                    path = workspace.root / path
                checks.append(
                    _check(
                        "ok" if path.is_file() else "fail",
                        f"scripted model {path}",
                        "response file exists" if path.is_file() else "file is missing",
                        scopes=("model",),
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
                        scopes=("assistant",),
                    )
                )

        if live:
            unique_specs = dict.fromkeys(specs)
            for spec in unique_specs:
                try:
                    _live_model_check(
                        spec,
                        environment,
                        project_root=workspace.root,
                    )
                except Exception as exc:
                    checks.append(
                        _check(
                            "fail",
                            f"live model {spec}",
                            f"{type(exc).__name__}: {exc}",
                            scopes=("model",),
                        )
                    )
                else:
                    checks.append(
                        _check(
                            "ok",
                            f"live model {spec}",
                            "reachable",
                            scopes=("model",),
                        )
                    )
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
        seen_sites: set[tuple[str, str, str]] = set()
        raw_model_overrides = resolved_models.get("overrides") or {}
        model_overrides = (
            {str(key): value for key, value in raw_model_overrides.items()}
            if isinstance(raw_model_overrides, Mapping)
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
                effective_routing.append(
                    {
                        "participant": participant,
                        "action": action,
                        "kind": "model",
                        "configuration": configuration,
                        "effective": selected,
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

    return {
        "site_root": str(workspace.directory),
        "project": {
            "name": manifest.get("name"),
            "root": str(workspace.root),
            "manifest": str(workspace.manifest_path),
            "workflow": workflow_spec,
            "specification": str(workspace.specification_path),
        },
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


def _load_project_workflow(workspace: Workspace, workflow_spec: str):
    from zippergen.workflow_io import load_workflow_spec, project_directory

    with project_directory(workspace.root):
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
    for name in sorted(used):
        configuration = configurations.get(name) or {}
        provider = configuration.get("provider")
        connection = str(configuration.get("connection") or "")
        telegram = workspace.provider_secret(connection, "bot_token")
        if provider == "telegram" and telegram:
            try:
                from zippergen.telegram_notify import TelegramBotClient

                client = TelegramBotClient(telegram, timeout=5)
                client.request("getMe")
                client.request("getChat", chat_id=configuration.get("chat_id"))
            except Exception as exc:
                checks.append(
                    _check(
                        "fail",
                        f"live connector {name}",
                        f"{type(exc).__name__}: {exc}",
                        scopes=("connector",),
                    )
                )
            else:
                checks.append(
                    _check(
                        "ok",
                        f"live connector {name}",
                        "reachable",
                        scopes=("connector",),
                    )
                )
    if workflow is None or module is None:
        return
    requirements = connector_requirements_from_module(module)
    google_pairs: dict[str, list[tuple[str, str]]] = {}
    for item in requirements:
        configuration = configurations.get(bindings.get(item.name, ""), {})
        connection = str(configuration.get("connection") or "")
        if (
            bindings.get(item.name) in used
            and item.kind in {"gmail", "google-sheets"}
            and connection
        ):
            google_pairs.setdefault(connection, []).append(
                (item.kind, item.access)
            )
    for name in extra_names:
        configuration = configurations.get(name) or {}
        kind = configuration.get("kind")
        connection = str(configuration.get("connection") or "")
        pair = (str(kind), "read-only")
        pairs = google_pairs.setdefault(connection, []) if connection else []
        if kind in {"gmail", "google-sheets"} and pair not in pairs:
            pairs.append(pair)
    for connection, pairs in sorted(google_pairs.items()):
        credential = workspace.provider_secret(connection, "authorized_user_json")
        if not credential:
            continue
        try:
            from zippergen.google_auth import (
                check_google_authorization,
                google_scopes_for_access,
            )

            refreshed = check_google_authorization(
                credential,
                scopes=google_scopes_for_access(pairs),
            )
            workspace.save_provider_secret(
                connection, "authorized_user_json", refreshed
            )
        except Exception as exc:
            checks.append(
                _check(
                    "fail",
                    f"live provider {connection}",
                    f"{type(exc).__name__}: {exc}",
                    scopes=("connector", "provider"),
                )
            )
        else:
            checks.append(
                _check(
                    "ok",
                    f"live provider {connection}",
                    "Google authorization refreshed",
                    scopes=("connector", "provider"),
                )
            )


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


def _idle_release_display(item: Mapping[str, object]) -> str:
    value = item.get("idle_timeout")
    provider = _provider(str(item.get("spec") or ""))
    if provider != "local":
        return "not applicable"
    if value is None or str(value).strip() == "":
        return "not set"
    seconds = float(str(value))
    return "after each call" if seconds == 0 else f"after {seconds:g} s"


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
    renderer.framed_section("Providers")
    providers = report.get("providers") or {}
    assert isinstance(providers, dict)
    _render_columns_or_empty(
        renderer,
        "Connections",
        ("Name", "Kind", "Site endpoint"),
        [
            (
                item.get("name"),
                item.get("kind"),
                item.get("base_url") or "provider default",
            )
            for item in providers.get("connections") or []
            if isinstance(item, dict)
        ],
        empty="No connections.",
    )
    renderer.framed_section("Models")
    models = report["models"]
    assert isinstance(models, dict)
    configurations = models.get("configurations") or []
    model_configuration_rows = [
        (
            item.get("name"),
            item.get("connection") or "-",
            item.get("model") or "-",
            _idle_release_display(item),
            item.get("source"),
        )
        for item in configurations
        if isinstance(item, dict)
    ]
    _render_columns_or_empty(
        renderer,
        "Configurations",
        ("Name", "Connection", "Model", "Idle release", "Source"),
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
            item.get("connection") or "-",
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
        ("Name", "Kind", "Connection", "Resource"),
        connector_configuration_rows,
        empty="No configurations.",
    )
    _render_columns_or_empty(
        renderer,
        "Slots",
        ("Target", "What it is", "Configuration"),
        [
            (item["target"], item["meaning"], item["configuration"])
            for item in (connectors.get("slots") or [])
            if isinstance(item, dict)
        ],
        empty="This workflow has no connector slots.",
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
    raw_site_facts = report.get("site_facts") or []
    site_facts = raw_site_facts if isinstance(raw_site_facts, list) else []
    _render_columns_or_empty(
        renderer,
        "Local requirements",
        ("Status", "Kind", "Requirement"),
        [
            (
                renderer.status_mark(
                    "success" if item.get("available") else "error"
                ),
                item.get("kind"),
                item.get("name"),
            )
            for item in site_facts
            if isinstance(item, dict)
        ],
        empty="No local credentials or tools are required.",
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


def render_readiness(
    report: dict[str, object],
    renderer: TerminalRenderer,
) -> None:
    """Render one live readiness view grouped by participant and dependency."""

    renderer.framed_section("Project readiness")
    project = report.get("project") or {}
    assert isinstance(project, dict)
    renderer.table(
        "Workflow",
        [
            ("Project", project.get("name"), None),
            ("Workflow", project.get("workflow") or "not resolved", None),
            (
                "Overall",
                "ready" if report.get("valid") else "not ready",
                "success" if report.get("valid") else "error",
            ),
        ],
    )
    routes = report.get("effective_routing") or []
    route_rows = routes if isinstance(routes, list) else []
    _render_columns_or_empty(
        renderer,
        "Effective routing",
        ("Status", "Participant", "Action", "Kind", "Configuration", "Effective"),
        [
            (
                renderer.status_mark(
                    "success" if item.get("available") else "error"
                ),
                item.get("participant"),
                item.get("action"),
                item.get("kind"),
                item.get("configuration"),
                item.get("effective"),
            )
            for item in route_rows
            if isinstance(item, dict)
        ],
        empty="No model, assistant, human, or connector routes.",
    )
    raw_checks = report.get("checks") or []
    checks = raw_checks if isinstance(raw_checks, list) else []
    categories = (
        (
            "Structure and assignments",
            lambda name: "live " not in name
            and "credential" not in name
            and " CLI " not in f" {name} ",
        ),
        (
            "Credentials and local tools",
            lambda name: "credential" in name or " CLI " in f" {name} ",
        ),
        ("Live providers", lambda name: name.startswith("live ")),
    )
    for title, selected in categories:
        rows = []
        for item in checks:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "")
            if not selected(name):
                continue
            rows.append(
                (
                    renderer.status_mark(
                        "success" if item.get("status") == "ok" else (
                            "warning" if item.get("status") == "warn" else "error"
                        )
                    ),
                    name,
                    item.get("detail"),
                )
            )
        _render_columns_or_empty(
            renderer,
            title,
            ("Status", "Check", "Detail"),
            rows,
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
            item.get("connection") or "-",
            item.get("model") or "-",
            _idle_release_display(item),
            item.get("source"),
        )
        for item in models.get("configurations") or []
        if isinstance(item, dict)
    ]
    _render_columns_or_empty(
        renderer,
        "Configurations",
        ("Name", "Connection", "Model", "Idle release", "Source"),
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
    _render_effective_routing(
        renderer,
        report,
        ("model",),
        subject="Action",
        resolved_header="Resolves to",
        resolved=lambda item: item.get("effective"),
        empty="No participant calls a model.",
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
    _render_effective_routing(
        renderer,
        report,
        ("assistant",),
        subject="Action",
        resolved_header="Backend",
        resolved=lambda item: item.get("effective"),
        empty="No participant runs an assistant.",
    )
    # Access, tools and shell are declared on the `@assistant` action, not
    # configured, so they answer "what may it do" rather than "what will it
    # use". Kept out of the routing table, where they sat at the right edge
    # and read as more routing.
    _render_columns_or_empty(
        renderer,
        "Permissions" if compact_titles else "Assistant permissions",
        ("Target", "Access", "Tools", "Shell"),
        [
            (
                item.get("target"),
                item.get("access"),
                item.get("external_tools"),
                item.get("shell"),
            )
            for item in assistants.get("resolved") or []
            if isinstance(item, dict)
        ],
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
            item.get("connection") or "-",
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
        ("Name", "Kind", "Connection", "Resource"),
        configuration_rows,
        empty="No configurations.",
    )
    _render_columns_or_empty(
        renderer,
        "Slots",
        ("Target", "What it is", "Configuration"),
        [
            (item["target"], item["meaning"], item["configuration"])
            for item in (connectors.get("slots") or [])
            if isinstance(item, dict)
        ],
        empty="This workflow has no connector slots.",
    )
    def _short(value: object) -> str:
        """Keep a long resource from squeezing out the columns beside it.

        The full value is in the Configurations table directly above, so this
        one only has to be recognisable.
        """

        text = str(value)
        return text if len(text) <= 22 else text[:21] + "\u2026"

    resources = {
        str(item.get("name")): (
            item.get("chat_id")
            or item.get("spreadsheet_id")
            or item.get("query")
            or item.get("account")
            or "-"
        )
        for item in connectors.get("configurations") or []
        if isinstance(item, dict)
    }
    _render_effective_routing(
        renderer,
        report,
        ("human", "telegram", "gmail", "google-sheets", "google-calendar"),
        subject="Slot",
        resolved_header="Resource",
        resolved=lambda item: _short(
            resources.get(str(item.get("configuration")), "-")
        ),
        empty="Nothing reaches outside this workflow.",
    )
    if show_checks:
        _render_selected_checks(report, renderer, "connector")


def render_provider_configuration(
    report: dict[str, object],
    renderer: TerminalRenderer,
    *,
    show_checks: bool = True,
) -> None:
    """Render named provider connections and their local readiness."""

    renderer.framed_section("Providers")
    providers = report.get("providers") or {}
    assert isinstance(providers, dict)
    _render_columns_or_empty(
        renderer,
        "Connections",
        ("Name", "Kind", "Site endpoint"),
        [
            (
                item.get("name"),
                item.get("kind"),
                item.get("base_url") or "provider default",
            )
            for item in providers.get("connections") or []
            if isinstance(item, dict)
        ],
        empty="No connections.",
    )
    if show_checks:
        _render_selected_checks(report, renderer, "provider")


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
    connection: str,
    model: str,
    *,
    idle_timeout: float | None = None,
) -> dict[str, str]:
    selected_connection = connection.strip()
    selected_model = model.strip()
    values = {
        "connection": selected_connection,
        "model": selected_model,
    }
    if idle_timeout is not None:
        if not math.isfinite(idle_timeout) or idle_timeout < 0:
            raise WorkspaceError("Idle timeout must be a non-negative finite number.")
        values["idle_timeout"] = str(idle_timeout)
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
    profile = workspace.model_assignment_profile(workflow)
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
    return workspace.save_model_assignment_profile(
        workflow,
        default=default,
        lifelines=lifelines,
        actions=actions,
    )


def configure_assistant(
    workspace: Workspace,
    name: str,
    backend: str,
) -> dict[str, str]:
    """Save one named Codex or Claude configuration."""

    return workspace.save_assistant_configuration(name, backend)


def assistant_target_problem(workspace: Workspace, target: str) -> str | None:
    """Say why this is not an assistant target, or nothing if it is one.

    The CLI asks before it prompts for anything else, and the assignment asks
    again before it writes. One rule, so the two cannot disagree.
    """

    workflow = workspace.resolve_workflow()
    loaded, module = _load_project_workflow(workspace, workflow)
    known = set(assistant_targets(loaded, module))
    if target in known:
        return None
    explanation = (
        " Assistant assignments can target 'default', a lifeline that "
        "runs an @assistant action, or an exact Participant.action for "
        "an @assistant action."
    )
    if known == {"default"}:
        explanation += " This workflow defines no @assistant actions."
    return (
        f"Unknown assistant assignment target {target!r}."
        + explanation
        + " Available: "
        + (", ".join(sorted(known)) or "none")
    )


def assign_assistant(
    workspace: Workspace,
    target: str,
    configuration: str | None,
) -> dict[str, object]:
    """Assign a named coding-assistant configuration to one target."""

    workflow = workspace.resolve_workflow()
    problem = assistant_target_problem(workspace, target)
    if problem is not None:
        raise WorkspaceError(problem)
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


CONNECTOR_REQUIREMENT = "connector requirement"
HUMAN_ACTION = "human action"
AMBIGUOUS_TARGET = "ambiguous"


def connector_target_kinds(workflow, module) -> dict[str, str]:
    """Every name ``zg connector assign`` accepts, and what each one means.

    A workflow offers two kinds of slot. It declares service requirements by
    name, and it has participants -- or single actions of theirs -- that ask a
    human something. Both are filled with a named connector configuration, so
    there is one verb for both and the name itself says which is meant.

    A name that means both is reported as ambiguous rather than resolved by a
    tie-break nobody could predict. That is a fault in the workflow, and only
    the workflow can fix it.
    """

    sites = human_action_sites(workflow, module)
    targets = {
        name: HUMAN_ACTION
        for name in (
            *(("default",) if sites else ()),
            *sites,
            *(
                f"{participant}.{action}"
                for participant, actions in sites.items()
                for action in actions
            ),
        )
    }
    for requirement in connector_requirements_from_module(module):
        targets[requirement.name] = (
            AMBIGUOUS_TARGET
            if requirement.name in targets
            else CONNECTOR_REQUIREMENT
        )
    return targets


def project_google_scopes(workspace: Workspace, connection: str) -> tuple[str, ...]:
    """Work out which Google scopes this project actually needs.

    The workflow already states it: every requirement declares its kind and
    how much access it wants, and the configuration assigned to it names the
    connection. So the answer is derivable, and asking a person to retype it
    only invites granting too much.
    """

    from zippergen.google_auth import google_scopes_for_access

    workflow = workspace.resolve_workflow()
    _loaded, module = _load_project_workflow(workspace, workflow)
    requirements = connector_requirements_from_module(module)
    bindings = workspace.connector_binding_profile(workflow)
    configurations = workspace.connector_configurations()
    assigned = [
        (requirement.kind, requirement.access)
        for requirement in requirements
        if (
            configurations.get(bindings.get(requirement.name) or "") or {}
        ).get("connection") == connection
    ]
    if assigned:
        return google_scopes_for_access(assigned)
    # Nothing is assigned yet, which is the normal state: authorizing is one
    # of the first things anybody does. When the project has a single Google
    # connection, every Google requirement in the workflow must end up on it,
    # so the answer is known even before anything is wired.
    google_connections = [
        name
        for name, profile in workspace.provider_connections().items()
        if str(profile.get("kind") or "") == "google"
    ]
    if google_connections == [connection]:
        return google_scopes_for_access(
            (requirement.kind, requirement.access)
            for requirement in requirements
        )
    return ()


def connector_target_problem(workspace: Workspace, target: str) -> str | None:
    """Say why this is not a connector target, or nothing if it is one.

    The CLI asks before it prompts for anything else, and the assignment asks
    again before it writes. One rule, so the two cannot disagree.
    """

    workflow = workspace.resolve_workflow()
    loaded, module = _load_project_workflow(workspace, workflow)
    return _connector_target_problem(
        target, connector_target_kinds(loaded, module), module
    )


def _connector_target_problem(
    target: str,
    targets: dict[str, str],
    module,
) -> str | None:
    kind = targets.get(target)
    if kind == AMBIGUOUS_TARGET:
        return (
            f"{target!r} is both a declared connector requirement and a "
            "human-action target, so there is no way to tell which one you "
            "mean. Rename the requirement in the workflow."
        )
    if kind is not None:
        return None
    return _unknown_connector_target(target, targets, module)


def _unknown_connector_target(
    target: str,
    targets: dict[str, str],
    module,
) -> str:
    """Explain a rejected target, and point at the one that was probably meant.

    The two kinds of slot are keyed differently -- a requirement by its own
    name, a human action by the participant running it -- so naming a
    participant that owns a requirement is the natural first guess and
    deserves an answer rather than a list.
    """

    owned = [
        requirement.name
        for requirement in connector_requirements_from_module(module)
        if requirement.participant == target
    ]
    if owned:
        return (
            f"{target!r} has no human action, so there is nothing to route to "
            f"a person. It owns the connector requirement"
            + (
                f" {owned[0]!r}. Assign that instead."
                if len(owned) == 1
                else "s " + ", ".join(repr(name) for name in owned)
                + ". Assign one of those instead."
            )
        )
    return (
        f"Unknown connector target {target!r}. Available: "
        + (", ".join(sorted(targets)) or "none")
    )


def assign_connector(
    workspace: Workspace,
    target: str,
    configuration: str | None,
) -> str:
    """Attach a configuration to whatever ``target`` names, and say which kind.

    One verb covers both kinds of slot. Which one is being filled follows from
    the name, so the caller never has to say it.
    """

    workflow = workspace.resolve_workflow()
    loaded, module = _load_project_workflow(workspace, workflow)
    targets = connector_target_kinds(loaded, module)
    problem = _connector_target_problem(target, targets, module)
    if problem is not None:
        raise WorkspaceError(problem)
    kind = targets[target]
    if kind == CONNECTOR_REQUIREMENT:
        _assign_requirement(workspace, workflow, module, target, configuration)
    else:
        _assign_human_action(workspace, workflow, target, configuration)
    return kind


def _assign_human_action(
    workspace: Workspace,
    workflow: str,
    target: str,
    configuration: str | None,
) -> dict[str, dict[str, str]]:
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
    default = str(profile.get("default") or "")
    lifelines = dict(profile.get("lifelines") or {})
    actions = dict(profile.get("actions") or {})
    if target == "default":
        default = configuration or ""
    else:
        selected = actions if "." in target else lifelines
        if configuration is None:
            selected.pop(target, None)
        else:
            selected[target] = configuration
    return workspace.save_connector_assignment_profile(
        workflow,
        default=default,
        lifelines=lifelines,
        actions=actions,
    )


def _assign_requirement(
    workspace: Workspace,
    workflow: str,
    module,
    requirement: str,
    configuration: str | None,
) -> dict[str, str]:
    if configuration is None:
        return workspace.unbind_connector(workflow, requirement)
    requirements = {
        item.name: item for item in connector_requirements_from_module(module)
    }
    selected_requirement = requirements[requirement]
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
