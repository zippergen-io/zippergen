"""Evaluate static and live project-configuration readiness."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

from zippergen.configuration_inventory import _provider, _used_connector_names
from zippergen.connector_wiring import human_action_sites
from zippergen.connectors import connector_requirements_from_module
from zippergen.google_auth import google_support_installed
from zippergen.models import ModelSettings, selected_llm_specs
from zippergen.provider_connections import (
    provider_credential_field,
    provider_credential_label,
    provider_standard_environment,
    split_model_spec,
)
from zippergen.syntax import LLMAction, Workflow
from zippergen.workspace import Workspace


Check = dict[str, object]


def _check(
    status: str,
    name: str,
    detail: str,
    *,
    scopes: Sequence[str] = (),
    fix: str = "",
) -> Check:
    """Describe one check, what it depends on, and how to satisfy it.

    A diagnostic that reports a hole without naming the command that fills it
    leaves the reader to reconstruct the order of a dozen setup commands from
    the manual. `fix` carries that command, so reading the check is enough.
    """

    return {
        "status": status,
        "name": name,
        "detail": detail,
        "scopes": tuple(scopes),
        "fix": fix,
    }


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
                    fix=(
                        f"zippergen connector configure NAME CONNECTION "
                        f"{requirement.kind}, then zippergen connector assign "
                        f"{requirement.name} NAME"
                    ),
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
                    fix=(
                        f"zippergen connector configure {configuration} "
                        f"CONNECTION {requirement.kind}"
                    ),
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


def _settings_for_specs(
    resolved_models: dict[str, object],
) -> dict[str, "ModelSettings"]:
    """Map each routed spec to the settings that will be used with it.

    Settings are recorded per target -- a participant or an action -- while a
    live check is per spec, because one spec may serve several targets. Where
    targets on one spec disagree, no settings are used rather than a guess.
    """

    from zippergen.models import model_settings_from_mapping

    raw = resolved_models.get("settings")
    if not isinstance(raw, Mapping):
        return {}
    overrides = resolved_models.get("overrides")
    default_spec = str(resolved_models.get("default") or "")
    routes = dict(overrides) if isinstance(overrides, Mapping) else {}

    by_spec: dict[str, ModelSettings] = {}
    conflicting: set[str] = set()
    for target, value in raw.items():
        spec = str(routes.get(target, routes.get(str(target).partition(".")[0], default_spec)))
        chosen = model_settings_from_mapping(value, subject=str(target))
        if spec in by_spec and by_spec[spec] != chosen:
            conflicting.add(spec)
        by_spec[spec] = chosen
    for spec in conflicting:
        by_spec.pop(spec, None)
    return by_spec


def _live_model_check(
    spec: str,
    environment: dict[str, str],
    *,
    project_root: Path,
    settings: "ModelSettings | None" = None,
) -> None:
    """Reach the provider the way the workflow will, settings included.

    Checking with a backend's own defaults can pass where the configured
    timeout would fail, or fail where the configured one would not. A readiness
    check that does not use the configured settings is not checking readiness.
    """

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
            settings=settings,
        )
        backend(action, {})


def _site_checks(
    workspace: Workspace,
    *,
    include_site_checks: bool,
    live: bool,
    provider_names: tuple[str, ...],
    model_names: tuple[str, ...],
    assistant_names: tuple[str, ...],
    connector_names: tuple[str, ...],
    resolved_models: dict[str, object],
    model_configurations: dict[str, dict[str, str]],
    assistant_configurations: dict[str, dict[str, str]],
    resolved_assistants: list[dict[str, object]],
    connector_configurations: dict[str, dict[str, str]],
    connector_assignments: dict[str, Any],
    connector_bindings: dict[str, str],
    provider_connections: dict[str, dict[str, str]],
    workflow: Workflow | None,
    module: ModuleType | None,
    checks: list[Check],
) -> list[dict[str, object]]:
    """Return site facts and append their static/live results to ``checks``."""

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
                        fix=f"zippergen provider configure {connection} KIND",
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
                    fix=(
                        ""
                        if available
                        else (
                            f"zippergen provider authorize {connection}"
                            if kind == "google"
                            else f"zippergen provider set-credential {connection}"
                        )
                    ),
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
            settings_by_spec = _settings_for_specs(resolved_models)
            unique_specs = dict.fromkeys(specs)
            for spec in unique_specs:
                try:
                    _live_model_check(
                        spec,
                        environment,
                        project_root=workspace.root,
                        settings=settings_by_spec.get(spec),
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
    return site_facts


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
