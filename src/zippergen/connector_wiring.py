"""Turn a project's connector configuration into deployment wiring.

A workflow says *what* it needs — a place to ask a person, a mailbox to read, a
spreadsheet to write. The project says *which* one: which chat, which
spreadsheet, which query. This module joins the two and produces the routing a
deployment runs with, plus the secret values that routing refers to by name.

The split is the point. The snapshot is durable, portable and free of
credentials; the environment holds the values and never leaves the machine that
supplied them.

This is a pure configuration layer. Nothing here renders or prompts.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

from zippergen.semantic import workflow_semantics
from zippergen.syntax import Workflow, _ordered_workflow_lifelines
from zippergen.workspace import Workspace
from zippergen.deployment_platform import slug


class ConnectorWiringError(RuntimeError):
    """The project's connectors cannot be wired as configured."""


def _environment_name(configuration: str, suffix: str) -> str:
    return (
        "ZIPPERGEN_CONNECTOR_"
        + slug(configuration).replace("-", "_").upper()
        + suffix
    )


def human_action_sites(
    workflow: Workflow,
    module: ModuleType | None,
) -> dict[str, list[str]]:
    """Return each participant's `@human` actions, in protocol order."""

    model = workflow_semantics(workflow, module)
    actions: dict[str, list[str]] = {}
    sites = model.get("action_sites") or []
    if isinstance(sites, list):
        for site in sites:
            if not isinstance(site, dict) or site.get("kind") != "human":
                continue
            participant = str(site.get("lifeline"))
            action = str(site.get("action"))
            actions.setdefault(participant, [])
            if action not in actions[participant]:
                actions[participant].append(action)
    ordered = [item.name for item in _ordered_workflow_lifelines(workflow)]
    return {name: actions[name] for name in ordered if name in actions}


def _human_targets(workflow: Workflow, module: ModuleType | None) -> set[str]:
    sites = human_action_sites(workflow, module)
    return {
        *sites,
        *(f"{participant}.{action}"
          for participant, actions in sites.items()
          for action in actions),
    }


def _check_google_authorization(
    workspace: Workspace,
    requirements,
    bindings: dict[str, str],
) -> None:
    """Refuse before deploying if the granted scopes do not cover the workflow."""

    from zippergen.google_auth import (
        google_scope_names,
        google_scopes_cover,
        google_scopes_for_access,
    )

    pairs = [
        (item.kind, item.access)
        for item in requirements
        if item.name in bindings and item.kind in {"gmail", "google-sheets"}
    ]
    if not pairs:
        return

    required = google_scopes_for_access(pairs)
    profile = workspace.connector_provider_profiles().get("google")
    raw = (profile or {}).get("granted_scopes")
    granted: tuple[str, ...] = ()
    if isinstance(raw, (list, tuple)):
        granted = tuple(str(value) for value in raw)
    elif isinstance(raw, str) and raw:
        import json

        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw.split(",")
        if isinstance(value, list):
            granted = tuple(str(item) for item in value)

    if not granted:
        raise ConnectorWiringError(
            "Google is not authorized on this machine. Use "
            "'zippergen connector authorize google --scopes ...'."
        )
    if not google_scopes_cover(granted, required):
        missing = [
            name
            for scope, name in zip(required, google_scope_names(required), strict=True)
            if not google_scopes_cover(granted, (scope,))
        ]
        raise ConnectorWiringError(
            "Google authorization does not cover this workflow: "
            + ", ".join(missing)
            + ". Re-run 'zippergen connector authorize google' with those scopes."
        )


def connector_runtime(
    workspace: Workspace,
    workflow_spec: str,
    workflow: Workflow,
    module: ModuleType | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Return (snapshot, environment) for this project's connectors.

    The snapshot is durable routing with no secret in it. The environment maps
    the names the snapshot refers to onto the values this machine holds.
    """

    from zippergen.connectors import connector_requirements_from_module

    requirements = connector_requirements_from_module(module)
    assignments = workspace.connector_assignment_profile(workflow_spec)
    lifeline_assignments = dict(assignments.get("lifelines") or {})
    action_assignments = dict(assignments.get("actions") or {})
    if not requirements and not lifeline_assignments and not action_assignments:
        return {}, {}

    bindings = workspace.connector_binding_profile(workflow_spec)
    configurations = workspace.connector_configurations()

    unbound = [
        item.name
        for item in requirements
        if item.required and item.name not in bindings
    ]
    if unbound:
        raise ConnectorWiringError(
            "These connector requirements are not bound to a configuration: "
            + ", ".join(unbound)
            + ". Use 'zippergen connector bind REQUIREMENT CONFIGURATION'."
        )
    _check_google_authorization(workspace, requirements, bindings)

    snapshot: dict[str, Any] = {}
    environment: dict[str, str] = {}

    def resolved(name: str) -> tuple[dict[str, str], str, dict[str, str]]:
        configuration = configurations.get(name)
        if configuration is None:
            raise ConnectorWiringError(
                f"Connector configuration {name!r} does not exist."
            )
        provider = str(
            configuration.get("provider") or configuration.get("kind") or ""
        )
        secrets: dict[str, str] = {}
        if provider == "telegram":
            token = workspace.connector_provider_secret(
                "telegram", "bot_token"
            ) or workspace.connector_secret(name, "bot_token")
            if not token:
                raise ConnectorWiringError(
                    "The Telegram bot token is missing on this machine. Use "
                    "'zippergen connector configure NAME telegram'."
                )
            secrets["bot_token"] = token
        elif provider == "google":
            credential = workspace.connector_provider_secret(
                "google", "authorized_user_json"
            )
            if not credential:
                raise ConnectorWiringError(
                    "Google is not authorized on this machine. Use "
                    "'zippergen connector authorize google --scopes ...'."
                )
            secrets["authorized_user_json"] = credential
        return configuration, provider, secrets

    known_targets = _human_targets(workflow, module)
    for target, name in [*lifeline_assignments.items(), *action_assignments.items()]:
        if target not in known_targets:
            raise ConnectorWiringError(
                f"{target} is assigned a connector but has no human action. "
                "Remove the assignment, or check the participant name."
            )
        configuration, provider, secrets = resolved(name)
        if provider != "telegram":
            raise ConnectorWiringError(
                f"{target} needs a connector that can ask a person, but "
                f"{name} is {provider or 'unconfigured'}."
            )
        token_env = _environment_name(provider, "_TOKEN")
        environment[token_env] = secrets["bot_token"]
        participant, _, action = target.partition(".")
        snapshot[f"human:{target}"] = {
            "type": "human",
            "target": target,
            "participant": participant,
            "action": action or None,
            "kind": provider,
            "provider": provider,
            "configuration": name,
            "chat_id": configuration.get("chat_id"),
            "channel": configuration.get("channel") or f"telegram:{name}",
            "token_env": token_env,
        }

    for requirement in requirements:
        name = bindings.get(requirement.name)
        if name is None:
            continue
        configuration, provider, secrets = resolved(name)
        kind = str(configuration.get("kind") or "")
        if kind != requirement.kind:
            raise ConnectorWiringError(
                f"{requirement.name} needs a {requirement.kind} connector, "
                f"but {name} is {kind or 'unknown'}."
            )
        record: dict[str, Any] = {
            **requirement.as_dict(),
            "provider": provider,
            "configuration": name,
            "channel": configuration.get("channel") or requirement.name,
        }
        if requirement.kind == "telegram":
            token_env = _environment_name(name, "_TOKEN")
            record.update(
                {"chat_id": configuration.get("chat_id"), "token_env": token_env}
            )
            environment[token_env] = secrets["bot_token"]
        elif requirement.kind in {"google-sheets", "gmail"}:
            credential_env = _environment_name(name, "_GOOGLE_CREDENTIAL")
            if requirement.kind == "google-sheets":
                record.update(
                    {
                        "spreadsheet_id": configuration.get("spreadsheet_id"),
                        "tab": configuration.get("tab"),
                        "credential_env": credential_env,
                    }
                )
            else:
                record.update(
                    {
                        "account": configuration.get("account") or "me",
                        "query": configuration.get("query")
                        or "is:unread in:inbox",
                        "credential_env": credential_env,
                    }
                )
            environment[credential_env] = secrets["authorized_user_json"]
        snapshot[f"requirement:{requirement.name}"] = record

    return snapshot, environment
