"""Validate and apply project configuration changes."""

from __future__ import annotations

import math

from zippergen.assistant_configuration import assistant_targets
from zippergen.configuration_inventory import _load_project_workflow, _model_targets
from zippergen.connector_wiring import human_action_sites
from zippergen.connectors import connector_requirements_from_module
from zippergen.workspace import Workspace, WorkspaceError


CONNECTOR_REQUIREMENT = "connector requirement"
HUMAN_ACTION = "human action"
AMBIGUOUS_TARGET = "ambiguous"


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
) -> dict[str, object]:
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
