"""Discover the configuration slots and routes offered by a workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zippergen.workspace import ProjectManifest

from collections.abc import Mapping
from types import ModuleType

from zippergen.connector_wiring import human_action_sites
from zippergen.connectors import connector_requirements_from_module
from zippergen.semantic import workflow_semantics
from zippergen.syntax import Workflow
from zippergen.workspace import Workspace


def _load_project_workflow(workspace: Workspace, workflow_spec: str):
    """Load the selected workflow with paths relative to the project root."""

    from zippergen.workflow_io import load_workflow_spec, project_directory

    with project_directory(workspace.root):
        return load_workflow_spec(workspace.absolute_spec(workflow_spec))


def _provider(spec: str) -> str:
    from zippergen.provider_connections import split_model_spec

    try:
        provider, _connection, _model = split_model_spec(spec)
    except ValueError:
        return spec.partition(":")[0].partition("@")[0].strip().casefold()
    return provider


def _project_model_source(
    project: "ProjectManifest",
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
    assignments: Mapping[str, object],
    bindings: Mapping[str, str],
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


# Translate assistant-specific resolution labels into the vocabulary shared
# by models and connectors.
_ASSISTANT_SOURCE_WORDS = {
    "action assignment": "action",
    "participant assignment": "participant",
    "default assignment": "default",
    "runtime default": "default",
    "missing": "unassigned",
}
