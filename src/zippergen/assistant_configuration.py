"""Named project routing for repository-aware coding-assistant actions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Protocol

from zippergen.syntax import AssistantAction, Workflow
from zippergen.assistant_backends import ASSISTANT_BACKENDS


class _AssistantWorkspace(Protocol):
    def has_assistant_assignment_profile(self, workflow_spec: str) -> bool: ...

    def assistant_assignment_profile(
        self,
        workflow_spec: str,
    ) -> dict[str, object]: ...

    def assistant_configurations(self) -> dict[str, dict[str, str]]: ...


@dataclass(frozen=True)
class AssistantRouting:
    """Concrete assistant backends resolved from named project settings."""

    default_backend: str | None
    overrides: dict[str, str]


@dataclass(frozen=True)
class ResolvedAssistantAction:
    """Effective backend and source for one assistant action site."""

    target: str
    participant: str
    action: str
    backend: str | None
    configuration: str | None
    source: str
    access: str
    external_tools: str
    shell: str


def _assistant_sites(
    workflow: Workflow,
    module: ModuleType | None = None,
) -> list[tuple[str, str]]:
    from zippergen.semantic import workflow_semantics

    raw = workflow_semantics(workflow, module).get("action_sites") or []
    result: list[tuple[str, str]] = []
    for site in raw if isinstance(raw, list) else []:
        if not isinstance(site, dict) or site.get("kind") != "assistant":
            continue
        pair = (str(site.get("lifeline") or ""), str(site.get("action") or ""))
        if all(pair) and pair not in result:
            result.append(pair)
    return result


def assistant_targets(
    workflow: Workflow,
    module: ModuleType | None = None,
) -> list[str]:
    """Return valid participant and exact-action assignment targets."""

    targets: list[str] = ["default"]
    for participant, action in _assistant_sites(workflow, module):
        for target in (participant, f"{participant}.{action}"):
            if target not in targets:
                targets.append(target)
    return targets


def normalize_assistant_overrides(values: object) -> dict[str, str]:
    """Return a validated participant/action to backend mapping."""

    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise SystemExit("Assistant routing must be an object.")
    normalized: dict[str, str] = {}
    for target, backend in values.items():
        name = str(target).strip()
        selected = str(backend).strip().casefold()
        if not name or selected not in set(ASSISTANT_BACKENDS):
            raise SystemExit(
                "Assistant routes require TARGET=codex or TARGET=claude."
            )
        normalized[name] = selected
    return normalized


def effective_assistant_routes(
    workflow: Workflow,
    default_backend: str | None,
    overrides: Mapping[str, str] | None = None,
    *,
    module: ModuleType | None = None,
) -> AssistantRouting:
    """Validate concrete assistant routes against the workflow."""

    default = str(default_backend or "").strip().casefold() or None
    if default not in {None, *ASSISTANT_BACKENDS}:
        raise SystemExit("The default assistant backend must be codex or claude.")
    selected = normalize_assistant_overrides(overrides)
    known = set(assistant_targets(workflow, module)) - {"default"}
    unknown = sorted(set(selected) - known)
    if unknown:
        raise SystemExit(
            "Unknown participant or assistant action target(s): "
            + ", ".join(unknown)
            + ". Available targets: "
            + (", ".join(sorted(known)) or "none")
            + "."
        )
    return AssistantRouting(default, selected)


def project_assistant_routing(
    workspace: _AssistantWorkspace,
    workflow_spec: str,
    workflow: Workflow,
    *,
    module: ModuleType | None = None,
    fallback_default: str | None = None,
) -> AssistantRouting:
    """Resolve named project assignments to concrete CLI backends."""

    if not workspace.has_assistant_assignment_profile(workflow_spec):
        return effective_assistant_routes(
            workflow,
            fallback_default,
            module=module,
        )
    profile = workspace.assistant_assignment_profile(workflow_spec)
    configurations = workspace.assistant_configurations()

    def configuration(name: object, *, target: str) -> tuple[str, str]:
        selected = str(name or "").strip()
        value = configurations.get(selected)
        if value is None:
            raise SystemExit(
                f"Assistant assignment for {target} names unknown "
                f"configuration {selected!r}. Define it in "
                "[assistants.configurations] in zippergen.toml."
            )
        backend = str(value.get("backend") or "").strip().casefold()
        if backend not in set(ASSISTANT_BACKENDS):
            raise SystemExit(
                f"Assistant configuration {selected!r} must select codex or "
                "claude."
            )
        return selected, backend

    default_name = str(profile.get("default") or "").strip()
    default = fallback_default
    if default_name:
        _name, default = configuration(default_name, target="the default")
    overrides: dict[str, str] = {}
    for group in ("lifelines", "actions"):
        raw = profile.get(group) or {}
        if not isinstance(raw, Mapping):
            raise SystemExit(
                f"Project assistant {group} assignments must be an object."
            )
        for target, name in raw.items():
            target_name = str(target).strip()
            _configuration, backend = configuration(name, target=target_name)
            overrides[target_name] = backend
    return effective_assistant_routes(
        workflow,
        default,
        overrides,
        module=module,
    )


def apply_assistant_overrides(
    routing: AssistantRouting,
    *,
    default_backend: str | None = None,
    overrides: Mapping[str, str] | None = None,
    workflow: Workflow,
    module: ModuleType | None = None,
) -> AssistantRouting:
    """Apply an explicit runtime override after project routing."""

    merged = dict(routing.overrides)
    merged.update(normalize_assistant_overrides(overrides))
    return effective_assistant_routes(
        workflow,
        default_backend if default_backend is not None else routing.default_backend,
        merged,
        module=module,
    )


def resolved_assistant_actions(
    workflow: Workflow,
    routing: AssistantRouting,
    *,
    module: ModuleType | None = None,
    assignments: Mapping[str, object] | None = None,
) -> list[ResolvedAssistantAction]:
    """Describe the effective backend of every assistant action site."""

    from zippergen.validation import assistant_actions

    definitions: dict[str, AssistantAction] = {
        action.name: action for action in assistant_actions(workflow)
    }
    profile = assignments or {}
    lifelines = profile.get("lifelines") or {}
    actions = profile.get("actions") or {}
    default_configuration = str(profile.get("default") or "") or None
    assert isinstance(lifelines, Mapping)
    assert isinstance(actions, Mapping)
    rows: list[ResolvedAssistantAction] = []
    for participant, action_name in _assistant_sites(workflow, module):
        target = f"{participant}.{action_name}"
        action = definitions[action_name]
        configuration: str | None = None
        if target in actions:
            configuration = str(actions[target])
            backend = routing.overrides.get(target)
            source = "action assignment"
        elif participant in lifelines:
            configuration = str(lifelines[participant])
            backend = routing.overrides.get(participant)
            source = "participant assignment"
        elif default_configuration:
            configuration = default_configuration
            backend = routing.default_backend
            source = "default assignment"
        elif routing.default_backend:
            backend = routing.default_backend
            source = "runtime default"
        else:
            backend = None
            source = "missing"
        rows.append(
            ResolvedAssistantAction(
                target=target,
                participant=participant,
                action=action_name,
                backend=backend,
                configuration=configuration,
                source=source,
                access=action.access,
                external_tools=action.external_tools,
                shell=action.shell,
            )
        )
    return rows


__all__ = [
    "AssistantRouting",
    "ResolvedAssistantAction",
    "apply_assistant_overrides",
    "assistant_targets",
    "effective_assistant_routes",
    "normalize_assistant_overrides",
    "project_assistant_routing",
    "resolved_assistant_actions",
]
