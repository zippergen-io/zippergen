"""Named project routing for repository-aware coding-assistant actions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Protocol

from zippergen.syntax import AssistantAction, Workflow
from zippergen.assistant_backends import (
    ASSISTANT_BACKENDS, AssistantSelection, assistant_settings,
    normalize_assistant_selection,
)


class _AssistantWorkspace(Protocol):
    def has_assistant_assignment_profile(self, workflow_spec: str) -> bool: ...

    def assistant_assignment_profile(
        self,
        workflow_spec: str,
    ) -> dict[str, object]: ...

    def assistant_configurations(self) -> dict[str, dict[str, str]]: ...


@dataclass(frozen=True)
class AssistantRouting:
    """Concrete CLI selections, including model and effort pins."""

    default_backend: AssistantSelection | None
    overrides: dict[str, AssistantSelection]


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
    model: str | None = None
    reasoning_effort: str | None = None


def _assistant_sites(
    workflow: Workflow,
    module: ModuleType | None = None,
    semantics: Mapping[str, object] | None = None,
) -> list[tuple[str, str]]:
    from zippergen.semantic import workflow_semantics

    model = (
        semantics
        if semantics is not None
        else workflow_semantics(workflow, module)
    )
    raw = model.get("action_sites") or []
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
    semantics: Mapping[str, object] | None = None,
) -> list[str]:
    """Return valid participant and exact-action assignment targets."""

    targets: list[str] = ["default"]
    for participant, action in _assistant_sites(workflow, module, semantics):
        for target in (participant, f"{participant}.{action}"):
            if target not in targets:
                targets.append(target)
    return targets


def normalize_assistant_overrides(values: object) -> dict[str, AssistantSelection]:
    """Return validated participant/action CLI selections."""

    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise SystemExit("Assistant routing must be an object.")
    normalized: dict[str, AssistantSelection] = {}
    for target, backend in values.items():
        name = str(target).strip()
        try:
            selected = normalize_assistant_selection(backend)
        except ValueError as exc:
            raise SystemExit(f"Assistant route {name!r}: {exc}") from exc
        if not name or selected is None:
            raise SystemExit(
                "Assistant routes require TARGET to select one of: "
                + ", ".join(ASSISTANT_BACKENDS)
                + "."
            )
        normalized[name] = selected
    return normalized


def effective_assistant_routes(
    workflow: Workflow,
    default_backend: AssistantSelection | None,
    overrides: Mapping[str, AssistantSelection] | None = None,
    *,
    module: ModuleType | None = None,
    semantics: Mapping[str, object] | None = None,
) -> AssistantRouting:
    """Validate concrete assistant routes against the workflow."""

    try:
        default = normalize_assistant_selection(default_backend)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    selected = normalize_assistant_overrides(overrides)
    known = set(assistant_targets(workflow, module, semantics)) - {"default"}
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
    fallback_default: AssistantSelection | None = None,
    semantics: Mapping[str, object] | None = None,
) -> AssistantRouting:
    """Resolve named project assignments to concrete CLI backends."""

    if not workspace.has_assistant_assignment_profile(workflow_spec):
        return effective_assistant_routes(
            workflow,
            fallback_default,
            module=module,
            semantics=semantics,
        )
    profile = workspace.assistant_assignment_profile(workflow_spec)
    configurations = workspace.assistant_configurations()

    def configuration(name: object, *, target: str) -> tuple[str, AssistantSelection]:
        selected = str(name or "").strip()
        value = configurations.get(selected)
        if value is None:
            raise SystemExit(
                f"Assistant assignment for {target} names unknown "
                f"configuration {selected!r}. Define it in "
                "[assistants.configurations] in zippergen.toml."
            )
        try:
            concrete = normalize_assistant_selection(value)
        except ValueError as exc:
            raise SystemExit(f"Assistant configuration {selected!r}: {exc}") from exc
        assert concrete is not None
        return selected, concrete

    default_name = str(profile.get("default") or "").strip()
    default = fallback_default
    if default_name:
        _name, default = configuration(default_name, target="the default")
    overrides: dict[str, AssistantSelection] = {}
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
        semantics=semantics,
    )


def apply_assistant_overrides(
    routing: AssistantRouting,
    *,
    default_backend: AssistantSelection | None = None,
    overrides: Mapping[str, AssistantSelection] | None = None,
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
    semantics: Mapping[str, object] | None = None,
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
    for participant, action_name in _assistant_sites(workflow, module, semantics):
        target = f"{participant}.{action_name}"
        action = definitions[action_name]
        configuration: str | None = None
        selected = (
            routing.overrides.get(target)
            or routing.overrides.get(participant)
            or routing.default_backend
        )
        settings = assistant_settings(selected) if selected else {}
        if target in routing.overrides:
            configuration = str(actions[target]) if target in actions else None
            source = "action assignment" if configuration else "runtime action override"
        elif participant in routing.overrides:
            configuration = str(lifelines[participant]) if participant in lifelines else None
            source = "participant assignment" if configuration else "runtime participant override"
        elif default_configuration:
            configuration = default_configuration
            source = "default assignment"
        elif routing.default_backend:
            source = "runtime default"
        else:
            source = "missing"
        rows.append(
            ResolvedAssistantAction(
                target=target,
                participant=participant,
                action=action_name,
                backend=settings.get("backend"),
                configuration=configuration,
                source=source,
                access=action.access,
                external_tools=action.external_tools,
                shell=action.shell,
                model=settings.get("model"),
                reasoning_effort=settings.get("reasoning_effort"),
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
