"""Shared helpers for default, participant, and action LLM configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Protocol

from zippergen.syntax import Workflow, _ordered_workflow_lifelines


class _ModelWorkspace(Protocol):
    def has_model_assignment_profile(self, workflow_spec: str) -> bool: ...

    def model_assignment_profile(
        self,
        workflow_spec: str,
        *,
        default: str = "mock",
    ) -> dict[str, object]: ...

    def model_configurations(self) -> dict[str, dict[str, str]]: ...


@dataclass(frozen=True)
class ModelRouting:
    """Concrete runtime model specs resolved from named project settings."""

    default_spec: str
    overrides: dict[str, str]
    idle_timeouts: dict[str, float]


def normalize_llm_overrides(values: object) -> dict[str, str]:
    """Return a string mapping from persisted or CLI-provided model overrides."""

    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise SystemExit(
            "Participant and action LLM configuration must be an object."
        )
    normalized: dict[str, str] = {}
    for lifeline, spec in values.items():
        name = str(lifeline).strip()
        model = str(spec).strip()
        if not name or not model:
            raise SystemExit(
                "LLM route entries require PARTICIPANT_OR_ACTION=SPEC."
            )
        normalized[name] = model
    return normalized


def effective_llm_routes(
    workflow: Workflow,
    default_spec: str,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Expand defaults for participants that actually contain LLM actions.

    All participant names remain valid assignment targets so that a stored
    profile can survive workflow edits.  The returned runtime routes are
    deliberately narrower: participants without an LLM action never call an
    LLM backend and therefore must not acquire model-lifecycle policy.
    """

    default = str(default_spec).strip()
    if not default:
        raise SystemExit("The default LLM spec must not be empty.")
    names = [lifeline.name for lifeline in _ordered_workflow_lifelines(workflow)]
    from zippergen.semantic import workflow_semantics

    raw_sites = workflow_semantics(workflow).get("action_sites", [])
    sites = raw_sites if isinstance(raw_sites, list) else []
    action_targets = {
        f"{site['lifeline']}.{site['action']}"
        for site in sites
        if isinstance(site, dict) and site.get("kind") == "llm"
    }
    active_participants = {
        str(site["lifeline"])
        for site in sites
        if (
            isinstance(site, dict)
            and site.get("kind") == "llm"
            and site.get("lifeline")
        )
    }
    selected = normalize_llm_overrides(overrides)
    known = {*names, *action_targets}
    unknown = sorted(set(selected) - known)
    if unknown:
        raise SystemExit(
            "Unknown participant or LLM action target(s): "
            + ", ".join(unknown)
            + ". Available targets: "
            + ", ".join([*names, *sorted(action_targets)])
            + "."
        )
    routes = {
        name: selected.get(name, default)
        for name in names
        if name in active_participants
    }
    routes.update(
        {
            target: selected[target]
            for target in sorted(action_targets)
            if target in selected
        }
    )
    return routes


def selected_llm_specs(
    default_spec: object,
    overrides: object = None,
) -> tuple[str, ...]:
    """List unique model specs used by conditional secret declarations."""

    specs: list[str] = []
    if default_spec is not None and str(default_spec).strip():
        specs.append(str(default_spec).strip())
    for spec in normalize_llm_overrides(overrides).values():
        if spec not in specs:
            specs.append(spec)
    return tuple(specs)


def project_model_routing(
    workspace: _ModelWorkspace,
    workflow_spec: str,
    workflow: Workflow,
    *,
    fallback_default: str = "mock",
) -> ModelRouting:
    """Resolve portable named assignments to concrete runtime model specs.

    A project assignment names a configuration, while the runtime needs the
    configuration's compact provider spec.  Keeping that translation here
    gives plain runs, durable runs, and deployments exactly one interpretation
    of ``zippergen.toml``.

    When the project has no model assignment profile, the workflow's ordinary
    fallback remains in force.  Reading configuration never creates a profile.
    """

    fallback = str(fallback_default).strip() or "mock"
    if not workspace.has_model_assignment_profile(workflow_spec):
        return ModelRouting(fallback, {}, {})

    profile = workspace.model_assignment_profile(
        workflow_spec,
        default="mock",
    )
    configurations = workspace.model_configurations()

    def configuration(name: object, *, target: str) -> dict[str, str]:
        selected = str(name).strip()
        value = configurations.get(selected)
        if value is None:
            raise SystemExit(
                f"Model assignment for {target} names unknown configuration "
                f"{selected!r}. Define it in "
                "[models.configurations] in zippergen.toml."
            )
        spec = str(value.get("spec") or "").strip()
        if not spec:
            raise SystemExit(
                f"Model configuration {selected!r} has no model spec."
            )
        return value

    default_name = str(profile.get("default") or "mock")
    default_configuration = configuration(default_name, target="the default")
    default_spec = str(default_configuration["spec"])

    named_overrides: dict[str, str] = {}
    target_configurations: dict[str, dict[str, str]] = {}
    for group in ("lifelines", "actions"):
        raw = profile.get(group) or {}
        if not isinstance(raw, Mapping):
            raise SystemExit(f"Project model {group} assignments must be an object.")
        for target, name in raw.items():
            target_name = str(target).strip()
            selected = configuration(name, target=target_name)
            named_overrides[target_name] = str(selected["spec"])
            target_configurations[target_name] = selected

    # Validate names and determine which participant defaults are active.  A
    # stored assignment may outlive an action only until validation, where the
    # explicit error is much safer than silently routing another model.
    routes = effective_llm_routes(workflow, default_spec, named_overrides)
    idle_timeouts: dict[str, float] = {}
    for target in routes:
        participant = target.partition(".")[0]
        selected = target_configurations.get(
            target,
            target_configurations.get(participant, default_configuration),
        )
        raw_timeout = str(selected.get("idle_timeout") or "").strip()
        if not raw_timeout:
            continue
        try:
            timeout = float(raw_timeout)
        except ValueError as exc:
            raise SystemExit(
                f"Model configuration idle timeout for {target} must be a number."
            ) from exc
        if not math.isfinite(timeout) or timeout < 0:
            raise SystemExit(
                f"Model configuration idle timeout for {target} must be a "
                "non-negative finite number."
            )
        idle_timeouts[target] = timeout

    return ModelRouting(default_spec, named_overrides, idle_timeouts)


def apply_model_overrides(
    routing: ModelRouting,
    *,
    default_spec: str | None = None,
    overrides: Mapping[str, str] | None = None,
    idle_timeouts: Mapping[str, float] | None = None,
) -> ModelRouting:
    """Apply direct command-line choices over one project routing snapshot.

    A global ``--llm`` means every action, so it intentionally clears project
    participant and action assignments.  A targeted override changes only that
    target.  ``inherit`` and ``default`` remove a targeted assignment.
    """

    if default_spec is not None:
        selected_default = str(default_spec).strip()
        if not selected_default:
            raise SystemExit("The default LLM spec must not be empty.")
        selected_overrides: dict[str, str] = {}
        selected_idle_timeouts: dict[str, float] = {}
    else:
        selected_default = routing.default_spec
        selected_overrides = dict(routing.overrides)
        selected_idle_timeouts = dict(routing.idle_timeouts)

    for target, spec in normalize_llm_overrides(overrides).items():
        selected_idle_timeouts.pop(target, None)
        if spec.casefold() in {"inherit", "default"}:
            selected_overrides.pop(target, None)
        else:
            selected_overrides[target] = spec
    selected_idle_timeouts.update(
        {str(target): float(value) for target, value in (idle_timeouts or {}).items()}
    )
    return ModelRouting(
        selected_default,
        selected_overrides,
        selected_idle_timeouts,
    )
