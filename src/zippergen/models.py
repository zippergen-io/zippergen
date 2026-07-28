"""Shared helpers for default, participant, and action LLM configuration."""

from __future__ import annotations

from collections.abc import Mapping

from zippergen.syntax import Workflow, _ordered_workflow_lifelines


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
    """Expand defaults and validate participant or ``Participant.action`` routes."""

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
    routes = {name: selected.get(name, default) for name in names}
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
