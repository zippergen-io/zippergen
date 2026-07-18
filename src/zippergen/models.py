"""Shared helpers for default and per-lifeline LLM configuration."""

from __future__ import annotations

from collections.abc import Mapping

from zippergen.syntax import Workflow, _ordered_workflow_lifelines


def normalize_llm_overrides(values: object) -> dict[str, str]:
    """Return a string mapping from persisted or CLI-provided model overrides."""

    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise SystemExit("Per-lifeline LLM configuration must be an object.")
    normalized: dict[str, str] = {}
    for lifeline, spec in values.items():
        name = str(lifeline).strip()
        model = str(spec).strip()
        if not name or not model:
            raise SystemExit("Per-lifeline LLM entries require LIFELINE=SPEC.")
        normalized[name] = model
    return normalized


def effective_llm_routes(
    workflow: Workflow,
    default_spec: str,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Expand a default plus overrides to an exact route for every lifeline."""

    default = str(default_spec).strip()
    if not default:
        raise SystemExit("The default LLM spec must not be empty.")
    names = [lifeline.name for lifeline in _ordered_workflow_lifelines(workflow)]
    selected = normalize_llm_overrides(overrides)
    unknown = sorted(set(selected) - set(names))
    if unknown:
        raise SystemExit(
            "Unknown lifeline(s) in LLM configuration: "
            + ", ".join(unknown)
            + ". Available lifelines: "
            + ", ".join(names)
            + "."
        )
    return {name: selected.get(name, default) for name in names}


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
