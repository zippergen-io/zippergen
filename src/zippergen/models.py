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
class ModelSettings:
    """How one model target is invoked, as a single value.

    These belong together because they are answered together and travel
    together: a setting reaches a backend through the routing, the runtime, the
    workflow and the deployment profile. Carrying one dictionary per setting
    instead meant that adding an ordinary knob like ``max_tokens`` touched
    roughly twenty call sites, so it was easier to route it through an
    environment variable than to add it properly -- which is how a standard
    inference setting came to be configured differently from ``temperature``.

    ``None`` means "not set here": the backend keeps its own default.
    """

    temperature: float | None = None
    max_tokens: int | None = None
    timeout: float | None = None
    idle_timeout: float | None = None

    def merged_with(self, other: "ModelSettings") -> "ModelSettings":
        """Return *other*'s stated values laid over this one's."""

        return ModelSettings(
            temperature=(
                self.temperature if other.temperature is None else other.temperature
            ),
            max_tokens=(
                self.max_tokens if other.max_tokens is None else other.max_tokens
            ),
            timeout=self.timeout if other.timeout is None else other.timeout,
            idle_timeout=(
                self.idle_timeout
                if other.idle_timeout is None
                else other.idle_timeout
            ),
        )

    def as_dict(self) -> dict[str, float | int]:
        """Return only the settings that were actually stated."""

        return {
            name: value
            for name, value in (
                ("temperature", self.temperature),
                ("max_tokens", self.max_tokens),
                ("timeout", self.timeout),
                ("idle_timeout", self.idle_timeout),
            )
            if value is not None
        }

    @property
    def is_empty(self) -> bool:
        return not self.as_dict()


MODEL_SETTING_NAMES = ("temperature", "max_tokens", "timeout", "idle_timeout")


def model_setting_text(value: float | int) -> str:
    """Render one setting the way a person wrote it.

    A whole number stays whole: a temperature entered as 0 is stored as "0",
    not "0.0", so a configuration round-trips through TOML unchanged.
    """

    number = float(value)
    return str(int(number)) if number.is_integer() else str(number)


def model_settings_from_mapping(
    value: Mapping[str, object] | None,
    *,
    subject: str,
) -> ModelSettings:
    """Read settings from stored or command-line data, rejecting bad numbers."""

    if not value:
        return ModelSettings()
    return ModelSettings(
        temperature=_setting_number(
            value.get("temperature"), name="temperature",
            subject=subject, maximum=1.0,
        ),
        max_tokens=_setting_integer(value.get("max_tokens"), subject=subject),
        timeout=_setting_number(
            value.get("timeout"), name="timeout", subject=subject, positive=True
        ),
        idle_timeout=_setting_number(
            value.get("idle_timeout"), name="idle timeout", subject=subject
        ),
    )


def _setting_number(
    raw: object,
    *,
    name: str,
    subject: str,
    maximum: float | None = None,
    positive: bool = False,
) -> float | None:
    if raw is None or str(raw).strip() == "":
        return None
    try:
        number = float(str(raw).strip())
    except ValueError as exc:
        raise SystemExit(
            f"Model {name} for {subject} must be a number."
        ) from exc
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        raise SystemExit(
            f"Model {name} for {subject} must be a "
            f"{'positive' if positive else 'non-negative'} finite number."
        )
    if maximum is not None and number > maximum:
        raise SystemExit(
            f"Model {name} for {subject} must be between 0 and {maximum:g}."
        )
    return number


def _setting_integer(raw: object, *, subject: str) -> int | None:
    if raw is None or str(raw).strip() == "":
        return None
    try:
        number = int(str(raw).strip())
    except ValueError as exc:
        raise SystemExit(
            f"Model max tokens for {subject} must be a whole number."
        ) from exc
    if number <= 0:
        raise SystemExit(
            f"Model max tokens for {subject} must be greater than zero."
        )
    return number


@dataclass(frozen=True)
class ModelRouting:
    """Concrete runtime model specs resolved from named project settings."""

    default_spec: str
    overrides: dict[str, str]
    settings: dict[str, ModelSettings]


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


def fake_model_notice(routes: Mapping[str, str]) -> str | None:
    """Describe which participants are not talking to a real model.

    A fresh project answers with the mock so it runs before anybody owns an
    API key, which is right. Staying quiet about it is not: a mostly-fake run
    looks exactly like a real one, and its output is plausible either way.

    Returns the sentence to show, or None when every participant is real.
    """

    mocked = sorted(name for name, spec in routes.items() if spec == "mock")
    if not mocked:
        return None
    return (
        "No real model is in use: every participant answers with the mock."
        if len(mocked) == len(routes)
        else "Mock model (not a real one) for: " + ", ".join(mocked) + "."
    ) + " Assign one with 'zg model assign TARGET NAME'."


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
    settings: dict[str, ModelSettings] = {}
    for target in routes:
        participant = target.partition(".")[0]
        selected = target_configurations.get(
            target,
            target_configurations.get(participant, default_configuration),
        )
        resolved = model_settings_from_mapping(selected, subject=target)
        if not resolved.is_empty:
            settings[target] = resolved

    return ModelRouting(default_spec, named_overrides, settings)


def apply_model_overrides(
    routing: ModelRouting,
    *,
    default_spec: str | None = None,
    overrides: Mapping[str, str] | None = None,
    settings: Mapping[str, ModelSettings] | None = None,
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
        selected_settings: dict[str, ModelSettings] = {}
    else:
        selected_default = routing.default_spec
        selected_overrides = dict(routing.overrides)
        selected_settings = dict(routing.settings)

    for target, spec in normalize_llm_overrides(overrides).items():
        # Routing a target somewhere else discards settings chosen for the
        # model it used to use.
        selected_settings.pop(target, None)
        if spec.casefold() in {"inherit", "default"}:
            selected_overrides.pop(target, None)
        else:
            selected_overrides[target] = spec
    for target, value in (settings or {}).items():
        name = str(target)
        selected_settings[name] = selected_settings.get(
            name, ModelSettings()
        ).merged_with(value)
    return ModelRouting(
        selected_default,
        selected_overrides,
        selected_settings,
    )
