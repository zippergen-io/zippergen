"""Studio model provider, configuration, and assignment management."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit

from zippergen.dev import default_llm_spec
from zippergen.rendering import StatusKind
from zippergen.workspace import WorkspaceError

# This mixin intentionally depends on Studio's small rendering, selection, and
# context interface. Keeping the domain implementation here prevents the main
# command shell from becoming the model subsystem.
# pyright: reportAttributeAccessIssue=false, reportUnknownMemberType=false

_PROVIDER_ALIASES = {
    "claude": "anthropic",
    "ollama": "local",
}
_PROVIDER_SECRETS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}
_PROVIDER_DEFAULT_MODELS = {
    "local": ("OLLAMA_MODEL", "qwen2.5:7b"),
    "openai": ("OPENAI_MODEL", "gpt-4o-mini"),
    "anthropic": ("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
    "mistral": ("MISTRAL_MODEL", "mistral-small-latest"),
}
_SUPPORTED_PROVIDERS = ("mock", "local", "openai", "anthropic", "mistral")


@dataclass(frozen=True)
class _LocalProviderCheck:
    checked_at: str
    model_count: int
    model_ids: tuple[str, ...]


@dataclass(frozen=True)
class _ModelVerification:
    kind: StatusKind
    message: str


class _LocalProviderError(RuntimeError):
    """A local OpenAI-compatible endpoint could not be verified."""


def _canonical_provider(value: str) -> str:
    provider = value.partition(":")[0].strip().lower()
    return _PROVIDER_ALIASES.get(provider, provider)


def _validate_model_spec(value: str) -> str:
    spec = value.strip()
    provider, separator, model = spec.partition(":")
    canonical = _canonical_provider(provider)
    if not spec or canonical not in _SUPPORTED_PROVIDERS:
        raise SystemExit(
            "Model provider must be mock, local/ollama, openai, "
            "anthropic/claude, or mistral."
        )
    if separator and not model.strip():
        raise SystemExit(f"Model spec {value!r} is missing a model after ':'.")
    if canonical == "mock":
        if separator:
            raise SystemExit(
                "The built-in mock model is written simply as 'mock'."
            )
        return "mock"
    return f"{canonical}:{model.strip()}" if separator else canonical


class StudioModelsMixin:
    @staticmethod
    def _idle_seconds_summary(seconds: float) -> str:
        if seconds == 0:
            return "after every call"
        value = f"{seconds:g} s"
        if seconds >= 60 and seconds % 60 == 0:
            value += f" ({seconds / 60:g} min)"
        return f"after {value}"

    @classmethod
    def _model_idle_routes_summary(cls, values: object) -> str:
        if not isinstance(values, dict) or not values:
            return "none"
        return " · ".join(
            f"{target} {cls._idle_seconds_summary(float(seconds))}"
            for target, seconds in sorted(values.items())
        )

    @staticmethod
    def _configuration_idle_timeout(
        configuration: dict[str, str] | None,
    ) -> float | None:
        if not configuration or not configuration.get("idle_timeout"):
            return None
        return float(configuration["idle_timeout"])

    @classmethod
    def _configuration_idle_summary(
        cls,
        configuration: dict[str, str] | None,
    ) -> str:
        if not configuration:
            return "unknown"
        provider = _canonical_provider(
            configuration.get("provider") or configuration.get("spec") or ""
        )
        if provider != "local":
            return "not applicable" if provider == "mock" else "API managed"
        timeout = cls._configuration_idle_timeout(configuration)
        if timeout is None:
            return "never"
        return cls._idle_seconds_summary(timeout)

    def _model_idle_timeout_routes(
        self,
        current: str,
        workflow,
        module,
        *,
        default_override: str | None = None,
    ) -> dict[str, float]:
        """Resolve configuration-level idle release for active LLM routes."""

        active = self._llm_action_lifelines(workflow, module)
        if not active:
            return {}
        assignments = self.workspace.model_assignment_profile(
            current,
            default=default_llm_spec(module),
        )
        configurations = self.workspace.model_configurations()
        default_name = str(assignments["default"])
        participant_overrides = assignments.get("lifelines") or {}
        action_overrides = assignments.get("actions") or {}
        assert isinstance(participant_overrides, dict)
        assert isinstance(action_overrides, dict)

        routes: dict[str, float] = {}
        for participant, actions in active.items():
            configuration_name = participant_overrides.get(participant)
            if configuration_name is None and default_override is None:
                configuration_name = default_name
            timeout = self._configuration_idle_timeout(
                configurations.get(str(configuration_name))
                if configuration_name is not None
                else None
            )
            if timeout is not None:
                routes[participant] = timeout
            for action_name in actions:
                target = f"{participant}.{action_name}"
                action_configuration = action_overrides.get(target)
                if action_configuration is None:
                    continue
                action_timeout = self._configuration_idle_timeout(
                    configurations.get(str(action_configuration))
                )
                if action_timeout is not None:
                    routes[target] = action_timeout
        return routes

    def _run_model_profile(self) -> dict[str, object]:
        current = self.workspace.current_workflow
        if not current:
            return {"default": None, "lifelines": {}}
        _current, _workflow, module = self._current_context()
        profile = self.workspace.model_profile(
            current,
            default=default_llm_spec(module),
        )
        lifelines = dict(profile.get("lifelines") or {})
        lifelines.update(dict(profile.get("actions") or {}))
        return {"default": profile.get("default"), "lifelines": lifelines}

    def _check_workflow_models(
        self,
        current: str,
        workflow,
        module,
        *,
        default_override: str | None = None,
        for_run: bool = False,
    ) -> None:
        """Check exactly the configurations used by LLM-active participants."""

        active = self._llm_action_lifelines(workflow, module)
        title = "Run model checks" if for_run else "Assignment checks"
        if not active:
            self._emit_table(
                title,
                [("Status", "not needed; no LLM actions", "success")],
            )
            return

        assignments = self.workspace.model_assignment_profile(
            current,
            default=default_llm_spec(module),
        )
        configurations = self.workspace.model_configurations()
        default_name = str(assignments["default"])
        participant_overrides = assignments.get("lifelines") or {}
        action_overrides = assignments.get("actions") or {}
        assert isinstance(participant_overrides, dict)
        assert isinstance(action_overrides, dict)

        routes: list[tuple[str, str, str]] = []
        for participant, actions in active.items():
            for action_name in actions:
                target = f"{participant}.{action_name}"
                if target in action_overrides:
                    configuration_name = str(action_overrides[target])
                    configuration = configurations.get(configuration_name)
                    if configuration is None:
                        raise SystemExit(
                            f"Model configuration {configuration_name!r}, "
                            f"assigned to {target}, no longer exists."
                        )
                    spec = str(configuration["spec"])
                elif participant in participant_overrides:
                    configuration_name = str(
                        participant_overrides[participant]
                    )
                    configuration = configurations.get(configuration_name)
                    if configuration is None:
                        raise SystemExit(
                            f"Model configuration {configuration_name!r}, assigned "
                            f"to {participant}, no longer exists."
                        )
                    spec = str(configuration["spec"])
                elif default_override is not None:
                    configuration_name = "run override"
                    spec = default_override
                else:
                    configuration_name = default_name
                    configuration = configurations.get(configuration_name)
                    if configuration is None:
                        raise SystemExit(
                            f"Default model configuration {configuration_name!r} "
                            "no longer exists."
                        )
                    spec = str(configuration["spec"])
                routes.append((target, configuration_name, spec))

        checks: dict[tuple[str, str], _ModelVerification] = {}
        failures: list[str] = []
        details: list[str] = []
        checked_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        for _target, configuration_name, spec in routes:
            key = (configuration_name, spec)
            if key in checks:
                continue
            configuration = configurations.get(configuration_name)
            try:
                verification = self._verify_model_spec(
                    configuration_name,
                    spec,
                    for_save=False,
                )
            except SystemExit as exc:
                verification = _ModelVerification(
                    "error",
                    str(exc),
                )
            checks[key] = verification
            if configuration is not None and configuration_name != "mock":
                status = (
                    "available"
                    if verification.kind == "success"
                    else "unavailable"
                    if verification.kind == "error"
                    else "unverified"
                )
                self.workspace.save_model_configuration(
                    configuration_name,
                    {
                        **configuration,
                        "check_status": status,
                        "check_detail": verification.message[:240],
                        "checked_at": checked_at,
                    },
                )
            if verification.kind != "success":
                failures.append(configuration_name)
                details.append(verification.message)

        rows: list[tuple[object, ...]] = []
        for target, configuration_name, spec in routes:
            verification = checks[(configuration_name, spec)]
            kind: StatusKind = (
                "success" if verification.kind == "success" else "error"
            )
            status_text = {
                "success": "available",
                "error": "unavailable",
                "warning": "not verified",
            }[verification.kind]
            status = f"{self._status_mark(kind)} {status_text}"
            rows.append((target, configuration_name, spec, status))
        self._emit_columns(
            title,
            ("Participant.action", "Configuration", "Model", "Status"),
            rows,
        )
        if failures:
            for detail in dict.fromkeys(details):
                self._error(detail, indent=2)
            unique = ", ".join(dict.fromkeys(failures))
            next_step = (
                "check the run override or run without it"
                if failures[0] == "run override"
                else f"run 'model config check {failures[0]}'"
            )
            if not for_run:
                raise SystemExit(
                    f"Assignment check failed because {unique} could not be "
                    "verified. Restore the connection or configuration, then "
                    f"{next_step} or use 'model assignments check' again."
                )
            raise SystemExit(
                f"Run stopped before collecting inputs because {unique} "
                "could not be verified. Restore the connection or configuration, "
                f"then {next_step} or try 'run' again."
            )
        self._success(
            "All models used by this run are reachable."
            if for_run
            else "All assigned models are reachable."
        )
        self._emit()

    def _configuration_status_kind(
        self,
        configuration: dict[str, str],
    ) -> StatusKind:
        status = configuration.get("check_status", "not_checked")
        if status == "available":
            return "success"
        if status == "unavailable":
            return "error"
        return "warning"

    def _assignment_check_summary(
        self,
        configuration: dict[str, str] | None,
    ) -> str:
        """Render compact cached check state without implying a live request."""

        if configuration is None:
            return f"{self._status_mark('error')} missing"
        provider = configuration.get("provider") or _canonical_provider(
            configuration.get("spec", "")
        )
        if provider == "mock":
            return f"{self._status_mark('success')} built in"
        status = configuration.get("check_status", "not_checked")
        kind = self._configuration_status_kind(configuration)
        if status == "not_checked":
            return f"{self._status_mark(kind)} never"
        checked_at = str(configuration.get("checked_at") or "").strip()
        when = "unknown"
        if len(checked_at) >= 16 and checked_at[10] == "T":
            when = (
                checked_at[11:16]
                if checked_at[:10] == time.strftime("%Y-%m-%d")
                else checked_at[:10]
            )
        return f"{self._status_mark(kind)} {when}"

    def _emit_model_configurations(self, *, include_next: bool = True) -> None:
        configurations = self.workspace.model_configurations()
        rows: list[tuple[object, ...]] = []
        for name, configuration in configurations.items():
            status = configuration.get("check_status", "not_checked")
            provider = configuration.get("provider") or _canonical_provider(
                configuration["spec"]
            )
            model = configuration.get("model") or (
                "built in" if provider == "mock" else "default"
            )
            mark = self._status_mark(
                self._configuration_status_kind(configuration)
            )
            rows.append(
                (
                    name,
                    provider,
                    model,
                    self._configuration_idle_summary(configuration),
                    f"{mark} {status.replace('_', ' ')}",
                )
            )
        self._emit_columns(
            "Model configurations",
            ("Name", "Provider", "Model", "Idle release", "Status"),
            rows,
        )
        if include_next:
            self._emit_next(
                "model config create [NAME] · model config check [NAME]"
            )

    def _emit_model_assignments(
        self,
        *,
        workflow,
        module,
        assignments: dict[str, object],
    ) -> None:
        active = self._llm_action_lifelines(workflow, module)
        configurations = self.workspace.model_configurations()
        default = str(assignments["default"])
        participant_overrides = assignments.get("lifelines") or {}
        action_overrides = assignments.get("actions") or {}
        assert isinstance(participant_overrides, dict)
        assert isinstance(action_overrides, dict)
        if not active:
            self._emit_table(
                "Model assignments",
                [
                    ("Workflow", workflow.name, None),
                    ("Default", default, None),
                    (
                        "Status",
                        "no participants contain LLM actions",
                        "warning",
                    ),
                ],
            )
            return
        rows: list[tuple[object, ...]] = []
        for lifeline, actions in active.items():
            for action_name in actions:
                target = f"{lifeline}.{action_name}"
                action_explicit = action_overrides.get(target)
                participant_explicit = participant_overrides.get(lifeline)
                effective = str(
                    action_explicit or participant_explicit or default
                )
                spec = configurations.get(effective, {}).get("spec", "missing")
                if action_explicit:
                    source = "action override"
                elif participant_explicit:
                    source = "participant"
                else:
                    source = "default"
                rows.append(
                    (
                        lifeline,
                        action_name,
                        effective,
                        spec,
                        source,
                        self._assignment_check_summary(
                            configurations.get(effective)
                        ),
                    )
                )
        self._emit_columns(
            "Model assignments",
            (
                "Participant",
                "LLM action",
                "Configuration",
                "Model",
                "Source",
                "Last check",
            ),
            rows,
        )
        self._emit_section_title("Execution")
        self._info(
            "Configurations can be shared; calls remain independent and may "
            "run in parallel.",
            indent=2,
        )
        self._emit()
        self._emit_next("model assignments check")

    def _model_configuration_name(self, requested: str) -> str:
        configurations = self.workspace.model_configurations()
        canonical = {
            name.casefold(): name for name in configurations
        }.get(requested.casefold())
        if canonical is None:
            available = ", ".join(configurations) or "none"
            raise SystemExit(
                f"Unknown model configuration {requested!r}. Available: "
                f"{available}. Use 'model config create' to create one."
            )
        return canonical

    def _configure_model_configuration(
        self,
        args: list[str],
        *,
        edit_only: bool = False,
        provider_override: str | None = None,
    ) -> str:
        if len(args) > 1:
            command = "edit NAME" if edit_only else "create [NAME]"
            raise SystemExit(f"Use model config {command}.")
        configurations = self.workspace.model_configurations()
        requested = args[0] if args else None
        existing_name = None
        if requested:
            existing_name = {
                name.casefold(): name for name in configurations
            }.get(requested.casefold())
        if edit_only and existing_name is None:
            raise SystemExit(
                f"Unknown model configuration {requested!r}. "
                "Use 'model config list' to see available names."
            )
        if existing_name == "mock":
            raise SystemExit("The built-in mock configuration cannot be edited.")
        if existing_name is not None and not edit_only:
            raise SystemExit(
                f"Model configuration {existing_name!r} already exists. "
                f"Use 'model config edit {existing_name}'."
            )

        existing = configurations.get(existing_name or "", {})
        connected = [
            provider
            for provider in _SUPPORTED_PROVIDERS
            if provider != "mock" and self._provider_is_connected(provider)
        ]
        default_provider = (
            existing.get("provider")
            or provider_override
            or (connected[0] if connected else "mock")
        )
        if provider_override:
            provider = _canonical_provider(provider_override)
        else:
            entered_provider = self.input(
                f"Provider [{default_provider}]: "
            ).strip()
            provider = _canonical_provider(entered_provider or default_provider)
        if provider not in _SUPPORTED_PROVIDERS:
            raise SystemExit(
                "Provider must be mock, local/ollama, openai, "
                "anthropic/claude, or mistral."
            )
        if provider != "mock" and not self._provider_is_connected(provider):
            raise SystemExit(
                f"Provider {provider!r} is not configured. "
                f"Next: model provider configure {provider}"
            )
        if provider == "mock":
            model = ""
            spec = "mock"
            idle_timeout = None
        else:
            _environment_name, fallback_model = _PROVIDER_DEFAULT_MODELS[provider]
            default_model = (
                existing.get("model")
                if existing.get("provider") == provider
                else None
            ) or fallback_model
            model = self.input(
                f"Model identifier [{default_model}]: "
            ).strip() or default_model
            spec = _validate_model_spec(f"{provider}:{model}")
            idle_timeout = None
            if provider == "local":
                if (
                    existing_name
                    and existing.get("provider") == "local"
                ):
                    default_idle = existing.get("idle_timeout") or "never"
                else:
                    default_idle = "300"
                entered_idle = self.input(
                    "Release local model after idle seconds "
                    f"[{default_idle}, or type 'never']: "
                ).strip()
                selected_idle = entered_idle or default_idle
                if selected_idle.casefold() not in {
                    "never",
                    "none",
                    "off",
                    "disabled",
                }:
                    try:
                        parsed_idle = float(selected_idle)
                    except ValueError as exc:
                        raise SystemExit(
                            "Idle release must be a non-negative number of "
                            "seconds or 'never'."
                        ) from exc
                    if not math.isfinite(parsed_idle) or parsed_idle < 0:
                        raise SystemExit(
                            "Idle release must be a non-negative number of "
                            "seconds or 'never'."
                        )
                    idle_timeout = (
                        str(int(parsed_idle))
                        if parsed_idle.is_integer()
                        else str(parsed_idle)
                    )

        if existing_name:
            name = existing_name
        elif requested:
            name = requested
        else:
            name = self.workspace.automatic_model_configuration_name(spec)
            if name in configurations:
                self._success(
                    f"Model configuration already exists: {name} ({spec})"
                )
                self._emit_next(
                    f"model config check {name} · "
                    f"model assign PARTICIPANT {name}"
                )
                return name
        try:
            self.workspace.save_model_configuration(
                name,
                {
                    "provider": provider,
                    "model": model,
                    "spec": spec,
                    "check_status": (
                        "available" if provider == "mock" else "not_checked"
                    ),
                    "check_detail": (
                        "built in"
                        if provider == "mock"
                        else "run 'model config check' before assignment"
                    ),
                    **(
                        {"idle_timeout": idle_timeout}
                        if idle_timeout is not None
                        else {}
                    ),
                },
            )
        except WorkspaceError as exc:
            raise SystemExit(str(exc)) from exc
        verb = "Updated" if existing_name else "Created"
        self._success(f"{verb} model configuration: {name} ({spec})")
        self._emit_next(
            f"model config check {name} · "
            f"model assign PARTICIPANT {name}"
        )
        return name

    def _check_model_configurations(self, target: str) -> None:
        configurations = self.workspace.model_configurations()
        if target.casefold() == "all":
            selected = list(configurations)
        else:
            selected = [self._model_configuration_name(target)]
        self._emit_section_title("Configuration checks")
        failures: list[str] = []
        for name in selected:
            configuration = configurations[name]
            if name == "mock":
                self._success("mock: built in and available.", indent=2)
                continue
            try:
                verification = self._verify_model_spec(
                    name,
                    configuration["spec"],
                    for_save=False,
                )
            except SystemExit as exc:
                detail = str(exc)
                self.workspace.save_model_configuration(
                    name,
                    {
                        **configuration,
                        "check_status": "unavailable",
                        "check_detail": detail[:240],
                        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    },
                )
                self._error(f"{name}: {detail}", indent=2)
                failures.append(name)
                continue
            check_status = (
                "available"
                if verification.kind == "success"
                else "unverified"
            )
            self.workspace.save_model_configuration(
                name,
                {
                    **configuration,
                    "check_status": check_status,
                    "check_detail": verification.message,
                    "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                },
            )
            self._status(
                verification.kind,
                verification.message,
                indent=2,
            )
        if failures:
            raise SystemExit(
                "Model configuration check failed for "
                + ", ".join(failures)
                + ". Assignments were not changed."
            )
        self._emit()
        self._success(
            "Model configuration checks complete; assignments unchanged."
        )

    def _show_model_configuration(self, requested: str) -> None:
        name = self._model_configuration_name(requested)
        configuration = self.workspace.model_configurations()[name]
        usage = self.workspace.model_configuration_usage(name)
        self._emit_table(
            "Model configuration",
            [
                ("Name", name, None),
                ("Provider", configuration.get("provider") or "unknown", None),
                (
                    "Model",
                    configuration.get("model")
                    or ("built in" if name == "mock" else "default"),
                    None,
                ),
                ("Effective", configuration["spec"], None),
                (
                    "Idle release",
                    self._configuration_idle_summary(configuration),
                    None,
                ),
                (
                    "Check",
                    configuration.get("check_status", "not checked"),
                    self._configuration_status_kind(configuration),
                ),
                (
                    "Check detail",
                    configuration.get("check_detail", "not checked"),
                    None,
                ),
                (
                    "Assigned in",
                    ", ".join(usage) if usage else "no workflows",
                    None,
                ),
                (
                    "Sharing",
                    "may be assigned to several participants; calls stay independent",
                    "success",
                ),
                ("Next", f"model config check {name}", None),
            ],
        )

    def _rename_model_configuration(self, old: str, new: str) -> None:
        old_name = self._model_configuration_name(old)
        try:
            result = self.workspace.rename_model_configuration(old_name, new)
        except WorkspaceError as exc:
            raise SystemExit(str(exc)) from exc
        new_name = str(result["new_name"])
        configuration = result.get("configuration")
        spec = (
            str(configuration.get("spec") or "unknown")
            if isinstance(configuration, dict)
            else "unknown"
        )
        rows: list[tuple[str, object, StatusKind | None]] = [
            ("From", old_name, None),
            ("To", new_name, "success"),
            ("Model", spec, None),
            (
                "Check",
                (
                    str(configuration.get("check_status") or "not checked")
                    if isinstance(configuration, dict)
                    else "not checked"
                )
                + "; preserved",
                None,
            ),
        ]
        references = result.get("references")
        reference_count = 0
        if isinstance(references, (tuple, list)):
            for reference in references:
                if not isinstance(reference, dict):
                    continue
                reference_count += 1
                locations: list[str] = []
                if reference.get("default"):
                    locations.append("default")
                lifelines = reference.get("lifelines")
                if isinstance(lifelines, (tuple, list)):
                    locations.extend(str(value) for value in lifelines)
                rows.append(
                    (
                        f"Reference {reference_count}",
                        f"{reference.get('workflow')} — "
                        + ", ".join(locations),
                        "success",
                    )
                )
        if reference_count == 0:
            rows.append(("References", "none assigned", None))
        rows.append(("Next", "model", None))
        self._emit_table("Model configuration renamed", rows)

    def _show_model_assignments(self) -> None:
        if not self.workspace.current_workflow:
            self._emit_table(
                "Workflow assignments",
                [
                    (
                        "Status",
                        "no workflow selected",
                        "warning",
                    ),
                    ("Next", "workflow list · workflow select", None),
                ],
            )
            return
        current, workflow, module = self._current_context()
        assignments = self.workspace.model_assignment_profile(
            current,
            default=default_llm_spec(module),
        )
        self._emit_model_assignments(
            workflow=workflow,
            module=module,
            assignments=assignments,
        )

    def _show_models_dashboard(self) -> None:
        self._emit_section_title("Models", major=True)
        self._emit()
        self._emit_model_connections(include_next=False)
        self._emit_model_configurations(include_next=False)
        self._show_model_assignments()
        self._emit_table(
            "Model workflow",
            [
                (
                    "Lifecycle",
                    "provider configure/check → config create/check → assign",
                    None,
                ),
                ("Next", "model setup", None),
            ],
        )

    def _emit_guided_model_step(
        self,
        step: int,
        action: str,
        *,
        context: tuple[str, object, StatusKind | None] | None = None,
    ) -> None:
        actions = (
            "configure and check a provider",
            "create and check a named configuration",
            "assign configurations to LLM participants",
        )
        rows: list[tuple[str, object, StatusKind | None]] = [
            ("Current", f"Step {step} of 3 — {action}", "info"),
        ]
        if context is not None:
            rows.append(context)
        if step < len(actions):
            rows.append(
                (
                    "Next",
                    f"Step {step + 1} of 3 — {actions[step]}",
                    None,
                )
            )
        if step == 1:
            rows.append(("Then", f"Step 3 of 3 — {actions[2]}", None))
        self._emit_table("Guided model setup", rows)

    def _guided_models_setup(self) -> None:
        self._emit_guided_model_step(
            1,
            "configure and check a provider",
        )
        provider = str(
            self._select(
                "Available providers",
                list(_SUPPORTED_PROVIDERS),
                prompt="Select provider",
            )
        )
        was_connected = self._provider_is_connected(provider)
        if provider != "mock" and not was_connected:
            self._connect_model_provider([provider])
        if provider != "mock" and (provider != "local" or was_connected):
            self._check_model_providers(provider)
        self._emit_guided_model_step(
            2,
            "create and check a named configuration",
            context=("Provider", provider, "success"),
        )
        entered_name = self.input(
            "Configuration name (press Enter for an automatic name): "
        ).strip()
        configuration_name = self._configure_model_configuration(
            [entered_name] if entered_name else [],
            provider_override=provider,
        )
        self._check_model_configurations(configuration_name)
        self._emit_guided_model_step(
            3,
            "assign configurations to LLM participants",
            context=("Configuration", configuration_name, "success"),
        )

        if not self.workspace.current_workflow:
            self._emit_table(
                "Assignment paused",
                [
                    ("Configuration", configuration_name, "success"),
                    ("Status", "no workflow selected", "warning"),
                    (
                        "Next",
                        "workflow list · workflow select · model setup",
                        None,
                    ),
                ],
            )
            return
        current, workflow, module = self._current_context()
        active = self._llm_action_lifelines(workflow, module)
        assignments = self.workspace.model_assignment_profile(
            current,
            default=default_llm_spec(module),
        )
        default = str(assignments["default"])
        overrides = dict(assignments.get("lifelines") or {})
        action_overrides = dict(assignments.get("actions") or {})
        choices = list(self.workspace.model_configurations())
        for lifeline in active:
            selected = self._select(
                f"Available configurations for {lifeline}",
                choices,
                prompt=f"Select configuration for {lifeline}",
            )
            overrides[lifeline] = str(selected)
        saved = self.workspace.save_model_assignment_profile(
            current,
            default=default,
            lifelines=overrides,
            actions=action_overrides,
        )
        self._success(
            f"Model setup complete for {workflow.name}; "
            "configurations remain reusable and calls remain independent."
        )
        self._emit()
        self._emit_model_assignments(
            workflow=workflow,
            module=module,
            assignments=saved,
        )

    def configure_models(self, args: list[str]) -> None:
        action = args[0].lower() if args else "show"

        if (action == "show" and len(args) == 1) or not args:
            self._show_models_dashboard()
            return

        if action == "setup" and len(args) == 1:
            self._guided_models_setup()
            return

        if action == "provider":
            subaction = args[1].lower() if len(args) > 1 else "list"
            rest = args[2:]
            if subaction in {"list", "show"} and not rest:
                self._emit_model_connections()
                return
            if subaction == "configure" and len(rest) in {1, 2}:
                self._connect_model_provider(rest)
                if _canonical_provider(rest[0]) not in {"mock", "local"}:
                    self._check_model_providers(rest[0], strict=False)
                self._emit()
                self._emit_model_connections()
                return
            if subaction == "check" and len(rest) <= 1:
                self._check_model_providers(rest[0] if rest else "all")
                return
            if subaction == "remove" and len(rest) == 1:
                self._disconnect_model_provider(rest[0])
                self._emit()
                self._emit_model_connections()
                return
            raise SystemExit(
                "Use model provider list, model provider configure NAME "
                "[URL], model provider check [NAME|all], or "
                "model provider remove NAME."
            )

        if action == "config":
            subaction = args[1].lower() if len(args) > 1 else "list"
            rest = args[2:]
            if subaction in {"list"} and not rest:
                self._emit_model_configurations()
                return
            if subaction == "show" and len(rest) <= 1:
                if not rest or rest[0].casefold() == "all":
                    self._emit_model_configurations()
                else:
                    self._show_model_configuration(rest[0])
                return
            if subaction == "create" and len(rest) <= 1:
                self._configure_model_configuration(rest)
                return
            if subaction == "edit" and len(rest) == 1:
                self._configure_model_configuration(rest, edit_only=True)
                return
            if subaction == "check" and len(rest) <= 1:
                self._check_model_configurations(rest[0] if rest else "all")
                return
            if subaction == "rename" and len(rest) == 2:
                self._rename_model_configuration(rest[0], rest[1])
                return
            if subaction == "remove" and len(rest) == 1:
                name = self._model_configuration_name(rest[0])
                try:
                    self.workspace.remove_model_configuration(name)
                except WorkspaceError as exc:
                    raise SystemExit(str(exc)) from exc
                self._success(f"Removed model configuration: {name}")
                return
            raise SystemExit(
                "Use model config list, model config create [NAME], "
                "model config show [NAME], model config check [NAME|all], "
                "model config edit NAME, model config rename OLD NEW, or "
                "model config remove NAME."
            )

        if action == "assignments":
            if len(args) == 1:
                self._show_model_assignments()
                return
            if len(args) == 2 and args[1].casefold() == "check":
                current, workflow, module = self._current_context()
                self._check_workflow_models(current, workflow, module)
                return
            raise SystemExit(
                "Use model assignments or model assignments check."
            )

        if action not in {"assign", "default", "inherit"}:
            raise SystemExit(
                "Use model, model setup, model provider ..., "
                "model config ..., model assignments, "
                "model assign PARTICIPANT_OR_ACTION NAME, model default NAME, "
                "or model inherit PARTICIPANT_OR_ACTION."
            )

        current, workflow, module = self._current_context()
        assignments = self.workspace.model_assignment_profile(
            current,
            default=default_llm_spec(module),
        )
        default = str(assignments["default"])
        overrides = dict(assignments.get("lifelines") or {})
        action_overrides = dict(assignments.get("actions") or {})
        active = self._llm_action_lifelines(workflow, module)
        action_targets = self._llm_action_targets(workflow, module)
        changed_configuration: str | None = None
        result_message: str

        if action == "default" and len(args) == 2:
            default = self._model_configuration_name(args[1])
            changed_configuration = default
            result_message = f"Set the default configuration to {default}."
        elif action == "assign" and len(args) == 3:
            entered_target, entered_configuration = args[1:]
            lifeline = {
                name.casefold(): name for name in active
            }.get(entered_target.casefold())
            action_target = {
                name.casefold(): name for name in action_targets
            }.get(entered_target.casefold())
            if lifeline is None and action_target is None:
                available = ", ".join(
                    [*active, *action_targets]
                ) or "none"
                raise SystemExit(
                    f"{entered_target!r} is not an LLM participant or action. "
                    f"Available targets: "
                    f"{available}."
                )
            configuration = self._model_configuration_name(
                entered_configuration
            )
            target = action_target or lifeline
            assert target is not None
            if action_target is not None:
                action_overrides[action_target] = configuration
            else:
                overrides[lifeline] = configuration  # type: ignore[index]
            changed_configuration = configuration
            result_message = f"Assigned {configuration} to {target}."
        elif action == "inherit" and len(args) == 2:
            entered_target = args[1]
            lifeline = {
                name.casefold(): name for name in active
            }.get(entered_target.casefold())
            action_target = {
                name.casefold(): name for name in action_targets
            }.get(entered_target.casefold())
            if lifeline is None and action_target is None:
                available = ", ".join(
                    [*active, *action_targets]
                ) or "none"
                raise SystemExit(
                    f"{entered_target!r} is not an LLM participant or action. "
                    f"Available targets: "
                    f"{available}."
                )
            if action_target is not None:
                action_overrides.pop(action_target, None)
                participant = action_targets[action_target][0]
                inherited = overrides.get(participant, default)
                result_message = (
                    f"{action_target} now inherits {inherited} from "
                    f"{'its participant' if participant in overrides else 'the default'}."
                )
            else:
                assert lifeline is not None
                overrides.pop(lifeline, None)
                result_message = (
                    f"{lifeline} now inherits the default configuration {default}."
                )
        else:
            raise SystemExit(
                "Use model assign PARTICIPANT_OR_ACTION NAME, model default NAME, or "
                "model inherit PARTICIPANT_OR_ACTION."
            )

        if changed_configuration is not None:
            configuration = self.workspace.model_configurations()[
                changed_configuration
            ]
            status = configuration.get("check_status")
            if status == "unavailable":
                raise SystemExit(
                    f"{changed_configuration} is unavailable. Run "
                    f"'model config check {changed_configuration}' again, "
                    "or edit the configuration before assigning it."
                )
            if status != "available":
                self._warning(
                    f"{changed_configuration} is "
                    f"{status or 'not checked'}; "
                    f"use 'model config check {changed_configuration}'."
                )
        saved = self.workspace.save_model_assignment_profile(
            current,
            default=default,
            lifelines=overrides,
            actions=action_overrides,
        )
        self._success(result_message)
        self._emit()
        self._emit_model_assignments(
            workflow=workflow,
            module=module,
            assignments=saved,
        )

    def _resolved_model(self, spec: str) -> tuple[str, str | None]:
        provider = _canonical_provider(spec)
        _prefix, separator, entered_model = spec.partition(":")
        if provider == "mock":
            return provider, None
        if separator:
            return provider, entered_model.strip()
        environment_name, fallback = _PROVIDER_DEFAULT_MODELS[provider]
        return provider, os.environ.get(environment_name, fallback)

    def _provider_api_key(self, provider: str) -> str | None:
        secret_name = _PROVIDER_SECRETS.get(provider)
        if secret_name is None:
            return None
        return os.environ.get(secret_name) or self.workspace.load_secrets().get(
            secret_name
        )

    def _api_model_request(
        self,
        provider: str,
        model: str,
        api_key: str,
    ) -> request.Request:
        encoded_model = quote(model, safe="")
        if provider == "openai":
            base_url = os.environ.get(
                "OPENAI_BASE_URL",
                "https://api.openai.com/v1",
            )
            models_url = self._local_models_url(base_url)
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "ZipperGen-Studio/0.1",
            }
        elif provider == "anthropic":
            models_url = "https://api.anthropic.com/v1/models"
            headers = {
                "Accept": "application/json",
                "anthropic-version": "2023-06-01",
                "x-api-key": api_key,
                "User-Agent": "ZipperGen-Studio/0.1",
            }
        else:
            assert provider == "mistral"
            models_url = "https://api.mistral.ai/v1/models"
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "ZipperGen-Studio/0.1",
            }
        return request.Request(
            f"{models_url}/{encoded_model}",
            headers=headers,
            method="GET",
        )

    def _remote_model_available(
        self,
        provider: str,
        model: str,
        api_key: str,
    ) -> tuple[bool | None, str | None]:
        req = self._api_model_request(provider, model, api_key)
        try:
            with request.urlopen(req, timeout=3.0) as response:
                raw = response.read(1_048_577)
        except HTTPError as exc:
            raw_detail = exc.read(512).decode("utf-8", errors="replace")
            detail = self._local_check_error(raw_detail) if raw_detail.strip() else ""
            suffix = f": {detail}" if detail else ""
            if exc.code in {400, 404, 422}:
                return False, f"HTTP {exc.code}{suffix}"
            if exc.code in {408, 429} or exc.code >= 500:
                return None, f"HTTP {exc.code}{suffix}"
            raise SystemExit(
                f"Could not verify model {model!r} with {provider}: "
                f"HTTP {exc.code}{suffix}. The configuration was not changed."
            ) from exc
        except URLError as exc:
            return None, self._local_check_error(exc.reason)
        except (TimeoutError, OSError) as exc:
            return None, self._local_check_error(exc)
        if len(raw) > 1_048_576:
            return None, "the model response exceeded the 1 MiB safety limit"
        return True, None

    def _verify_model_spec(
        self,
        label: str,
        spec: str,
        *,
        for_save: bool = True,
    ) -> _ModelVerification:
        provider, model = self._resolved_model(spec)
        if provider == "mock":
            return _ModelVerification(
                "success",
                f"{label}: mock is built in and available.",
            )
        assert model is not None
        resolved = f"{spec} (resolved model: {model})" if ":" not in spec else spec
        if provider == "local":
            profile = self.workspace.provider_profiles().get("local", {})
            base_url = profile.get(
                "base_url",
                os.environ.get(
                    "OLLAMA_BASE_URL",
                    "http://127.0.0.1:11434/v1",
                ),
            )
            try:
                result = self._check_local_provider(base_url)
            except _LocalProviderError as exc:
                message = (
                    f"saved {resolved}, but availability could not be checked"
                    if for_save
                    else f"{resolved} could not be checked"
                )
                return _ModelVerification(
                    "warning",
                    f"{label}: {message} at {base_url}: {exc}.",
                )
            if model not in result.model_ids:
                available = ", ".join(result.model_ids[:8]) or "none"
                raise SystemExit(
                    f"Model {model!r} is not available from the local provider "
                    f"at {base_url}. Available models: {available}. "
                    "The configuration was not changed."
                )
            return _ModelVerification(
                "success",
                f"{label}: {resolved} is available from the local provider.",
            )

        api_key = self._provider_api_key(provider)
        if not api_key:
            message = (
                f"saved {resolved}, but it was not checked"
                if for_save
                else f"{resolved} could not be checked"
            )
            return _ModelVerification(
                "warning",
                f"{label}: {message} because "
                f"{provider} is not configured. Use "
                f"'model provider configure {provider}'.",
            )
        available, detail = self._remote_model_available(provider, model, api_key)
        if available is False:
            suffix = f" ({detail})" if detail else ""
            raise SystemExit(
                f"Model {model!r} is not available with the configured "
                f"{provider} API key{suffix}. The configuration was not changed."
            )
        if available is None:
            suffix = f": {detail}" if detail else ""
            message = (
                f"saved {resolved}, but {provider} availability could not be checked"
                if for_save
                else f"{resolved} could not be checked with {provider}"
            )
            return _ModelVerification(
                "warning",
                f"{label}: {message}{suffix}.",
            )
        return _ModelVerification(
            "success",
            f"{label}: {resolved} is available with the configured "
            f"{provider} API key.",
        )

    def _api_models_collection_request(
        self,
        provider: str,
        api_key: str,
    ) -> request.Request:
        if provider == "openai":
            base_url = os.environ.get(
                "OPENAI_BASE_URL",
                "https://api.openai.com/v1",
            )
            models_url = self._local_models_url(base_url)
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "ZipperGen-Studio/0.1",
            }
        elif provider == "anthropic":
            models_url = "https://api.anthropic.com/v1/models"
            headers = {
                "Accept": "application/json",
                "anthropic-version": "2023-06-01",
                "x-api-key": api_key,
                "User-Agent": "ZipperGen-Studio/0.1",
            }
        else:
            assert provider == "mistral"
            models_url = "https://api.mistral.ai/v1/models"
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "ZipperGen-Studio/0.1",
            }
        return request.Request(models_url, headers=headers, method="GET")

    def _check_api_provider(
        self,
        provider: str,
        api_key: str,
    ) -> _LocalProviderCheck:
        req = self._api_models_collection_request(provider, api_key)
        try:
            with request.urlopen(req, timeout=3.0) as response:
                raw = response.read(1_048_577)
        except HTTPError as exc:
            raw_detail = exc.read(512).decode("utf-8", errors="replace")
            detail = (
                self._local_check_error(raw_detail)
                if raw_detail.strip()
                else ""
            )
            suffix = f": {detail}" if detail else ""
            raise _LocalProviderError(f"HTTP {exc.code}{suffix}") from exc
        except URLError as exc:
            raise _LocalProviderError(
                self._local_check_error(exc.reason)
            ) from exc
        except (TimeoutError, OSError) as exc:
            raise _LocalProviderError(self._local_check_error(exc)) from exc
        if len(raw) > 1_048_576:
            raise _LocalProviderError(
                "the model-list response exceeded the 1 MiB safety limit"
            )
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _LocalProviderError(
                "the model-list endpoint did not return valid UTF-8 JSON"
            ) from exc
        models = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            raise _LocalProviderError(
                "the provider response did not contain a JSON 'data' list"
            )
        return _LocalProviderCheck(
            checked_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            model_count=len(models),
            model_ids=tuple(
                str(item["id"])
                for item in models
                if isinstance(item, dict)
                and isinstance(item.get("id"), str)
            ),
        )

    def _check_model_providers(
        self,
        target: str,
        *,
        strict: bool = True,
    ) -> None:
        canonical_target = _canonical_provider(target)
        if target.casefold() == "all":
            selected = list(_SUPPORTED_PROVIDERS)
        elif canonical_target in _SUPPORTED_PROVIDERS:
            selected = [canonical_target]
        else:
            raise SystemExit(
                "Provider must be mock, local/ollama, openai, anthropic/claude, "
                "or mistral."
            )
        self._emit_section_title("Provider checks")
        failures: list[str] = []
        for provider in selected:
            if provider == "mock":
                self._success("mock: built in and available.", indent=2)
                continue
            if not self._provider_is_connected(provider):
                message = (
                    f"{provider}: not configured; use "
                    f"'model provider configure {provider}'"
                )
                if len(selected) == 1:
                    self._error(message, indent=2)
                    failures.append(provider)
                else:
                    self._warning(message, indent=2)
                continue
            checked_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            try:
                if provider == "local":
                    profile = self.workspace.provider_profiles().get("local", {})
                    base_url = profile.get(
                        "base_url",
                        "http://127.0.0.1:11434/v1",
                    )
                    result = self._check_local_provider(base_url)
                    self._save_local_provider_check(base_url, result)
                else:
                    api_key = self._provider_api_key(provider)
                    assert api_key is not None
                    result = self._check_api_provider(provider, api_key)
                    profile = self.workspace.provider_profiles().get(provider, {})
                    self.workspace.save_provider_profile(
                        provider,
                        {
                            **profile,
                            "kind": "api",
                            "key_env": _PROVIDER_SECRETS[provider],
                            "check_status": "reachable",
                            "checked_at": result.checked_at,
                            "model_count": str(result.model_count),
                        },
                    )
            except _LocalProviderError as exc:
                profile = self.workspace.provider_profiles().get(provider, {})
                self.workspace.save_provider_profile(
                    provider,
                    {
                        **profile,
                        "check_status": "unreachable",
                        "checked_at": checked_at,
                        "check_error": str(exc)[:240],
                    },
                )
                self._error(f"{provider}: {exc}", indent=2)
                failures.append(provider)
                continue
            noun = "model" if result.model_count == 1 else "models"
            self._success(
                f"{provider}: connection accepted; "
                f"{result.model_count} {noun} reported.",
                indent=2,
            )
        if failures and strict:
            raise SystemExit(
                "Provider check failed for "
                + ", ".join(failures)
                + ". Saved credentials and endpoints were preserved."
            )
        self._emit()
        if failures:
            self._warning(
                "Provider configuration was saved, but its check did not pass."
            )
        else:
            self._success("Provider checks complete.")

    def _provider_configuration_status(
        self,
        provider: str,
    ) -> tuple[StatusKind, str]:
        canonical = _canonical_provider(provider)
        if canonical == "mock":
            return "success", "available; built in"
        profiles = self.workspace.provider_profiles()
        if canonical == "local":
            profile = profiles.get("local", {})
            base_url = profile.get(
                "base_url",
                os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"),
            )
            check_status = profile.get("check_status")
            checked_at = profile.get("checked_at")
            if check_status == "reachable" and checked_at:
                count = profile.get("model_count", "0")
                noun = "model" if count == "1" else "models"
                kind: StatusKind = "success" if count != "0" else "warning"
                state = (
                    "last check succeeded"
                    if count != "0"
                    else "last check reached the endpoint but found no models"
                )
                return (
                    kind,
                    f"{state}; endpoint {base_url}; {count} {noun}; "
                    f"checked {checked_at}",
                )
            if check_status == "unreachable" and checked_at:
                detail = profile.get("check_error", "connection failed")
                return (
                    "error",
                    f"last check failed; endpoint {base_url}; "
                    f"checked {checked_at}: {detail}",
                )
            return (
                "warning",
                "not configured; use 'model provider configure local'",
            )
        secret_name = _PROVIDER_SECRETS.get(canonical)
        if secret_name is None:
            return "error", "unsupported"
        profile = profiles.get(canonical, {})
        check_status = profile.get("check_status")
        checked_at = profile.get("checked_at")
        if check_status == "reachable" and checked_at:
            count = profile.get("model_count", "0")
            noun = "model" if count == "1" else "models"
            return (
                "success",
                f"last check succeeded; {count} {noun}; checked {checked_at}",
            )
        if check_status == "unreachable" and checked_at:
            detail = profile.get("check_error", "connection failed")
            return (
                "error",
                f"last check failed; checked {checked_at}: {detail}",
            )
        if os.environ.get(secret_name):
            return (
                "warning",
                f"configured; {secret_name} is in the environment; not tested",
            )
        if self.workspace.load_secrets().get(secret_name):
            return (
                "warning",
                f"configured; {secret_name} is in private Studio storage; "
                "not tested",
            )
        return (
            "warning",
            f"not configured; use 'model provider configure {canonical}'",
        )

    def _provider_status(self, provider: str) -> str:
        return self._provider_configuration_status(provider)[1]

    def _emit_model_connections(self, *, include_next: bool = True) -> None:
        rows: list[tuple[object, ...]] = []
        for provider in _SUPPORTED_PROVIDERS:
            kind, status = self._provider_configuration_status(provider)
            if provider == "mock":
                state = "built in"
            elif "not configured" in status:
                state = "not configured"
            elif "not tested" in status:
                state = "configured"
            elif kind == "success":
                state = "checked"
            elif kind == "error":
                state = "check failed"
            else:
                state = "unverified"
            rows.append(
                (
                    provider,
                    state,
                    f"{self._status_mark(kind)} {status}",
                )
            )
        self._emit_columns(
            "Provider connections",
            ("Provider", "State", "Details"),
            rows,
        )
        self._emit_section_title("Secrets")
        self._emit("API-key values are never displayed or written to the project.")
        self._emit()
        if include_next:
            self._emit_next(
                "model provider configure NAME · model provider check [NAME]"
            )

    def _local_models_url(self, base_url: str) -> str:
        parsed = urlsplit(base_url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise SystemExit(
                "A local provider URL must be a complete http:// or https:// URL."
            )
        if parsed.username or parsed.password:
            raise SystemExit(
                "Do not embed credentials in a local provider URL."
            )
        path = parsed.path.rstrip("/")
        return urlunsplit(
            (parsed.scheme, parsed.netloc, f"{path}/models", "", "")
        )

    def _local_check_error(self, value: object) -> str:
        printable = "".join(
            character if character.isprintable() else " "
            for character in str(value)
        )
        return " ".join(printable.split())[:240] or "connection failed"

    def _check_local_provider(self, base_url: str) -> _LocalProviderCheck:
        models_url = self._local_models_url(base_url)
        req = request.Request(
            models_url,
            headers={
                "Accept": "application/json",
                "User-Agent": "ZipperGen-Studio/0.1",
            },
            method="GET",
        )
        try:
            with request.urlopen(req, timeout=3.0) as response:
                raw = response.read(1_048_577)
        except HTTPError as exc:
            raw_detail = exc.read(512).decode("utf-8", errors="replace")
            detail = self._local_check_error(raw_detail) if raw_detail.strip() else ""
            suffix = f": {detail}" if detail else ""
            raise _LocalProviderError(f"HTTP {exc.code}{suffix}") from exc
        except URLError as exc:
            raise _LocalProviderError(
                self._local_check_error(exc.reason)
            ) from exc
        except (TimeoutError, OSError) as exc:
            raise _LocalProviderError(self._local_check_error(exc)) from exc
        if len(raw) > 1_048_576:
            raise _LocalProviderError(
                "the /models response exceeded the 1 MiB safety limit"
            )
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _LocalProviderError(
                "the /models endpoint did not return valid UTF-8 JSON"
            ) from exc
        models = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            raise _LocalProviderError(
                "the /models response is not OpenAI-compatible "
                "(expected a JSON 'data' list)"
            )
        return _LocalProviderCheck(
            checked_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            model_count=len(models),
            model_ids=tuple(
                str(item["id"])
                for item in models
                if isinstance(item, dict)
                and isinstance(item.get("id"), str)
            ),
        )

    def _save_local_provider_check(
        self,
        base_url: str,
        result: _LocalProviderCheck,
    ) -> None:
        self.workspace.save_provider_profile(
            "local",
            {
                "kind": "local",
                "base_url": base_url,
                "check_status": "reachable",
                "checked_at": result.checked_at,
                "model_count": str(result.model_count),
            },
        )

    def _provider_is_connected(self, provider: str) -> bool:
        canonical = _canonical_provider(provider)
        if canonical == "mock":
            return True
        if canonical == "local":
            profile = self.workspace.provider_profiles().get("local", {})
            return bool(profile.get("base_url"))
        return bool(self._provider_api_key(canonical))

    def _connect_model_provider(self, args: list[str]) -> None:
        if not args:
            raise SystemExit("Use model provider configure NAME [URL].")
        provider = _canonical_provider(args[0])
        if provider not in _SUPPORTED_PROVIDERS:
            raise SystemExit(
                "Provider must be local/ollama, openai, anthropic/claude, "
                "or mistral."
            )
        if provider == "mock":
            if len(args) != 1:
                raise SystemExit("The built-in mock model takes no settings.")
            self._success("mock is built in and always available.")
            return
        if provider == "local":
            if len(args) > 2:
                raise SystemExit(
                    "Use model provider configure local [BASE_URL]."
                )
            existing = self.workspace.provider_profiles().get("local", {}).get(
                "base_url"
            )
            base_url = (
                args[1]
                if len(args) == 2
                else self.input(
                    "Local OpenAI-compatible base URL "
                    f"[{existing or 'http://127.0.0.1:11434/v1'}]: "
                ).strip()
                or existing
                or "http://127.0.0.1:11434/v1"
            )
            # Prove OpenAI compatibility before replacing a working endpoint.
            self._local_models_url(base_url)
            try:
                result = self._check_local_provider(base_url)
            except _LocalProviderError as exc:
                raise SystemExit(
                    f"Could not connect to the local provider at {base_url}: "
                    f"{exc}. Check that the model server and any SSH tunnel are "
                    "running; the connection was not saved."
                ) from exc
            self._save_local_provider_check(base_url, result)
            noun = "model" if result.model_count == 1 else "models"
            message = (
                f"Connected local provider: reachable; "
                f"{result.model_count} {noun}; endpoint {base_url}"
            )
            if result.model_count:
                self._success(message)
            else:
                self._warning(message + "; install or load a model before running")
            return
        if len(args) != 1:
            raise SystemExit(f"Use model provider configure {provider}.")
        secret_name = _PROVIDER_SECRETS[provider]
        secrets = self.workspace.load_secrets()
        from_environment = bool(os.environ.get(secret_name))
        if from_environment:
            self._success(
                f"Using {secret_name} from the current environment; "
                "its value was not copied."
            )
        else:
            existing = secrets.get(secret_name)
            suffix = " (press Enter to keep the saved value)" if existing else ""
            entered = self.secret_input(f"{secret_name}{suffix}: ").strip()
            if entered:
                secrets[secret_name] = entered
                self.workspace.save_secrets(secrets)
            elif not existing:
                raise SystemExit(f"{secret_name} must not be empty.")
        self.workspace.save_provider_profile(
            provider,
            {"kind": "api", "key_env": secret_name},
        )
        self._success(
            f"Configured {provider}: API key is available for checking."
        )

    def _disconnect_model_provider(self, name: str) -> None:
        provider = _canonical_provider(name)
        if provider not in _SUPPORTED_PROVIDERS or provider == "mock":
            raise SystemExit(
                "Provider removal must name local, openai, anthropic, or mistral."
            )
        self.workspace.remove_provider_profile(provider)
        secret_name = _PROVIDER_SECRETS.get(provider)
        removed_secret = False
        if secret_name:
            secrets = self.workspace.load_secrets()
            if secret_name in secrets:
                secrets.pop(secret_name)
                self.workspace.save_secrets(secrets)
                removed_secret = True
        detail = (
            " and removed its privately stored API key"
            if removed_secret
            else ""
        )
        self._success(f"Removed provider {provider}{detail}.")
        if secret_name and os.environ.get(secret_name):
            self._warning(
                f"{secret_name} is still present in the current environment."
            )
