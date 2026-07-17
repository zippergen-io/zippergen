"""Declarative deployment metadata for ZipperGen workflows.

Workflow modules may expose a ``zippergen_deployment`` value containing a
``DeploymentSpec`` (or the equivalent plain dictionary).  The deployment CLI
uses the declaration to collect configuration, keep secrets out of profiles,
install optional packages, run one-time setup, and diagnose a deployment.

The types intentionally contain data only.  Besides making declarations easy
to inspect and test, this gives workflow-generating LLMs a small, stable schema
to emit alongside a workflow.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import ModuleType
from typing import Any, Mapping


_FIELD_TARGETS = {"option", "env", "llm", "services", "input"}


@dataclass(frozen=True)
class DeploymentField:
    """One value collected while configuring a deployment.

    ``target`` controls where the value is stored.  Options and inputs are kept
    in the public deployment profile.  Environment values are loaded before
    the workflow module is imported; when ``secret`` is true their value is
    written to a separate mode-0600 file and never copied into the profile.
    """

    name: str
    prompt: str
    target: str = "option"
    default: object = None
    required: bool = False
    secret: bool = False
    env: str | None = None
    choices: tuple[str, ...] = ()
    help: str | None = None
    when: str | None = None
    when_values: tuple[str, ...] = ()
    path_exists: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("deployment field name must not be empty")
        if self.target not in _FIELD_TARGETS:
            raise ValueError(
                f"deployment field {self.name!r} has invalid target {self.target!r}; "
                f"expected one of {sorted(_FIELD_TARGETS)}"
            )
        if self.secret and self.target != "env":
            raise ValueError(f"secret deployment field {self.name!r} must target env")
        if self.target == "env" and not (self.env or self.name):
            raise ValueError("environment deployment fields require an env name")

    @property
    def target_name(self) -> str:
        return self.env or self.name


@dataclass(frozen=True)
class DeploymentPackage:
    """A Python requirement needed by a deployed workflow."""

    requirement: str
    import_name: str | None = None

    def __post_init__(self) -> None:
        if not self.requirement:
            raise ValueError("deployment package requirement must not be empty")


@dataclass(frozen=True)
class DeploymentSetup:
    """A shell-free one-time setup command run in the deployment environment."""

    name: str
    description: str
    command: tuple[str, ...]
    when: str | None = None
    when_values: tuple[str, ...] = ()
    creates_env: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("deployment setup name must not be empty")
        if not self.command:
            raise ValueError(f"deployment setup {self.name!r} requires a command")


@dataclass(frozen=True)
class DeploymentSpec:
    """Deployment requirements declared by a workflow module."""

    name: str | None = None
    description: str = ""
    fields: tuple[DeploymentField, ...] = ()
    packages: tuple[DeploymentPackage, ...] = ()
    setup: tuple[DeploymentSetup, ...] = ()
    files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        names = [field.name for field in self.fields]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate deployment fields: {', '.join(duplicates)}")

    def as_dict(self) -> dict[str, Any]:
        """Return the declaration as plain JSON-compatible data."""

        return asdict(self)


def _tuple_strings(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)  # type: ignore[union-attr]


def _field_from_mapping(value: Mapping[str, object]) -> DeploymentField:
    data = dict(value)
    data["choices"] = _tuple_strings(data.get("choices"))
    data["when_values"] = _tuple_strings(data.get("when_values"))
    return DeploymentField(**data)  # type: ignore[arg-type]


def _package_from_value(value: object) -> DeploymentPackage:
    if isinstance(value, DeploymentPackage):
        return value
    if isinstance(value, str):
        return DeploymentPackage(value)
    if isinstance(value, Mapping):
        return DeploymentPackage(**dict(value))  # type: ignore[arg-type]
    raise TypeError(f"invalid deployment package: {value!r}")


def _setup_from_mapping(value: Mapping[str, object]) -> DeploymentSetup:
    data = dict(value)
    data["command"] = _tuple_strings(data.get("command"))
    data["when_values"] = _tuple_strings(data.get("when_values"))
    return DeploymentSetup(**data)  # type: ignore[arg-type]


def normalize_deployment_spec(value: object) -> DeploymentSpec:
    """Validate and normalize a typed or dictionary deployment declaration."""

    if isinstance(value, DeploymentSpec):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("zippergen_deployment must be a DeploymentSpec or mapping")
    data = dict(value)
    fields = data.get("fields") or ()
    packages = data.get("packages") or ()
    setup = data.get("setup") or ()
    data["fields"] = tuple(
        item if isinstance(item, DeploymentField) else _field_from_mapping(item)
        for item in fields  # type: ignore[union-attr]
    )
    data["packages"] = tuple(_package_from_value(item) for item in packages)  # type: ignore[union-attr]
    data["setup"] = tuple(
        item if isinstance(item, DeploymentSetup) else _setup_from_mapping(item)
        for item in setup  # type: ignore[union-attr]
    )
    data["files"] = _tuple_strings(data.get("files"))
    return DeploymentSpec(**data)  # type: ignore[arg-type]


def deployment_spec_from_module(module: ModuleType) -> DeploymentSpec:
    """Load a module declaration, returning an empty spec when absent."""

    value = getattr(module, "zippergen_deployment", None)
    if value is None:
        return DeploymentSpec()
    return normalize_deployment_spec(value)


__all__ = [
    "DeploymentField",
    "DeploymentPackage",
    "DeploymentSetup",
    "DeploymentSpec",
    "deployment_spec_from_module",
    "normalize_deployment_spec",
]
