"""Logical connector requirements and private runtime bindings."""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import ModuleType

__all__ = [
    "ConnectorRequirement",
    "connector_requirements_from_module",
]


_NAME = re.compile(r"[A-Za-z][A-Za-z0-9._-]{0,63}")
_KINDS = {
    "telegram",
    "gmail",
    "google-sheets",
    "google-calendar",
}
_ACCESS = {"read-only", "write", "read-write"}


@dataclass(frozen=True)
class ConnectorRequirement:
    """A code-visible external capability without machine credentials."""

    name: str
    kind: str
    participant: str
    capabilities: tuple[str, ...] = ()
    access: str = "read-write"
    description: str = ""
    required: bool = True

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise ValueError(
                "Connector requirement names must start with a letter and "
                "contain only letters, digits, '.', '_' or '-'."
            )
        if self.kind not in _KINDS:
            raise ValueError(
                f"Unsupported connector kind {self.kind!r}; expected one of "
                + ", ".join(sorted(_KINDS))
                + "."
            )
        if not self.participant.strip():
            raise ValueError("A connector requirement needs a participant.")
        if self.access not in _ACCESS:
            raise ValueError(
                "Connector access must be 'read-only', 'write', or "
                f"'read-write', got {self.access!r}."
            )
        if any(not item.strip() for item in self.capabilities):
            raise ValueError("Connector capabilities must not be empty.")

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "participant": self.participant,
            "capabilities": list(self.capabilities),
            "access": self.access,
            "description": self.description,
            "required": self.required,
        }


def connector_requirements_from_module(
    module: ModuleType | None,
) -> tuple[ConnectorRequirement, ...]:
    """Load and validate ``zippergen_connectors`` from a workflow module."""

    if module is None:
        return ()
    raw = getattr(module, "zippergen_connectors", ())
    if raw is None:
        return ()
    if not isinstance(raw, (tuple, list)):
        raise TypeError("zippergen_connectors must be a tuple or list.")
    requirements: list[ConnectorRequirement] = []
    seen: set[str] = set()
    for value in raw:
        if not isinstance(value, ConnectorRequirement):
            raise TypeError(
                "Every zippergen_connectors entry must be a "
                "ConnectorRequirement."
            )
        key = value.name.casefold()
        if key in seen:
            raise ValueError(
                f"Duplicate connector requirement name: {value.name}."
            )
        seen.add(key)
        requirements.append(value)
    return tuple(requirements)
