"""Logical connector requirements and private runtime bindings."""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import ModuleType

__all__ = [
    "ConnectorRequirement",
    "connector_requirements_from_module",
]


#: The deployment hands each connector its routing through the environment, so
#: a workflow never carries an endpoint or a credential in its source.
CONNECTORS_ENV = "ZIPPERGEN_CONNECTORS_JSON"


_NAME = re.compile(r"[A-Za-z][A-Za-z0-9._-]{0,63}")
#: The connector kinds ZipperGen supports, in the order they are shown.
#:
#: This is the only declaration. Everything that offers, routes, or renders a
#: connector kind derives its list from here, so a kind cannot be accepted in a
#: workflow without a provider that serves it and a command that configures it.
#: ``tests/test_connector_kinds.py`` holds the two halves together.
CONNECTOR_KINDS = ("telegram", "gmail", "google-sheets")

_KINDS = frozenset(CONNECTOR_KINDS)
_ACCESS = {"read-only", "write", "read-write"}


@dataclass(frozen=True)
class ConnectorRequirement:
    """A code-visible external capability without machine credentials."""

    name: str
    kind: str
    participant: str
    #: What this connector is used for, written for a reader. ZipperGen does
    #: not enforce these, and cannot: a capability is a claim about a remote
    #: service, and the token that reaches it carries whatever scopes the
    #: provider granted. Narrowing what a connector may do is the provider's
    #: job, done when you authorize it. These strings tell a reviewer what to
    #: check for; they are not a sandbox.
    capabilities: tuple[str, ...] = ()
    #: Likewise descriptive. ``access`` records the intent under review.
    access: str = "read-only"
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


def requirement_binding(
    requirement: str,
    *,
    kind: str,
    error: type[Exception],
) -> dict[str, object]:
    """Return the runtime binding recorded for one declared requirement.

    Every connector module needs the same four answers: is any routing active,
    is it well formed, is this requirement bound, and is it bound to the kind
    the caller can actually speak. Each module raises its own error type, which
    is the only thing that ever differed between the copies of this.
    """

    import json
    import os

    raw = os.environ.get(CONNECTORS_ENV, "")
    if not raw:
        raise error(
            "No connector runtime configuration is active. Configure it with "
            f"'zippergen connector configure NAME CONNECTION {kind}', bind it "
            "with 'zippergen connector assign TARGET NAME', then deploy."
        )
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise error("The connector runtime configuration is malformed.") from exc
    if not isinstance(value, dict):
        raise error("The connector runtime configuration must be an object.")

    record = value.get(f"requirement:{requirement}") or value.get(requirement)
    if not isinstance(record, dict):
        raise error(f"Connector requirement is not bound: {requirement}.")
    bound = str(record.get("kind") or "")
    if bound != kind:
        raise error(
            f"Connector requirement {requirement!r} is bound to "
            f"{bound or 'an unknown connector'}, not {kind}."
        )
    return dict(record)
