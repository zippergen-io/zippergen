"""Logical connector requirements and private runtime bindings."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import ModuleType

__all__ = [
    "ConnectorRequirement",
    "connector_requirements_from_module",
]


#: The deployment hands each connector its routing through the environment, so
#: a workflow never carries an endpoint or a credential in its source.
CONNECTORS_ENV = "ZIPPERGEN_CONNECTORS_JSON"


_NAME = re.compile(r"[A-Za-z][A-Za-z0-9._-]{0,63}")


@dataclass(frozen=True)
class ConnectorSettingSpec:
    """One portable configuration value for a connector kind."""

    name: str
    label: str
    help: str
    metavar: str
    required: bool = True
    prompt: bool = True
    default: str | None = None
    default_from: str | None = None


@dataclass(frozen=True)
class ConnectorReadiness:
    """The kind adapter's answer to one deployment readiness check."""

    status: str
    detail: str


def _telegram_description(values: Mapping[str, str]) -> str:
    return (
        f"chat {values['chat_id']}, trusted user {values['allowed_user_id']}"
    )


def _sheets_description(values: Mapping[str, str]) -> str:
    return f"tab {values['tab']}"


def _gmail_description(values: Mapping[str, str]) -> str:
    return f"query {values['query']!r}"


def _telegram_readiness(
    requirement: "ConnectorRequirement",
    binding: Mapping[str, object],
    environment: Mapping[str, str],
    live: bool,
) -> ConnectorReadiness:
    token = environment.get(str(binding.get("token_env") or ""))
    chat_id = str(binding.get("chat_id") or "")
    if not token or not chat_id:
        return ConnectorReadiness("fail", "Telegram token or chat id is missing")
    if not live:
        return ConnectorReadiness(
            "ok",
            f"Telegram chat {chat_id} is configured; live availability was not checked",
        )
    try:
        from zippergen.telegram_notify import TelegramBotClient

        client = TelegramBotClient(token, timeout=5)
        client.request("getMe")
        client.request("getChat", chat_id=chat_id)
    except Exception as exc:
        return ConnectorReadiness("fail", f"Telegram is unavailable: {exc}")
    return ConnectorReadiness("ok", f"Telegram chat {chat_id} is reachable")


def _sheets_readiness(
    requirement: "ConnectorRequirement",
    binding: Mapping[str, object],
    environment: Mapping[str, str],
    live: bool,
) -> ConnectorReadiness:
    credential = environment.get(str(binding.get("credential_env") or ""))
    spreadsheet_id = str(binding.get("spreadsheet_id") or "")
    tab = str(binding.get("tab") or "")
    if not credential or not spreadsheet_id or not tab:
        return ConnectorReadiness(
            "fail", "Google credential, spreadsheet ID, or tab is missing"
        )
    if not live:
        return ConnectorReadiness(
            "ok", f"Google spreadsheet {spreadsheet_id}, tab {tab} is configured"
        )
    try:
        from zippergen.google_sheets import GoogleSheetsTable

        info = GoogleSheetsTable(
            requirement=requirement.name,
            spreadsheet_id=spreadsheet_id,
            tab=tab,
            credential_json=credential,
            access=requirement.access,
        ).inspect()
    except Exception as exc:
        return ConnectorReadiness("fail", f"Google Sheets is unavailable: {exc}")
    return ConnectorReadiness(
        "ok", f"{info['title']}, tab {info['tab']} is reachable"
    )


def _gmail_readiness(
    requirement: "ConnectorRequirement",
    binding: Mapping[str, object],
    environment: Mapping[str, str],
    live: bool,
) -> ConnectorReadiness:
    credential = environment.get(str(binding.get("credential_env") or ""))
    account = str(binding.get("account") or "me")
    query = str(binding.get("query") or "is:unread in:inbox")
    if not credential:
        return ConnectorReadiness("fail", "Google credential is missing")
    if not live:
        return ConnectorReadiness(
            "ok", f"Gmail account {account}, query {query!r} is configured"
        )
    try:
        from zippergen.google_gmail import GmailMailbox

        info = GmailMailbox(
            requirement=requirement.name,
            account=account,
            query=query,
            credential_json=credential,
            access=requirement.access,
        ).inspect()
    except Exception as exc:
        return ConnectorReadiness("fail", f"Gmail is unavailable: {exc}")
    return ConnectorReadiness(
        "ok", f"Gmail account {info['email']} is reachable"
    )


@dataclass(frozen=True)
class ConnectorKindSpec:
    """Everything that is true of one connector kind, in one place.

    Adding a kind used to mean touching a provider mapping, a configuration
    branch, a credential-wiring branch, an authorization branch, a readiness
    branch and an install extra -- each in a different module, each looking
    reasonable alone. A kind could satisfy every name-based completeness test
    and still be impossible to configure or wire.

    A kind exists here or not at all, and ``tests/test_connector_kinds.py``
    checks each field against the code that consumes it.
    """

    #: The name a workflow writes in a ``ConnectorRequirement``.
    name: str
    #: The provider connection kind that serves it.
    provider: str
    #: The credential field the provider stores for it.
    credential: str
    #: Portable configuration keys a person answers, beyond name/connection.
    settings: tuple[ConnectorSettingSpec, ...]
    #: Field in the durable binding that names the credential environment key.
    credential_environment_field: str
    #: Human-readable summary after configuration.
    describe: Callable[[Mapping[str, str]], str]
    #: Validate configuration and optionally contact the remote service.
    readiness: Callable[
        ["ConnectorRequirement", Mapping[str, object], Mapping[str, str], bool],
        ConnectorReadiness,
    ]
    #: The optional install extra a deployment needs, if any.
    extra: str | None = None
    #: OAuth scopes by access level, for kinds whose provider uses OAuth.
    scopes: Mapping[str, str] = field(default_factory=dict)


#: The connector kinds ZipperGen supports, in the order they are offered.
CONNECTOR_KIND_SPECS: tuple[ConnectorKindSpec, ...] = (
    ConnectorKindSpec(
        name="telegram",
        provider="telegram",
        credential="bot_token",
        settings=(
            ConnectorSettingSpec(
                "chat_id",
                "Telegram chat id",
                "Telegram chat id that receives approval messages.",
                "CHAT_ID",
            ),
            ConnectorSettingSpec(
                "allowed_user_id",
                "Telegram allowed user id",
                "Telegram user id allowed to answer approvals.",
                "USER_ID",
                required=False,
                prompt=False,
                default_from="chat_id",
            ),
        ),
        credential_environment_field="token_env",
        describe=_telegram_description,
        readiness=_telegram_readiness,
    ),
    ConnectorKindSpec(
        name="gmail",
        provider="google",
        credential="authorized_user_json",
        settings=(
            ConnectorSettingSpec(
                "account",
                "Gmail account",
                "Mailbox to read.",
                "ACCOUNT",
                prompt=False,
                default="me",
            ),
            ConnectorSettingSpec(
                "query",
                "Gmail query",
                "Gmail search that selects the messages to handle.",
                "QUERY",
                prompt=False,
                default="is:unread in:inbox",
            ),
        ),
        credential_environment_field="credential_env",
        describe=_gmail_description,
        readiness=_gmail_readiness,
        extra="google",
        scopes={
            "read-only": "https://www.googleapis.com/auth/gmail.readonly",
            "write": "https://www.googleapis.com/auth/gmail.modify",
            "read-write": "https://www.googleapis.com/auth/gmail.modify",
        },
    ),
    ConnectorKindSpec(
        name="google-sheets",
        provider="google",
        credential="authorized_user_json",
        settings=(
            ConnectorSettingSpec(
                "spreadsheet_id",
                "Google spreadsheet id",
                "Spreadsheet id, the long value in its URL.",
                "ID",
            ),
            ConnectorSettingSpec(
                "tab", "Google Sheets tab", "Sheet tab name.", "TAB"
            ),
        ),
        credential_environment_field="credential_env",
        describe=_sheets_description,
        readiness=_sheets_readiness,
        extra="google",
        scopes={
            "read-only": "https://www.googleapis.com/auth/spreadsheets.readonly",
            "write": "https://www.googleapis.com/auth/spreadsheets",
            "read-write": "https://www.googleapis.com/auth/spreadsheets",
        },
    ),
)

CONNECTOR_KINDS = tuple(spec.name for spec in CONNECTOR_KIND_SPECS)

def _distinct_setting_specs() -> tuple[ConnectorSettingSpec, ...]:
    """Return each CLI setting once and reject conflicting declarations."""

    by_name: dict[str, ConnectorSettingSpec] = {}
    for spec in CONNECTOR_KIND_SPECS:
        for setting in spec.settings:
            previous = by_name.get(setting.name)
            if previous is not None and previous != setting:
                raise RuntimeError(
                    f"Connector setting {setting.name!r} has conflicting specs."
                )
            by_name.setdefault(setting.name, setting)
    return tuple(by_name.values())


CONNECTOR_SETTING_SPECS = _distinct_setting_specs()
CONNECTOR_SETTING_NAMES = frozenset(
    setting.name for setting in CONNECTOR_SETTING_SPECS
)

_SPECS = {spec.name: spec for spec in CONNECTOR_KIND_SPECS}

_KINDS = frozenset(CONNECTOR_KINDS)


def connector_kind_spec(kind: object) -> ConnectorKindSpec | None:
    """Return the spec for one kind, or None when nothing declares it."""

    return _SPECS.get(str(kind or "").strip())


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
