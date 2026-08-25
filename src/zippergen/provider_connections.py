"""Named access paths to external model and connector providers.

A provider kind says which protocol ZipperGen speaks (OpenAI, Telegram,
Google, and so on).  A provider connection is one named identity at that
provider: it owns the site endpoint and private credential that several model
or connector configurations may share.
"""

from __future__ import annotations

import re

from zippergen.connectors import CONNECTOR_KINDS


# ``mock`` is a runtime backend, not a configurable external connection, so it
# is accepted by the model-spec parser but deliberately absent from this list.
PROVIDER_KINDS = (
    "openai",
    "anthropic",
    "mistral",
    "local",
    "scripted",
    "telegram",
    "google",
)
_RUNTIME_PROVIDER_KINDS = (*PROVIDER_KINDS, "mock")

_ALIASES = {"claude": "anthropic", "ollama": "local"}
_MODEL_KINDS = frozenset({"openai", "anthropic", "mistral", "local", "scripted"})
_CONNECTOR_KINDS = {
    "telegram": frozenset({"telegram"}),
    "google": frozenset({"gmail", "google-sheets"}),
}
_CREDENTIAL_FIELDS = {
    "openai": "api_key",
    "anthropic": "api_key",
    "mistral": "api_key",
    "telegram": "bot_token",
    "google": "authorized_user_json",
}
_CREDENTIAL_LABELS = {
    "openai": "OpenAI API key",
    "anthropic": "Anthropic API key",
    "mistral": "Mistral API key",
    "telegram": "Telegram bot token",
    "google": "Google authorization",
}
_STANDARD_ENVIRONMENT = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}


def canonical_provider_kind(value: object) -> str:
    """Normalize one provider kind without accepting an unknown value."""

    raw = str(value or "").strip().casefold()
    kind = _ALIASES.get(raw, raw)
    if kind not in _RUNTIME_PROVIDER_KINDS:
        raise ValueError(
            f"Unsupported provider kind {raw!r}. Supported: "
            + ", ".join(PROVIDER_KINDS)
        )
    return kind


def provider_supports_models(kind: object) -> bool:
    return canonical_provider_kind(kind) in _MODEL_KINDS


def connector_kinds_for_provider(kind: object) -> tuple[str, ...]:
    """The connector kinds one provider serves, in display order.

    This is the only statement of provider-to-connector compatibility. Every
    place that offers, filters, or validates a kind asks here -- otherwise a
    new kind can be accepted by the parser and rejected by the interactive
    command, which is exactly what happened when this was written out by hand
    in three places.
    """

    served = _CONNECTOR_KINDS.get(str(kind or "").strip(), frozenset())
    return tuple(name for name in CONNECTOR_KINDS if name in served)


def providers_serving_connectors() -> frozenset[str]:
    """Provider kinds that serve at least one connector kind."""

    return frozenset(
        name for name, served in _CONNECTOR_KINDS.items() if served
    )


def provider_supports_connector(kind: object, connector_kind: object) -> bool:
    provider = canonical_provider_kind(kind)
    return str(connector_kind or "").strip() in _CONNECTOR_KINDS.get(
        provider, frozenset()
    )


def provider_credential_field(kind: object) -> str | None:
    return _CREDENTIAL_FIELDS.get(canonical_provider_kind(kind))


def provider_credential_label(kind: object) -> str | None:
    return _CREDENTIAL_LABELS.get(canonical_provider_kind(kind))


def provider_standard_environment(kind: object) -> str | None:
    return _STANDARD_ENVIRONMENT.get(canonical_provider_kind(kind))


def provider_environment_name(connection: str, field: str) -> str:
    """Return a process-local variable name for one named connection field."""

    encoded = {"-": "_DASH_", ".": "_DOT_", "_": "_UNDERSCORE_"}
    stem = "".join(
        character.upper() if character.isalnum() else encoded[character]
        for character in connection
    )
    suffix = re.sub(r"[^A-Za-z0-9]+", "_", field).strip("_").upper()
    return f"ZIPPERGEN_PROVIDER_{stem}_{suffix}"


def connected_model_spec(connection: str, kind: object, model: object) -> str:
    """Encode a resolved project model route for the runtime.

    The ``provider@connection:model`` form is an internal runtime snapshot. A
    direct user override such as ``openai:gpt-4o-mini`` remains valid and uses
    the provider's conventional environment variable.
    """

    provider = canonical_provider_kind(kind)
    selected_model = str(model or "").strip()
    if not selected_model:
        raise ValueError("A model configuration requires a model name or path.")
    return f"{provider}@{connection}:{selected_model}"


def split_model_spec(spec: str) -> tuple[str, str | None, str | None]:
    """Return ``(provider kind, connection name, model)`` for a runtime spec."""

    provider_token, separator, model = spec.strip().partition(":")
    raw_provider, connection_separator, connection = provider_token.partition("@")
    provider = canonical_provider_kind(raw_provider)
    selected_connection = connection.strip() if connection_separator else None
    selected_model = model.strip() if separator else None
    if connection_separator and not selected_connection:
        raise ValueError(f"Model spec {spec!r} has an empty provider connection.")
    if separator and not selected_model:
        raise ValueError(f"Model spec {spec!r} is missing a model after ':'.")
    return provider, selected_connection, selected_model


__all__ = [
    "PROVIDER_KINDS",
    "canonical_provider_kind",
    "connected_model_spec",
    "provider_credential_field",
    "provider_credential_label",
    "provider_environment_name",
    "provider_standard_environment",
    "provider_supports_connector",
    "provider_supports_models",
    "split_model_spec",
]
