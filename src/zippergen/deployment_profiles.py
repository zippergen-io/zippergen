"""Reading a deployment's stored profile.

A deployment records what it needs to run: which workflow, which store, which
environment values. These are the accessors for that record. They read; they
never prompt and never start anything.

Extracted from the CLI dispatcher, which is not where domain logic belongs.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from zippergen.deployment import DeploymentField
from zippergen.deployment_platform import (
    deployment_profile_path as _deployment_profile_path,
    slug as _slug,
    zippergen_home as _zippergen_home,
)
from zippergen.value_codec import decode_value


DEPLOYMENT_PROFILE_SCHEMA_VERSION = 3


def _require_deployment_profile_schema(
    profile: dict[str, object], path: Path
) -> None:
    """Refuse a profile whose format this version does not understand."""

    version = profile.get("schema_version")
    if version == DEPLOYMENT_PROFILE_SCHEMA_VERSION:
        return
    if not isinstance(version, int) or isinstance(version, bool):
        raise SystemExit(
            f"Deployment profile {path} does not say which schema it uses "
            f"({version!r}). Remove the file and run 'zippergen deploy' to "
            "create a current one; you will be asked for its settings again."
        )
    if version > DEPLOYMENT_PROFILE_SCHEMA_VERSION:
        raise SystemExit(
            f"Deployment profile {path} uses schema {version}, but this "
            f"ZipperGen reads {DEPLOYMENT_PROFILE_SCHEMA_VERSION}. It was "
            "written by a newer ZipperGen; upgrade this one to use it."
        )
    raise SystemExit(
        f"Deployment profile {path} uses schema {version}, but this ZipperGen "
        f"reads {DEPLOYMENT_PROFILE_SCHEMA_VERSION}. No migration is available. "
        "Remove the file and run 'zippergen deploy' to create a current one."
    )


def _load_deployment_profile(name: str) -> dict[str, object]:
    path = _deployment_profile_path(name)
    if not path.exists():
        raise SystemExit(f"Deployment profile not found: {name}")
    try:
        profile = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Deployment profile is not valid JSON: {path}") from exc
    if not isinstance(profile, dict):
        raise SystemExit(f"Deployment profile is not an object: {path}")
    _require_deployment_profile_schema(profile, path)
    try:
        inputs = decode_value(profile.get("inputs"))
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Deployment inputs are malformed in {path}.") from exc
    if not isinstance(inputs, dict):
        raise SystemExit(f"Deployment inputs are not an object in {path}.")
    profile["inputs"] = inputs
    return profile


def _default_deployment_store_path(name: str) -> str:
    return str(_zippergen_home() / "runs" / f"{_slug(name)}.sqlite")


def _default_deployment_log_path(name: str) -> str:
    return str(_zippergen_home() / "logs" / f"{_slug(name)}.log")


def _load_deployment_secrets(profile: dict[str, object]) -> dict[str, str]:
    raw_path = profile.get("secrets_file")
    if not raw_path:
        return {}
    path = Path(str(raw_path)).expanduser()
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Deployment secrets file is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"Deployment secrets file is not an object: {path}")
    return {str(key): str(item) for key, item in value.items()}


def _profile_mapping(profile: dict[str, object], key: str) -> dict[str, object]:
    raw = profile.get(key)
    if not isinstance(raw, dict):
        return {}
    return {str(name): value for name, value in raw.items()}


def _deployment_environment(profile: dict[str, object]) -> dict[str, str]:
    raw = profile.get("environment") or {}
    if not isinstance(raw, dict):
        raise SystemExit("Deployment profile environment must be an object.")
    values = {str(key): str(value) for key, value in raw.items()}
    values.update(_load_deployment_secrets(profile))
    connectors = profile.get("connectors")
    if isinstance(connectors, dict):
        values["ZIPPERGEN_CONNECTORS_JSON"] = json.dumps(
            connectors,
            sort_keys=True,
        )
    return values


@contextmanager
def _profile_environment(profile: dict[str, object]):
    values = _deployment_environment(profile)
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield values
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _profile_options(profile: dict[str, object]) -> dict[str, object]:
    return _profile_mapping(profile, "options")


def _profile_field_value(
    profile: dict[str, object],
    field: DeploymentField,
    secrets: dict[str, str],
) -> object:
    if field.target == "input":
        values = profile.get("inputs") or {}
        return values.get(field.target_name) if isinstance(values, dict) else None
    if field.target == "option":
        values = profile.get("options") or {}
        return values.get(field.target_name) if isinstance(values, dict) else None
    if field.secret:
        return secrets.get(field.target_name)
    values = profile.get("environment") or {}
    return values.get(field.target_name) if isinstance(values, dict) else None


def _field_enabled(field: DeploymentField, values: dict[str, object]) -> bool:
    if not field.when:
        return True
    candidates = [values.get(field.when)]
    llm_field_names = values.get("__llm_field_names__")
    if (
        field.when == "llm"
        or (
            isinstance(llm_field_names, (list, tuple, set))
            and field.when in llm_field_names
        )
    ):
        configured = values.get("__llm_specs__")
        if isinstance(configured, (list, tuple, set)):
            candidates.extend(configured)
    if not field.when_values:
        return any(bool(current) for current in candidates)
    return any(
        str(current).startswith(expected[:-1])
        if expected.endswith("*")
        else str(current) == expected
        for current in candidates
        for expected in field.when_values
    )
