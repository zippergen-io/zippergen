"""Shared OAuth support for Google connector providers."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


GOOGLE_SHEETS_READONLY_SCOPE = (
    "https://www.googleapis.com/auth/spreadsheets.readonly"
)
GOOGLE_SHEETS_SCOPE = "https://www.googleapis.com/auth/spreadsheets"
GOOGLE_GMAIL_READONLY_SCOPE = (
    "https://www.googleapis.com/auth/gmail.readonly"
)
GOOGLE_GMAIL_MODIFY_SCOPE = "https://www.googleapis.com/auth/gmail.modify"
GOOGLE_SCOPE_ALIASES = {
    "gmail.readonly": GOOGLE_GMAIL_READONLY_SCOPE,
    "gmail.modify": GOOGLE_GMAIL_MODIFY_SCOPE,
    "spreadsheets.readonly": GOOGLE_SHEETS_READONLY_SCOPE,
    "spreadsheets": GOOGLE_SHEETS_SCOPE,
}
_GOOGLE_SCOPE_NAMES = {
    value: name for name, value in GOOGLE_SCOPE_ALIASES.items()
}
_GOOGLE_AUTHORIZATION_PREFIX = "zg-google-v1"


class GoogleConnectorError(RuntimeError):
    """A Google connector error phrased for a person to act on."""


@dataclass(frozen=True)
class GoogleAuthorization:
    """Portable result of a browser-side Google authorization."""

    authorized_user_json: str
    granted_scopes: tuple[str, ...]
    client_id: str
    expiry: str | None = None


def google_imports():
    try:
        from google.auth.transport.requests import AuthorizedSession, Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as exc:
        raise GoogleConnectorError(
            "Google connector support is not installed. Install it with "
            'pip install "zippergen[google]".'
        ) from exc
    return AuthorizedSession, Request, Credentials, InstalledAppFlow


def normalize_google_scopes(scopes: Iterable[str]) -> tuple[str, ...]:
    values = tuple(sorted({str(scope).strip() for scope in scopes if str(scope).strip()}))
    if not values:
        raise ValueError("At least one Google OAuth scope is required.")
    return values


def parse_google_scopes(value: str | Iterable[str]) -> tuple[str, ...]:
    """Parse friendly names or complete Google OAuth scope URLs."""

    if isinstance(value, str):
        raw = value.split(",")
    else:
        raw = value
    scopes: list[str] = []
    unknown: list[str] = []
    for item in raw:
        candidate = str(item).strip()
        if not candidate:
            continue
        resolved = GOOGLE_SCOPE_ALIASES.get(candidate.casefold(), candidate)
        if not resolved.startswith("https://www.googleapis.com/auth/"):
            unknown.append(candidate)
        else:
            scopes.append(resolved)
    if unknown:
        raise GoogleConnectorError(
            "Unknown Google OAuth scope: "
            + ", ".join(unknown)
            + ". Use gmail.readonly, gmail.modify, "
            "spreadsheets.readonly, or spreadsheets."
        )
    return normalize_google_scopes(scopes)


def google_scope_names(scopes: Iterable[str]) -> tuple[str, ...]:
    """Render known Google scopes as short user-facing names."""

    return tuple(
        _GOOGLE_SCOPE_NAMES.get(scope, scope)
        for scope in normalize_google_scopes(scopes)
    )


def google_scope_for_access(kind: str, access: str) -> str:
    """Return the narrowest supported OAuth scope for one requirement."""

    if access not in {"read-only", "write", "read-write"}:
        raise ValueError(f"Unsupported connector access: {access!r}.")
    if kind == "gmail":
        return (
            GOOGLE_GMAIL_READONLY_SCOPE
            if access == "read-only"
            else GOOGLE_GMAIL_MODIFY_SCOPE
        )
    if kind == "google-sheets":
        return (
            GOOGLE_SHEETS_READONLY_SCOPE
            if access == "read-only"
            else GOOGLE_SHEETS_SCOPE
        )
    raise ValueError(f"Unsupported Google connector kind: {kind!r}.")


def google_scopes_for_access(
    requirements: Iterable[tuple[str, str]],
) -> tuple[str, ...]:
    """Plan minimal scopes for ``(kind, access)`` requirement pairs."""

    strongest: dict[str, str] = {}
    for kind, access in requirements:
        if kind not in {"gmail", "google-sheets"}:
            continue
        previous = strongest.get(kind)
        if previous in {"write", "read-write"}:
            continue
        strongest[kind] = access
    return normalize_google_scopes(
        google_scope_for_access(kind, access)
        for kind, access in strongest.items()
    )


def google_scopes_cover(
    configured: Iterable[str],
    required: Iterable[str],
) -> bool:
    """Return whether configured scopes provide every required permission."""

    available = set(configured)
    implications = {
        GOOGLE_GMAIL_READONLY_SCOPE: {GOOGLE_GMAIL_MODIFY_SCOPE},
        GOOGLE_SHEETS_READONLY_SCOPE: {GOOGLE_SHEETS_SCOPE},
    }
    return all(
        scope in available
        or bool(available.intersection(implications.get(scope, set())))
        for scope in required
    )


def normalize_google_client_json(value: str) -> str:
    """Validate and normalize one Google desktop OAuth client document."""

    try:
        document = json.loads(value)
    except json.JSONDecodeError as exc:
        raise GoogleConnectorError(
            f"Google OAuth client JSON is invalid: {exc.msg}"
        ) from exc
    if not isinstance(document, dict):
        raise GoogleConnectorError(
            "Google OAuth client JSON must contain one JSON object."
        )
    installed = document.get("installed")
    if not isinstance(installed, dict):
        raise GoogleConnectorError(
            "Google OAuth client JSON must describe a Desktop app. "
            "Create a Desktop app OAuth client in Google Cloud and download "
            "its JSON file."
        )
    missing = [
        field
        for field in ("client_id", "client_secret", "auth_uri", "token_uri")
        if not str(installed.get(field) or "").strip()
    ]
    if missing:
        raise GoogleConnectorError(
            "Google OAuth Desktop app JSON is missing: "
            + ", ".join(missing)
            + "."
        )
    return json.dumps(document, sort_keys=True, separators=(",", ":"))


def authorize_google_client_result(
    client_json: str,
    *,
    scopes: Iterable[str],
    open_browser: bool = True,
) -> GoogleAuthorization:
    """Authorize a validated private desktop client JSON document."""

    requested = normalize_google_scopes(scopes)
    normalized = normalize_google_client_json(client_json)
    _session, _request, _credentials, flow_type = google_imports()
    try:
        flow = flow_type.from_client_config(
            json.loads(normalized),
            scopes=list(requested),
        )
        credentials = flow.run_local_server(
            host="127.0.0.1",
            port=0,
            open_browser=open_browser,
            authorization_prompt_message=(
                "Open this URL in your browser to authorize Google services:\n{url}"
            ),
            success_message=(
                "Google authorization completed. You may close this tab."
            ),
            timeout_seconds=300,
            access_type="offline",
            prompt="consent",
        )
    except Exception as exc:
        raise GoogleConnectorError(f"Google authorization failed: {exc}") from exc
    # OAuth servers may omit ``scope`` only when the granted set is identical
    # to the requested set. Preserve the explicit grant when Google returns it
    # because Credentials.to_json() serializes requested scopes instead.
    raw_granted = getattr(credentials, "granted_scopes", None) or requested
    if isinstance(raw_granted, str):
        raw_granted = raw_granted.split()
    granted = tuple(str(scope) for scope in raw_granted)
    authorized_user_json = credentials.to_json()
    try:
        document = json.loads(authorized_user_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise GoogleConnectorError(
            "Google returned an invalid authorized-user credential."
        ) from exc
    client_id = str(
        document.get("client_id")
        or json.loads(normalized)["installed"]["client_id"]
    )
    expiry = document.get("expiry")
    return GoogleAuthorization(
        authorized_user_json=json.dumps(
            document, sort_keys=True, separators=(",", ":")
        ),
        granted_scopes=normalize_google_scopes(granted),
        client_id=client_id,
        expiry=str(expiry) if expiry else None,
    )


def authorize_google_client(
    client_json: str,
    *,
    scopes: Iterable[str],
    open_browser: bool = True,
) -> str:
    """Authorize a desktop client and return authorized-user JSON."""

    return authorize_google_client_result(
        client_json,
        scopes=scopes,
        open_browser=open_browser,
    ).authorized_user_json


def authorize_google(
    credentials_file: str | Path,
    *,
    scopes: Iterable[str],
    open_browser: bool = True,
) -> str:
    """Run Google's desktop OAuth flow and return authorized-user JSON."""

    path = Path(credentials_file).expanduser().resolve()
    if not path.is_file():
        raise GoogleConnectorError(
            f"Google OAuth desktop credentials file does not exist: {path}"
        )
    try:
        client_json = path.read_text()
    except OSError as exc:
        raise GoogleConnectorError(
            f"Could not read Google OAuth client JSON: {exc}"
        ) from exc
    return authorize_google_client(
        client_json,
        scopes=scopes,
        open_browser=open_browser,
    )


def encode_google_authorization(result: GoogleAuthorization) -> str:
    """Encode one private, checksummed browser-to-CLI handoff."""

    try:
        credential = json.loads(result.authorized_user_json)
    except json.JSONDecodeError as exc:
        raise GoogleConnectorError(
            "Google authorized-user JSON is invalid."
        ) from exc
    if not isinstance(credential, dict):
        raise GoogleConnectorError(
            "Google authorized-user JSON must contain one object."
        )
    payload = json.dumps(
        {
            "credential": credential,
            "granted_scopes": list(
                normalize_google_scopes(result.granted_scopes)
            ),
            "client_id": result.client_id,
            "expiry": result.expiry,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    encoded = base64.urlsafe_b64encode(payload).decode().rstrip("=")
    checksum = hashlib.sha256(payload).hexdigest()[:16]
    return f"{_GOOGLE_AUTHORIZATION_PREFIX}.{encoded}.{checksum}"


def decode_google_authorization(value: str) -> GoogleAuthorization:
    """Decode and validate one private browser-to-CLI handoff."""

    parts = value.strip().split(".")
    if len(parts) != 3 or parts[0] != _GOOGLE_AUTHORIZATION_PREFIX:
        raise GoogleConnectorError(
            "The Google authorization result is not a ZipperGen v1 handoff. "
            "Run the displayed command again and paste its complete final line."
        )
    encoded, expected_checksum = parts[1:]
    try:
        payload = base64.urlsafe_b64decode(
            encoded + "=" * (-len(encoded) % 4)
        )
    except (ValueError, TypeError) as exc:
        raise GoogleConnectorError(
            "The Google authorization result is truncated or malformed."
        ) from exc
    actual_checksum = hashlib.sha256(payload).hexdigest()[:16]
    if not hmac.compare_digest(actual_checksum, expected_checksum):
        raise GoogleConnectorError(
            "The Google authorization result is truncated or changed. "
            "Run the displayed command again and paste its complete final line."
        )
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GoogleConnectorError(
            "The Google authorization result contains invalid data."
        ) from exc
    if not isinstance(document, dict):
        raise GoogleConnectorError(
            "The Google authorization result must contain one object."
        )
    credential = document.get("credential")
    granted_scopes = document.get("granted_scopes")
    client_id = str(document.get("client_id") or "")
    if (
        not isinstance(credential, dict)
        or not isinstance(granted_scopes, list)
        or not client_id
    ):
        raise GoogleConnectorError(
            "The Google authorization result is incomplete."
        )
    return GoogleAuthorization(
        authorized_user_json=json.dumps(
            credential, sort_keys=True, separators=(",", ":")
        ),
        granted_scopes=normalize_google_scopes(
            str(scope) for scope in granted_scopes
        ),
        client_id=client_id,
        expiry=(
            str(document["expiry"])
            if document.get("expiry")
            else None
        ),
    )


def google_authorization_summary(
    result: GoogleAuthorization,
) -> tuple[str, str, str]:
    """Return non-secret confirmation fields for a private authorization."""

    prefix = result.client_id.split(".", 1)[0][:12]
    return (
        ", ".join(google_scope_names(result.granted_scopes)),
        f"{prefix}…",
        result.expiry or "refreshable; no fixed expiry",
    )


def credentials_from_json(value: str, *, scopes: Iterable[str]):
    requested = normalize_google_scopes(scopes)
    _session, request_type, credentials_type, _flow = google_imports()
    try:
        info = json.loads(value)
        if not isinstance(info, dict):
            raise TypeError("credential JSON must be an object")
        # The refresh token already represents the scopes granted during the
        # browser authorization.  Google refresh requests do not need to
        # renegotiate them, and some Google OAuth clients reject a repeated
        # scope parameter with ``invalid_scope``.  ZipperGen keeps the verified
        # grant separately in the private provider profile, so remove any
        # serialized scope hint before reconstructing the credential.
        serialized_scopes = info.pop("scopes", None)
        if serialized_scopes:
            if isinstance(serialized_scopes, str):
                recorded = tuple(serialized_scopes.split())
            elif isinstance(serialized_scopes, (tuple, list)):
                recorded = tuple(str(scope) for scope in serialized_scopes)
            else:
                raise TypeError("credential scopes must be a list or string")
            if not google_scopes_cover(recorded, requested):
                missing = ", ".join(
                    google_scope_names(
                        scope
                        for scope in requested
                        if not google_scopes_cover(recorded, (scope,))
                    )
                )
                raise ValueError(
                    "stored credential does not cover required scope(s): "
                    + missing
                )
        credentials = credentials_type.from_authorized_user_info(
            info,
        )
        if not credentials.valid:
            credentials.refresh(request_type())
    except Exception as exc:
        raise GoogleConnectorError(
            f"Google OAuth credential is unavailable: {exc}"
        ) from exc
    return credentials


def check_google_authorization(value: str, *, scopes: Iterable[str]) -> str:
    """Refresh an authorized-user credential and return normalized JSON."""

    return credentials_from_json(value, scopes=scopes).to_json()


__all__ = [
    "GOOGLE_GMAIL_MODIFY_SCOPE",
    "GOOGLE_GMAIL_READONLY_SCOPE",
    "GOOGLE_SCOPE_ALIASES",
    "GOOGLE_SHEETS_SCOPE",
    "GOOGLE_SHEETS_READONLY_SCOPE",
    "GoogleAuthorization",
    "GoogleConnectorError",
    "authorize_google",
    "authorize_google_client",
    "authorize_google_client_result",
    "check_google_authorization",
    "credentials_from_json",
    "decode_google_authorization",
    "encode_google_authorization",
    "google_authorization_summary",
    "google_imports",
    "google_scope_for_access",
    "google_scope_names",
    "google_scopes_cover",
    "google_scopes_for_access",
    "normalize_google_scopes",
    "normalize_google_client_json",
    "parse_google_scopes",
]
