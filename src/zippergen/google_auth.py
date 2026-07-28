"""Shared OAuth support for Google connector providers."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path


GOOGLE_SHEETS_READONLY_SCOPE = (
    "https://www.googleapis.com/auth/spreadsheets.readonly"
)
GOOGLE_SHEETS_SCOPE = "https://www.googleapis.com/auth/spreadsheets"
GOOGLE_GMAIL_READONLY_SCOPE = (
    "https://www.googleapis.com/auth/gmail.readonly"
)
GOOGLE_GMAIL_MODIFY_SCOPE = "https://www.googleapis.com/auth/gmail.modify"


class GoogleConnectorError(RuntimeError):
    """A clear Google connector error suitable for Studio output."""


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


def authorize_google(
    credentials_file: str | Path,
    *,
    scopes: Iterable[str],
    open_browser: bool = True,
) -> str:
    """Run Google's desktop OAuth flow and return authorized-user JSON."""

    requested = normalize_google_scopes(scopes)
    _session, _request, _credentials, flow_type = google_imports()
    path = Path(credentials_file).expanduser().resolve()
    if not path.is_file():
        raise GoogleConnectorError(
            f"Google OAuth desktop credentials file does not exist: {path}"
        )
    try:
        flow = flow_type.from_client_secrets_file(str(path), scopes=list(requested))
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
    return credentials.to_json()


def credentials_from_json(value: str, *, scopes: Iterable[str]):
    requested = normalize_google_scopes(scopes)
    _session, request_type, credentials_type, _flow = google_imports()
    try:
        info = json.loads(value)
        if not isinstance(info, dict):
            raise TypeError("credential JSON must be an object")
        credentials = credentials_type.from_authorized_user_info(
            info,
            scopes=list(requested),
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
    "GOOGLE_SHEETS_SCOPE",
    "GOOGLE_SHEETS_READONLY_SCOPE",
    "GoogleConnectorError",
    "authorize_google",
    "check_google_authorization",
    "credentials_from_json",
    "google_imports",
    "google_scope_for_access",
    "google_scopes_cover",
    "google_scopes_for_access",
    "normalize_google_scopes",
]
