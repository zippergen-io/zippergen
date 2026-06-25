"""Gmail API wrapper for personal_assistant.py.

Handles OAuth2 auth, fetching unread emails, marking spam, and creating drafts.

Required packages:
    pip install google-auth google-auth-oauthlib google-api-python-client

One-time auth setup:
    python examples/gmail_client.py --setup

Default credential paths (override via env vars):
    ZIPPERGEN_GMAIL_CREDENTIALS  — path to credentials.json (default: ~/.zippergen_google_credentials.json)
    ZIPPERGEN_GMAIL_TOKEN        — path where the OAuth2 token is cached (auto-created)
"""

from __future__ import annotations

import base64
import email as _email_lib
import email.mime.text
import json
import os
import sys
from pathlib import Path
from typing import TypedDict


# ---------------------------------------------------------------------------
# Credential paths
# ---------------------------------------------------------------------------

CREDENTIALS_PATH = Path(
    os.environ.get("ZIPPERGEN_GMAIL_CREDENTIALS",
                   Path.home() / ".zippergen_google_credentials.json")
)
TOKEN_PATH = Path(
    os.environ.get("ZIPPERGEN_GMAIL_TOKEN",
                   Path.home() / ".zippergen_gmail_token.json")
)

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _get_service():
    """Return an authenticated Gmail API service."""
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        sys.exit(
            "Gmail API libraries missing.\n"
            "Install with:  pip install google-auth google-auth-oauthlib google-api-python-client"
        )

    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None
        if not creds or not creds.valid:
            if not CREDENTIALS_PATH.exists():
                sys.exit(
                    f"OAuth2 credentials not found at {CREDENTIALS_PATH}\n"
                    "Download from Google Cloud Console → APIs & Services → Credentials\n"
                    "and set ZIPPERGEN_GMAIL_CREDENTIALS or place the file at the default path."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_text(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def setup_auth() -> None:
    """Run the OAuth2 flow and cache the token. Call once before first use."""
    _get_service()
    print(f"Auth successful. Token saved to {TOKEN_PATH}")


# ---------------------------------------------------------------------------
# Email metadata
# ---------------------------------------------------------------------------

class EmailMeta(TypedDict):
    id: str
    sender: str
    subject: str
    body: str


def _extract_body(payload: dict) -> str:
    """Return plain-text body from a Gmail message payload."""
    mime_type = payload.get("mimeType", "")
    if mime_type == "text/plain":
        raw = payload["body"].get("data", "")
        return base64.urlsafe_b64decode(raw).decode("utf-8", errors="replace")
    for part in payload.get("parts", []):
        text = _extract_body(part)
        if text:
            return text
    return ""


def _format_email(meta: EmailMeta) -> str:
    """Return a text representation suitable for LLM prompts."""
    return f"From: {meta['sender']}\nSubject: {meta['subject']}\n\n{meta['body']}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def count_unread() -> int:
    """Return the number of unread messages in the inbox."""
    service = _get_service()
    result = service.users().messages().list(
        userId="me", q="is:unread in:inbox", maxResults=100
    ).execute()
    return result.get("resultSizeEstimate", 0)


def fetch_one() -> EmailMeta | None:
    """Fetch the oldest unread message, mark it as read, and return its metadata.

    Returns None if the inbox is empty.
    """
    service = _get_service()
    result = service.users().messages().list(
        userId="me", q="is:unread in:inbox", maxResults=1
    ).execute()
    messages = result.get("messages", [])
    if not messages:
        return None

    msg_id = messages[0]["id"]
    msg = service.users().messages().get(
        userId="me", id=msg_id, format="full"
    ).execute()

    headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
    body = _extract_body(msg["payload"]).strip()

    # Mark as read immediately so we don't pick it up again.
    service.users().messages().modify(
        userId="me", id=msg_id, body={"removeLabelIds": ["UNREAD"]}
    ).execute()

    return EmailMeta(
        id=msg_id,
        sender=headers.get("From", ""),
        subject=headers.get("Subject", "(no subject)"),
        body=body[:2000],  # cap for LLM context
    )


def mark_as_spam(msg_id: str) -> None:
    """Move a message to the spam folder."""
    service = _get_service()
    service.users().messages().modify(
        userId="me",
        id=msg_id,
        body={"addLabelIds": ["SPAM"], "removeLabelIds": ["INBOX"]},
    ).execute()


def create_draft(sender: str, subject: str, reply_body: str) -> str:
    """Create a Gmail draft reply. Returns the draft ID."""
    msg = email.mime.text.MIMEText(reply_body)
    msg["To"] = sender
    msg["Subject"] = f"Re: {subject}"
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()

    service = _get_service()
    draft = service.users().drafts().create(
        userId="me", body={"message": {"raw": raw}}
    ).execute()
    return draft["id"]


# ---------------------------------------------------------------------------
# CLI — one-time setup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--setup" in sys.argv:
        setup_auth()
    else:
        print(__doc__)
