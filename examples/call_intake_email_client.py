"""Gmail API client for examples/call_intake.py.

This client intentionally fetches unread messages without marking them read.
The workflow calls mark_processed only after it has ignored, recorded, or
replied to the message.

Required packages:
    pip install google-auth google-auth-oauthlib google-api-python-client

One-time auth setup:
    python examples/call_intake_email_client.py --setup

Default credential paths:
    ZIPPERGEN_CALL_GMAIL_CREDENTIALS  path to credentials.json
    ZIPPERGEN_CALL_GMAIL_TOKEN        OAuth2 token cache
"""

from __future__ import annotations

import base64
import email.mime.text
import os
import sys
from email.utils import getaddresses, parseaddr
from pathlib import Path
from typing import TypedDict


CREDENTIALS_PATH = Path(
    os.environ.get(
        "ZIPPERGEN_CALL_GMAIL_CREDENTIALS",
        os.environ.get("ZIPPERGEN_GMAIL_CREDENTIALS", str(Path.home() / ".zippergen_google_credentials.json")),
    )
)
TOKEN_PATH = Path(
    os.environ.get(
        "ZIPPERGEN_CALL_GMAIL_TOKEN",
        str(Path.home() / ".zippergen_call_gmail_token.json"),
    )
)

SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.compose",
]


class EmailMeta(TypedDict, total=False):
    id: str
    thread_id: str
    sender: str
    sender_email: str
    to: str
    cc: str
    delivered_to: str
    x_original_to: str
    envelope_to: str
    subject: str
    body: str
    message_id: str
    in_reply_to: str
    references: str


def _get_service():
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        sys.exit(
            "Gmail API libraries missing.\n"
            "Install with: pip install google-auth google-auth-oauthlib google-api-python-client"
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
                    "Download credentials.json from Google Cloud Console and set "
                    "ZIPPERGEN_CALL_GMAIL_CREDENTIALS."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_PATH.write_text(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def setup_auth() -> None:
    _get_service()
    print(f"Auth successful. Token saved to {TOKEN_PATH}")


def _extract_body(payload: dict) -> str:
    mime_type = payload.get("mimeType", "")
    if mime_type == "text/plain":
        raw = payload.get("body", {}).get("data", "")
        if raw:
            return base64.urlsafe_b64decode(raw).decode("utf-8", errors="replace")
    if mime_type == "text/html":
        raw = payload.get("body", {}).get("data", "")
        if raw:
            return base64.urlsafe_b64decode(raw).decode("utf-8", errors="replace")
    for part in payload.get("parts", []):
        text = _extract_body(part)
        if text:
            return text
    return ""


def _looks_like_reply_quote_header(line: str) -> bool:
    text = line.strip().lower()
    if not text:
        return False
    reply_openers = ("on ", "le ", "am ", "el ", "il ", "op ")
    reply_markers = ("wrote:", "a écrit", "schrieb", "escribió", "ha scritto")
    return text.startswith(reply_openers) and any(marker in text for marker in reply_markers)


def _strip_quoted_reply(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        if _looks_like_reply_quote_header(line):
            break
        if line.lstrip().startswith(">"):
            continue
        lines.append(line)
    stripped = "\n".join(lines).strip()
    return stripped or text.strip()


def _headers(payload: dict) -> dict[str, str]:
    return {
        str(header.get("name", "")).lower(): str(header.get("value", ""))
        for header in payload.get("headers", [])
    }


def count_unread() -> int:
    service = _get_service()
    query = os.environ.get("ZIPPERGEN_CALL_GMAIL_QUERY", "is:unread in:inbox")
    result = service.users().messages().list(
        userId="me",
        q=query,
        maxResults=10,
    ).execute()
    return int(result.get("resultSizeEstimate", 0) or 0)


def fetch_one_unread() -> EmailMeta | None:
    service = _get_service()
    query = os.environ.get("ZIPPERGEN_CALL_GMAIL_QUERY", "is:unread in:inbox")
    result = service.users().messages().list(
        userId="me",
        q=query,
        maxResults=1,
    ).execute()
    messages = result.get("messages", [])
    if not messages:
        return None

    msg_id = messages[0]["id"]
    msg = service.users().messages().get(
        userId="me",
        id=msg_id,
        format="full",
    ).execute()
    headers = _headers(msg["payload"])
    sender = headers.get("from", "")
    return EmailMeta(
        id=msg_id,
        thread_id=msg.get("threadId", ""),
        sender=sender,
        sender_email=parseaddr(sender)[1].lower(),
        to=headers.get("to", ""),
        cc=headers.get("cc", ""),
        delivered_to=headers.get("delivered-to", ""),
        x_original_to=headers.get("x-original-to", ""),
        envelope_to=headers.get("envelope-to", ""),
        subject=headers.get("subject", "(no subject)"),
        body=_strip_quoted_reply(_extract_body(msg["payload"])),
        message_id=headers.get("message-id", ""),
        in_reply_to=headers.get("in-reply-to", ""),
        references=headers.get("references", ""),
    )


def _looks_like_email_address(address: str) -> bool:
    local, sep, domain = address.partition("@")
    return bool(local and sep and domain)


def _configured_reply_to() -> str:
    raw = (
        os.environ.get("ZIPPERGEN_CALL_INTAKE_REPLY_TO")
        or os.environ.get("ZIPPERGEN_CALL_INTAKE_RECIPIENTS")
        or os.environ.get("ZIPPERGEN_CALL_INTAKE_ADDRESS")
        or ""
    )
    addresses = [
        address.strip().lower()
        for _name, address in getaddresses([raw])
        if _looks_like_email_address(address.strip().lower())
    ]
    return addresses[0] if addresses else ""


def _validated_reply_recipient(meta: dict) -> str:
    sender = str(meta.get("sender", "")).strip()
    parsed_sender = parseaddr(sender)[1].strip().lower()
    sender_email = str(meta.get("sender_email", "")).strip().lower()
    if not sender_email:
        sender_email = parsed_sender
    if not _looks_like_email_address(sender_email):
        raise ValueError("Refusing to send response: sender email address is not valid.")
    if parsed_sender and parsed_sender != sender_email:
        raise ValueError(
            "Refusing to send response: parsed sender address does not match "
            "the response recipient."
        )
    return sender_email


def _message_for_reply(meta: dict, subject: str, body: str) -> email.mime.text.MIMEText:
    msg = email.mime.text.MIMEText(body)
    msg["To"] = _validated_reply_recipient(meta)
    reply_to = _configured_reply_to()
    if reply_to:
        msg["Reply-To"] = reply_to
    msg["Subject"] = subject if subject.lower().startswith("re:") else f"Re: {subject}"
    original_message_id = meta.get("message_id")
    references = meta.get("references") or original_message_id
    if original_message_id:
        msg["In-Reply-To"] = original_message_id
    if references:
        msg["References"] = references
    return msg


def _raw_message(msg: email.mime.text.MIMEText) -> str:
    return base64.urlsafe_b64encode(msg.as_bytes()).decode()


def create_draft(meta: dict, subject: str, body: str) -> str:
    service = _get_service()
    msg = _message_for_reply(meta, subject, body)
    draft_body: dict = {"message": {"raw": _raw_message(msg)}}
    if meta.get("thread_id"):
        draft_body["message"]["threadId"] = meta["thread_id"]
    draft = service.users().drafts().create(userId="me", body=draft_body).execute()
    return str(draft.get("id", ""))


def send_email(meta: dict, subject: str, body: str) -> str:
    service = _get_service()
    msg = _message_for_reply(meta, subject, body)
    send_body: dict = {"raw": _raw_message(msg)}
    if meta.get("thread_id"):
        send_body["threadId"] = meta["thread_id"]
    sent = service.users().messages().send(userId="me", body=send_body).execute()
    return str(sent.get("id", ""))


def mark_processed(meta: dict) -> None:
    msg_id = meta.get("id") or meta.get("gmail_id") or meta.get("gmail-id")
    if not msg_id:
        return
    service = _get_service()
    service.users().messages().modify(
        userId="me",
        id=msg_id,
        body={"removeLabelIds": ["UNREAD"]},
    ).execute()


if __name__ == "__main__":
    if "--setup" in sys.argv:
        setup_auth()
    else:
        print(__doc__)
