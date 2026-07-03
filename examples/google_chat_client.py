"""Google Chat API wrapper for command_center.py.

Polls spaces the bot is a member of for new messages, then sends replies.

Required packages (same as other Google clients):
    pip install google-auth google-auth-oauthlib google-api-python-client

Bot setup:
    1. In Google Cloud Console, enable the Google Chat API.
    2. Create a Chat app and add it to a space.
    3. Run: python examples/google_chat_client.py --setup

Default credential paths (override via env vars):
    ZIPPERGEN_GCHAT_CREDENTIALS  — path to credentials.json
    ZIPPERGEN_GCHAT_TOKEN        — path where the OAuth2 token is cached
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TypedDict


CREDENTIALS_PATH = Path(
    os.environ.get("ZIPPERGEN_GCHAT_CREDENTIALS",
                   Path.home() / ".zippergen_google_credentials.json")
)
TOKEN_PATH = Path(
    os.environ.get("ZIPPERGEN_GCHAT_TOKEN",
                   Path.home() / ".zippergen_gchat_token.json")
)

SCOPES = [
    "https://www.googleapis.com/auth/chat.messages",
    "https://www.googleapis.com/auth/chat.spaces.readonly",
]

# In-memory set of message names already processed (resets on restart).
_seen_messages: set[str] = set()


class ChatMeta(TypedDict):
    name: str        # "spaces/AAAA/messages/BBBB"
    space: str       # "spaces/AAAA"
    space_name: str  # display name of the space
    sender: str      # sender display name
    text: str        # message body
    thread: str      # "spaces/AAAA/threads/CCCC" — for in-thread reply


def _get_service():
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise ImportError(
            "Install Google API packages: "
            "pip install google-auth google-auth-oauthlib google-api-python-client"
        ) from exc

    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_PATH), SCOPES
            )
            creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_text(creds.to_json())
    return build("chat", "v1", credentials=creds)


def _list_spaces() -> list[dict]:
    service = _get_service()
    result = service.spaces().list(pageSize=20).execute()
    return result.get("spaces", [])


def count_unread_messages() -> int:
    """Return number of unprocessed messages across all spaces (last 5 minutes)."""
    service = _get_service()
    since = (datetime.now(timezone.utc) - timedelta(minutes=5)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    count = 0
    for space in _list_spaces():
        result = service.spaces().messages().list(
            parent=space["name"],
            filter=f'createTime > "{since}"',
            pageSize=20,
        ).execute()
        for msg in result.get("messages", []):
            if msg.get("sender", {}).get("type") == "BOT":
                continue
            if msg["name"] not in _seen_messages:
                count += 1
    return count


def fetch_one_message() -> ChatMeta | None:
    """Return the oldest unprocessed message from any space, or None."""
    service = _get_service()
    since = (datetime.now(timezone.utc) - timedelta(minutes=5)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    for space in _list_spaces():
        space_name = space["name"]
        space_display = space.get("displayName", space_name)
        result = service.spaces().messages().list(
            parent=space_name,
            filter=f'createTime > "{since}"',
            pageSize=20,
            orderBy="createTime asc",
        ).execute()
        for msg in result.get("messages", []):
            if msg.get("sender", {}).get("type") == "BOT":
                continue
            if msg["name"] in _seen_messages:
                continue
            _seen_messages.add(msg["name"])
            sender = msg.get("sender", {}).get("displayName", "unknown")
            text = msg.get("text", msg.get("argumentText", "")).strip()
            thread = msg.get("thread", {}).get("name", "")
            return ChatMeta(
                name=msg["name"],
                space=space_name,
                space_name=space_display,
                sender=sender,
                text=text,
                thread=thread,
            )
    return None


def send_message(space: str, thread: str, text: str) -> str:
    """Send a reply into a space, threading when possible."""
    service = _get_service()
    body: dict = {"text": text}
    if thread:
        body["thread"] = {"name": thread}
    result = service.spaces().messages().create(
        parent=space,
        body=body,
        messageReplyOption="REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD" if thread else None,
    ).execute()
    msg_name = result.get("name", "?")
    print(f"[Chat] Sent: {msg_name}")
    return f"sent:{msg_name}"


def setup_auth() -> None:
    print("Opening browser for Google Chat OAuth2 authorisation …")
    _get_service()
    print(f"Token saved to {TOKEN_PATH}")


if __name__ == "__main__":
    if "--setup" in sys.argv:
        setup_auth()
    else:
        print("Run with --setup to authenticate.")
