"""Google Tasks API wrapper for command_center.py.

Creates tasks from email action items.

Required packages (same as other Google clients):
    pip install google-auth google-auth-oauthlib google-api-python-client

One-time auth setup:
    python examples/google_tasks_client.py --setup

Default credential paths (override via env vars):
    ZIPPERGEN_GTASKS_CREDENTIALS  — path to credentials.json
    ZIPPERGEN_GTASKS_TOKEN        — path where the OAuth2 token is cached
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


CREDENTIALS_PATH = Path(
    os.environ.get("ZIPPERGEN_GTASKS_CREDENTIALS",
                   Path.home() / ".zippergen_google_credentials.json")
)
TOKEN_PATH = Path(
    os.environ.get("ZIPPERGEN_GTASKS_TOKEN",
                   Path.home() / ".zippergen_gtasks_token.json")
)

SCOPES = ["https://www.googleapis.com/auth/tasks"]


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
    return build("tasks", "v1", credentials=creds)


def _default_tasklist() -> str:
    service = _get_service()
    result = service.tasklists().list(maxResults=1).execute()
    items = result.get("items", [])
    return items[0]["id"] if items else "@default"


def create_task(title: str, notes: str) -> str:
    """Create a task in the default task list and return a confirmation string."""
    service = _get_service()
    task_list = _default_tasklist()
    body: dict = {"title": title}
    if notes:
        body["notes"] = notes
    result = service.tasks().insert(tasklist=task_list, body=body).execute()
    task_id = result.get("id", "?")
    print(f"[Tasks] Created: '{title}' (id={task_id})")
    return f"task_created:{task_id}"


def setup_auth() -> None:
    print("Opening browser for Google Tasks OAuth2 authorisation …")
    _get_service()
    print(f"Token saved to {TOKEN_PATH}")


if __name__ == "__main__":
    if "--setup" in sys.argv:
        setup_auth()
    else:
        print("Run with --setup to authenticate.")
