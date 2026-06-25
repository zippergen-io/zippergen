"""Google Calendar API wrapper for command_center.py.

Handles OAuth2 auth, listing pending invites (RSVP needed), accepting /
declining events, checking slot availability, and proposing free slots.

Required packages (same as gmail_client.py):
    pip install google-auth google-auth-oauthlib google-api-python-client

One-time auth setup:
    python examples/google_calendar_client.py --setup

Default credential paths (override via env vars):
    ZIPPERGEN_GCAL_CREDENTIALS  — path to credentials.json (default: ~/.zippergen_google_credentials.json)
    ZIPPERGEN_GCAL_TOKEN        — path where the OAuth2 token is cached (separate from the Gmail token)
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TypedDict


# ---------------------------------------------------------------------------
# Credential paths
# ---------------------------------------------------------------------------

CREDENTIALS_PATH = Path(
    os.environ.get("ZIPPERGEN_GCAL_CREDENTIALS",
                   Path.home() / ".zippergen_google_credentials.json")
)
TOKEN_PATH = Path(
    os.environ.get("ZIPPERGEN_GCAL_TOKEN",
                   Path.home() / ".zippergen_gcal_token.json")
)

SCOPES = ["https://www.googleapis.com/auth/calendar"]


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _get_service():
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        sys.exit(
            "Google API libraries missing.\n"
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
                    "and set ZIPPERGEN_GCAL_CREDENTIALS or place the file at the default path."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_text(creds.to_json())

    return build("calendar", "v3", credentials=creds)


def setup_auth() -> None:
    """Run the OAuth2 flow and cache the token. Call once before first use."""
    _get_service()
    print(f"Auth successful. Token saved to {TOKEN_PATH}")


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class InviteMeta(TypedDict):
    id: str
    summary: str
    start: str
    end: str
    organizer: str
    description: str


class EventMeta(TypedDict):
    id: str
    summary: str
    start: str
    end: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_dt(dt_str: str | None, date_str: str | None) -> str:
    if dt_str:
        try:
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            return dt.strftime("%A %d %b %Y, %H:%M")
        except ValueError:
            return dt_str
    return date_str or "unknown"


def _busy_periods() -> list[tuple[datetime, datetime]]:
    service = _get_service()
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=7)
    result = service.freebusy().query(body={
        "timeMin": now.isoformat(),
        "timeMax": end.isoformat(),
        "items": [{"id": "primary"}],
    }).execute()
    out = []
    for b in result.get("calendars", {}).get("primary", {}).get("busy", []):
        s = datetime.fromisoformat(b["start"].replace("Z", "+00:00"))
        e = datetime.fromisoformat(b["end"].replace("Z", "+00:00"))
        out.append((s, e))
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def count_pending_invites() -> int:
    """Return the number of upcoming events where RSVP is still needed."""
    service = _get_service()
    now = datetime.now(timezone.utc).isoformat()
    result = service.events().list(
        calendarId="primary",
        timeMin=now,
        maxResults=50,
        singleEvents=True,
        orderBy="startTime",
    ).execute()
    return sum(
        1
        for event in result.get("items", [])
        for att in event.get("attendees", [])
        if att.get("self") and att.get("responseStatus") == "needsAction"
    )


def fetch_one_invite() -> InviteMeta | None:
    """Return the next upcoming event where RSVP is needed, or None."""
    service = _get_service()
    now = datetime.now(timezone.utc).isoformat()
    result = service.events().list(
        calendarId="primary",
        timeMin=now,
        maxResults=50,
        singleEvents=True,
        orderBy="startTime",
    ).execute()
    for event in result.get("items", []):
        for att in event.get("attendees", []):
            if att.get("self") and att.get("responseStatus") == "needsAction":
                start = event.get("start", {})
                end   = event.get("end",   {})
                return InviteMeta(
                    id=event["id"],
                    summary=event.get("summary", "(no title)"),
                    start=_fmt_dt(start.get("dateTime"), start.get("date")),
                    end=_fmt_dt(end.get("dateTime"), end.get("date")),
                    organizer=event.get("organizer", {}).get("email", "unknown"),
                    description=(event.get("description") or "").strip()[:400],
                )
    return None


def count_upcoming_meetings(window_minutes: int = 30) -> int:
    """Return number of accepted meetings starting within the next window_minutes."""
    service = _get_service()
    now = datetime.now(timezone.utc)
    soon = now + timedelta(minutes=window_minutes)
    result = service.events().list(
        calendarId="primary",
        timeMin=now.isoformat(),
        timeMax=soon.isoformat(),
        singleEvents=True,
        orderBy="startTime",
    ).execute()
    count = 0
    for event in result.get("items", []):
        attendees = event.get("attendees", [])
        if not attendees:
            count += 1  # events with no attendees list are implicitly accepted
            continue
        for att in attendees:
            if att.get("self") and att.get("responseStatus") in ("accepted", "tentative"):
                count += 1
                break
    return count


def fetch_next_meeting(window_minutes: int = 30) -> InviteMeta | None:
    """Return the next accepted meeting starting within window_minutes, or None."""
    service = _get_service()
    now = datetime.now(timezone.utc)
    soon = now + timedelta(minutes=window_minutes)
    result = service.events().list(
        calendarId="primary",
        timeMin=now.isoformat(),
        timeMax=soon.isoformat(),
        singleEvents=True,
        orderBy="startTime",
    ).execute()
    for event in result.get("items", []):
        attendees = event.get("attendees", [])
        accepted = not attendees or any(
            att.get("self") and att.get("responseStatus") in ("accepted", "tentative")
            for att in attendees
        )
        if not accepted:
            continue
        start = event.get("start", {})
        end   = event.get("end",   {})
        all_attendees = [
            att.get("email", "") for att in attendees if not att.get("self")
        ]
        return InviteMeta(
            id=event["id"],
            summary=event.get("summary", "(no title)"),
            start=_fmt_dt(start.get("dateTime"), start.get("date")),
            end=_fmt_dt(end.get("dateTime"), end.get("date")),
            organizer=event.get("organizer", {}).get("email", "unknown"),
            description=(
                ("Attendees: " + ", ".join(all_attendees) if all_attendees else "")
                + ("\n\n" + event.get("description", "").strip()[:400]
                   if event.get("description") else "")
            ).strip(),
        )
    return None


def find_events(query: str, days_ahead: int = 30) -> list[EventMeta]:
    """Return upcoming events whose title or description matches query."""
    service = _get_service()
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=days_ahead)
    result = service.events().list(
        calendarId="primary",
        timeMin=now.isoformat(),
        timeMax=end.isoformat(),
        q=query,
        singleEvents=True,
        orderBy="startTime",
        maxResults=5,
    ).execute()
    out = []
    for event in result.get("items", []):
        start = event.get("start", {})
        end_e = event.get("end", {})
        out.append(EventMeta(
            id=event["id"],
            summary=event.get("summary", "(no title)"),
            start=_fmt_dt(start.get("dateTime"), start.get("date")),
            end=_fmt_dt(end_e.get("dateTime"), end_e.get("date")),
        ))
    return out


def delete_event(event_id: str) -> None:
    """Permanently delete a calendar event by ID."""
    service = _get_service()
    service.events().delete(calendarId="primary", eventId=event_id).execute()
    print(f"[Calendar] Deleted event: {event_id}")


def accept_event(event_id: str) -> None:
    """Set the authenticated user's RSVP to accepted and notify organizer."""
    service = _get_service()
    event = service.events().get(calendarId="primary", eventId=event_id).execute()
    for att in event.get("attendees", []):
        if att.get("self"):
            att["responseStatus"] = "accepted"
    service.events().patch(
        calendarId="primary",
        eventId=event_id,
        sendUpdates="all",
        body={"attendees": event["attendees"]},
    ).execute()


def decline_event(event_id: str) -> None:
    """Set the authenticated user's RSVP to declined and notify organizer."""
    service = _get_service()
    event = service.events().get(calendarId="primary", eventId=event_id).execute()
    for att in event.get("attendees", []):
        if att.get("self"):
            att["responseStatus"] = "declined"
    service.events().patch(
        calendarId="primary",
        eventId=event_id,
        sendUpdates="all",
        body={"attendees": event["attendees"]},
    ).execute()


def check_slot(slot_hint: str) -> str:
    """
    Check the next 7 days for conflicts and return a human-readable status.
    slot_hint is passed through for context (e.g. the email proposing a time).
    """
    busy = _busy_periods()
    if not busy:
        return "Calendar is clear for the next 7 days."
    desc = "; ".join(
        f"{s.strftime('%a %d %b %H:%M')}–{e.strftime('%H:%M')}"
        for s, e in busy[:6]
    )
    return f"Busy periods in the next 7 days: {desc}."


def list_free_slots() -> str:
    """
    Return up to 6 free 1-hour slots during working hours (09:00–17:00,
    Mon–Fri) over the next 7 days, using the system's local timezone.
    """
    busy = _busy_periods()  # UTC datetimes

    # Generate candidates in local time so working-hours check is correct
    local_tz = datetime.now().astimezone().tzinfo
    now_local = datetime.now(local_tz)
    candidate = now_local.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

    free: list[str] = []
    while candidate < now_local + timedelta(days=7) and len(free) < 6:
        if candidate.weekday() < 5 and 9 <= candidate.hour < 17:
            slot_end = candidate + timedelta(hours=1)
            # Compare against busy periods in UTC
            cand_utc = candidate.astimezone(timezone.utc)
            end_utc  = slot_end.astimezone(timezone.utc)
            if not any(s < end_utc and e > cand_utc for s, e in busy):
                free.append(candidate.strftime("%A %d %b, %H:%M"))
        candidate += timedelta(hours=1)
        if candidate.hour >= 17:
            candidate = (candidate + timedelta(days=1)).replace(
                hour=9, minute=0, second=0, microsecond=0
            )
            while candidate.weekday() >= 5:
                candidate += timedelta(days=1)

    return "\n".join(free) if free else "No free slots found in the next 7 days."


def create_event(summary: str, start_iso: str, end_iso: str, attendee_email: str) -> str:
    """Create a calendar event in local time and return its event ID."""
    local_tz = datetime.now().astimezone().tzinfo

    def _localize(iso: str) -> str:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=local_tz)
        return dt.isoformat()

    service = _get_service()
    body: dict = {
        "summary": summary,
        "start": {"dateTime": _localize(start_iso)},
        "end":   {"dateTime": _localize(end_iso)},
    }
    if attendee_email:
        body["attendees"] = [{"email": attendee_email}]
    result = service.events().insert(
        calendarId="primary",
        body=body,
        sendUpdates="all" if attendee_email else "none",
    ).execute()
    print(f"[Calendar] Created event '{summary}' ({result['id']})")
    return result["id"]


# ---------------------------------------------------------------------------
# CLI — one-time setup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--setup" in sys.argv:
        setup_auth()
    else:
        print(__doc__)
