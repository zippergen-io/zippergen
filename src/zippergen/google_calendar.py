"""Google Calendar events through a configured, privately authorized connector."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import quote

from zippergen.connectors import requirement_binding
from zippergen.google_auth import (
    GoogleConnectorError, credentials_from_json, google_imports, google_scope_for_access,
)

__all__ = ["GoogleCalendar", "GoogleCalendarError"]

_API = "https://www.googleapis.com/calendar/v3/calendars"
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})")
_RECEIPT = "zippergenRequestHash"


class GoogleCalendarError(GoogleConnectorError):
    """A calendar operation could not be established as successful."""


def _interval(start: str, end: str) -> None:
    def parse(value):
        if not isinstance(value, str) or not _TIMESTAMP.fullmatch(value):
            raise ValueError("Calendar times must be RFC3339 timestamps with an explicit UTC offset.")
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    if parse(end) <= parse(start):
        raise ValueError("Calendar end must be after start.")


@dataclass
class GoogleCalendar:
    """Read events and create single timed events on one configured calendar.

    A stable request ID identifies one creation, including across process restarts.
    Keep it with the approved proposal. Reusing it for different content is an error.
    This uses Google's caller-supplied event ID, not a local receipt database.
    """

    requirement: str
    calendar_id: str
    credential_json: str = field(repr=False)
    access: str = "read-only"

    def __post_init__(self):
        if not self.calendar_id.strip():
            raise ValueError("A calendar ID is required.")
        if self.access not in {"read-only", "write", "read-write"}:
            raise ValueError("Invalid calendar access level.")

    @classmethod
    def from_requirement(cls, requirement: str) -> "GoogleCalendar":
        binding = requirement_binding(requirement, kind="google-calendar", error=GoogleCalendarError)
        credential = os.environ.get(str(binding.get("credential_env") or ""), "")
        if not credential:
            raise GoogleCalendarError(f"Private Google credential is missing for {requirement!r}.")
        return cls(requirement, str(binding.get("calendar_id") or ""), credential,
                   str(binding.get("access") or "read-only"))

    def _session(self):
        session_type, _request, _credentials, _flow = google_imports()
        return session_type(credentials_from_json(
            self.credential_json, scopes=(google_scope_for_access("google-calendar", self.access),),
        ))

    def _url(self, event_id: str | None = None) -> str:
        url = f"{_API}/{quote(self.calendar_id, safe='')}/events"
        return url if event_id is None else f"{url}/{quote(event_id, safe='')}"

    @staticmethod
    def _request(session, method, url, **kwargs):
        try:
            return getattr(session, method)(url, timeout=20, **kwargs)
        except Exception:
            # Exception text can contain authorization headers or event contents.
            raise GoogleCalendarError(
                "Google Calendar could not be reached. A write may have succeeded. "
                "Retry with the same request ID and proposal."
            ) from None

    @staticmethod
    def _json(response, operation: str) -> dict[str, Any]:
        if not 200 <= response.status_code < 300:
            raise GoogleCalendarError(
                f"Google Calendar {operation} failed (HTTP {response.status_code}). "
                "Check calendar access and Google authorization."
            )
        try:
            value = response.json()
        except (ValueError, TypeError):
            raise GoogleCalendarError(f"Google Calendar {operation} returned invalid JSON.") from None
        if not isinstance(value, dict):
            raise GoogleCalendarError(f"Google Calendar {operation} returned an invalid response.")
        return value

    def inspect(self) -> dict[str, object]:
        # Events scopes suffice. calendars.get would need a broader OAuth scope.
        value = self._json(self._request(self._session(), "get", self._url(),
                                        params={"maxResults": 1, "fields": "summary,timeZone"}),
                           "configuration check")
        return {"calendar_id": self.calendar_id, "title": value.get("summary", self.calendar_id),
                "time_zone": value.get("timeZone", "")}

    def list_events(self, start: str, end: str, *, max_events: int = 10000) -> list[dict[str, Any]]:
        """Return events overlapping [start, end), expanding recurring instances.

        All-day events retain Google's date fields. This is an observation, not
        a reservation. Fail rather than silently truncate a busy calendar.
        """
        _interval(start, end)
        if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events < 1:
            raise ValueError("max_events must be a positive integer.")
        params = {"timeMin": start, "timeMax": end, "singleEvents": "true",
                  "orderBy": "startTime", "showDeleted": "false", "maxResults": 250}
        events = []
        seen = set()
        session = self._session()
        while True:
            value = self._json(self._request(session, "get", self._url(), params=dict(params)), "read")
            items = value.get("items", [])
            if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
                raise GoogleCalendarError("Google Calendar returned malformed events.")
            events.extend(items)
            if len(events) > max_events:
                raise GoogleCalendarError("Calendar result exceeds max_events. Use a smaller time range.")
            token = value.get("nextPageToken")
            if not token:
                return events
            if not isinstance(token, str) or token in seen:
                raise GoogleCalendarError("Google Calendar returned an invalid page token.")
            seen.add(token)
            params["pageToken"] = token

    def create_event(
        self, request_id: str, *, summary: str, start: str, end: str,
        description: str = "", location: str = "", attendees: tuple[str, ...] = (),
    ) -> str:
        """Create one event and return its stable Google event ID.

        Attendees receive invitations (sendUpdates=all). A recovered creation
        returns its ID without sending another insert. No availability lock or
        exactly-once notification delivery is promised by this operation.
        """
        if self.access == "read-only":
            raise GoogleCalendarError("Calendar event creation requires write access. This connector is read-only.")
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("Calendar creation needs a stable, non-empty request ID.")
        _interval(start, end)
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("Calendar event summary must not be empty.")
        if not isinstance(description, str) or not isinstance(location, str):
            raise ValueError("Calendar description and location must be strings.")
        if not isinstance(attendees, (tuple, list)) or any(
            not isinstance(email, str) or not re.fullmatch(r"[^\s@]+@[^\s@]+", email)
            for email in attendees
        ):
            raise ValueError("Calendar attendees must be a sequence of email addresses.")
        payload = {"summary": summary, "start": {"dateTime": start},
                   "end": {"dateTime": end}, "description": description, "location": location,
                   "attendees": [{"email": email} for email in sorted(set(attendees))]}
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        event_id = hashlib.sha256(("zippergen-calendar-v1:" + request_id).encode()).hexdigest()
        session = self._session()

        def recover(response):
            value = self._json(response, "creation recovery")
            properties = value.get("extendedProperties") or {}
            private = properties.get("private", {}) if isinstance(properties, dict) else {}
            if (value.get("id") != event_id or value.get("status") == "cancelled"
                    or not isinstance(private, dict) or private.get(_RECEIPT) != fingerprint):
                raise GoogleCalendarError(
                    "Calendar request ID already exists with different content or was cancelled. "
                    "Review the existing event before starting a new request."
                )
            return event_id

        existing = self._request(session, "get", self._url(event_id))
        if existing.status_code != 404:
            return recover(existing)
        payload["id"] = event_id
        payload["extendedProperties"] = {"private": {_RECEIPT: fingerprint}}
        response = self._request(session, "post", self._url(), json=payload,
                                 params={"sendUpdates": "all" if attendees else "none"})
        if response.status_code == 409:
            return recover(self._request(session, "get", self._url(event_id)))
        value = self._json(response, "creation")
        if value.get("id") != event_id:
            raise GoogleCalendarError("Calendar creation returned an unexpected event ID. Retry the same request.")
        return event_id
