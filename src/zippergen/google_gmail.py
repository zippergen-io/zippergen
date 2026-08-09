"""Gmail connector runtime support.

The workflow names a logical mailbox requirement. The project supplies the Google
account, search query, and private OAuth credential at run or deployment time.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import parseaddr
from html import unescape
from typing import Any
from urllib.parse import quote

from zippergen.google_auth import (
    GoogleConnectorError,
    credentials_from_json,
    google_imports,
    google_scope_for_access,
)


_CONNECTORS_ENV = "ZIPPERGEN_CONNECTORS_JSON"
_GMAIL_API = "https://gmail.googleapis.com/gmail/v1/users"


class GmailError(GoogleConnectorError):
    """A clear Gmail connector error suitable for command output."""


def _binding_records() -> dict[str, dict[str, object]]:
    raw = os.environ.get(_CONNECTORS_ENV, "")
    if not raw:
        raise GmailError(
            "No connector runtime configuration is active. Configure it with "
            "'zippergen connector configure NAME gmail', bind it with "
            "'zippergen connector bind REQUIREMENT NAME', then deploy."
        )
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GmailError("The connector runtime configuration is malformed.") from exc
    if not isinstance(value, dict):
        raise GmailError("The connector runtime configuration must be an object.")
    return {
        str(name): dict(record)
        for name, record in value.items()
        if isinstance(record, dict)
    }


def _requirement_binding(requirement: str) -> dict[str, object]:
    records = _binding_records()
    value = records.get(f"requirement:{requirement}") or records.get(requirement)
    if value is None:
        raise GmailError(f"Gmail connector requirement is not bound: {requirement}.")
    kind = str(value.get("kind") or "")
    if kind != "gmail":
        raise GmailError(
            f"Connector requirement {requirement!r} is bound to "
            f"{kind or 'an unknown connector'}, not gmail."
        )
    return value


def _decode_body(data: str) -> str:
    if not data:
        return ""
    padded = data + "=" * (-len(data) % 4)
    try:
        return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")
    except (ValueError, UnicodeError):
        return ""


def _payload_text(payload: dict[str, Any]) -> str:
    mime_type = str(payload.get("mimeType") or "")
    body = payload.get("body")
    if mime_type == "text/plain" and isinstance(body, dict):
        text = _decode_body(str(body.get("data") or ""))
        if text:
            return text
    parts = payload.get("parts")
    if isinstance(parts, list):
        for part in parts:
            if isinstance(part, dict):
                text = _payload_text(part)
                if text:
                    return text
    if mime_type == "text/html" and isinstance(body, dict):
        text = _decode_body(str(body.get("data") or ""))
        if text:
            import re

            return unescape(re.sub(r"<[^>]+>", " ", text))
    if isinstance(body, dict):
        return _decode_body(str(body.get("data") or ""))
    return ""


def _headers(payload: dict[str, Any]) -> dict[str, str]:
    raw = payload.get("headers")
    if not isinstance(raw, list):
        return {}
    result: dict[str, str] = {}
    for item in raw:
        if isinstance(item, dict) and item.get("name"):
            result[str(item["name"]).casefold()] = str(item.get("value") or "")
    return result


@dataclass
class GmailMailbox:
    """One configured Gmail account and search query."""

    requirement: str
    account: str
    query: str
    credential_json: str
    access: str = "read-write"

    @classmethod
    def from_requirement(cls, requirement: str) -> "GmailMailbox":
        binding = _requirement_binding(requirement)
        credential_env = str(binding.get("credential_env") or "")
        credential = os.environ.get(credential_env, "") if credential_env else ""
        if not credential:
            raise GmailError(
                f"Private Google credential is missing for {requirement!r}."
            )
        return cls(
            requirement=requirement,
            account=str(binding.get("account") or "me"),
            query=str(binding.get("query") or "is:unread in:inbox"),
            credential_json=credential,
            access=str(binding.get("access") or "read-only"),
        )

    def _session(self):
        session_type, _request, _credentials, _flow = google_imports()
        credentials = credentials_from_json(
            self.credential_json,
            scopes=(google_scope_for_access("gmail", self.access),),
        )
        return session_type(credentials)

    def _url(self, suffix: str) -> str:
        account = quote(self.account or "me", safe="")
        return f"{_GMAIL_API}/{account}/{suffix.lstrip('/')}"

    def _require_write(self, operation: str) -> None:
        if self.access == "read-only":
            raise GmailError(
                f"Gmail {operation} requires write access, but connector "
                f"{self.requirement!r} is read-only."
            )

    @staticmethod
    def _response_json(response, operation: str) -> dict[str, Any]:
        try:
            response.raise_for_status()
            value = response.json()
        except Exception as exc:
            detail = getattr(response, "text", "") or str(exc)
            raise GmailError(f"Gmail {operation} failed: {detail}") from exc
        if not isinstance(value, dict):
            raise GmailError(f"Gmail {operation} returned an invalid response.")
        return value

    def inspect(self) -> dict[str, object]:
        response = self._session().get(self._url("profile"), timeout=10)
        value = self._response_json(response, "configuration check")
        return {
            "email": str(value.get("emailAddress") or self.account),
            "messages": int(value.get("messagesTotal") or 0),
            "threads": int(value.get("threadsTotal") or 0),
        }

    def _list(self, *, maximum: int = 1) -> list[dict[str, Any]]:
        response = self._session().get(
            self._url("messages"),
            params={"q": self.query, "maxResults": maximum},
            timeout=20,
        )
        value = self._response_json(response, "message search")
        messages = value.get("messages") or []
        return [dict(item) for item in messages if isinstance(item, dict)]

    def count_unread(self) -> int:
        response = self._session().get(
            self._url("messages"),
            params={"q": self.query, "maxResults": 1},
            timeout=20,
        )
        value = self._response_json(response, "message search")
        return int(value.get("resultSizeEstimate") or 0)

    def fetch_one_unread(self) -> dict[str, str] | None:
        messages = self._list(maximum=1)
        if not messages:
            return None
        message_id = str(messages[0].get("id") or "")
        response = self._session().get(
            self._url(f"messages/{quote(message_id, safe='')}"),
            params={"format": "full"},
            timeout=20,
        )
        value = self._response_json(response, "message read")
        payload = value.get("payload")
        payload_record = payload if isinstance(payload, dict) else {}
        headers = _headers(payload_record)
        return {
            "id": str(value.get("id") or message_id),
            "gmail_id": str(value.get("id") or message_id),
            "thread_id": str(value.get("threadId") or ""),
            "message_id": headers.get("message-id", ""),
            "in_reply_to": headers.get("in-reply-to", ""),
            "references": headers.get("references", ""),
            "sender": headers.get("from", ""),
            "from": headers.get("from", ""),
            "to": headers.get("to", ""),
            "cc": headers.get("cc", ""),
            "delivered_to": headers.get("delivered-to", ""),
            "x_original_to": headers.get("x-original-to", ""),
            "envelope_to": headers.get("envelope-to", ""),
            "subject": headers.get("subject", ""),
            "body": _payload_text(payload_record),
        }

    @staticmethod
    def _recipient(meta: dict[str, object]) -> str:
        raw = str(meta.get("sender") or meta.get("from") or "")
        address = parseaddr(raw)[1].strip()
        if not address or "@" not in address:
            raise GmailError("The source message has no valid reply address.")
        return address

    @staticmethod
    def _raw_message(
        meta: dict[str, object],
        subject: str,
        body: str,
    ) -> dict[str, str]:
        message = EmailMessage()
        message["To"] = GmailMailbox._recipient(meta)
        reply_to = str(meta.get("response_reply_to") or "").strip()
        if reply_to:
            message["Reply-To"] = reply_to
        message["Subject"] = (
            subject if subject.casefold().startswith("re:") else f"Re: {subject}"
        )
        source_message_id = str(meta.get("message_id") or "").strip()
        if source_message_id:
            message["In-Reply-To"] = source_message_id
            message["References"] = source_message_id
        message.set_content(body)
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")
        return {"raw": raw}

    def create_draft(
        self,
        meta: dict[str, object],
        subject: str,
        body: str,
    ) -> str:
        self._require_write("draft creation")
        message = self._raw_message(meta, subject, body)
        thread_id = str(meta.get("thread_id") or "")
        if thread_id:
            message["threadId"] = thread_id
        response = self._session().post(
            self._url("drafts"),
            json={"message": message},
            timeout=20,
        )
        value = self._response_json(response, "draft creation")
        return str(value.get("id") or "")

    def send_email(
        self,
        meta: dict[str, object],
        subject: str,
        body: str,
    ) -> str:
        self._require_write("message sending")
        message = self._raw_message(meta, subject, body)
        thread_id = str(meta.get("thread_id") or "")
        if thread_id:
            message["threadId"] = thread_id
        response = self._session().post(
            self._url("messages/send"),
            json=message,
            timeout=20,
        )
        value = self._response_json(response, "message send")
        return str(value.get("id") or "")

    def mark_processed(self, meta: dict[str, object] | str) -> None:
        self._require_write("message update")
        message_id = (
            str(meta)
            if isinstance(meta, str)
            else str(meta.get("id") or meta.get("gmail_id") or "")
        )
        if not message_id:
            raise GmailError("Cannot mark a Gmail message without its ID.")
        response = self._session().post(
            self._url(f"messages/{quote(message_id, safe='')}/modify"),
            json={"removeLabelIds": ["UNREAD"]},
            timeout=20,
        )
        self._response_json(response, "message update")


__all__ = ["GmailError", "GmailMailbox"]
