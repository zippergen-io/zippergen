import json
import base64

import pytest

from zippergen.google_gmail import GmailError, GmailMailbox


class _Response:
    def __init__(self, value):
        self.value = value
        self.text = json.dumps(value)

    def raise_for_status(self):
        return None

    def json(self):
        return self.value


class _Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(("get", url, kwargs))
        return _Response(next(self.responses))

    def post(self, url, **kwargs):
        self.calls.append(("post", url, kwargs))
        return _Response(next(self.responses))


def _runtime_environment(monkeypatch):
    credential_env = "ZIPPERGEN_CONNECTOR_CALL_MAILBOX_GOOGLE_CREDENTIAL"
    monkeypatch.setenv(credential_env, '{"refresh_token":"private"}')
    monkeypatch.setenv(
        "ZIPPERGEN_CONNECTORS_JSON",
        json.dumps({
            "requirement:call-mailbox": {
                "kind": "gmail",
                "provider": "google",
                "account": "me",
                "query": "is:unread label:Calls",
                "credential_env": credential_env,
            }
        }),
    )


def test_gmail_requirement_resolves_private_runtime_binding(monkeypatch):
    _runtime_environment(monkeypatch)

    mailbox = GmailMailbox.from_requirement("call-mailbox")

    assert mailbox.account == "me"
    assert mailbox.query == "is:unread label:Calls"
    assert mailbox.credential_json == '{"refresh_token":"private"}'
    assert mailbox.access == "read-only"


def test_gmail_requirement_fails_clearly_without_runtime_binding(monkeypatch):
    monkeypatch.delenv("ZIPPERGEN_CONNECTORS_JSON", raising=False)

    with pytest.raises(GmailError, match="No connector runtime"):
        GmailMailbox.from_requirement("call-mailbox")


def test_gmail_raw_reply_keeps_thread_headers():
    encoded = GmailMailbox._raw_message(
        {
            "sender": "Alice <alice@example.com>",
            "message_id": "<source@example.com>",
        },
        "Review",
        "Thanks",
    )

    raw = base64.urlsafe_b64decode(encoded["raw"]).decode()
    assert "To: alice@example.com" in raw
    assert "In-Reply-To: <source@example.com>" in raw
    assert "References: <source@example.com>" in raw


def test_mark_processed_requires_a_stable_message_id(monkeypatch):
    mailbox = GmailMailbox(
        requirement="call-mailbox",
        account="me",
        query="is:unread",
        credential_json="private",
    )

    with pytest.raises(GmailError, match="without its ID"):
        mailbox.mark_processed({})


def test_read_only_gmail_binding_blocks_modification():
    mailbox = GmailMailbox(
        requirement="call-mailbox",
        account="me",
        query="is:unread",
        credential_json="private",
        access="read-only",
    )

    with pytest.raises(GmailError, match="read-only"):
        mailbox.mark_processed({"id": "gmail-1"})
    with pytest.raises(GmailError, match="read-only"):
        mailbox.create_draft(
            {"sender": "alice@example.com"},
            "Review",
            "Thanks",
        )


def test_gmail_reads_one_message_without_marking_it_processed(monkeypatch):
    body = base64.urlsafe_b64encode(b"Call details").decode()
    session = _Session([
        {"messages": [{"id": "gmail-1"}], "resultSizeEstimate": 1},
        {
            "id": "gmail-1",
            "threadId": "thread-1",
            "payload": {
                "mimeType": "text/plain",
                "headers": [
                    {"name": "From", "value": "Alice <alice@example.com>"},
                    {"name": "To", "value": "calls@example.com"},
                    {"name": "Subject", "value": "New call"},
                    {"name": "Message-ID", "value": "<source@example.com>"},
                ],
                "body": {"data": body},
            },
        },
    ])
    mailbox = GmailMailbox(
        requirement="call-mailbox",
        account="me",
        query="is:unread label:Calls",
        credential_json="private",
    )
    monkeypatch.setattr(mailbox, "_session", lambda: session)

    message = mailbox.fetch_one_unread()

    assert message is not None
    assert message["id"] == "gmail-1"
    assert message["sender"] == "Alice <alice@example.com>"
    assert message["body"] == "Call details"
    assert all("/modify" not in call[1] for call in session.calls)


def test_gmail_marks_the_exact_message_processed(monkeypatch):
    session = _Session([{"id": "gmail-1"}])
    mailbox = GmailMailbox(
        requirement="call-mailbox",
        account="me",
        query="is:unread",
        credential_json="private",
    )
    monkeypatch.setattr(mailbox, "_session", lambda: session)

    mailbox.mark_processed({"id": "gmail-1"})

    method, url, kwargs = session.calls[0]
    assert method == "post"
    assert url.endswith("/messages/gmail-1/modify")
    assert kwargs["json"] == {"removeLabelIds": ["UNREAD"]}
