import json
import traceback

import pytest

from zippergen.google_auth import GoogleAuthorization, credentials_from_json
from zippergen.google_calendar import GoogleCalendar
from zippergen.google_gmail import GmailMailbox
from zippergen.google_sheets import GoogleSheetsTable


PRIVATE = "private-refresh-token-and-message-content"


@pytest.fixture(params=[GmailMailbox, GoogleSheetsTable, GoogleCalendar])
def connector(request):
    if request.param is GmailMailbox:
        return GmailMailbox("mail", "me", "is:unread", PRIVATE)
    if request.param is GoogleSheetsTable:
        return GoogleSheetsTable("table", "sheet", "Inbox", PRIVATE)
    return GoogleCalendar("agenda", "primary", PRIVATE)


def test_connector_repr_does_not_print_oauth_credentials(connector):
    assert PRIVATE not in repr(connector)


def test_handoff_repr_does_not_print_oauth_credentials():
    authorization = GoogleAuthorization(PRIVATE, ("scope",), "client")
    assert PRIVATE not in repr(authorization)


@pytest.mark.parametrize("failure", ["http", "json", "transport"])
def test_connector_errors_hide_upstream_contents(connector, monkeypatch, failure):
    class Response:
        status_code = 403 if failure == "http" else 200
        text = PRIVATE

        def raise_for_status(self):
            if failure == "http":
                raise RuntimeError(PRIVATE)

        def json(self):
            raise ValueError(PRIVATE)

    class Session:
        def get(self, *args, **kwargs):
            if failure == "transport":
                raise RuntimeError(PRIVATE)
            return Response()

    monkeypatch.setattr(connector, "_session", Session)
    with pytest.raises(RuntimeError) as caught:
        connector.inspect()
    assert PRIVATE not in str(caught.value)
    assert PRIVATE not in "".join(traceback.format_exception(caught.value))
    if failure == "http":
        assert "403" in str(caught.value)


@pytest.mark.parametrize("stage", ["load", "refresh"])
def test_oauth_library_errors_hide_secrets(monkeypatch, stage):
    class Credentials:
        valid = False

        @classmethod
        def from_authorized_user_info(cls, info):
            if stage == "load":
                raise ValueError(PRIVATE)
            return cls()

        def refresh(self, request):
            raise RuntimeError(PRIVATE)

    monkeypatch.setattr("zippergen.google_auth.google_imports",
                        lambda: (object, object, Credentials, object))
    with pytest.raises(RuntimeError) as caught:
        credentials_from_json(json.dumps({"refresh_token": PRIVATE}),
                              scopes=("https://www.googleapis.com/auth/gmail.readonly",))
    assert PRIVATE not in str(caught.value)
    assert PRIVATE not in "".join(traceback.format_exception(caught.value))


def test_browser_authorization_errors_hide_secrets(monkeypatch):
    from zippergen.google_auth import authorize_google_client_result

    class Flow:
        @classmethod
        def from_client_config(cls, *args, **kwargs):
            raise RuntimeError(PRIVATE)

    monkeypatch.setattr("zippergen.google_auth.google_imports",
                        lambda: (object, object, object, Flow))
    client = {"installed": {"client_id": "client", "client_secret": PRIVATE,
                            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                            "token_uri": "https://oauth2.googleapis.com/token"}}
    with pytest.raises(RuntimeError) as caught:
        authorize_google_client_result(json.dumps(client), scopes=("scope",))
    assert PRIVATE not in "".join(traceback.format_exception(caught.value))
