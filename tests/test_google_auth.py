import json

import pytest

from zippergen.google_auth import (
    GOOGLE_GMAIL_MODIFY_SCOPE,
    GOOGLE_GMAIL_READONLY_SCOPE,
    GOOGLE_SHEETS_READONLY_SCOPE,
    GOOGLE_SHEETS_SCOPE,
    GoogleAuthorization,
    decode_google_authorization,
    encode_google_authorization,
    google_scope_for_access,
    google_scope_names,
    google_scopes_cover,
    google_scopes_for_access,
    normalize_google_client_json,
    parse_google_scopes,
)


def test_google_scopes_follow_connector_access():
    assert google_scope_for_access("gmail", "read-only") == (
        GOOGLE_GMAIL_READONLY_SCOPE
    )
    assert google_scope_for_access("gmail", "write") == (
        GOOGLE_GMAIL_MODIFY_SCOPE
    )
    assert google_scope_for_access("google-sheets", "read-only") == (
        GOOGLE_SHEETS_READONLY_SCOPE
    )
    assert google_scope_for_access("google-sheets", "read-write") == (
        GOOGLE_SHEETS_SCOPE
    )


def test_google_scope_plan_keeps_only_the_strongest_scope_per_service():
    scopes = google_scopes_for_access((
        ("gmail", "read-only"),
        ("gmail", "read-write"),
        ("google-sheets", "read-only"),
    ))

    assert scopes == tuple(sorted((
        GOOGLE_GMAIL_MODIFY_SCOPE,
        GOOGLE_SHEETS_READONLY_SCOPE,
    )))


def test_broad_google_scopes_cover_read_only_requirements():
    assert google_scopes_cover(
        (GOOGLE_GMAIL_MODIFY_SCOPE, GOOGLE_SHEETS_SCOPE),
        (GOOGLE_GMAIL_READONLY_SCOPE, GOOGLE_SHEETS_READONLY_SCOPE),
    )
    assert not google_scopes_cover(
        (GOOGLE_GMAIL_READONLY_SCOPE,),
        (GOOGLE_GMAIL_MODIFY_SCOPE,),
    )


def test_unknown_google_scope_inputs_are_rejected():
    with pytest.raises(ValueError, match="connector access"):
        google_scope_for_access("gmail", "admin")
    with pytest.raises(ValueError, match="connector kind"):
        google_scope_for_access("google-drive", "read-only")


def test_google_desktop_client_json_is_validated_and_normalized():
    normalized = normalize_google_client_json(
        """
        {
          "installed": {
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_secret": "secret",
            "client_id": "example.apps.googleusercontent.com",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth"
          }
        }
        """
    )

    assert normalized == (
        '{"installed":{"auth_uri":"https://accounts.google.com/o/oauth2/auth",'
        '"client_id":"example.apps.googleusercontent.com",'
        '"client_secret":"secret",'
        '"token_uri":"https://oauth2.googleapis.com/token"}}'
    )


def test_google_client_json_must_be_a_desktop_app():
    with pytest.raises(
        RuntimeError,
        match="must describe a Desktop app",
    ):
        normalize_google_client_json('{"web":{"client_id":"example"}}')


def test_google_scope_names_round_trip_from_cli_aliases():
    scopes = parse_google_scopes("gmail.readonly,spreadsheets")

    assert scopes == tuple(sorted((
        GOOGLE_GMAIL_READONLY_SCOPE,
        GOOGLE_SHEETS_SCOPE,
    )))
    assert google_scope_names(scopes) == tuple(sorted((
        "gmail.readonly",
        "spreadsheets",
    )))


def test_google_authorization_handoff_is_checked_and_round_trips():
    result = GoogleAuthorization(
        authorized_user_json=(
            '{"client_id":"example.apps.googleusercontent.com",'
            '"refresh_token":"private-token","token_uri":"https://token"}'
        ),
        granted_scopes=(
            GOOGLE_GMAIL_READONLY_SCOPE,
            GOOGLE_SHEETS_SCOPE,
        ),
        client_id="example.apps.googleusercontent.com",
        expiry="2026-08-01T12:00:00Z",
    )

    encoded = encode_google_authorization(result)
    decoded = decode_google_authorization(encoded)

    assert encoded.startswith("zg-google-v1.")
    assert decoded == result
    assert "private-token" not in encoded


def test_google_authorization_handoff_rejects_truncation():
    encoded = encode_google_authorization(
        GoogleAuthorization(
            authorized_user_json=(
                '{"client_id":"example.apps.googleusercontent.com",'
                '"refresh_token":"private-token"}'
            ),
            granted_scopes=(GOOGLE_GMAIL_READONLY_SCOPE,),
            client_id="example.apps.googleusercontent.com",
        )
    )

    with pytest.raises(RuntimeError, match="truncated or changed"):
        decode_google_authorization(encoded[:-1] + "0")


def test_google_refresh_uses_existing_grant_without_resending_scopes(
    monkeypatch,
):
    from zippergen.google_auth import credentials_from_json

    constructed: dict[str, object] = {}

    class FakeCredentials:
        valid = False

        @classmethod
        def from_authorized_user_info(cls, info, scopes=None):
            constructed["info"] = info
            constructed["scopes"] = scopes
            return cls()

        def refresh(self, transport):
            constructed["transport"] = transport
            self.valid = True

    class FakeRequest:
        pass

    monkeypatch.setattr(
        "zippergen.google_auth.google_imports",
        lambda: (object, FakeRequest, FakeCredentials, object),
    )
    credential = credentials_from_json(
        json.dumps(
            {
                "client_id": "client",
                "client_secret": "secret",
                "refresh_token": "refresh",
                "scopes": [
                    GOOGLE_GMAIL_MODIFY_SCOPE,
                    GOOGLE_SHEETS_SCOPE,
                ],
            }
        ),
        scopes=(GOOGLE_GMAIL_MODIFY_SCOPE, GOOGLE_SHEETS_SCOPE),
    )

    assert isinstance(credential, FakeCredentials)
    assert "scopes" not in constructed["info"]
    assert constructed["scopes"] is None
    assert isinstance(constructed["transport"], FakeRequest)


def test_google_refresh_rejects_a_serialized_grant_missing_required_scope(
    monkeypatch,
):
    from zippergen.google_auth import credentials_from_json

    monkeypatch.setattr(
        "zippergen.google_auth.google_imports",
        lambda: (object, object, object, object),
    )

    with pytest.raises(
        RuntimeError,
        match="does not cover required scope.*spreadsheets",
    ):
        credentials_from_json(
            json.dumps(
                {
                    "client_id": "client",
                    "client_secret": "secret",
                    "refresh_token": "refresh",
                    "scopes": [GOOGLE_GMAIL_MODIFY_SCOPE],
                }
            ),
            scopes=(GOOGLE_GMAIL_MODIFY_SCOPE, GOOGLE_SHEETS_SCOPE),
        )
