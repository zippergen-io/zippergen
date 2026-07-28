import pytest

from zippergen.google_auth import (
    GOOGLE_GMAIL_MODIFY_SCOPE,
    GOOGLE_GMAIL_READONLY_SCOPE,
    GOOGLE_SHEETS_READONLY_SCOPE,
    GOOGLE_SHEETS_SCOPE,
    google_scope_for_access,
    google_scopes_cover,
    google_scopes_for_access,
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
