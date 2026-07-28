import json

import pytest

from zippergen.google_sheets import (
    GoogleSheetsError,
    GoogleSheetsTable,
    read_json_rows,
    upsert_json_row,
)


def _runtime_environment(monkeypatch):
    credential_env = "ZIPPERGEN_CONNECTOR_CALL_RECORDS_GOOGLE_CREDENTIAL"
    monkeypatch.setenv(credential_env, '{"refresh_token":"private"}')
    monkeypatch.setenv(
        "ZIPPERGEN_CONNECTORS_JSON",
        json.dumps({
            "requirement:call-records": {
                "kind": "google-sheets",
                "provider": "google",
                "access": "read-only",
                "spreadsheet_id": "sheet-123",
                "tab": "Calls",
                "credential_env": credential_env,
            }
        }),
    )


def test_google_sheets_requirement_resolves_private_runtime_binding(
    monkeypatch,
):
    _runtime_environment(monkeypatch)

    table = GoogleSheetsTable.from_requirement("call-records")

    assert table.spreadsheet_id == "sheet-123"
    assert table.tab == "Calls"
    assert table.credential_json == '{"refresh_token":"private"}'
    assert table.access == "read-only"


def test_google_sheets_requirement_fails_clearly_without_runtime_binding(
    monkeypatch,
):
    monkeypatch.delenv("ZIPPERGEN_CONNECTORS_JSON", raising=False)

    with pytest.raises(GoogleSheetsError, match="No connector runtime"):
        GoogleSheetsTable.from_requirement("call-records")


def test_google_sheets_upsert_uses_stable_key_for_retry_safety(monkeypatch):
    table = GoogleSheetsTable(
        requirement="call-records",
        spreadsheet_id="sheet-123",
        tab="Calls",
        credential_json="private",
    )
    updates = []
    appends = []
    monkeypatch.setattr(
        table,
        "read_rows",
        lambda columns: [{"call_id": "call-1", "title": "Old"}],
    )
    monkeypatch.setattr(
        table,
        "_update",
        lambda range_text, values: updates.append((range_text, values)),
    )
    monkeypatch.setattr(
        table,
        "_append",
        lambda range_text, values: appends.append((range_text, values)),
    )

    result = table.upsert_row(
        {"call_id": "call-1", "title": "New"},
        columns=("call_id", "title"),
        key_field="call_id",
    )

    assert result == "updated"
    assert updates == [("'Calls'!A2:B2", [["call-1", "New"]])]
    assert appends == []


def test_read_only_google_sheets_binding_blocks_writes():
    table = GoogleSheetsTable(
        requirement="call-records",
        spreadsheet_id="sheet-123",
        tab="Calls",
        credential_json="private",
        access="read-only",
    )

    with pytest.raises(GoogleSheetsError, match="read-only"):
        table.replace_rows([], columns=("call_id",))


def test_json_helpers_keep_workflow_values_serializable(monkeypatch):
    _runtime_environment(monkeypatch)
    monkeypatch.setattr(
        GoogleSheetsTable,
        "read_rows",
        lambda self, columns: [{"call_id": "call-1", "title": "Example"}],
    )
    captured = {}

    def fake_upsert(self, record, *, columns, key_field):
        captured.update(record)
        return "created"

    monkeypatch.setattr(GoogleSheetsTable, "upsert_row", fake_upsert)

    rows = read_json_rows(
        "call-records",
        columns=("call_id", "title"),
    )
    status = upsert_json_row(
        "call-records",
        '{"call_id":"call-2","title":"New"}',
        columns=("call_id", "title"),
        key_field="call_id",
    )

    assert json.loads(rows)[0]["call_id"] == "call-1"
    assert status == "created"
    assert captured == {"call_id": "call-2", "title": "New"}
