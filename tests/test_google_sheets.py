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


@pytest.mark.parametrize("rows", [
    {},
    None,
    [["call_id", "title"], {"call_id": "broken"}, ["target", "Old"]],
    ["call_id", ["target", "Old"]],
])
def test_upsert_rejects_malformed_rows_without_writing(monkeypatch, rows):
    table = GoogleSheetsTable("records", "sheet", "Calls", "private")
    writes = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"values": rows}

    class Session:
        def get(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr(table, "_session", Session)
    monkeypatch.setattr(table, "_update", lambda *args: writes.append(args))
    monkeypatch.setattr(table, "_append", lambda *args: writes.append(args))
    with pytest.raises(GoogleSheetsError, match="malformed"):
        table.upsert_row({"call_id": "target", "title": "New"},
                         columns=("call_id", "title"), key_field="call_id")
    assert not writes


def test_upsert_preserves_blank_row_positions(monkeypatch):
    table = GoogleSheetsTable("records", "sheet", "Calls", "private")
    writes = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"values": [["call_id", "title"], [], ["target", "Old"]]}

    class Session:
        def get(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr(table, "_session", Session)
    monkeypatch.setattr(table, "_update", lambda *args: writes.append(args))
    table.upsert_row({"call_id": "target", "title": "New"},
                     columns=("call_id", "title"), key_field="call_id")
    assert writes == [("'Calls'!A3:B3", [["target", "New"]])]


def test_replacement_validates_values_before_clearing_sheet(monkeypatch):
    table = GoogleSheetsTable("records", "sheet", "Calls", "private")
    requests = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {}

    class Session:
        def post(self, *args, **kwargs):
            requests.append((args, kwargs))
            return Response()

    monkeypatch.setattr(table, "_session", Session)
    with pytest.raises(TypeError):
        table.replace_rows([{"call_id": "1", "title": object()}],
                           columns=("call_id", "title"))
    assert not requests, "invalid replacement must leave the existing sheet intact"


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
