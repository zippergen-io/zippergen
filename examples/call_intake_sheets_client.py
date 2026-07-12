"""Google Sheets table client for examples/call_intake.py.

The workflow writes to an existing spreadsheet. Create the spreadsheet in the
Google account, share it view-only as needed, then set ZIPPERGEN_CALL_SHEET_ID.

Required packages:
    pip install google-auth google-auth-oauthlib google-api-python-client

One-time auth setup:
    python examples/call_intake_sheets_client.py --setup

Default credential paths:
    ZIPPERGEN_CALL_SHEETS_CREDENTIALS  path to credentials.json
    ZIPPERGEN_CALL_SHEETS_TOKEN        OAuth2 token cache
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


CREDENTIALS_PATH = Path(
    os.environ.get(
        "ZIPPERGEN_CALL_SHEETS_CREDENTIALS",
        os.environ.get(
            "ZIPPERGEN_CALL_GMAIL_CREDENTIALS",
            os.environ.get("ZIPPERGEN_GMAIL_CREDENTIALS", str(Path.home() / ".zippergen_google_credentials.json")),
        ),
    )
)
TOKEN_PATH = Path(
    os.environ.get(
        "ZIPPERGEN_CALL_SHEETS_TOKEN",
        str(Path.home() / ".zippergen_call_sheets_token.json"),
    )
)

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


def _get_service():
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        sys.exit(
            "Google Sheets API libraries missing.\n"
            "Install with: pip install google-auth google-auth-oauthlib google-api-python-client"
        )

    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
        if hasattr(creds, "has_scopes") and not creds.has_scopes(SCOPES):
            creds = None

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
                    "Download credentials.json from Google Cloud Console and set "
                    "ZIPPERGEN_CALL_SHEETS_CREDENTIALS."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_PATH.write_text(creds.to_json())

    return build("sheets", "v4", credentials=creds)


def setup_auth() -> None:
    _get_service()
    print(f"Auth successful. Token saved to {TOKEN_PATH}")


def _column_letter(index: int) -> str:
    letters = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def _quote_sheet_name(name: str) -> str:
    return "'" + name.replace("'", "''") + "'"


def _managed_range(sheet_name: str, field_count: int, row_count: int | None = None) -> str:
    last_col = _column_letter(field_count)
    quoted = _quote_sheet_name(sheet_name)
    if row_count is None:
        return f"{quoted}!A:{last_col}"
    return f"{quoted}!A1:{last_col}{max(row_count, 1)}"


def _ensure_sheet(service, spreadsheet_id: str, sheet_name: str) -> None:
    spreadsheet = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    for sheet in spreadsheet.get("sheets", []):
        properties = sheet.get("properties", {})
        if properties.get("title") == sheet_name:
            return
    service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"requests": [{"addSheet": {"properties": {"title": sheet_name}}}]},
    ).execute()


def _normalise_row(values: list[object], headers: list[str], fields: list[str]) -> dict[str, str]:
    raw = {header: str(value or "") for header, value in zip(headers, values)}
    return {field: raw.get(field, "") for field in fields}


def read_rows(spreadsheet_id: str, sheet_name: str, fields: list[str]) -> list[dict[str, str]]:
    service = _get_service()
    _ensure_sheet(service, spreadsheet_id, sheet_name)
    result = service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range=_managed_range(sheet_name, len(fields)),
    ).execute()
    values = result.get("values", [])
    if not values:
        write_rows(spreadsheet_id, sheet_name, fields, [])
        return []
    headers = [str(value) for value in values[0]]
    if headers != fields:
        raise ValueError(
            f"Unexpected Google Sheet header in {sheet_name!r}. "
            "The first row must be the ZipperGen call table header."
        )
    return [_normalise_row(row, headers, fields) for row in values[1:]]


def write_rows(spreadsheet_id: str, sheet_name: str, fields: list[str], rows: list[dict[str, str]]) -> None:
    service = _get_service()
    _ensure_sheet(service, spreadsheet_id, sheet_name)
    managed_range = _managed_range(sheet_name, len(fields))
    values = [fields] + [[row.get(field, "") for field in fields] for row in rows]
    service.spreadsheets().values().clear(
        spreadsheetId=spreadsheet_id,
        range=managed_range,
        body={},
    ).execute()
    service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=_managed_range(sheet_name, len(fields), len(values)),
        valueInputOption="RAW",
        body={"values": values},
    ).execute()


if __name__ == "__main__":
    if "--setup" not in sys.argv:
        sys.exit("Usage: python examples/call_intake_sheets_client.py --setup")
    setup_auth()
