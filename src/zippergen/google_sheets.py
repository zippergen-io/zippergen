"""Google Sheets connector runtime and OAuth support.

Workflow code refers to a logical connector requirement. The project supplies the
machine-specific spreadsheet, tab, and private OAuth credential at run or
deployment time.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote


from zippergen.google_http import (
    request as google_request,
    response_json as google_response_json,
)
from zippergen.connectors import requirement_binding
from zippergen.google_auth import (
    GOOGLE_SHEETS_SCOPE,
    GoogleConnectorError,
    authorize_google,
    check_google_authorization as _check_google_authorization,
    credentials_from_json,
    google_imports,
    google_scope_for_access,
)


_SHEETS_API = "https://sheets.googleapis.com/v4/spreadsheets"


class GoogleSheetsError(GoogleConnectorError):
    """A clear connector error suitable for command and workflow output."""


def authorize_google_sheets(
    credentials_file: str | Path,
    *,
    open_browser: bool = True,
) -> str:
    """Run Google's desktop OAuth flow and return private authorized-user JSON."""

    return authorize_google(
        credentials_file,
        scopes=(GOOGLE_SHEETS_SCOPE,),
        open_browser=open_browser,
    )


def check_google_authorization(value: str) -> str:
    """Refresh an authorized-user credential and return its normalized JSON."""

    return _check_google_authorization(
        value,
        scopes=(GOOGLE_SHEETS_SCOPE,),
    )


def _requirement_binding(requirement: str) -> dict[str, object]:
    return requirement_binding(requirement, kind='google-sheets', error=GoogleSheetsError)


def _column_letter(index: int) -> str:
    if index <= 0:
        raise ValueError("Column indices start at one.")
    letters = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def _quote_tab(name: str) -> str:
    return "'" + name.replace("'", "''") + "'"


def _normalise_columns(columns: Sequence[str]) -> tuple[str, ...]:
    values = tuple(str(column).strip() for column in columns)
    if not values or any(not column for column in values):
        raise ValueError("Google Sheets columns must not be empty.")
    if len(values) != len(set(values)):
        raise ValueError("Google Sheets columns must be unique.")
    return values


def _cell_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (str, bool, int, float)):
        return value
    return json.dumps(value, sort_keys=True)


@dataclass
class GoogleSheetsTable:
    """One configured spreadsheet tab."""

    requirement: str
    spreadsheet_id: str
    tab: str
    credential_json: str = field(repr=False)
    access: str = "read-write"

    @classmethod
    def from_requirement(cls, requirement: str) -> "GoogleSheetsTable":
        binding = _requirement_binding(requirement)
        credential_env = str(binding.get("credential_env") or "")
        credential = os.environ.get(credential_env, "") if credential_env else ""
        if not credential:
            raise GoogleSheetsError(
                f"Private Google credential is missing for {requirement!r}."
            )
        spreadsheet_id = str(binding.get("spreadsheet_id") or "")
        tab = str(binding.get("tab") or "")
        if not spreadsheet_id or not tab:
            raise GoogleSheetsError(
                f"Google Sheets resource is incomplete for {requirement!r}."
            )
        return cls(
            requirement,
            spreadsheet_id,
            tab,
            credential,
            str(binding.get("access") or "read-only"),
        )

    def _session(self):
        session_type, _request, _credentials, _flow = google_imports()
        return session_type(credentials_from_json(
            self.credential_json,
            scopes=(
                google_scope_for_access("google-sheets", self.access),
            ),
        ))

    def _values_url(self, range_text: str, suffix: str = "") -> str:
        spreadsheet = quote(self.spreadsheet_id, safe="")
        encoded_range = quote(range_text, safe="")
        return f"{_SHEETS_API}/{spreadsheet}/values/{encoded_range}{suffix}"

    def _require_write(self, operation: str) -> None:
        if self.access == "read-only":
            raise GoogleSheetsError(
                f"Google Sheets {operation} requires write access, but "
                f"connector {self.requirement!r} is read-only."
            )

    @staticmethod
    def _response_json(response, operation: str) -> dict[str, object]:
        return google_response_json(
            response, operation, service="Google Sheets", error=GoogleSheetsError,
        )

    def _request(self, method: str, url: str, **kwargs):
        return google_request(
            self._session(), method, url, service="Google Sheets", error=GoogleSheetsError, **kwargs,
        )

    def inspect(self) -> dict[str, object]:
        spreadsheet = quote(self.spreadsheet_id, safe="")
        response = self._request(
            "get",
            f"{_SHEETS_API}/{spreadsheet}",
            params={"fields": "properties.title,sheets.properties.title"},
            timeout=10,
        )
        value = self._response_json(response, "configuration check")
        raw_sheets = value.get("sheets")
        sheets = raw_sheets if isinstance(raw_sheets, list) else []
        tabs = [
            str(properties.get("title"))
            for sheet in sheets
            if isinstance(sheet, dict)
            and isinstance((properties := sheet.get("properties")), dict)
            and properties.get("title")
        ]
        if self.tab not in tabs:
            raise GoogleSheetsError(
                f"Spreadsheet is reachable, but tab {self.tab!r} does not "
                f"exist. Available tabs: {', '.join(tabs) or 'none'}."
            )
        properties = value.get("properties")
        title = (
            str(properties.get("title"))
            if isinstance(properties, dict)
            else self.spreadsheet_id
        )
        return {"title": title, "tab": self.tab, "tabs": tabs}

    def read_rows(self, columns: Sequence[str]) -> list[dict[str, object]]:
        fields = _normalise_columns(columns)
        end = _column_letter(len(fields))
        range_text = f"{_quote_tab(self.tab)}!A:{end}"
        response = self._request(
            "get",
            self._values_url(range_text),
            params={"majorDimension": "ROWS", "valueRenderOption": "UNFORMATTED_VALUE"},
            timeout=20,
        )
        value = self._response_json(response, "read")
        rows = value.get("values", [])
        if not isinstance(rows, list) or any(not isinstance(row, list) for row in rows):
            raise GoogleSheetsError("Google Sheets returned malformed row data.")
        if not rows:
            return []
        header = tuple(str(item) for item in rows[0])
        if header != fields:
            raise GoogleSheetsError(
                f"Unexpected header in tab {self.tab!r}. Expected "
                f"{list(fields)!r}, found {list(header)!r}."
            )
        result: list[dict[str, object]] = []
        for raw_row in rows[1:]:
            result.append({
                field: raw_row[index] if index < len(raw_row) else ""
                for index, field in enumerate(fields)
            })
        return result

    def _update(self, range_text: str, values: list[list[object]]) -> None:
        self._require_write("update")
        payload: Any = {"majorDimension": "ROWS", "values": values}
        response = self._request(
            "put",
            self._values_url(range_text),
            params={"valueInputOption": "RAW"},
            json=payload,
            timeout=20,
        )
        self._response_json(response, "write")

    def _append(self, range_text: str, values: list[list[object]]) -> None:
        self._require_write("append")
        payload: Any = {"majorDimension": "ROWS", "values": values}
        response = self._request(
            "post",
            self._values_url(range_text, ":append"),
            params={
                "valueInputOption": "RAW",
                "insertDataOption": "INSERT_ROWS",
            },
            json=payload,
            timeout=20,
        )
        self._response_json(response, "append")

    def replace_rows(
        self,
        rows: Sequence[Mapping[str, object]],
        *,
        columns: Sequence[str],
    ) -> None:
        """Replace the managed table while keeping its explicit schema.

        All values are validated before clearing. Clear and update are still
        separate requests: a remote failure between them can leave an empty
        table. Retry with the same replacement data to finish the operation.
        """

        self._require_write("replacement")
        fields = _normalise_columns(columns)
        end = _column_letter(len(fields))
        range_text = f"{_quote_tab(self.tab)}!A:{end}"
        # Validate the entire replacement before the first destructive call.
        values = [
            list(fields),
            *[
                [_cell_value(row.get(field, "")) for field in fields]
                for row in rows
            ],
        ]
        json.dumps(values, allow_nan=False)
        response = self._request(
            "post",
            self._values_url(range_text, ":clear"),
            json={},
            timeout=20,
        )
        self._response_json(response, "clear")
        self._update(
            f"{_quote_tab(self.tab)}!A1:{end}{len(values)}",
            values,
        )

    def upsert_row(
        self,
        record: Mapping[str, object],
        *,
        columns: Sequence[str],
        key_field: str,
    ) -> str:
        """Create or replace one keyed row.

        A stable key lets a sequential retry find a previously appended row.
        This read-then-write sequence is not atomic. Concurrent writers can
        still append duplicates, so the table needs a single writer.
        """

        self._require_write("upsert")
        fields = _normalise_columns(columns)
        if key_field not in fields:
            raise ValueError(
                f"Google Sheets key field {key_field!r} is not in columns."
            )
        key = record.get(key_field)
        if key is None or str(key) == "":
            raise ValueError(
                f"Google Sheets record needs a non-empty {key_field!r}."
            )
        rows = self.read_rows(fields)
        values = [_cell_value(record.get(field, "")) for field in fields]
        end = _column_letter(len(fields))
        for index, existing in enumerate(rows, start=2):
            if str(existing.get(key_field, "")) == str(key):
                self._update(
                    f"{_quote_tab(self.tab)}!A{index}:{end}{index}",
                    [values],
                )
                return "updated"
        if not rows:
            # An empty read means the tab has no header yet.
            self._update(
                f"{_quote_tab(self.tab)}!A1:{end}1",
                [list(fields)],
            )
        self._append(
            f"{_quote_tab(self.tab)}!A:{end}",
            [values],
        )
        return "created"


def read_json_rows(
    requirement: str,
    *,
    columns: Sequence[str],
) -> str:
    """Read configured rows and return a JSON array for workflow variables."""

    rows = GoogleSheetsTable.from_requirement(requirement).read_rows(columns)
    return json.dumps(rows, sort_keys=True)


def upsert_json_row(
    requirement: str,
    record_json: str,
    *,
    columns: Sequence[str],
    key_field: str,
) -> str:
    """Upsert one JSON object and return ``created`` or ``updated``."""

    try:
        value: Any = json.loads(record_json)
    except json.JSONDecodeError as exc:
        raise ValueError("Google Sheets record must be valid JSON.") from exc
    if not isinstance(value, dict):
        raise ValueError("Google Sheets record must be a JSON object.")
    return GoogleSheetsTable.from_requirement(requirement).upsert_row(
        value,
        columns=columns,
        key_field=key_field,
    )


__all__ = [
    "GOOGLE_SHEETS_SCOPE",
    "GoogleSheetsError",
    "GoogleSheetsTable",
    "authorize_google_sheets",
    "check_google_authorization",
    "read_json_rows",
    "upsert_json_row",
]
