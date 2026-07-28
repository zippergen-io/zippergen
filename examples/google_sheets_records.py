"""Small workflow with a configured Google Sheets resource.

The workflow owns the table schema and the meaning of each operation. Studio
owns Google authorization and the concrete spreadsheet:

    zippergen
    zippergen [google_sheet_records]> connector setup
    zippergen [google_sheet_records]> run
"""

import json

from zippergen import (
    ConnectorRequirement,
    DeploymentField,
    DeploymentSpec,
    Lifeline,
    effect,
    pure,
    read_json_rows,
    upsert_json_row,
    workflow,
)


Requester = Lifeline("Requester")
Records = Lifeline("Records")

RECORD_COLUMNS = ("record_id", "title", "status")

zippergen_connectors = (
    ConnectorRequirement(
        name="project-records",
        kind="google-sheets",
        participant="Records",
        capabilities=("read-rows", "upsert-row"),
        access="read-write",
        description="The project record table.",
    ),
)

zippergen_deployment = DeploymentSpec(
    name="google-sheets-records",
    description=(
        "Write one keyed JSON record to Google Sheets and read the table back."
    ),
    fields=(
        DeploymentField(
            "record_json",
            "JSON record",
            target="input",
            default=json.dumps(
                {
                    "record_id": "demo-1",
                    "title": "First record",
                    "status": "new",
                }
            ),
            required=True,
        ),
    ),
    files=("examples/google_sheets_records.py",),
)


@effect(connector="project-records", operation="upsert-json-row")
def write_record(record_json: str) -> str:
    return upsert_json_row(
        "project-records",
        record_json,
        columns=RECORD_COLUMNS,
        key_field="record_id",
    )


@effect(connector="project-records", operation="read-json-rows")
def read_records() -> str:
    return read_json_rows(
        "project-records",
        columns=RECORD_COLUMNS,
    )


@pure
def result_json(write_status: str, rows_json: str) -> str:
    return json.dumps(
        {
            "write_status": write_status,
            "rows": json.loads(rows_json),
        },
        sort_keys=True,
    )


@workflow
def google_sheet_records(record_json: str @ Requester) -> str:
    Requester(record_json) >> Records(record_json)
    Records: write_status = write_record(record_json)
    Records: rows_json = read_records()
    Records: result = result_json(write_status, rows_json)
    Records(result) >> Requester(result)
    return result @ Requester
