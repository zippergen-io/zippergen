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
    Json,
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
            "record",
            "JSON record",
            target="input",
            default={
                "record_id": "demo-1",
                "title": "First record",
                "status": "new",
            },
            required=True,
        ),
    ),
    files=("examples/google_sheets_records.py",),
)


@effect(connector="project-records", operation="upsert-json-row")
def write_record(record: Json) -> str:
    return upsert_json_row(
        "project-records",
        json.dumps(record, sort_keys=True),
        columns=RECORD_COLUMNS,
        key_field="record_id",
    )


@effect(connector="project-records", operation="read-json-rows")
def read_records() -> Json:
    return json.loads(
        read_json_rows(
            "project-records",
            columns=RECORD_COLUMNS,
        )
    )


@pure
def result_record(write_status: str, rows: Json) -> Json:
    return {"write_status": write_status, "rows": rows}


@workflow
def google_sheet_records(record: Json @ Requester) -> Json:
    Requester(record) >> Records(record)
    Records: write_status = write_record(record)
    Records: rows = read_records()
    Records: result = result_record(write_status, rows)
    Records(result) >> Requester(result)
    return result @ Requester
