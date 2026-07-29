from types import ModuleType
from typing import Any, cast

import pytest

from zippergen import ConnectorRequirement
from zippergen.connectors import connector_requirements_from_module
from zippergen.semantic import workflow_semantics
from zippergen.serve import load_workflow_spec
from zippergen.serve import _validate_workflow
from zippergen.view import ViewOptions, render_workflow


def test_connector_requirement_validates_logical_metadata():
    requirement = ConnectorRequirement(
        name="human-approval",
        kind="telegram",
        participant="Reviewer",
        capabilities=("notify", "approve"),
        access="read-write",
    )

    assert requirement.as_dict()["kind"] == "telegram"
    assert ConnectorRequirement(
        name="read-default",
        kind="gmail",
        participant="Reviewer",
    ).access == "read-only"
    with pytest.raises(ValueError, match="Unsupported connector kind"):
        ConnectorRequirement(
            name="unknown",
            kind="carrier-pigeon",
            participant="Reviewer",
        )
    with pytest.raises(ValueError, match="Connector access"):
        ConnectorRequirement(
            name="approval",
            kind="telegram",
            participant="Reviewer",
            access="admin",
        )


def test_connector_loader_rejects_duplicate_requirements():
    module = ModuleType("duplicate_connectors")
    requirement = ConnectorRequirement(
        name="approval",
        kind="telegram",
        participant="Reviewer",
    )
    module.zippergen_connectors = (requirement, requirement)

    with pytest.raises(ValueError, match="Duplicate connector"):
        connector_requirements_from_module(module)


def test_tutorial_human_delivery_needs_no_redundant_connector_requirement():
    workflow, module = load_workflow_spec(
        "examples/tutorial_review.py:tutorial_review"
    )
    semantics = cast(dict[str, Any], workflow_semantics(workflow, module))
    code = render_workflow(
        workflow,
        module,
        options=ViewOptions(detail="full"),
    )

    assert semantics.get("connectors", {}) == {}
    assert any(
        site["lifeline"] == "Reviewer"
        and site["action"] == "approve_reply"
        and site["kind"] == "human"
        for site in semantics["action_sites"]
    )
    assert "approve_reply" in code
    assert "# Connector requirements" not in code


def test_effect_connector_is_semantic_and_must_be_declared(tmp_path):
    source = tmp_path / "sheet_workflow.py"
    source.write_text(
        """
from zippergen import ConnectorRequirement, Lifeline, effect, workflow

User = Lifeline("User")
Records = Lifeline("Records")

@effect(connector="call-records", operation="upsert-json-row")
def save_record(record: str) -> str:
    return "created"

zippergen_connectors = (
    ConnectorRequirement(
        name="call-records",
        kind="google-sheets",
        participant="Records",
        capabilities=("read-rows", "upsert-row"),
        access="read-write",
    ),
)

@workflow
def sample(record: str @ User) -> str:
    User(record) >> Records(record)
    Records: status = save_record(record)
    Records(status) >> User(status)
    return status @ User
"""
    )
    workflow, module = load_workflow_spec(f"{source}:sample")

    semantics = cast(dict[str, Any], workflow_semantics(workflow, module))
    definition = semantics["action_definitions"]["save_record"]

    assert definition["connector"] == "call-records"
    assert definition["operation"] == "upsert-json-row"
    assert semantics["connectors"]["call-records"]["kind"] == "google-sheets"


def test_google_sheets_records_example_is_valid_and_connector_explicit():
    workflow, module = load_workflow_spec(
        "examples/google_sheets_records.py:google_sheet_records"
    )

    validation = _validate_workflow(workflow, module)
    semantics = cast(dict[str, Any], workflow_semantics(workflow, module))

    assert validation["valid"] is True
    assert semantics["connectors"]["project-records"]["access"] == (
        "read-write"
    )
    assert semantics["action_definitions"]["write_record"]["operation"] == (
        "upsert-json-row"
    )
    assert semantics["action_definitions"]["read_records"]["operation"] == (
        "read-json-rows"
    )
    assert semantics["inputs"]["record"]["type"] == "Json"
    assert semantics["outputs"]["result@Requester"]["type"] == "Json"


def test_validation_rejects_connector_effect_on_the_wrong_participant(
    tmp_path,
):
    source = tmp_path / "wrong_owner.py"
    source.write_text(
        """
from zippergen import ConnectorRequirement, Lifeline, effect, workflow

User = Lifeline("User")
Records = Lifeline("Records")

@effect(connector="records", operation="read-json-rows")
def read_records() -> str:
    return "[]"

zippergen_connectors = (
    ConnectorRequirement(
        name="records",
        kind="google-sheets",
        participant="Records",
        capabilities=("read-rows",),
        access="read-only",
    ),
)

@workflow
def sample(request: str @ User) -> str:
    User: result = read_records()
    return result @ User
"""
    )
    workflow, module = load_workflow_spec(f"{source}:sample")

    validation = _validate_workflow(workflow, module)

    assert validation["valid"] is False
    assert any(
        check["status"] == "fail"
        and check["name"] == "effect connector owner read_records"
        for check in validation["checks"]
    )
