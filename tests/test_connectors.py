from types import ModuleType
from typing import Any, cast

import pytest

from zippergen import ConnectorRequirement
from zippergen.connectors import connector_requirements_from_module
from zippergen.semantic import workflow_semantics
from zippergen.serve import load_workflow_spec
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
