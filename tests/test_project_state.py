"""Project state as properties, with no source or specification provenance."""

import shutil
from pathlib import Path

import pytest

from zippergen.project_state import (
    MissingRequirement,
    ProjectState,
    next_action,
    read_project_state,
)
from zippergen.workspace import Workspace

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "diagnosis.py"
ENTRY = "workflow.py:diagnosis_consensus"


def _project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    shutil.copy(EXAMPLE, root / "workflow.py")
    return Workspace(root, home=tmp_path / "home")


def _state(**overrides) -> ProjectState:
    values = {
        "root": "/project",
        "name": "project",
        "manifest_present": True,
        "specification": "present",
        "workflow": "present",
        "workflow_entry": ENTRY,
        "validation": "valid",
        "validation_detail": None,
        "missing": (),
    }
    values.update(overrides)
    return ProjectState(**values)  # type: ignore[arg-type]


def test_a_configured_project_is_valid_and_complete(tmp_path):
    workspace = _project(tmp_path)
    workspace.select_workflow(ENTRY, cwd=workspace.root)
    workspace.save_specification("Two reviewers must agree.")

    state = read_project_state(workspace)

    assert state.workflow == "present"
    assert state.validation == "valid"
    assert state.specification == "present"
    assert state.configuration == "complete"
    assert state.next_action.command == "zippergen run"


def test_the_workflow_entry_resolves_against_the_project_not_the_caller(
    tmp_path,
    monkeypatch,
):
    """The entry is relative to the project, wherever the caller stands."""

    workspace = _project(tmp_path)
    workspace.select_workflow(ENTRY, cwd=workspace.root)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert read_project_state(workspace).validation == "valid"


def test_a_broken_workflow_is_invalid_and_says_why(tmp_path):
    workspace = _project(tmp_path)
    workspace.select_workflow(ENTRY, cwd=workspace.root)
    (workspace.root / "workflow.py").write_text("this is not python (((")

    state = read_project_state(workspace)

    assert state.validation == "invalid"
    assert "SyntaxError" in str(state.validation_detail)
    assert state.next_action.command == "zippergen verify"


def test_a_missing_credential_is_named_but_never_valued(tmp_path):
    workspace = _project(tmp_path)
    workspace.select_workflow(ENTRY, cwd=workspace.root)
    workspace.save_specification("Two reviewers must agree.")
    workspace.ensure_model_configuration("openai:gpt-4o")

    state = read_project_state(workspace)

    assert state.configuration == "incomplete"
    assert [item.what for item in state.missing] == ["OPENAI_API_KEY"]
    assert state.next_action.command == "zippergen provider configure openai"
    # Entering a credential is never something the agent may do.
    assert state.next_action.may_agent_run is False


def test_a_supplied_credential_completes_the_configuration(tmp_path):
    workspace = _project(tmp_path)
    workspace.select_workflow(ENTRY, cwd=workspace.root)
    workspace.save_specification("Two reviewers must agree.")
    workspace.ensure_model_configuration("openai:gpt-4o")
    workspace.save_secrets({"OPENAI_API_KEY": "sk-not-a-real-key"})

    state = read_project_state(workspace)

    assert state.configuration == "complete"
    assert state.next_action.command == "zippergen run"


def test_state_carries_no_source_or_specification_provenance(tmp_path):
    """No `external`, no `stale`, no lock — see AGENT-HARNESS-DESIGN §10.1.

    Editing the workflow is the normal case once an agent maintains it, so a
    byte fingerprint would report a permanent anomaly and mean nothing.
    """

    workspace = _project(tmp_path)
    workspace.select_workflow(ENTRY, cwd=workspace.root)
    workspace.save_specification("Two reviewers must agree.")
    before = read_project_state(workspace)

    source = workspace.root / "workflow.py"
    source.write_text(source.read_text().replace("MAX_ROUNDS = 5", "MAX_ROUNDS = 3"))
    after = read_project_state(workspace)

    assert before == after
    assert not hasattr(after, "source")
    assert "external" not in repr(after)
    assert "stale" not in repr(after)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"manifest_present": False}, "zippergen project init"),
        ({"workflow": "absent", "workflow_entry": None}, "zippergen adopt"),
        ({"validation": "invalid"}, "zippergen verify"),
        ({"specification": "absent"}, "write specification.md"),
        (
            {"missing": (MissingRequirement("K", "zippergen provider configure x"),)},
            "zippergen provider configure x",
        ),
        ({}, "zippergen run"),
    ],
)
def test_next_action_is_the_first_matching_row(overrides, expected):
    assert next_action(_state(**overrides)).command == expected


def test_an_invalid_workflow_outranks_a_missing_specification():
    """Ordering is load-bearing; callers must read it, not re-derive it."""

    state = _state(validation="invalid", specification="absent")

    assert next_action(state).command == "zippergen verify"
