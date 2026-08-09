import json
import shutil
from pathlib import Path

from zippergen.locator import resolve_path, statement_node_paths
from zippergen.projection import project
from zippergen.serve import load_workflow_spec, main
from zippergen.store import open_store, write_execution_state
from zippergen.syntax import ActStmt, _ordered_workflow_lifelines
from zippergen.workspace import Workspace


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "email_approval.py"


def _observed_run(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    shutil.copy(EXAMPLE, root / "workflow.py")
    home = tmp_path / "home"
    workspace = Workspace(root, home=home)
    workspace.initialize_project(name="inspection")
    record = workspace.new_run(
        workflow_spec="workflow.py:email_approval",
        workflow_name="email_approval",
        fingerprint="test",
        inputs={"request": "hello"},
        llm="mock",
    )
    workflow, _module = load_workflow_spec(
        str(root / "workflow.py") + ":email_approval"
    )
    writer = next(
        participant
        for participant in _ordered_workflow_lifelines(workflow)
        if participant.name == "Writer"
    )
    local = project(workflow, writer)
    action_path = next(
        path
        for path in statement_node_paths(local).values()
        if isinstance(resolve_path(local, path), ActStmt)
    )
    connection = open_store(str(record["store"]))
    write_execution_state(
        connection,
        "Writer",
        "running_model",
        [action_path],
        {"action": "draft_reply", "kind": "model"},
    )
    connection.close()
    monkeypatch.chdir(root)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    return record


def test_inspect_shows_the_current_local_program_pointer(
    tmp_path, monkeypatch, capsys
):
    _observed_run(tmp_path, monkeypatch)

    assert main(["inspect", "--agent", "Writer"]) == 0

    output = capsys.readouterr().out
    assert "Execution positions" in output
    assert "Writer" in output
    assert "running model action" in output
    assert "▶" in output
    assert "draft_reply" in output
    assert "request" not in output


def test_inspect_has_a_machine_readable_view(tmp_path, monkeypatch, capsys):
    record = _observed_run(tmp_path, monkeypatch)

    assert main(["inspect", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["subject"] == f"run {record['run_id']}"
    assert payload["focus"] == "Writer"
    writer = next(
        item for item in payload["positions"]
        if item["participant"] == "Writer"
    )
    assert writer["state"] == "running_model"
    assert writer["detail"] == {
        "action": "draft_reply",
        "kind": "model",
    }
