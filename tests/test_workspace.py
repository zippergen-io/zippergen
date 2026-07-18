import json
from pathlib import Path

from zippergen.workspace import (
    Workspace,
    discover_project_root,
    discover_workflow_specs,
)


def test_project_root_prefers_containing_git_checkout(tmp_path):
    root = tmp_path / "project"
    nested = root / "src" / "package"
    nested.mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "src" / "pyproject.toml").write_text("[project]\nname='nested'\n")

    assert discover_project_root(nested) == root


def test_workflow_discovery_uses_ast_without_importing_modules(tmp_path):
    (tmp_path / ".git").mkdir()
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "review.py").write_text(
        "raise RuntimeError('must not import during discovery')\n\n"
        "@workflow\n"
        "def review(request: str):\n"
        "    return request\n"
    )
    ignored = tmp_path / ".venv"
    ignored.mkdir()
    (ignored / "hidden.py").write_text("@workflow\ndef hidden(): pass\n")

    assert discover_workflow_specs(tmp_path) == ["workflows/review.py:review"]


def test_workspace_state_lives_outside_checkout_and_remembers_workflow(tmp_path):
    root = tmp_path / "project"
    home = tmp_path / "state"
    root.mkdir()
    (root / ".git").mkdir()
    workflow_path = root / "review.py"
    workflow_path.write_text("@workflow\ndef review(): pass\n")

    workspace = Workspace(root, home=home)
    selected = workspace.select_workflow(str(workflow_path) + ":review")

    assert selected == "review.py:review"
    assert workspace.current_workflow == "review.py:review"
    assert workspace.absolute_spec(selected) == str(workflow_path) + ":review"
    assert workspace.state_path.is_relative_to(home)
    assert not (root / ".zippergen").exists()


def test_workspace_creates_unique_managed_runs(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    workspace = Workspace(root, home=tmp_path / "state")

    first = workspace.new_run(
        workflow_spec="review.py:review",
        workflow_name="review",
        fingerprint="abc",
        inputs={"request": "Explain the sky", "max_retries": 2},
        llm="mock",
    )
    second = workspace.new_run(
        workflow_spec="review.py:review",
        workflow_name="review",
        fingerprint="abc",
        inputs={"request": "Explain deployment", "max_retries": 3},
        llm="mock",
    )

    assert first["run_id"] != second["run_id"]
    assert Path(first["store"]).parent == workspace.runs_directory
    assert not Path(first["store"]).exists()
    assert workspace.current_run_id == second["run_id"]
    assert [run["run_id"] for run in workspace.list_runs()] == [
        second["run_id"],
        first["run_id"],
    ]


def test_workspace_updates_run_and_saves_assistant_request(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    workspace = Workspace(root, home=tmp_path / "state")
    run = workspace.new_run(
        workflow_spec="review.py:review",
        workflow_name="review",
        fingerprint="abc",
        inputs={"request": "Hello"},
        llm="mock",
    )

    updated = workspace.update_run(run["run_id"], status="done", result="Hello!")
    request = workspace.save_request(
        kind="create",
        prompt="Create a review workflow",
        content="Use $zippergen-workflows.\nCreate a review workflow.",
    )

    assert updated["status"] == "done"
    assert updated["result"] == "Hello!"
    assert Path(request["content_file"]).read_text().startswith(
        "Use $zippergen-workflows."
    )
    metadata = json.loads(
        (workspace.requests_directory / f"{request['request_id']}.json").read_text()
    )
    assert metadata["prompt"] == "Create a review workflow"
