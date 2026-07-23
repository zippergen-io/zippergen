import json
import tomllib
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


def test_project_root_prefers_nearest_zippergen_manifest(tmp_path):
    outer = tmp_path / "framework-checkout"
    project = outer / "tutorial"
    nested = project / "workflows"
    nested.mkdir(parents=True)
    (outer / ".git").mkdir()
    (project / "zippergen.toml").write_text(
        'schema_version = 1\nname = "tutorial"\nprompts_directory = "prompts"\n'
    )

    assert discover_project_root(nested) == project


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
    assert workspace.current_task_path == root / ".zippergen" / "current-task.md"
    assert workspace.current_task_path.read_text() == Path(
        request["content_file"]
    ).read_text()
    assert workspace.current_request()["request_id"] == request["request_id"]
    assert workspace.list_requests()[0]["request_id"] == request["request_id"]
    assert workspace.load()["current_request"] == request["request_id"]
    metadata = json.loads(
        (workspace.requests_directory / f"{request['request_id']}.json").read_text()
    )
    assert metadata["prompt"] == "Create a review workflow"
    assert metadata["task_file"] == str(workspace.current_task_path)
    assert metadata["status"] == "prepared"

    workspace.update_request(
        str(request["request_id"]),
        status="awaiting_review",
        assistant="Codex",
    )
    assert workspace.current_request()["status"] == "awaiting_review"
    assert workspace.current_request()["assistant"] == "Codex"

    workspace.current_task_path.write_text("stale task\n")
    workspace.current_request()
    assert workspace.current_task_path.read_text() == Path(
        request["content_file"]
    ).read_text()


def test_workspace_reset_archives_private_state_and_keeps_project_files(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / ".git").mkdir()
    workflow = root / "workflow.py"
    workflow.write_text("@workflow\ndef review(): pass\n")
    workspace = Workspace(root, home=tmp_path / "state")
    workspace.initialize_project(name="Review project")
    workspace.add_prompt(kind="initial", content="Keep source visible.")
    workspace.select_workflow("workflow.py:review", cwd=root)
    workspace.new_run(
        workflow_spec="workflow.py:review",
        workflow_name="review",
        fingerprint="abc",
        inputs={},
        llm="mock",
    )
    workspace.save_request(
        kind="create",
        prompt="Keep source visible.",
        content="Create the workflow.\n",
    )
    workspace.save_model_profile(
        "workflow.py:review",
        default="mock",
        lifelines={},
    )
    workspace.save_provider_profile(
        "openai",
        {"kind": "api", "key_env": "OPENAI_API_KEY"},
    )
    workspace.save_secrets({"OPENAI_API_KEY": "private"})
    workspace.update(last_deployment="review-service")
    drafts = root / ".zippergen" / "prompt-drafts"
    drafts.mkdir(parents=True)
    (drafts / "unfinished.md").write_text("Do not lose this draft.\n")
    deployment = workspace.home / "deployments" / "review-service.json"
    deployment.parent.mkdir(parents=True)
    deployment.write_text("{}\n")

    summary = workspace.private_state_summary()
    assert summary["runs"] == 1
    assert summary["requests"] == 1
    assert summary["development_secrets"] == 1
    assert summary["project_local_exists"] is True

    result = workspace.reset_private_state()

    backup = Path(result["backup_directory"])
    assert result["workspace_moved"] is True
    assert result["project_local_moved"] is True
    assert (backup / "workspace" / "workspace.json").exists()
    assert (backup / "workspace" / "development.secrets.json").exists()
    assert list((backup / "workspace" / "runs").glob("*.json"))
    assert list((backup / "workspace" / "requests").glob("*.json"))
    assert (backup / "project-local" / "current-task.md").exists()
    assert (
        backup / "project-local" / "prompt-drafts" / "unfinished.md"
    ).exists()
    metadata = json.loads((backup / "reset.json").read_text())
    assert metadata["project_root"] == str(root)
    assert metadata["workspace_moved"] is True
    assert metadata["project_local_moved"] is True

    assert workflow.exists()
    assert workspace.manifest_path.exists()
    assert workspace.prompt("P001")["content"] == "Keep source visible."
    assert (root / ".git").exists()
    assert deployment.exists()
    assert not workspace.directory.exists()
    assert not workspace.current_task_path.exists()
    assert workspace.current_workflow is None
    assert workspace.current_run_id is None
    assert workspace.load()["last_deployment"] is None
    assert workspace.load_secrets() == {}

    empty = workspace.reset_private_state()
    assert empty["backup_directory"] is None


def test_workspace_reset_can_archive_unreadable_private_state(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    workspace = Workspace(root, home=tmp_path / "state")
    workspace.directory.mkdir(parents=True)
    workspace.state_path.write_text("{broken")
    workspace.secrets_path.write_text("{also-broken")

    summary = workspace.private_state_summary()

    assert len(summary["warnings"]) == 2
    assert summary["development_secrets"] == "present but unreadable"

    result = workspace.reset_private_state()

    backup = Path(result["backup_directory"])
    assert (backup / "workspace" / "workspace.json").read_text() == "{broken"
    assert (
        backup / "workspace" / "development.secrets.json"
    ).read_text() == "{also-broken"
    assert workspace.load()["current_workflow"] is None


def test_workspace_reset_supports_home_inside_project_tooling_directory(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    workspace = Workspace(root, home=root / ".zippergen")
    workspace.select_workflow("workflow.py:review", cwd=root)
    workspace.save_request(
        kind="create",
        prompt="Create a review workflow.",
        content="Create the workflow.\n",
    )

    result = workspace.reset_private_state()

    backup = Path(result["backup_directory"])
    assert backup.is_relative_to(root / ".zippergen" / "resets")
    assert (backup / "workspace" / "workspace.json").exists()
    assert (backup / "project-local" / "current-task.md").exists()
    assert workspace.current_workflow is None
    assert workspace.private_state_summary()["project_local_exists"] is False


def test_workspace_keeps_model_profiles_per_workflow(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    workspace = Workspace(root, home=tmp_path / "state")

    saved = workspace.save_model_profile(
        "review.py:review",
        default="mock",
        lifelines={"Writer": "openai:gpt-4o-mini"},
    )

    assert saved == {
        "default": "mock",
        "lifelines": {"Writer": "openai:gpt-4o-mini"},
    }
    assert workspace.model_profile("review.py:review") == saved
    assert workspace.model_profile(
        "summary.py:summary",
        default="claude:claude-sonnet-4-6",
    ) == {
        "default": "claude:claude-sonnet-4-6",
        "lifelines": {},
    }


def test_workspace_initializes_visible_project_and_manages_prompt_ledger(tmp_path):
    root = tmp_path / "review-project"
    root.mkdir()
    workspace = Workspace(root, home=tmp_path / "state")

    manifest = workspace.initialize_project(name="Reviewed answers")
    initial = workspace.add_prompt(
        kind="initial",
        content="# Create reviewed answers\n\nNever return an unapproved draft.",
    )
    source = tmp_path / "retry change.md"
    source.write_text(
        "# Add bounded retries\n\nReturn an explicit failure after exhaustion.\n"
    )
    refinement = workspace.add_prompt(
        kind="refinement",
        content=source.read_text(),
        source_path=source,
        workflow_spec="workflows/review.py:review",
    )

    assert manifest["name"] == "Reviewed answers"
    assert workspace.manifest_path.exists()
    assert workspace.prompt_index_path.exists()
    assert initial["id"] == "P001"
    assert initial["file"] == "prompts/001-create-reviewed-answers.md"
    assert refinement["id"] == "P002"
    assert refinement["file"] == "prompts/002-add-bounded-retries.md"
    assert source.exists()
    assert "P001 [initial]" in workspace.prompt_context()
    assert workspace.prompt_context().index("P001") < workspace.prompt_context().index(
        "P002"
    )
    parsed = tomllib.loads(workspace.prompt_index_path.read_text())
    assert [entry["id"] for entry in parsed["prompts"]] == ["P001", "P002"]

    workspace.set_prompt_active("1", active=False)
    workspace.move_prompt("P002", relation="before", other_id="P001")
    replacement = workspace.replace_prompt(
        "P002",
        content="# Use three retries\n\nThe retry limit is three.",
    )

    records = workspace.list_prompts()
    assert [record["id"] for record in records] == ["P002", "P003", "P001"]
    assert records[0]["active"] is False
    assert replacement["replaces"] == "P002"
    assert replacement["active"] is True
    context = workspace.prompt_context()
    assert "Use three retries" in context
    assert "Add bounded retries" not in context
    assert "Create reviewed answers" not in context


def test_workspace_registers_existing_project_prompt_idempotently(tmp_path):
    root = tmp_path / "project"
    prompt_directory = root / "prompts"
    prompt_directory.mkdir(parents=True)
    source = prompt_directory / "design.md"
    source.write_text("# Design\n\nKeep the source visible.\n")
    workspace = Workspace(root, home=tmp_path / "state")

    first = workspace.add_prompt(
        kind="initial",
        content=source.read_text(),
        source_path=source,
    )
    second = workspace.add_prompt(
        kind="initial",
        content=source.read_text(),
        source_path=source,
    )

    assert first["created"] is True
    assert second["created"] is False
    assert second["id"] == first["id"]
    assert len(workspace.list_prompts()) == 1

    workspace.set_prompt_active("P001", active=False)
    third = workspace.add_prompt(
        kind="initial",
        content=source.read_text(),
        source_path=source,
    )
    assert third["created"] is False
    assert third["active"] is True
    assert workspace.prompt("P001")["active"] is True


def test_workspace_updates_prompt_content_without_changing_identity(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    workspace = Workspace(root, home=tmp_path / "state")
    original = workspace.add_prompt(
        kind="initial",
        content="# Original title\n\nOriginal requirement.",
    )

    updated = workspace.update_prompt_content(
        "P001",
        content="# Clearer title\n\nCorrected wording.",
    )

    assert updated["id"] == original["id"]
    assert updated["kind"] == original["kind"]
    assert updated["file"] == original["file"]
    assert updated["title"] == "Clearer title"
    assert workspace.prompt("P001")["content"] == (
        "# Clearer title\n\nCorrected wording."
    )
    assert len(workspace.list_prompts()) == 1


def test_workspace_prompt_fingerprint_tracks_active_content_and_order(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    workspace = Workspace(root, home=tmp_path / "state")
    workspace.add_prompt(kind="initial", content="Original requirement.")
    workspace.add_prompt(kind="refinement", content="Later requirement.")

    original = workspace.prompt_ledger_fingerprint()

    workspace.move_prompt("P002", relation="before", other_id="P001")
    reordered = workspace.prompt_ledger_fingerprint()
    assert reordered != original

    workspace.update_prompt_content("P001", content="Corrected requirement.")
    edited = workspace.prompt_ledger_fingerprint()
    assert edited != reordered

    workspace.set_prompt_active("P002", active=False)
    archived = workspace.prompt_ledger_fingerprint()
    assert archived != edited

    workspace.set_prompt_active("P002", active=True)
    assert workspace.prompt_ledger_fingerprint() == edited


def test_workspace_manages_one_canonical_spec_and_one_pending_refinement(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    workspace = Workspace(root, home=tmp_path / "state")

    manifest = workspace.initialize_project(name="Review project")

    assert manifest["specification_file"] == "specification.md"
    assert workspace.specification_path == root / "specification.md"
    assert workspace.specification() is None
    assert not workspace.prompt_index_path.exists()
    assert "prompts_directory" not in workspace.manifest_path.read_text()

    workspace.save_specification("# Reviewed answer\n\nRequire human approval.")
    accepted_fingerprint = workspace.specification_fingerprint(
        include_pending=False
    )
    first = workspace.save_pending_refinement("Add bounded retries.")
    second = workspace.save_pending_refinement(
        "Return an explicit failure after exhaustion.",
        append=True,
    )

    assert first["path"] == root / ".zippergen" / "pending-refinement.md"
    assert first["created"] is True
    assert second["created"] is False
    assert workspace.pending_refinement() == (
        "Add bounded retries.\n\nReturn an explicit failure after exhaustion."
    )
    assert workspace.load()["pending_specification_fingerprint"] == (
        accepted_fingerprint
    )
    assert workspace.specification_fingerprint() != accepted_fingerprint

    workspace.save_pending_refinement("bounded", append=True)
    assert workspace.pending_refinement().endswith("\n\nbounded")

    archived = workspace.archive_pending_refinement(status="reconciled")

    assert archived["status"] == "reconciled"
    assert workspace.pending_refinement() is None
    assert Path(archived["history_path"]).read_text().startswith(
        "Add bounded retries."
    )
    assert workspace.list_spec_history()[0]["status"] == "reconciled"


def test_workspace_migrates_active_legacy_prompts_without_deleting_history(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    workspace = Workspace(root, home=tmp_path / "state")
    first = workspace.add_prompt(kind="initial", content="Create a reviewer.")
    second = workspace.add_prompt(
        kind="refinement",
        content="Add bounded retries.",
    )

    migrated = workspace.ensure_specification()

    assert migrated["migrated"] is True
    assert "Create a reviewer." in workspace.specification()
    assert "Add bounded retries." in workspace.specification()
    assert (root / str(first["file"])).exists()
    assert (root / str(second["file"])).exists()
    assert len(workspace.list_prompts()) == 2


def test_workspace_provider_configuration_keeps_secrets_private(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    workspace = Workspace(root, home=tmp_path / "state")
    workspace.save_provider_profile(
        "openai",
        {"kind": "api", "key_env": "OPENAI_API_KEY"},
    )
    workspace.save_provider_profile(
        "local",
        {"kind": "local", "base_url": "http://localhost:11434/v1"},
    )
    workspace.save_secrets({"OPENAI_API_KEY": "private-key"})

    environment = workspace.development_provider_environment(
        ("openai:gpt-4o-mini", "ollama:qwen2.5:7b")
    )

    assert environment == {
        "OPENAI_API_KEY": "private-key",
        "OLLAMA_BASE_URL": "http://localhost:11434/v1",
    }
    assert workspace.secrets_path.stat().st_mode & 0o077 == 0
    assert not workspace.manifest_path.exists()


def test_project_init_recognizes_and_ignores_nested_framework_checkout(tmp_path):
    root = tmp_path / "tutorial"
    framework = root / "zippergen"
    workflows = root / "workflows"
    framework.mkdir(parents=True)
    workflows.mkdir()
    (root / ".git").mkdir()
    (framework / ".git").mkdir()
    (framework / "pyproject.toml").write_text('[project]\nname = "zippergen"\n')
    (framework / "example.py").write_text("@workflow\ndef framework_example(): pass\n")
    (workflows / "answer.py").write_text("@workflow\ndef answer(): pass\n")
    workspace = Workspace(root, home=tmp_path / "state")

    manifest = workspace.initialize_project(name="Tutorial")

    assert manifest["framework_directory"] == "zippergen"
    assert "/zippergen/" in (root / ".gitignore").read_text().splitlines()
    assert "/.zippergen/" in (root / ".gitignore").read_text().splitlines()
    assert "/tutorial-runtime/" in (root / ".gitignore").read_text().splitlines()
    assert workspace.discover_workflows() == ["workflows/answer.py:answer"]

    (root / ".gitignore").write_text("")
    workspace.initialize_project()
    assert "/zippergen/" in (root / ".gitignore").read_text().splitlines()
