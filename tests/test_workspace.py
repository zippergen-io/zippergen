import json
import os
import tomllib
from pathlib import Path

import pytest

from zippergen.workspace import (
    Workspace,
    WorkspaceError,
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
        'schema_version = 1\nname = "tutorial"\n'
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


def test_workflow_entry_lives_in_visible_project_manifest(tmp_path):
    root = tmp_path / "project"
    home = tmp_path / "state"
    root.mkdir()
    (root / ".git").mkdir()
    workflow_path = root / "review.py"
    workflow_path.write_text("@workflow\ndef review(): pass\n")

    workspace = Workspace(root, home=home)
    workspace.initialize_project()
    selected = "review.py:review"
    workspace.select_workflow(selected)

    assert selected == "review.py:review"
    assert workspace.workflow_entry == "review.py:review"
    assert tomllib.loads(workspace.manifest_path.read_text())["workflow_entry"] == (
        "review.py:review"
    )
    assert workspace.absolute_spec(selected) == str(workflow_path) + ":review"
    assert workspace.state_path.is_relative_to(home)
    assert not (root / ".zippergen").exists()


def test_fresh_clone_resolves_workflow_from_manifest_without_private_state(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "workflow.py").write_text("@workflow\ndef review(): pass\n")
    original = Workspace(root, home=tmp_path / "original-home")
    original.initialize_project()
    original.select_workflow("workflow.py:review", cwd=root)

    clone = Workspace(root, home=tmp_path / "empty-home")

    assert not clone.state_path.exists()
    entry = clone.workflow_entry
    assert entry == "workflow.py:review"
    assert clone.absolute_spec(entry) == (
        str(root / "workflow.py") + ":review"
    )




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


def test_workspace_updates_run(tmp_path):
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
    assert updated["status"] == "done"
    assert updated["result"] == "Hello!"


def test_workspace_configuration_edits_update_every_assignment(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    workspace = Workspace(root, home=tmp_path / "state")
    workspace.save_model_configuration(
        "writer",
        {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "spec": "openai:gpt-4o-mini",
        },
    )
    workspace.save_model_assignment_profile(
        "review.py:review",
        default="mock",
        lifelines={"Writer": "writer"},
    )

    workspace.save_model_configuration(
        "writer",
        {
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "spec": "openai:gpt-4.1-mini",
        },
    )

    assert workspace.model_assignment_profile("review.py:review")["lifelines"] == {
        "Writer": "writer"
    }
    assert workspace.model_configurations()["writer"]["spec"] == (
        "openai:gpt-4.1-mini"
    )


def test_workspace_validates_local_model_idle_release(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    workspace = Workspace(root, home=tmp_path / "state")

    saved = workspace.save_model_configuration(
        "local-writer",
        {
            "provider": "local",
            "model": "qwen2.5:7b",
            "spec": "local:qwen2.5:7b",
            "idle_timeout": "300.0",
        },
    )

    assert saved["idle_timeout"] == "300"
    assert (
        workspace.model_configurations()["local-writer"]["idle_timeout"]
        == "300"
    )
    with pytest.raises(WorkspaceError, match="only available for local"):
        workspace.save_model_configuration(
            "remote-writer",
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "spec": "openai:gpt-4o-mini",
                "idle_timeout": "300",
            },
        )




def test_workspace_manages_the_visible_specification(tmp_path):
    """`specification.md` is a plain project file the agent maintains."""

    root = tmp_path / "project"
    root.mkdir()
    (root / ".git").mkdir()
    workspace = Workspace(root, home=tmp_path / "state")

    manifest = workspace.initialize_project(name="Review project")

    assert manifest["specification_file"] == "specification.md"
    assert workspace.specification_path == root / "specification.md"
    assert not workspace.specification_path.exists()

    workspace.specification_path.write_text(
        "# Reviewed answer\n\nRequire human approval.\n",
        encoding="utf-8",
    )

    assert "Require human approval." in workspace.specification_path.read_text()
    ignored = (root / ".gitignore").read_text(encoding="utf-8")
    assert "/tutorial-runtime/" in ignored.splitlines()


def test_reinitializing_the_same_path_creates_a_new_private_identity(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    first = Workspace(root, home=tmp_path / "state")
    first_manifest = first.initialize_project(name="First")
    first_directory = first.directory

    first.manifest_path.unlink()
    second = Workspace(root, home=tmp_path / "state")
    second_manifest = second.initialize_project(name="Second")

    assert second_manifest["project_id"] != first_manifest["project_id"]
    assert second.directory != first_directory


def test_a_manifest_without_a_project_id_keeps_its_workspace_after_a_write(
    tmp_path,
):
    """A project older than project_id must not be moved by a config write.

    The workspace key hashes project_id, so writing anything in place of an
    absent one, including the string "None", sends the project to a different
    workspace directory and strands the credentials already saved there.

    One write is enough to catch this. A placeholder is stable once stored, so
    comparing two later writes to each other would pass on the broken code.
    """

    root = tmp_path / "project"
    root.mkdir()
    (root / "workflow.py").touch()
    (root / "zippergen.toml").write_text(
        'schema_version = 1\nname = "demo"\n'
        'specification_file = "workflow.py"\n',
        encoding="utf-8",
    )
    before = Workspace(root, home=tmp_path / "state").directory

    Workspace(root, home=tmp_path / "state").save_model_configuration(
        "writer",
        {"spec": "openai:gpt-4o-mini", "provider": "openai"},
    )

    written = (root / "zippergen.toml").read_text(encoding="utf-8")
    assert "project_id" not in written
    assert Workspace(root, home=tmp_path / "state").directory == before


def test_an_absent_manifest_value_is_never_written_as_text(tmp_path):
    from zippergen.workspace import _toml_string

    with pytest.raises(WorkspaceError):
        _toml_string(None)


def test_workspace_provider_configuration_keeps_secrets_private(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    workspace = Workspace(root, home=tmp_path / "state")
    workspace.update(
        providers={
            "openai": {"kind": "api", "key_env": "OPENAI_API_KEY"},
            "local": {
                "kind": "local",
                "base_url": "http://localhost:11434/v1",
            },
        }
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
    assert "/tutorial-runtime/" in (root / ".gitignore").read_text().splitlines()
    assert workspace.discover_workflows() == ["workflows/answer.py:answer"]

    (root / ".gitignore").write_text("")
    workspace.initialize_project()
    assert "/zippergen/" in (root / ".gitignore").read_text().splitlines()


def test_project_configuration_survives_a_fresh_clone(
    tmp_path,
):
    root = tmp_path / "project"
    root.mkdir()
    (root / "workflow.py").write_text("def sample(): pass\n")
    original = Workspace(root, home=tmp_path / "original-state")
    original.initialize_project(name="Portable project")
    original.select_workflow("workflow.py:sample", cwd=root)
    original.save_model_configuration(
        "reviewer",
        {
            "provider": "anthropic",
            "model": "claude-opus-5",
            "spec": "anthropic:claude-opus-5",
            "check_status": "available",
        },
    )
    original.save_model_configuration(
        "local-writer",
        {
            "provider": "local",
            "model": "qwen2.5:14b",
            "spec": "local:qwen2.5:14b",
            "idle_timeout": "300",
        },
    )
    original.save_model_assignment_profile(
        "workflow.py:sample",
        default="reviewer",
        lifelines={"Writer": "local-writer"},
    )
    original.save_connector_provider_profile("google", {"kind": "google"})
    original.save_connector_provider_profile(
        "telegram", {"kind": "telegram"}
    )
    original.save_connector_configuration(
        "records",
        {
            "provider": "google",
            "kind": "google-sheets",
            "spreadsheet_id": "sheet-123",
            "tab": "Calls",
        },
    )
    original.save_connector_configuration(
        "approvals",
        {
            "provider": "telegram",
            "kind": "telegram",
            "chat_id": "42",
        },
    )
    original.bind_connector("workflow.py:sample", "call-records", "records")
    original.save_connector_assignment_profile(
        "workflow.py:sample",
        lifelines={"Human": "approvals"},
    )
    original.update(
        providers={
            "local": {
                "kind": "local",
                "base_url": "http://gpu:11434/v1",
            }
        }
    )
    original.save_secrets({"ANTHROPIC_API_KEY": "private"})
    original.save_connector_provider_secret(
        "google", "authorized_user_json", "private"
    )
    original.save_connector_provider_secret("telegram", "bot_token", "private")

    clone = Workspace(root, home=tmp_path / "fresh-clone-state")

    assert clone.workflow_entry == "workflow.py:sample"
    assert clone.model_assignment_profile("workflow.py:sample") == {
        "default": "reviewer",
        "lifelines": {"Writer": "local-writer"},
        "actions": {},
    }
    assert clone.model_configurations()["reviewer"]["spec"] == (
        "anthropic:claude-opus-5"
    )
    assert "idle_timeout" not in clone.model_configurations()["local-writer"]
    assert clone.connector_binding_profile("workflow.py:sample") == {
        "call-records": "records"
    }
    assert clone.connector_assignment_profile("workflow.py:sample") == {
        "lifelines": {"Human": "approvals"},
        "actions": {},
    }
    manifest_text = clone.manifest_path.read_text()
    assert "claude-opus-5" in manifest_text
    assert "sheet-123" in manifest_text
    assert "idle_timeout" not in manifest_text
    assert "http://gpu:11434/v1" not in manifest_text
    assert "private" not in manifest_text
