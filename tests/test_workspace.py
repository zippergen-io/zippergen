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


def test_workflow_entry_lives_in_visible_project_manifest(tmp_path):
    root = tmp_path / "project"
    home = tmp_path / "state"
    root.mkdir()
    (root / ".git").mkdir()
    workflow_path = root / "review.py"
    workflow_path.write_text("@workflow\ndef review(): pass\n")

    workspace = Workspace(root, home=home)
    selected = workspace.select_workflow(str(workflow_path) + ":review")

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
    original.select_workflow("workflow.py:review", cwd=root)

    clone = Workspace(root, home=tmp_path / "empty-home")

    assert not clone.state_path.exists()
    entry = clone.workflow_entry
    assert entry == "workflow.py:review"
    assert clone.absolute_spec(entry) == (
        str(root / "workflow.py") + ":review"
    )




def test_workspace_migrates_private_workflow_pointer_into_manifest(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "workflow.py").write_text("@workflow\ndef review(): pass\n")
    workspace = Workspace(root, home=tmp_path / "home")
    workspace.initialize_project()
    state = workspace.default_state()
    state["current_workflow"] = "workflow.py:review"
    workspace.state_path.parent.mkdir(parents=True)
    workspace.state_path.write_text(json.dumps(state))

    result = workspace.migrate_workflow_entry()

    assert result["source"] == "private workspace"
    assert workspace.workflow_entry == "workflow.py:review"
    assert "current_workflow" not in workspace.load()


def test_workspace_migration_uses_one_discovered_workflow(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "workflow.py").write_text("@workflow\ndef review(): pass\n")
    workspace = Workspace(root, home=tmp_path / "home")
    workspace.initialize_project()

    result = workspace.migrate_workflow_entry()

    assert result["source"] == "discovery"
    assert workspace.workflow_entry == "workflow.py:review"


def test_workspace_migration_rejects_nonportable_external_pointer(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    external = tmp_path / "external.py"
    external.write_text("@workflow\ndef review(): pass\n")
    workspace = Workspace(root, home=tmp_path / "home")
    workspace.initialize_project()
    state = workspace.default_state()
    state["current_workflow"] = f"{external}:review"
    workspace.state_path.parent.mkdir(parents=True)
    workspace.state_path.write_text(json.dumps(state))

    with pytest.raises(WorkspaceError, match="workflow_entry in zippergen.toml"):
        workspace.migrate_workflow_entry()

    assert workspace.workflow_entry is None


def test_workspace_discards_legacy_current_store_pointer(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    workspace = Workspace(root, home=tmp_path / "state")
    state = workspace.load()
    state["current_store"] = str(tmp_path / "ambiguous.sqlite")
    workspace.state_path.parent.mkdir(parents=True, exist_ok=True)
    workspace.state_path.write_text(json.dumps(state))

    loaded = workspace.load()

    assert "current_store" not in loaded


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


def test_workspace_migrates_direct_models_to_named_configurations(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    workspace = Workspace(root, home=tmp_path / "state")
    workspace.save_model_profile(
        "review.py:review",
        default="mock",
        lifelines={"Writer": "openai:gpt-4o-mini"},
    )

    assignments = workspace.model_assignment_profile("review.py:review")

    assert assignments == {
        "default": "mock",
        "lifelines": {"Writer": "openai-gpt-4o-mini"},
    }
    assert workspace.model_configurations()["openai-gpt-4o-mini"]["spec"] == (
        "openai:gpt-4o-mini"
    )
    assert workspace.model_profile("review.py:review") == {
        "default": "mock",
        "lifelines": {"Writer": "openai:gpt-4o-mini"},
    }


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

    assert workspace.model_profile("review.py:review")["lifelines"] == {
        "Writer": "openai:gpt-4.1-mini"
    }
    with pytest.raises(WorkspaceError, match="still assigned"):
        workspace.remove_model_configuration("writer")


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


def test_workspace_connector_rename_updates_assignments_and_bindings(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    workspace = Workspace(root, home=tmp_path / "state")
    workspace.save_connector_configuration(
        "approvals",
        {
            "provider": "telegram",
            "kind": "telegram",
            "chat_id": "123",
        },
    )
    workspace.save_connector_assignment_profile(
        "review.py:review",
        lifelines={"Human": "approvals"},
        actions={"Human.escalate": "approvals"},
    )
    workspace.bind_connector(
        "review.py:review",
        "audit-log",
        "approvals",
    )
    workspace.save_connector_secret(
        "approvals", "legacy-token", "private"
    )

    assert workspace.connector_configuration_usage("approvals") == (
        "review.py:review",
    )
    assert workspace.connector_configuration_references("approvals") == (
        ("review.py:review", "action", "Human.escalate"),
        ("review.py:review", "participant", "Human"),
        ("review.py:review", "requirement", "audit-log"),
    )
    with pytest.raises(
        WorkspaceError,
        match=(
            r"still referenced by:.*action Human\.escalate.*"
            r"participant Human.*requirement audit-log"
        ),
    ):
        workspace.remove_connector_configuration("approvals")

    workspace.rename_connector_configuration("approvals", "team-chat")

    assert "approvals" not in workspace.connector_configurations()
    assert "team-chat" in workspace.connector_configurations()
    assert workspace.connector_assignment_profile("review.py:review") == {
        "lifelines": {"Human": "team-chat"},
        "actions": {"Human.escalate": "team-chat"},
    }
    assert workspace.connector_binding_profile("review.py:review") == {
        "audit-log": "team-chat"
    }
    assert workspace.connector_secret(
        "team-chat", "legacy-token"
    ) == "private"


def test_workspace_removing_unused_connector_removes_legacy_secrets(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    workspace = Workspace(root, home=tmp_path / "state")
    workspace.save_connector_configuration(
        "unused",
        {
            "provider": "telegram",
            "kind": "telegram",
            "chat_id": "123",
        },
    )
    workspace.save_connector_secret("unused", "bot_token", "private")

    workspace.remove_connector_configuration("unused")

    assert "unused" not in workspace.connector_configurations()
    assert workspace.connector_secret("unused", "bot_token") is None




def test_workspace_manages_the_visible_specification(tmp_path):
    """`specification.md` is a plain project file the agent maintains."""

    root = tmp_path / "project"
    root.mkdir()
    (root / ".git").mkdir()
    workspace = Workspace(root, home=tmp_path / "state")

    manifest = workspace.initialize_project(name="Review project")

    assert manifest["specification_file"] == "specification.md"
    assert workspace.specification_path == root / "specification.md"
    assert workspace.specification() is None

    workspace.save_specification("# Reviewed answer\n\nRequire human approval.")

    assert "Require human approval." in workspace.specification()
    ignored = (root / ".gitignore").read_text(encoding="utf-8")
    assert "/tutorial-runtime/" in ignored.splitlines()


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
    assert "/tutorial-runtime/" in (root / ".gitignore").read_text().splitlines()
    assert workspace.discover_workflows() == ["workflows/answer.py:answer"]

    (root / ".gitignore").write_text("")
    workspace.initialize_project()
    assert "/zippergen/" in (root / ".gitignore").read_text().splitlines()


def test_project_configuration_survives_a_fresh_clone_and_reports_only_site_gaps(
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
    original.save_provider_profile(
        "local", {"kind": "local", "base_url": "http://gpu:11434/v1"}
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
    assert [
        (item["kind"], item["name"], item["command"])
        for item in clone.missing_site_requirements()
    ] == [
        (
            "secret",
            "ANTHROPIC_API_KEY",
            "model provider configure anthropic",
        ),
        (
            "secret",
            "Google authorization",
            "connector provider configure google",
        ),
        (
            "secret",
            "Telegram bot token",
            "connector provider configure telegram",
        ),
        (
            "site fact",
            "local model endpoint",
            "model provider configure local",
        ),
    ]
    manifest_text = clone.manifest_path.read_text()
    assert "claude-opus-5" in manifest_text
    assert "sheet-123" in manifest_text
    assert "idle_timeout" not in manifest_text
    assert "http://gpu:11434/v1" not in manifest_text
    assert "private" not in manifest_text


def test_legacy_configuration_migration_copies_portable_fields_without_deleting(
    tmp_path,
):
    root = tmp_path / "project"
    root.mkdir()
    workspace = Workspace(root, home=tmp_path / "state")
    workspace.initialize_project()
    workspace.select_workflow("workflow.py:sample", cwd=root)
    workspace.update(
        model_configurations={
            "writer": {
                "provider": "local",
                "model": "qwen2.5:14b",
                "spec": "local:qwen2.5:14b",
                "idle_timeout": "300",
            }
        },
        model_profiles={
            "workflow.py:sample": {
                "default_configuration": "writer",
                "lifeline_configurations": {},
            }
        },
        connector_providers={"google": {"kind": "google", "scopes": "mail"}},
        connector_configurations={
            "mailbox": {
                "provider": "google",
                "kind": "gmail",
                "account": "me",
                "check_status": "available",
            }
        },
        connector_bindings={"workflow.py:sample": {"mail": "mailbox"}},
    )

    result = workspace.migrate_project_configuration()

    assert result["migrated"] is True
    manifest = workspace.project_manifest()
    assert manifest["models"]["configurations"]["writer"] == {
        "provider": "local",
        "model": "qwen2.5:14b",
        "spec": "local:qwen2.5:14b",
    }
    assert manifest["models"]["assignments"]["default"] == "writer"
    assert manifest["connectors"]["providers"] == {
        "google": {"kind": "google"}
    }
    assert manifest["connectors"]["configurations"]["mailbox"] == {
        "provider": "google",
        "kind": "gmail",
        "account": "me",
    }
    assert manifest["connectors"]["bindings"] == {"mail": "mailbox"}
    state = workspace.load()
    assert state["model_configurations"]["writer"]["idle_timeout"] == "300"
    assert state["connector_providers"]["google"]["scopes"] == "mail"
    assert state["connector_configurations"]["mailbox"]["check_status"] == (
        "available"
    )
