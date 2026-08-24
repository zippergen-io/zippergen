import hashlib
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
    # The project's local state holds only its identity and stable private
    # address; everything the manifest describes is resolved from the manifest.
    assert sorted(
        path.name for path in (root / ".zippergen").iterdir()
    ) == [".gitignore", "project-id", "workspace-name"]


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


def test_run_record_preserves_typed_inputs_without_private_memory(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    workspace = Workspace(root, home=tmp_path / "state")
    run = workspace.new_run(
        workflow_spec="review.py:review",
        workflow_name="review",
        fingerprint="abc",
        inputs={"coordinates": (1, [2, 3])},
        llm="mock",
    )

    raw = json.loads(workspace.run_path(run["run_id"]).read_text())
    assert "__zippergen_typed_value_v1__" in raw["inputs"]
    restored = Workspace(root, home=tmp_path / "state").load_run(run["run_id"])
    assert restored["inputs"]["coordinates"] == (1, [2, 3])
    assert type(restored["inputs"]["coordinates"]) is tuple


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
    workspace.save_provider_connection("openai-main", {"kind": "openai"})
    workspace.save_model_configuration(
        "writer",
        {
            "connection": "openai-main",
            "model": "gpt-4o-mini",
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
            "connection": "openai-main",
            "model": "gpt-4.1-mini",
        },
    )

    assert workspace.model_assignment_profile("review.py:review")["lifelines"] == {
        "Writer": "writer"
    }
    assert workspace.model_configurations()["writer"]["spec"] == (
        "openai@openai-main:gpt-4.1-mini"
    )


def test_workspace_validates_local_model_idle_release(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    workspace = Workspace(root, home=tmp_path / "state")
    workspace.save_provider_connection(
        "ollama-gpu", {"kind": "local", "base_url": "http://gpu/v1"}
    )
    workspace.save_provider_connection("openai-main", {"kind": "openai"})

    saved = workspace.save_model_configuration(
        "local-writer",
        {
            "connection": "ollama-gpu",
            "model": "qwen2.5:7b",
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
                "connection": "openai-main",
                "model": "gpt-4o-mini",
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


def test_moving_a_project_keeps_its_workspace_and_deployment_name(tmp_path):
    home = tmp_path / "state"
    original_root = tmp_path / "before"
    original_root.mkdir()
    original = Workspace(original_root, home=home)
    original.initialize_project(name="movable")
    workspace_directory = original.directory
    deployment_name = original.directory.name
    assert deployment_name.startswith("before-")
    original.update(current_run="run-before-move")

    moved_root = tmp_path / "after"
    original_root.rename(moved_root)
    moved = Workspace(moved_root, home=home)

    assert moved.directory == workspace_directory
    assert moved.directory.name == deployment_name
    assert moved.load()["current_run"] == "run-before-move"
    assert moved.load()["project_root"] == str(moved_root)


def test_an_existing_path_derived_workspace_keeps_its_address_after_upgrade(
    tmp_path,
):
    home = tmp_path / "state"
    original_root = tmp_path / "call-intake"
    original_root.mkdir()
    original = Workspace(original_root, home=home)
    original.initialize_project(name="call-intake")
    identity = original.project_id_path.read_text().strip()
    workspace_name_path = original.project_state_directory / "workspace-name"
    workspace_name_path.unlink(missing_ok=True)
    old_digest = hashlib.sha256(
        f"{original_root.resolve()}\0{identity}".encode()
    ).hexdigest()[:10]
    old_directory = home / "workspaces" / f"call-intake-{old_digest}"
    old_directory.mkdir(parents=True)

    assert original.directory == old_directory

    moved_root = tmp_path / "renamed"
    original_root.rename(moved_root)
    moved = Workspace(moved_root, home=home)

    assert moved.directory == old_directory


def test_identity_digest_recovers_a_workspace_after_its_prefix_changes(tmp_path):
    home = tmp_path / "state"
    original_root = tmp_path / "original"
    original_root.mkdir()
    original = Workspace(original_root, home=home)
    original.initialize_project(name="original")
    identity = original.project_id_path.read_text().strip()
    digest = hashlib.sha256(identity.encode()).hexdigest()[:10]
    original.workspace_name_path.unlink()
    existing = home / "workspaces" / f"project-{digest}"
    existing.mkdir(parents=True)
    moved_root = tmp_path / "renamed"
    original_root.rename(moved_root)

    moved = Workspace(moved_root, home=home)

    assert moved.directory == existing
    assert moved.workspace_name_path.read_text().strip() == existing.name


def test_a_legacy_project_moved_before_lookup_refuses_an_empty_workspace(tmp_path):
    home = tmp_path / "state"
    original_root = tmp_path / "legacy"
    original_root.mkdir()
    original = Workspace(original_root, home=home)
    original.initialize_project(name="legacy")
    identity = original.project_id_path.read_text().strip()
    original.workspace_name_path.unlink()
    legacy_digest = hashlib.sha256(
        f"{original_root.resolve()}\0{identity}".encode()
    ).hexdigest()[:10]
    legacy_name = f"legacy-{legacy_digest}"
    legacy_directory = home / "workspaces" / legacy_name
    legacy_directory.mkdir(parents=True)

    moved_root = tmp_path / "moved"
    original_root.rename(moved_root)
    moved = Workspace(moved_root, home=home)

    with pytest.raises(WorkspaceError) as raised:
        _ = moved.directory

    message = str(raised.value)
    assert "No workspace was found for this project identity" in message
    assert str(moved.workspace_name_path) in message
    assert not moved.workspace_name_path.exists()
    identity_digest = hashlib.sha256(identity.encode()).hexdigest()[:10]
    assert not (home / "workspaces" / f"moved-{identity_digest}").exists()

    moved.workspace_name_path.write_text(f"{legacy_name}\n")
    assert moved.directory == legacy_directory


def test_a_manifest_without_a_project_id_keeps_its_workspace_after_a_write(
    tmp_path,
):
    """A project without a local identity must not move on a config write.

    The workspace key hashes the local identity when one exists, so inventing
    one sends the project to a different workspace directory and strands the
    credentials already saved there.

    One write is enough to catch this. A placeholder is stable once stored, so
    comparing two later writes to each other would pass on the broken code.
    """

    root = tmp_path / "project"
    root.mkdir()
    (root / "workflow.py").touch()
    (root / "zippergen.toml").write_text(
        'schema_version = 2\nname = "demo"\n'
        'specification_file = "workflow.py"\n',
        encoding="utf-8",
    )
    before = Workspace(root, home=tmp_path / "state").directory

    workspace = Workspace(root, home=tmp_path / "state")
    workspace.save_provider_connection("openai-main", {"kind": "openai"})
    workspace.save_model_configuration(
        "writer", {"connection": "openai-main", "model": "gpt-4o-mini"}
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
    workspace.save_provider_connection("openai-main", {"kind": "openai"})
    workspace.save_provider_secret("openai-main", "api_key", "private-key")
    workspace.save_provider_connection(
        "ollama-local",
        {"kind": "local", "base_url": "http://localhost:11434/v1"},
    )

    environment = workspace.development_provider_environment(
        (
            "openai@openai-main:gpt-4o-mini",
            "local@ollama-local:qwen2.5:7b",
        )
    )

    assert environment == {
        "ZIPPERGEN_PROVIDER_OPENAI_DASH_MAIN_API_KEY": "private-key",
        "ZIPPERGEN_PROVIDER_OLLAMA_DASH_LOCAL_BASE_URL": "http://localhost:11434/v1",
    }
    assert workspace.secrets_path.stat().st_mode & 0o077 == 0
    assert workspace.manifest_path.exists()


def test_reconfiguring_provider_preserves_site_values_and_protects_references(
    tmp_path,
):
    root = tmp_path / "project"
    root.mkdir()
    workspace = Workspace(root, home=tmp_path / "state")
    workspace.save_provider_connection(
        "ollama-local",
        {"kind": "local", "base_url": "http://localhost:11434/v1"},
    )

    workspace.save_provider_connection("ollama-local", {"kind": "local"})

    assert workspace.provider_connections()["ollama-local"]["base_url"] == (
        "http://localhost:11434/v1"
    )
    workspace.save_model_configuration(
        "writer", {"connection": "ollama-local", "model": "qwen2.5:7b"}
    )
    with pytest.raises(WorkspaceError, match="cannot change from local to openai"):
        workspace.save_provider_connection("ollama-local", {"kind": "openai"})


def test_changing_unused_provider_kind_clears_incompatible_private_state(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    workspace = Workspace(root, home=tmp_path / "state")
    workspace.save_provider_connection(
        "service", {"kind": "local", "base_url": "http://localhost:11434/v1"}
    )
    workspace.save_provider_secret("service", "api_key", "obsolete")

    workspace.save_provider_connection("service", {"kind": "telegram"})

    assert workspace.provider_connections()["service"] == {"kind": "telegram"}
    assert workspace.provider_secret("service", "api_key") is None


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
    original.save_provider_connection("anthropic-main", {"kind": "anthropic"})
    original.save_provider_connection(
        "ollama-gpu", {"kind": "local", "base_url": "http://gpu:11434/v1"}
    )
    original.save_provider_connection("google-work", {"kind": "google"})
    original.save_provider_connection("approval-bot", {"kind": "telegram"})
    original.save_model_configuration(
        "reviewer",
        {
            "connection": "anthropic-main",
            "model": "claude-opus-5",
        },
    )
    original.save_model_configuration(
        "local-writer",
        {
            "connection": "ollama-gpu",
            "model": "qwen2.5:14b",
            "idle_timeout": "300",
        },
    )
    original.save_model_assignment_profile(
        "workflow.py:sample",
        default="reviewer",
        lifelines={"Writer": "local-writer"},
    )
    original.save_connector_configuration(
        "records",
        {
            "connection": "google-work",
            "kind": "google-sheets",
            "spreadsheet_id": "sheet-123",
            "tab": "Calls",
        },
    )
    original.save_connector_configuration(
        "approvals",
        {
            "connection": "approval-bot",
            "kind": "telegram",
            "chat_id": "42",
        },
    )
    original.bind_connector("workflow.py:sample", "call-records", "records")
    original.save_connector_assignment_profile(
        "workflow.py:sample",
        lifelines={"Human": "approvals"},
    )
    original.save_provider_secret("anthropic-main", "api_key", "private")
    original.save_provider_secret(
        "google-work", "authorized_user_json", "private"
    )
    original.save_provider_secret("approval-bot", "bot_token", "private")

    clone = Workspace(root, home=tmp_path / "fresh-clone-state")

    assert clone.workflow_entry == "workflow.py:sample"
    assert clone.model_assignment_profile("workflow.py:sample") == {
        "default": "reviewer",
        "lifelines": {"Writer": "local-writer"},
        "actions": {},
    }
    assert clone.model_configurations()["reviewer"]["spec"] == (
        "anthropic@anthropic-main:claude-opus-5"
    )
    assert "idle_timeout" not in clone.model_configurations()["local-writer"]
    assert clone.connector_binding_profile("workflow.py:sample") == {
        "call-records": "records"
    }
    assert clone.connector_assignment_profile("workflow.py:sample") == {
        "default": "",
        "lifelines": {"Human": "approvals"},
        "actions": {},
    }
    manifest_text = clone.manifest_path.read_text()
    assert "claude-opus-5" in manifest_text
    assert "sheet-123" in manifest_text
    assert "idle_timeout" not in manifest_text
    assert "http://gpu:11434/v1" not in manifest_text
    assert "private" not in manifest_text


def test_an_ambiguous_identity_refuses_even_when_one_claimant_is_canonical(
    tmp_path,
):
    """Ambiguity is decided before any preference is applied.

    Checking the canonical name first let one claimant win silently, which is
    the outcome this refusal exists to prevent.
    """

    import hashlib

    home = tmp_path / "home"
    root = tmp_path / "mailbox"
    root.mkdir()
    workspace = Workspace(root, home=home)
    workspace.initialize_project(name="mailbox")
    identity = workspace._project_id()
    assert identity is not None
    digest = hashlib.sha256(identity.encode()).hexdigest()[:10]
    (home / "workspaces" / f"mailbox-{digest}").mkdir(parents=True, exist_ok=True)
    (home / "workspaces" / f"project-{digest}").mkdir(parents=True, exist_ok=True)
    workspace.workspace_name_path.unlink(missing_ok=True)

    with pytest.raises(WorkspaceError) as caught:
        Workspace(root, home=home).directory

    message = str(caught.value)
    assert f"mailbox-{digest}" in message and f"project-{digest}" in message


def test_an_unreadable_identity_is_not_treated_as_an_absent_one(tmp_path):
    """A clone legitimately has none; a corrupt file is a different thing.

    Treating them alike sent credential lookup to a different workspace with
    no indication that anything had happened.
    """

    home = tmp_path / "home"
    root = tmp_path / "mailbox"
    root.mkdir()
    workspace = Workspace(root, home=home)
    workspace.initialize_project(name="mailbox")
    workspace.workspace_name_path.unlink(missing_ok=True)
    workspace.project_id_path.write_bytes(b"\xff\xfe not utf-8")

    with pytest.raises(WorkspaceError) as caught:
        Workspace(root, home=home).directory

    assert "identity file cannot be read" in str(caught.value)


def test_a_clone_without_any_identity_still_resolves(tmp_path):
    """The absent case must keep working: it is how every clone starts."""

    home = tmp_path / "home"
    root = tmp_path / "cloned"
    root.mkdir()
    (root / "zippergen.toml").write_text(
        'schema_version = 2\nname = "cloned"\nspecification_file = "spec.md"\n'
    )

    assert Workspace(root, home=home).directory.name.startswith("cloned-")
