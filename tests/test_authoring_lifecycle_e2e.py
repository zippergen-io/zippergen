from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from zippergen.studio import Studio
from zippergen.workspace import Workspace


WORKFLOW_SOURCE = """\
from zippergen import Lifeline, llm, workflow

User = Lifeline("User")
Writer = Lifeline("Writer")

@llm(
    system="Return one deterministic result.",
    user="Return ready.",
    parse="text",
    outputs=(("result", str),),
)
def prepare_result() -> None: ...

@workflow
def sample() -> str:
    Writer: result = prepare_result()
    Writer(result) >> User(result)
    return result @ User
"""


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def _initialize_git(root: Path) -> None:
    _git(root, "init")
    _git(root, "config", "user.name", "ZipperGen Test")
    _git(root, "config", "user.email", "zippergen-test@example.invalid")


def test_compound_legacy_migration_preserves_every_kind_of_state(
    tmp_path: Path,
) -> None:
    root = tmp_path / "legacy-project"
    root.mkdir()
    (root / "workflow.py").write_text(WORKFLOW_SOURCE, encoding="utf-8")
    workspace = Workspace(root, home=tmp_path / "legacy-home")
    workspace.initialize_project(name="Legacy project")
    workspace.save_specification("Return one deterministic result.")
    fingerprint = workspace.specification_fingerprint()
    request = workspace.save_request(
        kind="implementation",
        prompt="Return one deterministic result.",
        content="Implement the legacy workflow.\n",
        specification_fingerprint=fingerprint,
    )
    workspace.update_request(
        str(request["request_id"]),
        status="awaiting_review",
    )

    accepted = workspace.directory / "accepted"
    (accepted / "sha256-legacy").mkdir(parents=True)
    (accepted / "sha256-legacy" / "workflow.py").write_text(
        "# accepted source snapshot\n",
        encoding="utf-8",
    )
    legacy_refinement = root / ".zippergen" / "pending-refinement.md"
    legacy_refinement.parent.mkdir(parents=True, exist_ok=True)
    refinement_bytes = (
        "Keep this exact prose.\n\nAdd retries for café callers.\n"
    ).encode("utf-8")
    legacy_refinement.write_bytes(refinement_bytes)

    workspace.update(
        current_workflow="workflow.py:sample",
        accepted_reviews={
            "workflow.py:sample": {
                "request_id": request["request_id"],
                "specification_fingerprint": fingerprint,
                "accepted_source": {
                    "files": [
                        {
                            "path": "workflow.py",
                            "role": "entry point",
                            "sha256": hashlib.sha256(
                                WORKFLOW_SOURCE.encode("utf-8")
                            ).hexdigest(),
                        }
                    ]
                },
            }
        },
        deployment_review_overrides=[{"reason": "legacy override"}],
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
        provider_profiles={
            "local": {
                "kind": "local",
                "base_url": "http://legacy-host:11434/v1",
            }
        },
        connector_providers={
            "google": {"kind": "google", "scopes": "gmail.readonly"}
        },
        connector_configurations={
            "mailbox": {
                "provider": "google",
                "kind": "gmail",
                "account": "me",
                "query": "is:unread",
                "check_status": "available",
            }
        },
        connector_bindings={
            "workflow.py:sample": {"incoming-mail": "mailbox"}
        },
        connector_assignments={
            "workflow.py:sample": {
                "lifelines": {"User": "mailbox"},
                "actions": {},
            }
        },
    )
    output: list[str] = []

    studio = Studio(workspace, output_func=output.append)
    studio.welcome()

    assert not accepted.exists()
    archives = list((workspace.home / "trash" / "accepted").iterdir())
    assert len(archives) == 1
    assert (
        archives[0] / "sha256-legacy" / "workflow.py"
    ).read_text(encoding="utf-8") == "# accepted source snapshot\n"
    assert legacy_refinement.read_bytes() == refinement_bytes

    state = workspace.load()
    assert "accepted_reviews" not in state
    assert "deployment_review_overrides" not in state
    assert "current_workflow" not in state
    assert state["model_configurations"]["writer"]["idle_timeout"] == "300"
    assert state["provider_profiles"]["local"]["base_url"] == (
        "http://legacy-host:11434/v1"
    )
    assert state["connector_providers"]["google"]["scopes"] == (
        "gmail.readonly"
    )
    assert state["connector_configurations"]["mailbox"]["check_status"] == (
        "available"
    )

    migrated_request = workspace.load_request(str(request["request_id"]))
    assert migrated_request["workflow_spec"] == "workflow.py:sample"
    assert migrated_request["result_specification_fingerprint"] == fingerprint
    assert migrated_request["accepted_source_files"] == [
        {
            "path": "workflow.py",
            "sha256": hashlib.sha256(WORKFLOW_SOURCE.encode("utf-8")).hexdigest(),
        }
    ]
    assert workspace.workflow_entry == "workflow.py:sample"
    assert workspace.implementation_state()["state"] == "current"

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
        "query": "is:unread",
    }
    assert manifest["connectors"]["bindings"] == {
        "incoming-mail": "mailbox"
    }
    assert manifest["connectors"]["assignments"] == {
        "lifelines": {"User": "mailbox"},
        "actions": {},
    }

    assert any("Archived retired workflow source snapshots" in line for line in output)
    assert any("former private Studio state" in line for line in output)
    assert any("Moved portable model and connector configuration" in line for line in output)
    assert any("zippergen.lock" in line for line in output)
    assert any("left untouched" in line for line in output)


def test_acceptance_migration_does_not_mark_changed_source_current(
    tmp_path: Path,
) -> None:
    root = tmp_path / "changed-project"
    root.mkdir()
    (root / "workflow.py").write_text(
        WORKFLOW_SOURCE + "\n# hand-edited after acceptance\n",
        encoding="utf-8",
    )
    workspace = Workspace(root, home=tmp_path / "changed-home")
    workspace.initialize_project(name="Changed project")
    workspace.save_specification("Return one deterministic result.")
    fingerprint = workspace.specification_fingerprint()
    request = workspace.save_request(
        kind="implementation",
        prompt="Return one deterministic result.",
        content="Implement the legacy workflow.\n",
        specification_fingerprint=fingerprint,
    )
    workspace.update_request(
        str(request["request_id"]),
        status="awaiting_review",
    )
    workspace.update(
        current_workflow="workflow.py:sample",
        accepted_reviews={
            "workflow.py:sample": {
                "request_id": request["request_id"],
                "specification_fingerprint": fingerprint,
                "accepted_source": {
                    "files": [
                        {
                            "path": "workflow.py",
                            "role": "entry point",
                            "sha256": hashlib.sha256(
                                WORKFLOW_SOURCE.encode("utf-8")
                            ).hexdigest(),
                        }
                    ]
                },
            }
        },
    )

    Studio(workspace, output_func=lambda _line: None).welcome()

    assert not workspace.implementation_lock_path.exists()
    assert workspace.implementation_state()["state"] == "external"


def test_fresh_project_full_authoring_flow_survives_a_fresh_clone(
    tmp_path: Path,
    monkeypatch,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("Git is not installed")
    root = tmp_path / "project"
    root.mkdir()
    first_home = tmp_path / "first-home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(first_home))
    workspace = Workspace(root, home=first_home)
    answers = iter(["", "Lifecycle implementation"])
    output: list[str] = []
    studio = Studio(
        workspace,
        input_func=lambda _prompt: next(answers),
        output_func=output.append,
    )
    studio.execute("project init Lifecycle")
    _initialize_git(root)
    studio.execute(
        'workflow edit-spec "Create a deterministic workflow that returns ready."'
    )
    studio.execute(
        'workflow edit-refinement "Clarify that the result is produced by Writer."'
    )

    real_run = subprocess.run
    real_which = shutil.which

    def fake_which(name: str):
        if name == "codex":
            return "/fake/codex"
        return real_which(name)

    def fake_run(arguments, **kwargs):
        if arguments and arguments[0] == "/fake/codex":
            cwd = Path(kwargs["cwd"])
            if cwd.name.startswith("zippergen-refine-spec-"):
                specification = cwd / "specification.md"
                specification.write_text(
                    specification.read_text(encoding="utf-8").rstrip()
                    + "\nThe Writer participant produces the result.\n",
                    encoding="utf-8",
                )
            else:
                (root / "workflow.py").write_text(
                    WORKFLOW_SOURCE,
                    encoding="utf-8",
                )
                workspace.assistant_result_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "verification": "passed",
                            "summary": "Generated and validated the workflow.",
                            "checks": [
                                {
                                    "command": "zippergen validate workflow.py:sample",
                                    "status": "passed",
                                    "detail": "Workflow is valid.",
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
            return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")
        return real_run(arguments, **kwargs)

    monkeypatch.setattr("zippergen.studio.shutil.which", fake_which)
    monkeypatch.setattr("zippergen.studio.subprocess.run", fake_run)

    studio.execute("workflow refine-spec codex")
    assert workspace.refinement_buffer() is None
    assert "Writer participant" in (workspace.specification() or "")

    studio._interactive_offers_enabled = True
    studio.execute("workflow implement codex")
    assert workspace.implementation_state()["state"] == "current"
    assert _git(root, "log", "-1", "--pretty=%s").stdout.strip() == (
        "Lifecycle implementation"
    )

    entry = workspace.workflow_entry
    assert entry == "workflow.py:sample"
    workspace.save_model_configuration(
        "production",
        {
            "provider": "anthropic",
            "model": "claude-opus-5",
            "spec": "anthropic:claude-opus-5",
        },
    )
    workspace.save_model_assignment_profile(
        entry,
        default="production",
        lifelines={},
    )
    workspace.save_model_assignment_profile(
        entry,
        default="mock",
        lifelines={},
        site=True,
    )
    _git(root, "add", "zippergen.toml")
    _git(root, "commit", "-m", "Configure production model")

    studio.execute("run")
    studio.deploy_workflow(["lifecycle-first", "--no-start"])

    clone_root = tmp_path / "clone"
    real_run(
        ["git", "clone", str(root), str(clone_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    clone_home = tmp_path / "clone-home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(clone_home))
    clone = Workspace(clone_root, home=clone_home)
    assert not clone.state_path.exists()
    assert clone.implementation_state()["state"] == "current"
    assert [
        (item["kind"], item["name"], item["command"])
        for item in clone.missing_site_requirements()
    ] == [
        (
            "secret",
            "ANTHROPIC_API_KEY",
            "model provider configure anthropic",
        )
    ]

    clone.save_model_assignment_profile(
        "workflow.py:sample",
        default="mock",
        lifelines={},
        site=True,
    )
    assert clone.missing_site_requirements() == ()
    clone_output: list[str] = []
    clone_studio = Studio(
        clone,
        input_func=lambda _prompt: "",
        output_func=clone_output.append,
    )
    clone_studio.deploy_workflow(["lifecycle-clone", "--no-start"])

    run = workspace.current_run()
    assert run is not None
    assert run["status"] == "done"
    assert any("Development run" in line for line in output)
    assert any("Deployment prepared" in line for line in output)
    assert any("Deployment prepared" in line for line in clone_output)
