from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from zippergen.studio import Studio
from zippergen.studio_git import (
    commit_implementation_unit,
    implementation_commit_unit,
    implementation_status_unit,
    uncommitted_commit_unit_paths,
)
from zippergen.workspace import Workspace


pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="Git unavailable")


WORKFLOW_SOURCE = """\
from zippergen import Lifeline, llm, workflow

User = Lifeline("User")
Writer = Lifeline("Writer")

@llm(
    system="Echo the value.",
    user="{value}",
    parse="text",
    outputs=(("result", str),),
)
def echo(value: str) -> None: ...

@workflow
def sample(value: str @ User) -> str:
    User(value) >> Writer(value)
    Writer: result = echo(value)
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


def _project(tmp_path: Path) -> tuple[Path, Workspace]:
    root = tmp_path / "project"
    root.mkdir()
    (root / "workflow.py").write_text(WORKFLOW_SOURCE, encoding="utf-8")
    workspace = Workspace(root, home=tmp_path / "home")
    workspace.initialize_project()
    workspace.save_specification("Echo the user's request through Writer.")
    workspace.select_workflow("workflow.py:sample", cwd=root)
    workspace.write_implementation_lock(["workflow.py"])
    return root, workspace


def _initialize_git(root: Path) -> None:
    _git(root, "init")
    _git(root, "config", "user.name", "ZipperGen Test")
    _git(root, "config", "user.email", "zippergen-test@example.invalid")


def test_commit_operation_commits_exact_unit_and_preserves_other_staged_work(
    tmp_path: Path,
) -> None:
    root, workspace = _project(tmp_path)
    _initialize_git(root)
    unrelated = root / "notes.txt"
    unrelated.write_text("keep staged but separate\n", encoding="utf-8")
    _git(root, "add", "notes.txt")

    unit = implementation_commit_unit(workspace, include_manifest=True)
    assert unit is not None
    assert set(unit.project_paths) == {
        "specification.md",
        "workflow.py",
        "zippergen.lock",
        "zippergen.toml",
    }

    revision = commit_implementation_unit(unit, "Regenerate workflow")

    assert revision != ""
    committed = set(
        _git(root, "show", "--pretty=format:", "--name-only", "HEAD")
        .stdout.splitlines()
    )
    assert committed == {
        "specification.md",
        "workflow.py",
        "zippergen.lock",
        "zippergen.toml",
    }
    assert _git(root, "diff", "--cached", "--name-only").stdout.splitlines() == [
        "notes.txt"
    ]


def test_uncommitted_lock_warns_until_the_complete_unit_is_committed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, workspace = _project(tmp_path)
    _initialize_git(root)
    _git(root, "add", "workflow.py", "specification.md", "zippergen.toml")
    _git(root, "commit", "-m", "Workflow without portable lock")

    unit = implementation_commit_unit(workspace, include_manifest=True)
    assert unit is not None
    assert uncommitted_commit_unit_paths(unit) == ("zippergen.lock",)

    output: list[str] = []
    studio = Studio(
        workspace,
        input_func=lambda _prompt: "",
        output_func=output.append,
    )
    studio.manage_task([])
    assert any(
        "zippergen.lock" in line and "fresh clone" in line for line in output
    )

    deploy_calls: list[list[str]] = []
    monkeypatch.setattr(
        "zippergen.serve.main",
        lambda arguments: deploy_calls.append(arguments) or 0,
    )
    output.clear()
    studio.deploy_workflow(["sample", "--no-start"])
    assert len(deploy_calls) == 1
    assert any(
        "zippergen.lock" in line and "Deployment will still proceed" in line
        for line in output
    )

    _git(root, "add", "zippergen.lock")
    _git(root, "commit", "-m", "Record portable implementation")

    assert uncommitted_commit_unit_paths(unit) == ()
    output.clear()
    studio.manage_task([])
    assert not any("fresh clone" in line for line in output)

    workspace.save_model_configuration(
        "drafting",
        {
            "provider": "openai",
            "model": "gpt-5-mini",
            "spec": "openai:gpt-5-mini",
        },
    )
    identity_unit = implementation_commit_unit(
        workspace,
        include_manifest="workflow_identity",
    )
    assert identity_unit is not None
    assert "zippergen.toml" not in identity_unit.project_paths
    assert uncommitted_commit_unit_paths(identity_unit) == ()

    workspace.implementation_lock_path.unlink()
    deleted_lock_unit = implementation_status_unit(workspace)
    assert deleted_lock_unit is not None
    assert uncommitted_commit_unit_paths(deleted_lock_unit) == (
        "zippergen.lock",
    )


def test_interactive_offer_uses_an_editable_default_commit_message(
    tmp_path: Path,
) -> None:
    root, workspace = _project(tmp_path)
    _initialize_git(root)
    answers = iter(["", "Implement sample with a clearer message"])
    output: list[str] = []
    studio = Studio(
        workspace,
        input_func=lambda _prompt: next(answers),
        output_func=output.append,
    )
    studio._interactive_offers_enabled = True

    studio._offer_implementation_commit(
        "workflow.py:sample",
        include_manifest=True,
    )

    assert _git(root, "log", "-1", "--pretty=%s").stdout.strip() == (
        "Implement sample with a clearer message"
    )
    assert any("Committed the specification" in line for line in output)


def test_non_git_project_has_no_commit_offer_or_status_warning(
    tmp_path: Path,
) -> None:
    _root, workspace = _project(tmp_path)
    assert implementation_commit_unit(workspace, include_manifest=True) is None

    output: list[str] = []

    def unexpected_prompt(_prompt: str) -> str:
        pytest.fail("a non-Git project must not receive a Git prompt")

    studio = Studio(
        workspace,
        input_func=unexpected_prompt,
        output_func=output.append,
    )
    studio.manage_task([])

    assert not any("Git" in line or "fresh clone" in line for line in output)


def test_commit_unit_stages_removal_of_a_dropped_implementation_file(
    tmp_path: Path,
) -> None:
    root, workspace = _project(tmp_path)
    helper = root / "helper.py"
    helper.write_text("VALUE = 1\n", encoding="utf-8")
    workspace.write_implementation_lock(["workflow.py", "helper.py"])
    _initialize_git(root)

    unit = implementation_commit_unit(workspace, include_manifest=True)
    assert unit is not None
    commit_implementation_unit(unit, "Implement with a generated helper")

    # A later implementation drops the helper; the new lock no longer names it.
    helper.unlink()
    workspace.write_implementation_lock(["workflow.py"])

    unit = implementation_commit_unit(workspace, include_manifest=True)
    assert unit is not None
    assert "helper.py" in unit.project_paths

    commit_implementation_unit(unit, "Regenerate without the helper")

    assert _git(root, "status", "--porcelain").stdout == ""
    tracked = _git(root, "ls-files").stdout.splitlines()
    assert "helper.py" not in tracked
