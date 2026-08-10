"""Named coding-assistant routing is portable and drives every run mode."""

import json
import subprocess
import threading
import tomllib
from pathlib import Path

import pytest

from zippergen.assistant_backends import AssistantCliCheck
from zippergen.serve import main
from zippergen.workspace import Workspace



def _one_deployment(home, suffix=".json"):
    """The project's one deployment, whatever name was derived for it."""

    found = sorted((home / "deployments").glob(f"*{suffix}"))
    if suffix == ".json":
        found = [p for p in found if not p.name.endswith(".secrets.json")]
    assert len(found) == 1, f"expected one {suffix} deployment file, got {found}"
    return found[0]

ASSISTANT_WORKFLOW = '''
from zippergen import Lifeline, assistant, workflow

Developer = Lifeline("Developer")
Reviewer = Lifeline("Reviewer")

@assistant(instructions="Implement the request.", access="write")
def implement(request: str) -> str: ...

@assistant(instructions="Review the result.", access="read-only")
def review(request: str) -> str: ...

@workflow
def maintenance(request: str @ Developer) -> str:
    Developer: result = implement(request)
    Developer(result) >> Reviewer(result)
    Reviewer: report = review(result)
    Reviewer(report) >> Developer(report)
    return report @ Developer
'''


def _project(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Workspace]:
    root = tmp_path / "project"
    root.mkdir()
    (root / "workflow.py").write_text(ASSISTANT_WORKFLOW, encoding="utf-8")
    home = tmp_path / "home"
    workspace = Workspace(root, home=home)
    workspace.initialize_project(name="assistant-routing")
    workspace.select_workflow("workflow.py:maintenance")
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    monkeypatch.chdir(root)
    return root, home, workspace


def _supported(backend: str) -> AssistantCliCheck:
    return AssistantCliCheck(
        backend=backend,
        executable=f"/tools/{backend}",
        supported=True,
        detail="supports the required safety options",
    )


def test_named_assistant_configuration_and_both_assignment_levels(
    tmp_path, monkeypatch, capsys
):
    root, _home, workspace = _project(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "zippergen.assistant_backends.check_cli_assistant",
        _supported,
    )

    assert main(["assistant", "configure", "coding", "codex"]) == 0
    assert main(["assistant", "configure", "reviewer", "claude"]) == 0
    assert main(["assistant", "assign", "Developer", "coding"]) == 0
    assert main([
        "assistant", "assign", "Developer.implement", "reviewer"
    ]) == 0
    assert main(["assistant", "assign", "Reviewer", "reviewer"]) == 0
    capsys.readouterr()

    manifest = tomllib.loads((root / "zippergen.toml").read_text())
    assert manifest["assistants"]["configurations"] == {
        "coding": {"backend": "codex"},
        "reviewer": {"backend": "claude"},
    }
    assert manifest["assistants"]["assignments"]["lifelines"] == {
        "Developer": "coding",
        "Reviewer": "reviewer",
    }
    assert manifest["assistants"]["assignments"]["actions"] == {
        "Developer.implement": "reviewer"
    }

    assert main(["config", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    resolved = {
        item["target"]: item for item in report["assistants"]["resolved"]
    }
    assert resolved["Developer.implement"] == {
        "target": "Developer.implement",
        "backend": "claude",
        "configuration": "reviewer",
        "source": "action assignment",
        "access": "write",
        "external_tools": "none",
        "shell": "restricted",
    }
    assert resolved["Reviewer.review"]["backend"] == "claude"
    assert resolved["Reviewer.review"]["source"] == "participant assignment"
    assert workspace.assistant_configurations()["coding"] == {
        "backend": "codex"
    }


def test_assistant_check_verifies_selected_cli_safety_options(
    tmp_path, monkeypatch, capsys
):
    _root, _home, _workspace = _project(tmp_path, monkeypatch)
    calls: list[str] = []

    def check(backend: str) -> AssistantCliCheck:
        calls.append(backend)
        return _supported(backend)

    monkeypatch.setattr(
        "zippergen.assistant_backends.check_cli_assistant",
        check,
    )
    main(["assistant", "configure", "coding", "codex"])
    main(["assistant", "assign", "default", "coding"])
    capsys.readouterr()

    assert main(["assistant", "check", "coding"]) == 0
    output = capsys.readouterr().out
    assert calls == ["codex"]
    assert "required safety options" in output


def test_assistant_configuration_and_assignment_are_guided_in_a_terminal(
    tmp_path, monkeypatch
):
    _root, _home, workspace = _project(tmp_path, monkeypatch)
    answers = iter(["coding-agent", "codex", "Developer", "coding-agent"])
    monkeypatch.setattr("zippergen.serve.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    assert main(["assistant", "configure"]) == 0
    assert main(["assistant", "assign"]) == 0

    assert workspace.assistant_configurations() == {
        "coding-agent": {"backend": "codex"}
    }
    assert workspace.assistant_assignment_profile(
        "workflow.py:maintenance"
    )["lifelines"] == {"Developer": "coding-agent"}


def test_assistant_completion_uses_project_names_and_action_targets(
    tmp_path, monkeypatch, capsys
):
    _root, _home, workspace = _project(tmp_path, monkeypatch)
    workspace.save_assistant_configuration("coding-agent", "codex")

    assert main(["__complete", "assistant-configurations"]) == 0
    assert capsys.readouterr().out.splitlines() == ["coding-agent"]
    assert main(["__complete", "assistant-backends"]) == 0
    assert capsys.readouterr().out.splitlines() == ["codex", "claude"]
    assert main(["__complete", "assistant-targets"]) == 0
    targets = capsys.readouterr().out.splitlines()
    assert "default" in targets
    assert "Developer" in targets
    assert "Developer.implement" in targets


def test_assistant_configuration_cannot_be_removed_while_assigned(
    tmp_path, monkeypatch
):
    _root, _home, workspace = _project(tmp_path, monkeypatch)
    main(["assistant", "configure", "coding-agent", "codex"])
    main(["assistant", "assign", "Developer", "coding-agent"])

    with pytest.raises(SystemExit, match="still assigned to: Developer"):
        main(["assistant", "remove", "coding-agent"])

    assert main(["assistant", "unassign", "Developer"]) == 0
    assert main(["assistant", "remove", "coding-agent"]) == 0
    assert workspace.assistant_configurations() == {}


def test_assistant_check_rejects_a_cli_missing_a_required_flag(
    monkeypatch,
):
    from zippergen.assistant_backends import check_cli_assistant

    monkeypatch.setattr(
        "zippergen.assistant_backends.shutil.which",
        lambda _name: "/tools/codex",
    )
    monkeypatch.setattr(
        "zippergen.assistant_backends.subprocess.run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout="--sandbox --cd",
            stderr="",
        ),
    )

    result = check_cli_assistant("codex")

    assert result.supported is False
    assert "--strict-config" in result.detail
    assert "--ignore-user-config" in result.detail


def test_effective_backend_uses_action_and_participant_project_routing(
    tmp_path, monkeypatch
):
    from zippergen import assistant
    from zippergen.assistant_backends import make_cli_assistant_backend

    commands: list[list[str]] = []
    monkeypatch.setattr(
        "zippergen.assistant_backends.shutil.which",
        lambda name: f"/tools/{name}",
    )
    monkeypatch.setattr(
        "zippergen.assistant_backends.subprocess.run",
        lambda command, **_kwargs: (
            commands.append(command)
            or subprocess.CompletedProcess(command, 0, stdout="done", stderr="")
        ),
    )

    @assistant(instructions="Implement it.", access="write")
    def implement(request: str) -> str: ...

    @assistant(instructions="Legacy fallback.", backend="claude", access="write")
    def fixed_backend(request: str) -> str: ...

    backend = make_cli_assistant_backend(
        "claude",
        project_root=tmp_path,
        routes={"Developer": "codex", "Developer.implement": "claude"},
    )
    thread = threading.current_thread()
    old_name = thread.name
    try:
        thread.name = "Developer"
        backend(implement, {"request": "change"})
        backend(fixed_backend, {"request": "change"})
    finally:
        thread.name = old_name

    assert commands[0][0] == "/tools/claude"
    assert commands[1][0] == "/tools/codex"


def test_durable_run_and_deployment_snapshot_project_assistant_routes(
    tmp_path, monkeypatch, capsys
):
    root, home, workspace = _project(tmp_path, monkeypatch)
    workspace.save_assistant_configuration("coding", "codex")
    workspace.save_assistant_configuration("reviewer", "claude")
    workspace.save_assistant_assignment_profile(
        "workflow.py:maintenance",
        default="coding",
        lifelines={"Reviewer": "reviewer"},
        actions={},
    )
    captured: list[tuple[str | None, dict[str, str]]] = []

    def fake_factory(default=None, *, project_root=None, routes=None):
        captured.append((default, dict(routes or {})))

        def backend(action, inputs):
            return {action.outputs[0][0]: f"done:{next(iter(inputs.values()))}"}

        return backend

    monkeypatch.setattr(
        "zippergen.assistant_backends.make_cli_assistant_backend",
        fake_factory,
    )

    assert main(["run", "--input", "request=change", "--yes"]) == 0
    capsys.readouterr()
    assert captured[-1] == ("codex", {"Reviewer": "claude"})

    assert main([
        "run", "--durable", "--input", "request=change", "--yes"
    ]) == 0
    capsys.readouterr()
    run_record = json.loads(next((home / "workspaces").glob("*/runs/*.json")).read_text())
    assert run_record["assistant"] == "codex"
    assert run_record["assistants"] == {"Reviewer": "claude"}
    assert captured[-1] == ("codex", {"Reviewer": "claude"})

    assert main([
        "deploy",
        "--input",
        "request=change",
        "--no-start",
        "--no-bundle",
        "--no-install",
        "--no-setup",
        "--no-doctor",
        "--yes",
    ]) == 0
    capsys.readouterr()
    profile = json.loads(
        _one_deployment(home).read_text()
    )
    assert profile["assistant"] == "codex"
    assert profile["assistants"] == {"Reviewer": "claude"}
    assert profile["source_cwd"] == str(root)


def test_fresh_clone_keeps_routing_and_reports_only_the_missing_cli(
    tmp_path, monkeypatch
):
    root, _home, workspace = _project(tmp_path, monkeypatch)
    workspace.save_assistant_configuration("coding", "codex")
    workspace.save_assistant_assignment_profile(
        "workflow.py:maintenance",
        default="coding",
        lifelines={},
        actions={},
    )
    fresh_home = tmp_path / "fresh-home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(fresh_home))
    monkeypatch.setattr(
        "zippergen.assistant_backends.check_cli_assistant",
        lambda backend: AssistantCliCheck(
            backend, None, False, "executable 'codex' is not on PATH"
        ),
    )
    from zippergen.project_configuration import configuration_report

    report = configuration_report(Workspace(root, home=fresh_home))

    resolved = report["assistants"]["resolved"]
    assert {item["backend"] for item in resolved} == {"codex"}
    assert report["site_facts"] == [
        {
            "kind": "assistant CLI",
            "name": "codex",
            "available": False,
        }
    ]
