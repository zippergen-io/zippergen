"""The CLI makes named project routing visible and manageable end to end."""

from __future__ import annotations

import json
import shutil
import tomllib
from pathlib import Path

import pytest

from zippergen.serve import _parse_cli_args, main
from zippergen.workspace import Workspace


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "email_approval.py"


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    shutil.copy(EXAMPLE, root / "workflow.py")
    home = tmp_path / "home"
    workspace = Workspace(root, home=home)
    workspace.initialize_project(name="configuration-test")
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    monkeypatch.chdir(root)
    return root, workspace


def test_model_configuration_and_assignment_are_ordinary_commands(
    project, capsys
):
    root, workspace = project

    assert main([
        "model", "configure", "writer", "openai:gpt-4o-mini"
    ]) == 0
    assert main(["model", "assign", "Writer", "writer"]) == 0
    capsys.readouterr()

    manifest = tomllib.loads((root / "zippergen.toml").read_text())
    assert manifest["models"]["configurations"]["writer"]["spec"] == (
        "openai:gpt-4o-mini"
    )
    assert manifest["models"]["assignments"]["lifelines"] == {
        "Writer": "writer"
    }
    assert workspace.model_assignment_profile(
        "workflow.py:email_approval"
    )["lifelines"] == {"Writer": "writer"}


def test_model_and_connector_configuration_share_name_first_grammar():
    _parser, model = _parse_cli_args(
        ["model", "configure", "writer", "openai:gpt-4o-mini"]
    )
    _parser, connector = _parse_cli_args(
        ["connector", "configure", "approval-chat", "telegram"]
    )

    assert (model.name, model.spec) == ("writer", "openai:gpt-4o-mini")
    assert (connector.name, connector.connector_provider) == (
        "approval-chat",
        "telegram",
    )


def test_model_assignment_rejects_a_target_that_cannot_call_an_llm(project):
    main(["model", "configure", "writer", "openai:gpt-4o-mini"])

    with pytest.raises(SystemExit, match="Unknown model assignment target"):
        main(["model", "assign", "Nobody", "writer"])


def test_model_unassign_and_remove_leave_no_stale_reference(project):
    _root, workspace = project
    main(["model", "configure", "writer", "openai:gpt-4o-mini"])
    main(["model", "assign", "Writer", "writer"])

    assert main(["model", "unassign", "Writer"]) == 0
    assert main(["model", "remove", "writer"]) == 0

    assert "writer" not in workspace.model_configurations()


def test_config_json_resolves_assignments_without_exposing_secrets(
    project, monkeypatch, capsys
):
    main(["model", "configure", "writer", "openai:gpt-4o-mini"])
    main(["model", "assign", "Writer", "writer"])
    capsys.readouterr()
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-printed")

    assert main(["config", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["models"]["resolved"]["overrides"] == {
        "Writer": "openai:gpt-4o-mini"
    }
    assert "must-not-be-printed" not in json.dumps(report)


def test_config_check_reports_missing_site_credentials(project, capsys):
    main(["model", "configure", "writer", "openai:gpt-4o-mini"])
    main(["model", "assign", "Writer", "writer"])
    capsys.readouterr()

    assert main(["config", "check"]) == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().out


def test_validate_catches_a_stale_project_assignment(project):
    root, _workspace = project
    manifest = root / "zippergen.toml"
    manifest.write_text(
        manifest.read_text()
        + '\n[models.assignments.lifelines]\n"Nobody" = "mock"\n',
        encoding="utf-8",
    )

    assert main(["validate"]) == 1
    assert main(["validate", "workflow.py:email_approval"]) == 1


def test_connector_unassign_and_remove_are_symmetric(project):
    _root, workspace = project
    workspace.save_connector_configuration(
        "approval-chat",
        {"provider": "telegram", "kind": "telegram", "chat_id": "42"},
    )

    assert main(["connector", "assign", "User", "approval-chat"]) == 0
    assert main(["connector", "unassign", "User"]) == 0
    assert main(["connector", "remove", "approval-chat"]) == 0

    assert "approval-chat" not in workspace.connector_configurations()


def test_completion_uses_current_project_names(project, capsys):
    _root, workspace = project
    workspace.save_model_configuration(
        "writer",
        {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "spec": "openai:gpt-4o-mini",
        },
    )

    assert main(["__complete", "model-configurations"]) == 0
    assert capsys.readouterr().out.splitlines() == ["mock", "writer"]
    assert main(["__complete", "model-targets"]) == 0
    targets = capsys.readouterr().out.splitlines()
    assert "Writer" in targets
    assert "Writer.draft_reply" in targets
    assert main(["__complete", "connector-providers"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "telegram",
        "gmail",
        "google-sheets",
    ]


@pytest.mark.parametrize("shell", ["zsh", "bash", "fish"])
def test_completion_scripts_are_available(shell, capsys):
    assert main(["completion", shell]) == 0
    assert "__complete" in capsys.readouterr().out


def test_completion_options_come_from_the_real_parser(capsys):
    assert main(["__complete", "options", "model", "configure"]) == 0
    options = capsys.readouterr().out.splitlines()
    assert "--base-url" in options
    assert "--idle-timeout" in options
