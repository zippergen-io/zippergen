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


def test_model_configuration_is_fully_guided_in_a_terminal(
    project, monkeypatch, capsys
):
    _root, workspace = project
    answers = iter(
        [
            "writer",
            "openai",
            "gpt-4o-mini",
            "Writer",
            "writer",
        ]
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("zippergen.serve.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.setattr("getpass.getpass", lambda _prompt: "private-key")

    assert main(["model", "configure"]) == 0
    assert main(["model", "assign"]) == 0

    assert workspace.model_configurations()["writer"]["spec"] == (
        "openai:gpt-4o-mini"
    )
    assert workspace.development_credential("OPENAI_API_KEY") == "private-key"
    assert "private-key" not in (_root / "zippergen.toml").read_text()
    assert workspace.secrets_path.stat().st_mode & 0o777 == 0o600
    assert workspace.model_assignment_profile(
        "workflow.py:email_approval"
    )["lifelines"] == {"Writer": "writer"}
    output = capsys.readouterr().out
    assert "Available model assignment targets" in output
    assert "Saved OPENAI_API_KEY in private storage" in output


def test_guided_scripted_model_asks_for_a_response_file(project, monkeypatch):
    _root, workspace = project
    answers = iter(["responses", "scripted", "answers.json"])
    monkeypatch.setattr("zippergen.serve.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    assert main(["model", "configure"]) == 0

    assert workspace.model_configurations()["responses"]["spec"] == (
        "scripted:answers.json"
    )


def test_existing_model_environment_credential_is_not_copied(
    project, monkeypatch, capsys
):
    _root, workspace = project
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    monkeypatch.setattr("zippergen.serve.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(
        "getpass.getpass",
        lambda _prompt: pytest.fail("an environment credential must be reused"),
    )

    assert main(
        ["model", "configure", "writer", "openai:gpt-4o-mini"]
    ) == 0

    assert workspace.development_credential("OPENAI_API_KEY") is None
    assert "Found OPENAI_API_KEY in the environment" in capsys.readouterr().out


def test_missing_required_values_do_not_prompt_outside_a_terminal(
    project, monkeypatch
):
    monkeypatch.setattr("zippergen.serve.sys.stdin.isatty", lambda: False)
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt: pytest.fail("a non-interactive command must not prompt"),
    )

    with pytest.raises(
        SystemExit,
        match=r"zg model configure NAME PROVIDER:MODEL",
    ):
        main(["model", "configure"])


def test_connector_configuration_and_assignment_are_guided_the_same_way(
    project, monkeypatch
):
    _root, workspace = project
    workspace.save_connector_provider_secret("telegram", "bot_token", "private")
    answers = iter(
        [
            "approval-chat",
            "telegram",
            "4242",
            "User",
            "approval-chat",
        ]
    )
    monkeypatch.setattr("zippergen.serve.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    assert main(["connector", "configure"]) == 0
    assert main(["connector", "assign"]) == 0

    assert workspace.connector_configurations()["approval-chat"]["chat_id"] == (
        "4242"
    )
    assert workspace.connector_assignment_profile(
        "workflow.py:email_approval"
    )["lifelines"] == {"User": "approval-chat"}


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


def test_config_display_and_check_have_distinct_jobs(project, monkeypatch, capsys):
    _root, workspace = project
    workspace.save_model_configuration(
        "writer",
        {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "spec": "openai:gpt-4o-mini",
        },
    )
    workspace.save_model_assignment_profile(
        "workflow.py:email_approval",
        default="mock",
        lifelines={"Writer": "writer"},
        actions={"Writer.draft_reply": "mock"},
    )
    monkeypatch.setattr(
        "zippergen.project_configuration._live_model_check",
        lambda *_args, **_kwargs: None,
    )

    assert main(["config"]) == 0
    display = capsys.readouterr().out
    assert "Models" in display
    assert "Assistants" in display
    assert "Connectors" in display
    assert "Site" in display
    assert "Writer" in display
    assert "  draft_reply" in display
    assert "Readiness" not in display
    assert "│ Models" in display
    assert "Models\n══════" not in display
    assert "Configurations\n══════════════" in display

    assert main(["config", "check"]) == 1
    checked = capsys.readouterr().out
    assert "Readiness" in checked
    assert "Checks\n══════" in checked
    assert "OPENAI_API_KEY" in checked


def test_empty_family_sections_omit_table_scaffolding(project, capsys):
    assert main(["assistant"]) == 0
    assistants = capsys.readouterr().out
    assert "│ Assistants" in assistants
    assert "No configurations." in assistants
    assert "No assignments." in assistants
    assert "No assistant actions." in assistants
    assert "Name  Backend" not in assistants

    assert main(["connector"]) == 0
    connectors = capsys.readouterr().out
    assert "│ Connectors" in connectors
    assert "No configurations." in connectors
    assert "No assignments or bindings." in connectors
    assert "Name  Kind  Resource" not in connectors


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


def test_human_action_assignment_rejects_a_service_connector(project):
    _root, workspace = project
    workspace.save_connector_configuration(
        "inbox",
        {
            "provider": "google",
            "kind": "gmail",
            "account": "me",
            "query": "is:unread",
        },
    )

    with pytest.raises(SystemExit, match="need a Telegram configuration"):
        main(["connector", "assign", "User", "inbox"])


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
