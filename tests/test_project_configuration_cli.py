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
    for name, kind in (
        ("openai-main", "openai"),
        ("local-main", "local"),
        ("scripted-main", "scripted"),
        ("approval-bot", "telegram"),
        ("google-work", "google"),
    ):
        values = {"kind": kind}
        if kind == "local":
            values["base_url"] = "http://127.0.0.1:11434/v1"
        workspace.save_provider_connection(name, values)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    monkeypatch.chdir(root)
    return root, workspace


def test_model_configuration_and_assignment_are_ordinary_commands(
    project, capsys
):
    root, workspace = project

    assert main([
        "model", "configure", "writer", "openai-main", "gpt-4o-mini"
    ]) == 0
    assert main(["model", "assign", "Writer", "writer"]) == 0
    capsys.readouterr()

    manifest = tomllib.loads((root / "zippergen.toml").read_text())
    assert manifest["models"]["configurations"]["writer"] == {
        "connection": "openai-main",
        "model": "gpt-4o-mini",
    }
    assert manifest["models"]["assignments"]["lifelines"] == {
        "Writer": "writer"
    }
    assert workspace.model_assignment_profile(
        "workflow.py:email_approval"
    )["lifelines"] == {"Writer": "writer"}


def test_model_and_connector_configuration_share_name_first_grammar():
    _parser, model = _parse_cli_args(
        ["model", "configure", "writer", "openai-main", "gpt-4o-mini"]
    )
    _parser, connector = _parse_cli_args(
        ["connector", "configure", "approval-chat", "approval-bot", "telegram"]
    )

    assert (model.name, model.connection, model.model) == (
        "writer", "openai-main", "gpt-4o-mini"
    )
    assert (connector.name, connector.connection, connector.kind) == (
        "approval-chat",
        "approval-bot",
        "telegram",
    )


def test_model_configuration_is_fully_guided_in_a_terminal(
    project, monkeypatch, capsys
):
    _root, workspace = project
    answers = iter(
        [
            "openai-main",
            "gpt-4o-mini",
            "writer",
            "Writer",
            "writer",
        ]
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("zippergen.serve.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    assert main(["model", "configure"]) == 0
    assert main(["model", "assign"]) == 0

    assert workspace.model_configurations()["writer"]["spec"] == (
        "openai@openai-main:gpt-4o-mini"
    )
    assert workspace.model_assignment_profile(
        "workflow.py:email_approval"
    )["lifelines"] == {"Writer": "writer"}
    output = capsys.readouterr().out
    assert "Available model assignment targets" in output


REJECTED_NAME = "not a name"


def _run_guided(monkeypatch, command, answers):
    """Run one guided dialogue, recording every question it asks."""

    prompts: list[str] = []
    remaining = iter(answers)

    def ask(prompt: str) -> str:
        prompts.append(prompt)
        return next(remaining)

    monkeypatch.setattr("zippergen.serve.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", ask)
    assert main(command) == 0
    return prompts


@pytest.mark.parametrize(
    ("command", "deciding", "name_label", "answers", "family", "saved"),
    [
        (
            ["provider", "configure"],
            "Provider kind",
            "Provider connection name",
            ["openai", REJECTED_NAME, "openai-spare"],
            "provider_connections",
            "openai-spare",
        ),
        (
            ["model", "configure"],
            "Provider connection",
            "Model configuration name",
            ["openai-main", "gpt-4o-mini", REJECTED_NAME, "writer"],
            "model_configurations",
            "writer",
        ),
        (
            ["connector", "configure"],
            "Provider connection",
            "Connector configuration name",
            ["approval-bot", REJECTED_NAME, "approval-chat", "4242"],
            "connector_configurations",
            "approval-chat",
        ),
        (
            ["assistant", "configure"],
            "Assistant backend",
            "Assistant configuration name",
            ["codex", REJECTED_NAME, "coding-agent"],
            "assistant_configurations",
            "coding-agent",
        ),
    ],
    ids=["provider", "model", "connector", "assistant"],
)
def test_every_guided_configure_dialogue_follows_the_same_grammar(
    project, monkeypatch, capsys, command, deciding, name_label, answers,
    family, saved,
):
    """The four setup dialogues are written out separately. They must still
    behave as one.

    Two rules bind them, and both were broken in a single family before this
    test existed: ask the deciding field before the invented name, and treat a
    rejected answer as a typo to correct rather than a reason to stop.
    """

    _root, workspace = project
    workspace.save_provider_secret("approval-bot", "bot_token", "private")

    prompts = _run_guided(monkeypatch, command, answers)

    assert prompts[0].startswith(deciding)
    named = [
        index for index, prompt in enumerate(prompts)
        if prompt.startswith(name_label)
    ]
    assert named and named[0] > 0, "the name must never be the first question"
    assert len(named) == 2, "a rejected name must be asked again"
    assert "must start with a letter or digit" in capsys.readouterr().out
    assert saved in getattr(workspace, family)()


def test_a_guided_prompt_asks_again_after_a_rejected_answer(
    project, monkeypatch, capsys
):
    """A typo at a prompt must not end the dialogue: the person is still there."""

    _root, workspace = project
    answers = iter(
        [
            "local-min",           # a connection that does not exist
            "local-main",
            "qwen3:14b",
            "local-qwen3:14b",     # a colon is not allowed in a name
            "mock",                # reserved by ZipperGen
            "local-qwen3",
        ]
    )
    monkeypatch.setattr("zippergen.serve.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    assert main(["model", "configure"]) == 0

    assert workspace.model_configurations()["local-qwen3"]["spec"] == (
        "local@local-main:qwen3:14b"
    )
    output = capsys.readouterr().out
    assert "Unknown provider connection 'local-min'" in output
    assert "must start with a letter or digit" in output
    assert "'mock' is reserved" in output


def test_guided_scripted_model_asks_for_a_response_file(project, monkeypatch):
    _root, workspace = project
    answers = iter(["scripted-main", "answers.json", "responses"])
    monkeypatch.setattr("zippergen.serve.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    assert main(["model", "configure"]) == 0

    assert workspace.model_configurations()["responses"]["spec"] == (
        "scripted@scripted-main:answers.json"
    )


def test_a_connection_can_fall_back_to_the_provider_environment(
    project, monkeypatch, capsys
):
    _root, workspace = project
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    assert main(
        ["model", "configure", "writer", "openai-main", "gpt-4o-mini"]
    ) == 0

    assert workspace.provider_secret("openai-main", "api_key") is None
    assert main(["config"]) == 0
    assert "openai-main" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("connection", "kind", "values", "missing"),
    [
        ("approval-bot", "telegram", {"chat_id": "42"}, "Telegram bot token"),
        (
            "google-work",
            "google-sheets",
            {"spreadsheet_id": "sheet-1", "tab": "Calls"},
            "Google authorization",
        ),
    ],
)
def test_connector_check_includes_its_provider_credential(
    project, capsys, connection, kind, values, missing
):
    _root, workspace = project
    workspace.save_connector_configuration(
        "selected",
        {"connection": connection, "kind": kind, **values},
    )

    assert main(["connector", "check", "selected"]) == 1
    output = capsys.readouterr().out
    assert missing in output
    assert "missing on this computer" in output


@pytest.mark.parametrize(
    ("connection", "model"),
    [("local-main", "qwen2.5:14b"), ("scripted-main", "answers.json")],
)
def test_credential_free_model_routes_are_available(
    project, capsys, connection, model
):
    root, workspace = project
    if connection == "scripted-main":
        (root / model).write_text("{}", encoding="utf-8")
    workspace.save_model_configuration(
        "writer", {"connection": connection, "model": model}
    )
    workspace.save_model_assignment_profile(
        "workflow.py:email_approval",
        default="mock",
        lifelines={"Writer": "writer"},
    )

    assert main(["config", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    writer = next(
        item
        for item in report["effective_routing"]
        if item["participant"] == "Writer" and item["kind"] == "model"
    )
    assert writer["available"] is True


def test_hand_edited_model_configuration_fails_cleanly(project):
    root, _workspace = project
    manifest = root / "zippergen.toml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8")
        + '\n[models.configurations.broken]\nconnection = "missing"\n',
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="requires connection and model"):
        main(["model", "check", "broken"])
    assert main(["validate"]) == 1


def test_hand_edited_connector_configuration_fails_cleanly(project):
    root, _workspace = project
    manifest = root / "zippergen.toml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8")
        + '\n[connectors.configurations.broken]\n'
        + 'connection = "approval-bot"\nkind = "telegram"\n',
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="missing required field.*chat_id"):
        main(["connector", "check", "broken"])
    assert main(["validate"]) == 1


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
        match=r"zg model configure NAME CONNECTION MODEL",
    ):
        main(["model", "configure"])


def test_configuration_explains_missing_provider_before_prompting(
    tmp_path, monkeypatch
):
    root = tmp_path / "project"
    root.mkdir()
    home = tmp_path / "home"
    Workspace(root, home=home).initialize_project(name="empty")
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    monkeypatch.chdir(root)
    monkeypatch.setattr("zippergen.serve.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt: pytest.fail("there is no valid answer to prompt for"),
    )

    with pytest.raises(SystemExit) as model_error:
        main(["model", "configure"])
    assert "No model-capable provider connection" in str(model_error.value)
    assert "zg provider configure openai-main openai" in str(model_error.value)

    with pytest.raises(SystemExit) as connector_error:
        main(["connector", "configure", "approval-chat"])
    assert "No connector-capable provider connection" in str(
        connector_error.value
    )
    assert "zg provider configure approval-bot telegram" in str(
        connector_error.value
    )


def test_connector_configuration_and_assignment_are_guided_the_same_way(
    project, monkeypatch
):
    _root, workspace = project
    workspace.save_provider_secret("approval-bot", "bot_token", "private")
    answers = iter(
        [
            "approval-bot",
            "approval-chat",
            "4242",
            "Mailbox",
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
    )["lifelines"] == {"Mailbox": "approval-chat"}


def test_model_assignment_rejects_a_target_that_cannot_call_an_llm(project):
    main(["model", "configure", "writer", "openai-main", "gpt-4o-mini"])

    with pytest.raises(SystemExit, match="Unknown model assignment target"):
        main(["model", "assign", "Nobody", "writer"])


def test_model_unassign_and_remove_leave_no_stale_reference(project):
    _root, workspace = project
    main(["model", "configure", "writer", "openai-main", "gpt-4o-mini"])
    main(["model", "assign", "Writer", "writer"])

    assert main(["model", "unassign", "Writer"]) == 0
    assert main(["model", "remove", "writer"]) == 0

    assert "writer" not in workspace.model_configurations()


def test_config_json_resolves_assignments_without_exposing_secrets(
    project, monkeypatch, capsys
):
    main(["model", "configure", "writer", "openai-main", "gpt-4o-mini"])
    main(["model", "assign", "Writer", "writer"])
    capsys.readouterr()
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-printed")

    assert main(["config", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["models"]["resolved"]["overrides"] == {
        "Writer": "openai@openai-main:gpt-4o-mini"
    }
    assert "must-not-be-printed" not in json.dumps(report)


def test_config_check_reports_missing_site_credentials(project, capsys):
    main(["model", "configure", "writer", "openai-main", "gpt-4o-mini"])
    main(["model", "assign", "Writer", "writer"])
    capsys.readouterr()

    assert main(["check"]) == 0
    assert "openai-main" in capsys.readouterr().out


def test_check_exit_code_says_whether_it_ran_not_what_it_found(project, capsys):
    """A missing credential is news, not a command failure.

    An interactive shell shows the last exit code, so a plain report must not
    look like a crash. Scripts that want a gate ask for one with --strict.
    """

    main(["model", "configure", "writer", "openai-main", "gpt-4o-mini"])
    main(["model", "assign", "Writer", "writer"])
    capsys.readouterr()

    assert main(["check"]) == 0
    assert "openai-main" in capsys.readouterr().out

    assert main(["check", "--strict"]) == 1
    assert "openai-main" in capsys.readouterr().out


def test_strict_check_stays_zero_when_the_project_is_ready(project, capsys):
    capsys.readouterr()

    assert main(["check", "--strict"]) == 0


def test_config_reports_an_unassigned_model_credential_without_contacting_it(
    project, monkeypatch, capsys
):
    main(["model", "configure", "unused", "openai-main", "gpt-4o-mini"])
    capsys.readouterr()
    monkeypatch.setattr(
        "zippergen.project_configuration._live_model_check",
        lambda *_args, **_kwargs: pytest.fail("config must remain offline"),
    )

    assert main(["config"]) == 0
    output = capsys.readouterr().out
    assert "openai-main" in output
    assert "Local requirements" in output


def test_provider_set_credential_saves_to_the_named_projects_private_file(
    project, monkeypatch, capsys
):
    _root, workspace = project
    monkeypatch.setattr("getpass.getpass", lambda _prompt: "new-secret")

    assert main(["provider", "set-credential", "openai-main"]) == 0
    assert workspace.provider_secret("openai-main", "api_key") == "new-secret"
    assert str(workspace.secrets_path) in capsys.readouterr().out


def test_reconfiguring_interactively_keeps_existing_values_as_defaults(
    project, monkeypatch
):
    _root, workspace = project
    main([
        "model", "configure", "writer", "local-main", "qwen2.5:14b",
        "--idle-timeout", "300",
    ])
    monkeypatch.setattr("zippergen.serve.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "")

    assert main(["model", "configure", "writer"]) == 0
    assert workspace.model_configurations()["writer"] == {
        "provider": "local",
        "connection": "local-main",
        "model": "qwen2.5:14b",
        "spec": "local@local-main:qwen2.5:14b",
        "idle_timeout": "300",
    }


def test_config_display_and_check_have_distinct_jobs(project, monkeypatch, capsys):
    _root, workspace = project
    workspace.save_model_configuration(
        "writer",
        {
            "connection": "openai-main",
            "model": "gpt-4o-mini",
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

    assert main(["check"]) == 0
    checked = capsys.readouterr().out
    assert "Project readiness" in checked
    assert "Credentials and local tools\n═══════════════════════════" in checked
    assert "openai-main" in checked


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
    workspace.save_provider_secret("approval-bot", "bot_token", "secret")
    workspace.save_connector_configuration(
        "approval-chat",
        {
            "connection": "approval-bot",
            "kind": "telegram",
            "chat_id": "42",
        },
    )

    assert main(["connector", "assign", "Mailbox", "approval-chat"]) == 0
    assert main(["connector", "unassign", "Mailbox"]) == 0
    assert main(["connector", "remove", "approval-chat"]) == 0

    assert "approval-chat" not in workspace.connector_configurations()


def test_human_action_assignment_rejects_a_service_connector(project):
    _root, workspace = project
    workspace.save_connector_configuration(
        "inbox",
        {
            "connection": "google-work",
            "kind": "gmail",
            "account": "me",
            "query": "is:unread",
        },
    )

    with pytest.raises(SystemExit, match="need a Telegram configuration"):
        main(["connector", "assign", "Mailbox", "inbox"])


def test_completion_uses_current_project_names(project, capsys):
    _root, workspace = project
    workspace.save_model_configuration(
        "writer",
        {
            "connection": "openai-main",
            "model": "gpt-4o-mini",
        },
    )

    assert main(["__complete", "model-configurations"]) == 0
    assert capsys.readouterr().out.splitlines() == ["mock", "writer"]
    assert main(["__complete", "model-targets"]) == 0
    targets = capsys.readouterr().out.splitlines()
    assert "Writer" in targets
    assert "Writer.draft_reply" in targets
    assert main(["__complete", "connector-kinds"]) == 0
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
    assert "--base-url" not in options
    assert "--idle-timeout" in options
    assert main(["__complete", "options", "provider", "configure"]) == 0
    assert "--base-url" in capsys.readouterr().out.splitlines()
