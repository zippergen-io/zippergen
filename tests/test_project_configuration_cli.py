"""The CLI makes named project routing visible and manageable end to end."""

from __future__ import annotations

import json
import shutil
import tomllib
from pathlib import Path

import pytest

from zippergen.serve import _parse_cli_args, main
from zippergen.workspace import Workspace, WorkspaceError


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


def test_zero_temperature_round_trips_as_a_toml_number(project):
    root, workspace = project

    assert main([
        "model", "configure", "classifier", "openai-main", "gpt-4o-mini",
        "--temperature", "0",
    ]) == 0

    manifest = tomllib.loads((root / "zippergen.toml").read_text())
    assert manifest["models"]["configurations"]["classifier"]["temperature"] == 0
    assert workspace.model_configurations()["classifier"]["temperature"] == "0"


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


@pytest.mark.parametrize(
    "command",
    [
        ["check"],
        ["provider", "check"],
        ["model", "check"],
        ["connector", "check"],
        ["assistant", "check"],
    ],
    ids=["project", "provider", "model", "connector", "assistant"],
)
def test_a_check_reports_by_default_and_only_gates_with_strict(
    project, monkeypatch, capsys, command
):
    """A check tells you what it found. It does not fail the shell for it.

    Nothing went wrong when a check reports a missing credential: the command
    was asked to look, and it looked. Only a script asking for a gate, with
    --strict, gets a non-zero exit. Each family decides its own readiness, so
    the rule is asserted for all of them at once.
    """

    _root, workspace = project
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "zippergen.assistant_backends.shutil.which", lambda _name: None
    )
    workspace.save_model_configuration(
        "writer", {"connection": "openai-main", "model": "gpt-4o-mini"}
    )
    workspace.save_connector_configuration(
        "approval-chat",
        {"connection": "approval-bot", "kind": "telegram", "chat_id": "42"},
    )
    workspace.save_assistant_configuration("coding-agent", "codex")

    assert main(command) == 0, "a check must not fail the shell on its own"
    capsys.readouterr()
    assert main([*command, "--strict"]) == 1, "--strict is the gate"


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

    assert main(["connector", "check", "selected", "--strict"]) == 1
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


def test_config_reports_every_effective_model_setting(project, capsys):
    _root, workspace = project
    workspace.save_model_configuration(
        "writer",
        {
            "connection": "local-main",
            "model": "qwen3:14b",
            "temperature": "0.4",
            "max_tokens": "4096",
            "timeout": "120",
            "idle_timeout": "30",
        },
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

    assert writer["settings"] == {
        "temperature": 0.4,
        "max_tokens": 4096,
        "timeout": 120.0,
        "idle_timeout": 30.0,
    }
    configured = next(
        item for item in report["models"]["configurations"]
        if item["name"] == "writer"
    )
    assert configured["max_tokens"] == "4096"
    assert configured["timeout"] == "120"

    assert main(["model"]) == 0
    output = capsys.readouterr().out
    assert "Max tokens" in output
    assert "Timeout" in output
    assert "4096" in output
    assert "120" in output


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
        "zippergen.configuration_checks._live_model_check",
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
        "--idle-timeout", "300", "--temperature", "0",
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
        "temperature": "0",
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
        "zippergen.configuration_checks._live_model_check",
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
    assert "Name  Kind  Resource" not in connectors
    # Slots are what the workflow offers, not what has been filled in, so the
    # table stays useful before anything is assigned: it is the list of names
    # you may type.
    assert "Mailbox.approve_reply" in connectors
    assert "not assigned" in connectors


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
    assert "--temperature" in options
    assert main(["__complete", "options", "provider", "configure"]) == 0
    assert "--base-url" in capsys.readouterr().out.splitlines()


def test_every_assignable_family_offers_the_same_three_levels(project):
    """Models, assistants and connectors must agree on what can be assigned.

    Each family is written separately, so the shared grammar -- one default
    for the whole workflow, one per participant, one per exact action -- is
    only real if something asserts it. Connectors lacked the default level
    until this test existed.
    """

    from zippergen.completion import completion_candidates

    levels = {}
    for kind in ("model-targets", "connector-targets", "assistant-targets"):
        names = completion_candidates(kind)
        levels[kind] = {
            "default": "default" in names,
            "participant": any(
                "." not in name and name != "default" for name in names
            ),
            "action": any("." in name for name in names),
        }

    assert levels["model-targets"] == {
        "default": True, "participant": True, "action": True
    }
    assert levels["connector-targets"] == {
        "default": True, "participant": True, "action": True
    }
    # This workflow has no @assistant action, so only the default exists here.
    assert levels["assistant-targets"]["default"] is True


def test_a_value_given_on_the_command_line_is_checked_before_more_questions(
    project, monkeypatch
):
    """A wrong argument must not cost you an unrelated prompt first.

    `zg connector assign Gatekeeper` used to ask which configuration to use
    and only then say the target was unknown.
    """

    _root, workspace = project
    workspace.save_provider_secret("approval-bot", "bot_token", "private")
    workspace.save_connector_configuration(
        "approval-chat",
        {"connection": "approval-bot", "kind": "telegram", "chat_id": "42"},
    )
    monkeypatch.setattr("zippergen.serve.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt: pytest.fail("a known-bad argument must not prompt"),
    )

    with pytest.raises(SystemExit) as error:
        main(["connector", "assign", "Nobody"])

    assert "Unknown connector target 'Nobody'" in str(error.value)


def test_a_missing_google_library_is_reported_before_authorizing(
    project, monkeypatch, capsys
):
    """The optional extra is part of readiness, not a surprise at the browser.

    Credentials and code are two different ways of being unready, and only one
    of them used to be checked.
    """

    _root, workspace = project
    monkeypatch.setattr(
        "zippergen.configuration_checks.google_support_installed",
        lambda: False,
    )

    assert main(["provider", "check", "--strict"]) == 1

    output = capsys.readouterr().out
    assert "google support installed" in output
    assert "not installed here" in output


def test_google_support_is_only_mentioned_when_google_is_configured(
    project, capsys
):
    """A project with no Google connection must not be told about the extra."""

    _root, workspace = project
    workspace.remove_provider_connection("google-work")

    assert main(["provider", "check"]) == 0

    assert "google support" not in capsys.readouterr().out


def _fake_google_browser(monkeypatch, tmp_path):
    """Stand in for the browser flow, returning a credential to be stored."""

    from zippergen.google_auth import GoogleAuthorization

    client = tmp_path / "google-client.json"
    client.write_text(json.dumps({"installed": {
        "client_id": "example.apps.googleusercontent.com",
        "client_secret": "private-client-secret",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    }}))
    monkeypatch.setattr("builtins.input", lambda _prompt: str(client))
    monkeypatch.setattr(
        "zippergen.google_auth.authorize_google_client_result",
        lambda value, *, scopes: GoogleAuthorization(
            authorized_user_json=json.dumps({
                "client_id": "example.apps.googleusercontent.com",
                "refresh_token": "private-refresh-token",
            }),
            granted_scopes=tuple(scopes),
            client_id="example.apps.googleusercontent.com",
        ),
    )


def test_authorizing_inside_a_project_saves_without_a_copy_paste(
    project, monkeypatch, capsys, tmp_path
):
    """The machine you authorize on is usually the one that needs it.

    Printing a live refresh token for somebody to paste back into the same
    computer puts a credential through the screen and the shell history for
    no reason.
    """

    _root, workspace = project
    _fake_google_browser(monkeypatch, tmp_path)

    assert main([
        "provider", "authorize", "google-work", "--scopes", "gmail.readonly"
    ]) == 0

    output = capsys.readouterr().out
    assert workspace.provider_secret("google-work", "authorized_user_json")
    assert "private-refresh-token" not in output
    assert "provider accept" not in output


def test_handoff_still_prints_for_another_computer(
    project, monkeypatch, capsys, tmp_path
):
    """A server has no browser, so the laptop must still be able to hand over."""

    _root, workspace = project
    _fake_google_browser(monkeypatch, tmp_path)

    assert main([
        "provider", "authorize", "google-work",
        "--scopes", "gmail.readonly", "--handoff",
    ]) == 0

    output = capsys.readouterr().out
    assert "provider accept" in output
    assert workspace.provider_secret("google-work", "authorized_user_json") is None


def test_renaming_a_provider_connection_takes_its_credential_with_it(
    project, capsys
):
    """The credential is keyed by the connection name.

    A rename that only edited the manifest would strand it, and quietly send
    somebody back through a Google browser flow to get it again.
    """

    root, workspace = project
    workspace.save_provider_secret("approval-bot", "bot_token", "private")
    workspace.save_connector_configuration(
        "approval-chat",
        {"connection": "approval-bot", "kind": "telegram", "chat_id": "42"},
    )
    workspace.save_model_configuration(
        "writer", {"connection": "openai-main", "model": "gpt-4o-mini"}
    )

    assert main(["provider", "rename", "approval-bot", "alerts-bot"]) == 0
    capsys.readouterr()

    assert workspace.provider_secret("alerts-bot", "bot_token") == "private"
    assert workspace.provider_secret("approval-bot", "bot_token") is None
    assert "alerts-bot" in workspace.provider_connections()
    assert "approval-bot" not in workspace.provider_connections()
    assert workspace.connector_configurations()["approval-chat"][
        "connection"
    ] == "alerts-bot"
    manifest = tomllib.loads((root / "zippergen.toml").read_text())
    assert "alerts-bot" in manifest["providers"]["connections"]


def test_renaming_a_local_connection_keeps_its_endpoint(project):
    """The endpoint is site state, keyed the same way, so it moves too."""

    _root, workspace = project

    assert main(["provider", "rename", "local-main", "gpu-box"]) == 0

    assert workspace.provider_connections()["gpu-box"]["base_url"] == (
        "http://127.0.0.1:11434/v1"
    )


@pytest.mark.parametrize(
    ("family", "configure", "assign", "lookup"),
    [
        ("model", ("model", "configure", "old", "openai-main", "gpt-4o-mini"),
         ("model", "assign", "Writer", "old"), "model_assignment_profile"),
        ("connector", None, None, "connector_assignment_profile"),
    ],
    ids=["model", "connector"],
)
def test_a_rename_repoints_every_assignment(
    project, capsys, family, configure, assign, lookup
):
    """A rename must leave nothing pointing at the old name."""

    _root, workspace = project
    if family == "model":
        assert main(list(configure)) == 0
        assert main(list(assign)) == 0
    else:
        workspace.save_provider_secret("approval-bot", "bot_token", "private")
        workspace.save_connector_configuration(
            "old",
            {"connection": "approval-bot", "kind": "telegram", "chat_id": "42"},
        )
        assert main(["connector", "assign", "Mailbox", "old"]) == 0
    capsys.readouterr()

    assert main([family, "rename", "old", "new"]) == 0

    profile = getattr(workspace, lookup)("workflow.py:email_approval")
    assert "old" not in profile["lifelines"].values()
    assert "new" in profile["lifelines"].values()


@pytest.mark.parametrize(
    ("command", "label"),
    [
        (["provider", "authorize"], "Google connection"),
        (["provider", "accept"], "Google connection"),
        (["workflow", "select"], "Workflow"),
        (["completion"], "Shell"),
    ],
    ids=["authorize", "accept", "select", "completion"],
)
def test_no_user_facing_command_dies_on_a_missing_argument(
    project, monkeypatch, command, label
):
    """Every command a person types should ask, not print a usage line.

    argparse's own error stops with `the following arguments are required`.
    These commands each know the answers they accept, so they offer them.
    """

    prompts: list[str] = []
    monkeypatch.setattr("zippergen.serve.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(
        "builtins.input", lambda prompt: (prompts.append(prompt), "")[1]
    )
    monkeypatch.setattr(
        "getpass.getpass", lambda prompt: (prompts.append(prompt), "")[1]
    )

    try:
        main(command)
    except SystemExit as exit_code:
        assert "required: " not in str(exit_code), "argparse refused instead"

    assert any(label.casefold() in prompt.casefold() for prompt in prompts)


def test_a_single_candidate_is_offered_as_the_default(project, monkeypatch):
    """One candidate is its own suggestion, so it is shown in brackets."""

    prompts: list[str] = []
    monkeypatch.setattr("zippergen.serve.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(
        "builtins.input", lambda prompt: (prompts.append(prompt), "")[1]
    )
    monkeypatch.setattr(
        "getpass.getpass", lambda prompt: (prompts.append(prompt), "")[1]
    )

    try:
        main(["provider", "accept"])
    except SystemExit:
        pass

    assert any("[google-work]" in prompt for prompt in prompts)


def test_effective_routing_says_what_runs_and_where_to_change_it(
    project, capsys
):
    """The tables answer a different question from the checks.

    `zg check` answers "can this run at all". These answer "what exactly will
    it use, and which level do I edit" -- which previously had to be
    reconstructed by reading three tables and knowing the precedence rule.
    """

    _root, workspace = project
    workspace.save_model_configuration(
        "writer", {"connection": "openai-main", "model": "gpt-4o-mini"}
    )
    assert main(["model", "assign", "Writer", "writer"]) == 0
    capsys.readouterr()

    assert main(["model"]) == 0

    routing = capsys.readouterr().out.split("Effective routing")[1]
    assert "Writer" in routing
    assert "draft_reply" in routing
    assert "writer" in routing
    assert "participant" in routing, "the level to edit must be visible"


def test_a_participant_falling_back_to_mock_is_visible_without_running(
    project, capsys
):
    """The silent-fake case is the one worth seeing before execution."""

    _root, _workspace = project

    assert main(["model"]) == 0

    routing = capsys.readouterr().out.split("Effective routing")[1]
    assert "mock" in routing
    assert "default" in routing, "it must say the value came from the default"


def test_routing_separates_configured_from_actually_reached(project, capsys):
    """Offline and live are different claims, so they get different marks."""

    from zippergen.project_configuration import _routing_status
    from zippergen.rendering import TerminalRenderer

    renderer = TerminalRenderer(lambda _text: None, color=False)

    reached = _routing_status(renderer, {"available": True, "verified": True})
    configured = _routing_status(renderer, {"available": True, "verified": False})
    broken = _routing_status(renderer, {"available": False, "verified": True})

    assert reached != configured, "a live check must not look like an offline one"
    assert broken not in {reached, configured}


@pytest.mark.parametrize(
    "command",
    [["config"], ["check"], ["model"], ["connector"], ["assistant"]],
    ids=["config", "check", "model", "connector", "assistant"],
)
def test_every_view_uses_one_routing_grammar(project, capsys, command):
    """Five commands show routing. They must not each invent a layout.

    The grammar is: a status marker in the first narrow column, the
    participant, what it is, and `From` naming the key you would type after
    `assign`.
    """

    _root, workspace = project
    workspace.save_model_configuration(
        "writer", {"connection": "openai-main", "model": "gpt-4o-mini"}
    )
    assert main(["model", "assign", "Writer", "writer"]) == 0
    capsys.readouterr()

    assert main(command) in (0, 1)

    output = capsys.readouterr().out
    assert "Effective routing" in output, "every routing view carries the heading"
    table = output.split("Effective routing")[1]
    header = next(
        (line for line in table.splitlines() if "Participant" in line), None
    )
    if header is None:
        # This workflow has nothing of that family to route; the heading and
        # its empty line are the whole contract there.
        return
    assert header.lstrip().startswith("Participant"), (
        "the status marker column carries no heading"
    )
    assert header.rstrip().endswith("From"), "From is the last column"


def test_a_rename_interrupted_before_the_switch_still_works(project, monkeypatch):
    """Three files cannot be written atomically, so order is the guarantee.

    Private values are copied under the new name before the manifest switches,
    so a crash at the worst moment leaves a project that still works under the
    old name. Moving them instead would leave the manifest naming a connection
    whose credential had already gone.
    """

    _root, workspace = project
    workspace.save_provider_secret("approval-bot", "bot_token", "private")

    def stop_before_switch(self, **_kw):
        raise KeyboardInterrupt("interrupted at the worst moment")

    monkeypatch.setattr(
        type(workspace), "_write_project_configuration", stop_before_switch
    )

    with pytest.raises(KeyboardInterrupt):
        workspace.rename_provider_connection("approval-bot", "alerts-bot")

    monkeypatch.undo()
    assert workspace.provider_secret("approval-bot", "bot_token") == "private", (
        "the name the manifest still points at must keep working"
    )
    assert "approval-bot" in workspace.provider_connections()


def test_a_completed_rename_leaves_no_duplicate_behind(project):
    """Cleanup is the third step, and it must actually happen."""

    _root, workspace = project
    workspace.save_provider_secret("approval-bot", "bot_token", "private")

    workspace.rename_provider_connection("approval-bot", "alerts-bot")

    assert workspace.provider_secret("alerts-bot", "bot_token") == "private"
    assert workspace.provider_secret("approval-bot", "bot_token") is None
    assert "approval-bot" not in workspace.provider_connections()


def _interrupt_after_the_switch(workspace, monkeypatch, rename):
    """Run a rename that dies immediately after the manifest is written."""

    real_write = type(workspace)._write_project_configuration

    def stop_after_the_switch(self, **kwargs):
        real_write(self, **kwargs)
        raise KeyboardInterrupt("interrupted during cleanup")

    monkeypatch.setattr(
        type(workspace), "_write_project_configuration", stop_after_the_switch
    )
    with pytest.raises(KeyboardInterrupt):
        rename()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "channel",
    ["secret", "site-endpoint", "model-override"],
)
def test_rerunning_an_interrupted_rename_finishes_every_storage_channel(
    project, monkeypatch, channel
):
    """Interruption after the switch is the one window copy-first leaves open.

    Each private channel is cleaned up by a different line, and testing only
    one of them is how a recovery boundary hides a gap.
    """

    _root, workspace = project

    if channel == "model-override":
        workspace.save_model_configuration(
            "old-model",
            {"connection": "local-main", "model": "qwen3:14b", "idle_timeout": 30},
        )
        assert workspace.load()["model_configuration_overrides"].get("old-model")
        _interrupt_after_the_switch(
            workspace,
            monkeypatch,
            lambda: workspace.rename_model_configuration("old-model", "new-model"),
        )
        overrides = workspace.load()["model_configuration_overrides"]
        assert set(overrides) >= {"old-model", "new-model"}, "copied, not moved"

        assert workspace.rename_model_configuration("old-model", "new-model") == (
            "new-model"
        )

        overrides = workspace.load()["model_configuration_overrides"]
        assert "old-model" not in overrides
        assert overrides["new-model"]
        return

    if channel == "secret":
        workspace.save_provider_secret("approval-bot", "bot_token", "private")
    else:
        workspace.save_provider_connection(
            "approval-bot", {"kind": "telegram", "base_url": "http://x:1/v1"}
        )

    _interrupt_after_the_switch(
        workspace,
        monkeypatch,
        lambda: workspace.rename_provider_connection("approval-bot", "alerts-bot"),
    )

    if channel == "secret":
        assert workspace.provider_secret("approval-bot", "bot_token") == "private"
        assert workspace.provider_secret("alerts-bot", "bot_token") == "private"
    else:
        overrides = workspace.load()["provider_connection_overrides"]
        assert set(overrides) >= {"approval-bot", "alerts-bot"}, "copied, not moved"

    assert workspace.rename_provider_connection("approval-bot", "alerts-bot") == (
        "alerts-bot"
    )

    if channel == "secret":
        assert workspace.provider_secret("approval-bot", "bot_token") is None
        assert workspace.provider_secret("alerts-bot", "bot_token") == "private"
    else:
        overrides = workspace.load()["provider_connection_overrides"]
        assert "approval-bot" not in overrides
        assert overrides["alerts-bot"]["base_url"] == "http://x:1/v1"


def test_recovery_reruns_even_when_the_command_had_stray_whitespace(
    project, monkeypatch
):
    """The identical command must reach the branch it reached the first time."""

    _root, workspace = project
    workspace.save_provider_secret("approval-bot", "bot_token", "private")
    _interrupt_after_the_switch(
        workspace,
        monkeypatch,
        lambda: workspace.rename_provider_connection("approval-bot", " alerts-bot "),
    )

    assert workspace.rename_provider_connection(
        "approval-bot", " alerts-bot "
    ) == "alerts-bot"
    assert workspace.provider_secret("approval-bot", "bot_token") is None


def test_recovery_does_not_excuse_a_genuine_typo(project):
    """Only an actual leftover counts as an interrupted rename."""

    _root, workspace = project

    with pytest.raises(WorkspaceError, match="does not exist"):
        workspace.rename_provider_connection("never-existed", "approval-bot")


@pytest.mark.parametrize(
    ("orphan_value", "existing_value"),
    [("unrelated", "different"), ("shared", "shared")],
    ids=["different-values", "the-same-key-in-both"],
)
def test_an_unrelated_orphan_credential_is_never_silently_deleted(
    project, orphan_value, existing_value
):
    """Only a recorded rename authorises the cleanup, not matching values.

    One API key shared by two connections looks exactly like a half-finished
    rename if the evidence is value equality. It is common enough to share a
    key that equality cannot establish provenance, so a marker written before
    the copy is what says a rename was under way.
    """

    _root, workspace = project
    secrets = workspace.load_secrets()
    secrets["provider:orphan:api_key"] = orphan_value
    secrets["provider:approval-bot:api_key"] = existing_value
    workspace.save_secrets(secrets)

    with pytest.raises(WorkspaceError, match="does not exist"):
        workspace.rename_provider_connection("orphan", "approval-bot")

    assert workspace.load_secrets()["provider:orphan:api_key"] == orphan_value


def test_a_rename_with_stray_whitespace_actually_renames(project):
    """The guard normalised the old name; the mutation did not, and silently
    reported success while changing nothing."""

    _root, workspace = project
    workspace.save_provider_secret("approval-bot", "bot_token", "private")

    assert workspace.rename_provider_connection(
        " approval-bot ", "alerts-bot"
    ) == "alerts-bot"

    assert "approval-bot" not in workspace.provider_connections()
    assert "alerts-bot" in workspace.provider_connections()
    assert workspace.provider_secret("alerts-bot", "bot_token") == "private"
    assert workspace.provider_secret("approval-bot", "bot_token") is None


def test_the_rename_marker_is_cleared_when_the_rename_finishes(project):
    """A marker left behind would authorise a later, unrelated cleanup."""

    _root, workspace = project
    workspace.save_provider_secret("approval-bot", "bot_token", "private")

    workspace.rename_provider_connection("approval-bot", "alerts-bot")

    assert workspace.load().get("rename_in_progress") is None


def test_an_unfinished_rename_cannot_be_overwritten(
    project, monkeypatch
):
    """A second rename must not erase the recovery proof for the first."""

    _root, workspace = project
    workspace.save_provider_secret("approval-bot", "bot_token", "private")
    _interrupt_after_the_switch(
        workspace,
        monkeypatch,
        lambda: workspace.rename_provider_connection(
            "approval-bot", "alerts-bot"
        ),
    )
    marker = workspace.load()["rename_in_progress"]

    with pytest.raises(WorkspaceError, match="rename is unfinished"):
        workspace.rename_provider_connection("openai-main", "openai-backup")

    assert workspace.load()["rename_in_progress"] == marker
    assert workspace.rename_provider_connection(
        "approval-bot", "alerts-bot"
    ) == "alerts-bot"
    assert workspace.load().get("rename_in_progress") is None


def test_a_malformed_rename_marker_has_an_explicit_recovery_path(project):
    """Private-state corruption must refuse safely without stranding the user."""

    _root, workspace = project
    workspace.update(rename_in_progress="not-a-rename")

    with pytest.raises(WorkspaceError) as raised:
        workspace.rename_provider_connection("approval-bot", "alerts-bot")

    message = str(raised.value)
    assert str(workspace.state_path) in message
    assert "'rename_in_progress' key" in message
    assert "remove only" in message
    assert workspace.load()["rename_in_progress"] == "not-a-rename"
    assert "approval-bot" in workspace.provider_connections()
    assert "alerts-bot" not in workspace.provider_connections()


# A value stored in the project belongs to the project, so every command that
# uses it reads it from there. Wiring only `deploy` to it left `run` asking for
# answers already written down, and left `zg config` silent about a section of
# the file it exists to show.

_REMEMBERING_WORKFLOW = '''
from zippergen import Lifeline, pure, workflow
from zippergen.deployment import DeploymentField, DeploymentSpec

A = Lifeline("A")

zippergen_deployment = DeploymentSpec(
    description="under test",
    fields=(
        DeploymentField("recipient", "Address", target="input", required=True),
        DeploymentField("rounds", "Rounds", target="input", default=3),
    ),
)


@pure
def echo(recipient: str, rounds: int) -> str:
    return f"{recipient} x{rounds}"


@workflow
def remembering(recipient: str @ A, rounds: int @ A) -> str:
    A: out = echo(recipient, rounds)
    return out @ A
'''


def _remembering_project(tmp_path, monkeypatch):
    from zippergen.workspace import Workspace

    home = tmp_path / "home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    project = tmp_path / "project"
    project.mkdir()
    (project / "workflow.py").write_text(_REMEMBERING_WORKFLOW)
    workspace = Workspace(project, home=home)
    workspace.initialize_project(name="remembering")
    workspace.select_workflow("workflow.py:remembering", cwd=project)
    return workspace


def _collect(workspace, provided):
    import os

    from zippergen.durable_runs import collect_workflow_inputs
    from zippergen.workflow_io import load_workflow_spec

    previous = os.getcwd()
    try:
        os.chdir(workspace.root)
        workflow, module = load_workflow_spec("workflow.py:remembering")
        return collect_workflow_inputs(
            workflow,
            module,
            provided,
            interactive=False,
            workspace=workspace,
        )
    finally:
        os.chdir(previous)


def test_a_run_records_its_answers_in_the_project(tmp_path, monkeypatch):
    workspace = _remembering_project(tmp_path, monkeypatch)

    _collect(workspace, {"recipient": "alice@example.org"})

    assert workspace.configuration_values() == {
        "recipient": "alice@example.org",
        "rounds": 3,
    }


def test_a_later_run_needs_no_answer_it_already_has(tmp_path, monkeypatch):
    """Without this, `run` asks again for a value `deploy` had written down."""

    workspace = _remembering_project(tmp_path, monkeypatch)
    workspace.write_configuration_values({"recipient": "kept@example.org"})

    collected = _collect(workspace, {})

    assert collected["recipient"] == "kept@example.org"
    assert collected["rounds"] == 3, "a declared default still applies"


def test_a_stored_answer_outranks_the_declared_default(tmp_path, monkeypatch):
    workspace = _remembering_project(tmp_path, monkeypatch)
    workspace.write_configuration_values(
        {"recipient": "kept@example.org", "rounds": 9}
    )

    assert _collect(workspace, {})["rounds"] == 9


def test_an_explicit_input_wins_and_is_remembered(tmp_path, monkeypatch):
    workspace = _remembering_project(tmp_path, monkeypatch)
    workspace.write_configuration_values(
        {"recipient": "kept@example.org", "rounds": 3}
    )

    collected = _collect(workspace, {"rounds": 9})

    assert collected["rounds"] == 9
    assert workspace.configuration_values()["rounds"] == 9


def test_zg_config_shows_the_projects_answers(tmp_path, monkeypatch, capsys):
    """The command that shows a project's configuration shows all of it."""

    from zippergen.serve import main

    workspace = _remembering_project(tmp_path, monkeypatch)
    workspace.write_configuration_values({"recipient": "shown@example.org"})
    monkeypatch.chdir(workspace.root)

    assert main(["config"]) == 0

    output = capsys.readouterr().out
    assert "Configuration" in output
    assert "shown@example.org" in output
    assert "Address" in output, "the question is shown beside the answer"
