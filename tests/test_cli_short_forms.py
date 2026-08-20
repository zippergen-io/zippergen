"""The common path should be short enough not to undercut the story.

A first tutorial that says "just tell the coding agent what you want" is
undermined if every check then costs a workflow spec, an execution-mode flag
and an inline input. A single-workflow project already records its entry point,
so these commands infer it.

    zippergen validate workflow.py:email_approval --execution memory ...
    zg validate
"""

import argparse
import shutil
from pathlib import Path

import pytest

from zippergen import serve
from zippergen.workspace import Workspace

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
EXAMPLE = EXAMPLES / "email_approval.py"
# The tutorial workflow watches a mailbox and never ends on its own, so every
# run of it here carries a message budget.
BUDGET = ["--option", "max_messages=1"]


def _mailbox(root, *messages):
    box = root / "mailbox"
    box.mkdir(exist_ok=True)
    for index, text in enumerate(messages, start=1):
        (box / f"{index:02d}.txt").write_text(text, encoding="utf-8")
    return box


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "email-approval"
    root.mkdir()
    shutil.copy(EXAMPLE, root / "workflow.py")
    workspace = Workspace(root, home=tmp_path / "home")
    workspace.initialize_project(name="email-approval")
    monkeypatch.chdir(root)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))
    return root


@pytest.fixture
def input_project(tmp_path, monkeypatch):
    """A workflow that actually takes an input, for the prompting tests."""

    root = tmp_path / "hello"
    root.mkdir()
    shutil.copy(EXAMPLES / "hello.py", root / "workflow.py")
    workspace = Workspace(root, home=tmp_path / "home")
    workspace.initialize_project(name="hello")
    monkeypatch.chdir(root)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))
    return root


def test_validate_infers_the_project_workflow(project, capsys):
    assert serve.main(["validate"]) == 0

    output = capsys.readouterr().out
    assert "email_approval: valid" in output
    assert (
        "workflow inputs: none, the run starts without setup questions"
        in output
    )


def test_show_infers_the_project_workflow(project, capsys):
    assert serve.main(["show", "--agent", "Writer"]) == 0

    rendered = capsys.readouterr().out
    assert "email_approval__Writer" in rendered
    # The Writer takes no part in Mailbox's decision, so it has no branch.
    assert "if " not in rendered


def test_run_infers_the_project_workflow(project, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
    mailbox = _mailbox(project, "Can we meet on Thursday")

    rc = serve.main(["run", "--llm", "mock", "--yes", *BUDGET])

    assert rc == 0
    assert "result" in capsys.readouterr().out
    assert not (mailbox / "01.txt").exists()
    assert (mailbox / "01.done").read_text(encoding="utf-8") == (
        "Can we meet on Thursday"
    )


def test_an_explicit_spec_still_wins(project, capsys):
    assert serve.main(["validate", "workflow.py:email_approval"]) == 0

    assert "email_approval: valid" in capsys.readouterr().out


def test_a_run_leaves_nothing_behind_by_default(project, monkeypatch, capsys):
    """`--execution memory` was an implementation detail in the reader's way."""

    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
    _mailbox(project, "Could we move our meeting?")

    serve.main(["run", "--llm", "mock", "--yes", *BUDGET])

    assert "Store:" not in capsys.readouterr().err
    assert not list(project.glob("*.sqlite"))


def test_a_missing_input_is_asked_for_in_a_terminal(
    input_project, monkeypatch, capsys
):
    """Rather than refusing with a usage error."""

    monkeypatch.setattr(serve.sys.stdin, "isatty", lambda: True)
    answers = iter(["deployment", "y"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    rc = serve.main(["run", "--llm", "mock"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "Workflow inputs\n═══════════════\n" in output
    assert "result" in output


def test_validate_lists_required_workflow_inputs(input_project, capsys):
    assert serve.main(["validate"]) == 0

    assert "workflow inputs: topic (str) @ User, required" in (
        capsys.readouterr().out
    )


def test_plain_run_uses_a_declared_input_default_without_prompting(
    tmp_path, monkeypatch, capsys
):
    root = tmp_path / "default-input"
    root.mkdir()
    (root / "workflow.py").write_text(
        """
from zippergen import DeploymentField, DeploymentSpec, Lifeline, pure, workflow

User = Lifeline("User")
zippergen_deployment = DeploymentSpec(fields=(
    DeploymentField(
        "directory",
        "Directory",
        target="input",
        default="mailbox",
    ),
))

@pure
def identity(value: str) -> str:
    return value

@workflow
def default_input(directory: str @ User) -> str:
    User: result = identity(directory)
    return result @ User
""",
        encoding="utf-8",
    )
    workspace = Workspace(root, home=tmp_path / "home")
    workspace.initialize_project(name="default-input")
    monkeypatch.chdir(root)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))

    assert serve.main(["run", "--llm", "mock", "--yes"]) == 0
    assert '"result": "mailbox"' in capsys.readouterr().out


def test_a_missing_input_outside_a_terminal_still_refuses_clearly(
    input_project, monkeypatch
):
    monkeypatch.setattr(serve.sys.stdin, "isatty", lambda: False)

    with pytest.raises(SystemExit) as error:
        serve.main(["run", "--llm", "mock"])

    assert "topic" in str(error.value)


def test_a_directory_without_a_project_says_so(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))

    with pytest.raises(SystemExit, match="Not a ZipperGen project"):
        serve.main(["validate"])


@pytest.mark.parametrize(
    "command",
    (["config"], ["model"], ["connector"], ["run", "--llm", "mock"]),
)
def test_project_commands_never_initialize_an_accidental_directory(
    tmp_path, monkeypatch, command
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))

    with pytest.raises(SystemExit, match="Not a ZipperGen project"):
        serve.main(command)
    assert not (tmp_path / "zippergen.toml").exists()


def test_diff_compares_a_baseline_against_the_project_workflow(
    project, tmp_path, capsys
):
    """One saved baseline plus one argument is the whole change-check ritual."""

    baseline = tmp_path / "before.json"
    assert serve.main(["snapshot", str(baseline)]) == 0
    capsys.readouterr()

    workflow = project / "workflow.py"
    workflow.write_text(
        workflow.read_text().replace(
            "Writer(draft) >> Mailbox(draft)",
            "Writer(draft) >> Mailbox(draft)\n        Mailbox(draft) >> Writer(draft)",
            1,
        )
    )

    assert serve.main(["diff", str(baseline)]) == 0

    assert "Mailbox(draft) >> Writer(draft)" in capsys.readouterr().out


def test_a_single_workflow_is_inferred_without_a_manifest_entry(
    tmp_path, monkeypatch, capsys
):
    """`zg init` runs before any workflow exists, so it cannot record one.

    A beginner should reach `zg validate` without first hand-writing
    workflow_entry into zippergen.toml.
    """

    root = tmp_path / "beginner"
    root.mkdir()
    shutil.copy(EXAMPLE, root / "workflow.py")
    workspace = Workspace(root, home=tmp_path / "home")
    workspace.initialize_project(name="beginner")
    monkeypatch.chdir(root)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))

    assert serve.main(["validate"]) == 0

    assert "email_approval: valid" in capsys.readouterr().out
    # Inference is a convenience; it must not quietly rewrite the manifest.
    assert "workflow_entry" not in workspace.manifest_path.read_text()


def test_workflow_shows_inferred_entry_and_selects_an_explicit_one(
    project, capsys
):
    assert serve.main(["workflow"]) == 0
    assert capsys.readouterr().out.strip() == "workflow.py:email_approval (inferred)"

    assert serve.main(
        ["workflow", "select", "workflow.py:email_approval"]
    ) == 0
    assert "Project workflow: workflow.py:email_approval" in (
        capsys.readouterr().out
    )
    assert Workspace(project).workflow_entry == "workflow.py:email_approval"


def test_a_durable_run_infers_the_only_workflow_without_a_manifest_entry(
    tmp_path, monkeypatch, capsys
):
    root = tmp_path / "durable-beginner"
    root.mkdir()
    shutil.copy(EXAMPLES / "hello.py", root / "workflow.py")
    workspace = Workspace(root, home=tmp_path / "home")
    workspace.initialize_project(name="durable-beginner")
    monkeypatch.chdir(root)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))

    assert serve.main(
        ["run", "--durable", "--llm", "mock", "--yes", "--input", "topic=hello"]
    ) == 0

    assert workspace.current_run()["status"] == "done"
    assert "Workflow hello: valid" in capsys.readouterr().out


def test_connector_assignment_infers_the_only_workflow_without_manifest_entry(
    tmp_path, monkeypatch, capsys
):
    root = tmp_path / "connector-beginner"
    root.mkdir()
    shutil.copy(EXAMPLE, root / "workflow.py")
    workspace = Workspace(root, home=tmp_path / "home")
    workspace.initialize_project(name="connector-beginner")
    workspace.save_provider_connection("approval-bot", {"kind": "telegram"})
    workspace.save_connector_configuration(
        "approval-chat",
        {
            "connection": "approval-bot",
            "kind": "telegram",
            "chat_id": "123",
        },
    )
    monkeypatch.chdir(root)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))

    assert serve.main(["connector", "assign", "Mailbox", "approval-chat"]) == 0

    assert workspace.connector_assignment_profile(
        "workflow.py:email_approval"
    )["lifelines"] == {"Mailbox": "approval-chat"}
    manifest = workspace.manifest_path.read_text(encoding="utf-8")
    assert "[connectors.assignments.lifelines]" in manifest
    assert '"Mailbox" = "approval-chat"' in manifest
    assert "workflow_entry" not in manifest
    assert "asked through approval-chat" in capsys.readouterr().out


def test_several_workflows_ask_for_an_explicit_choice(tmp_path, monkeypatch):
    root = tmp_path / "ambiguous"
    root.mkdir()
    shutil.copy(EXAMPLE, root / "workflow.py")
    shutil.copy(EXAMPLE, root / "second.py")
    workspace = Workspace(root, home=tmp_path / "home")
    workspace.initialize_project(name="ambiguous")
    monkeypatch.chdir(root)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))

    with pytest.raises(SystemExit, match="several workflows"):
        serve.main(["validate"])


def test_a_manifest_entry_wins_over_discovery(tmp_path, monkeypatch, capsys):
    root = tmp_path / "explicit"
    root.mkdir()
    shutil.copy(EXAMPLE, root / "workflow.py")
    shutil.copy(EXAMPLE, root / "second.py")
    workspace = Workspace(root, home=tmp_path / "home")
    workspace.initialize_project(name="explicit")
    workspace.select_workflow("second.py:email_approval", cwd=root)
    monkeypatch.chdir(root)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))

    assert serve.main(["validate"]) == 0

    assert "email_approval: valid" in capsys.readouterr().out


def test_top_level_help_hides_legacy_internal_commands(capsys):
    with pytest.raises(SystemExit) as exc:
        serve.main(["--help"])
    assert exc.value.code == 0

    output = capsys.readouterr().out
    assert "==SUPPRESS==" not in output
    for command in ("__run-deployment", "notify", "serve"):
        assert f"    {command} " not in output

    with pytest.raises(SystemExit) as deploy_help:
        serve.main(["deploy", "--help"])
    assert deploy_help.value.code == 0
    assert "    run " not in capsys.readouterr().out


def test_completion_uses_the_registered_command_tree():
    from zippergen.completion import completion_candidates, render_completion
    from zippergen.serve import HIDDEN_COMMANDS, _parse_cli_args

    parser, _arguments = _parse_cli_args([])
    top_level = next(
        action for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    expected_commands = [
        name for name in top_level.choices if name not in HIDDEN_COMMANDS
    ]
    deploy_parser = top_level.choices["deploy"]
    deploy = next(
        action for action in deploy_parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    run_parser = top_level.choices["run"]
    run = next(
        action for action in run_parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )

    assert completion_candidates("commands") == expected_commands
    assert completion_candidates("run-actions") == list(run.choices)
    assert completion_candidates("deploy-actions") == list(deploy.choices)
    assert "run" not in completion_candidates("deploy-actions")
    for family in (
        "workflow",
        "provider",
        "model",
        "assistant",
        "connector",
        "run",
        "deploy",
    ):
        actions = completion_candidates(f"{family}-actions")
        # Fish has to spell out the actions in its predicates; zsh and bash
        # obtain them dynamically from ``${cmd}-actions``.
        fish_script = render_completion("fish")
        assert all(action in fish_script for action in actions)
    for shell in ("zsh", "bash", "fish"):
        script = render_completion(shell)
        assert "deploy-actions" in script or "${cmd}-actions" in script
        assert "kind=deployments" not in script
    fish = render_completion("fish")
    assert (
        "__fish_seen_subcommand_from status reset inspect trace tasks approve"
        in fish
    )
    assert "__zg_complete_options" in fish
    # Every shell is generated from one positional table, so each must carry
    # every rule. Asserting coverage rather than syntax is what catches a new
    # command reaching one shell and not the other two.
    from zippergen.completion import POSITIONAL_COMPLETIONS

    for shell in ("zsh", "bash", "fish"):
        script = render_completion(shell)
        for (command, action, _index), candidate in POSITIONAL_COMPLETIONS.items():
            assert candidate in script, (shell, candidate)
            assert action in script, (shell, command, action)


def test_a_projection_that_promises_a_result_shows_where_it_comes_from(
    project, capsys
):
    """The signature and the body have to agree.

    The local view typed `Mailbox` as `-> int` while rendering a function that
    never returned. A reader who knows Python notices that before they notice
    anything else.
    """

    assert serve.main(["show", "--agent", "Mailbox"]) == 0
    mailbox = capsys.readouterr().out

    assert "-> int" in mailbox
    assert "return handled" in mailbox

    # The Writer owns no output, so it must keep promising nothing.
    assert serve.main(["show", "--agent", "Writer"]) == 0
    writer = capsys.readouterr().out

    assert "-> None" in writer
    assert "return" not in writer


def test_generated_completion_positions_agree_across_shells():
    """Presence is not enough: a rule can be emitted at the wrong word.

    One table feeds three shells that count words differently. zsh's key
    counts from the program and includes the action; bash's COMP_CWORD counts
    the program as word 0; fish counts the words already typed. A shared
    off-by-one puts candidates one argument early -- suggesting a provider
    kind where the name belongs -- while still passing a presence check.
    """

    import re

    from zippergen.completion import POSITIONAL_COMPLETIONS, render_completion

    zsh = render_completion("zsh")
    bash = render_completion("bash")
    fish = render_completion("fish")

    for (command, action, index), candidate in POSITIONAL_COMPLETIONS.items():
        assert f"    {command}:{action}:{index + 2}) kind={candidate} ;;" in zsh
        assert (
            f"$cmd == {command} && $action == {action} "
            f"&& $COMP_CWORD -eq {index + 1} ]]; then kind={candidate}"
        ) in bash
        assert (
            f"__fish_seen_subcommand_from {command}; "
            f"and __fish_seen_subcommand_from {action}; "
            f"and test (count (commandline -opc)) -eq {index + 1}' "
            f"-a '(zg __complete {candidate} 2>/dev/null)'"
        ) in fish

    # One anchor checked against the hand-written script this replaced, so the
    # three formulas cannot drift together into a consistent wrong answer.
    assert (
        "$cmd == provider && $action == configure && $COMP_CWORD -eq 4 ]]; "
        "then kind=provider-kinds"
    ) in bash
    assert "    provider:configure:5) kind=provider-kinds ;;" in zsh
