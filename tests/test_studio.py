import json
from pathlib import Path

from zippergen.studio import Studio
from zippergen.workspace import Workspace


WORKFLOW_SOURCE = """
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


def _studio(tmp_path, responses=()):
    root = tmp_path / "project"
    root.mkdir()
    (root / "workflow.py").write_text(WORKFLOW_SOURCE)
    workspace = Workspace(root, home=tmp_path / "home")
    answers = iter(responses)
    output: list[str] = []
    studio = Studio(
        workspace,
        input_func=lambda prompt: next(answers),
        output_func=output.append,
    )
    return studio, workspace, output


def test_studio_use_discovers_workflow_without_importing_for_selection(tmp_path):
    studio, workspace, output = _studio(tmp_path, responses=["1"])

    studio.use_workflow([])

    assert workspace.current_workflow == "workflow.py:sample"
    assert output[0] == "Workflows"
    assert "workflow.py:sample" in output[1]
    assert output[-1].startswith("Current workflow: workflow.py:sample")


def test_studio_show_menu_renders_communication_code(tmp_path):
    studio, workspace, output = _studio(tmp_path, responses=["3"])
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)

    studio.show_workflow([])

    rendered = output[-1]
    assert "User(value) >> Writer(value)" in rendered
    assert "Writer(result) >> User(result)" in rendered
    assert "echo(value)" not in rendered
    assert workspace.load()["last_view"] == "communications"


def test_studio_show_agent_renders_exact_projection(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)

    studio.show_workflow(["agent", "Writer"])

    assert "Generated local projection for Writer" in output[-1]
    assert "value = recv('User')" in output[-1]


def test_studio_create_saves_code_first_assistant_handoff(tmp_path):
    studio, workspace, output = _studio(tmp_path)

    studio.create_request("Draft an answer and ask a reviewer to approve it.")

    records = list(workspace.requests_directory.glob("*-create.json"))
    assert len(records) == 1
    content = Path(records[0].with_suffix(".md")).read_text()
    assert "Use $zippergen-workflows." in content
    assert "visible Python source" in content
    assert "Do not deploy" in content
    assert output[0].startswith("Creation brief:")


def test_studio_create_reads_multiline_prompt_from_project_file(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    prompts = workspace.root / "prompts"
    prompts.mkdir()
    (prompts / "reviewed answer.md").write_text(
        "Create a reviewed answer workflow.\n\n"
        "Never return an unapproved draft.\n",
        encoding="utf-8",
    )

    studio.execute('create --file "prompts/reviewed answer.md"')

    records = list(workspace.requests_directory.glob("*-create.json"))
    assert len(records) == 1
    metadata = json.loads(records[0].read_text())
    assert metadata["prompt"] == (
        "Create a reviewed answer workflow.\n\n"
        "Never return an unapproved draft."
    )
    assert output[0] == "Prompt file: prompts/reviewed answer.md"
    assert output[1].startswith("Creation brief:")


def test_studio_refine_saves_semantic_baseline_and_handoff(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)

    studio.refine_request("Add a human review before returning the result.")

    baselines = list(workspace.requests_directory.glob("*-semantic-before.json"))
    briefs = list(workspace.requests_directory.glob("*-refine.md"))
    assert len(baselines) == 1
    assert len(briefs) == 1
    content = briefs[0].read_text()
    assert str(baselines[0]) in content
    assert "Preserve all behavior not explicitly changed" in content
    assert "zippergen diff" in content
    assert output[0].startswith("Refinement brief:")


def test_studio_refine_reads_prompt_from_absolute_file(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    prompt_file = tmp_path / "change.md"
    prompt_file.write_text(
        "Add human review.\nPreserve the existing model call.\n",
        encoding="utf-8",
    )

    studio.execute(f'refine --file "{prompt_file}"')

    briefs = list(workspace.requests_directory.glob("*-refine.md"))
    assert len(briefs) == 1
    content = briefs[0].read_text()
    assert "Add human review.\nPreserve the existing model call." in content
    assert output[0] == f"Prompt file: {prompt_file}"
    assert output[1].startswith("Refinement brief:")


def test_studio_prompt_file_errors_are_actionable(tmp_path):
    studio, workspace, _output = _studio(tmp_path)
    (workspace.root / "empty.md").write_text("", encoding="utf-8")
    (workspace.root / "prompt-directory").mkdir()
    (workspace.root / "not-utf8.md").write_bytes(b"\xff")

    for command, expected in (
        ("create --file", "Use create --file PATH."),
        ("create --file missing.md", "Prompt file does not exist:"),
        ("create --file empty.md", "Prompt file is empty:"),
        ("create --file prompt-directory", "Prompt path is a directory:"),
        ("create --file not-utf8.md", "Prompt file must contain UTF-8 text:"),
    ):
        try:
            studio.execute(command)
        except SystemExit as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"{command!r} should fail")


def test_studio_commands_are_discoverable(tmp_path):
    studio, _workspace, output = _studio(tmp_path)

    assert studio.execute("help") is True
    assert "show | inspect" in output[-1]
    assert "create --file PATH" in output[-1]
    assert "refine --file PATH" in output[-1]
    assert studio.execute("not-a-command") is True
    assert output[-1].startswith("Unknown command")
    assert studio.execute("exit") is False


def test_studio_deploys_current_workflow_and_remembers_name(tmp_path, monkeypatch):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "zippergen.serve.main",
        lambda arguments: calls.append(arguments) or 0,
    )

    studio.deploy_workflow(["sample-test"])

    assert calls == [[
        "deploy",
        str(workspace.root / "workflow.py") + ":sample",
        "--name",
        "sample-test",
        "--llm",
        "mock",
    ]]
    assert workspace.load()["last_deployment"] == "sample-test"
    assert output[-1] == "Guided deployment: sample-test"


def test_studio_can_prepare_deployment_without_starting_it(tmp_path, monkeypatch):
    studio, workspace, _output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "zippergen.serve.main",
        lambda arguments: calls.append(arguments) or 0,
    )

    studio.deploy_workflow(["sample-test", "--no-start"])

    assert calls[0] == [
        "deploy",
        str(workspace.root / "workflow.py") + ":sample",
        "--name",
        "sample-test",
        "--no-start",
        "--llm",
        "mock",
    ]


def test_studio_operates_remembered_deployment(tmp_path, monkeypatch):
    studio, workspace, _output = _studio(tmp_path)
    workspace.update(last_deployment="sample-test")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "zippergen.serve.main",
        lambda arguments: calls.append(arguments) or 0,
    )

    studio.deployment_action("restart", [])

    assert calls == [["restart", "sample-test"]]


def test_studio_run_accepts_an_llm_override(tmp_path, monkeypatch):
    studio, _workspace, _output = _studio(tmp_path)
    calls = []
    monkeypatch.setattr(
        "zippergen.studio.run_dev",
        lambda workspace, **kwargs: calls.append(kwargs),
    )

    studio.execute("run openai:gpt-4o-mini")

    assert calls[0]["llm"] == "openai:gpt-4o-mini"


def test_studio_models_configures_and_displays_llm_active_lifelines(tmp_path):
    studio, workspace, output = _studio(
        tmp_path,
        responses=["2", "openai:gpt-4o-mini"],
    )
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)

    studio.execute("models")

    assert workspace.model_profile("workflow.py:sample") == {
        "default": "mock",
        "lifelines": {"Writer": "openai:gpt-4o-mini"},
    }
    assert any("Writer (echo)" in line for line in output)
    assert output[-1] == (
        "  Writer: openai:gpt-4o-mini (override; actions: echo)"
    )


def test_studio_model_profile_is_used_for_run_and_deploy(tmp_path, monkeypatch):
    studio, workspace, _output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_model_profile(
        "workflow.py:sample",
        default="mock",
        lifelines={"Writer": "claude:claude-sonnet-4-6"},
    )
    run_calls = []
    cli_calls: list[list[str]] = []
    monkeypatch.setattr(
        "zippergen.studio.run_dev",
        lambda workspace, **kwargs: run_calls.append(kwargs),
    )
    monkeypatch.setattr(
        "zippergen.serve.main",
        lambda arguments: cli_calls.append(arguments) or 0,
    )

    studio.execute("run")
    studio.deploy_workflow(["sample-routed"])

    assert run_calls[0]["llm"] == "mock"
    assert run_calls[0]["llms"] == {
        "Writer": "claude:claude-sonnet-4-6"
    }
    assert cli_calls[0][-4:] == [
        "--llm",
        "mock",
        "--llm-for",
        "Writer=claude:claude-sonnet-4-6",
    ]


def test_studio_models_rejects_lifelines_without_llm_actions(tmp_path):
    studio, workspace, _output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)

    try:
        studio.execute("models set User openai:gpt-4o-mini")
    except SystemExit as exc:
        assert "has no LLM actions" in str(exc)
    else:
        raise AssertionError("a non-LLM lifeline should not accept a model override")


def test_studio_reports_command_interruption_without_a_traceback(
    tmp_path, monkeypatch
):
    studio, workspace, output = _studio(tmp_path, responses=["run", "exit"])
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    monkeypatch.setattr(
        "zippergen.studio.run_dev",
        lambda workspace, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    assert studio.run() == 0

    assert any("Command interrupted" in line for line in output)
    assert any("use 'resume'" in line for line in output)
