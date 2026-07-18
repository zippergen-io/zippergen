from pathlib import Path

from zippergen.studio import Studio
from zippergen.workspace import Workspace


WORKFLOW_SOURCE = """
from zippergen import Lifeline, pure, workflow

User = Lifeline("User")
Writer = Lifeline("Writer")

@pure
def echo(value: str) -> str:
    return value

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


def test_studio_commands_are_discoverable(tmp_path):
    studio, _workspace, output = _studio(tmp_path)

    assert studio.execute("help") is True
    assert "show | inspect" in output[-1]
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
    ]]
    assert workspace.load()["last_deployment"] == "sample-test"
    assert output[-1] == "Guided deployment: sample-test"


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
