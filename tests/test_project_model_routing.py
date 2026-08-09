import json
from pathlib import Path

from zippergen.serve import main
from zippergen.workspace import Workspace


MODEL_WORKFLOW = """
from zippergen import Lifeline, llm, workflow

User = Lifeline("User")
Writer = Lifeline("Writer")

@llm(
    system="Draft a reply.",
    user="{topic}",
    parse="text",
    outputs=(("draft", str),),
)
def draft_reply(topic: str) -> None: ...

@workflow
def answer(topic: str @ User) -> str:
    User(topic) >> Writer(topic)
    Writer: draft = draft_reply(topic)
    Writer(draft) >> User(draft)
    return draft @ User
"""


def _configured_project(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    root = tmp_path / "project"
    root.mkdir()
    workflow_path = root / "workflow.py"
    workflow_path.write_text(MODEL_WORKFLOW)
    replies = root / "replies.json"
    replies.write_text(
        json.dumps({"Writer.draft_reply": {"draft": "assigned model"}})
    )
    home = tmp_path / "home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    workspace = Workspace(root, home=home)
    workspace.initialize_project()
    workspace.select_workflow("workflow.py:answer")
    workspace.save_model_configuration(
        "writer-model",
        {
            "provider": "scripted",
            "model": str(replies),
            "spec": f"scripted:{replies}",
        },
    )
    workspace.save_model_assignment_profile(
        "workflow.py:answer",
        default="mock",
        lifelines={"Writer": "writer-model"},
    )
    monkeypatch.chdir(root)
    return root, home


def test_project_model_assignment_drives_plain_and_durable_runs(
    tmp_path, monkeypatch, capsys
):
    root, home = _configured_project(tmp_path, monkeypatch)

    assert main(["run", "--input", "topic=hello", "--yes"]) == 0
    assert json.loads(capsys.readouterr().out) == {"result": "assigned model"}

    assert main([
        "run",
        "--durable",
        "--project",
        str(root),
        "--input",
        "topic=hello",
        "--yes",
    ]) == 0
    capsys.readouterr()
    records = list((home / "workspaces").glob("*/runs/*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["result"] == "assigned model"
    assert record["llm"] == "mock"
    assert record["llms"] == {"Writer": f"scripted:{root / 'replies.json'}"}


def test_action_assignment_overrides_its_participant_assignment(
    tmp_path, monkeypatch, capsys
):
    root, home = _configured_project(tmp_path, monkeypatch)
    action_replies = root / "action-replies.json"
    action_replies.write_text(
        json.dumps({"Writer.draft_reply": {"draft": "action model"}})
    )
    workspace = Workspace(root, home=home)
    workspace.save_model_configuration(
        "draft-model",
        {
            "provider": "scripted",
            "model": str(action_replies),
            "spec": f"scripted:{action_replies}",
        },
    )
    workspace.save_model_assignment_profile(
        "workflow.py:answer",
        default="mock",
        lifelines={"Writer": "writer-model"},
        actions={"Writer.draft_reply": "draft-model"},
    )

    assert main(["run", "--input", "topic=hello", "--yes"]) == 0
    assert json.loads(capsys.readouterr().out) == {"result": "action model"}


def test_deployment_snapshots_project_model_assignments(
    tmp_path, monkeypatch, capsys
):
    root, home = _configured_project(tmp_path, monkeypatch)

    assert main([
        "deploy",
        "--name",
        "answer-prod",
        "--input",
        "topic=hello",
        "--no-start",
        "--no-bundle",
        "--no-install",
        "--no-setup",
        "--no-doctor",
        "--yes",
    ]) == 0
    capsys.readouterr()

    profile = json.loads(
        (home / "deployments" / "answer-prod.json").read_text()
    )
    assert profile["llm"] == "mock"
    assert profile["llms"] == {
        "Writer": f"scripted:{root / 'replies.json'}"
    }


def test_global_cli_model_replaces_project_assignments(
    tmp_path, monkeypatch, capsys
):
    _root, home = _configured_project(tmp_path, monkeypatch)

    assert main([
        "deploy",
        "--name",
        "mock-prod",
        "--llm",
        "mock",
        "--input",
        "topic=hello",
        "--no-start",
        "--no-bundle",
        "--no-install",
        "--no-setup",
        "--no-doctor",
        "--yes",
    ]) == 0
    capsys.readouterr()

    profile = json.loads((home / "deployments" / "mock-prod.json").read_text())
    assert profile["llm"] == "mock"
    assert profile["llms"] == {}
