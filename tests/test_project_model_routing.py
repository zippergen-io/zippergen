import json
from pathlib import Path

from zippergen.serve import main
from zippergen.workspace import Workspace



def _one_deployment(home, suffix=".json"):
    """The project's one deployment, whatever name was derived for it."""

    found = sorted((home / "deployments").glob(f"*{suffix}"))
    if suffix == ".json":
        found = [p for p in found if not p.name.endswith(".secrets.json")]
    assert len(found) == 1, f"expected one {suffix} deployment file, got {found}"
    return found[0]

MODEL_WORKFLOW = """
from zippergen import DeploymentField, DeploymentSpec, Lifeline, llm, workflow

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

zippergen_deployment = DeploymentSpec(fields=(
    DeploymentField("topic", "Topic", target="input", required=True),
))
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
    workspace.save_provider_connection("scripted-tests", {"kind": "scripted"})
    workspace.save_model_configuration(
        "writer-model",
        {
            "connection": "scripted-tests",
            "model": str(replies),
            "temperature": "0.4",
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
    assert record["llms"] == {
        "Writer": f"scripted@scripted-tests:{root / 'replies.json'}"
    }
    assert record["llm_temperatures"] == {"Writer": 0.4}


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
            "connection": "scripted-tests",
            "model": str(action_replies),
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
        "--set",
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
        _one_deployment(home).read_text()
    )
    assert profile["llm"] == "mock"
    assert profile["llms"] == {
        "Writer": f"scripted@scripted-tests:{root / 'replies.json'}"
    }


def test_deploy_does_not_expose_a_global_model_override(
    tmp_path, monkeypatch, capsys
):
    _configured_project(tmp_path, monkeypatch)

    try:
        main(["deploy", "--llm", "mock"])
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover
        raise AssertionError("deploy should use project model assignments")


def test_global_cli_model_replaces_project_assignments_for_plain_run(
    tmp_path, monkeypatch, capsys
):
    _configured_project(tmp_path, monkeypatch)

    assert main([
        "run",
        "--llm",
        "mock",
        "--input",
        "topic=hello",
        "--yes",
    ]) == 0

    assert json.loads(capsys.readouterr().out) == {
        "result": "[draft_reply:draft]"
    }
