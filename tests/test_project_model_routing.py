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
    assert record["llm_settings"] == {"Writer": {"temperature": 0.4}}


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
    monkeypatch.setattr("zippergen.serve._bundle_deployment", lambda *_args: None)
    monkeypatch.setattr(
        "zippergen.serve._prepare_deployment_environment", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr("zippergen.serve._run_deployment_setup", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("zippergen.serve._doctor_checks", lambda *_args, **_kwargs: [])

    assert main([
        "deploy",
        "--set",
        "topic=hello",
        "--no-start",
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


def test_model_settings_are_configured_beside_the_model(tmp_path, monkeypatch):
    """A standard inference setting is model configuration, not an env var.

    `temperature` was configurable here while `max_tokens` was reachable only
    through `OLLAMA_MAX_TOKENS`, so a workflow had to declare a deployment
    field just to set one. Both now live in the same place.
    """

    import tomllib

    from zippergen.configuration_mutations import configure_model
    from zippergen.workspace import Workspace

    home = tmp_path / "home"
    root = tmp_path / "project"
    root.mkdir()
    workspace = Workspace(root, home=home)
    workspace.initialize_project(name="settings")
    workspace.save_provider_connection("local-main", {"kind": "local"})

    configure_model(
        workspace,
        "qwen",
        "local-main",
        "qwen3",
        temperature=0.2,
        max_tokens=4096,
        timeout=120,
    )

    stored = tomllib.loads(workspace.manifest_path.read_text())
    configured = stored["models"]["configurations"]["qwen"]
    assert configured["temperature"] == 0.2
    assert configured["max_tokens"] == 4096
    assert configured["timeout"] == 120.0


def test_a_run_record_written_before_settings_were_one_value_still_resumes(
    tmp_path,
):
    """Old runs carry a dictionary per setting; reading both is the migration."""

    from zippergen.durable_runs import _recorded_model_settings

    recovered = _recorded_model_settings({
        "llm_temperatures": {"Writer": 0.4},
        "llm_idle_timeouts": {"Writer": 300, "Reviewer": 0},
    })

    assert recovered["Writer"].temperature == 0.4
    assert recovered["Writer"].idle_timeout == 300
    assert recovered["Reviewer"].idle_timeout == 0
    assert recovered["Reviewer"].temperature is None


def test_a_profile_written_before_settings_were_one_value_still_deploys():
    from zippergen.serve import _profile_model_settings

    recovered = _profile_model_settings({
        "llm_temperatures": {"Writer": 0.4},
        "llm_idle_timeouts": {"Writer": 300},
    })

    assert recovered["Writer"].temperature == 0.4
    assert recovered["Writer"].idle_timeout == 300
