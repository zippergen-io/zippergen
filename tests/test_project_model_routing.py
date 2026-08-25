import json
from pathlib import Path

import pytest

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


# A readiness check that enumerates specs assumes one spec is one invocation.
# It is not, and checking one of several and reporting the model ready says
# more than was tested.


def test_two_targets_on_one_model_are_two_invocations():
    from zippergen.models import ModelSettings, model_invocations

    pairs = model_invocations(
        "openai:gpt",
        {"Writer": "openai:gpt", "Reviewer": "openai:gpt"},
        {"Writer": ModelSettings(timeout=1)},
    )

    timeouts = {chosen.timeout for _spec, chosen in pairs}
    assert timeouts == {1.0, None}, (
        "the target using backend defaults is its own invocation"
    )


def test_conflicting_settings_are_both_checked_not_discarded():
    from zippergen.models import ModelSettings, model_invocations

    pairs = model_invocations(
        "openai:gpt",
        {"Writer": "openai:gpt", "Reviewer": "openai:gpt"},
        {
            "Writer": ModelSettings(timeout=1),
            "Reviewer": ModelSettings(timeout=9),
        },
    )

    assert sorted(chosen.timeout for _spec, chosen in pairs) == [1.0, 9.0]


def test_a_named_but_unrouted_configuration_is_checked_with_its_settings():
    """Checking a configuration by name must use the settings it would use."""

    from zippergen.models import ModelSettings, model_invocations

    pairs = model_invocations(
        "mock",
        {},
        {},
        extra=[("local@local-main:qwen", ModelSettings(timeout=12))],
    )

    assert ("local@local-main:qwen", ModelSettings(timeout=12)) in pairs


def test_identical_invocations_are_checked_once():
    from zippergen.models import ModelSettings, model_invocations

    pairs = model_invocations(
        "openai:gpt",
        {"Writer": "openai:gpt", "Reviewer": "openai:gpt"},
        {
            "Writer": ModelSettings(timeout=5),
            "Reviewer": ModelSettings(timeout=5),
        },
    )

    assert len(pairs) == 1


# The cases below cross the two axes that matter -- where an invocation comes
# from (routed or named) against whether two of them share a spec -- because
# testing one cell of that square is how a whole column went unchecked.


@pytest.mark.parametrize("routed", [True, False])
def test_two_invocations_on_one_spec_survive_however_they_arise(routed):
    """Routed or merely named, one spec can still be two invocations."""

    from zippergen.configuration_checks import _model_invocations
    from zippergen.models import ModelSettings, model_settings_from_mapping

    if routed:
        resolved = {
            "default": "openai:gpt",
            "overrides": {"Writer": "openai:gpt", "Reviewer": "openai:gpt"},
            "settings": {
                "Writer": {"timeout": 1},
                "Reviewer": {"timeout": 120},
            },
        }
        pairs = _model_invocations(resolved, {}, ())
    else:
        resolved = {"default": "mock", "overrides": {}, "settings": {}}
        configurations = {
            "fast": {"spec": "openai:gpt", "timeout": "1"},
            "slow": {"spec": "openai:gpt", "timeout": "120"},
        }
        pairs = _model_invocations(resolved, configurations, ("fast", "slow"))

    timeouts = sorted(
        chosen.timeout
        for spec, chosen in pairs
        if spec == "openai:gpt" and chosen.timeout is not None
    )
    assert timeouts == [1.0, 120.0]


def test_two_named_configurations_that_agree_are_checked_once():
    from zippergen.configuration_checks import _model_invocations

    resolved = {"default": "mock", "overrides": {}, "settings": {}}
    configurations = {
        "a": {"spec": "openai:gpt", "timeout": "5"},
        "b": {"spec": "openai:gpt", "timeout": "5"},
    }

    pairs = _model_invocations(resolved, configurations, ("a", "b"))

    assert sum(1 for spec, _ in pairs if spec == "openai:gpt") == 1
