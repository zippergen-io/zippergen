import json
import subprocess

import pytest

from zippergen.natural_language import (
    NaturalCommandPlan,
    NaturalLanguageStore,
    generalize_interpretation,
    parse_cli_plan,
)
from zippergen.studio import Studio
from zippergen.workspace import Workspace


WORKFLOW_SOURCE = """
from zippergen import Lifeline, llm, workflow

User = Lifeline("User")
Writer = Lifeline("Writer")

@llm(
    system="Write.",
    user="{request}",
    parse="text",
    outputs=(("draft", str),),
)
def draft(request: str) -> None: ...

@workflow
def sample(request: str @ User) -> str:
    User(request) >> Writer(request)
    Writer: result = draft(request)
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


def test_natural_current_request_executes_without_a_model(tmp_path):
    studio, workspace, output = _studio(tmp_path)

    studio.execute("What is the current state?")

    assert "Natural-language request" in output
    assert any("current" in line for line in output)
    assert "Current" in output
    assert "Project" in output
    history = NaturalLanguageStore(
        workspace.directory / "natural-language.json"
    ).history()
    assert history[-1]["source"] == "deterministic"
    assert history[-1]["status"] == "executed"


def test_natural_show_phrase_wins_over_invalid_show_syntax(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)

    studio.execute("Show me the whole protocol")

    assert any("show protocol" in line for line in output)
    assert any("@workflow" in line for line in output)


def test_natural_prose_with_an_apostrophe_is_not_treated_as_broken_shell_syntax(
    tmp_path,
):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)

    studio.execute("Show Writer's local view")

    assert any("show agent Writer" in line for line in output)
    assert workspace.load()["last_view"] == "agent Writer"


def test_natural_model_assignment_is_canonical_and_reversible(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)

    studio.execute("Assign the mock model to Writer")

    profile = workspace.model_profile("workflow.py:sample", default="mock")
    assert profile["lifelines"] == {"Writer": "mock"}
    assert any("models set Writer mock" in line for line in output)
    assert any("Natural-language command plan completed" in line for line in output)


def test_how_do_i_request_previews_without_execution(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)

    studio.execute("How do I show the whole protocol?")

    assert any("preview only" in line for line in output)
    assert any("Plan shown without execution" in line for line in output)
    assert not any("@workflow" in line for line in output)


def test_natural_run_phrase_is_not_misparsed_as_an_exact_run_command(tmp_path):
    studio, workspace, output = _studio(tmp_path, responses=["n"])
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)

    studio.execute("Run the workflow")

    assert any("Command plan" in line for line in output)
    assert any(line.strip().endswith("run") for line in output)
    assert any("Safety" in line and "execution" in line for line in output)
    assert any("nothing was executed" in line for line in output)


def test_codex_fallback_is_read_only_and_learns_a_parameterized_plan(
    tmp_path, monkeypatch
):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    calls: list[tuple[list[str], dict[str, object]]] = []

    monkeypatch.setattr(
        "zippergen.studio.shutil.which",
        lambda name: "/tools/codex" if name == "codex" else None,
    )

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        payload = {
            "summary": "Show Writer's local projection.",
            "commands": ["show agent Writer"],
            "clarification": None,
        }
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(payload), stderr=""
        )

    monkeypatch.setattr("zippergen.studio.subprocess.run", fake_run)

    studio.execute("What exactly can Writer observe?")
    studio.execute("What exactly can User observe?")

    assert len(calls) == 1
    assert calls[0][0][:6] == [
        "/tools/codex",
        "exec",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--cd",
    ]
    assert calls[0][0][-1] == "-"
    assert "selected_workflow" in str(calls[0][1]["input"])
    learned = NaturalLanguageStore(
        workspace.directory / "natural-language.json"
    ).learned()
    assert learned[0]["request_template"] == (
        "what exactly can {participant} observe"
    )
    assert learned[0]["commands"] == ["show agent {participant}"]
    assert learned[0]["uses"] == 1
    assert any("private learned interpretation L001" in line for line in output)


def test_cli_plan_cannot_escape_the_studio_command_catalog(tmp_path, monkeypatch):
    studio, _workspace, _output = _studio(tmp_path)
    monkeypatch.setattr(
        "zippergen.studio.shutil.which",
        lambda name: "/tools/codex" if name == "codex" else None,
    )
    payload = {
        "summary": "Delete files.",
        "commands": ["rm -rf ."],
        "clarification": None,
    }
    monkeypatch.setattr(
        "zippergen.studio.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(payload), stderr=""
        ),
    )

    with pytest.raises(SystemExit, match="unsupported Studio syntax"):
        studio.execute("Clean up absolutely everything")


def test_execution_plan_requires_confirmation(tmp_path, monkeypatch):
    studio, workspace, output = _studio(tmp_path, responses=["n"])
    workspace.update(last_deployment="review")
    monkeypatch.setattr(
        "zippergen.studio.shutil.which",
        lambda name: "/tools/codex" if name == "codex" else None,
    )
    payload = {
        "summary": "Stop the remembered deployment.",
        "commands": ["stop review"],
        "clarification": None,
    }
    monkeypatch.setattr(
        "zippergen.studio.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(payload), stderr=""
        ),
    )

    studio.execute("Shut down the deployed service")

    history = NaturalLanguageStore(
        workspace.directory / "natural-language.json"
    ).history()
    assert history[-1]["status"] == "cancelled"
    assert any("nothing was executed" in line for line in output)


def test_cli_fallback_can_return_a_validated_read_only_command_sequence(
    tmp_path, monkeypatch
):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    monkeypatch.setattr(
        "zippergen.studio.shutil.which",
        lambda name: "/tools/codex" if name == "codex" else None,
    )
    payload = {
        "summary": "Show project state and model routing.",
        "commands": ["current", "models show"],
        "clarification": None,
    }
    monkeypatch.setattr(
        "zippergen.studio.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(payload), stderr=""
        ),
    )

    studio.execute("Give me one combined operational and model summary")

    history = NaturalLanguageStore(workspace.natural_language_path).history()
    assert history[-1]["commands"] == ["current", "models show"]
    assert history[-1]["status"] == "executed"
    assert any("Executing 1/2: current" in line for line in output)
    assert any("Executing 2/2: models show" in line for line in output)


def test_cli_fallback_can_ask_for_a_missing_value(tmp_path, monkeypatch):
    studio, workspace, output = _studio(tmp_path)
    monkeypatch.setattr(
        "zippergen.studio.shutil.which",
        lambda name: "/tools/codex" if name == "codex" else None,
    )
    payload = {
        "summary": "A workflow must be identified.",
        "commands": [],
        "clarification": "Which workflow should Studio select?",
    }
    monkeypatch.setattr(
        "zippergen.studio.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(payload), stderr=""
        ),
    )

    studio.execute("Select the appropriate workflow")

    assert any("Which workflow should Studio select?" in line for line in output)
    history = NaturalLanguageStore(workspace.natural_language_path).history()
    assert history[-1]["status"] == "clarification"
    assert NaturalLanguageStore(workspace.natural_language_path).learned() == []


def test_language_controls_are_inspectable_and_learned_items_can_be_forgotten(
    tmp_path,
):
    studio, workspace, output = _studio(tmp_path)
    store = NaturalLanguageStore(workspace.directory / "natural-language.json")
    record = store.remember(
        "What exactly can Writer observe?",
        NaturalCommandPlan(
            "Show Writer.",
            ("show agent Writer",),
            "codex",
        ),
    )
    assert record is not None

    studio.execute("language")
    studio.execute(f"language forget {record['id']}")

    assert any("Natural-language commands" in line for line in output)
    assert store.learned() == []


def test_secret_looking_natural_request_is_neither_sent_nor_stored(
    tmp_path, monkeypatch
):
    studio, workspace, _output = _studio(tmp_path)

    def unexpected(*args, **kwargs):
        raise AssertionError("secret-looking text must not reach a CLI")

    monkeypatch.setattr("zippergen.studio.subprocess.run", unexpected)

    with pytest.raises(SystemExit, match="appears to contain a secret"):
        studio.execute("Set my OpenAI API key to sk-abcdefghijklmnop")

    store = NaturalLanguageStore(workspace.directory / "natural-language.json")
    assert store.history() == []
    assert store.learned() == []


def test_secret_looking_cli_output_is_discarded_before_display_or_learning(
    tmp_path, monkeypatch
):
    studio, workspace, output = _studio(tmp_path)
    monkeypatch.setattr(
        "zippergen.studio.shutil.which",
        lambda name: "/tools/codex" if name == "codex" else None,
    )
    payload = {
        "summary": "Use sk-abcdefghijklmnop",
        "commands": ["current"],
        "clarification": None,
    }
    monkeypatch.setattr(
        "zippergen.studio.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(payload), stderr=""
        ),
    )

    with pytest.raises(SystemExit, match="discarded the plan"):
        studio.execute("Resolve this unusual project question")

    assert not any("sk-abcdefghijklmnop" in line for line in output)
    store = NaturalLanguageStore(workspace.natural_language_path)
    assert store.learned() == []
    assert store.history()[-1]["commands"] == []


def test_private_state_reset_archives_language_history(tmp_path):
    studio, workspace, _output = _studio(tmp_path)
    studio.execute("What is the current state?")

    result = workspace.reset_private_state()

    backup = result["backup_directory"]
    assert backup is not None
    archived = backup / "workspace" / "natural-language.json"
    assert archived.exists()
    assert not workspace.natural_language_path.exists()


def test_cli_json_parser_uses_the_last_structured_plan():
    plan = parse_cli_plan(
        'diagnostic {"ignored": true}\n'
        '{"summary":"Inspect","commands":["current"],"clarification":null}',
        source="codex",
    )

    assert plan.commands == ("current",)
    assert plan.source == "codex"


def test_generalization_quotes_values_only_when_rendering(tmp_path):
    template, commands = generalize_interpretation(
        "Show what Lead Writer sees",
        ("show agent 'Lead Writer'",),
    )
    store = NaturalLanguageStore(tmp_path / "natural-language.json")
    store.remember(
        "Show what Lead Writer sees",
        NaturalCommandPlan("Show participant.", commands, "codex"),
    )

    assert template == "show what {participant} sees"
    assert commands == ("show agent {participant}",)
