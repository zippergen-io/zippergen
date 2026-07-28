import json
import shlex
import subprocess

import pytest

from zippergen.natural_language import (
    NaturalCommandPlan,
    NaturalLanguageStore,
    deterministic_plan,
    generalize_interpretation,
    parse_cli_plan,
    requirement_proposal,
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


def _studio(tmp_path, responses=(), secret_responses=()):
    root = tmp_path / "project"
    root.mkdir()
    (root / "workflow.py").write_text(WORKFLOW_SOURCE)
    workspace = Workspace(root, home=tmp_path / "home")
    answers = iter(responses)
    secret_answers = iter(secret_responses)
    output: list[str] = []
    studio = Studio(
        workspace,
        input_func=lambda prompt: next(answers),
        output_func=output.append,
        secret_input_func=lambda prompt: next(secret_answers),
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


@pytest.mark.parametrize(
    ("phrase", "command"),
    [
        ("Show me all stores", ("runs", "deploy list")),
        ("Show pending human tasks", ("run tasks",)),
        ("Show the store trace", ("run trace",)),
        ("What is the deployment status?", "deploy show"),
        ("Show the deploy logs", "deploy logs"),
        ("Stop the deployment", "deploy stop"),
        (
            "Remove deployment review-demo",
            "deploy remove review-demo",
        ),
        (
            "Permanently delete deployment review-demo",
            "deploy remove review-demo --purge",
        ),
    ],
)
def test_natural_operational_requests_use_the_namespaced_surface(
    phrase,
    command,
):
    plan = deterministic_plan(phrase)

    assert plan is not None
    expected = command if isinstance(command, tuple) else (command,)
    assert plan.commands == expected


def test_unmatched_design_prose_is_offered_as_initial_specification(tmp_path):
    studio, workspace, output = _studio(tmp_path, responses=["y"])
    request = (
        "Make a workflow where Writer drafts an answer and Reviewer approves it"
    )

    studio.execute(request)

    assert workspace.specification() == request
    current = workspace.current_request()
    assert current is not None
    assert current["kind"] == "create"
    assert any(
        "Treat this prose as the initial specification" in line
        for line in output
    )
    assert any("workflow create" in line for line in output)


def test_unmatched_design_prose_is_offered_as_one_pending_refinement(tmp_path):
    studio, workspace, _output = _studio(tmp_path, responses=["y"])
    workspace.save_specification("Writer drafts an answer.")
    request = "Add a Reviewer participant that must approve every answer"

    studio.execute(request)

    assert workspace.pending_refinement() == request
    current = workspace.current_request()
    assert current is not None
    assert current["kind"] == "refine"


@pytest.mark.parametrize(
    "sentence",
    [
        "A writer drafts a reply and a reviewer may request up to three revisions.",
        "Draft an answer, then a reviewer approves or rejects it.",
        "Two agents debate a topic until they reach consensus.",
        "Summarize incoming support emails and route urgent ones to a human.",
        "A researcher gathers sources, a critic checks them, a human signs off.",
        "Build a workflow where a writer drafts and a reviewer approves.",
        "I want an agent that triages GitHub issues with human approval.",
        "The reviewer should be able to reject a draft twice.",
        "Classify each invoice, then ask an accountant to confirm the total.",
        "Translate a document and have a native speaker verify the result.",
    ],
)
def test_realistic_declarative_requirements_are_offered_without_framework_words(
    sentence,
):
    plan = requirement_proposal(sentence, has_specification=False)

    assert plan is not None
    assert plan.requires_confirmation is True
    assert plan.commands == (
        f"workflow create {shlex.quote(sentence)}",
    )


@pytest.mark.parametrize(
    "sentence",
    [
        "How do I inspect the complete workflow?",
        "Can Studio check whether my local model is connected?",
        "A reviewer approves every draft?",
        "The deployment failed after the latest restart.",
        "Codex is installed but Studio cannot find it.",
        "The local provider connection timed out again.",
    ],
)
def test_questions_and_operational_failures_are_not_specification_proposals(
    sentence,
):
    assert requirement_proposal(sentence, has_specification=True) is None


def test_rejected_requirement_can_be_interpreted_as_a_command(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path, responses=["command"])
    monkeypatch.setattr(
        studio,
        "_interpret_with_cli",
        lambda request_text, *, configured: NaturalCommandPlan(
            "Show current project state.",
            ("current",),
            "codex",
        ),
    )

    studio.execute("Summarize incoming support emails for a human operator")

    assert workspace.specification() is None
    assert any("No specification was changed" in line for line in output)
    assert "Current" in output
    history = NaturalLanguageStore(workspace.natural_language_path).history()
    assert [entry["status"] for entry in history[-2:]] == [
        "redirected",
        "executed",
    ]


def test_explicit_ask_bypasses_the_requirement_offer(tmp_path, monkeypatch):
    studio, workspace, output = _studio(tmp_path)
    monkeypatch.setattr(
        studio,
        "_interpret_with_cli",
        lambda request_text, *, configured: NaturalCommandPlan(
            "Show current project state.",
            ("current",),
            "claude",
        ),
    )

    studio.execute("ask Summarize incoming support emails for a human operator")

    assert workspace.specification() is None
    assert "Current" in output


def test_short_start_over_phrase_requests_clarification(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    workspace.initialize_project(name="Tutorial")
    workspace.save_specification("Create a review workflow.")

    studio.execute("start over")

    assert workspace.specification() == "Create a review workflow."
    assert workspace.manifest_path.exists()
    assert "Clarification needed" in output
    assert any("project reset fresh" in line for line in output)


def test_explicit_reset_everything_still_proposes_recoverable_fresh_reset(
    tmp_path,
):
    studio, workspace, output = _studio(tmp_path, responses=["y"])
    workspace.initialize_project(name="Tutorial")
    workspace.save_specification("Create a review workflow.")

    studio.execute("reset everything")

    assert workspace.specification() is None
    assert not workspace.manifest_path.exists()
    assert any("project reset fresh" in line for line in output)


def test_confirmed_natural_discard_does_not_confirm_twice(
    tmp_path,
    monkeypatch,
):
    prompts: list[str] = []
    studio, workspace, _output = _studio(tmp_path)
    studio.input = lambda prompt: prompts.append(prompt) or "y"
    workspace.save_specification("Create a review workflow.")
    studio.refine_request("Add a Reviewer.")
    monkeypatch.setattr(
        studio,
        "_interpret_with_cli",
        lambda request_text, *, configured: NaturalCommandPlan(
            "Discard the pending refinement.",
            ("workflow discard",),
            "codex",
        ),
    )

    studio.execute("Remove the pending amendment")

    assert workspace.pending_refinement() is None
    assert prompts == ["Execute this destructive plan? [y/n]: "]


def test_help_me_get_started_is_local_and_deterministic(tmp_path):
    studio, workspace, output = _studio(tmp_path)

    studio.execute("help me get started")

    assert any("Show the short Studio path" in line for line in output)
    assert any("Getting started:" in line for line in output)
    history = NaturalLanguageStore(workspace.natural_language_path).history()
    assert history[-1]["source"] == "deterministic"


def test_natural_show_phrase_wins_over_invalid_show_syntax(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)

    studio.execute("Show me the whole protocol")

    assert any("workflow show protocol" in line for line in output)
    assert any("@workflow" in line for line in output)


def test_natural_workflow_discovery_and_source_requests_are_deterministic(
    tmp_path,
):
    studio, workspace, output = _studio(tmp_path)

    studio.execute("Show me the available workflows")

    assert any("workflow list" in line for line in output)
    assert any("workflow.py:sample" in line for line in output)

    output.clear()
    studio.execute("Show me the authored Python source")

    assert workspace.current_workflow == "workflow.py:sample"
    assert any("workflow show source" in line for line in output)
    assert any("Source: workflow.py" in line for line in output)


def test_natural_workflow_review_request_opens_the_guided_review(tmp_path):
    studio, workspace, output = _studio(tmp_path, responses=["6"])
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_specification("Echo the request through Writer.")
    studio.refine_request("Require human approval before returning.")
    workspace.save_specification(
        "Echo the request through Writer and require human approval "
        "before returning."
    )

    studio.execute("Review the current workflow implementation")

    assert any("workflow review" in line for line in output)
    assert "Workflow review" in output
    assert any("Review remains open" in line for line in output)
    history = NaturalLanguageStore(workspace.natural_language_path).history()
    assert history[-1]["source"] == "deterministic"
    assert history[-1]["commands"] == ["workflow review"]


def test_natural_prose_with_an_apostrophe_is_not_treated_as_broken_shell_syntax(
    tmp_path,
):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)

    studio.execute("Show Writer's local view")

    assert any("workflow show agent Writer" in line for line in output)
    assert workspace.load()["last_view"] == "agent Writer"


def test_natural_model_assignment_is_canonical_and_reversible(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)

    studio.execute("Assign the mock model to Writer")

    profile = workspace.model_profile("workflow.py:sample", default="mock")
    assert profile["lifelines"] == {"Writer": "mock"}
    assert any("model assign Writer mock" in line for line in output)
    assert any("Natural-language command plan completed" in line for line in output)


def test_natural_model_configuration_rename_is_deterministic(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    workspace.save_model_configuration(
        "fast-review",
        {
            "provider": "local",
            "model": "qwen3",
            "spec": "local:qwen3",
            "check_status": "available",
        },
    )

    studio.execute("Rename model configuration fast-review to editorial")

    assert "fast-review" not in workspace.model_configurations()
    assert workspace.model_configurations()["editorial"]["spec"] == "local:qwen3"
    assert any(
        "model config rename fast-review editorial" in line
        for line in output
    )
    history = NaturalLanguageStore(workspace.natural_language_path).history()
    assert history[-1]["source"] == "deterministic"


def test_natural_project_rename_and_global_learning_are_deterministic(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    studio.execute("project init Tutorial")

    studio.execute("Rename the project to Reviewed Answer")
    studio.execute("Turn learning off")

    assert workspace.project_manifest()["name"] == "Reviewed Answer"
    assert workspace.global_settings()["learning"] is False
    assert any(
        "project rename 'Reviewed Answer'" in line
        for line in output
    )
    assert any("settings set learning off" in line for line in output)
    history = NaturalLanguageStore(workspace.natural_language_path).history()
    assert [record["source"] for record in history[-2:]] == [
        "deterministic",
        "deterministic",
    ]


def test_natural_studio_restart_is_deterministic_and_confirmed(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path, responses=["y"])
    launcher = tmp_path / "zippergen"
    launcher.write_text("#!/bin/sh\n")
    launcher.chmod(0o755)
    calls: list[tuple[str, list[str]]] = []

    monkeypatch.setattr("zippergen.studio.sys.argv", [str(launcher)])
    monkeypatch.setattr(
        "zippergen.studio.os.execv",
        lambda executable, arguments: calls.append(
            (executable, list(arguments))
        ),
    )

    studio.execute("Please restart ZipperGen Studio")

    assert calls == [(str(launcher), [str(launcher)])]
    assert any("studio restart" in line for line in output)
    history = NaturalLanguageStore(workspace.natural_language_path).history()
    assert history[-1]["source"] == "deterministic"
    assert history[-1]["commands"] == ["studio restart"]
    assert history[-1]["status"] == "executed"


def test_natural_provider_configuration_uses_the_models_surface(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(
        tmp_path,
        secret_responses=["private-anthropic-key"],
    )

    class ModelsResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, limit=-1):
            return b'{"data":[{"id":"claude-sonnet-4-6"}]}'

    monkeypatch.setattr(
        "zippergen.studio_models.request.urlopen",
        lambda req, *, timeout: ModelsResponse(),
    )

    studio.execute("Configure Claude as a model provider")

    assert workspace.load_secrets()["ANTHROPIC_API_KEY"] == (
        "private-anthropic-key"
    )
    assert workspace.provider_profiles()["anthropic"]["check_status"] == (
        "reachable"
    )
    assert set(workspace.model_configurations()) == {"mock"}
    assert any(
        "model provider configure anthropic" in line
        for line in output
    )
    assert not any("providers set" in line for line in output)


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
            "commands": ["workflow show agent Writer"],
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
    assert learned[0]["commands"] == [
        "workflow show agent {participant}"
    ]
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
        "commands": ["deploy stop review"],
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
        "commands": ["current", "model"],
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
    assert history[-1]["commands"] == ["current", "model"]
    assert history[-1]["status"] == "executed"
    assert any("Executing 1/2: current" in line for line in output)
    assert any("Executing 2/2: model" in line for line in output)


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
            ("workflow show agent Writer",),
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
        ("workflow show agent 'Lead Writer'",),
    )
    store = NaturalLanguageStore(tmp_path / "natural-language.json")
    store.remember(
        "Show what Lead Writer sees",
        NaturalCommandPlan("Show participant.", commands, "codex"),
    )

    assert template == "show what {participant} sees"
    assert commands == ("workflow show agent {participant}",)


def test_generalization_uses_current_model_assignment_syntax():
    template, commands = generalize_interpretation(
        "Assign careful review to Lead Writer",
        ("model assign 'Lead Writer' 'careful review'",),
    )

    assert template == "assign {configuration} to {participant}"
    assert commands == (
        "model assign {participant} {configuration}",
    )
