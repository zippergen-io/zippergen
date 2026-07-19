import json
import subprocess
from io import StringIO
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


def test_studio_use_discovers_workflow_without_importing_for_selection(tmp_path):
    studio, workspace, output = _studio(tmp_path, responses=["1"])

    studio.use_workflow([])

    assert workspace.current_workflow == "workflow.py:sample"
    assert output[0] == "Workflows"
    assert "workflow.py:sample" in output[1]
    assert output[-1].startswith("✓ Current workflow: workflow.py:sample")


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
    assert content == workspace.current_task_path.read_text()
    assert output[0] == "Creation"
    assert any("✓ P001 registered" in line for line in output)
    assert any("✓ .zippergen/current-task.md" in line for line in output)
    assert any("assistant" in line for line in output)
    assert all("Pass this brief" not in line for line in output)


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
    assert output[0] == "Creation"
    assert any(
        "P001 registered — prompts/reviewed answer.md" in line
        for line in output
    )
    assert all("Loaded prompt file" not in line for line in output)


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
    assert output[0] == "Refinement"
    assert any("✓ P001 registered" in line for line in output)
    assert any("✓ .zippergen/current-task.md" in line for line in output)


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
    assert output[0] == "Refinement"
    assert any("✓ P001 registered" in line for line in output)
    assert all("Loaded prompt file" not in line for line in output)


def test_studio_task_commands_expose_one_stable_task_and_private_history(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    studio.create_request("Create a review workflow.")
    output.clear()

    studio.execute("task")
    assert output[0] == "Current task"
    assert any(".zippergen/current-task.md" in line for line in output)
    assert any("assistant" in line for line in output)

    output.clear()
    studio.execute("task path")
    assert output == [str(workspace.current_task_path)]

    output.clear()
    studio.execute("task show")
    assert output == [workspace.current_task_path.read_text().rstrip()]

    output.clear()
    studio.execute("task history")
    assert output[0] == "Task history"
    assert any("create" in line and "P001" in line for line in output)


def test_studio_assistant_launches_codex_in_project_on_the_stable_task(
    tmp_path, monkeypatch
):
    studio, workspace, output = _studio(tmp_path)
    studio.create_request("Create a review workflow.")
    output.clear()
    calls: list[tuple[list[str], Path, bool]] = []

    monkeypatch.setattr("zippergen.studio.shutil.which", lambda name: "/bin/codex")

    def fake_run(arguments, *, cwd, check):
        calls.append((arguments, cwd, check))
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr("zippergen.studio.subprocess.run", fake_run)

    studio.execute("assistant")

    assert calls[0][0][0:3] == [
        "/bin/codex",
        "--cd",
        str(workspace.root),
    ]
    assert ".zippergen/current-task.md" in calls[0][0][3]
    assert calls[0][1] == workspace.root
    assert calls[0][2] is False
    assert output[0] == "Assistant"
    assert any("MCP" in line and "not required" in line for line in output)
    assert output[-1].startswith("✓ Codex session ended")


def test_studio_assistant_reports_missing_codex_without_losing_task(
    tmp_path, monkeypatch
):
    studio, workspace, _output = _studio(tmp_path)
    studio.create_request("Create a review workflow.")
    monkeypatch.setattr("zippergen.studio.shutil.which", lambda name: None)

    try:
        studio.execute("assistant")
    except SystemExit as exc:
        assert "Codex CLI was not found" in str(exc)
        assert "codex login" in str(exc)
    else:
        raise AssertionError("assistant should fail when Codex is not installed")
    assert workspace.current_task_path.exists()


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
    assert "project init [NAME]" in output[-1]
    assert "prompts move ID before|after ID" in output[-1]
    assert "task show|path|history" in output[-1]
    assert "assistant" in output[-1]
    assert "create --file PATH" in output[-1]
    assert "refine --file PATH" in output[-1]
    assert "providers set local [URL]" in output[-1]
    assert studio.execute("not-a-command") is True
    assert output[-1].startswith("✗ Unknown command")
    assert studio.execute("exit") is False


def test_studio_status_marks_use_color_only_when_enabled(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    workspace = Workspace(root, home=tmp_path / "home")
    output: list[str] = []
    studio = Studio(workspace, output_func=output.append, color=True)

    studio.execute("project init Tutorial")
    studio.execute("not-a-command")

    assert output[0].startswith(
        "\033[32m✓\033[0m Project manifest created:"
    )
    assert output[-1].startswith(
        "\033[31m✗\033[0m Unknown command:"
    )


def test_studio_automatic_color_respects_no_color(tmp_path, monkeypatch):
    class InteractiveOutput(StringIO):
        def isatty(self):
            return True

    root = tmp_path / "project"
    root.mkdir()
    workspace = Workspace(root, home=tmp_path / "home")
    monkeypatch.setattr("sys.stdout", InteractiveOutput())
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TERM", raising=False)

    assert Studio(workspace).color is True

    monkeypatch.setenv("NO_COLOR", "")
    assert Studio(workspace).color is False


def test_studio_validation_marks_successful_checks(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)

    studio.validate()

    assert output[0] == "✓ Workflow sample: valid"
    assert all(line.startswith("  ✓ ") for line in output[1:])


def test_studio_interactive_errors_have_a_failure_mark(tmp_path):
    studio, _workspace, output = _studio(
        tmp_path,
        responses=["validate", "exit"],
    )

    assert studio.run() == 0

    assert any(
        line.startswith("✗ No workflow selected.")
        for line in output
    )


def test_studio_project_and_prompt_commands_manage_visible_design_context(tmp_path):
    studio, workspace, output = _studio(tmp_path)

    studio.execute("project init Tutorial")
    studio.execute("prompts add First requirement")
    studio.execute("prompts add Second requirement")
    studio.execute("prompts move P002 before P001")
    studio.execute("prompts remove P001")
    studio.execute("prompts context")

    assert workspace.project_manifest()["name"] == "Tutorial"
    assert workspace.manifest_path.exists()
    records = workspace.list_prompts()
    assert [record["id"] for record in records] == ["P002", "P001"]
    assert records[0]["active"] is True
    assert records[1]["active"] is False
    assert "Second requirement" in output[-1]
    assert "First requirement" not in output[-1]


def test_studio_replacement_preserves_prompt_history(tmp_path):
    studio, workspace, output = _studio(tmp_path)

    studio.execute("prompts add Keep retries bounded")
    studio.execute("prompts replace P001 Use exactly three retries")
    studio.execute("prompts")

    records = workspace.list_prompts()
    assert [record["id"] for record in records] == ["P001", "P002"]
    assert records[0]["active"] is False
    assert records[1]["replaces"] == "P001"
    assert any("P001" in line and "archived" in line for line in output)
    assert any("P002" in line and "active" in line for line in output)


def test_studio_handoff_contains_all_active_prompts_in_order(tmp_path):
    studio, workspace, _output = _studio(tmp_path)
    studio.create_request("Create a concise answer workflow.")
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    studio.refine_request("Add an explicit reviewer.")

    refine_records = list(workspace.requests_directory.glob("*-refine.json"))
    assert len(refine_records) == 1
    metadata = json.loads(refine_records[0].read_text())
    brief = refine_records[0].with_suffix(".md").read_text()

    assert metadata["prompt_id"] == "P002"
    assert metadata["active_prompt_ids"] == ["P001", "P002"]
    assert brief.index("P001 [initial]") < brief.index("P002 [refinement]")
    assert "Later prompts take precedence only where" in brief
    assert "preserve every unaffected earlier" in brief


def test_studio_current_is_a_complete_project_dashboard(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    studio.create_request("Create a sampled workflow.")
    output.clear()

    studio.show_current()

    assert output[0] == "Current"
    assert "Project" in output
    assert "Workflow" in output
    assert "Models" in output
    assert "Runtime" in output
    assert any("Name" in line and "project" in line for line in output)
    assert any("Prompts" in line and "1 active" in line for line in output)
    assert any("Selected" in line and "workflow.py:sample" in line for line in output)
    assert any("Name" in line and "sample" in line for line in output)
    assert any("Participants" in line and "User, Writer" in line for line in output)
    assert any("Connectors" in line and "none" in line for line in output)
    assert any("Validation" in line and "✓ valid" in line for line in output)
    assert any("Writer" in line and "mock" in line for line in output)
    assert any("Provider mock" in line and "ready; built in" in line for line in output)
    assert any("Run" in line and "none" in line for line in output)
    assert any("Deployment" in line and "none" in line for line in output)


def test_studio_current_is_explicit_before_a_workflow_exists(tmp_path):
    studio, _workspace, output = _studio(tmp_path)

    studio.show_current()

    assert output[0] == "Current"
    assert any("Selected" in line and "⚠ none" in line for line in output)
    assert any("Participants" in line and "0 — none" in line for line in output)
    assert any("Connectors" in line and "none" in line for line in output)
    assert any("Validation" in line and "⚠ not available" in line for line in output)
    assert any("Assignments" in line and "none" in line for line in output)
    assert any("Providers" in line and "none" in line for line in output)


def test_studio_configures_api_and_local_providers_without_displaying_secrets(
    tmp_path,
):
    studio, workspace, output = _studio(
        tmp_path,
        secret_responses=["super-secret-key"],
    )

    studio.execute("providers set openai")
    studio.execute("providers set local http://localhost:1234/v1")
    studio.execute("providers")

    assert workspace.load_secrets() == {"OPENAI_API_KEY": "super-secret-key"}
    assert workspace.provider_profiles()["local"]["base_url"] == (
        "http://localhost:1234/v1"
    )
    assert workspace.secrets_path.stat().st_mode & 0o077 == 0
    assert all("super-secret-key" not in line for line in output)
    assert any("openai: ready" in line for line in output)
    assert any("local: endpoint http://localhost:1234/v1" in line for line in output)

    studio.execute("providers reset openai")
    assert "OPENAI_API_KEY" not in workspace.load_secrets()
    assert "openai" not in workspace.provider_profiles()


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
    assert output[-1] == "✓ Deployment completed: sample-test"


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
    assert (
        "  Writer: openai:gpt-4o-mini (override; actions: echo)"
    ) in output
    assert output[-1].startswith("  ⚠ Provider openai: not configured")


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
