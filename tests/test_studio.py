import json
import subprocess
from io import BytesIO, StringIO
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from zippergen.studio import Studio, StudioCompleter
from zippergen.workspace import Workspace, WorkspaceError


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

DEPLOYMENT_SOURCE = WORKFLOW_SOURCE + """

from zippergen import DeploymentField, DeploymentSpec

zippergen_deployment = DeploymentSpec(
    name="sample",
    fields=(
        DeploymentField(
            "openai_api_key",
            "OpenAI API key",
            target="env",
            env="OPENAI_API_KEY",
            secret=True,
            required=True,
            when="llm",
            when_values=("openai*",),
        ),
    ),
)
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


def _completions(studio: Studio, text: str) -> list[str]:
    document = Document(text, cursor_position=len(text))
    event = CompleteEvent(completion_requested=True)
    return [
        completion.text
        for completion in StudioCompleter(studio).get_completions(document, event)
    ]


def test_studio_completion_is_context_and_project_aware(tmp_path):
    studio, workspace, _output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    (workspace.root / "requirements.md").write_text("Create a workflow.\n")
    notes = workspace.root / "notes folder"
    notes.mkdir()
    (notes / "spec.md").write_text("Another workflow.\n")

    assert _completions(studio, "sp") == ["spec"]
    assert {"refine", "reconcile"}.issubset(_completions(studio, "spec r"))
    assert _completions(studio, "use wor") == ["workflow.py:sample"]
    assert _completions(studio, "show agent W") == ["Writer"]
    assert _completions(studio, "models set W") == ["Writer"]
    assert _completions(studio, "providers set a") == ["anthropic"]
    assert _completions(studio, "providers ch") == ["check"]
    assert _completions(studio, "providers check l") == ["local"]
    assert _completions(studio, "create --file req") == ["requirements.md"]
    assert _completions(studio, "create --file 'notes f") == ["'notes folder/'"]
    assert _completions(studio, "create --file 'notes folder/'") == [
        "'notes folder/spec.md'"
    ]


def test_studio_explains_a_single_completion_match(tmp_path):
    studio, _workspace, _output = _studio(tmp_path)

    assert studio.completion_explanation("resu") == (
        " Tab: resume — resume the current incomplete run "
    )
    assert studio.completion_explanation("spec r") == ""
    assert studio.completion_explanation("") == ""


def test_studio_run_uses_prompt_toolkit_session_when_interactive(tmp_path):
    studio, _workspace, output = _studio(tmp_path)
    prompts: list[tuple[str, bool]] = []

    class FakeSession:
        def prompt(self, value: str, *, complete_in_thread: bool) -> str:
            prompts.append((value, complete_in_thread))
            return "exit"

    studio._prompt_toolkit_enabled = True
    studio._prompt_session = FakeSession()  # type: ignore[assignment]

    assert studio.run() == 0
    assert prompts == [("zippergen [no workflow]> ", True)]
    assert any("press Tab to complete" in line for line in output)


def test_studio_command_history_is_owner_only(tmp_path):
    studio, workspace, _output = _studio(tmp_path)
    workspace.directory.mkdir(parents=True)
    history = workspace.directory / "studio.history"
    history.write_text("# command\n+current\n")
    history.chmod(0o644)

    studio._protect_studio_history()

    assert history.stat().st_mode & 0o777 == 0o600


def test_studio_completion_never_breaks_input_on_invalid_private_state(tmp_path):
    studio, workspace, _output = _studio(tmp_path)
    workspace.directory.mkdir(parents=True)
    workspace.state_path.write_text("not valid JSON")

    assert _completions(studio, "status ") == []


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
    assert "Use $zippergen-workflows" in content
    assert "visible Python source" in content
    assert "Do not deploy" in content
    assert content == workspace.current_task_path.read_text()
    assert workspace.specification_path.name == "specification.md"
    assert workspace.specification() == (
        "Draft an answer and ask a reviewer to approve it."
    )
    assert output[0] == "Creation"
    assert any("✓ specification.md" in line for line in output)
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
    assert workspace.specification() == metadata["prompt"]
    assert metadata["specification_file"] == str(workspace.specification_path)
    assert output[0] == "Creation"
    assert any("✓ specification.md" in line for line in output)
    assert all("Loaded prompt file" not in line for line in output)


def test_studio_inline_create_does_not_replace_an_existing_specification(tmp_path):
    studio, workspace, _output = _studio(tmp_path)
    studio.create_request("Original accepted requirements.")

    try:
        studio.execute("create Different requirements")
    except SystemExit as exc:
        assert "canonical specification already exists" in str(exc)
    else:
        raise AssertionError("inline create must not overwrite accepted intent")

    assert workspace.specification() == "Original accepted requirements."


def test_studio_refine_saves_semantic_baseline_and_handoff(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_specification("Echo a value through Writer and return it.")

    studio.refine_request("Add a human review before returning the result.")

    baselines = list(workspace.requests_directory.glob("*-semantic-before.json"))
    briefs = list(workspace.requests_directory.glob("*-refine.md"))
    assert len(baselines) == 1
    assert len(briefs) == 1
    content = briefs[0].read_text()
    assert str(baselines[0]) in content
    assert "Preserve all behavior not explicitly changed" in content
    assert "zippergen diff" in content
    assert "# Canonical workflow specification" in content
    assert "# Pending refinement" in content
    assert workspace.pending_refinement() == (
        "Add a human review before returning the result."
    )
    assert output[0] == "Refinement"
    assert any("✓ created — .zippergen/pending-refinement.md" in line for line in output)
    assert any("✓ .zippergen/current-task.md" in line for line in output)


def test_studio_refine_reads_prompt_from_absolute_file(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_specification("Echo a value through Writer and return it.")
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
    assert workspace.pending_refinement() == (
        "Add human review.\nPreserve the existing model call."
    )
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
    assert any("create" in line for line in output)


def test_studio_task_show_refreshes_stale_specification_context_once(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    studio.create_request("Create a review workflow.")
    original = workspace.current_request()
    assert original is not None
    workspace.save_specification(
        "Create a review workflow and add an explicit failure result after "
        "retry exhaustion."
    )
    output.clear()

    studio.execute("task show")

    refreshed = workspace.current_request()
    assert refreshed is not None
    assert refreshed["request_id"] != original["request_id"]
    assert refreshed["refreshes_request"] == original["request_id"]
    assert refreshed["specification_fingerprint"] == (
        workspace.specification_fingerprint()
    )
    assert output[0] == "✓ Task refreshed from the current specification context."
    assert "add an explicit failure result" in output[1]
    assert len(workspace.list_requests()) == 2

    output.clear()
    studio.execute("task show")

    assert output == [workspace.current_task_path.read_text().rstrip()]
    assert len(workspace.list_requests()) == 2

    output.clear()
    studio.execute("task history")

    assert "Refreshes" in output[2]
    assert any(
        str(refreshed["request_id"]) in line
        and str(original["request_id"]) in line
        for line in output
    )


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
    assert any(line.startswith("✓ Codex session ended") for line in output)
    assert any(
        "Status" in line and "awaiting human review" in line for line in output
    )
    assert workspace.current_request()["status"] == "awaiting_review"


def test_studio_assistant_refreshes_edited_specification_before_launch(
    tmp_path, monkeypatch
):
    studio, workspace, output = _studio(tmp_path)
    studio.create_request("Original creation requirement.")
    original = workspace.current_request()
    assert original is not None
    workspace.save_specification("Corrected creation requirement.")
    output.clear()
    calls: list[tuple[list[str], Path, bool]] = []
    monkeypatch.setattr("zippergen.studio.shutil.which", lambda name: "/bin/codex")

    def fake_run(arguments, *, cwd, check):
        task = workspace.current_task_path.read_text()
        assert "Corrected creation requirement." in task
        assert "Original creation requirement." not in task
        calls.append((arguments, cwd, check))
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr("zippergen.studio.subprocess.run", fake_run)

    studio.execute("assistant")

    refreshed = workspace.current_request()
    assert refreshed is not None
    assert refreshed["request_id"] != original["request_id"]
    assert refreshed["refreshes_request"] == original["request_id"]
    assert refreshed["specification_fingerprint"] == (
        workspace.specification_fingerprint()
    )
    assert calls
    assert output[0] == "✓ Task refreshed from the current specification context."
    assert output[1] == "Assistant"


def test_studio_assistant_can_launch_claude_code_on_the_same_task(
    tmp_path, monkeypatch
):
    studio, workspace, output = _studio(tmp_path)
    studio.create_request("Create a review workflow.")
    output.clear()
    calls: list[tuple[list[str], Path, bool]] = []

    def find_assistant(name: str):
        assert name == "claude"
        return "/bin/claude"

    monkeypatch.setattr("zippergen.studio.shutil.which", find_assistant)

    def fake_run(arguments, *, cwd, check):
        calls.append((arguments, cwd, check))
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr("zippergen.studio.subprocess.run", fake_run)

    studio.execute("assistant claude")

    assert calls[0][0][0:4] == [
        "/bin/claude",
        "--print",
        "--permission-mode",
        "acceptEdits",
    ]
    assert len(calls[0][0]) == 5
    assert ".zippergen/current-task.md" in calls[0][0][4]
    assert calls[0][1] == workspace.root
    assert calls[0][2] is False
    assert any("Tool" in line and "Claude Code" in line for line in output)
    assert any("Mode" in line and "one-shot task" in line for line in output)
    assert any(line.startswith("✓ Claude Code session ended") for line in output)
    assert workspace.current_request()["status"] == "awaiting_review"


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


def test_studio_assistant_reports_missing_claude_and_rejects_unknown_tools(
    tmp_path, monkeypatch
):
    studio, workspace, _output = _studio(tmp_path)
    studio.create_request("Create a review workflow.")
    monkeypatch.setattr("zippergen.studio.shutil.which", lambda name: None)

    try:
        studio.execute("assistant claude")
    except SystemExit as exc:
        assert "Claude Code was not found" in str(exc)
        assert "first-run authentication" in str(exc)
    else:
        raise AssertionError("assistant claude should require Claude Code")

    try:
        studio.execute("assistant unknown")
    except SystemExit as exc:
        assert "assistant codex" in str(exc)
        assert "assistant claude" in str(exc)
    else:
        raise AssertionError("unknown assistants should be rejected")
    assert workspace.current_task_path.exists()


def test_studio_completed_refinement_task_waits_for_review_without_refreshing(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_specification("Echo the request through Writer.")
    studio.refine_request("Require human approval before returning.")
    original = workspace.current_request()
    assert original is not None
    monkeypatch.setattr("zippergen.studio.shutil.which", lambda name: "/bin/codex")

    def fake_run(arguments, *, cwd, check):
        workspace.save_specification(
            "Echo the request through Writer and require human approval "
            "before returning."
        )
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr("zippergen.studio.subprocess.run", fake_run)

    studio.execute("assistant codex")
    completed = workspace.current_request()
    assert completed is not None
    assert completed["request_id"] == original["request_id"]
    assert completed["status"] == "awaiting_review"
    assert completed["assistant"] == "Codex"
    assert completed["assistant_exit_code"] == 0
    output.clear()

    studio.execute("task")

    reviewed = workspace.current_request()
    assert reviewed is not None
    assert reviewed["request_id"] == original["request_id"]
    assert len(workspace.list_requests()) == 1
    assert any(
        "Status" in line and "awaiting human review" in line for line in output
    )
    assert any(
        "Execution" in line and "nothing is scheduled" in line for line in output
    )
    assert any(
        "Next" in line and "spec reconcile" in line for line in output
    )
    assert all("Task refreshed" not in line for line in output)

    output.clear()
    studio.execute("current")

    assert any(
        "Task" in line and "awaiting human review" in line for line in output
    )
    assert any(
        "Task next" in line and "spec reconcile" in line for line in output
    )
    assert any(
        "Refinement" in line and "awaiting human review" in line
        for line in output
    )

    output.clear()
    studio.execute("spec pending")

    assert any(
        "Status" in line and "awaiting human review" in line for line in output
    )
    assert any(
        "Next" in line and "spec reconcile" in line for line in output
    )

    with pytest.raises(SystemExit, match="already returned.*awaiting human review"):
        studio.execute("assistant codex")


def test_studio_explicit_assistant_rerun_prepares_a_new_task(
    tmp_path,
    monkeypatch,
):
    studio, workspace, _output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_specification("Echo the request through Writer.")
    studio.refine_request("Require human approval before returning.")
    first = workspace.current_request()
    assert first is not None
    monkeypatch.setattr("zippergen.studio.shutil.which", lambda name: "/bin/codex")

    def fake_run(arguments, *, cwd, check):
        workspace.save_specification(
            "Echo the request through Writer and require human approval "
            "before returning."
        )
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr("zippergen.studio.subprocess.run", fake_run)
    studio.execute("assistant codex")

    studio.execute("assistant codex --rerun")

    rerun = workspace.current_request()
    assert rerun is not None
    assert rerun["request_id"] != first["request_id"]
    assert rerun["refreshes_request"] == first["request_id"]
    assert rerun["status"] == "awaiting_review"
    assert len(workspace.list_requests()) == 2


def test_studio_failed_and_interrupted_assistants_have_explicit_task_states(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path)
    studio.create_request("Create a review workflow.")
    monkeypatch.setattr("zippergen.studio.shutil.which", lambda name: "/bin/codex")
    monkeypatch.setattr(
        "zippergen.studio.subprocess.run",
        lambda arguments, **kwargs: subprocess.CompletedProcess(arguments, 7),
    )

    with pytest.raises(SystemExit, match="exited with status 7"):
        studio.execute("assistant codex")

    failed = workspace.current_request()
    assert failed is not None
    assert failed["status"] == "assistant_failed"
    assert failed["assistant_exit_code"] == 7
    output.clear()
    studio.execute("task")
    assert any("Status" in line and "assistant failed" in line for line in output)

    def interrupt(arguments, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("zippergen.studio.subprocess.run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        studio.execute("assistant codex")

    interrupted = workspace.current_request()
    assert interrupted is not None
    assert interrupted["status"] == "assistant_interrupted"


def test_studio_recovers_an_orphaned_running_assistant_as_interrupted(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path)
    studio.create_request("Create a review workflow.")
    request = workspace.current_request()
    assert request is not None
    workspace.update_request(
        str(request["request_id"]),
        status="assistant_running",
        assistant="Codex",
        studio_process_id=12345,
    )

    def missing_process(pid, signal):
        assert (pid, signal) == (12345, 0)
        raise ProcessLookupError

    monkeypatch.setattr("zippergen.studio.os.kill", missing_process)
    output.clear()

    studio.execute("task")

    recovered = workspace.current_request()
    assert recovered is not None
    assert recovered["status"] == "assistant_interrupted"
    assert any(
        "Status" in line and "assistant interrupted" in line for line in output
    )


def test_studio_infers_review_state_for_an_existing_integrated_refinement(
    tmp_path,
):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_specification("Echo the request through Writer.")
    studio.refine_request("Require human approval before returning.")
    original = workspace.current_request()
    assert original is not None
    metadata_path = workspace.request_path(str(original["request_id"]))
    metadata = json.loads(metadata_path.read_text())
    metadata.pop("status")
    metadata_path.write_text(json.dumps(metadata))
    workspace.save_specification(
        "Echo the request through Writer and require human approval "
        "before returning."
    )
    output.clear()

    studio.execute("task")

    migrated = workspace.current_request()
    assert migrated is not None
    assert migrated["request_id"] == original["request_id"]
    assert migrated["status"] == "awaiting_review"
    assert migrated["lifecycle_inferred"] is True
    assert len(workspace.list_requests()) == 1
    assert any(
        "Status" in line and "awaiting human review" in line for line in output
    )
    assert any(
        "Assistant" in line and "review inferred" in line for line in output
    )


def test_studio_manual_spec_integration_is_reviewable_without_an_assistant(
    tmp_path,
):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_specification("Echo the request through Writer.")
    studio.refine_request("Require human approval before returning.")
    original = workspace.current_request()
    assert original is not None
    workspace.save_specification(
        "Echo the request through Writer and require human approval "
        "before returning."
    )
    output.clear()

    studio.execute("task")

    manual = workspace.current_request()
    assert manual is not None
    assert manual["request_id"] == original["request_id"]
    assert manual["status"] == "awaiting_review"
    assert manual["manual_integration"] is True
    assert any(
        "Assistant" in line and "edited manually" in line for line in output
    )
    assert any(
        "Execution" in line and "assistant not run" in line for line in output
    )
    assert any("Next" in line and "spec reconcile" in line for line in output)


def test_studio_task_close_keeps_history_and_refinements_use_reconcile(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    studio.create_request("Create a review workflow.")

    studio.execute("task close --yes")

    assert workspace.current_request() is None
    assert not workspace.current_task_path.exists()
    assert workspace.list_requests()[0]["status"] == "closed"
    assert any(line == "Task closed" for line in output)

    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_specification("Echo the request through Writer.")
    studio.refine_request("Require human approval.")
    with pytest.raises(SystemExit, match="spec reconcile.*spec discard"):
        studio.execute("task close --yes")


def test_studio_remembers_shows_and_resets_editor_preference(
    tmp_path, monkeypatch
):
    studio, workspace, output = _studio(tmp_path)
    monkeypatch.setattr(
        "zippergen.studio.shutil.which",
        lambda name: f"/usr/bin/{name}" if name == "micro" else None,
    )

    studio.execute("editor set micro")

    assert workspace.load()["editor_command"] == ["micro"]
    assert output[-1] == "✓ Editor preference: micro"

    output.clear()
    studio.execute("editor show")

    assert output[0] == "Editor"
    assert any("Preference" in line and "micro" in line for line in output)
    assert any("Effective" in line and "/usr/bin/micro" in line for line in output)
    assert any("Source" in line and "project preference" in line for line in output)

    studio.execute("editor reset")

    assert workspace.load()["editor_command"] is None
    assert output[-1] == "✓ Editor preference reset to automatic discovery."


def test_studio_edits_selected_workflow_with_preference_or_one_off_override(
    tmp_path, monkeypatch
):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.update(editor_command=["nano"])
    calls: list[tuple[list[str], Path, bool]] = []

    def find_editor(name: str):
        return {"nano": "/usr/bin/nano", "micro": "/opt/bin/micro"}.get(name)

    def fake_run(arguments, *, cwd, check):
        calls.append((arguments, cwd, check))
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr("zippergen.studio.shutil.which", find_editor)
    monkeypatch.setattr("zippergen.studio.subprocess.run", fake_run)

    studio.execute("edit workflow")
    studio.execute("edit workflow --editor micro")

    assert calls == [
        (["/usr/bin/nano", str(workspace.root / "workflow.py")], workspace.root, False),
        (["/opt/bin/micro", str(workspace.root / "workflow.py")], workspace.root, False),
    ]
    assert any("project preference" in line for line in output)
    assert any("one-off" in line for line in output)
    assert output[-1] == "Next: validate · show · run"


def test_studio_create_opens_automatic_specification_and_prepares_task(
    tmp_path, monkeypatch
):
    studio, workspace, output = _studio(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "zippergen.studio.shutil.which",
        lambda name: "/usr/bin/micro" if name == "micro" else None,
    )

    def fake_run(arguments, *, cwd, check):
        calls.append(arguments)
        target = Path(arguments[-1])
        guide = target.read_text(encoding="utf-8")
        assert "zippergen:specification-guide" in guide
        assert "Do not choose Python filenames" in guide
        target.write_text(
            "Create a reviewed answer workflow.\n"
            "Never return an unapproved draft.\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr("zippergen.studio.subprocess.run", fake_run)

    studio.execute("create --editor micro")

    assert calls == [[
        "/usr/bin/micro",
        str(workspace.specification_path),
    ]]
    assert workspace.specification() == (
        "Create a reviewed answer workflow.\n"
        "Never return an unapproved draft."
    )
    assert "zippergen:specification-guide" not in (
        workspace.specification_path.read_text()
    )
    assert workspace.current_task_path.exists()
    assert any("Editor closed" in line for line in output)
    assert any("specification.md" in line for line in output)


def test_studio_create_keeps_guide_when_no_requirements_are_written(
    tmp_path,
    monkeypatch,
):
    studio, workspace, _output = _studio(tmp_path)
    monkeypatch.setattr(
        "zippergen.studio.shutil.which",
        lambda name: "/usr/bin/micro" if name == "micro" else None,
    )
    monkeypatch.setattr(
        "zippergen.studio.subprocess.run",
        lambda arguments, **kwargs: subprocess.CompletedProcess(arguments, 0),
    )

    try:
        studio.execute("create --editor micro")
    except SystemExit as exc:
        assert "No application requirements were written" in str(exc)
    else:
        raise AssertionError("the untouched guide must not become a task")

    assert workspace.specification() is None
    assert "zippergen:specification-guide" in workspace.specification_path.read_text()
    assert workspace.current_request() is None


def test_studio_path_free_create_always_uses_canonical_specification_name(
    tmp_path, monkeypatch
):
    studio, workspace, _output = _studio(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "zippergen.studio.shutil.which",
        lambda name: "/usr/bin/micro" if name == "micro" else None,
    )

    def fake_run(arguments, *, cwd, check):
        calls.append(arguments)
        Path(arguments[-1]).write_text(
            "# Reviewed answer policy\n\n"
            "Never return an unapproved draft.\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr("zippergen.studio.subprocess.run", fake_run)

    studio.execute("create --edit --editor micro")

    assert workspace.specification_path == workspace.root / "specification.md"
    assert workspace.specification() == (
        "# Reviewed answer policy\n\nNever return an unapproved draft."
    )
    assert calls[0][-1] == str(workspace.specification_path)
    assert not (workspace.root / "prompts").exists()


def test_studio_spec_refine_reopens_one_pending_file(tmp_path, monkeypatch):
    studio, workspace, _output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_specification("Echo the input through Writer.")
    monkeypatch.setattr(
        "zippergen.studio.shutil.which",
        lambda name: "/usr/bin/micro" if name == "micro" else None,
    )

    calls = 0

    def fake_run(arguments, *, cwd, check):
        nonlocal calls
        target = Path(arguments[-1])
        assert target == workspace.pending_refinement_path
        if calls == 0:
            target.write_text("Add human approval before returning the result.\n")
        else:
            assert "Add human approval" in target.read_text()
            target.write_text(
                "Add human approval before returning the result.\n"
                "Use a yes/no decision.\n"
            )
        calls += 1
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr("zippergen.studio.subprocess.run", fake_run)

    studio.execute("spec refine --editor micro")
    first_baselines = list(
        workspace.requests_directory.glob("*-semantic-before.json")
    )
    studio.execute("spec refine --editor micro")

    request = workspace.current_request()
    assert request is not None
    assert request["kind"] == "refine"
    assert workspace.pending_refinement() == (
        "Add human approval before returning the result.\n"
        "Use a yes/no decision."
    )
    assert len(first_baselines) == 1
    assert list(workspace.requests_directory.glob("*-semantic-before.json")) == (
        first_baselines
    )


def test_studio_prompt_replacement_can_be_composed_in_editor(
    tmp_path, monkeypatch
):
    studio, workspace, _output = _studio(tmp_path)
    studio.execute("prompts add Keep retries bounded")
    monkeypatch.setattr(
        "zippergen.studio.shutil.which",
        lambda name: "/usr/bin/micro" if name == "micro" else None,
    )

    def fake_run(arguments, *, cwd, check):
        Path(arguments[-1]).write_text("Use exactly three retries.\n")
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr("zippergen.studio.subprocess.run", fake_run)

    studio.execute(
        "prompts replace P001 --edit prompts/retry_replacement.md "
        "--editor micro"
    )

    records = workspace.list_prompts()
    assert records[0]["active"] is False
    assert records[1]["replaces"] == "P001"
    assert records[1]["content"] == "Use exactly three retries."
    assert records[1]["file"] == "prompts/retry_replacement.md"


def test_studio_path_free_prompt_replacement_derives_name_and_preserves_history(
    tmp_path, monkeypatch
):
    studio, workspace, _output = _studio(tmp_path)
    studio.execute("prompts add Keep retries bounded")
    monkeypatch.setattr(
        "zippergen.studio.shutil.which",
        lambda name: "/usr/bin/micro" if name == "micro" else None,
    )

    def fake_run(arguments, *, cwd, check):
        draft = Path(arguments[-1])
        assert draft.read_text() == "Keep retries bounded\n"
        draft.write_text(
            "# Retry exactly three times\n\nThen return an explicit failure.\n"
        )
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr("zippergen.studio.subprocess.run", fake_run)

    studio.execute("prompts replace P001 --edit --editor micro")

    records = workspace.list_prompts()
    assert records[0]["active"] is False
    assert records[1]["id"] == "P002"
    assert records[1]["kind"] == "initial"
    assert records[1]["replaces"] == "P001"
    assert records[1]["file"] == "prompts/002-retry-exactly-three-times.md"
    assert list((workspace.root / ".zippergen" / "prompt-drafts").iterdir()) == []


def test_studio_edits_prompt_by_id_without_changing_identity(tmp_path, monkeypatch):
    studio, workspace, output = _studio(tmp_path)
    studio.execute("prompts add # Original title\nOriginal requirement")
    original = workspace.prompt("P001")
    monkeypatch.setattr(
        "zippergen.studio.shutil.which",
        lambda name: "/usr/bin/micro" if name == "micro" else None,
    )

    def fake_run(arguments, *, cwd, check):
        draft = Path(arguments[-1])
        assert ".zippergen/prompt-drafts" in str(draft)
        assert "Original requirement" in draft.read_text()
        draft.write_text("# Clearer title\n\nCorrected wording.\n")
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr("zippergen.studio.subprocess.run", fake_run)
    output.clear()

    studio.execute("prompts edit P001 --editor micro")

    records = workspace.list_prompts()
    assert len(records) == 1
    assert records[0]["id"] == "P001"
    assert records[0]["kind"] == "initial"
    assert records[0]["file"] == original["file"]
    assert records[0]["title"] == "Clearer title"
    assert records[0]["content"] == "# Clearer title\n\nCorrected wording."
    assert "Prompt updated" in output
    assert list((workspace.root / ".zippergen" / "prompt-drafts").iterdir()) == []


def test_studio_invalid_prompt_edit_leaves_registered_content_untouched(
    tmp_path, monkeypatch
):
    studio, workspace, _output = _studio(tmp_path)
    studio.execute("prompts add Original requirement")
    monkeypatch.setattr(
        "zippergen.studio.shutil.which",
        lambda name: "/usr/bin/micro" if name == "micro" else None,
    )

    def fake_run(arguments, *, cwd, check):
        Path(arguments[-1]).write_text("")
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr("zippergen.studio.subprocess.run", fake_run)

    try:
        studio.execute("prompts edit P001 --editor micro")
    except SystemExit as exc:
        assert "Prompt file is empty" in str(exc)
    else:
        raise AssertionError("an empty edit should be rejected")

    assert workspace.prompt("P001")["content"] == "Original requirement"
    assert len(list((workspace.root / ".zippergen" / "prompt-drafts").iterdir())) == 1


def test_studio_failed_create_editor_preserves_canonical_draft(
    tmp_path, monkeypatch
):
    studio, workspace, _output = _studio(tmp_path)
    monkeypatch.setattr(
        "zippergen.studio.shutil.which",
        lambda name: "/usr/bin/micro" if name == "micro" else None,
    )

    def fake_run(arguments, *, cwd, check):
        Path(arguments[-1]).write_text("# Important draft\n\nDo not lose this.\n")
        return subprocess.CompletedProcess(arguments, 3)

    monkeypatch.setattr("zippergen.studio.subprocess.run", fake_run)

    try:
        studio.execute("create --editor micro")
    except SystemExit as exc:
        assert "Editor exited with status 3" in str(exc)
    else:
        raise AssertionError("failed editor should stop creation")

    assert "Do not lose this" in workspace.specification_path.read_text()
    assert workspace.current_request() is None


def test_studio_editor_errors_are_safe_and_actionable(tmp_path, monkeypatch):
    studio, workspace, _output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    monkeypatch.setattr("zippergen.studio.shutil.which", lambda name: None)

    for command, expected in (
        ("editor set missing", "Editor executable was not found: missing"),
        (
            "create --editor missing",
            "Editor executable was not found: missing",
        ),
        ("edit workflow --editor missing", "Editor executable was not found: missing"),
    ):
        try:
            studio.execute(command)
        except SystemExit as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"{command!r} should fail")
    assert not (workspace.root.parent / "outside.md").exists()


def test_studio_does_not_prepare_task_after_failed_editor(
    tmp_path, monkeypatch
):
    studio, workspace, _output = _studio(tmp_path)
    monkeypatch.setattr(
        "zippergen.studio.shutil.which",
        lambda name: "/usr/bin/micro" if name == "micro" else None,
    )

    def fake_run(arguments, *, cwd, check):
        return subprocess.CompletedProcess(arguments, 3)

    monkeypatch.setattr("zippergen.studio.subprocess.run", fake_run)

    try:
        studio.execute("create --editor micro")
    except SystemExit as exc:
        assert "Editor exited with status 3" in str(exc)
    else:
        raise AssertionError("failed editor should stop creation")

    assert workspace.current_request() is None


def test_studio_refuses_manual_filename_for_create_editor(tmp_path, monkeypatch):
    studio, _workspace, _output = _studio(tmp_path)
    monkeypatch.setattr(
        "zippergen.studio.shutil.which",
        lambda name: "/usr/bin/micro" if name == "micro" else None,
    )

    try:
        studio.execute("create --edit prompts/custom.md --editor micro")
    except SystemExit as exc:
        assert "only used when create opens the specification editor" in str(exc)
    else:
        raise AssertionError("create should own the specification filename")


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
    assert "project reset fresh [--yes]" in output[-1]
    assert "project reset state [--yes]" in output[-1]
    assert "legacy prompt-ledger migration/compatibility" in output[-1]
    assert "task show|path|history" in output[-1]
    assert "assistant" in output[-1]
    assert "editor [show|set CMD|reset]" in output[-1]
    assert "edit [workflow|file PATH]" in output[-1]
    assert "create --file PATH" in output[-1]
    assert "spec edit" in output[-1]
    assert "spec refine" in output[-1]
    assert "refine --file PATH" in output[-1]
    assert "providers set local [URL]" in output[-1]
    assert "providers check local" in output[-1]
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


def test_studio_interactive_commands_have_a_clear_output_boundary(tmp_path):
    studio, _workspace, output = _studio(
        tmp_path,
        responses=["current", "exit"],
    )

    assert studio.run() == 0

    boundaries = [line for line in output if line.startswith("── Output:")]
    assert len(boundaries) == 1
    assert boundaries[0].startswith("── Output: current ")
    assert output[output.index(boundaries[0]) - 1] == ""
    assert all("exit" not in line for line in boundaries)


def test_studio_boundaries_hide_arguments_and_skip_empty_or_exit(tmp_path):
    studio, _workspace, output = _studio(tmp_path)

    studio.execute(
        "create Never expose SECRET_SENTINEL in a boundary",
        show_boundary=True,
    )

    boundary = next(line for line in output if line.startswith("── Output:"))
    assert boundary.startswith("── Output: create ")
    assert "SECRET_SENTINEL" not in boundary

    output.clear()
    assert studio.execute("", show_boundary=True) is True
    assert studio.execute("exit", show_boundary=True) is False
    assert output == []


def test_studio_parse_errors_receive_an_input_boundary(tmp_path):
    studio, _workspace, output = _studio(tmp_path)

    studio.execute('create "unterminated', show_boundary=True)

    assert output[0] == ""
    assert output[1].startswith("── Output: input ")
    assert output[2].startswith("✗ Could not parse command:")


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


def test_studio_project_reset_can_be_cancelled_without_changes(tmp_path):
    studio, workspace, output = _studio(tmp_path, responses=["n"])
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    studio.create_request("Create a review workflow.")
    output.clear()

    studio.execute("project reset state")

    assert workspace.current_workflow == "workflow.py:sample"
    assert workspace.current_request() is not None
    assert workspace.current_task_path.exists()
    assert not workspace.resets_directory.exists()
    assert output[0] == "Project reset preview"
    assert output[-1] == "⚠ Project reset cancelled; nothing was changed."


def test_studio_project_reset_interrupt_is_a_clean_cancellation(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)

    def interrupt(_prompt):
        raise KeyboardInterrupt

    studio.input = interrupt

    studio.execute("project reset state")

    assert workspace.current_workflow == "workflow.py:sample"
    assert output[-1] == "⚠ Project reset cancelled; nothing was changed."


def test_studio_project_reset_state_backs_up_only_private_context(tmp_path):
    studio, workspace, output = _studio(tmp_path, responses=["perhaps", "yes"])
    studio.execute("project init Tutorial")
    studio.create_request("Create a review workflow.")
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_secrets({"OPENAI_API_KEY": "private"})
    output.clear()

    studio.execute("project reset state")

    assert "⚠ Please enter 'y' or 'n'." in output
    assert workspace.current_workflow is None
    assert workspace.current_request() is None
    assert workspace.load_secrets() == {}
    assert not workspace.current_task_path.exists()
    assert workspace.manifest_path.exists()
    assert (workspace.root / "workflow.py").exists()
    assert workspace.specification() == "Create a review workflow."
    backups = list(workspace.resets_directory.iterdir())
    assert len(backups) == 1
    assert (backups[0] / "workspace" / "workspace.json").exists()
    assert (backups[0] / "project-local" / "current-task.md").exists()
    assert any(line == "Project reset" for line in output)
    assert any("✓ complete" in line for line in output)
    assert any("✓ " + str(backups[0]) in line for line in output)
    assert studio._prompt() == "zippergen [no workflow]> "


def test_studio_plain_project_reset_makes_both_scopes_explicit(tmp_path):
    studio, workspace, output = _studio(tmp_path, responses=["3"])
    studio.execute("project init Tutorial")
    studio.create_request("Create a review workflow.")
    output.clear()

    studio.execute("project reset")

    assert output[0] == "Choose reset scope"
    assert any("Fresh design cycle" in line for line in output)
    assert any("Studio state only" in line for line in output)
    assert output[-1] == "⚠ Project reset cancelled; nothing was changed."
    assert workspace.manifest_path.exists()
    assert workspace.specification() == "Create a review workflow."


def test_studio_project_reset_rejects_ambiguous_noninteractive_form(tmp_path):
    studio, workspace, _output = _studio(tmp_path)
    studio.execute("project init Tutorial")

    try:
        studio.execute("project reset --yes")
    except SystemExit as exc:
        assert "project reset fresh" in str(exc)
        assert "project reset state" in str(exc)
    else:
        raise AssertionError("a noninteractive reset must name its scope")

    assert workspace.manifest_path.exists()


def test_studio_project_reset_fresh_archives_design_then_init_is_new(tmp_path):
    studio, workspace, output = _studio(tmp_path, responses=["1", "yes"])
    studio.execute("project init Tutorial")
    workspace.add_prompt(kind="initial", content="Legacy requirement.")
    studio.create_request("Create a review workflow.")
    output.clear()

    studio.execute("project reset")

    assert not workspace.manifest_path.exists()
    assert not workspace.specification_path.exists()
    assert not (workspace.root / "prompts").exists()
    assert (workspace.root / "workflow.py").exists()
    assert workspace.current_request() is None
    backups = list(workspace.resets_directory.iterdir())
    assert len(backups) == 1
    archived = backups[0] / "project-visible"
    assert (archived / "zippergen.toml").exists()
    assert (archived / "specification.md").exists()
    assert (archived / "prompts" / "index.toml").exists()
    assert any("fresh design cycle" in line for line in output)
    assert any("project init · create" in line for line in output)

    output.clear()
    studio.execute('project init "Tutorial again"')
    assert any("Project manifest created" in line for line in output)
    assert workspace.project_manifest()["name"] == "Tutorial again"


def test_studio_project_reset_state_yes_is_noninteractive_and_idempotent(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)

    studio.execute("project reset state --yes")

    assert workspace.current_workflow is None
    assert any("✓ complete" in line for line in output)

    output.clear()
    studio.execute("project reset state --yes")

    assert output == [
        "⚠ Private Studio state is already empty. The manifest, "
        "specification, source, tests, and Git were not changed."
    ]


def test_studio_project_reset_handles_a_missing_project_directory(tmp_path):
    root = tmp_path / "deleted-project"
    workspace = Workspace(root, home=tmp_path / "home")
    workspace.update(current_workflow="workflows/deleted.py:deleted")
    output: list[str] = []
    studio = Studio(workspace, output_func=output.append)

    studio.execute("project reset state --yes")

    assert workspace.current_workflow is None
    assert any(str(root) in line and "missing" in line for line in output)
    assert any(
        "exit and recreate the project directory" in line for line in output
    )


def test_studio_spec_commands_use_automatic_paths_and_append_one_pending_change(
    tmp_path,
):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    studio.create_request("Echo the request through Writer.")
    output.clear()

    studio.execute("spec path")
    assert output == [str(workspace.root / "specification.md")]

    output.clear()
    studio.execute("spec refine Add bounded retries")
    studio.execute("refine Return an explicit failure after exhaustion")

    assert workspace.pending_refinement_path == (
        workspace.root / ".zippergen" / "pending-refinement.md"
    )
    assert workspace.pending_refinement() == (
        "Add bounded retries\n\nReturn an explicit failure after exhaustion"
    )
    assert len(
        list(workspace.requests_directory.glob("*-semantic-before.json"))
    ) == 1
    assert any("short alias for 'spec refine'" in line for line in output)
    assert not (workspace.root / "prompts").exists()

    output.clear()
    studio.execute("spec pending")
    assert output[0] == "Pending refinement"
    assert any("Add bounded retries" in line for line in output)


def test_studio_reconcile_requires_integrated_spec_and_keeps_private_history(
    tmp_path,
):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    studio.create_request("Echo the request through Writer.")
    studio.execute("spec refine Add a human approval before returning")

    try:
        studio.execute("spec reconcile --yes")
    except SystemExit as exc:
        assert "canonical specification has not changed" in str(exc)
    else:
        raise AssertionError("an unintegrated refinement must not be reconciled")

    workspace.save_specification(
        "Echo the request through Writer and require human approval before return."
    )
    output.clear()
    studio.execute("spec reconcile --yes")

    assert workspace.pending_refinement() is None
    assert workspace.current_request() is None
    assert not workspace.current_task_path.exists()
    assert workspace.list_spec_history()[0]["status"] == "reconciled"
    assert workspace.list_requests()[0]["status"] == "reconciled"
    assert any("✓ reconciled" in line for line in output)
    assert any("✓ cleared" in line for line in output)
    assert any(
        "Canonical" in line and "no automatic merge" in line for line in output
    )


def test_studio_spec_discard_is_explicit_and_recoverable(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    studio.create_request("Echo the request through Writer.")
    studio.execute("spec refine Remove Writer")
    output.clear()

    studio.execute("spec discard --yes")

    assert workspace.pending_refinement() is None
    assert workspace.list_spec_history()[0]["status"] == "discarded"
    assert any("⚠ discarded" in line for line in output)


def test_studio_spec_show_migrates_legacy_prompt_ledger_once(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    first = workspace.add_prompt(kind="initial", content="Create a reviewer.")
    second = workspace.add_prompt(
        kind="refinement",
        content="Add bounded retries.",
    )
    output.clear()

    studio.execute("spec show")

    assert "Create a reviewer." in workspace.specification()
    assert "Add bounded retries." in workspace.specification()
    assert (workspace.root / str(first["file"])).exists()
    assert (workspace.root / str(second["file"])).exists()
    assert any("Migrated the former active prompt ledger" in line for line in output)

    output.clear()
    studio.execute("spec show")
    assert all(
        not line.startswith("• Migrated the former active prompt ledger")
        for line in output
    )


def test_studio_legacy_prompt_ledger_is_read_only_after_migration(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    workspace.add_prompt(kind="initial", content="Create a reviewer.")
    studio.execute("spec show")

    try:
        studio.execute("prompts add This must not become hidden design intent")
    except SystemExit as exc:
        assert "canonical specification" in str(exc)
        assert "spec refine" in str(exc)
    else:
        raise AssertionError("a migrated legacy ledger must be read-only")

    assert len(workspace.list_prompts()) == 1
    output.clear()
    studio.execute("prompts inspect P001")
    assert "Create a reviewer." in output


def test_studio_prompt_table_inspection_path_archive_and_restore(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    studio.execute("prompts add Foundation: keep source visible")
    studio.execute("prompts add Add bounded retries")

    records = workspace.list_prompts()
    assert [record["kind"] for record in records] == ["initial", "refinement"]

    output.clear()
    studio.execute("prompts")
    assert output[0] == "Prompt summary"
    assert "Prompt ledger" in output
    assert any(
        "ID" in line and "Kind" in line and "Status" in line and "Title" in line
        for line in output
    )
    assert any("P001" in line and "initial" in line for line in output)
    assert any("P002" in line and "refinement" in line for line in output)

    output.clear()
    studio.execute("prompts inspect P002")
    assert output[0] == "Prompt P002"
    assert "Requirement" in output
    assert any("Position" in line and "2" in line for line in output)

    output.clear()
    studio.execute("prompts path P002")
    assert output == [str(workspace.root / str(records[1]["file"]))]

    studio.execute("prompts archive P002")
    assert workspace.prompt("P002")["active"] is False
    assert "Add bounded retries" not in workspace.prompt_context()

    studio.execute("prompts restore P002")
    assert workspace.prompt("P002")["active"] is True
    assert "Add bounded retries" in workspace.prompt_context()


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


def test_studio_handoff_contains_canonical_spec_and_pending_refinement(tmp_path):
    studio, workspace, _output = _studio(tmp_path)
    studio.create_request("Create a concise answer workflow.")
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    studio.refine_request("Add an explicit reviewer.")

    refine_records = list(workspace.requests_directory.glob("*-refine.json"))
    assert len(refine_records) == 1
    metadata = json.loads(refine_records[0].read_text())
    brief = refine_records[0].with_suffix(".md").read_text()

    assert metadata["prompt_id"] is None
    assert metadata["specification_fingerprint"] == (
        workspace.specification_fingerprint()
    )
    assert brief.index("# Canonical workflow specification") < brief.index(
        "# Pending refinement"
    )
    assert "Create a concise answer workflow." in brief
    assert "Add an explicit reviewer." in brief
    assert "preserve every unaffected requirement" in brief


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
    assert any("Specification" in line and "ready" in line for line in output)
    assert any("Refinement" in line and "none" in line for line in output)
    assert any("Editor" in line and "automatic" in line for line in output)
    assert any("Selected" in line and "workflow.py:sample" in line for line in output)
    assert any("Name" in line and "sample" in line for line in output)
    assert any("Participants" in line and "User, Writer" in line for line in output)
    assert any(
        "LLM-active participants" in line and "1 — Writer" in line
        for line in output
    )
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
    assert any(
        "LLM-active participants" in line and "0 — none" in line
        for line in output
    )
    assert any("Connectors" in line and "none" in line for line in output)
    assert any("Validation" in line and "⚠ not available" in line for line in output)
    assert any("Assignments" in line and "none" in line for line in output)
    assert any("Providers" in line and "none" in line for line in output)


def test_studio_configures_api_and_local_providers_without_displaying_secrets(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(
        tmp_path,
        secret_responses=["super-secret-key"],
    )
    requests: list[tuple[str, float]] = []

    class ModelsResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, limit=-1):
            assert limit == 1_048_577
            return json.dumps(
                {
                    "object": "list",
                    "data": [
                        {"id": "qwen2.5:7b"},
                        {"id": "llama3.2:3b"},
                    ],
                }
            ).encode("utf-8")

    def fake_urlopen(req, *, timeout):
        requests.append((req.full_url, timeout))
        return ModelsResponse()

    monkeypatch.setattr("zippergen.studio.request.urlopen", fake_urlopen)

    studio.execute("providers set openai")
    studio.execute("providers set local http://localhost:1234/v1")
    studio.execute("providers check local")
    studio.execute("providers")

    assert workspace.load_secrets() == {"OPENAI_API_KEY": "super-secret-key"}
    assert workspace.provider_profiles()["local"]["base_url"] == (
        "http://localhost:1234/v1"
    )
    assert workspace.provider_profiles()["local"]["check_status"] == "reachable"
    assert workspace.provider_profiles()["local"]["model_count"] == "2"
    assert requests == [
        ("http://localhost:1234/v1/models", 3.0),
        ("http://localhost:1234/v1/models", 3.0),
    ]
    assert workspace.secrets_path.stat().st_mode & 0o077 == 0
    assert all("super-secret-key" not in line for line in output)
    assert any("openai: ready" in line for line in output)
    assert any(
        "local: ready; endpoint http://localhost:1234/v1; 2 models; checked"
        in line
        for line in output
    )

    studio.execute("providers reset openai")
    assert "OPENAI_API_KEY" not in workspace.load_secrets()
    assert "openai" not in workspace.provider_profiles()


def test_studio_does_not_replace_local_endpoint_when_check_fails(
    tmp_path,
    monkeypatch,
):
    studio, workspace, _output = _studio(tmp_path)
    original = {
        "kind": "local",
        "base_url": "http://localhost:11434/v1",
        "check_status": "reachable",
        "checked_at": "2026-07-23T10:00:00+0200",
        "model_count": "1",
    }
    workspace.save_provider_profile("local", original)

    def fail_urlopen(req, *, timeout):
        raise URLError("connection refused")

    monkeypatch.setattr("zippergen.studio.request.urlopen", fail_urlopen)

    with pytest.raises(SystemExit, match="endpoint was not saved"):
        studio.execute("providers set local http://localhost:9999/v1")

    assert workspace.provider_profiles()["local"] == original


def test_studio_records_failed_local_provider_recheck(tmp_path, monkeypatch):
    studio, workspace, output = _studio(tmp_path)
    workspace.save_provider_profile(
        "local",
        {
            "kind": "local",
            "base_url": "http://localhost:11434/v1",
            "check_status": "reachable",
            "checked_at": "2026-07-23T10:00:00+0200",
            "model_count": "1",
        },
    )

    def fail_urlopen(req, *, timeout):
        raise URLError("connection refused")

    monkeypatch.setattr("zippergen.studio.request.urlopen", fail_urlopen)

    with pytest.raises(SystemExit, match="SSH tunnel"):
        studio.execute("providers check local")

    profile = workspace.provider_profiles()["local"]
    assert profile["check_status"] == "unreachable"
    assert profile["check_error"] == "connection refused"
    output.clear()
    studio.execute("providers")
    assert any(
        "local: endpoint http://localhost:11434/v1; unreachable" in line
        and "connection refused" in line
        for line in output
    )


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


def test_studio_offers_private_provider_key_reuse_for_first_deployment(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path, responses=[""])
    (workspace.root / "workflow.py").write_text(DEPLOYMENT_SOURCE)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_model_profile(
        "workflow.py:sample",
        default="openai:gpt-4o-mini",
        lifelines={},
    )
    workspace.save_secrets({"OPENAI_API_KEY": "development-secret"})
    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "deployment-home"))
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "zippergen.serve.main",
        lambda arguments: calls.append(arguments) or 0,
    )

    studio.deploy_workflow(["sample-reuse", "--no-start"])

    assert calls[0][-2:] == [
        "--set",
        "openai_api_key=development-secret",
    ]
    assert any(
        "Available" in line and "OPENAI_API_KEY" in line for line in output
    )
    assert any(
        line.startswith("✓ Reusing 1 configured credential") for line in output
    )
    assert all("development-secret" not in line for line in output)


def test_studio_can_decline_provider_key_reuse_for_deployment(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path, responses=["n"])
    (workspace.root / "workflow.py").write_text(DEPLOYMENT_SOURCE)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_model_profile(
        "workflow.py:sample",
        default="openai:gpt-4o-mini",
        lifelines={},
    )
    workspace.save_secrets({"OPENAI_API_KEY": "development-secret"})
    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "deployment-home"))
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "zippergen.serve.main",
        lambda arguments: calls.append(arguments) or 0,
    )

    studio.deploy_workflow(["sample-separate", "--no-start"])

    assert "--set" not in calls[0]
    assert any(
        "Credential reuse declined" in line for line in output
    )
    assert all("development-secret" not in line for line in output)


def test_studio_redeploy_keeps_existing_deployment_provider_key(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path)
    (workspace.root / "workflow.py").write_text(DEPLOYMENT_SOURCE)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_model_profile(
        "workflow.py:sample",
        default="openai:gpt-4o-mini",
        lifelines={},
    )
    profile_path = tmp_path / "deployment-home" / "sample-existing.json"
    profile_path.parent.mkdir()
    profile_path.write_text("{}")
    monkeypatch.setattr(
        "zippergen.serve._deployment_profile_path",
        lambda name: profile_path,
    )
    monkeypatch.setattr(
        "zippergen.serve._load_deployment_profile",
        lambda name: {"name": name, "secrets_file": "private.json"},
    )
    monkeypatch.setattr(
        "zippergen.serve._load_deployment_secrets",
        lambda profile: {"OPENAI_API_KEY": "existing-deployment-secret"},
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "zippergen.serve.main",
        lambda arguments: calls.append(arguments) or 0,
    )

    studio.deploy_workflow(["sample-existing", "--no-start"])

    assert calls[0][-2:] == [
        "--set",
        "openai_api_key=existing-deployment-secret",
    ]
    assert any(
        line.startswith("✓ Keeping 1 existing deployment credential")
        for line in output
    )
    assert all("existing-deployment-secret" not in line for line in output)


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


def test_studio_models_verifies_mistral_model_before_saving(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_provider_profile(
        "mistral",
        {"kind": "api", "key_env": "MISTRAL_API_KEY"},
    )
    workspace.save_secrets({"MISTRAL_API_KEY": "private-mistral-key"})
    requests = []

    class ModelResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, limit=-1):
            assert limit == 1_048_577
            return b'{"id":"mistral-small-latest","object":"model"}'

    def fake_urlopen(req, *, timeout):
        requests.append(req)
        assert timeout == 3.0
        return ModelResponse()

    monkeypatch.setattr("zippergen.studio.request.urlopen", fake_urlopen)

    studio.execute("models set Writer mistral:mistral-small-latest")

    assert workspace.model_profile("workflow.py:sample")["lifelines"] == {
        "Writer": "mistral:mistral-small-latest"
    }
    assert requests[0].full_url == (
        "https://api.mistral.ai/v1/models/mistral-small-latest"
    )
    assert requests[0].get_header("Authorization") == (
        "Bearer private-mistral-key"
    )
    assert all("private-mistral-key" not in line for line in output)
    assert any(
        line.startswith("✓ Writer:")
        and "is available with the configured mistral API key" in line
        for line in output
    )


def test_studio_models_rejects_unavailable_mistral_model_without_saving(
    tmp_path,
    monkeypatch,
):
    studio, workspace, _output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_provider_profile(
        "mistral",
        {"kind": "api", "key_env": "MISTRAL_API_KEY"},
    )
    workspace.save_secrets({"MISTRAL_API_KEY": "private-mistral-key"})

    def fake_urlopen(req, *, timeout):
        raise HTTPError(
            req.full_url,
            404,
            "Not Found",
            {},
            BytesIO(b'{"message":"model not found"}'),
        )

    monkeypatch.setattr("zippergen.studio.request.urlopen", fake_urlopen)

    with pytest.raises(SystemExit, match="not available.*routing was not changed"):
        studio.execute("models set Writer mistral:mistral-smol-latest")

    assert workspace.model_profile("workflow.py:sample")["lifelines"] == {}


def test_studio_models_saves_an_explicit_unchecked_route_when_provider_is_offline(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_provider_profile(
        "mistral",
        {"kind": "api", "key_env": "MISTRAL_API_KEY"},
    )
    workspace.save_secrets({"MISTRAL_API_KEY": "private-mistral-key"})

    def fake_urlopen(req, *, timeout):
        raise URLError("network is offline")

    monkeypatch.setattr("zippergen.studio.request.urlopen", fake_urlopen)

    studio.execute("models set Writer mistral:mistral-small-latest")

    assert workspace.model_profile("workflow.py:sample")["lifelines"] == {
        "Writer": "mistral:mistral-small-latest"
    }
    assert any(
        line.startswith("⚠ Writer:")
        and "availability could not be checked" in line
        and "network is offline" in line
        for line in output
    )


def test_studio_models_checks_local_model_identifiers(tmp_path, monkeypatch):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_provider_profile(
        "local",
        {
            "kind": "local",
            "base_url": "http://localhost:11434/v1",
        },
    )

    class ModelsResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, limit=-1):
            assert limit == 1_048_577
            return b'{"object":"list","data":[{"id":"qwen2.5:7b"}]}'

    monkeypatch.setattr(
        "zippergen.studio.request.urlopen",
        lambda req, *, timeout: ModelsResponse(),
    )

    studio.execute("models set Writer local:qwen2.5:7b")

    assert any(
        line.startswith("✓ Writer:")
        and "is available from the local provider" in line
        for line in output
    )

    with pytest.raises(SystemExit, match="Available models: qwen2.5:7b"):
        studio.execute("models set Writer local:missing")
    assert workspace.model_profile("workflow.py:sample")["lifelines"] == {
        "Writer": "local:qwen2.5:7b"
    }


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
