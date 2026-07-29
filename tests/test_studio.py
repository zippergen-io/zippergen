import json
import subprocess
import time
from io import BytesIO, StringIO
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from zippergen.studio import Studio, StudioCompleter
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

GOOGLE_DESKTOP_CLIENT = json.dumps(
    {
        "installed": {
            "client_id": "example.apps.googleusercontent.com",
            "client_secret": "private-client-secret",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }
)

TWO_LLM_PARTICIPANT_SOURCE = """
from zippergen import Lifeline, llm, workflow

User = Lifeline("User")
Writer = Lifeline("Writer")
Reviewer = Lifeline("Reviewer")

@llm(
    system="Process the value.",
    user="{value}",
    parse="text",
    outputs=(("result", str),),
)
def process(value: str) -> None: ...

@workflow
def sample(value: str @ User) -> str:
    User(value) >> Writer(value)
    Writer: draft = process(value)
    Writer(draft) >> Reviewer(draft)
    Reviewer: result = process(draft)
    Reviewer(result) >> User(result)
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

CONNECTOR_SOURCE = WORKFLOW_SOURCE + """

from zippergen import ConnectorRequirement

zippergen_connectors = (
    ConnectorRequirement(
        name="human-approval",
        kind="telegram",
        participant="Writer",
        capabilities=("notify", "approve"),
        required=True,
    ),
)
"""

HUMAN_SOURCE = """
from zippergen import Lifeline, human, workflow

User = Lifeline("User")
Human = Lifeline("Human")

@human(
    kind="select",
    instruction="Choose a time.",
    prefill="Thursday, 11 AM\\nFriday, 10 AM",
    outputs=["choice: str"],
)
def choose_time() -> None: ...

@workflow
def sample(request: str @ User) -> str:
    User(request) >> Human(request)
    Human: choice = choose_time()
    Human(choice) >> User(choice)
    return choice @ User
"""

GOOGLE_SHEETS_SOURCE = """
from zippergen import ConnectorRequirement, Lifeline, effect, workflow

User = Lifeline("User")
Records = Lifeline("Records")

@effect(connector="call-records", operation="upsert-json-row")
def save_record(record: str) -> str:
    return "created"

zippergen_connectors = (
    ConnectorRequirement(
        name="call-records",
        kind="google-sheets",
        participant="Records",
        capabilities=("read-rows", "upsert-row"),
        access="read-write",
    ),
)

@workflow
def sample(record: str @ User) -> str:
    User(record) >> Records(record)
    Records: status = save_record(record)
    Records(status) >> User(status)
    return status @ User
"""

GMAIL_AND_SHEETS_SOURCE = """
from zippergen import ConnectorRequirement, Lifeline, effect, workflow

User = Lifeline("User")
Mailbox = Lifeline("Mailbox")
Records = Lifeline("Records")

@effect(connector="mailbox", operation="read-messages")
def read_mail() -> str:
    return "message"

@effect(connector="records", operation="upsert-row")
def save_record(message: str) -> str:
    return "created"

zippergen_connectors = (
    ConnectorRequirement(
        name="mailbox",
        kind="gmail",
        participant="Mailbox",
        capabilities=("read-messages",),
        access="read-only",
    ),
    ConnectorRequirement(
        name="records",
        kind="google-sheets",
        participant="Records",
        capabilities=("upsert-row",),
        access="write",
    ),
)

@workflow
def sample() -> str:
    Mailbox: message = read_mail()
    Mailbox(message) >> Records(message)
    Records: status = save_record(message)
    Records(status) >> User(status)
    return status @ User
"""

TWO_SHEETS_SOURCE = """
from zippergen import ConnectorRequirement, Lifeline, effect, workflow

User = Lifeline("User")
Archivist = Lifeline("Archivist")

@effect(connector="source-catalog", operation="read-json-rows")
def read_catalog() -> str:
    return "items"

@effect(connector="target-dashboard", operation="upsert-json-row")
def update_dashboard(items: str) -> str:
    return "updated"

zippergen_connectors = (
    ConnectorRequirement(
        name="source-catalog",
        kind="google-sheets",
        participant="Archivist",
        capabilities=("read-rows",),
        access="read-only",
    ),
    ConnectorRequirement(
        name="target-dashboard",
        kind="google-sheets",
        participant="Archivist",
        capabilities=("upsert-row",),
        access="write",
    ),
)

@workflow
def sample() -> str:
    Archivist: items = read_catalog()
    Archivist: status = update_dashboard(items)
    Archivist(status) >> User(status)
    return status @ User
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

    assert _completions(studio, "wo") == ["workflow"]
    assert {"refine", "status"}.issubset(_completions(studio, "workflow "))
    assert _completions(studio, "workflow rev") == ["review"]
    assert _completions(studio, "workflow select wor") == [
        "workflow.py:sample"
    ]
    assert "reset" in _completions(studio, "deploy logs ")
    assert _completions(studio, "workflow show agent W") == ["Writer"]
    assert _completions(studio, "model assign W") == [
        "Writer",
        "Writer.echo",
    ]
    assert "check" in _completions(studio, "model config ch")
    assert _completions(studio, "model config check m") == ["mock"]
    assert "all" in _completions(studio, "model config check ")
    assert _completions(studio, "model assignments ") == ["check"]
    assert _completions(
        studio, "model provider configure a"
    ) == ["anthropic"]
    assert _completions(studio, "model inh") == ["inherit"]
    assert _completions(studio, "model inherit W") == [
        "Writer",
        "Writer.echo",
    ]
    assert _completions(studio, "settings set l") == ["learning"]
    assert _completions(studio, "settings set learning ") == ["on", "off"]
    assert "assistant" in _completions(studio, "settings reset ")
    assert _completions(studio, "project ren") == ["rename"]
    assert _completions(studio, "stu") == ["studio"]
    assert _completions(studio, "studio res") == ["restart"]
    assert _completions(studio, "sto") == []
    assert _completions(studio, "depl") == ["deploy"]
    assert "show" in _completions(studio, "deploy ")
    assert {"inspect", "tasks", "approve", "trace"}.issubset(
        _completions(studio, "run ")
    )
    assert "trace" in _completions(studio, "deploy ")
    assert "storage" in _completions(studio, "deploy ")
    assert _completions(studio, "deploy storage ") == ["compact"]
    assert _completions(
        studio, "deploy storage compact "
    ) == ["--yes"]
    assert "remove" in _completions(studio, "deploy ")
    assert _completions(studio, "run inspect W") == ["Writer"]
    assert "--watch" in _completions(studio, "run inspect ")
    assert _completions(studio, "run inspect Writer ") == ["--watch"]
    assert "--watch" in _completions(studio, "deploy inspect ")
    assert _completions(
        studio, "deploy inspect review-demo Reviewer "
    ) == ["--watch"]
    assert _completions(studio, "workflow create --file req") == [
        "requirements.md"
    ]
    assert _completions(
        studio, "workflow create --file 'notes f"
    ) == ["'notes folder/'"]
    assert _completions(
        studio, "workflow create --file 'notes folder/'"
    ) == [
        "'notes folder/spec.md'"
    ]
    assert _completions(studio, "workflow impo") == ["import"]


def test_studio_imports_external_workflow_dependencies_and_resources(
    tmp_path,
):
    studio, workspace, output = _studio(tmp_path)
    external = tmp_path / "external"
    (external / "flows").mkdir(parents=True)
    (external / "prompts").mkdir()
    (external / "pyproject.toml").write_text("[project]\nname='source'\n")
    (external / "flows" / "__init__.py").write_text("")
    (external / "flows" / "helper.py").write_text(
        "from zippergen import pure\n\n"
        "@pure\n"
        "def normalize(value: str) -> str:\n"
        "    return value.strip()\n"
    )
    (external / "prompts" / "instructions.md").write_text("Keep it short.\n")
    source = external / "flows" / "imported.py"
    source.write_text(
        "from zippergen import DeploymentSpec, Lifeline, workflow\n"
        "from .helper import normalize\n\n"
        "User = Lifeline('User')\n\n"
        "zippergen_deployment = DeploymentSpec(\n"
        "    name='imported',\n"
        "    files=('prompts/instructions.md',),\n"
        ")\n\n"
        "@workflow\n"
        "def imported(request: str @ User) -> str:\n"
        "    User: result = normalize(request)\n"
        "    return result @ User\n"
    )

    studio.execute(f"workflow import {source}")

    assert (workspace.root / "flows" / "imported.py").is_file()
    assert (workspace.root / "flows" / "__init__.py").is_file()
    assert (workspace.root / "flows" / "helper.py").is_file()
    assert (workspace.root / "prompts" / "instructions.md").is_file()
    assert workspace.current_workflow == "flows/imported.py:imported"
    assert any("Workflow imported" in line for line in output)
    assert any("workflow validate" in line for line in output)


def test_studio_import_refuses_to_overwrite_different_project_file(tmp_path):
    studio, workspace, _output = _studio(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    source = external / "imported.py"
    source.write_text(WORKFLOW_SOURCE)
    (workspace.root / "imported.py").write_text("# local work\n")

    with pytest.raises(SystemExit, match="would overwrite different"):
        studio.execute(f"workflow import {source}")

    assert (workspace.root / "imported.py").read_text() == "# local work\n"


def test_studio_import_selects_requested_entry_from_multi_workflow_file(
    tmp_path,
):
    studio, workspace, _output = _studio(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    source = external / "choices.py"
    source.write_text(
        "from zippergen import Lifeline, workflow\n"
        "User = Lifeline('User')\n\n"
        "@workflow\n"
        "def first(value: str @ User) -> str:\n"
        "    return value @ User\n\n"
        "@workflow\n"
        "def second(value: str @ User) -> str:\n"
        "    return value @ User\n"
    )

    studio.execute(f"workflow import {source}:second")

    assert workspace.current_workflow == "choices.py:second"


def test_studio_imports_from_nested_project_checkout(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    nested_checkout = workspace.root / "zippergen"
    examples = nested_checkout / "examples"
    examples.mkdir(parents=True)
    (nested_checkout / "pyproject.toml").write_text(
        "[project]\nname='zippergen-source'\n"
    )
    source = examples / "call_intake.py"
    source.write_text(
        "from zippergen import Lifeline, workflow\n"
        "User = Lifeline('User')\n\n"
        "@workflow\n"
        "def call_intake(value: str @ User) -> str:\n"
        "    return value @ User\n"
    )

    studio.execute(
        "workflow import zippergen/examples/call_intake.py:call_intake"
    )

    imported = workspace.root / "examples" / "call_intake.py"
    assert imported.is_file()
    assert imported.read_bytes() == source.read_bytes()
    assert workspace.current_workflow == "examples/call_intake.py:call_intake"
    assert any("Workflow imported" in line for line in output)


def test_studio_inspects_current_run_with_a_local_program_pointer(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    from zippergen.locator import resolve_path, statement_node_paths
    from zippergen.projection import project
    from zippergen.serve import load_workflow_spec
    from zippergen.store import open_store, write_execution_state
    from zippergen.syntax import ActStmt, _ordered_workflow_lifelines

    workflow, _module = load_workflow_spec(
        workspace.absolute_spec("workflow.py:sample")
    )
    record = workspace.new_run(
        workflow_spec="workflow.py:sample",
        workflow_name="sample",
        fingerprint="test",
        inputs={"value": "hello"},
        llm="mock",
    )
    workspace.update_run(str(record["run_id"]), status="running")
    local = project(
        workflow,
        next(
            lifeline
            for lifeline in _ordered_workflow_lifelines(workflow)
            if lifeline.name == "Writer"
        ),
    )
    action_path = next(
        path
        for path in statement_node_paths(local).values()
        if isinstance(resolve_path(local, path), ActStmt)
    )
    connection = open_store(str(record["store"]))
    write_execution_state(
        connection,
        "Writer",
        "running_model",
        [action_path],
        {"action": "echo", "kind": "model"},
    )
    connection.close()

    studio.execute("run inspect Writer")

    assert any(line == "Execution context" for line in output)
    assert any("Current position" in line and "Elapsed" in line for line in output)
    assert any("Writer" in line and "running model action" in line for line in output)
    assert any("▶" in line and "result = echo(value)" in line for line in output)
    assert any("workflow variables and action inputs" in line for line in output)


def test_studio_watches_current_run_once_per_second_without_stopping_it(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    record = workspace.new_run(
        workflow_spec="workflow.py:sample",
        workflow_name="sample",
        fingerprint="test",
        inputs={"value": "hello"},
        llm="mock",
    )
    workspace.update_run(str(record["run_id"]), status="running")
    studio._prompt_toolkit_enabled = True
    frames: list[str] = []

    def display_twice(frame_provider) -> bool:
        frames.extend([frame_provider(), frame_provider()])
        return True

    monkeypatch.setattr(studio, "_run_watch_display", display_twice)

    studio.execute("run inspect Writer --watch")

    assert len(frames) == 2
    assert all("Execution context" in frame for frame in frames)
    assert all("Refreshing once per second" in frame for frame in frames)
    assert all(
        "development run will keep running" in frame
        for frame in frames
    )
    assert not any(line == "Execution context" for line in output)
    assert any(
        "Stopped watching. The development run was not interrupted." in line
        for line in output
    )
    assert workspace.load_run(str(record["run_id"]))["status"] == "running"


def test_studio_restores_renderer_output_when_watch_capture_fails(tmp_path):
    studio, _workspace, _output = _studio(tmp_path)
    original_output = studio._renderer.output

    def fail() -> None:
        raise RuntimeError("inspection failed")

    with pytest.raises(RuntimeError, match="inspection failed"):
        studio._capture_watch_frame(
            fail,
            command="run inspect --watch",
            subject="development run",
        )

    assert studio._renderer.output is original_output


def test_studio_watch_display_exits_on_control_c(monkeypatch):
    from zippergen import studio as studio_module

    real_application = studio_module.Application
    with create_pipe_input() as pipe_input:
        pipe_input.send_text("\x03")

        def test_application(**kwargs):
            return real_application(
                input=pipe_input,
                output=DummyOutput(),
                **kwargs,
            )

        monkeypatch.setattr(studio_module, "Application", test_application)

        assert Studio._run_watch_display(lambda: "Stable frame") is True


def test_studio_rejects_watch_mode_outside_an_interactive_terminal(tmp_path):
    studio, workspace, _output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.new_run(
        workflow_spec="workflow.py:sample",
        workflow_name="sample",
        fingerprint="test",
        inputs={"value": "hello"},
        llm="mock",
    )

    with pytest.raises(SystemExit, match="requires an interactive terminal"):
        studio.execute("run inspect --watch")


def test_studio_explains_a_single_completion_match(tmp_path):
    studio, _workspace, _output = _studio(tmp_path)

    assert studio.completion_explanation("resu") == (
        " Tab: resume — resume the current incomplete run "
    )
    assert studio.completion_explanation("workflow ref") == (
        " Tab: refine — create or reopen the pending refinement "
    )
    assert studio.completion_explanation("studio res") == (
        " Tab: restart — replace this process and reload installed source "
    )
    assert studio.completion_explanation("") == ""


def test_welcome_and_studio_doctor_show_readiness_and_next_action(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path)
    monkeypatch.setattr(
        "zippergen.studio.shutil.which",
        lambda name: f"/tools/{name}" if name in {"codex", "micro"} else None,
    )

    studio.welcome()

    assert output[0].startswith("╭")
    assert "Session context" in output
    assert any(
        "Assistant" in line and "✓ Codex CLI found" in line
        for line in output
    )
    assert any("Type a command or describe" in line for line in output)
    next_title = output.index("Next")
    assert output[next_title + 1] == "═" * len("Next")
    assert output[next_title + 2] == "project init"

    output.clear()
    workspace.initialize_project(name="Tutorial")
    studio.execute("studio doctor")

    assert "Studio readiness" in output
    assert any("Codex CLI found" in line for line in output)
    assert any("workflow create" in line for line in output)

    output.clear()
    studio.welcome()

    assert any("Project" in line and "Tutorial" in line for line in output)
    assert not any("Type a command or describe" in line for line in output)


def test_studio_restart_replaces_the_original_process(tmp_path, monkeypatch):
    studio, workspace, output = _studio(tmp_path)
    launcher = tmp_path / "bin" / "zippergen"
    launcher.parent.mkdir()
    launcher.write_text("#!/bin/sh\n")
    launcher.chmod(0o755)
    calls: list[tuple[str, list[str]]] = []

    monkeypatch.setattr(
        "zippergen.studio.sys.argv",
        [str(launcher), "studio", "--project", str(workspace.root)],
    )
    monkeypatch.setattr(
        "zippergen.studio.os.execv",
        lambda executable, arguments: calls.append(
            (executable, list(arguments))
        ),
    )

    studio.execute("studio restart")

    assert calls == [
        (
            str(launcher),
            [str(launcher), "studio", "--project", str(workspace.root)],
        )
    ]
    assert "Studio restart" in output
    assert any(
        "saved project context will be reloaded" in line for line in output
    )
    assert any("Restarting ZipperGen Studio" in line for line in output)


def test_studio_restart_resolves_a_path_launcher(tmp_path, monkeypatch):
    studio, _workspace, _output = _studio(tmp_path)
    calls: list[tuple[str, list[str]]] = []

    monkeypatch.setattr("zippergen.studio.sys.argv", ["zippergen"])
    monkeypatch.setattr(
        "zippergen.studio.shutil.which",
        lambda command: "/tools/zippergen" if command == "zippergen" else None,
    )
    monkeypatch.setattr(
        "zippergen.studio.os.execv",
        lambda executable, arguments: calls.append(
            (executable, list(arguments))
        ),
    )

    studio.execute("studio restart")

    assert calls == [
        ("/tools/zippergen", ["/tools/zippergen"]),
    ]


def test_studio_restart_failure_keeps_the_current_session(
    tmp_path,
    monkeypatch,
):
    studio, _workspace, _output = _studio(tmp_path)
    launcher = tmp_path / "zippergen"
    launcher.write_text("#!/bin/sh\n")
    launcher.chmod(0o755)

    monkeypatch.setattr("zippergen.studio.sys.argv", [str(launcher)])

    def fail_exec(_executable, _arguments):
        raise OSError("permission denied")

    monkeypatch.setattr("zippergen.studio.os.execv", fail_exec)

    with pytest.raises(SystemExit, match="current Studio process is still running"):
        studio.execute("studio restart")


def test_studio_update_fast_forwards_and_restarts_same_project(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path)
    checkout = tmp_path / "zippergen-source"
    checkout.mkdir()
    calls: list[tuple[list[str], str]] = []
    restarted: list[Path] = []
    before = "1" * 40
    after = "2" * 40

    monkeypatch.setattr(
        studio,
        "_studio_source_checkout",
        lambda: checkout,
    )
    monkeypatch.setattr(
        "zippergen.studio.shutil.which",
        lambda command: f"/tools/{command}",
    )

    def run(arguments, *, operation):
        calls.append((list(arguments), operation))
        tail = arguments[3:]
        if tail == ["status", "--porcelain", "--untracked-files=no"]:
            stdout = ""
        elif tail == ["rev-parse", "--abbrev-ref", "HEAD"]:
            stdout = "main\n"
        elif tail == [
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        ]:
            stdout = "origin/main\n"
        elif tail == ["rev-parse", "HEAD"]:
            stdout = before + "\n" if sum(
                item[0][3:] == ["rev-parse", "HEAD"] for item in calls
            ) == 1 else after + "\n"
        elif tail == ["pull", "--ff-only"]:
            stdout = "Updating source\nFast-forward\n"
        elif tail == [
            "diff",
            "--name-only",
            before,
            after,
            "--",
            "pyproject.toml",
            "uv.lock",
        ]:
            stdout = ""
        else:
            raise AssertionError(arguments)
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(studio, "_update_subprocess", run)
    monkeypatch.setattr(
        studio,
        "restart_studio",
        lambda: restarted.append(workspace.root),
    )

    studio.execute("studio update")

    assert restarted == [workspace.root]
    assert any("fast-forwarded successfully" in line for line in output)
    assert any("installed bundles remain immutable" in line for line in output)
    assert not any(call[0][0] == "/tools/uv" for call in calls)


def test_studio_update_synchronizes_changed_dependency_metadata(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path)
    checkout = tmp_path / "zippergen-source"
    checkout.mkdir()
    (checkout / "uv.lock").write_text("locked")
    before = "a" * 40
    after = "b" * 40
    head_reads = 0
    calls: list[list[str]] = []

    monkeypatch.setattr(
        studio,
        "_studio_source_checkout",
        lambda: checkout,
    )
    monkeypatch.setattr(
        "zippergen.studio.shutil.which",
        lambda command: f"/tools/{command}",
    )

    def run(arguments, *, operation):
        nonlocal head_reads
        calls.append(list(arguments))
        if arguments[0] == "/tools/uv":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        tail = arguments[3:]
        if tail == ["status", "--porcelain", "--untracked-files=no"]:
            stdout = ""
        elif tail == ["rev-parse", "--abbrev-ref", "HEAD"]:
            stdout = "main\n"
        elif tail == [
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        ]:
            stdout = "origin/main\n"
        elif tail == ["rev-parse", "HEAD"]:
            head_reads += 1
            stdout = (before if head_reads == 1 else after) + "\n"
        elif tail == ["pull", "--ff-only"]:
            stdout = "Fast-forward\n"
        elif tail[:2] == ["diff", "--name-only"]:
            stdout = "pyproject.toml\n"
        else:
            raise AssertionError(arguments)
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(studio, "_update_subprocess", run)
    monkeypatch.setattr(studio, "restart_studio", lambda: None)

    studio.execute("studio update")

    assert [
        "/tools/uv",
        "sync",
        "--locked",
        "--project",
        str(checkout),
    ] in calls
    assert any("synchronized with uv" in line for line in output)


def test_studio_update_refuses_tracked_checkout_changes(
    tmp_path,
    monkeypatch,
):
    studio, _workspace, _output = _studio(tmp_path)
    checkout = tmp_path / "zippergen-source"
    checkout.mkdir()
    calls: list[list[str]] = []

    monkeypatch.setattr(
        studio,
        "_studio_source_checkout",
        lambda: checkout,
    )
    monkeypatch.setattr(
        "zippergen.studio.shutil.which",
        lambda command: f"/tools/{command}",
    )

    def run(arguments, *, operation):
        calls.append(list(arguments))
        return SimpleNamespace(
            returncode=0,
            stdout=" M src/zippergen/studio.py\n",
            stderr="",
        )

    monkeypatch.setattr(studio, "_update_subprocess", run)

    with pytest.raises(SystemExit, match="tracked local changes"):
        studio.execute("studio update")

    assert len(calls) == 1
    assert calls[0][3:] == [
        "status",
        "--porcelain",
        "--untracked-files=no",
    ]


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
    assert "press Tab to complete" in " ".join(line.strip() for line in output)


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


def test_studio_list_and_select_discover_workflow_entry_points(tmp_path):
    studio, workspace, output = _studio(tmp_path, responses=["1"])

    studio.list_workflows()

    assert workspace.current_workflow is None
    assert output[0] == "Available workflows"
    assert any("workflow.py:sample" in line for line in output)
    assert any("source scan only" in line for line in output)

    workspace.update(current_workflow=None)
    output.clear()
    studio.select_workflow([])

    assert workspace.current_workflow == "workflow.py:sample"
    assert "Workflow selected" in output
    assert any("Validation" in line and "not run" in line for line in output)


def test_studio_numbered_workflow_menus_name_the_requested_selection(tmp_path):
    studio, workspace, _output = _studio(tmp_path)
    (workspace.root / "alternate.py").write_text(
        WORKFLOW_SOURCE.replace("def sample(", "def alternate("),
        encoding="utf-8",
    )
    answers = iter(["2", "1", "2", "1,2"])
    prompts: list[str] = []

    def answer(prompt: str) -> str:
        prompts.append(prompt)
        return next(answers)

    studio.input = answer
    studio.execute("workflow select")
    studio.execute("workflow show")
    studio.execute("workflow show agent")
    studio.execute("workflow show agents")

    assert prompts == [
        "Select workflow [1-2]: ",
        "Select workflow view [1-8]: ",
        "Select participant [1-2]: ",
        "Select participants [1-2, comma-separated]: ",
    ]


def test_studio_show_prompts_for_entry_point_when_several_are_discovered(
    tmp_path,
):
    studio, workspace, output = _studio(tmp_path, responses=["2"])
    (workspace.root / "alternate.py").write_text(
        WORKFLOW_SOURCE.replace("def sample(", "def alternate("),
        encoding="utf-8",
    )

    studio.execute("workflow show protocol")

    assert output[0] == "Choose a workflow to inspect it"
    assert workspace.current_workflow in {
        "alternate.py:alternate",
        "workflow.py:sample",
    }
    assert any("validation has not run" in line for line in output)
    assert any("@workflow" in line for line in output)


def test_studio_lists_local_workflow_files_and_shows_selected_source(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    (workspace.root / "review_helpers.py").write_text(
        'REVIEW_POLICY = "human approval required"\n',
        encoding="utf-8",
    )
    workflow_path = workspace.root / "workflow.py"
    workflow_path.write_text(
        "import review_helpers\n\n" + workflow_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    studio.execute("workflow files")

    assert workspace.current_workflow == "workflow.py:sample"
    assert any(
        "workflow.py" in line and "entry point" in line for line in output
    )
    assert any(
        "review_helpers.py" in line and "local Python import" in line
        for line in output
    )
    assert any("validation has not run" in line.lower() for line in output)
    assert all("Workflow sample: valid" not in line for line in output)

    output.clear()
    studio.execute("workflow show source 2")

    assert output[0].startswith("Source: review_helpers.py")
    assert 'REVIEW_POLICY = "human approval required"' in output


def test_studio_show_menu_renders_communication_code(tmp_path):
    studio, workspace, output = _studio(tmp_path, responses=["4"])
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)

    studio.show_workflow([])

    rendered = "\n".join(output)
    assert "User(value) >> Writer(value)" in rendered
    assert "Writer(result) >> User(result)" in rendered
    assert "echo(value)" not in rendered
    assert workspace.load()["last_view"] == "communications"


def test_studio_show_agent_renders_exact_projection(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)

    studio.show_workflow(["agent", "Writer"])

    rendered = "\n".join(output)
    assert "Generated local projection for Writer" in rendered
    assert "value = recv('User')" in rendered


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
    assert any("✓ prepared" in line for line in output)
    assert any("workflow implement" in line for line in output)
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

    studio.execute('workflow create --file "prompts/reviewed answer.md"')

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
        studio.execute("workflow create Different requirements")
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
    assert any("Pending" in line and "✓ created" in line for line in output)
    assert all(".zippergen/pending-refinement.md" not in line for line in output)
    assert any("✓ prepared" in line for line in output)
    assert any(
        "Manual path" in line
        and "workflow edit code" in line
        and "workflow edit spec" in line
        for line in output
    )


def test_studio_refine_reads_prompt_from_absolute_file(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_specification("Echo a value through Writer and return it.")
    prompt_file = tmp_path / "change.md"
    prompt_file.write_text(
        "Add human review.\nPreserve the existing model call.\n",
        encoding="utf-8",
    )

    studio.execute(f'workflow refine --file "{prompt_file}"')

    briefs = list(workspace.requests_directory.glob("*-refine.md"))
    assert len(briefs) == 1
    content = briefs[0].read_text()
    assert "Add human review.\nPreserve the existing model call." in content
    assert output[0] == "Refinement"
    assert workspace.pending_refinement() == (
        "Add human review.\nPreserve the existing model call."
    )
    assert all("Loaded prompt file" not in line for line in output)


def test_studio_workflow_commands_expose_one_implementation_and_private_history(
    tmp_path,
):
    studio, workspace, output = _studio(tmp_path)
    studio.create_request("Create a review workflow.")
    output.clear()

    studio.execute("workflow status")
    assert output[0] == "Workflow implementation task"
    assert all(".zippergen/current-task.md" not in line for line in output)

    output.clear()
    studio.execute("workflow status --details")
    assert any(".zippergen/current-task.md" in line for line in output)
    assert any("assistant" in line for line in output)

    output.clear()
    studio.manage_task(["path"])
    assert output == [str(workspace.current_task_path)]

    output.clear()
    studio.manage_task(["show"])
    assert output == [workspace.current_task_path.read_text().rstrip()]

    output.clear()
    studio.execute("workflow history")
    assert output[0] == "Specification history"
    assert "Implementation history" in output
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

    studio.manage_task(["show"])

    refreshed = workspace.current_request()
    assert refreshed is not None
    assert refreshed["request_id"] != original["request_id"]
    assert refreshed["refreshes_request"] == original["request_id"]
    assert refreshed["specification_fingerprint"] == (
        workspace.specification_fingerprint()
    )
    assert output[0] == (
        "✓ Implementation request refreshed from the current specification context."
    )
    assert "add an explicit failure result" in output[1]
    assert len(workspace.list_requests()) == 2

    output.clear()
    studio.manage_task(["show"])

    assert output == [workspace.current_task_path.read_text().rstrip()]
    assert len(workspace.list_requests()) == 2

    output.clear()
    studio.execute("workflow history")

    assert any("Refreshes" in line for line in output)
    compact = "".join(line.strip() for line in output)
    assert str(refreshed["request_id"])[:16] in compact
    assert "…" in compact
    assert str(original["request_id"])[:16] in compact


def test_studio_refreshes_a_pre_verification_contract_task_before_launch(
    tmp_path,
):
    studio, workspace, output = _studio(tmp_path)
    studio.create_request("Create a review workflow.")
    original = workspace.current_request()
    assert original is not None
    metadata_path = workspace.request_path(str(original["request_id"]))
    metadata = json.loads(metadata_path.read_text())
    metadata.pop("task_contract_version")
    metadata_path.write_text(json.dumps(metadata))
    output.clear()

    studio.manage_task(["show"])

    refreshed = workspace.current_request()
    assert refreshed is not None
    assert refreshed["request_id"] != original["request_id"]
    assert refreshed["refreshes_request"] == original["request_id"]
    assert refreshed["task_contract_version"] == 3
    assert "assistant-result.json" in output[1]
    assert output[0] == (
        "✓ Implementation request refreshed from the current specification context."
    )


def test_studio_assistant_launches_codex_in_project_on_the_stable_task(
    tmp_path, monkeypatch
):
    studio, workspace, output = _studio(tmp_path)
    studio.create_request("Create a review workflow.")
    output.clear()
    calls: list[tuple[list[str], Path, bool]] = []

    monkeypatch.setattr("zippergen.studio.shutil.which", lambda name: "/bin/codex")

    def fake_run(arguments, *, cwd, check, **kwargs):
        assert kwargs == {"capture_output": True, "text": True}
        calls.append((arguments, cwd, check))
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr("zippergen.studio.subprocess.run", fake_run)

    studio.execute("workflow implement")

    assert calls[0][0][0:6] == [
        "/bin/codex",
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--cd",
        str(workspace.root),
    ]
    assert ".zippergen/current-task.md" in calls[0][0][6]
    assert calls[0][1] == workspace.root
    assert calls[0][2] is False
    assert output[0] == "Assistant"
    assert any(
        "Mode" in line and "returns to Studio automatically" in line
        for line in output
    )
    assert any("MCP" in line and "not required" in line for line in output)
    assert any("Codex session returned to Studio" in line for line in output)
    assert any("assistant checks are incomplete" in line.lower() for line in output)
    assert any(
        "Status" in line and "awaiting human review" in line for line in output
    )
    request = workspace.current_request()
    assert request is not None
    assert request["assistant_mode"] == "one_shot"
    assert request["status"] == "awaiting_review"
    assert request["assistant_verification"] == "incomplete"


def test_studio_condensed_assistant_reports_progress_and_precise_boundary(
    tmp_path, monkeypatch
):
    studio, _workspace, output = _studio(tmp_path)
    studio.create_request("Create a review workflow.")
    output.clear()
    monkeypatch.setattr("zippergen.studio.shutil.which", lambda name: "/bin/codex")
    monkeypatch.setattr("zippergen.studio._ASSISTANT_HEARTBEAT_SECONDS", 0.01)

    def fake_run(arguments, *, cwd, check, **kwargs):
        time.sleep(0.035)
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr("zippergen.studio.subprocess.run", fake_run)

    studio.execute("workflow implement", show_boundary=True)

    boundary = next(
        line
        for line in output
        if line.startswith("│ ZipperGen Studio · workflow implement ")
    )
    assert boundary.endswith("│")
    assert any(
        "Codex CLI is working" in line and "Control-C" in line
        for line in output
    )
    assert any(
        "Codex CLI is still working" in line and "elapsed" in line
        for line in output
    )


def test_studio_assistant_records_passed_verification_separately_from_exit(
    tmp_path, monkeypatch
):
    studio, workspace, output = _studio(tmp_path)
    studio.create_request("Create a review workflow.")
    output.clear()
    monkeypatch.setattr("zippergen.studio.shutil.which", lambda name: "/bin/codex")

    def fake_run(arguments, *, cwd, check, **kwargs):
        workspace.assistant_result_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "verification": "passed",
                    "summary": "Validation and focused application tests passed.",
                    "checks": [
                        {
                            "command": "uv run zippergen validate workflow.py:sample",
                            "status": "passed",
                            "detail": "Workflow is valid.",
                        },
                        {
                            "command": "uv run pytest tests/test_sample.py",
                            "status": "passed",
                            "detail": "2 passed.",
                        },
                    ],
                }
            )
        )
        events = [
            {"type": "thread.started", "thread_id": "thread-1"},
            {
                "type": "item.completed",
                "item": {
                    "type": "agent_message",
                    "text": "Updated the workflow and completed verification.",
                },
            },
            {"type": "turn.completed"},
        ]
        stderr = (
            "2026-07-23T21:05:59Z ERROR codex_models_manager::manager: "
            "failed to renew cache TTL: missing field "
            "`supports_reasoning_summaries`\n"
        )
        return subprocess.CompletedProcess(
            arguments,
            0,
            stdout="\n".join(json.dumps(event) for event in events),
            stderr=stderr,
        )

    monkeypatch.setattr("zippergen.studio.subprocess.run", fake_run)

    studio.execute("workflow implement")

    request = workspace.current_request()
    assert request is not None
    assert request["status"] == "awaiting_review"
    assert request["assistant_exit_code"] == 0
    assert request["assistant_verification"] == "passed"
    assert len(request["assistant_verification_checks"]) == 2
    assert request["assistant_suppressed_diagnostics"] == 1
    assert request["assistant_cli_diagnostics"] == []
    assert request["assistant_report"] == (
        "Updated the workflow and completed verification."
    )
    assert not workspace.assistant_result_path.exists()
    assert all("supports_reasoning_summaries" not in line for line in output)
    assert any("Assistant report" in line for line in output)
    assert any("Updated the workflow" in line for line in output)
    assert any("assistant checks passed" in line.lower() for line in output)
    assert any(
        "Assistant checks" in line and "passed" in line and "2 checks" in line
        for line in output
    )

    output.clear()
    studio.execute("workflow status")
    assert any(
        "Assistant checks" in line and "passed" in line and "2 checks" in line
        for line in output
    )
    assert any(
        "Check summary" in line and "focused application tests passed" in line
        for line in output
    )


def test_studio_implement_can_enter_guided_review_on_return(
    tmp_path,
    monkeypatch,
):
    studio, workspace, _output = _studio(tmp_path)
    studio.create_request("Create a review workflow.")
    monkeypatch.setattr(
        "zippergen.studio.shutil.which",
        lambda _name: "/bin/codex",
    )

    def fake_run(arguments, *, cwd, check, **kwargs):
        workspace.assistant_result_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "verification": "passed",
                    "summary": "Focused checks passed.",
                    "checks": [
                        {
                            "command": "uv run zippergen validate workflow.py:sample",
                            "status": "passed",
                            "detail": "Workflow is valid.",
                        }
                    ],
                }
            )
        )
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    monkeypatch.setattr("zippergen.studio.subprocess.run", fake_run)
    review_calls: list[bool] = []
    monkeypatch.setattr(
        studio,
        "review_workflow",
        lambda: review_calls.append(True),
    )

    studio.execute("workflow implement --review")

    assert review_calls == [True]


def test_studio_assistant_does_not_hide_a_failed_check_behind_zero_exit(
    tmp_path, monkeypatch
):
    studio, workspace, output = _studio(tmp_path)
    studio.create_request("Create a review workflow.")
    output.clear()
    monkeypatch.setattr("zippergen.studio.shutil.which", lambda name: "/bin/codex")

    def fake_run(arguments, *, cwd, check, **kwargs):
        workspace.assistant_result_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "verification": "passed",
                    "summary": "The focused test passed but broad collection failed.",
                    "checks": [
                        {
                            "command": "uv run pytest",
                            "status": "failed",
                            "detail": "prompt_toolkit was unavailable during collection.",
                        }
                    ],
                }
            )
        )
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr("zippergen.studio.subprocess.run", fake_run)

    studio.execute("workflow implement")

    request = workspace.current_request()
    assert request is not None
    assert request["status"] == "awaiting_review"
    assert request["assistant_exit_code"] == 0
    assert request["assistant_verification"] == "failed"
    assert any("assistant checks failed" in line.lower() for line in output)
    assert any(
        "Assistant checks" in line and "failed" in line for line in output
    )
    assert "Next" in output
    assert any("workflow implement codex --rerun" in line for line in output)
    assert "Failed or incomplete assistant checks" in output
    assert any("Command" in line and "uv run pytest" in line for line in output)
    assert any(
        "Result" in line and "prompt_toolkit was unavailable" in line
        for line in output
    )

    output.clear()
    studio.execute("workflow status")
    assert "Assistant checks" in output
    assert any("Command" in line and "uv run pytest" in line for line in output)
    assert any(
        "Result" in line and "prompt_toolkit was unavailable" in line
        for line in output
    )


def test_studio_verification_checks_are_wrapped_records_with_problem_priority(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path)
    monkeypatch.setattr(studio, "_output_columns", lambda: 72)
    studio.create_request("Create a review workflow.")
    record = workspace.current_request()
    assert record is not None
    long_command = (
        "UV_CACHE_DIR=/private/tmp/zippergen-uv-cache uv run --offline "
        "--no-sync --project zippergen zippergen validate "
        "workflows/reviewed_answer.py:reviewed_answer "
        + "very-long-option-" * 8
    )
    workspace.update_request(
        str(record["request_id"]),
        status="awaiting_review",
        assistant="Codex",
        assistant_verification="failed",
        assistant_verification_checks=[
            {
                "command": "uv run pytest tests",
                "status": "passed",
                "detail": "All focused tests passed.",
            },
            {
                "command": long_command,
                "status": "failed",
                "detail": (
                    "Validation found a deliberately long diagnostic that "
                    "must wrap without extending the terminal divider."
                ),
            },
            {
                "command": "uv run zippergen show --agent Writer",
                "status": "not_run",
                "detail": "Skipped after validation failed.",
            },
        ],
    )
    output.clear()

    studio.execute("workflow status")

    title = output.index("Assistant checks")
    checks = output[title:]
    assert "3 checks · 1 passed · 1 failed · 1 not run" in checks
    assert max(len(line) for line in checks) <= 72
    assert not any("Field" in line and "Value" in line for line in checks)
    failed = next(index for index, line in enumerate(checks) if "2. failed" in line)
    not_run = next(
        index for index, line in enumerate(checks) if "3. not run" in line
    )
    passed = next(index for index, line in enumerate(checks) if "1. passed" in line)
    assert failed < not_run < passed
    assert any(line.lstrip().startswith("Command") for line in checks)
    assert any(line.lstrip().startswith("Result") for line in checks)


def test_studio_task_explains_nested_framework_test_environment(tmp_path):
    studio, workspace, _output = _studio(tmp_path)
    framework = workspace.root / "zippergen"
    framework.mkdir()
    (framework / "pyproject.toml").write_text("[project]\nname = 'zippergen'\n")
    workspace.initialize_project()

    studio.create_request("Create a review workflow.")

    task = workspace.current_task_path.read_text()
    assert "uv run --offline --project zippergen zippergen" in task
    assert "uv run --offline --project zippergen pytest tests" in task
    assert "Do not run bare\n`uv run pytest`" in task
    assert "assistant-result.json" in task
    assert '"verification": "passed"' in task
    assert "non-human connector requirements" in task
    assert "Human delivery is inferred from `@human` action sites" in task


def test_studio_assistant_stops_before_launch_when_nested_tests_are_not_synced(
    tmp_path, monkeypatch
):
    studio, workspace, _output = _studio(tmp_path)
    framework = workspace.root / "zippergen"
    framework.mkdir()
    (framework / "pyproject.toml").write_text("[project]\nname = 'zippergen'\n")
    workspace.initialize_project()
    studio.create_request("Create a review workflow.")
    calls: list[list[str]] = []

    def find_tool(name: str):
        return {"codex": "/bin/codex", "uv": "/bin/uv"}.get(name)

    def fake_run(arguments, **kwargs):
        calls.append(arguments)
        return subprocess.CompletedProcess(
            arguments,
            1,
            stdout="",
            stderr="No module named pytest",
        )

    monkeypatch.setattr("zippergen.studio.shutil.which", find_tool)
    monkeypatch.setattr("zippergen.studio.subprocess.run", fake_run)

    with pytest.raises(SystemExit, match="uv sync --project zippergen"):
        studio.execute("workflow implement codex")

    assert calls == [
        [
            "/bin/uv",
            "run",
            "--offline",
            "--project",
            "zippergen",
            "python",
            "-c",
            "import pytest",
        ]
    ]
    request = workspace.current_request()
    assert request is not None
    assert request["status"] == "prepared"


def test_studio_assistant_codex_can_still_run_interactively(
    tmp_path, monkeypatch
):
    studio, workspace, output = _studio(tmp_path)
    studio.create_request("Create a review workflow.")
    output.clear()
    calls: list[tuple[list[str], Path, bool]] = []
    monkeypatch.setattr("zippergen.studio.shutil.which", lambda name: "/bin/codex")

    def fake_run(arguments, *, cwd, check, **kwargs):
        calls.append((arguments, cwd, check))
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr("zippergen.studio.subprocess.run", fake_run)

    studio.execute("workflow implement codex --interactive")

    assert calls[0][0][0:3] == [
        "/bin/codex",
        "--cd",
        str(workspace.root),
    ]
    assert ".zippergen/current-task.md" in calls[0][0][3]
    assert any(
        "Mode" in line and "interactive implementation session" in line
        for line in output
    )
    request = workspace.current_request()
    assert request is not None
    assert request["assistant_mode"] == "interactive"
    assert request["status"] == "awaiting_review"


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

    def fake_run(arguments, *, cwd, check, **kwargs):
        task = workspace.current_task_path.read_text()
        assert "Corrected creation requirement." in task
        assert "Original creation requirement." not in task
        calls.append((arguments, cwd, check))
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr("zippergen.studio.subprocess.run", fake_run)

    studio.execute("workflow implement")

    refreshed = workspace.current_request()
    assert refreshed is not None
    assert refreshed["request_id"] != original["request_id"]
    assert refreshed["refreshes_request"] == original["request_id"]
    assert refreshed["specification_fingerprint"] == (
        workspace.specification_fingerprint()
    )
    assert calls
    assert output[0] == (
        "✓ Implementation request refreshed from the current specification context."
    )
    assert output[1] == "Assistant"


def test_studio_assistant_can_launch_claude_code_on_the_same_task(
    tmp_path, monkeypatch
):
    studio, workspace, output = _studio(tmp_path)
    studio.create_request("Create a review workflow.")
    workspace.update_global_settings(assistant="claude")
    output.clear()
    calls: list[tuple[list[str], Path, bool]] = []

    def find_assistant(name: str):
        assert name == "claude"
        return "/bin/claude"

    monkeypatch.setattr("zippergen.studio.shutil.which", find_assistant)

    def fake_run(arguments, *, cwd, check, **kwargs):
        calls.append((arguments, cwd, check))
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr("zippergen.studio.subprocess.run", fake_run)

    studio.execute("workflow implement")

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
    assert any(
        "Mode" in line and "one-shot implementation" in line for line in output
    )
    assert any("Claude Code session returned to Studio" in line for line in output)
    assert workspace.current_request()["status"] == "awaiting_review"


def test_studio_assistant_reports_missing_codex_without_losing_task(
    tmp_path, monkeypatch
):
    studio, workspace, _output = _studio(tmp_path)
    studio.create_request("Create a review workflow.")
    monkeypatch.setattr("zippergen.studio.shutil.which", lambda name: None)

    try:
        studio.execute("workflow implement")
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
        studio.execute("workflow implement claude")
    except SystemExit as exc:
        assert "Claude Code was not found" in str(exc)
        assert "first-run authentication" in str(exc)
    else:
        raise AssertionError("assistant claude should require Claude Code")

    try:
        studio.execute("workflow implement unknown")
    except SystemExit as exc:
        assert "workflow implement codex" in str(exc)
        assert "workflow implement claude" in str(exc)
    else:
        raise AssertionError("unknown assistants should be rejected")

    with pytest.raises(
        SystemExit,
        match="only with workflow implement codex",
    ):
        studio.execute("workflow implement claude --interactive")
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

    def fake_run(arguments, *, cwd, check, **kwargs):
        workspace.save_specification(
            "Echo the request through Writer and require human approval "
            "before returning."
        )
        workspace.assistant_result_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "verification": "passed",
                    "summary": "Validation and focused tests passed.",
                    "checks": [
                        {
                            "command": "uv run zippergen validate workflow.py:sample",
                            "status": "passed",
                            "detail": "Workflow is valid.",
                        }
                    ],
                }
            )
        )
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr("zippergen.studio.subprocess.run", fake_run)

    studio.execute("workflow implement codex")
    completed = workspace.current_request()
    assert completed is not None
    assert completed["request_id"] == original["request_id"]
    assert completed["status"] == "awaiting_review"
    assert completed["assistant"] == "Codex"
    assert completed["assistant_exit_code"] == 0
    output.clear()

    studio.execute("workflow status")

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
    assert "Next" in output
    assert any("workflow review" in line for line in output)
    assert all("Implementation request refreshed" not in line for line in output)

    output.clear()
    studio.execute("current")

    assert any(
        "Implementation" in line and "awaiting human review" in line
        for line in output
    )
    assert any(
        "Task next" in line and "workflow review" in line for line in output
    )
    assert any(
        "Refinement" in line and "awaiting human review" in line
        for line in output
    )

    output.clear()
    studio.execute("workflow show pending")

    assert any(
        "Status" in line and "awaiting human review" in line for line in output
    )
    assert "Next" in output
    assert any("workflow review" in line for line in output)

    with pytest.raises(SystemExit, match="already returned.*awaiting human review"):
        studio.execute("workflow implement codex")


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

    def fake_run(arguments, *, cwd, check, **kwargs):
        workspace.save_specification(
            "Echo the request through Writer and require human approval "
            "before returning."
        )
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr("zippergen.studio.subprocess.run", fake_run)
    studio.execute("workflow implement codex")

    studio.execute("workflow implement codex --rerun")

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
        studio.execute("workflow implement codex")

    failed = workspace.current_request()
    assert failed is not None
    assert failed["status"] == "assistant_failed"
    assert failed["assistant_exit_code"] == 7
    output.clear()
    studio.execute("workflow status")
    assert any("Status" in line and "assistant failed" in line for line in output)

    def interrupt(arguments, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("zippergen.studio.subprocess.run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        studio.execute("workflow implement codex")

    interrupted = workspace.current_request()
    assert interrupted is not None
    assert interrupted["status"] == "assistant_interrupted"


def test_studio_run_explains_recovery_after_assistant_interrupt(
    tmp_path, monkeypatch
):
    studio, _workspace, output = _studio(
        tmp_path,
        responses=["workflow implement", "exit"],
    )
    studio.create_request("Create a review workflow.")
    output.clear()
    monkeypatch.setattr("zippergen.studio.shutil.which", lambda name: "/bin/codex")

    def interrupt(arguments, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("zippergen.studio.subprocess.run", interrupt)

    assert studio.run() == 0

    assert any(
        "Assistant interrupted" in line
        and "workflow status" in line
        and "preserved" in line
        for line in output
    )


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

    studio.execute("workflow status")

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

    studio.execute("workflow status")

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

    studio.execute("workflow status")

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
    assert "Next" in output
    assert any("workflow review" in line for line in output)


def test_studio_workflow_review_guides_requirements_and_can_be_resumed(
    tmp_path,
):
    studio, workspace, output = _studio(tmp_path, responses=["1", "6"])
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_specification("Echo the request through Writer.")
    studio.refine_request("Require human approval before returning.")
    workspace.save_specification(
        "Echo the request through Writer and require human approval "
        "before returning."
    )

    studio.execute("workflow review")

    assert workspace.current_request() is not None
    assert workspace.pending_refinement() is not None
    assert "Workflow review" in output
    assert "Review actions" in output
    assert any("Require human approval before returning." in line for line in output)
    assert any(
        "Echo the request through Writer and require human approval" in line
        for line in output
    )
    assert "Specification diff" in output
    assert "Semantic workflow diff" in output
    assert any("Review remains open" in line for line in output)


def test_studio_workflow_review_can_accept_the_reviewed_refinement(tmp_path):
    studio, workspace, output = _studio(tmp_path, responses=["4", "y"])
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_specification("Echo the request through Writer.")
    studio.refine_request("Require human approval before returning.")
    workspace.save_specification(
        "Echo the request through Writer and require human approval "
        "before returning."
    )

    studio.execute("workflow review")

    assert workspace.current_request() is None
    assert workspace.pending_refinement() is None
    assert any(line == "Specification refinement" for line in output)
    assert any(
        "Implementation" in line and "accepted" in line
        for line in output
    )


def test_studio_workflow_review_requires_a_returned_implementation(tmp_path):
    studio, _workspace, _output = _studio(tmp_path)
    studio.create_request("Create a review workflow.")

    with pytest.raises(SystemExit, match="not awaiting human review"):
        studio.execute("workflow review")


def test_studio_workflow_accept_keeps_history_and_accepts_refinements(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    studio.create_request("Create a review workflow.")

    studio.execute("workflow accept --yes")

    assert workspace.current_request() is None
    assert not workspace.current_task_path.exists()
    assert workspace.list_requests()[0]["status"] == "closed"
    assert "Review comparison" in output
    assert any(line == "Workflow implementation accepted" for line in output)

    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_specification("Echo the request through Writer.")
    studio.refine_request("Require human approval.")
    with pytest.raises(SystemExit, match="workflow accept.*workflow discard"):
        studio.manage_task(["close", "--yes"])


def test_studio_accepts_an_existing_selected_workflow_without_a_task(tmp_path):
    studio, workspace, output = _studio(tmp_path, responses=["y"])
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_specification("Echo the request through Writer.")

    studio.execute("workflow accept")

    accepted = workspace.accepted_review("workflow.py:sample")
    assert accepted is not None
    assert accepted["request_id"] is None
    assert workspace.current_request() is None
    assert "Existing workflow acceptance" in output
    assert "Workflow baseline accepted" in output
    assert any(
        "nothing was run or deployed" in line for line in output
    )


def test_studio_repeated_baseline_accept_is_idempotent(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_specification("Echo the request through Writer.")
    studio.execute("workflow accept --yes")
    first = workspace.accepted_review("workflow.py:sample")
    assert first is not None
    first_source = first["accepted_source"]
    assert isinstance(first_source, dict)
    first_root = first_source["root"]
    output.clear()

    studio.execute("workflow accept")

    second = workspace.accepted_review("workflow.py:sample")
    assert second is not None
    second_source = second["accepted_source"]
    assert isinstance(second_source, dict)
    assert second_source["root"] == first_root
    assert output[0] == "Workflow baseline already accepted"


def test_studio_can_reaccept_source_only_drift_without_a_dummy_refinement(
    tmp_path,
):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_specification("Echo the request through Writer.")
    studio.execute("workflow accept --yes")
    first = workspace.accepted_review("workflow.py:sample")
    assert first is not None
    first_source = first["accepted_source"]
    assert isinstance(first_source, dict)
    first_root = first_source["root"]
    source = workspace.root / "workflow.py"
    source.write_text(source.read_text() + "\n# Reviewed documentation note.\n")
    output.clear()

    studio.execute("workflow accept --yes")

    second = workspace.accepted_review("workflow.py:sample")
    assert second is not None
    second_source = second["accepted_source"]
    assert isinstance(second_source, dict)
    assert second_source["root"] != first_root
    assert any(
        "Source files" in line and "modified workflow.py" in line
        for line in output
    )
    assert "Workflow baseline accepted" in output


def test_studio_remembers_shows_and_resets_editor_preference(
    tmp_path, monkeypatch
):
    studio, workspace, output = _studio(tmp_path)
    monkeypatch.setattr(
        "zippergen.studio.shutil.which",
        lambda name: f"/usr/bin/{name}" if name == "micro" else None,
    )

    studio.execute("editor set micro")

    assert workspace.global_settings()["editor_command"] == ["micro"]
    assert output[-1] == "✓ Global editor preference: micro"

    output.clear()
    studio.execute("editor show")

    assert output[0] == "Editor"
    assert any("Preference" in line and "micro" in line for line in output)
    assert any("Effective" in line and "/usr/bin/micro" in line for line in output)
    assert any("Source" in line and "global preference" in line for line in output)

    studio.execute("editor reset")

    assert workspace.global_settings()["editor_command"] is None
    assert output[-1] == (
        "✓ Global editor preference reset to automatic discovery."
    )


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

    studio.execute("workflow edit code")
    studio.execute("workflow edit code --editor micro")

    assert calls == [
        (["/usr/bin/nano", str(workspace.root / "workflow.py")], workspace.root, False),
        (["/opt/bin/micro", str(workspace.root / "workflow.py")], workspace.root, False),
    ]
    assert any("global preference" in line for line in output)
    assert any("one-off" in line for line in output)
    assert "Next" in output
    assert "workflow validate · workflow show · run" in output


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

    studio.execute("workflow create --editor micro")

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
    rendered = "\n".join(output)
    assert "Workflow specification" in rendered
    assert "/usr/bin/micro" in rendered
    assert "save and exit the editor to continue in Studio" in rendered
    assert output.count("Workflow specification") == 1
    assert "Editor" not in output


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
        studio.execute("workflow create --editor micro")
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

    studio.execute("workflow create --edit --editor micro")

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

    studio.execute("workflow refine --editor micro")
    first_baselines = list(
        workspace.requests_directory.glob("*-semantic-before.json")
    )
    studio.execute("workflow refine --editor micro")

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


def test_studio_refine_can_start_implementation_without_a_second_command(
    tmp_path,
    monkeypatch,
):
    studio, workspace, _output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_specification("Echo the input through Writer.")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        studio,
        "run_assistant",
        lambda arguments: calls.append(arguments),
    )

    studio.execute(
        "workflow refine Increase the retry budget --implement"
    )

    assert workspace.pending_refinement() == "Increase the retry budget"
    assert calls == [[]]


def test_studio_refine_can_chain_implementation_into_guided_review(
    tmp_path,
    monkeypatch,
):
    studio, workspace, _output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_specification("Echo the input through Writer.")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        studio,
        "run_assistant",
        lambda arguments: calls.append(arguments),
    )

    studio.execute(
        "workflow refine Increase the retry budget --implement --review"
    )

    assert calls == [["--review"]]


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
        studio.execute("workflow create --editor micro")
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
            "workflow create --editor missing",
            "Editor executable was not found: missing",
        ),
        (
            "workflow edit code --editor missing",
            "Editor executable was not found: missing",
        ),
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
        studio.execute("workflow create --editor micro")
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
        studio.execute(
            "workflow create --edit prompts/custom.md --editor micro"
        )
    except SystemExit as exc:
        assert "only used when workflow create opens" in str(exc)
    else:
        raise AssertionError("create should own the specification filename")


def test_studio_prompt_file_errors_are_actionable(tmp_path):
    studio, workspace, _output = _studio(tmp_path)
    (workspace.root / "empty.md").write_text("", encoding="utf-8")
    (workspace.root / "prompt-directory").mkdir()
    (workspace.root / "not-utf8.md").write_bytes(b"\xff")

    for command, expected in (
        ("workflow create --file", "Use workflow create --file PATH."),
        ("workflow create --file missing.md", "Prompt file does not exist:"),
        ("workflow create --file empty.md", "Prompt file is empty:"),
        (
            "workflow create --file prompt-directory",
            "Prompt path is a directory:",
        ),
        (
            "workflow create --file not-utf8.md",
            "Prompt file must contain UTF-8 text:",
        ),
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
    assert "workflow create" in output[-1]
    assert "help all" in output[-1]

    output.clear()
    assert studio.execute("help all") is True
    assert "workflow show" in output[-1]
    assert "project init [NAME]" in output[-1]
    assert "project reset [fresh|state] [--yes]" in output[-1]
    assert "workflow history" in output[-1]
    assert "workflow status" in output[-1]
    assert "workflow review" in output[-1]
    assert "workflow implement" in output[-1]
    assert "editor set COMMAND" in output[-1]
    assert "edit file PATH" in output[-1]
    assert "workflow create" in output[-1]
    assert "workflow edit" in output[-1]
    assert "workflow refine" in output[-1]
    assert "model provider" in output[-1]
    assert "model config" in output[-1]
    assert "model config list|create|show|check" in output[-1]
    assert "model assign PARTICIPANT_OR_ACTION NAME" in output[-1]
    assert "NATURAL LANGUAGE" in output[-1]
    assert "language history" in output[-1]
    assert "language learned" in output[-1]
    assert studio.execute("exit") is False


def test_studio_retires_the_providers_command_with_targeted_guidance(tmp_path):
    studio, _workspace, _output = _studio(tmp_path)

    with pytest.raises(
        SystemExit,
        match=r"`providers` is not a Studio command.*model provider configure NAME",
    ):
        studio.execute("providers set openai")


def test_studio_retires_the_plural_models_command_with_targeted_guidance(
    tmp_path,
):
    studio, _workspace, _output = _studio(tmp_path)

    with pytest.raises(
        SystemExit,
        match=r"`models` was renamed to `model`",
    ):
        studio.execute("models assignments")


def test_studio_retires_the_deployment_namespace_with_targeted_guidance(
    tmp_path,
):
    studio, _workspace, _output = _studio(tmp_path)

    with pytest.raises(
        SystemExit,
        match=r"`deployment` was replaced by the single `deploy` namespace",
    ):
        studio.execute("deployment show")


@pytest.mark.parametrize(
    ("legacy", "replacement"),
    [
        ("status", "show"),
        ("doctor", "doctor"),
        ("logs", "logs"),
        ("start", "start"),
        ("restart", "restart"),
        ("stop", "stop"),
    ],
)
def test_studio_retires_legacy_deployment_verbs_with_targeted_guidance(
    tmp_path,
    legacy,
    replacement,
):
    studio, _workspace, _output = _studio(tmp_path)

    with pytest.raises(
        SystemExit,
        match=rf"`{legacy}` is no longer a Studio command.*"
        rf"`deploy {replacement} \[NAME\]`",
    ):
        studio.execute(legacy)


def test_studio_project_rename_changes_only_the_logical_manifest_name(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    studio.execute("project init Tutorial")
    original_root = workspace.root
    output.clear()

    studio.execute('project rename "Reviewed Answer Tutorial"')

    manifest = workspace.project_manifest()
    assert manifest["name"] == "Reviewed Answer Tutorial"
    assert workspace.root == original_root
    assert manifest["specification_file"] == "specification.md"
    assert output[0] == "Project renamed"
    assert any("From" in line and "Tutorial" in line for line in output)
    assert any(
        "To" in line and "Reviewed Answer Tutorial" in line
        for line in output
    )
    rendered = "\n".join(output)
    assert "Root" in rendered
    assert "unchanged" in rendered


def test_studio_settings_are_global_while_language_data_stays_project_local(
    tmp_path,
):
    studio, workspace, output = _studio(tmp_path)

    studio.execute("settings set learning off")
    studio.execute("settings set output compact")

    second_root = tmp_path / "second-project"
    second_root.mkdir()
    second_workspace = Workspace(second_root, home=workspace.home)
    second_studio = Studio(
        second_workspace,
        input_func=lambda _prompt: "",
        output_func=lambda _value: None,
    )

    assert second_workspace.global_settings()["learning"] is False
    assert second_workspace.global_settings()["output_style"] == "compact"
    assert second_studio._global_settings()["learning"] is False
    assert workspace.natural_language_path != second_workspace.natural_language_path
    assert workspace.global_settings_path == second_workspace.global_settings_path
    assert workspace.global_settings_path.stat().st_mode & 0o077 == 0
    assert any("all local ZipperGen projects" in line for line in output)

    # A full reset writes explicit defaults, so stale pre-global project
    # preferences cannot be migrated back on the next command.
    workspace.update(editor_command=["nano"])
    studio.execute("settings reset all")

    assert studio._global_settings()["learning"] is True
    assert studio._global_settings()["editor_command"] is None
    assert studio._global_settings()["output_style"] == "banner"


def test_studio_banner_is_connected_and_compact_style_is_configurable(tmp_path):
    studio, workspace, output = _studio(tmp_path)

    studio.execute("current", show_boundary=True)

    assert output[0] == ""
    assert output[1] == "╭" + "─" * 58 + "╮"
    assert output[2].startswith("│ ZipperGen Studio · current ")
    assert output[2].endswith("│")
    assert len(output[1]) == len(output[2]) == len(output[3])
    assert output[3] == "╰" + "─" * 58 + "╯"

    workspace.update_global_settings(output_style="compact")
    output.clear()
    studio.execute("current", show_boundary=True)

    assert output[1].startswith("── ZipperGen Studio · current ")
    assert not any(line.startswith("╭") for line in output)


def test_studio_structured_output_separates_titles_headers_rows_and_next(
    tmp_path,
):
    studio, _workspace, output = _studio(tmp_path)

    studio.execute("workflow", show_boundary=True)

    title = output.index("Workflow development")
    assert output[title + 1] == "═" * len("Workflow development")
    assert output[title + 2].split() == ["Field", "Value"]
    assert set(output[title + 3].replace(" ", "")) == {"─"}
    assert output[title + 4].lstrip().startswith("Specification")

    next_title = output.index("Next")
    assert output[next_title + 1] == "═" * len("Next")
    assert output[next_title + 2] == "workflow create"


def test_studio_status_marks_use_color_only_when_enabled(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    workspace = Workspace(root, home=tmp_path / "home")
    output: list[str] = []
    studio = Studio(workspace, output_func=output.append, color=True)

    studio.execute("project init Tutorial")
    studio._error("Example failure")

    assert output[0].startswith(
        "\033[32m✓\033[0m Project manifest created:"
    )
    assert output[-1].startswith(
        "\033[31m✗\033[0m Example failure"
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

    assert output[0] == "Workflow validation"
    assert output[1] == "═" * len("Workflow validation")
    assert output[2] == "✓ Workflow sample: valid"
    distinction = output.index("Validation and acceptance")
    assert all(line.startswith("  ✓ ") for line in output[3:distinction])
    assert any(
        "Technical validation" in line and "passed" in line
        for line in output[distinction:]
    )
    assert any(
        "Human acceptance" in line and "not recorded" in line
        for line in output[distinction:]
    )
    assert any(
        "validate checks workflow structure" in line
        for line in output[distinction:]
    )


def test_studio_validate_automatically_selects_one_discovered_workflow(tmp_path):
    studio, _workspace, output = _studio(
        tmp_path,
        responses=["workflow validate", "exit"],
    )

    assert studio.run() == 0

    assert any("Automatically selected workflow.py:sample" in line for line in output)
    assert any(line.startswith("✓ Workflow sample: valid") for line in output)


def test_studio_interactive_commands_have_a_clear_output_boundary(tmp_path):
    studio, _workspace, output = _studio(
        tmp_path,
        responses=["current", "exit"],
    )

    assert studio.run() == 0

    boundaries = [
        line
        for line in output
        if line.startswith("│ ZipperGen Studio · ")
    ]
    assert len(boundaries) == 1
    assert boundaries[0].startswith("│ ZipperGen Studio · current ")
    boundary_index = output.index(boundaries[0])
    assert output[boundary_index - 1].startswith("╭")
    assert output[boundary_index + 1].startswith("╰")
    assert output[boundary_index - 2] == ""
    assert all("exit" not in line for line in boundaries)


def test_studio_boundaries_hide_arguments_and_skip_empty_or_exit(tmp_path):
    studio, _workspace, output = _studio(tmp_path)

    studio.execute(
        "workflow create Never expose SECRET_SENTINEL in a boundary",
        show_boundary=True,
    )

    boundary = next(
        line
        for line in output
        if line.startswith("│ ZipperGen Studio · workflow create ")
    )
    assert "SECRET_SENTINEL" not in boundary

    output.clear()
    assert studio.execute("", show_boundary=True) is True
    assert studio.execute("exit", show_boundary=True) is False
    assert output == []


def test_studio_parse_errors_receive_an_input_boundary(tmp_path):
    studio, _workspace, output = _studio(tmp_path)

    studio.execute('workflow create "unterminated', show_boundary=True)

    assert output[0] == ""
    assert output[1].startswith("╭")
    assert output[2].startswith("│ ZipperGen Studio · input ")
    assert output[3].startswith("╰")
    assert output[4].startswith("✗ Could not parse command:")


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
    compact = "".join(line.strip() for line in output)
    assert str(backups[0]) in compact
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
    assert any("project init · workflow create" in line for line in output)

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
    compact = "".join(line.strip() for line in output)
    assert str(root) in compact
    assert "missing" in compact
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

    studio.execute("workflow path")
    assert output == [str(workspace.root / "specification.md")]

    output.clear()
    studio.execute("workflow refine Add bounded retries")
    studio.execute(
        "workflow refine Return an explicit failure after exhaustion"
    )

    assert workspace.pending_refinement_path == (
        workspace.root / ".zippergen" / "pending-refinement.md"
    )
    assert workspace.pending_refinement() == (
        "Add bounded retries\n\nReturn an explicit failure after exhaustion"
    )
    assert len(
        list(workspace.requests_directory.glob("*-semantic-before.json"))
    ) == 1
    assert any("Implementation" in line and "prepared" in line for line in output)
    assert not (workspace.root / "prompts").exists()

    output.clear()
    studio.execute("workflow show pending")
    assert output[0] == "Pending refinement"
    assert any("Add bounded retries" in line for line in output)


def test_studio_reconcile_requires_integrated_spec_and_keeps_private_history(
    tmp_path,
):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    studio.create_request("Echo the request through Writer.")
    studio.execute(
        "workflow refine Add a human approval before returning"
    )

    try:
        studio.execute("workflow accept --yes")
    except SystemExit as exc:
        assert "canonical specification has not changed" in str(exc)
    else:
        raise AssertionError("an unintegrated refinement must not be reconciled")

    workspace.save_specification(
        "Echo the request through Writer and require human approval before return."
    )
    output.clear()
    studio.execute("workflow accept --yes")

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
    accepted = workspace.accepted_review("workflow.py:sample")
    assert accepted is not None
    assert accepted["specification"] == (
        "Echo the request through Writer and require human approval before return."
    )
    assert isinstance(accepted["semantic_snapshot"], dict)
    history = workspace.list_spec_history()[0]
    assert Path(str(history["specification_before_file"])).read_text().strip() == (
        "Echo the request through Writer."
    )
    assert Path(str(history["specification_after_file"])).read_text().strip() == (
        "Echo the request through Writer and require human approval before return."
    )


def test_studio_review_diff_compares_intent_and_semantics_before_acceptance(
    tmp_path,
):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_specification("Echo the request through Writer.")
    studio.refine_request("Require explicit human approval.")
    workspace.save_specification(
        "Echo the request through Writer and require explicit human approval."
    )
    output.clear()

    studio.execute("workflow diff")

    assert "Review comparison" in output
    assert "Specification diff" in output
    assert "Semantic workflow diff" in output
    assert any("-Echo the request through Writer." in line for line in output)
    assert any(
        "+Echo the request through Writer and require explicit human approval."
        in line
        for line in output
    )
    assert any("# No semantic changes." in line for line in output)


def test_studio_accept_records_review_and_reports_later_intent_drift(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    studio.create_request("Echo the request through Writer.")

    studio.execute("workflow accept --yes")
    output.clear()
    studio.execute("workflow validate")

    assert any(
        "Human acceptance" in line
        and "matches accepted intent and workflow semantics" in line
        for line in output
    )

    workspace.save_specification(
        "Echo the request through Writer. Add a future human review."
    )
    output.clear()
    studio.execute("workflow validate")

    assert any(
        "Human acceptance" in line
        and "specification changed since the last accepted review" in line
        for line in output
    )


def test_studio_reports_semantic_drift_after_an_accepted_source_edit(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    studio.create_request("Echo the request through Writer.")
    studio.execute("workflow accept --yes")

    source = workspace.root / "workflow.py"
    source.write_text(
        source.read_text().replace(
            'system="Echo the value."',
            'system="Echo the value in one short sentence."',
        )
    )
    output.clear()

    studio.execute("workflow validate")

    assert any(
        "Human acceptance" in line
        and "workflow semantics changed since the last accepted review" in line
        for line in output
    )


def test_studio_spec_discard_is_explicit_and_recoverable(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    studio.create_request("Echo the request through Writer.")
    studio.execute("workflow refine Remove Writer")
    output.clear()

    studio.execute("workflow discard --yes")

    assert workspace.pending_refinement() is None
    assert workspace.list_spec_history()[0]["status"] == "discarded"
    assert any("⚠ discarded" in line for line in output)
    assert any(
        "Working tree" in line and "not reverted" in line for line in output
    )


def test_studio_spec_show_migrates_legacy_prompt_ledger_once(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    first = workspace.add_prompt(kind="initial", content="Create a reviewer.")
    second = workspace.add_prompt(
        kind="refinement",
        content="Add bounded retries.",
    )
    output.clear()

    studio.execute("workflow show spec")

    assert "Create a reviewer." in workspace.specification()
    assert "Add bounded retries." in workspace.specification()
    assert (workspace.root / str(first["file"])).exists()
    assert (workspace.root / str(second["file"])).exists()
    assert any("Migrated the former active prompt ledger" in line for line in output)

    output.clear()
    studio.execute("workflow show spec")
    assert all(
        not line.startswith("• Migrated the former active prompt ledger")
        for line in output
    )


def test_studio_legacy_prompt_ledger_is_migrated_but_has_no_command_surface(
    tmp_path,
):
    studio, workspace, output = _studio(tmp_path)
    workspace.add_prompt(kind="initial", content="Create a reviewer.")
    studio.execute("workflow show spec")

    with pytest.raises(SystemExit, match="was retired"):
        studio.execute(
            "workflow prompts add This must not become hidden design intent"
        )

    assert len(workspace.list_prompts()) == 1
    output.clear()
    studio.execute("workflow show spec")
    assert "Create a reviewer." in output


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
    assert any("Implementation task" in line for line in output)
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
    assert any(
        "Provider mock" in line and "available; built in" in line
        for line in output
    )
    assert any("Run" in line and "none" in line for line in output)
    assert any("Deployment" in line and "none" in line for line in output)


def test_studio_project_is_the_headless_project_inventory(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    workspace.initialize_project(name="Sample project")
    current = workspace.select_workflow(
        "workflow.py:sample",
        cwd=workspace.root,
    )
    default_configuration = workspace.ensure_model_configuration(
        "local:qwen2.5:14b"
    )
    workspace.save_model_assignment_profile(
        current,
        default=default_configuration,
        lifelines={},
    )
    workspace.specification_path.write_text("Create an echo workflow.\n")

    studio.execute("project")

    rendered = "\n".join(output)
    assert "Project · Sample project" in rendered
    assert "├── Workflow · sample · workflow.py:sample" in rendered
    assert "│   ├── Specification" in rendered
    assert (
        f"├── Models · default {default_configuration} · "
        "0 overrides · 1 LLM action"
    ) in rendered
    assert "├── Connectors" in rendered
    assert "├── Runs" in rendered
    assert "└── Deployments" in rendered
    assert "show" not in _completions(studio, "project ")


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


def test_studio_configures_checks_and_binds_a_telegram_connector(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(
        tmp_path,
        responses=["123456"],
        secret_responses=["private-bot-token"],
    )
    (workspace.root / "workflow.py").write_text(HUMAN_SOURCE)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)

    def fake_request(_client, method, **params):
        if method == "getMe":
            return {"ok": True, "result": {"username": "zippergen_test"}}
        assert method == "getChat"
        assert params["chat_id"] == "123456"
        return {"ok": True, "result": {"id": 123456}}

    monkeypatch.setattr(
        "zippergen.telegram_notify.TelegramBotClient.request",
        fake_request,
    )

    studio.execute("connector provider configure telegram")
    studio.execute("connector config create review-telegram")
    studio.execute("connector assign Human review-telegram")

    configuration = workspace.connector_configurations()["review-telegram"]
    assert configuration["provider"] == "telegram"
    assert configuration["check_status"] == "available"
    assert workspace.connector_provider_secret(
        "telegram", "bot_token"
    ) == "private-bot-token"
    assert workspace.connector_assignment_profile(
        "workflow.py:sample"
    ) == {
        "lifelines": {"Human": "review-telegram"},
        "actions": {},
    }
    assert all("private-bot-token" not in line for line in output)
    assert _completions(
        studio, "connector assign Human "
    ) == ["review-telegram"]

    current, workflow, module = studio._current_context()
    arguments = studio._deployment_connector_arguments(
        workflow_spec=current,
        workflow=workflow,
        module=module,
    )
    snapshot = json.loads(
        arguments[arguments.index("--connectors-json") + 1]
    )
    binding = snapshot["human:Human"]
    assert binding["configuration"] == "review-telegram"
    assert binding["target"] == "Human"
    assert binding["chat_id"] == "123456"
    assert binding["token_env"].startswith("ZIPPERGEN_CONNECTOR_")
    secret_argument = arguments[arguments.index("--connector-secret") + 1]
    assert secret_argument.endswith("=private-bot-token")


def test_studio_guides_google_sheet_setup_and_builds_private_runtime_context(
    tmp_path,
    monkeypatch,
):
    credentials = tmp_path / "google-desktop.json"
    credentials.write_text(GOOGLE_DESKTOP_CLIENT)
    sheet_url = (
        "https://docs.google.com/spreadsheets/d/sheet-123/edit#gid=0"
    )
    studio, workspace, output = _studio(
        tmp_path,
        responses=[str(credentials), sheet_url, "Calls"],
    )
    (workspace.root / "workflow.py").write_text(GOOGLE_SHEETS_SOURCE)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    monkeypatch.setattr(
        studio,
        "_needs_remote_google_browser",
        lambda: False,
    )
    monkeypatch.setattr(
        "zippergen.google_auth.authorize_google_client_result",
        lambda value, *, scopes: __import__(
            "zippergen.google_auth", fromlist=["GoogleAuthorization"]
        ).GoogleAuthorization(
            authorized_user_json=(
                '{"refresh_token":"private-google-token"}'
            ),
            granted_scopes=tuple(scopes),
            client_id="example.apps.googleusercontent.com",
        ),
    )
    monkeypatch.setattr(
        "zippergen.google_auth.check_google_authorization",
        lambda value, *, scopes: value,
    )
    monkeypatch.setattr(
        "zippergen.google_sheets.GoogleSheetsTable.inspect",
        lambda self: {
            "title": "Call records",
            "tab": self.tab,
            "tabs": [self.tab],
        },
    )

    studio.execute("connector setup")

    assert workspace.connector_binding_profile(
        "workflow.py:sample"
    ) == {"call-records": "call-records"}
    configuration = workspace.connector_configurations()["call-records"]
    assert configuration["provider"] == "google"
    assert configuration["kind"] == "google-sheets"
    assert configuration["spreadsheet_id"] == "sheet-123"
    assert configuration["tab"] == "Calls"
    assert configuration["check_status"] == "available"
    assert workspace.connector_provider_secret(
        "google", "authorized_user_json"
    ) == '{"refresh_token":"private-google-token"}'
    assert workspace.connector_provider_secret(
        "google", "oauth_client_json"
    ) is None
    assert (
        workspace.connector_provider_profiles()["google"]["client_storage"]
        == "not retained by Studio"
    )
    assert workspace.connector_provider_profiles()["google"][
        "granted_scopes"
    ]
    assert "credentials_file" not in (
        workspace.connector_provider_profiles()["google"]
    )

    current, workflow, module = studio._current_context()
    environment = studio._workflow_connector_environment(
        workflow_spec=current,
        workflow=workflow,
        module=module,
    )
    snapshot = json.loads(environment["ZIPPERGEN_CONNECTORS_JSON"])
    binding = snapshot["requirement:call-records"]
    assert binding["provider"] == "google"
    assert binding["kind"] == "google-sheets"
    assert binding["spreadsheet_id"] == "sheet-123"
    assert binding["tab"] == "Calls"
    assert environment[binding["credential_env"]] == (
        '{"refresh_token":"private-google-token"}'
    )
    assert all("private-google-token" not in line for line in output)
    assert "google" in _completions(
        studio, "connector provider configure "
    )


def test_studio_guides_one_google_authorization_for_gmail_and_sheets(
    tmp_path,
    monkeypatch,
):
    credentials = tmp_path / "google-desktop.json"
    credentials.write_text(GOOGLE_DESKTOP_CLIENT)
    studio, workspace, _output = _studio(
        tmp_path,
        responses=[
            str(credentials),
            "is:unread label:Calls",
            "sheet-123",
            "Calls",
        ],
    )
    (workspace.root / "workflow.py").write_text(GMAIL_AND_SHEETS_SOURCE)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    monkeypatch.setattr(
        studio,
        "_needs_remote_google_browser",
        lambda: False,
    )
    requested_scopes = []
    monkeypatch.setattr(
        "zippergen.google_auth.authorize_google_client_result",
        lambda value, *, scopes: (
            requested_scopes.extend(scopes)
            or __import__(
                "zippergen.google_auth", fromlist=["GoogleAuthorization"]
            ).GoogleAuthorization(
                authorized_user_json=(
                    '{"refresh_token":"private-google-token"}'
                ),
                granted_scopes=tuple(scopes),
                client_id="example.apps.googleusercontent.com",
            )
        ),
    )
    monkeypatch.setattr(
        "zippergen.google_auth.check_google_authorization",
        lambda value, *, scopes: value,
    )
    monkeypatch.setattr(
        "zippergen.google_gmail.GmailMailbox.inspect",
        lambda self: {"email": "calls@example.com"},
    )
    monkeypatch.setattr(
        "zippergen.google_sheets.GoogleSheetsTable.inspect",
        lambda self: {
            "title": "Call records",
            "tab": self.tab,
            "tabs": [self.tab],
        },
    )

    studio.execute("connector setup")

    assert set(workspace.connector_binding_profile("workflow.py:sample")) == {
        "mailbox",
        "records",
    }
    assert workspace.connector_configurations()["mailbox"]["kind"] == "gmail"
    assert (
        workspace.connector_configurations()["mailbox"]["query"]
        == "is:unread label:Calls"
    )
    assert (
        workspace.connector_configurations()["records"]["kind"]
        == "google-sheets"
    )
    from zippergen.google_auth import (
        GOOGLE_GMAIL_READONLY_SCOPE,
        GOOGLE_SHEETS_SCOPE,
    )

    assert set(requested_scopes) == {
        GOOGLE_GMAIL_READONLY_SCOPE,
        GOOGLE_SHEETS_SCOPE,
    }
    current, workflow, module = studio._current_context()
    environment = studio._workflow_connector_environment(
        workflow_spec=current,
        workflow=workflow,
        module=module,
    )
    snapshot = json.loads(environment["ZIPPERGEN_CONNECTORS_JSON"])
    mailbox = snapshot["requirement:mailbox"]
    records = snapshot["requirement:records"]
    assert mailbox["kind"] == "gmail"
    assert mailbox["access"] == "read-only"
    assert mailbox["query"] == "is:unread label:Calls"
    assert records["kind"] == "google-sheets"
    assert records["access"] == "write"
    assert environment[mailbox["credential_env"]] == (
        '{"refresh_token":"private-google-token"}'
    )


def test_connector_removal_names_deployments_that_keep_snapshots(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    workspace.save_connector_configuration(
        "old-route",
        {
            "kind": "telegram",
            "provider": "telegram",
            "chat_id": "123",
        },
    )
    deployments = workspace.home / "deployments"
    deployments.mkdir(parents=True)
    (deployments / "review-demo.json").write_text(
        json.dumps(
            {
                "name": "review-demo",
                "project_root": str(workspace.root),
                "store": str(workspace.home / "runs" / "review-demo.sqlite"),
                "connectors": {
                    "human:Reviewer": {
                        "type": "human",
                        "configuration": "old-route",
                    }
                },
            }
        )
    )

    studio.execute("connector config remove old-route")

    assert "old-route" not in workspace.connector_configurations()
    assert any(
        "Existing deployments keep their private connector snapshots: "
        "review-demo" in line
        for line in output
    )


def test_connector_setup_keeps_two_sheet_requirements_independent(
    tmp_path,
    monkeypatch,
):
    credentials = tmp_path / "google-desktop.json"
    credentials.write_text(GOOGLE_DESKTOP_CLIENT)
    studio, workspace, output = _studio(
        tmp_path,
        responses=[
            str(credentials),
            "sheet-source",
            "Source",
            "1",
            "sheet-target",
            "Dashboard",
        ],
    )
    (workspace.root / "workflow.py").write_text(TWO_SHEETS_SOURCE)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    monkeypatch.setattr(
        studio,
        "_needs_remote_google_browser",
        lambda: False,
    )
    monkeypatch.setattr(
        "zippergen.google_auth.authorize_google_client_result",
        lambda value, *, scopes: __import__(
            "zippergen.google_auth", fromlist=["GoogleAuthorization"]
        ).GoogleAuthorization(
            authorized_user_json=(
                '{"refresh_token":"private-google-token"}'
            ),
            granted_scopes=tuple(scopes),
            client_id="example.apps.googleusercontent.com",
        ),
    )
    monkeypatch.setattr(
        "zippergen.google_auth.check_google_authorization",
        lambda value, *, scopes: value,
    )
    monkeypatch.setattr(
        "zippergen.google_sheets.GoogleSheetsTable.inspect",
        lambda self: {
            "title": self.spreadsheet_id,
            "tab": self.tab,
            "tabs": [self.tab],
        },
    )

    studio.execute("connector setup")

    assert workspace.connector_binding_profile(
        "workflow.py:sample"
    ) == {
        "source-catalog": "source-catalog",
        "target-dashboard": "target-dashboard",
    }
    configurations = workspace.connector_configurations()
    assert configurations["source-catalog"]["spreadsheet_id"] == (
        "sheet-source"
    )
    assert configurations["source-catalog"]["tab"] == "Source"
    assert configurations["target-dashboard"]["spreadsheet_id"] == (
        "sheet-target"
    )
    assert configurations["target-dashboard"]["tab"] == "Dashboard"
    assert any(
        "Resource for target-dashboard" in line for line in output
    )
    assert all("call-intake" not in line.casefold() for line in output)


def test_studio_receives_remote_google_authorization_as_hidden_handoff(
    tmp_path,
    monkeypatch,
):
    from zippergen.google_auth import (
        GOOGLE_SHEETS_SCOPE,
        GoogleAuthorization,
        encode_google_authorization,
    )

    handoff = encode_google_authorization(
        GoogleAuthorization(
            authorized_user_json=(
                '{"client_id":"example.apps.googleusercontent.com",'
                '"refresh_token":"private-google-token"}'
            ),
            granted_scopes=(GOOGLE_SHEETS_SCOPE,),
            client_id="example.apps.googleusercontent.com",
        )
    )
    studio, workspace, output = _studio(
        tmp_path,
        secret_responses=[handoff],
    )
    secret_prompts: list[str] = []

    def paste_handoff(prompt: str) -> str:
        secret_prompts.append(prompt)
        return handoff

    studio.secret_input = paste_handoff
    # Batch Studio commands do not use prompt_toolkit. SSH detection must
    # still select the local authorization handoff instead of run_local_server.
    assert studio._prompt_toolkit_enabled is False
    monkeypatch.setenv("SSH_CONNECTION", "local remote")
    monkeypatch.setattr(
        "zippergen.google_auth.check_google_authorization",
        lambda value, *, scopes: value,
    )

    studio.execute("connector provider configure google")

    assert workspace.connector_provider_secret(
        "google", "authorized_user_json"
    ) == (
        '{"client_id":"example.apps.googleusercontent.com",'
        '"refresh_token":"private-google-token"}'
    )
    assert any(
        line == (
            "uvx --from 'zippergen[google] @ "
            "git+https://github.com/zippergen-io/zippergen.git@main' "
            "zippergen connector authorize google --scopes spreadsheets"
        )
        for line in output
    )
    assert any(
        line == "1. Open a terminal on your own computer (not this server)."
        for line in output
    )
    assert any(
        line == "3. Copy that whole zg-google-v1... line and return here."
        for line in output
    )
    assert secret_prompts == ["Paste the complete zg-google-v1... line: "]
    assert all("private-client-secret" not in line for line in output)
    assert all("private-google-token" not in line for line in output)
    profile = workspace.connector_provider_profiles()["google"]
    assert "authorization_ssh_host" not in profile
    assert json.loads(profile["granted_scopes"]) == [GOOGLE_SHEETS_SCOPE]


def test_google_scope_drift_blocks_runtime_context_before_resource_access(
    tmp_path,
):
    from zippergen.google_auth import (
        GOOGLE_SHEETS_READONLY_SCOPE,
        GOOGLE_SHEETS_SCOPE,
    )

    studio, workspace, _output = _studio(tmp_path)
    (workspace.root / "workflow.py").write_text(GOOGLE_SHEETS_SOURCE)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_connector_provider_profile(
        "google",
        {
            "kind": "google",
            "scopes": json.dumps([GOOGLE_SHEETS_SCOPE]),
            "granted_scopes": json.dumps(
                [GOOGLE_SHEETS_READONLY_SCOPE]
            ),
            "check_status": "available",
        },
    )
    workspace.save_connector_provider_secret(
        "google",
        "authorized_user_json",
        '{"refresh_token":"private-google-token"}',
    )
    workspace.save_connector_configuration(
        "call-records",
        {
            "provider": "google",
            "kind": "google-sheets",
            "spreadsheet_id": "sheet-123",
            "tab": "Calls",
            "check_status": "available",
        },
    )
    workspace.bind_connector(
        "workflow.py:sample",
        "call-records",
        "call-records",
    )
    current, workflow, module = studio._current_context()

    with pytest.raises(SystemExit, match="missing spreadsheets"):
        studio._workflow_connector_environment(
            workflow_spec=current,
            workflow=workflow,
            module=module,
        )


def test_studio_human_connector_assignment_needs_no_extra_requirement(
    tmp_path,
):
    studio, workspace, output = _studio(tmp_path)
    (workspace.root / "workflow.py").write_text(HUMAN_SOURCE)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_connector_provider_profile(
        "telegram",
        {"kind": "telegram", "check_status": "available"},
    )
    workspace.save_connector_configuration(
        "approvals",
        {
            "provider": "telegram",
            "kind": "telegram",
            "chat_id": "123",
            "check_status": "available",
        },
    )

    studio.execute("connector assign Human approvals")

    assert workspace.connector_assignment_profile(
        "workflow.py:sample"
    )["lifelines"] == {"Human": "approvals"}
    assert any("Human" in line and "approvals" in line for line in output)


def test_studio_no_longer_exposes_manual_deploy_notify(
    tmp_path,
):
    studio, _workspace, _output = _studio(tmp_path)
    assert "notify" not in _completions(studio, "deploy ")
    assert "connector" in _completions(studio, "con")


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

    monkeypatch.setattr("zippergen.studio_models.request.urlopen", fake_urlopen)

    studio.execute("model provider configure openai")
    studio.execute(
        "model provider configure local http://localhost:1234/v1"
    )
    studio.execute("model")

    assert workspace.load_secrets() == {"OPENAI_API_KEY": "super-secret-key"}
    assert workspace.provider_profiles()["local"]["base_url"] == (
        "http://localhost:1234/v1"
    )
    assert workspace.provider_profiles()["local"]["check_status"] == "reachable"
    assert workspace.provider_profiles()["local"]["model_count"] == "2"
    assert requests == [
        ("https://api.openai.com/v1/models", 3.0),
        ("http://localhost:1234/v1/models", 3.0),
    ]
    assert workspace.secrets_path.stat().st_mode & 0o077 == 0
    assert all("super-secret-key" not in line for line in output)
    assert any("openai" in line and "last check succeeded" in line for line in output)
    assert any(
        "local" in line
        and "http://localhost:1234/v1" in line
        and "2 models" in line
        for line in output
    )
    assert any(line == "Provider connections" for line in output)
    assert any(
        "model provider configure NAME" in line
        and "model provider check" in line
        for line in output
    )

    studio.execute("model provider remove openai")
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

    monkeypatch.setattr("zippergen.studio_models.request.urlopen", fail_urlopen)

    with pytest.raises(SystemExit, match="connection was not saved"):
        studio.execute(
            "model provider configure local http://localhost:9999/v1"
        )

    assert workspace.provider_profiles()["local"] == original


def test_studio_records_failed_local_configuration_check(tmp_path, monkeypatch):
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
    workspace.save_model_configuration(
        "local-reviewer",
        {
            "provider": "local",
            "model": "qwen2.5:7b",
            "spec": "local:qwen2.5:7b",
            "check_status": "not_checked",
        },
    )

    def fail_urlopen(req, *, timeout):
        raise URLError("connection refused")

    monkeypatch.setattr("zippergen.studio_models.request.urlopen", fail_urlopen)

    studio.execute("model config check local-reviewer")

    configuration = workspace.model_configurations()["local-reviewer"]
    assert configuration["check_status"] == "unverified"
    assert "connection refused" in configuration["check_detail"]
    output.clear()
    studio.execute("model")
    assert any(
        "local-reviewer" in line and "unverified" in line
        for line in output
    )


def test_studio_retires_the_public_store_namespace(tmp_path):
    studio, _workspace, _output = _studio(tmp_path)

    with pytest.raises(SystemExit, match="not a Studio command"):
        studio.execute("store list")


def test_studio_guides_approval_in_the_current_run(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    prompts: list[str] = []
    studio.input = lambda prompt: prompts.append(prompt) or "y"
    run = workspace.new_run(
        workflow_spec="workflow.py:sample",
        workflow_name="sample",
        fingerprint="fingerprint",
        inputs={},
        llm="mock",
    )
    store_path = Path(str(run["store"]))
    from zippergen.store import ensure_human_task, load_human_task, open_store

    connection = open_store(str(store_path))
    ensure_human_task(
        connection,
        task_id="review-1",
        role="HumanApprover",
        locator=[0],
        action="approve_answer",
        input_hash=None,
        inputs={"draft": "candidate"},
        spec={
            "kind": "confirm",
            "output": "approved",
            "output_type": "bool",
            "submit_label": "Approve",
            "cancel_label": "Revise",
            "rendered": {
                "instruction": "Approve this draft?",
                "context": (
                    "Draft:\nCandidate answer\n\n"
                    "Automated reviewer notes:\nMissing citation."
                ),
                "prefill": None,
            },
        },
    )
    connection.close()

    studio.execute("run tasks")
    studio.execute("run approve")

    connection = open_store(str(store_path))
    task = load_human_task(connection, "review-1")
    connection.close()
    assert task is not None
    assert task["status"] == "done"
    assert task["result"] == {"approved": True}
    assert sum(line == "Human decision" for line in output) == 2
    assert sum("Approve this draft?" in line for line in output) == 2
    assert sum("Candidate answer" in line for line in output) == 2
    assert sum("Missing citation." in line for line in output) == 2
    assert prompts == ["Decision — Approve or Revise? [y/n]: "]
    context_index = max(
        index for index, line in enumerate(output) if "Missing citation." in line
    )
    completed_index = next(
        index
        for index, line in enumerate(output)
        if "Completed human task review-1" in line
    )
    assert context_index < completed_index
    assert any("Completed human task review-1" in line for line in output)


def test_studio_run_trace_renders_the_persisted_event(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    run = workspace.new_run(
        workflow_spec="workflow.py:sample",
        workflow_name="sample",
        fingerprint="fingerprint",
        inputs={},
        llm="mock",
    )
    store_path = Path(str(run["store"]))
    from zippergen.store import open_store, record_trace_event

    connection = open_store(str(store_path))
    record_trace_event(
        connection,
        "Writer",
        {
            "type": "send",
            "from": "Writer",
            "to": "Reviewer",
            "channel": "draft",
            "values": ["candidate"],
        },
    )
    connection.close()

    output.clear()
    studio.execute("run trace")

    assert any(
        "Writer send Writer->Reviewer draft" in line for line in output
    )


def test_studio_deployment_list_includes_an_expected_missing_state(
    tmp_path,
):
    studio, workspace, output = _studio(tmp_path)
    deployments = workspace.home / "deployments"
    deployments.mkdir(parents=True)
    store = workspace.home / "runs" / "reviewed-answer.sqlite"
    (deployments / "reviewed-answer.json").write_text(
        json.dumps(
            {
                "name": "reviewed-answer",
                "project_root": str(workspace.root),
                "workflow": "workflow.py:sample",
                "cwd": str(workspace.root),
                "store": str(store),
                "log": str(workspace.home / "logs" / "reviewed-answer.log"),
            }
        )
    )

    studio.execute("deploy list")

    assert any(
        "Selected" in line and "Deployment" in line
        for line in output
    )
    assert any(
        "reviewed-answer" in line
        and "missing" in line
        for line in output
    )


def test_studio_deploy_remove_archives_the_deployment_and_clears_selection(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(workspace.home))
    deployments = workspace.home / "deployments"
    deployments.mkdir(parents=True)
    store = workspace.home / "runs" / "reviewed-answer.sqlite"
    store.parent.mkdir(parents=True)
    store.write_bytes(b"sqlite")
    profile = {
        "name": "reviewed-answer",
        "project_root": str(workspace.root),
        "workflow": "workflow.py:sample",
        "store": str(store),
        "log": str(workspace.home / "logs" / "reviewed-answer.log"),
    }
    (deployments / "reviewed-answer.json").write_text(json.dumps(profile))
    workspace.update(
        last_deployment="reviewed-answer",
    )
    monkeypatch.setattr(
        "zippergen.serve._deployment_service_status",
        lambda _name: {
            "state": "not-loaded",
            "detail": "service is not installed",
        },
    )
    monkeypatch.setattr(
        "zippergen.studio_deployments.unregister_deployment_service",
        lambda _name: "service was not installed",
    )

    studio.execute("deploy remove reviewed-answer --yes")

    assert not (deployments / "reviewed-answer.json").exists()
    assert not store.exists()
    archives = list(
        (workspace.home / "trash" / "deployments").iterdir()
    )
    assert len(archives) == 1
    assert (archives[0] / "profile/deployment.json").exists()
    state = workspace.load()
    assert state["last_deployment"] is None
    assert "current_store" not in state
    assert any(
        "Deployment removed from active use: reviewed-answer" in line
        for line in output
    )
    rendered = "\n".join(output).replace("\n", "")
    assert "Archive" in rendered
    assert archives[0].name in rendered


def test_studio_deploy_remove_purge_requires_a_name_and_deletes_permanently(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(workspace.home))
    deployments = workspace.home / "deployments"
    deployments.mkdir(parents=True)
    profile = {
        "name": "reviewed-answer",
        "project_root": str(workspace.root),
        "workflow": "workflow.py:sample",
        "store": str(workspace.home / "runs" / "reviewed-answer.sqlite"),
        "log": str(workspace.home / "logs" / "reviewed-answer.log"),
    }
    (deployments / "reviewed-answer.json").write_text(json.dumps(profile))
    workspace.update(last_deployment="reviewed-answer")
    monkeypatch.setattr(
        "zippergen.serve._deployment_service_status",
        lambda _name: {
            "state": "not-loaded",
            "detail": "service is not installed",
        },
    )
    monkeypatch.setattr(
        "zippergen.studio_deployments.unregister_deployment_service",
        lambda _name: "service was not installed",
    )

    with pytest.raises(
        SystemExit,
        match="Permanent removal requires an explicit deployment name",
    ):
        studio.execute("deploy remove --purge")

    studio.execute("deploy remove reviewed-answer --purge --yes")

    assert not (deployments / "reviewed-answer.json").exists()
    trash = workspace.home / "trash" / "deployments"
    assert trash.is_dir()
    assert list(trash.iterdir()) == []
    assert any(
        "Deployment permanently purged: reviewed-answer" in line
        for line in output
    )
    assert any("none; deletion was permanent" in line for line in output)


def test_studio_deployment_list_hides_another_projects_state(
    tmp_path,
):
    studio, workspace, output = _studio(tmp_path)
    deployments = workspace.home / "deployments"
    deployments.mkdir(parents=True)
    other_store = workspace.home / "runs" / "other.sqlite"
    (deployments / "other.json").write_text(
        json.dumps(
            {
                "name": "other",
                "project_root": str(tmp_path / "another-project"),
                "workflow": "workflow.py:other",
                "cwd": str(tmp_path / "another-project"),
                "store": str(other_store),
                "log": str(workspace.home / "logs" / "other.log"),
            }
        )
    )

    studio.execute("deploy list")

    assert all("other.sqlite" not in line for line in output)
    assert all("workflow.py:other" not in line for line in output)


def test_studio_deployment_show_separates_service_run_and_store(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(workspace.home))
    deployments = workspace.home / "deployments"
    deployments.mkdir(parents=True)
    bundle = workspace.home / "apps" / "reviewed-answer" / "version"
    bundle.mkdir(parents=True)
    log = workspace.home / "logs" / "reviewed-answer.log"
    log.parent.mkdir(parents=True)
    log.write_text("RuntimeError: MISTRAL_API_KEY is not set.\n")
    profile = {
        "name": "reviewed-answer",
        "project_root": str(workspace.root),
        "workflow": "workflow.py:sample",
        "cwd": str(bundle),
        "bundle": str(bundle),
        "store": str(workspace.home / "runs" / "reviewed-answer.sqlite"),
        "log": str(log),
        "llm": "mock",
        "llms": {"Reviewer": "mistral:mistral-small-latest"},
    }
    (deployments / "reviewed-answer.json").write_text(json.dumps(profile))
    monkeypatch.setattr(
        "zippergen.serve._deployment_service_status",
        lambda _name: {
            "state": "restarting",
            "detail": "loaded but not running; last exit code 1",
        },
    )
    monkeypatch.setattr("zippergen.serve._doctor_checks", lambda *a, **k: [])

    studio.execute("deploy list")

    assert any(line == "Deployments" for line in output)
    assert any("reviewed-answer" in line for line in output)
    output.clear()
    studio.execute("deploy show reviewed-answer")

    assert any(line == "Deployment state" for line in output)
    assert any("Bundle" in line and "installed" in line for line in output)
    assert any("Service" in line and "last exit code 1" in line for line in output)
    assert any("Boot" in line and "startup" in line for line in output)
    assert any("Run" in line and "deployment store is missing" in line for line in output)
    assert any("Store" in line and "missing" in line for line in output)
    assert any("Cause" in line and "MISTRAL_API_KEY is not set" in line for line in output)


def test_studio_deployment_show_separates_ready_store_from_starting_run(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(workspace.home))
    deployments = workspace.home / "deployments"
    deployments.mkdir(parents=True)
    bundle = workspace.home / "apps" / "reviewed-answer" / "version"
    bundle.mkdir(parents=True)
    store = workspace.home / "runs" / "reviewed-answer.sqlite"
    store.parent.mkdir(parents=True)
    from zippergen.store import open_store

    connection = open_store(str(store))
    connection.close()
    profile = {
        "name": "reviewed-answer",
        "project_root": str(workspace.root),
        "workflow": "workflow.py:sample",
        "cwd": str(bundle),
        "bundle": str(bundle),
        "store": str(store),
        "log": str(workspace.home / "logs" / "reviewed-answer.log"),
        "llm": "mock",
        "llms": {},
    }
    (deployments / "reviewed-answer.json").write_text(json.dumps(profile))
    monkeypatch.setattr(
        "zippergen.serve._deployment_service_status",
        lambda _name: {
            "state": "running",
            "detail": "service is running",
        },
    )
    monkeypatch.setattr("zippergen.serve._doctor_checks", lambda *a, **k: [])

    studio.execute("deploy show reviewed-answer")

    assert any(
        "Run" in line and "starting, no durable events recorded yet" in line
        for line in output
    )
    assert any(
        "Store" in line and "ready" in line and "no run data yet" in line
        for line in output
    )


def test_studio_deployment_show_labels_old_log_error_as_historical(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(workspace.home))
    deployments = workspace.home / "deployments"
    deployments.mkdir(parents=True)
    bundle = workspace.home / "apps" / "reviewed-answer" / "version"
    bundle.mkdir(parents=True)
    log = workspace.home / "logs" / "reviewed-answer.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        "ValueError: Local model 'local:qwen2.5:14b' has conflicting "
        "idle release policies.\n"
    )
    profile = {
        "name": "reviewed-answer",
        "project_root": str(workspace.root),
        "workflow": "workflow.py:sample",
        "cwd": str(bundle),
        "bundle": str(bundle),
        "store": str(workspace.home / "runs" / "reviewed-answer.sqlite"),
        "log": str(log),
        "llm": "mock",
        "llms": {},
        "zippergen_runtime": {
            "kind": "source-checkout",
            "version": "0.1.0a2",
            "revision": "c0089d5f00d123456789",
        },
    }
    (deployments / "reviewed-answer.json").write_text(json.dumps(profile))
    monkeypatch.setattr(
        "zippergen.serve._deployment_service_status",
        lambda _name: {
            "state": "running",
            "detail": "service is running",
        },
    )
    monkeypatch.setattr("zippergen.serve._doctor_checks", lambda *a, **k: [])

    studio.execute("deploy show reviewed-answer")

    assert any(
        "Runtime" in line and "c0089d5f00d1" in line
        for line in output
    )
    assert any(
        "Cause" in line and "no immediate failure detected" in line
        for line in output
    )
    assert any(
        "Previous failure" in line
        and "historical log entry" in line
        for line in output
    )


def test_deployment_log_cause_uses_current_generation_boundary(tmp_path):
    log = tmp_path / "deployment.log"
    old = "ValueError: old failure\n"
    log.write_text(old + "service started\nRuntimeError: current failure\n")

    assert Studio._deployment_log_cause(
        {
            "log": str(log),
            "log_generation_offset": len(old.encode()),
        }
    ) == "RuntimeError: current failure"
    assert (
        Studio._deployment_log_cause(
            {
                "log": str(log),
                "log_generation_offset": log.stat().st_size,
            }
        )
        is None
    )


def test_studio_resets_deployment_log_without_touching_the_service(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path, responses=("yes",))
    monkeypatch.setenv("ZIPPERGEN_HOME", str(workspace.home))
    deployments = workspace.home / "deployments"
    deployments.mkdir(parents=True)
    log = workspace.home / "logs" / "reviewed-answer.log"
    log.parent.mkdir(parents=True)
    old = b"older generation\n"
    current = b"current failure\ncurrent retry\n"
    log.write_bytes(old + current)
    profile_path = deployments / "reviewed-answer.json"
    profile_path.write_text(
        json.dumps(
            {
                "name": "reviewed-answer",
                "project_root": str(workspace.root),
                "workflow": "workflow.py:sample",
                "cwd": str(workspace.root),
                "store": str(
                    workspace.home / "runs" / "reviewed-answer.sqlite"
                ),
                "log": str(log),
                "log_generation_offset": len(old),
            }
        )
    )
    workspace.update(last_deployment="reviewed-answer")

    studio.execute("deploy logs reset")

    updated = json.loads(profile_path.read_text())
    archives = list(
        (workspace.home / "trash" / "deployment-logs").glob(
            "reviewed-answer-*.log"
        )
    )
    assert updated["log_generation_offset"] == len(old + current)
    assert len(archives) == 1
    assert archives[0].read_bytes() == current
    assert log.read_bytes() == old + current
    assert any(
        "Service" in line and "no stop or restart" in line
        for line in output
    )
    assert any(
        "Current history" in line and "empty" in line
        for line in output
    )


def test_studio_shows_and_compacts_deployment_storage(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(workspace.home))
    deployments = workspace.home / "deployments"
    deployments.mkdir(parents=True)
    store = workspace.home / "runs" / "reviewed-answer.sqlite"
    store.parent.mkdir(parents=True)
    log = workspace.home / "logs" / "reviewed-answer.log"
    log.parent.mkdir(parents=True)
    log.write_text("old deployment output\n")
    from zippergen.store import open_store, record_trace_event

    connection = open_store(str(store))
    connection.execute(
        "INSERT INTO events(sender,receiver,channel,kind,payload) "
        "VALUES('Writer',NULL,NULL,'seed','{}')"
    )
    for index in range(3):
        record_trace_event(
            connection,
            "Writer",
            {"type": "step", "index": index},
        )
    connection.close()
    profile = {
        "name": "reviewed-answer",
        "project_root": str(workspace.root),
        "workflow": "workflow.py:sample",
        "cwd": str(workspace.root),
        "store": str(store),
        "log": str(log),
        "recovery_compaction_version": 1,
        "trace_retention_version": 1,
    }
    (deployments / "reviewed-answer.json").write_text(json.dumps(profile))
    workspace.update(last_deployment="reviewed-answer")
    service = {
        "state": "not-loaded",
        "detail": "service is stopped",
    }
    monkeypatch.setattr(
        "zippergen.serve._deployment_service_status",
        lambda _name: service,
    )
    monkeypatch.setattr(
        "zippergen.studio_deployments._deployment_service_status",
        lambda _name: service,
    )

    studio.execute("deploy storage reviewed-answer")

    assert any(line == "Deployment storage" for line in output)
    assert any("Without snapshot" in line and "Writer" in line for line in output)
    assert any("trace" in line and "3" in line for line in output)
    assert any(
        "Task audit" in line and "retained by design" in line
        for line in output
    )
    output.clear()

    studio.execute("deploy storage compact reviewed-answer --yes")

    connection = open_store(str(store))
    assert connection.execute(
        "SELECT COUNT(*) FROM events WHERE kind='trace'"
    ).fetchone()[0] == 3
    connection.close()
    assert log.read_bytes() == b""
    assert any("Log archived" in line for line in output)


def test_studio_requires_redeploy_before_compacting_an_older_bundle(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(workspace.home))
    deployments = workspace.home / "deployments"
    deployments.mkdir(parents=True)
    store = workspace.home / "runs" / "legacy.sqlite"
    store.parent.mkdir(parents=True)
    from zippergen.store import open_store

    open_store(str(store)).close()
    (deployments / "legacy.json").write_text(
        json.dumps(
            {
                "name": "legacy",
                "workflow": "workflow.py:sample",
                "cwd": str(workspace.root),
                "store": str(store),
                "log": str(workspace.home / "logs" / "legacy.log"),
            }
        )
    )

    studio.execute("deploy storage legacy")
    assert any(
        "Trace retention" in line
        and "redeploy once to enable" in line
        for line in output
    )

    with pytest.raises(SystemExit, match="Redeploy it once"):
        studio.execute("deploy storage compact legacy --yes")


def test_studio_operates_human_tasks_through_the_deployment(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(workspace.home))
    deployments = workspace.home / "deployments"
    deployments.mkdir(parents=True)
    bundle = workspace.home / "apps" / "reviewed-answer" / "version"
    bundle.mkdir(parents=True)
    (bundle / "dual.py").write_text(TWO_LLM_PARTICIPANT_SOURCE)
    store = workspace.home / "runs" / "reviewed-answer.sqlite"
    store.parent.mkdir(parents=True)
    log = workspace.home / "logs" / "reviewed-answer.log"
    profile = {
        "name": "reviewed-answer",
        "project_root": str(workspace.root),
        "workflow": "dual.py:sample",
        "cwd": str(bundle),
        "bundle": str(bundle),
        "store": str(store),
        "log": str(log),
        "llm": "mock",
        "llms": {
            "Writer": "local:qwen2.5:7b",
            "Reviewer": "mistral:mistral-small-latest",
        },
    }
    (deployments / "reviewed-answer.json").write_text(json.dumps(profile))

    from zippergen.locator import resolve_path, statement_node_paths
    from zippergen.projection import project
    from zippergen.serve import load_workflow_spec
    from zippergen.store import (
        ensure_human_task,
        load_human_task,
        open_store,
        record_trace_event,
        write_execution_state,
    )
    from zippergen.syntax import ActStmt, _ordered_workflow_lifelines

    connection = open_store(str(store))
    ensure_human_task(
        connection,
        task_id="review-1",
        role="HumanApprover",
        locator=[0],
        action="approve_answer",
        input_hash=None,
        inputs={
            "draft": "Candidate answer",
            "concerns": "Missing citation.",
        },
        spec={
            "kind": "confirm",
            "output": "approved",
            "output_type": "bool",
            "submit_label": "Approve",
            "cancel_label": "Revise",
            "rendered": {
                "instruction": "Approve this draft?",
                "context": (
                    "Draft:\nCandidate answer\n\n"
                    "Automated reviewer notes:\nMissing citation."
                ),
                "prefill": None,
            },
        },
    )
    deployed_workflow, _module = load_workflow_spec(
        str(bundle / "dual.py") + ":sample"
    )
    reviewer = next(
        item
        for item in _ordered_workflow_lifelines(deployed_workflow)
        if item.name == "Reviewer"
    )
    reviewer_local = project(deployed_workflow, reviewer)
    reviewer_action = next(
        path
        for path in statement_node_paths(reviewer_local).values()
        if isinstance(resolve_path(reviewer_local, path), ActStmt)
    )
    write_execution_state(
        connection,
        "Reviewer",
        "running_model",
        [reviewer_action],
        {"action": "process", "kind": "model"},
    )
    record_trace_event(
        connection,
        "Reviewer",
        {
            "type": "act",
            "role": "Reviewer",
            "action": "process",
        },
    )
    connection.close()
    monkeypatch.setattr(
        "zippergen.serve._deployment_service_status",
        lambda _name: {
            "state": "running",
            "detail": "service is running",
        },
    )
    monkeypatch.setattr("zippergen.serve._doctor_checks", lambda *a, **k: [])

    studio.execute("deploy show reviewed-answer")

    assert any(
        "Store" in line and "ready" in line and "pending" in line
        for line in output
    )
    models = next(line for line in output if "Models" in line)
    assert "Writer.process=local:qwen2.5:7b" in models
    assert "Reviewer.process=mistral:mistral-small-latest" in models
    assert "mock" not in models
    assert any("deploy tasks" in line for line in output)
    assert _completions(studio, "deploy tasks r") == [
        "reviewed-answer"
    ]

    output.clear()
    studio.execute("current")
    assert not any(
        "Deployment" in line and "reviewed-answer" in line
        for line in output
    )
    assert not any("store" in line.casefold() for line in output)

    output.clear()
    studio.execute("deploy tasks reviewed-answer")

    assert any("Candidate answer" in line for line in output)
    assert any("Missing citation." in line for line in output)
    assert any("deploy approve reviewed-answer" in line for line in output)
    assert workspace.load()["last_deployment"] == "reviewed-answer"

    output.clear()
    studio.execute("deploy trace reviewed-answer")
    assert any("Reviewer act action process" in line for line in output)

    output.clear()
    studio.execute("deploy inspect reviewed-answer Reviewer")
    assert any(
        "Reviewer" in line and "running model action" in line
        for line in output
    )
    assert any(
        "▶" in line and "result = process(draft)" in line
        for line in output
    )

    watched: list[tuple[str, str]] = []

    def render_watched_deployment(
        render_once,
        *,
        command: str,
        subject: str,
    ) -> None:
        watched.append((command, subject))
        render_once()

    monkeypatch.setattr(
        studio,
        "_watch_execution",
        render_watched_deployment,
    )
    output.clear()
    studio.execute("deploy inspect reviewed-answer Reviewer --watch")

    assert watched == [
        (
            "deploy inspect reviewed-answer Reviewer --watch",
            "deployment",
        )
    ]
    assert any(
        "Reviewer" in line and "running model action" in line
        for line in output
    )

    prompts: list[str] = []
    studio.input = lambda prompt: prompts.append(prompt) or "y"
    output.clear()
    studio.execute("deploy approve reviewed-answer")

    connection = open_store(str(store))
    task = load_human_task(connection, "review-1")
    connection.close()
    assert task is not None
    assert task["status"] == "done"
    assert task["result"] == {"approved": True}
    assert prompts == ["Decision — Approve or Revise? [y/n]: "]
    assert any("Candidate answer" in line for line in output)
    assert list((workspace.home / "runs").glob("*.sqlite")) == [store]


def test_studio_deploys_current_workflow_and_remembers_name(tmp_path, monkeypatch):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "zippergen.serve.main",
        lambda arguments: calls.append(arguments) or 0,
    )

    studio.deploy_workflow(["sample-test"])

    assert len(calls) == 1
    assert calls[0][:6] == [
        "deploy",
        str(workspace.root / "workflow.py") + ":sample",
        "--name",
        "sample-test",
        "--project-root",
        str(workspace.root),
    ]
    assert "--project-alignment-json" in calls[0]
    assert "--concise" in calls[0]
    assert ["--llm", "mock"] == calls[0][
        calls[0].index("--llm"):calls[0].index("--llm") + 2
    ]
    assert ["--assistant", "codex"] == calls[0][
        calls[0].index("--assistant"):calls[0].index("--assistant") + 2
    ]
    assert calls[0][-2:] == ["--llm-idle-timeouts-json", "{}"]
    assert workspace.load()["last_deployment"] == "sample-test"
    assert output[-1] == "✓ Deployment completed: sample-test"
    assert any(
        "No Studio acceptance is recorded" in line for line in output
    )


def test_studio_resume_uses_recorded_connector_routing(tmp_path, monkeypatch):
    studio, workspace, _output = _studio(tmp_path)
    snapshot = {
        "human:Writer.echo": {
            "type": "human",
            "target": "Writer.echo",
            "participant": "Writer",
            "action": "echo",
            "kind": "telegram",
            "provider": "telegram",
            "configuration": "old-route",
            "chat_id": "123",
            "channel": "telegram:old-route",
            "token_env": "ZIPPERGEN_CONNECTOR_TELEGRAM_TOKEN",
        }
    }
    workspace.save_connector_provider_secret(
        "telegram",
        "bot_token",
        "private-token",
    )
    run = workspace.new_run(
        workflow_spec="workflow.py:sample",
        workflow_name="sample",
        fingerprint="unused-by-mocked-run",
        inputs={"value": "hello"},
        llm="mock",
        connectors=snapshot,
    )
    workspace.update_run(str(run["run_id"]), status="waiting")
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "zippergen.studio.run_dev",
        lambda _workspace, **kwargs: calls.append(kwargs) or {},
    )

    studio.execute("resume")

    assert calls[0]["connector_environment"] == {
        "ZIPPERGEN_CONNECTORS_JSON": json.dumps(
            snapshot,
            sort_keys=True,
        ),
        "ZIPPERGEN_CONNECTOR_TELEGRAM_TOKEN": "private-token",
    }
    assert callable(calls[0]["human_connector_factory"])


def test_deployment_alignment_covers_review_and_configuration(tmp_path):
    from zippergen.semantic import semantic_snapshot
    from zippergen.serve import load_workflow_spec

    studio, workspace, _output = _studio(tmp_path)
    current = workspace.select_workflow(
        "workflow.py:sample",
        cwd=workspace.root,
    )
    workspace.specification_path.write_text("Create an echo workflow.\n")
    workflow, module = load_workflow_spec(
        workspace.absolute_spec(current)
    )
    semantics = semantic_snapshot(workflow, module)
    specification = workspace.specification()
    assert specification is not None
    specification_fingerprint = workspace.specification_fingerprint(
        include_pending=False
    )
    workspace.record_accepted_review(
        current,
        specification=specification,
        specification_fingerprint=specification_fingerprint,
        semantic_snapshot=semantics,
        request_id=None,
        accepted_source={"root": str(workspace.root)},
    )
    profile: dict[str, object] = {
        "project_root": str(workspace.root),
        "workflow": "workflow.py:sample",
        "llm": "mock",
        "llms": {},
        "llm_idle_timeouts": {},
        "assistant": "codex",
        "connectors": {},
        "environment": {},
        "project_alignment": {
            "schema_version": 1,
            "workflow_spec": current,
            "specification_fingerprint": specification_fingerprint,
            "semantic_fingerprint": (
                studio._semantic_snapshot_fingerprint(semantics)
            ),
            "review": "accepted",
        },
    }

    message, kind, changed = studio._deployment_project_alignment(profile)

    assert kind == "success"
    assert changed == ()
    assert "matches current specification" in message

    workspace.save_model_profile(
        current,
        default="openai:gpt-4o-mini",
        lifelines={},
    )
    message, kind, changed = studio._deployment_project_alignment(profile)

    assert kind == "warning"
    assert "models" in changed
    assert "Redeploy" in message


def test_studio_deploys_the_immutable_accepted_source_by_default(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    studio.create_request("Echo the request through Writer.")
    studio.execute("workflow accept --yes")
    accepted = workspace.accepted_review("workflow.py:sample")
    assert accepted is not None
    accepted_source = accepted["accepted_source"]
    assert isinstance(accepted_source, dict)
    accepted_root = Path(str(accepted_source["root"]))
    assert (accepted_root / "workflow.py").read_text() == (
        workspace.root / "workflow.py"
    ).read_text()
    output.clear()
    calls: list[tuple[list[str], Path]] = []

    def fake_main(arguments):
        calls.append((arguments, Path.cwd()))
        return 0

    monkeypatch.setattr("zippergen.serve.main", fake_main)

    studio.deploy_workflow(["sample-accepted", "--no-start"])

    assert calls[0][1] == accepted_root
    assert calls[0][0][0] == "deploy"
    assert str(accepted_root) in calls[0][0][1]
    assert any(
        "Source" in line and "immutable accepted source snapshot" in line
        for line in output
    )


def test_studio_divergent_deploy_can_use_the_accepted_version(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path, responses=["1"])
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    studio.create_request("Echo the request through Writer.")
    studio.execute("workflow accept --yes")
    accepted = workspace.accepted_review("workflow.py:sample")
    assert accepted is not None
    accepted_source = accepted["accepted_source"]
    assert isinstance(accepted_source, dict)
    accepted_root = Path(str(accepted_source["root"]))
    source = workspace.root / "workflow.py"
    source.write_text(
        source.read_text().replace(
            'system="Echo the value."',
            'system="Echo a changed value."',
        )
    )
    output.clear()
    calls: list[tuple[list[str], Path]] = []

    def fake_main(arguments):
        calls.append((arguments, Path.cwd()))
        return 0

    monkeypatch.setattr("zippergen.serve.main", fake_main)

    studio.deploy_workflow(["sample-accepted", "--no-start"])

    assert calls[0][1] == accepted_root
    assert str(accepted_root) in calls[0][0][1]
    assert "Accepted semantic workflow diff" in output
    assert any(
        "system_prompt" in line and "Echo a changed value." in line
        for line in output
    )


def test_studio_divergent_deploy_override_requires_and_records_a_reason(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    studio.create_request("Echo the request through Writer.")
    studio.execute("workflow accept --yes")
    source = workspace.root / "workflow.py"
    source.write_text(
        source.read_text().replace(
            'system="Echo the value."',
            'system="Emergency candidate behavior."',
        )
    )
    output.clear()
    calls: list[tuple[list[str], Path]] = []

    def fake_main(arguments):
        calls.append((arguments, Path.cwd()))
        return 0

    monkeypatch.setattr("zippergen.serve.main", fake_main)

    studio.deploy_workflow(
        [
            "sample-override",
            "--no-start",
            "--unreviewed",
            "--reason",
            "Emergency production correction",
        ]
    )

    assert calls[0][1] == workspace.root
    assert calls[0][0][1] == str(workspace.root / "workflow.py") + ":sample"
    overrides = workspace.load()["deployment_review_overrides"]
    assert isinstance(overrides, list)
    assert overrides[-1]["reason"] == "Emergency production correction"
    assert any(
        "Unaccepted deployment override recorded" in line for line in output
    )


def test_studio_blocks_a_never_accepted_generated_result_awaiting_review(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    studio.create_request("Echo the request through Writer.")
    request = workspace.current_request()
    assert request is not None
    workspace.update_request(
        str(request["request_id"]),
        status="awaiting_review",
    )
    monkeypatch.setattr(
        "zippergen.serve.main",
        lambda _arguments: pytest.fail("deployment must not start"),
    )

    with pytest.raises(
        SystemExit,
        match="human-review boundary",
    ):
        studio.deploy_workflow(["sample-blocked"])

    assert any(
        "Reason" in line and "awaiting human review" in line
        for line in output
    )


def test_studio_can_prepare_deployment_without_starting_it(tmp_path, monkeypatch):
    studio, workspace, _output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "zippergen.serve.main",
        lambda arguments: calls.append(arguments) or 0,
    )

    studio.deploy_workflow(["sample-test", "--no-start"])

    assert calls[0][:6] == [
        "deploy",
        str(workspace.root / "workflow.py") + ":sample",
        "--name",
        "sample-test",
        "--project-root",
        str(workspace.root),
    ]
    assert "--project-alignment-json" in calls[0]
    assert "--concise" in calls[0]
    assert "--no-start" in calls[0]
    assert calls[0][-2:] == ["--llm-idle-timeouts-json", "{}"]


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


def test_studio_reuses_a_selected_provider_key_without_a_declared_field(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path, responses=[""])
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_model_profile(
        "workflow.py:sample",
        default="mock",
        lifelines={"Writer": "mistral:mistral-small-latest"},
    )
    workspace.save_secrets({"MISTRAL_API_KEY": "private-mistral-key"})
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "zippergen.serve.main",
        lambda arguments: calls.append(arguments) or 0,
    )

    studio.deploy_workflow(["sample-mistral", "--no-start"])

    secret_index = calls[0].index("--provider-secret")
    assert calls[0][secret_index + 1] == (
        "MISTRAL_API_KEY=private-mistral-key"
    )
    assert "--set" not in calls[0]
    assert all("private-mistral-key" not in line for line in output)


def test_studio_carries_the_configured_local_endpoint_into_deployment(
    tmp_path,
    monkeypatch,
):
    studio, workspace, _output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_provider_profile(
        "local",
        {
            "kind": "local",
            "base_url": "http://127.0.0.1:11434/v1",
        },
    )
    workspace.save_model_profile(
        "workflow.py:sample",
        default="mock",
        lifelines={"Writer": "local:qwen2.5:7b"},
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "zippergen.serve.main",
        lambda arguments: calls.append(arguments) or 0,
    )

    studio.deploy_workflow(["sample-local", "--no-start"])

    environment_index = calls[0].index("--provider-env")
    assert calls[0][environment_index + 1] == (
        "OLLAMA_BASE_URL=http://127.0.0.1:11434/v1"
    )


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


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("start", ["start", "sample-test", "--enable"]),
        ("restart", ["restart", "sample-test"]),
    ],
)
def test_studio_operates_remembered_deployment(
    action,
    expected,
    tmp_path,
    monkeypatch,
):
    studio, workspace, _output = _studio(tmp_path)
    workspace.update(last_deployment="sample-test")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "zippergen.serve.main",
        lambda arguments: calls.append(arguments) or 0,
    )

    studio.deployment_action(action, [])

    assert calls == [expected]


def test_studio_run_accepts_an_llm_override(tmp_path, monkeypatch):
    studio, _workspace, _output = _studio(tmp_path)
    calls = []
    monkeypatch.setattr(
        studio,
        "_verify_model_spec",
        lambda label, spec, for_save=False: SimpleNamespace(
            kind="success",
            message=f"{label}: {spec} is available.",
        ),
    )
    monkeypatch.setattr(
        "zippergen.studio.run_dev",
        lambda workspace, **kwargs: calls.append(kwargs),
    )

    studio.execute("run openai:gpt-4o-mini")

    assert calls[0]["llm"] == "openai:gpt-4o-mini"
    assert calls[0]["renderer"] is studio._renderer


def test_studio_run_accepts_an_assistant_action_backend(tmp_path, monkeypatch):
    studio, _workspace, _output = _studio(tmp_path)
    calls = []
    monkeypatch.setattr(
        "zippergen.studio.run_dev",
        lambda workspace, **kwargs: calls.append(kwargs),
    )

    studio.execute("run mock --assistant claude")

    assert calls[0]["llm"] == "mock"
    assert calls[0]["assistant"] == "claude"


def test_studio_run_starts_assigned_human_connector_automatically(
    tmp_path,
    monkeypatch,
):
    studio, workspace, _output = _studio(tmp_path)
    (workspace.root / "workflow.py").write_text(HUMAN_SOURCE)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_connector_provider_profile(
        "telegram",
        {"kind": "telegram", "check_status": "available"},
    )
    workspace.save_connector_provider_secret(
        "telegram", "bot_token", "private-token"
    )
    workspace.save_connector_configuration(
        "approvals",
        {
            "provider": "telegram",
            "kind": "telegram",
            "chat_id": "123",
            "channel": "telegram:approvals",
            "check_status": "available",
        },
    )
    workspace.save_connector_assignment_profile(
        "workflow.py:sample",
        lifelines={"Human": "approvals"},
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        studio,
        "_check_connector_configuration",
        lambda name: name == "approvals",
    )
    monkeypatch.setattr(
        "zippergen.studio.run_dev",
        lambda workspace, **kwargs: calls.append(kwargs),
    )

    studio.execute("run")

    factory = calls[0]["human_connector_factory"]
    assert callable(factory)
    notifier = factory("/tmp/example.sqlite")
    assert notifier.assignments == {"Human": "approvals"}
    assert notifier.routes["approvals"]["chat_id"] == "123"


def test_studio_models_displays_connections_and_llm_active_lifelines(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_model_profile(
        "workflow.py:sample",
        default="mock",
        lifelines={"Writer": "openai:gpt-4o-mini"},
    )

    studio.execute("model")

    assert workspace.model_profile("workflow.py:sample") == {
        "default": "mock",
        "lifelines": {"Writer": "openai:gpt-4o-mini"},
    }
    assert any(line == "Provider connections" for line in output)
    assert any(
        "Writer" in line
        and "openai:gpt-4o-mini" in line
        for line in output
    )
    assert any(
        "openai" in line and "not configured" in line
        for line in output
    )


def test_studio_models_configure_check_then_assign(
    tmp_path,
    monkeypatch,
):
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

    monkeypatch.setattr("zippergen.studio_models.request.urlopen", fake_urlopen)

    studio, workspace, output = _studio(
        tmp_path,
        responses=["mistral", "mistral-small-latest"],
    )
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_provider_profile(
        "mistral",
        {"kind": "api", "key_env": "MISTRAL_API_KEY"},
    )
    workspace.save_secrets({"MISTRAL_API_KEY": "private-mistral-key"})

    studio.execute("model config create fast-review")
    assert requests == []
    assert workspace.model_configurations()["fast-review"]["check_status"] == (
        "not_checked"
    )

    studio.execute("model config check fast-review")
    studio.execute("model assign Writer fast-review")

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
        "fast-review:" in line
        and "is available with the configured mistral API key" in line
        for line in output
    )
    assert any(
        "Writer" in line and "fast-review" in line
        and "mistral:mistral-small-latest" in line
        for line in output
    )


def test_studio_local_model_configuration_includes_idle_release(tmp_path):
    studio, workspace, output = _studio(
        tmp_path,
        responses=["local", "qwen2.5:7b", "300"],
    )
    workspace.save_provider_profile(
        "local",
        {
            "kind": "local",
            "base_url": "http://localhost:11434/v1",
            "check_status": "reachable",
            "checked_at": "2026-07-28T10:00:00+0200",
            "model_count": "1",
        },
    )

    studio.execute("model config create local-writer")
    configuration = workspace.model_configurations()["local-writer"]

    assert configuration["spec"] == "local:qwen2.5:7b"
    assert configuration["idle_timeout"] == "300"
    output.clear()
    studio.execute("model config show local-writer")
    assert any(
        "Idle release" in line and "after 300 s (5 min)" in line
        for line in output
    )


def test_studio_guided_model_setup_labels_progress_and_each_selection(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    (workspace.root / "dual.py").write_text(TWO_LLM_PARTICIPANT_SOURCE)
    workspace.select_workflow("dual.py:sample", cwd=workspace.root)
    answers = iter(["1", "", "1", "1"])
    prompts: list[str] = []

    def answer(prompt: str) -> str:
        prompts.append(prompt)
        return next(answers)

    studio.input = answer
    studio.execute("model setup")

    assert prompts == [
        "Select provider [1-5]: ",
        "Configuration name (press Enter for an automatic name): ",
        "Select configuration for Writer [1-1]: ",
        "Select configuration for Reviewer [1-1]: ",
    ]
    assert any(
        "Current" in line
        and "Step 1 of 3" in line
        and "configure and check a provider" in line
        for line in output
    )
    assert any(
        "Current" in line
        and "Step 2 of 3" in line
        and "create and check a named configuration" in line
        for line in output
    )
    assert any(
        "Current" in line
        and "Step 3 of 3" in line
        and "assign configurations to LLM participants" in line
        for line in output
    )
    assert not any(
        line.strip().startswith("1") and "configure and check" in line
        for line in output
    )
    assert workspace.model_assignment_profile(
        "dual.py:sample",
        default="mock",
    )["lifelines"] == {
        "Writer": "mock",
        "Reviewer": "mock",
    }


def test_studio_models_rename_preserves_check_and_updates_all_assignments(
    tmp_path,
):
    studio, workspace, output = _studio(tmp_path)
    workspace.save_model_configuration(
        "fast-review",
        {
            "provider": "mistral",
            "model": "mistral-small-latest",
            "spec": "mistral:mistral-small-latest",
            "check_status": "available",
            "check_detail": "model is available",
            "checked_at": "2026-07-24T16:00:00+0200",
        },
    )
    workspace.save_model_assignment_profile(
        "workflow.py:sample",
        default="fast-review",
        lifelines={"Writer": "fast-review"},
    )
    workspace.save_model_assignment_profile(
        "other.py:other",
        default="mock",
        lifelines={"Reviewer": "fast-review"},
    )

    studio.execute("model config rename fast-review editorial")

    configurations = workspace.model_configurations()
    assert "fast-review" not in configurations
    assert configurations["editorial"]["spec"] == (
        "mistral:mistral-small-latest"
    )
    assert configurations["editorial"]["check_status"] == "available"
    assert configurations["editorial"]["checked_at"] == (
        "2026-07-24T16:00:00+0200"
    )
    profiles = workspace.load()["model_profiles"]
    assert profiles["workflow.py:sample"]["default_configuration"] == "editorial"
    assert profiles["workflow.py:sample"]["lifeline_configurations"] == {
        "Writer": "editorial"
    }
    assert profiles["other.py:other"]["lifeline_configurations"] == {
        "Reviewer": "editorial"
    }
    assert profiles["workflow.py:sample"]["default"] == (
        "mistral:mistral-small-latest"
    )
    assert any(line == "Model configuration renamed" for line in output)
    assert any("Check" in line and "available; preserved" in line for line in output)
    assert any(
        "workflow.py:sample" in line and "default, Writer" in line
        for line in output
    )
    assert any(
        "other.py:other" in line and "Reviewer" in line for line in output
    )


def test_studio_models_rename_rejects_mock_and_name_collisions(tmp_path):
    studio, workspace, _output = _studio(tmp_path)
    for name in ("first", "second"):
        workspace.save_model_configuration(
            name,
            {
                "provider": "local",
                "model": name,
                "spec": f"local:{name}",
            },
        )

    with pytest.raises(SystemExit, match="built-in mock.*cannot be renamed"):
        studio.execute("model config rename mock replacement")
    with pytest.raises(SystemExit, match="conflicts with existing"):
        studio.execute("model config rename first SECOND")

    assert {"first", "second"}.issubset(workspace.model_configurations())


def test_studio_model_configuration_guides_missing_provider_connection(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(
        tmp_path,
        responses=["anthropic", "claude-sonnet-4-6"],
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

    studio.execute("model provider configure anthropic")
    studio.execute("model config create anthropic-review")

    assert workspace.load_secrets()["ANTHROPIC_API_KEY"] == (
        "private-anthropic-key"
    )
    configuration = workspace.model_configurations()["anthropic-review"]
    assert configuration["spec"] == "anthropic:claude-sonnet-4-6"
    assert configuration["check_status"] == "not_checked"
    assert any("Configured anthropic" in line for line in output)
    assert all("private-anthropic-key" not in line for line in output)


def test_studio_model_assignment_requires_a_saved_configuration(
    tmp_path,
):
    studio, workspace, _output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)

    with pytest.raises(SystemExit, match="Unknown model configuration"):
        studio.execute("model assign Writer openai:gpt-4o-mini")

    assert workspace.model_profile("workflow.py:sample")["lifelines"] == {}


def test_studio_model_configuration_requires_a_configured_provider(tmp_path):
    studio, workspace, _output = _studio(
        tmp_path,
        responses=["anthropic"],
    )

    with pytest.raises(
        SystemExit,
        match=r"Provider 'anthropic' is not configured.*"
        r"model provider configure anthropic",
    ):
        studio.execute("model config create editorial")

    assert "editorial" not in workspace.model_configurations()
    assert workspace.provider_profiles() == {}
    assert workspace.load_secrets() == {}


def test_studio_configuration_is_reusable_without_serializing_calls(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    (workspace.root / "dual.py").write_text(TWO_LLM_PARTICIPANT_SOURCE)
    workspace.select_workflow("dual.py:sample", cwd=workspace.root)
    workspace.save_model_configuration(
        "shared-local",
        {
            "provider": "local",
            "model": "qwen3",
            "spec": "local:qwen3",
            "check_status": "available",
        },
    )

    studio.execute("model assign Writer shared-local")
    studio.execute("model assign Reviewer shared-local")

    assignments = workspace.model_assignment_profile("dual.py:sample")
    assert assignments["lifelines"] == {
        "Reviewer": "shared-local",
        "Writer": "shared-local",
    }
    assert sum(
        "shared-local" in line and participant in line
        for line in output
        for participant in ("Writer", "Reviewer")
    ) >= 2
    assert any(
        "calls remain independent and may run in parallel" in line
        for line in output
    )
    title = max(
        index
        for index, line in enumerate(output)
        if line == "Model assignments"
    )
    assert output[title + 2].split() == [
        "Participant",
        "LLM",
        "action",
        "Configuration",
        "Model",
        "Source",
        "Last",
        "check",
    ]
    assert set(output[title + 3].replace(" ", "")) == {"─"}
    assert any(
        line.startswith("✓ Assigned shared-local to Reviewer.")
        for line in output
    )


def test_studio_model_action_override_precedes_participant_assignment(
    tmp_path,
):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_model_configuration(
        "participant-model",
        {
            "provider": "local",
            "model": "qwen3",
            "spec": "local:qwen3",
            "check_status": "available",
        },
    )
    workspace.save_model_configuration(
        "action-model",
        {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "spec": "openai:gpt-4o-mini",
            "check_status": "available",
        },
    )

    studio.execute("model assign Writer participant-model")
    studio.execute("model assign Writer.echo action-model")

    assignments = workspace.model_assignment_profile(
        "workflow.py:sample"
    )
    assert assignments["lifelines"] == {
        "Writer": "participant-model"
    }
    assert assignments["actions"] == {
        "Writer.echo": "action-model"
    }
    profile = workspace.model_profile("workflow.py:sample")
    assert profile["lifelines"]["Writer"] == "local:qwen3"
    assert profile["actions"]["Writer.echo"] == "openai:gpt-4o-mini"
    assert any(
        "Writer" in line and "echo" in line and "action override" in line
        for line in output
    )


def test_studio_assignment_listing_is_cached_and_check_is_targeted(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path)
    (workspace.root / "dual.py").write_text(TWO_LLM_PARTICIPANT_SOURCE)
    workspace.select_workflow("dual.py:sample", cwd=workspace.root)
    workspace.save_model_configuration(
        "shared-local",
        {
            "provider": "local",
            "model": "qwen3",
            "spec": "local:qwen3",
            "check_status": "not_checked",
        },
    )
    workspace.save_model_configuration(
        "unused-mistral",
        {
            "provider": "mistral",
            "model": "mistral-small-latest",
            "spec": "mistral:mistral-small-latest",
            "check_status": "not_checked",
        },
    )
    workspace.save_model_assignment_profile(
        "dual.py:sample",
        default="mock",
        lifelines={
            "Writer": "shared-local",
            "Reviewer": "shared-local",
        },
    )
    checks: list[tuple[str, str]] = []

    def verify(label, spec, *, for_save=False):
        checks.append((label, spec))
        return SimpleNamespace(
            kind="success",
            message=f"{label}: {spec} is available.",
        )

    monkeypatch.setattr(studio, "_verify_model_spec", verify)

    studio.execute("model assignments")

    assert checks == []
    assert any("Last check" in line for line in output)
    assert any("never" in line for line in output)

    output.clear()
    studio.execute("model assignments check")

    assert checks == [("shared-local", "local:qwen3")]
    assert any(line == "Assignment checks" for line in output)
    assert any("All assigned models are reachable" in line for line in output)
    assert workspace.model_configurations()["shared-local"]["check_status"] == (
        "available"
    )
    assert workspace.model_configurations()["unused-mistral"]["check_status"] == (
        "not_checked"
    )


def test_studio_assignment_check_rejects_real_local_idle_policy_conflict(
    tmp_path,
    monkeypatch,
):
    studio, workspace, _output = _studio(tmp_path)
    (workspace.root / "dual.py").write_text(TWO_LLM_PARTICIPANT_SOURCE)
    workspace.select_workflow("dual.py:sample", cwd=workspace.root)
    for name, idle_timeout in (
        ("release-local", "300"),
        ("resident-local", None),
    ):
        configuration = {
            "provider": "local",
            "model": "qwen2.5:14b",
            "spec": "local:qwen2.5:14b",
            "check_status": "available",
        }
        if idle_timeout is not None:
            configuration["idle_timeout"] = idle_timeout
        workspace.save_model_configuration(name, configuration)
    workspace.save_model_assignment_profile(
        "dual.py:sample",
        default="release-local",
        lifelines={"Reviewer": "resident-local"},
    )
    checks: list[str] = []
    monkeypatch.setattr(
        studio,
        "_verify_model_spec",
        lambda label, spec, for_save=False: checks.append(label),
    )

    with pytest.raises(
        SystemExit,
        match=(
            "conflicting idle release policies: "
            "Writer=300 s, Reviewer=never"
        ),
    ):
        studio.execute("model assignments check")

    assert checks == []


def test_studio_models_inherit_removes_a_participant_assignment(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_model_profile(
        "workflow.py:sample",
        default="mock",
        lifelines={"Writer": "local:qwen2.5:7b"},
    )

    studio.execute("model inherit Writer")

    assert workspace.model_profile("workflow.py:sample")["lifelines"] == {}
    assert any(
        "Writer" in line and "echo" in line and "mock" in line
        and "default" in line
        for line in output
    )


def test_studio_models_check_updates_configuration_not_assignments(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_model_configuration(
        "review-model",
        {
            "provider": "mistral",
            "model": "mistral-small-latest",
            "spec": "mistral:mistral-small-latest",
            "check_status": "not_checked",
        },
    )
    workspace.save_provider_profile(
        "mistral",
        {"kind": "api", "key_env": "MISTRAL_API_KEY"},
    )
    workspace.save_secrets({"MISTRAL_API_KEY": "private-mistral-key"})
    workspace.save_model_assignment_profile(
        "workflow.py:sample",
        default="review-model",
        lifelines={},
    )
    before = workspace.model_assignment_profile("workflow.py:sample")
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

    monkeypatch.setattr("zippergen.studio_models.request.urlopen", fake_urlopen)

    studio.execute("model config check review-model")

    assert len(requests) == 1
    assert workspace.model_assignment_profile("workflow.py:sample") == before
    assert workspace.model_configurations()["review-model"]["check_status"] == (
        "available"
    )
    assert any(line == "Configuration checks" for line in output)
    assert any(
        "review-model:" in line
        and "is available with the configured mistral API key" in line
        for line in output
    )
    assert any("assignments unchanged" in line for line in output)
    assert all("private-mistral-key" not in line for line in output)


def test_studio_models_check_records_an_unavailable_configuration(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_model_configuration(
        "broken-reviewer",
        {
            "provider": "mistral",
            "model": "mistral-smol-latest",
            "spec": "mistral:mistral-smol-latest",
            "check_status": "not_checked",
        },
    )
    workspace.save_provider_profile(
        "mistral",
        {"kind": "api", "key_env": "MISTRAL_API_KEY"},
    )
    workspace.save_secrets({"MISTRAL_API_KEY": "private-mistral-key"})
    before = workspace.model_assignment_profile("workflow.py:sample")

    def fake_urlopen(req, *, timeout):
        raise HTTPError(
            req.full_url,
            404,
            "Not Found",
            {},
            BytesIO(b'{"message":"model not found"}'),
        )

    monkeypatch.setattr("zippergen.studio_models.request.urlopen", fake_urlopen)

    with pytest.raises(
        SystemExit,
        match="check failed for broken-reviewer.*Assignments were not changed",
    ):
        studio.execute("model config check broken-reviewer")

    assert workspace.model_assignment_profile("workflow.py:sample") == before
    configuration = workspace.model_configurations()["broken-reviewer"]
    assert configuration["check_status"] == "unavailable"
    assert any(
        "broken-reviewer:" in line
        and "not available with the configured mistral API key" in line
        for line in output
    )
    with pytest.raises(SystemExit, match="broken-reviewer is unavailable"):
        studio.execute("model assign Writer broken-reviewer")
    assert workspace.model_profile("workflow.py:sample")["lifelines"] == {}


def test_studio_models_check_accepts_a_case_insensitive_configuration(tmp_path):
    studio, _workspace, output = _studio(tmp_path)

    studio.execute("model config check MOCK")

    assert any(
        line.startswith("  ✓ mock:") and "built in" in line
        for line in output
    )


def test_studio_models_dashboard_does_not_change_routing(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    before = workspace.model_profile("workflow.py:sample")

    studio.execute("model")

    assert workspace.model_profile("workflow.py:sample") == before
    assert any(line == "Provider connections" for line in output)
    assert any(line == "Model configurations" for line in output)
    assert any(line == "Model assignments" for line in output)


def test_studio_models_assignment_warns_when_configuration_is_unchecked(tmp_path):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_model_configuration(
        "review-model",
        {
            "provider": "mistral",
            "model": "mistral-small-latest",
            "spec": "mistral:mistral-small-latest",
            "check_status": "not_checked",
        },
    )

    studio.execute("model assign Writer review-model")
    assert workspace.model_profile("workflow.py:sample")["lifelines"] == {
        "Writer": "mistral:mistral-small-latest"
    }
    assert any(
        "review-model is not_checked" in line
        and "model config check review-model" in line
        for line in output
    )


def test_studio_models_checks_local_configuration_identifiers(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path)
    workspace.save_provider_profile(
        "local",
        {
            "kind": "local",
            "base_url": "http://localhost:11434/v1",
        },
    )
    workspace.save_model_configuration(
        "local-writer",
        {
            "provider": "local",
            "model": "qwen2.5:7b",
            "spec": "local:qwen2.5:7b",
            "check_status": "not_checked",
        },
    )
    workspace.save_model_configuration(
        "missing-local",
        {
            "provider": "local",
            "model": "missing",
            "spec": "local:missing",
            "check_status": "not_checked",
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
        "zippergen.studio_models.request.urlopen",
        lambda req, *, timeout: ModelsResponse(),
    )

    studio.execute("model config check local-writer")

    assert any(
        "local-writer:" in line
        and "is available from the local provider" in line
        for line in output
    )

    with pytest.raises(SystemExit, match="check failed for missing-local"):
        studio.execute("model config check missing-local")
    assert workspace.model_configurations()["missing-local"]["check_status"] == (
        "unavailable"
    )
    assert any("Available models: qwen2.5:7b" in line for line in output)


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
        studio,
        "_verify_model_spec",
        lambda label, spec, for_save=False: SimpleNamespace(
            kind="success",
            message=f"{label}: {spec} is available.",
        ),
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
    assert ["--llm", "mock"] == cli_calls[0][
        cli_calls[0].index("--llm"):cli_calls[0].index("--llm") + 2
    ]
    assert [
        "--llm-for",
        "Writer=anthropic:claude-sonnet-4-6",
    ] == cli_calls[0][
        cli_calls[0].index("--llm-for"):
        cli_calls[0].index("--llm-for") + 2
    ]
    assert ["--assistant", "codex"] == cli_calls[0][
        cli_calls[0].index("--assistant"):
        cli_calls[0].index("--assistant") + 2
    ]
    assert cli_calls[0][-2:] == ["--llm-idle-timeouts-json", "{}"]


def test_studio_propagates_local_configuration_idle_release(
    tmp_path,
    monkeypatch,
):
    studio, workspace, _output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_provider_profile(
        "local",
        {
            "kind": "local",
            "base_url": "http://localhost:11434/v1",
            "check_status": "reachable",
            "checked_at": "2026-07-28T10:00:00+0200",
            "model_count": "1",
        },
    )
    workspace.save_model_configuration(
        "local-writer",
        {
            "provider": "local",
            "model": "qwen2.5:7b",
            "spec": "local:qwen2.5:7b",
            "idle_timeout": "300",
            "check_status": "available",
        },
    )
    workspace.save_model_assignment_profile(
        "workflow.py:sample",
        default="mock",
        lifelines={"Writer": "local-writer"},
    )
    run_calls = []
    cli_calls: list[list[str]] = []
    monkeypatch.setattr(
        "zippergen.studio.run_dev",
        lambda workspace, **kwargs: run_calls.append(kwargs),
    )
    monkeypatch.setattr(
        studio,
        "_verify_model_spec",
        lambda label, spec, for_save=False: SimpleNamespace(
            kind="success",
            message=f"{label}: {spec} is available.",
        ),
    )
    monkeypatch.setattr(
        "zippergen.serve.main",
        lambda arguments: cli_calls.append(arguments) or 0,
    )

    studio.execute("run")
    studio.deploy_workflow(["local-routed"])

    assert run_calls[0]["llm_idle_timeouts"] == {"Writer": 300.0}
    option = cli_calls[0].index("--llm-idle-timeouts-json")
    assert json.loads(cli_calls[0][option + 1]) == {"Writer": 300.0}


def test_studio_run_checks_only_models_used_by_llm_participants(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_model_configuration(
        "local-writer",
        {
            "provider": "local",
            "model": "qwen2.5:7b",
            "spec": "local:qwen2.5:7b",
            "check_status": "not_checked",
        },
    )
    workspace.save_model_assignment_profile(
        "workflow.py:sample",
        default="mock",
        lifelines={"Writer": "local-writer"},
    )
    checks: list[tuple[str, str]] = []
    run_calls: list[dict[str, object]] = []

    def verify(label, spec, *, for_save=False):
        checks.append((label, spec))
        return SimpleNamespace(
            kind="success",
            message=f"{label}: {spec} is available.",
        )

    monkeypatch.setattr(studio, "_verify_model_spec", verify)
    monkeypatch.setattr(
        "zippergen.studio.run_dev",
        lambda workspace, **kwargs: run_calls.append(kwargs),
    )

    studio.execute("run")

    assert checks == [("local-writer", "local:qwen2.5:7b")]
    assert len(run_calls) == 1
    assert any(line == "Run model checks" for line in output)
    assert any(
        "Writer" in line
        and "local-writer" in line
        and "local:qwen2.5:7b" in line
        for line in output
    )
    assert not any(
        line.strip().startswith("Writer") and "mock" in line
        for line in output
    )


def test_studio_run_stops_before_inputs_when_a_used_model_is_unreachable(
    tmp_path,
    monkeypatch,
):
    studio, workspace, output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    workspace.save_provider_profile(
        "local",
        {
            "kind": "local",
            "base_url": "http://localhost:11434/v1",
        },
    )
    workspace.save_model_configuration(
        "local-writer",
        {
            "provider": "local",
            "model": "qwen2.5:7b",
            "spec": "local:qwen2.5:7b",
            "check_status": "available",
        },
    )
    workspace.save_model_assignment_profile(
        "workflow.py:sample",
        default="mock",
        lifelines={"Writer": "local-writer"},
    )
    run_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        "zippergen.studio_models.request.urlopen",
        lambda req, *, timeout: (_ for _ in ()).throw(
            URLError("connection refused")
        ),
    )
    monkeypatch.setattr(
        "zippergen.studio.run_dev",
        lambda workspace, **kwargs: run_calls.append(kwargs),
    )

    with pytest.raises(
        SystemExit,
        match="Run stopped before collecting inputs.*local-writer",
    ):
        studio.execute("run")

    assert run_calls == []
    assert any(line == "Run model checks" for line in output)
    assert any("local-writer" in line and "not verified" in line for line in output)
    assert any("connection refused" in line for line in output)
    assert workspace.model_configurations()["local-writer"]["check_status"] == (
        "unverified"
    )


def test_studio_run_reports_runtime_provider_failure_without_a_traceback(
    tmp_path,
    monkeypatch,
):
    studio, workspace, _output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)
    monkeypatch.setattr(
        "zippergen.studio.run_dev",
        lambda workspace, **kwargs: (_ for _ in ()).throw(
            RuntimeError(
                "Lifeline 'Writer' raised: Could not reach API: "
                "connection refused"
            )
        ),
    )

    with pytest.raises(
        SystemExit,
        match=r"Run failed: Lifeline 'Writer'.*store was preserved.*resume",
    ):
        studio.execute("run")


def test_studio_models_rejects_lifelines_without_llm_actions(tmp_path):
    studio, workspace, _output = _studio(tmp_path)
    workspace.select_workflow("workflow.py:sample", cwd=workspace.root)

    try:
        studio.execute("model assign User openai:gpt-4o-mini")
    except SystemExit as exc:
        assert "not an LLM participant or action" in str(exc)
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
