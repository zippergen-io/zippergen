import os
import threading
from pathlib import Path

import pytest

from zippergen import Json
from zippergen.durable_runs import _coerce_input, _parse_guided_value, run_durable
from zippergen.rendering import TerminalRenderer
from zippergen.store import open_store
from zippergen.workspace import Workspace

TUTORIAL_SPEC = "examples/tutorial_review.py:tutorial_review"


SECRET_WORKFLOW_SOURCE = """
import os

from zippergen import DeploymentField, DeploymentSpec, Lifeline, pure, workflow

User = Lifeline("User")

zippergen_deployment = DeploymentSpec(
    fields=(
        DeploymentField(
            "api_key",
            "Development API key",
            target="env",
            env="OPENAI_API_KEY",
            secret=True,
            required=True,
            when="llm",
            when_values=("openai*",),
        ),
    ),
)

@pure
def check_secret(value: str) -> str:
    return f"{value}:{bool(os.environ.get('OPENAI_API_KEY'))}"

@workflow
def secret_demo(value: str @ User) -> str:
    User: result = check_secret(value)
    return result @ User
"""


ROUTED_WORKFLOW_SOURCE = """
from zippergen import DeploymentField, DeploymentSpec, Lifeline, llm, workflow

User = Lifeline("User")
Writer = Lifeline("Writer")
Reviewer = Lifeline("Reviewer")

zippergen_deployment = DeploymentSpec(
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
        DeploymentField(
            "anthropic_api_key",
            "Anthropic API key",
            target="env",
            env="ANTHROPIC_API_KEY",
            secret=True,
            required=True,
            when="llm",
            when_values=("claude*", "anthropic*"),
        ),
    ),
)

@llm(system="Draft.", user="{value}", parse="text", outputs=(("draft", str),))
def draft(value: str) -> None: ...

@llm(system="Review.", user="{draft}", parse="text", outputs=(("review", str),))
def review(draft: str) -> None: ...

@workflow
def routed(value: str @ User) -> str:
    User(value) >> Writer(value)
    Writer: draft_value = draft(value)
    Writer(draft_value) >> Reviewer(draft_value)
    Reviewer: result = review(draft_value)
    Reviewer(result) >> User(result)
    return result @ User
"""


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_guided_input_accepts_structured_json_values():
    parsed = _parse_guided_value(
        '{"caller":"Alice","slots":["Thursday",null]}'
    )

    assert _coerce_input("record", parsed, Json) == {
        "caller": "Alice",
        "slots": ["Thursday", None],
    }


def test_guided_input_rejects_non_json_values():
    with pytest.raises(SystemExit, match="not a valid Json value"):
        _coerce_input("record", {"bad": (1, 2)}, Json)


def test_durable_run_collects_multiple_inputs_and_reviews_inline(tmp_path):
    workspace = Workspace(_repository_root(), home=tmp_path / "home")
    responses = iter(
        [
            "Explain durable execution.",
            "3",
            "n",
            "y",
        ]
    )
    output: list[str] = []

    record = run_durable(
        workspace,
        workflow_spec=TUTORIAL_SPEC,
        input_func=lambda prompt: next(responses),
        output_func=output.append,
    )

    assert record["status"] == "done"
    assert record["inputs"] == {
        "request": "Explain durable execution.",
        "max_retries": 3,
    }
    assert record["result"] == "[revise_reply:draft]"
    assert Path(record["store"]).exists()
    conn = open_store(record["store"])
    assert conn.execute("SELECT COUNT(*) FROM human_tasks").fetchone()[0] == 2
    assert conn.execute(
        "SELECT COUNT(*) FROM human_tasks WHERE status='done'"
    ).fetchone()[0] == 2
    assert any(line.startswith("Workflow tutorial_review: valid") for line in output)
    assert output[-2] == "Result: [revise_reply:draft]"


def test_durable_run_records_non_secret_connector_routing(tmp_path):
    workspace = Workspace(_repository_root(), home=tmp_path / "home")
    responses = iter(
        [
            "Explain durable execution.",
            "1",
            "y",
        ]
    )
    snapshot = {
        "human:Reviewer.approve_reply": {
            "type": "human",
            "target": "Reviewer.approve_reply",
            "configuration": "telegram-review",
            "provider": "telegram",
            "chat_id": "123",
            "token_env": "ZIPPERGEN_CONNECTOR_TELEGRAM_TOKEN",
        }
    }

    record = run_durable(
        workspace,
        workflow_spec=TUTORIAL_SPEC,
        input_func=lambda prompt: next(responses),
        output_func=lambda _line: None,
        connector_snapshot=snapshot,
    )

    stored = workspace.load_run(str(record["run_id"]))
    assert stored["connectors"] == snapshot


def test_durable_run_uses_the_shared_renderer(tmp_path):
    workspace = Workspace(_repository_root(), home=tmp_path / "home")
    responses = iter(["Explain durable execution.", "1", "y"])
    output: list[str] = []
    renderer = TerminalRenderer(output.append, color=False)

    record = run_durable(
        workspace,
        workflow_spec=TUTORIAL_SPEC,
        input_func=lambda prompt: next(responses),
        output_func=output.append,
        renderer=renderer,
    )

    assert record["status"] == "done"
    assert "Durable run" in output
    assert "Run completed" in output
    assert any("✓ Workflow tutorial_review validated" in line for line in output)
    assert any("✓ done" in line for line in output)
    assert any("Model · Writer" in line for line in output)
    assert not any(line.strip().startswith("Model  mock") for line in output)
    assert not any(line.startswith("Workflow tutorial_review:") for line in output)


def test_durable_run_records_and_uses_assistant_backend(tmp_path, monkeypatch):
    workspace = Workspace(_repository_root(), home=tmp_path / "home")
    selections = []

    def fake_factory(default=None, *, project_root=None):
        selections.append((default, project_root))

        def backend(action, inputs):
            return {
                action.outputs[0][0]: f"updated:{inputs['change']}"
            }

        return backend

    monkeypatch.setattr(
        "zippergen.assistant_backends.make_cli_assistant_backend",
        fake_factory,
    )

    record = run_durable(
        workspace,
        workflow_spec=(
            "examples/assistant_maintenance.py:assistant_maintenance"
        ),
        provided_inputs={"change": "Document retries"},
        assistant="codex",
        interactive=False,
        output_func=lambda line: None,
    )

    assert record["status"] == "done"
    assert record["assistant"] == "codex"
    assert record["result"] == "updated:Document retries"
    assert selections[-1] == ("codex", str(workspace.root))


def test_durable_resume_claims_the_existing_pending_terminal_task(tmp_path):
    workspace = Workspace(_repository_root(), home=tmp_path / "home")

    def terminal_closed(prompt: str) -> str:
        raise RuntimeError("terminal closed during review")

    with pytest.raises(RuntimeError, match="terminal closed during review"):
        run_durable(
            workspace,
            workflow_spec=TUTORIAL_SPEC,
            provided_inputs={"request": "Resume me", "max_retries": 1},
            input_func=terminal_closed,
            output_func=lambda line: None,
        )

    interrupted = workspace.current_run()
    assert interrupted is not None
    assert interrupted["status"] == "failed"
    conn = open_store(interrupted["store"])
    assert conn.execute(
        "SELECT COUNT(*) FROM human_tasks WHERE status='pending'"
    ).fetchone()[0] == 1

    resumed = run_durable(
        workspace,
        resume=True,
        input_func=lambda prompt: "y",
        output_func=lambda line: None,
    )

    assert resumed["run_id"] == interrupted["run_id"]
    assert resumed["status"] == "done"
    assert resumed["result"] == "[draft_reply:draft]"
    assert conn.execute(
        "SELECT COUNT(*) FROM human_tasks WHERE status='done'"
    ).fetchone()[0] == 1


def test_durable_ctrl_c_keeps_terminal_input_on_main_thread_and_resumes(tmp_path):
    workspace = Workspace(_repository_root(), home=tmp_path / "home")
    input_threads = []

    def interrupt_review(prompt: str) -> str:
        input_threads.append(threading.current_thread())
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_durable(
            workspace,
            workflow_spec=TUTORIAL_SPEC,
            provided_inputs={"request": "Interrupt me", "max_retries": 1},
            input_func=interrupt_review,
            output_func=lambda line: None,
        )

    interrupted = workspace.current_run()
    assert interrupted is not None
    assert interrupted["status"] == "interrupted"
    assert input_threads == [threading.main_thread()]
    assert not any(
        thread.is_alive()
        and thread.name in {"Requester", "Writer", "Reviewer"}
        for thread in threading.enumerate()
    )
    conn = open_store(interrupted["store"])
    assert conn.execute(
        "SELECT COUNT(*) FROM human_tasks WHERE status='pending'"
    ).fetchone()[0] == 1

    resumed = run_durable(
        workspace,
        resume=True,
        input_func=lambda prompt: input_threads.append(threading.current_thread())
        or "y",
        output_func=lambda line: None,
    )

    assert resumed["run_id"] == interrupted["run_id"]
    assert resumed["status"] == "done"
    assert input_threads == [threading.main_thread(), threading.main_thread()]
    assert conn.execute(
        "SELECT COUNT(*) FROM human_tasks WHERE status='done'"
    ).fetchone()[0] == 1


def test_durable_run_rejects_resume_after_semantic_change(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    workflow_path = root / "workflow.py"
    workflow_path.write_text(
        "from zippergen import Lifeline, pure, workflow\n"
        "User = Lifeline('User')\n"
        "@pure\n"
        "def answer(value: str) -> str: return value\n"
        "@workflow\n"
        "def sample(value: str @ User) -> str:\n"
        "    User: result = answer(value)\n"
        "    return result @ User\n"
    )
    workspace = Workspace(root, home=tmp_path / "home")
    record = run_durable(
        workspace,
        workflow_spec="workflow.py:sample",
        provided_inputs={"value": "first"},
        output_func=lambda line: None,
    )
    workspace.update_run(record["run_id"], status="failed")
    workflow_path.write_text(workflow_path.read_text().replace("return value", "return value + '!'"))

    with pytest.raises(SystemExit, match="workflow meaning changed"):
        run_durable(
            workspace,
            resume=True,
            output_func=lambda line: None,
        )


def test_durable_run_collects_and_reuses_private_declared_secret(
    tmp_path, monkeypatch
):
    root = tmp_path / "project"
    root.mkdir()
    (root / "secret_workflow.py").write_text(SECRET_WORKFLOW_SOURCE)
    workspace = Workspace(root, home=tmp_path / "home")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    output: list[str] = []

    first = run_durable(
        workspace,
        workflow_spec="secret_workflow.py:secret_demo",
        provided_inputs={"value": "first"},
        llm="openai:test",
        secret_input_func=lambda prompt: "top-secret",
        output_func=output.append,
    )

    assert first["result"] == "first:True"
    assert workspace.load_secrets() == {"OPENAI_API_KEY": "top-secret"}
    assert workspace.secrets_path.stat().st_mode & 0o077 == 0
    assert "top-secret" not in workspace.run_path(first["run_id"]).read_text()
    assert "OPENAI_API_KEY" not in os.environ
    assert any("private development secret storage" in line for line in output)

    second = run_durable(
        workspace,
        workflow_spec="secret_workflow.py:secret_demo",
        provided_inputs={"value": "second"},
        llm="openai:test",
        secret_input_func=lambda prompt: pytest.fail("saved secret should be reused"),
        output_func=lambda line: None,
    )

    assert second["result"] == "second:True"


def test_durable_run_routes_models_and_collects_each_provider_secret(
    tmp_path, monkeypatch
):
    root = tmp_path / "project"
    root.mkdir()
    (root / "routed_workflow.py").write_text(ROUTED_WORKFLOW_SOURCE)
    workspace = Workspace(root, home=tmp_path / "home")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def fake_backend_from_spec(spec, **_kwargs):
        def backend(action, _inputs):
            name = action.outputs[0][0]
            return {name: f"{spec}:{action.name}"}

        return backend, spec

    monkeypatch.setattr(
        "zippergen.backends.backend_from_spec",
        fake_backend_from_spec,
    )
    secrets = iter(["openai-secret", "anthropic-secret"])

    record = run_durable(
        workspace,
        workflow_spec="routed_workflow.py:routed",
        provided_inputs={"value": "hello"},
        llm="mock",
        llms={
            "Writer": "openai:gpt-4o-mini",
            "Reviewer": "claude:claude-sonnet-4-6",
        },
        secret_input_func=lambda _prompt: next(secrets),
        output_func=lambda _line: None,
    )

    assert record["result"] == "claude:claude-sonnet-4-6:review"
    assert record["llms"] == {
        "Writer": "openai:gpt-4o-mini",
        "Reviewer": "claude:claude-sonnet-4-6",
    }
    assert workspace.load_secrets() == {
        "OPENAI_API_KEY": "openai-secret",
        "ANTHROPIC_API_KEY": "anthropic-secret",
    }
    record_text = workspace.run_path(record["run_id"]).read_text()
    assert "openai-secret" not in record_text
    assert "anthropic-secret" not in record_text


def test_run_durable_and_run_resume_reach_the_same_execution(tmp_path, monkeypatch):
    """One verb runs workflows; durability is an explicit mode of that verb."""

    from zippergen import serve

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        serve, "_durable_run_command", lambda args: calls.append(vars(args)) or 0
    )

    assert serve.main(["run", "wf.py:w", "--durable"]) == 0
    assert serve.main(["run", "--resume"]) == 0

    assert calls[0]["durable"] is True
    assert calls[1]["resume"] is True
    # Resuming does not need the workflow named again.
    assert calls[1]["workflow"] is None


def test_a_plain_run_outside_a_project_says_what_is_missing(tmp_path, monkeypatch):
    """Inside a project the workflow is inferred; outside there is none."""

    from zippergen import serve

    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit, match="none was found in this project"):
        serve.main(["run"])


def test_run_does_not_expose_run_ids():
    from zippergen import serve

    with pytest.raises(SystemExit) as exc:
        serve.main(["run", "wf.py:w", "--run-id", "old"])
    assert exc.value.code == 2


def test_run_does_not_expose_store_or_execution_mode():
    from zippergen import serve

    with pytest.raises(SystemExit) as exc:
        serve.main(
            ["run", "wf.py:w", "--store", "run.sqlite", "--execution", "memory"]
        )
    assert exc.value.code == 2
