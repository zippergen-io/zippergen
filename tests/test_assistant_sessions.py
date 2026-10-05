"""Native conversations are scoped to an execution and physical participant."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from zippergen import (
    AssistantExecutionError, Lifeline, Var, assistant, fragment,
    make_cli_assistant_backend, pure, run, run_sqlite, skip, spawn, workflow,
    render_workflow, ViewOptions, workflow_semantics,
)
from zippergen.assistant_backends import _bind_assistant_backend


Developer, Reviewer = Lifeline("Developer"), Lifeline("Reviewer")
count = Var("count", int, default=0)


@assistant(instructions="Inspect the request.")
def inspect_request(request: str) -> str: ...


@assistant(instructions="Implement the result.", access="write")
def implement_request(request: str) -> str: ...


@pure
def increment(value: int) -> int:
    return value + 1


@workflow
def editing_round(request: str @ Developer) -> str:
    Developer: diagnosis = inspect_request(request)
    Developer: report = implement_request(diagnosis)
    return report @ Developer


@workflow
def looping_rounds():
    while (count < 2) @ Developer:
        Developer: report = inspect_request("review")
        Developer: count = increment(count)
    return count @ Developer


@fragment
def review_child():
    skip(Developer)
    Reviewer: diagnosis = inspect_request("review")
    Reviewer: report = implement_request(diagnosis)


@workflow
def fresh_reviewers():
    spawn(review_child, at=Developer, retain=(Developer,), fresh="all")
    spawn(review_child, at=Developer, retain=(Developer,), fresh="all")


@workflow
def inherited_reviewers():
    with spawn(fresh="all"):
        review_child()
    with spawn(fresh="all"):
        review_child()


def _resume_id(command):
    flag = "resume" if command[1:2] == ["exec"] else "--resume"
    return command[command.index(flag) + 1] if flag in command else None


def _reply(command, session_id, result="done"):
    if command[1:2] == ["exec"]:
        return "\n".join(map(json.dumps, [
            {"type": "thread.started", "thread_id": session_id},
            {"type": "item.completed", "item": {
                "type": "agent_message", "text": result,
            }},
            {"type": "turn.completed"},
        ]))
    return json.dumps({
        "type": "result", "subtype": "success", "is_error": False,
        "session_id": session_id, "result": result,
    })


@pytest.fixture
def cli(monkeypatch):
    calls = []
    lock = threading.Lock()
    monkeypatch.setattr(
        "zippergen.assistant_backends.shutil.which", lambda name: f"/tools/{name}",
    )

    def fake_run(command, **kwargs):
        with lock:
            session_id = _resume_id(command) or f"conversation-{len(calls) + 1}"
            calls.append((command, kwargs, session_id))
        return subprocess.CompletedProcess(
            command, 0, stdout=_reply(command, session_id), stderr="",
        )

    monkeypatch.setattr("zippergen.assistant_backends.subprocess.run", fake_run)
    return calls


@pytest.mark.parametrize("backend_name", ["codex", "claude"])
@pytest.mark.parametrize("runner", [run, run_sqlite])
def test_different_actions_share_a_conversation(cli, tmp_path, backend_name, runner):
    backend = make_cli_assistant_backend(backend_name, project_root=tmp_path)
    assert runner(
        editing_round, [Developer], {"Developer": {"request": "fix"}},
        assistant_backend=backend, timeout=5,
    ) == "done"
    assert [_resume_id(call[0]) for call in cli] == [None, cli[0][2]]
    assert "Inspect the request." in cli[0][1]["input"]
    assert "Implement the result." in cli[1][1]["input"]
    assert '"request": "done"' in cli[1][1]["input"]
    # Capabilities must be recalculated when the action changes.
    if backend_name == "codex":
        assert "read-only" in cli[0][0]
        assert "workspace-write" in cli[1][0]
    else:
        assert "plan" in cli[0][0]
        assert "acceptEdits" in cli[1][0]
        assert "Read,Glob,Grep,Edit,Write" in cli[1][0]


@pytest.mark.parametrize("backend_name", ["codex", "claude"])
@pytest.mark.parametrize("runner", [run, run_sqlite])
def test_loop_rounds_share_a_conversation(cli, tmp_path, backend_name, runner):
    runner(
        looping_rounds, [Developer], {}, timeout=5,
        assistant_backend=make_cli_assistant_backend(backend_name, project_root=tmp_path),
    )
    assert [_resume_id(call[0]) for call in cli] == [None, cli[0][2]]


@pytest.mark.parametrize("runner", [run, run_sqlite])
def test_reusing_a_backend_for_a_new_execution_starts_fresh(cli, tmp_path, runner):
    backend = make_cli_assistant_backend("codex", project_root=tmp_path)
    for _ in range(2):
        runner(
            editing_round, [Developer], {"Developer": {"request": "fix"}},
            assistant_backend=backend, timeout=5,
        )
    assert [_resume_id(call[0]) for call in cli] == [
        None, cli[0][2], None, cli[2][2],
    ]
    assert cli[0][2] != cli[2][2]


@pytest.mark.parametrize("backend_name", ["codex", "claude"])
@pytest.mark.parametrize("runner", [run, run_sqlite])
def test_fresh_spawned_participants_use_static_routes_and_separate_sessions(
    cli, tmp_path, backend_name, runner,
):
    backend = make_cli_assistant_backend(
        project_root=tmp_path, routes={"Reviewer": backend_name},
    )
    runner(fresh_reviewers, [Developer, Reviewer], {}, assistant_backend=backend, timeout=5)
    starts = [call for call in cli if _resume_id(call[0]) is None]
    assert len(cli) == 4 and len(starts) == 2
    assert starts[0][2] != starts[1][2]
    for start in starts:
        assert sum(call[2] == start[2] for call in cli) == 2


@pytest.mark.parametrize("backend_name", ["codex", "claude"])
def test_fresh_action_does_not_replace_participant_conversation(cli, tmp_path, backend_name):
    backend = make_cli_assistant_backend(backend_name, project_root=tmp_path)
    backend(inspect_request, {"request": "first"})
    backend(replace(inspect_request, session="fresh"), {"request": "independent"})
    backend(implement_request, {"request": "continue"})
    assert [_resume_id(call[0]) for call in cli] == [None, None, cli[0][2]]
    flag = "--ephemeral" if backend_name == "codex" else "--no-session-persistence"
    assert flag in cli[1][0]
    assert flag not in cli[0][0] and flag not in cli[2][0]


@pytest.mark.parametrize("runner", [run, run_sqlite])
def test_inherited_participant_keeps_its_conversation(cli, tmp_path, runner):
    runner(
        inherited_reviewers, [Developer, Reviewer], {}, timeout=5,
        assistant_backend=make_cli_assistant_backend("codex", project_root=tmp_path),
    )
    assert len(cli) == 4
    assert [_resume_id(call[0]) for call in cli] == [None] + [cli[0][2]] * 3


def test_participants_backends_and_workspaces_are_separate(cli, tmp_path):
    (tmp_path / "other").mkdir()
    backend = make_cli_assistant_backend(
        "codex", project_root=tmp_path, routes={"Developer.implement_request": "claude"},
    )
    developer = _bind_assistant_backend(backend, "Developer")
    reviewer = _bind_assistant_backend(backend, "Reviewer")
    developer(inspect_request, {"request": "first"})
    reviewer(inspect_request, {"request": "review"})
    developer(implement_request, {"request": "switch backend"})
    developer(replace(inspect_request, workspace="other"), {"request": "switch workspace"})
    developer(inspect_request, {"request": "return"})
    assert [_resume_id(call[0]) for call in cli] == [None, None, None, None, cli[0][2]]


@pytest.mark.parametrize("backend_name", ["codex", "claude"])
def test_missing_conversation_id_fails_clearly(cli, tmp_path, monkeypatch, backend_name):
    monkeypatch.setattr(
        "zippergen.assistant_backends.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout=_reply(command, None), stderr="",
        ),
    )
    backend = make_cli_assistant_backend(backend_name, project_root=tmp_path)
    with pytest.raises(AssistantExecutionError, match="without a conversation ID"):
        backend(inspect_request, {"request": "review"})


@pytest.mark.parametrize("backend_name", ["codex", "claude"])
def test_resuming_a_different_conversation_is_rejected(cli, tmp_path, backend_name):
    backend = make_cli_assistant_backend(backend_name, project_root=tmp_path)
    backend(inspect_request, {"request": "first"})
    # Replace the output adapter only, leaving the actual command untouched.
    from unittest.mock import patch
    with patch(
        "zippergen.assistant_backends.subprocess.run",
        side_effect=lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout=_reply(command, "wrong-conversation"), stderr="",
        ),
    ), pytest.raises(AssistantExecutionError, match="different conversation"):
        backend(inspect_request, {"request": "continue"})


@pytest.mark.parametrize("backend_name", ["codex", "claude"])
def test_fresh_process_resume_restores_exact_cli_conversation(tmp_path, backend_name):
    """Use a local CLI stub across interpreter exits; no model requests."""
    executable = tmp_path / backend_name
    log = tmp_path / "cli-calls.jsonl"
    marker = tmp_path / "failed-once"
    executable.write_text(f"#!{sys.executable}\n" + f'''
import json, sys, uuid
from pathlib import Path
command = sys.argv
flag = "resume" if Path(__file__).name == "codex" else "--resume"
resumed = command[command.index(flag) + 1] if flag in command else None
session_id = resumed or str(uuid.uuid4())
with Path({str(log)!r}).open("a") as stream:
    stream.write(json.dumps({{"resumed": resumed, "session_id": session_id, "command": command}}) + "\\n")
sys.stdin.read()
marker = Path({str(marker)!r})
if resumed and not marker.exists():
    marker.touch()
    print("Interrupted turn", file=sys.stderr)
    sys.exit(1)
if Path(__file__).name == "codex":
    for event in [
        {{"type": "thread.started", "thread_id": session_id}},
        {{"type": "item.completed", "item": {{"type": "agent_message", "text": "done"}}}},
        {{"type": "turn.completed"}},
    ]:
        print(json.dumps(event))
else:
    print(json.dumps({{"type": "result", "subtype": "success", "is_error": False,
                      "session_id": session_id, "result": "done"}}))
''')
    executable.chmod(0o700)
    worker = tmp_path / "worker.py"
    worker.write_text(f'''
import json, sys
sys.path.insert(0, {str(Path(__file__).parent)!r})
from test_assistant_sessions import editing_round, Developer
from zippergen import run_sqlite, make_cli_assistant_backend
result = run_sqlite(
    editing_round, [Developer], {{"Developer": {{"request": "fix"}}}},
    store_path=sys.argv[1], timeout=5,
    assistant_backend=make_cli_assistant_backend(
        {{"backend": {backend_name!r}, "model": "exact-model", "reasoning_effort": "high"}},
        project_root={str(tmp_path)!r},
    ),
)
print(json.dumps(result))
''')
    environment = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]}

    def execute(store):
        return subprocess.run(
            [sys.executable, str(worker), str(store)], env=environment,
            capture_output=True, text=True, timeout=15,
        )

    store = tmp_path / "run.sqlite"
    interrupted = execute(store)
    assert interrupted.returncode != 0
    assert "Interrupted turn" in interrupted.stderr
    resumed = execute(store)
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads(resumed.stdout) == "done"
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 3
    assert calls[0]["resumed"] is None
    assert calls[1]["resumed"] == calls[2]["resumed"] == calls[0]["session_id"]
    for call in calls:
        command = call["command"]
        assert command[command.index("--model") + 1] == "exact-model"
        if backend_name == "codex":
            assert 'model_reasoning_effort="high"' in command
        else:
            assert command[command.index("--effort") + 1] == "high"
    # Completed recovery returns the committed result without another CLI call.
    assert execute(store).returncode == 0
    assert len(log.read_text().splitlines()) == 3
    assert execute(tmp_path / "new-run.sqlite").returncode == 0
    new_calls = [json.loads(line) for line in log.read_text().splitlines()][3:]
    assert new_calls[0]["resumed"] is None
    assert new_calls[0]["session_id"] != calls[0]["session_id"]


def test_session_policy_is_validated_and_included_in_views_and_semantics():
    with pytest.raises(ValueError, match="session must be"):
        assistant(instructions="Review", session="latest")
    assert "session='participant'" in render_workflow(
        editing_round, options=ViewOptions(detail="full"),
    )
    assert workflow_semantics(editing_round)["action_definitions"]["inspect_request"]["session"] == "participant"


def test_session_policy_changes_durable_compatibility_and_semantics():
    from zippergen.control import program_fingerprint
    from zippergen.syntax import ActStmt, LitExpr

    node = ActStmt(
        Developer, inspect_request, (LitExpr("review", str),), (Var("diagnosis", str),),
    )
    independent = replace(node, action=replace(inspect_request, session="fresh"))
    assert program_fingerprint({"Developer": node}) != program_fingerprint({"Developer": independent})
    wf = replace(editing_round, body=independent)
    assert "session='fresh'" in render_workflow(wf, options=ViewOptions(detail="full"))
    assert workflow_semantics(wf)["action_definitions"]["inspect_request"]["session"] == "fresh"
