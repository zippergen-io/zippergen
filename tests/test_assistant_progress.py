"""Public tool telemetry must not become a workflow result or expose reasoning."""

import json
import subprocess
import sys
import threading
import time
from dataclasses import replace

import pytest

from zippergen import AssistantExecutionError, assistant, make_cli_assistant_backend
from zippergen.assistant_backends import (
    _BACKENDS,
    _bind_assistant_backend,
    _new_assistant_execution,
)
from zippergen.assistant_progress import claude_progress, codex_progress, run_claude, run_codex


@assistant(instructions="Do the work.", access="write")
def work(task: str) -> str: ...


def test_public_parsers_exclude_reasoning_text_output_and_unknown_tools():
    for kind in ("reasoning", "agent_message", "unknown"):
        assert (
            codex_progress(
                {
                    "type": "item.completed",
                    "item": {
                        "type": kind,
                        "text": "private",
                        "id": "i1",
                    },
                }
            )
            == []
        )
    event = {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "thinking", "thinking": "private"},
                {"type": "text", "text": "public but not a tool"},
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "a.py"}},
                {
                    "type": "tool_use",
                    "id": "t2",
                    "name": "Edit",
                    "input": {
                        "file_path": "b.py",
                        "old_string": "private",
                        "new_string": "private",
                    },
                },
            ]
        },
    }
    events = claude_progress(event)
    assert [e["operation"] for e in events] == ["read", "edit"]
    assert "private" not in json.dumps(events)
    assert codex_progress(
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "id": "t3",
                "command": "pytest",
                "exit_code": 1,
                "aggregated_output": "private",
            },
        }
    ) == [
        {
            "kind": "command",
            "phase": "completed",
            "tool_id": "t3",
            "command": "pytest",
            "exit_code": 1,
        }
    ]


@pytest.mark.parametrize("backend", ["codex", "claude"])
def test_live_progress_before_result_and_separate_resumed_sessions(tmp_path, monkeypatch, backend):
    commands, events = [], []
    marker = tmp_path / "observed"
    session_number = 0

    def command(executable, workspace, action, session_id, settings=None):
        nonlocal session_number
        if session_id is None:
            session_number += 1
            session_id = f"session-{session_number}"
        commands.append((session_id, action.access, settings))
        if backend == "codex":
            records = [
                {"type": "thread.started", "thread_id": session_id},
                {
                    "type": "item.started",
                    "item": {
                        "type": "command_execution",
                        "id": "tool",
                        "command": "python -m unittest",
                    },
                },
            ]
            final = [
                {"type": "item.completed", "item": {"type": "agent_message", "text": "done"}},
                {"type": "turn.completed"},
            ]
        else:
            records = [
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tool",
                                "name": "Bash",
                                "input": {"command": "python -m unittest"},
                            }
                        ]
                    },
                }
            ]
            final = [
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "session_id": session_id,
                    "result": "done",
                }
            ]
        script = (
            "import sys,json,time,pathlib\n"
            "sys.stdin.read()\n"
            f"for record in {records!r}: print(json.dumps(record),flush=True)\n"
            f"marker=pathlib.Path({str(marker)!r})\n"
            "deadline=time.monotonic()+3\n"
            "while not marker.exists() and time.monotonic()<deadline: time.sleep(.01)\n"
            "assert marker.exists(), 'progress was buffered until exit'\n"
            f"for record in {final!r}: print(json.dumps(record),flush=True)\n"
        )
        return [sys.executable, "-c", script, "--output-format", "json"]

    def observe(event):
        events.append(event)
        if event["kind"] == "command":
            marker.write_text("seen")

    monkeypatch.setattr("zippergen.assistant_backends.shutil.which", lambda name: sys.executable)
    monkeypatch.setitem(_BACKENDS, backend, replace(_BACKENDS[backend], command=command))
    factory = make_cli_assistant_backend(backend, project_root=tmp_path, on_progress=observe)
    native = _new_assistant_execution(factory)
    implementer = _bind_assistant_backend(native, "Implementer")
    reviewer = _bind_assistant_backend(native, "Reviewer")
    for call in [implementer, reviewer, implementer]:
        marker.unlink(missing_ok=True)
        assert list(call(work, {"task": "work"}).values()) == ["done"]
    assert [c[0] for c in commands] == ["session-1", "session-2", "session-1"]
    starts = [e for e in events if e["kind"] == "invocation.started"]
    assert len({e["invocation_id"] for e in starts}) == 3
    assert [e["participant"] for e in starts] == ["Implementer", "Reviewer", "Implementer"]
    assert all(e["action"] == "work" for e in events)
    assert len([e for e in events if e["kind"] == "command"]) == 3


def test_slow_or_failing_observer_cannot_block_cli_or_result():
    entered, release = threading.Event(), threading.Event()

    def observer(event):
        entered.set()
        release.wait(5)
        raise RuntimeError("UI disconnected")

    event = {"type": "item.started", "item": {"type": "command_execution", "command": "pytest"}}
    script = f'import time; print({json.dumps(event)!r}, flush=True); time.sleep(.2); print("done")'
    started = time.monotonic()
    try:
        result = run_codex([sys.executable, "-c", script], on_progress=observer)
        assert entered.is_set()
        assert result.returncode == 0 and result.stdout.endswith("done\n")
        assert time.monotonic() - started < 2
    finally:
        release.set()


def test_large_input_and_both_output_pipes_do_not_deadlock():
    script = (
        'import sys; sys.stderr.write("e"*200000); print("x"*200000); print(len(sys.stdin.read()))'
    )
    result = run_codex(
        [sys.executable, "-c", script], input="p" * 200000, timeout=3, on_progress=lambda e: None
    )
    assert len(result.stderr) == 200000
    assert result.stdout.endswith("200000\n")


def test_stream_timeout_kills_child(tmp_path):
    marker = tmp_path / "after-timeout"
    script = f"import time,pathlib; time.sleep(1); pathlib.Path({str(marker)!r}).touch()"
    with pytest.raises(subprocess.TimeoutExpired):
        run_codex([sys.executable, "-c", script], timeout=0.1, on_progress=lambda e: None)
    time.sleep(1.1)
    assert not marker.exists()


def test_claude_failure_retains_exit_status_and_diagnostic():
    result = run_claude(
        [
            sys.executable,
            "-c",
            'import sys; sys.stderr.write("login failed"); sys.exit(9)',
            "--output-format",
            "json",
        ],
        on_progress=lambda e: None,
    )
    assert result.returncode == 9 and result.stderr == "login failed"


def test_progress_cannot_replace_missing_final_result(tmp_path, monkeypatch):
    monkeypatch.setattr("zippergen.assistant_backends.shutil.which", lambda name: sys.executable)
    event = {
        "type": "item.completed",
        "item": {"type": "command_execution", "command": "pytest", "exit_code": 0},
    }
    monkeypatch.setitem(
        _BACKENDS,
        "codex",
        replace(
            _BACKENDS["codex"],
            command=lambda *a: [sys.executable, "-c", f"print({json.dumps(event)!r})"],
        ),
    )
    backend = make_cli_assistant_backend("codex", project_root=tmp_path, on_progress=lambda e: None)
    with pytest.raises(AssistantExecutionError):
        backend(work, {"task": "work"})
