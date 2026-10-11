"""Optional public CLI tool telemetry. Never protocol messages or model reasoning.

Progress is best-effort: a bounded observer queue cannot backpressure the CLI.
The ordinary backend still validates the final output and retains sessions.
"""

from __future__ import annotations

import json
import queue
import re
import subprocess
import threading
from collections.abc import Callable
from typing import Any

ProgressObserver = Callable[[dict[str, Any]], None]


# Status is deliberately authored for a UI, never extracted from reasoning or
# guessed from arbitrary assistant prose. Keep final-result parsing unchanged.
UI_STATUS_VERSION = 1


def ui_status(text: Any) -> list[dict[str, Any]]:
    if not isinstance(text, str) or len(text) > 2000:
        return []
    match = re.fullmatch(r"\s*<ui-status>([^<>]{1,240})</ui-status>\s*", text)
    if not match:
        return []
    lines = [" ".join(line.split()) for line in match[1].splitlines() if line.strip()]
    if not 1 <= len(lines) <= 3:
        return []
    return [{"kind": "ui_status", "phase": "completed", "text": "\n".join(lines)}]


def codex_progress(event: dict[str, Any]) -> list[dict[str, Any]]:
    event_type = event.get("type")
    if not isinstance(event_type, str):
        return []
    phase = {"item.started": "started", "item.completed": "completed"}.get(event_type)
    item = event.get("item")
    if phase is None or not isinstance(item, dict):
        return []
    if phase == "completed" and item.get("type") == "agent_message":
        return ui_status(item.get("text"))
    common = {"phase": phase, "tool_id": item.get("id")}
    if item.get("type") == "command_execution":
        return [
            {
                **common,
                "kind": "command",
                "command": item.get("command", ""),
                "exit_code": item.get("exit_code"),
                **_output(item.get("aggregated_output")),
            }
        ]
    if item.get("type") == "file_change":
        changes = item.get("changes", [])
        if not isinstance(changes, list):
            return []
        return [
            {
                **common,
                "kind": "file",
                "operation": change.get("kind", "update"),
                "path": change.get("path", ""),
                "status": item.get("status"),
            }
            for change in changes
            if isinstance(change, dict)
        ]
    # In particular: reasoning, unmarked agent messages and unknown items
    # do not cross this observational boundary.
    return []


def claude_progress(event: dict[str, Any]) -> list[dict[str, Any]]:
    if event.get("type") != "assistant":
        return []
    message = event.get("message")
    content = message.get("content", []) if isinstance(message, dict) else []
    if not isinstance(content, list):
        return []
    result = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            result.extend(ui_status(block.get("text")))
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        args = block.get("input")
        if not isinstance(args, dict):
            continue
        name = block.get("name")
        common = {"phase": "started", "tool_id": block.get("id")}
        if name == "Bash":
            result.append({**common, "kind": "command", "command": args.get("command", "")})
        elif name in {"Read", "Edit", "Write", "Glob", "Grep"}:
            result.append(
                {
                    **common,
                    "kind": "file",
                    "operation": name.lower(),
                    "path": args.get("file_path", args.get("path", "")),
                    **{
                        key: args[key]
                        for key in ("offset", "limit")
                        if name == "Read" and key in args
                    },
                }
            )
    return result


def _output(value: Any) -> dict[str, Any]:
    """Bound observer memory without interpreting public command results."""
    if not isinstance(value, str):
        return {}
    return {"output": value[:262144], "output_truncated": len(value) > 262144}


class ClaudeProgress:
    """Tool correlation belongs to one CLI invocation, not the workflow/session."""

    def __init__(self) -> None:
        self.pending: dict[str, dict[str, Any]] = {}

    def __call__(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        started = claude_progress(event)
        for item in started:
            if isinstance(item.get("tool_id"), str):
                self.pending[item["tool_id"]] = item
                if len(self.pending) > 256:
                    del self.pending[next(iter(self.pending))]
        if event.get("type") != "user":
            return started
        message = event.get("message")
        blocks = message.get("content", []) if isinstance(message, dict) else []
        if not isinstance(blocks, list):
            return []
        result = []
        for block in blocks:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            tool_id = block.get("tool_use_id")
            original = self.pending.pop(tool_id, None) if isinstance(tool_id, str) else None
            if original is None:
                continue
            completed = {
                **original,
                "phase": "completed",
                "status": "failed" if block.get("is_error") else "completed",
            }
            if original["kind"] == "command":
                content = block.get("content")
                if isinstance(content, list):
                    content = "\n".join(
                        b["text"]
                        for b in content
                        if isinstance(b, dict)
                        and b.get("type") == "text"
                        and isinstance(b.get("text"), str)
                    )
                completed.update(_output(content))
                # is_error is not an exit code. Preserve that distinction.
            result.append(completed)
        return result


class _Observer:
    def __init__(self, callback: ProgressObserver):
        self.callback = callback
        self.pending: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=64)
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self._drain, daemon=True)
        self.thread.start()

    def _drain(self) -> None:
        while not self.closed.is_set():
            item = self.pending.get()
            if item is None or self.closed.is_set():
                return
            try:
                self.callback(item)
            except Exception:
                # Presentation must not invalidate an otherwise successful action.
                pass

    def put(self, event: dict[str, Any]) -> None:
        try:
            self.pending.put_nowait(event)
        except queue.Full:
            pass

    def close(self) -> None:
        self.closed.set()
        # Do not wait for an embedding's stalled callback. Its invocation ID lets
        # consumers reject a late callback after the action has ended.
        try:
            self.pending.put_nowait(None)
        except queue.Full:
            pass


def _stream(
    command: list[str],
    *,
    decoder: Callable,
    on_progress: ProgressObserver,
    final_only: bool = False,
    input: str = "",
    timeout: float | None = None,
    text: bool = True,
    capture_output: bool = True,
    check: bool = False,
    **kwargs: Any,
) -> subprocess.CompletedProcess:
    """Drain both pipes concurrently while retaining subprocess.run timeout semantics."""
    observer = _Observer(on_progress)
    stdout: list[str] = []
    stderr: list[str] = []
    errors: list[BaseException] = []

    def read_output(pipe: Any, output: list[str], observe: bool) -> None:
        try:
            for line in pipe:
                output.append(line)
                if observe:
                    try:
                        event = json.loads(line)
                        if isinstance(event, dict):
                            for progress in decoder(event):
                                observer.put(progress)
                    except (ValueError, TypeError, KeyError):
                        # The backend's final decoder remains responsible for
                        # malformed output. Telemetry is not a success signal.
                        continue
        except BaseException as exc:
            errors.append(exc)

    def write_input(pipe: Any) -> None:
        try:
            pipe.write(input)
            pipe.flush()
        except (BrokenPipeError, OSError):
            pass
        finally:
            pipe.close()

    try:
        with subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            **kwargs,
        ) as process:
            readers = [
                threading.Thread(
                    target=read_output, args=(process.stdout, stdout, True), daemon=True
                ),
                threading.Thread(
                    target=read_output, args=(process.stderr, stderr, False), daemon=True
                ),
            ]
            writer = threading.Thread(target=write_input, args=(process.stdin,), daemon=True)
            for reader in readers:
                reader.start()
            writer.start()
            try:
                process.wait(timeout=timeout)
            except BaseException:
                process.kill()
                process.wait()
                raise
            finally:
                for reader in readers:
                    reader.join()
                writer.join()
        if errors:
            raise errors[0]
        output = "".join(stdout)
        if final_only and process.returncode == 0:
            # Claude stream-json has the same final result envelope used by the
            # buffered decoder/session validator, plus preceding progress records.
            results = []
            for line in stdout:
                record = json.loads(line)
                if isinstance(record, dict) and record.get("type") == "result":
                    results.append(line)
            if len(results) != 1:
                raise ValueError("Claude stream must contain exactly one final result envelope.")
            output = results[0]
        return subprocess.CompletedProcess(command, process.returncode, output, "".join(stderr))
    finally:
        observer.close()


def run_codex(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    return _stream(command, decoder=codex_progress, **kwargs)


def run_claude(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    command = list(command)
    command[command.index("--output-format") + 1] = "stream-json"
    command.append("--verbose")
    return _stream(command, decoder=ClaudeProgress(), final_only=True, **kwargs)
