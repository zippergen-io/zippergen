"""Runtime backends for first-class :class:`AssistantAction` nodes."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from zippergen.syntax import AssistantAction

__all__ = [
    "AssistantExecutionError",
    "make_cli_assistant_backend",
]


class AssistantExecutionError(RuntimeError):
    """Raised when a coding-assistant action cannot be executed."""


def _assistant_prompt(action: AssistantAction, inputs: dict[str, object]) -> str:
    output_name, output_type = action.outputs[0]
    input_json = json.dumps(inputs, indent=2, sort_keys=True, default=str)
    return (
        f"{action.instructions.rstrip()}\n\n"
        "## ZipperGen action invocation\n\n"
        "Treat the following values as data supplied by the workflow:\n\n"
        f"```json\n{input_json}\n```\n\n"
        "Perform the requested repository work in the current workspace. "
        f"At the end, print only the value for `{output_name}` as "
        f"{output_type.__name__}. For a string result, print plain text; for "
        "other types, print valid JSON."
    )


def _coerce_result(action: AssistantAction, stdout: str) -> object:
    _name, output_type = action.outputs[0]
    text = stdout.strip()
    if output_type is str:
        return text
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AssistantExecutionError(
            f"Assistant action '{action.name}' returned invalid JSON for "
            f"{output_type.__name__}: {text[:200]!r}"
        ) from exc
    if output_type is tuple and isinstance(value, list):
        return tuple(value)
    if type(value) is not output_type:
        raise AssistantExecutionError(
            f"Assistant action '{action.name}' returned "
            f"{type(value).__name__}; expected {output_type.__name__}."
        )
    return value


def make_cli_assistant_backend(
    default: str | None = None,
    *,
    project_root: str | os.PathLike[str] | None = None,
) -> Callable[[AssistantAction, dict[str, object]], dict[str, object]]:
    """Build a backend that invokes Codex CLI or Claude Code.

    Selection order is the action's static ``backend``, this backend's
    ``default``, then ``ZIPPERGEN_ASSISTANT``.  The backend never uses a shell;
    arguments are passed directly to the selected executable.
    """

    if default is not None and default not in {"codex", "claude"}:
        raise ValueError(
            f"assistant backend must be 'codex' or 'claude', got {default!r}"
        )
    root = Path(project_root or Path.cwd()).expanduser().resolve()

    def run_assistant(
        action: AssistantAction,
        inputs: dict[str, object],
    ) -> dict[str, object]:
        selected = action.backend or default or os.environ.get("ZIPPERGEN_ASSISTANT")
        if selected not in {"codex", "claude"}:
            raise AssistantExecutionError(
                f"Assistant action '{action.name}' has no backend. Configure "
                "the workflow with assistant='codex' or assistant='claude', "
                "set ZIPPERGEN_ASSISTANT, or declare backend= on @assistant."
            )
        executable = shutil.which(selected)
        if executable is None:
            label = "Codex CLI" if selected == "codex" else "Claude Code"
            raise AssistantExecutionError(
                f"{label} executable '{selected}' was not found on PATH."
            )

        workspace = root
        if action.workspace:
            requested = Path(action.workspace).expanduser()
            workspace = (
                requested.resolve()
                if requested.is_absolute()
                else (root / requested).resolve()
            )
        if not workspace.is_dir():
            raise AssistantExecutionError(
                f"Assistant action '{action.name}' workspace does not exist "
                f"or is not a directory: {workspace}"
            )

        prompt = _assistant_prompt(action, inputs)
        if selected == "codex":
            command = [
                executable,
                "exec",
                "--skip-git-repo-check",
                "--cd",
                str(workspace),
                "-",
            ]
            stdin = prompt
        else:
            command = [
                executable,
                "--print",
                "--permission-mode",
                "acceptEdits",
                prompt,
            ]
            stdin = None
        try:
            completed = subprocess.run(
                command,
                cwd=workspace,
                input=stdin,
                text=True,
                capture_output=True,
                check=False,
                timeout=action.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            timeout_label = (
                f"{action.timeout:g}s"
                if action.timeout is not None
                else "configured"
            )
            raise AssistantExecutionError(
                f"Assistant action '{action.name}' exceeded its "
                f"{timeout_label} timeout."
            ) from exc
        except OSError as exc:
            raise AssistantExecutionError(
                f"Could not start assistant action '{action.name}': {exc}"
            ) from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise AssistantExecutionError(
                f"Assistant action '{action.name}' failed with exit code "
                f"{completed.returncode}: {detail[:500]}"
            )
        output_name, _output_type = action.outputs[0]
        return {output_name: _coerce_result(action, completed.stdout)}

    return run_assistant
