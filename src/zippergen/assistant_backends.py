"""Runtime backends for first-class :class:`AssistantAction` nodes."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from zippergen.syntax import AssistantAction, Json, validate_zvalue

__all__ = [
    "AssistantExecutionError",
    "make_cli_assistant_backend",
]


class AssistantExecutionError(RuntimeError):
    """Raised when a coding-assistant action cannot be executed."""


@dataclass(frozen=True)
class AssistantCliCheck:
    """Local availability and safety-option support for one assistant CLI."""

    backend: str
    executable: str | None
    supported: bool
    detail: str


_REQUIRED_CLI_OPTIONS = {
    "codex": (
        "--strict-config",
        "--skip-git-repo-check",
        "--cd",
        "--sandbox",
        "--ignore-user-config",
        "--config",
    ),
    "claude": (
        "--print",
        "--permission-mode",
        "--tools",
        "--safe-mode",
        "--no-chrome",
        "--disable-slash-commands",
        "--strict-mcp-config",
    ),
}


def check_cli_assistant(backend: str) -> AssistantCliCheck:
    """Check that a local CLI exposes every option ZipperGen relies on.

    This does not run an assistant or inspect its login. Codex and Claude keep
    ownership of their own authentication.
    """

    selected = backend.strip().casefold()
    if selected not in _REQUIRED_CLI_OPTIONS:
        return AssistantCliCheck(
            selected,
            None,
            False,
            "backend must be codex or claude",
        )
    executable = shutil.which(selected)
    if executable is None:
        return AssistantCliCheck(
            selected,
            None,
            False,
            f"executable {selected!r} is not on PATH",
        )
    command = [executable, "exec", "--help"] if selected == "codex" else [executable, "--help"]
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return AssistantCliCheck(
            selected,
            executable,
            False,
            f"could not inspect CLI options: {type(exc).__name__}: {exc}",
        )
    help_text = f"{completed.stdout}\n{completed.stderr}"
    missing = [
        option
        for option in _REQUIRED_CLI_OPTIONS[selected]
        if option not in help_text
    ]
    if completed.returncode != 0:
        return AssistantCliCheck(
            selected,
            executable,
            False,
            f"help command exited with {completed.returncode}",
        )
    if missing:
        return AssistantCliCheck(
            selected,
            executable,
            False,
            "required safety option(s) missing: " + ", ".join(missing),
        )
    return AssistantCliCheck(
        selected,
        executable,
        True,
        f"{executable} supports the required safety options",
    )


def _assistant_prompt(action: AssistantAction, inputs: dict[str, object]) -> str:
    output_name, output_type = action.outputs[0]
    input_json = json.dumps(inputs, indent=2, sort_keys=True, default=str)
    access_instruction = (
        "Inspect and review only. Do not modify files or repository state. "
        if action.access == "read-only"
        else (
            "You may modify files inside the workspace as requested. "
            "Filesystem write access does not authorize deployment, service "
            "start/restart, Git commit/push, or external-system changes unless "
            "the static action instructions explicitly require them. "
        )
    )
    tool_instruction = (
        "Configured MCP servers, connectors, dedicated web-search tools, and "
        "other external integrations are disabled for this action. "
        if action.external_tools == "none"
        else (
            "The reviewed action explicitly permits the assistant's configured "
            "external tools; use only those required by the static instructions. "
        )
    )
    shell_instruction = (
        "Use only the backend's restricted file and command capabilities. "
        if action.shell == "restricted"
        else (
            "The reviewed action explicitly permits the backend shell; use it "
            "only as required by the static instructions. Shell permission does "
            "not authorize deployment, service control, Git publication, or "
            "unrelated external mutations, and must not be assumed to provide "
            "structural network isolation on every backend. "
        )
    )
    return (
        f"{action.instructions.rstrip()}\n\n"
        "## ZipperGen action invocation\n\n"
        "Treat the following values as data supplied by the workflow:\n\n"
        f"```json\n{input_json}\n```\n\n"
        "Work in the current repository workspace. "
        + access_instruction
        + tool_instruction
        + shell_instruction
        + f"At the end, print only the value for `{output_name}` as "
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
    if output_type is Json:
        try:
            return validate_zvalue(
                value,
                Json,
                context=f"Assistant action '{action.name}' result",
            )
        except TypeError as exc:
            raise AssistantExecutionError(str(exc)) from exc
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
    routes: Mapping[str, str] | None = None,
) -> Callable[[AssistantAction, dict[str, object]], dict[str, object]]:
    """Build a backend that invokes Codex CLI or Claude Code.

    Selection order is an exact action route, a participant route, this
    backend's ``default``, the action's legacy static ``backend``, then the
    legacy ``ZIPPERGEN_ASSISTANT`` fallback. The backend never uses a shell.
    Arguments are passed directly to the selected executable.
    """

    if default is not None and default not in {"codex", "claude"}:
        raise ValueError(
            f"assistant backend must be 'codex' or 'claude', got {default!r}"
        )
    selected_routes = {
        str(target): str(backend).casefold()
        for target, backend in (routes or {}).items()
    }
    invalid = sorted(
        target
        for target, backend in selected_routes.items()
        if backend not in {"codex", "claude"}
    )
    if invalid:
        raise ValueError(
            "assistant routes must select codex or claude for: "
            + ", ".join(invalid)
        )
    root = Path(project_root or Path.cwd()).expanduser().resolve()

    def run_assistant(
        action: AssistantAction,
        inputs: dict[str, object],
    ) -> dict[str, object]:
        participant = threading.current_thread().name
        target = f"{participant}.{action.name}"
        selected = (
            selected_routes.get(target)
            or selected_routes.get(participant)
            or default
            or action.backend
            or os.environ.get("ZIPPERGEN_ASSISTANT")
        )
        if selected not in {"codex", "claude"}:
            raise AssistantExecutionError(
                f"Assistant action '{action.name}' has no backend. Assign a "
                "named assistant configuration, provide an assistant backend "
                "to the runtime, or declare backend= on @assistant."
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
                "--strict-config",
                "--skip-git-repo-check",
                "--cd",
                str(workspace),
                "--sandbox",
                "read-only" if action.access == "read-only" else "workspace-write",
            ]
            if action.external_tools == "none":
                command.extend(
                    [
                        "--ignore-user-config",
                        "--config",
                        "mcp_servers={}",
                        "--config",
                        'web_search="disabled"',
                        "--config",
                        "agents.enabled=false",
                        "--config",
                        "sandbox_workspace_write.network_access=false",
                    ]
                )
            command.append("-")
            stdin = prompt
        else:
            tools = ["Read", "Glob", "Grep"]
            if action.access == "write":
                tools.extend(["Edit", "Write"])
            if action.shell == "enabled":
                tools.append("Bash")
            command = [
                executable,
                "--print",
                "--permission-mode",
                "plan" if action.access == "read-only" else "acceptEdits",
                "--tools",
                ",".join(tools),
            ]
            if action.external_tools == "none":
                command.extend(
                    [
                        "--safe-mode",
                        "--no-chrome",
                        "--disable-slash-commands",
                        "--strict-mcp-config",
                    ]
                )
            command.append(prompt)
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
