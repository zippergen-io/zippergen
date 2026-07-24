"""A lightweight, discoverable project shell for ZipperGen development."""

from __future__ import annotations

import ast
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit

from prompt_toolkit import PromptSession
from prompt_toolkit.application.current import get_app
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion, CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory

from zippergen.dev import default_llm_spec, run_dev
from zippergen.models import normalize_llm_overrides
from zippergen.natural_language import (
    NaturalCommandPlan,
    NaturalLanguageStore,
    deterministic_plan,
    interpreter_prompt,
    looks_sensitive,
    parse_cli_plan,
)
from zippergen.semantic import semantic_snapshot, workflow_semantics
from zippergen.view import ViewOptions, workflow_view_data
from zippergen.workspace import (
    ASSISTANT_TASK_CONTRACT_VERSION,
    SPECIFICATION_GUIDE,
    Workspace,
    WorkspaceError,
)


InputFunc = Callable[[str], str]
OutputFunc = Callable[[str], object]
SecretInputFunc = Callable[[str], str]
StatusKind = Literal["success", "warning", "error", "info"]
CommandRisk = Literal["read-only", "configuration", "execution", "destructive"]
AssistantVerification = Literal["passed", "failed", "incomplete"]
_ASSISTANT_HEARTBEAT_SECONDS = 10.0


@dataclass(frozen=True)
class _PromptInput:
    content: str
    source_path: Path | None = None
    draft_path: Path | None = None


@dataclass(frozen=True)
class _LocalProviderCheck:
    checked_at: str
    model_count: int
    model_ids: tuple[str, ...]


@dataclass(frozen=True)
class _ModelVerification:
    kind: StatusKind
    message: str


@dataclass(frozen=True)
class _AssistantResult:
    verification: AssistantVerification
    summary: str
    checks: tuple[dict[str, str], ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class _CodexOutput:
    report: str | None = None
    diagnostics: tuple[str, ...] = ()
    suppressed_diagnostics: int = 0


class _LocalProviderError(RuntimeError):
    """A local OpenAI-compatible endpoint could not be verified."""


_PROVIDER_ALIASES = {
    "claude": "anthropic",
    "ollama": "local",
}
_PROVIDER_SECRETS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}
_PROVIDER_DEFAULT_MODELS = {
    "local": ("OLLAMA_MODEL", "qwen2.5:7b"),
    "openai": ("OPENAI_MODEL", "gpt-4o-mini"),
    "anthropic": ("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
    "mistral": ("MISTRAL_MODEL", "mistral-small-latest"),
}
_SUPPORTED_PROVIDERS = ("mock", "local", "openai", "anthropic", "mistral")
_STATUS_MARKS = {
    "success": "✓",
    "warning": "⚠",
    "error": "✗",
    "info": "•",
}
_STATUS_COLORS = {
    "success": "32",
    "warning": "33",
    "error": "31",
    "info": "36",
}
_STUDIO_COMMANDS = {
    "?",
    "ask",
    "current",
    "deploy",
    "doctor",
    "edit",
    "editor",
    "exit",
    "help",
    "logs",
    "models",
    "language",
    "plan",
    "project",
    "quit",
    "restart",
    "resume",
    "run",
    "runs",
    "settings",
    "start",
    "status",
    "stop",
    "workflow",
}

_COMMAND_COMPLETIONS = (
    ("project", "initialize, inspect, or reset the project"),
    ("workflow", "design, implement, inspect, and validate the workflow"),
    ("models", "configure, check, and assign reusable models"),
    ("run", "start a managed development run"),
    ("resume", "resume the current incomplete run"),
    ("runs", "list managed development runs"),
    ("deploy", "prepare or start a named deployment"),
    ("status", "show deployment status"),
    ("doctor", "check deployment readiness"),
    ("logs", "show deployment logs"),
    ("start", "start a deployment"),
    ("restart", "restart a deployment"),
    ("stop", "stop a deployment"),
    ("current", "show workflow, model, run, and deployment context"),
    ("settings", "inspect or configure global Studio preferences"),
    ("language", "inspect or configure natural-language commands"),
    ("ask", "interpret and execute an explicit natural-language request"),
    ("plan", "interpret natural language without executing it"),
    ("editor", "inspect or configure the terminal editor"),
    ("edit", "edit another project file"),
    ("help", "show all Studio commands"),
    ("exit", "leave Studio"),
    ("quit", "alias for exit"),
)

_SUBCOMMAND_COMPLETIONS = {
    "project": (
        ("init", "create the visible project manifest"),
        ("rename", "change the logical project name"),
        ("show", "show visible project configuration"),
        ("reset", "back up and reset private project state"),
    ),
    "settings": (
        ("show", "show global Studio preferences"),
        ("set", "set a global Studio preference"),
        ("reset", "reset one preference or all preferences"),
    ),
    "workflow": (
        ("create", "write the initial accepted specification"),
        ("refine", "create or reopen the pending refinement"),
        ("edit", "edit the specification or selected Python source"),
        ("show", "inspect specifications or code-first semantic views"),
        ("list", "list discovered workflow entry points"),
        ("select", "select a workflow entry point for inspection"),
        ("files", "list files used by the selected workflow"),
        ("status", "show the current implementation lifecycle"),
        ("implement", "run Codex or Claude on the current implementation"),
        ("validate", "validate the selected workflow and projections"),
        ("accept", "accept the reviewed workflow implementation"),
        ("discard", "archive an unwanted pending refinement"),
        ("history", "show specification and implementation history"),
        ("path", "print the automatic specification path"),
    ),
    "language": (
        ("show", "show interpreter, learning, and history status"),
        ("set", "choose auto, Codex, Claude, or no CLI fallback"),
        ("learning", "turn private project learning on or off"),
        ("history", "show interpreted requests and outcomes"),
        ("learned", "show reusable private interpretations"),
        ("forget", "forget one learned interpretation or all"),
    ),
    "editor": (
        ("show", "show the effective editor"),
        ("set", "remember a project editor"),
        ("reset", "restore automatic editor discovery"),
    ),
    "edit": (
        ("workflow", "edit the selected workflow source"),
        ("file", "edit a project file"),
    ),
    "show": (
        ("overview", "compact workflow summary"),
        ("protocol", "global protocol code"),
        ("communications", "communications only"),
        ("actions", "actions and prompts"),
        ("full", "complete workflow code"),
        ("agent", "one exact local projection"),
        ("agents", "selected-participant focus view"),
    ),
    "models": (
        ("show", "show configurations, connections, and assignments"),
        ("list", "list saved model configurations"),
        ("configure", "create or reopen a model configuration"),
        ("edit", "edit a saved model configuration"),
        ("rename", "rename a configuration and update every assignment"),
        ("remove", "remove an unused model configuration"),
        ("connect", "configure a provider connection"),
        ("disconnect", "remove a provider connection"),
        ("check", "check model configurations without changing assignments"),
        ("default", "set the inherited default configuration"),
        ("assign", "assign a configuration to one LLM-active participant"),
        ("inherit", "restore the default for one participant"),
    ),
}

_MODEL_COMPLETIONS = (
    ("mock", "deterministic built-in model"),
    ("local:", "local OpenAI-compatible model"),
    ("openai:", "OpenAI model"),
    ("anthropic:", "Anthropic model"),
    ("mistral:", "Mistral model"),
)


def _is_explicit_studio_syntax(parts: list[str]) -> bool:
    """Distinguish exact Studio syntax from natural prose beginning similarly."""

    if not parts:
        return False
    command = parts[0].casefold()
    args = parts[1:]
    if command not in _STUDIO_COMMANDS:
        return False
    if command in {"exit", "quit", "help", "?", "current"}:
        return not args
    if command in {
        "ask",
        "plan",
    }:
        return True
    if command == "run":
        if not args:
            return True
        if len(args) == 1:
            return not args[0].startswith("-")
        if len(args) == 2:
            return args[0] == "--assistant"
        return len(args) == 3 and args[1] == "--assistant"
    if command in {"resume", "runs"}:
        return not args
    if command in {"status", "doctor", "logs", "start", "restart", "stop", "deploy"}:
        if args and args[0].casefold() in {"of", "please", "the"}:
            return False
        return len(args) <= 2
    allowed: dict[str, set[str]] = {
        "project": {"init", "rename", "show", "reset"},
        "settings": {"show", "set", "reset"},
        "workflow": {
            "create",
            "refine",
            "edit",
            "show",
            "list",
            "select",
            "files",
            "status",
            "implement",
            "validate",
            "accept",
            "discard",
            "history",
            "path",
            "prompts",
        },
        "editor": {"show", "set", "reset"},
        "edit": {"file"},
        "models": {
            "show",
            "list",
            "configure",
            "edit",
            "rename",
            "remove",
            "connect",
            "disconnect",
            "check",
            "default",
            "assign",
            "inherit",
        },
        "language": {
            "show",
            "set",
            "learning",
            "history",
            "learned",
            "forget",
        },
    }
    if command in allowed:
        return not args or args[0].casefold() in allowed[command]
    return True


def _is_allowed_natural_plan_command(parts: list[str]) -> bool:
    """Validate the strict command subset exposed to repository interpreters."""

    if not parts:
        return False
    command = parts[0].casefold()
    args = parts[1:]
    lowered = [value.casefold() for value in args]
    if command in {"current", "resume", "runs"}:
        return not args
    if command == "project":
        return (
            len(args) == 1
            and lowered[0] == "show"
            or 1 <= len(args) <= 2
            and lowered[0] in {"init", "rename"}
        )
    if command == "settings":
        if not args:
            return True
        if len(args) == 1:
            return lowered[0] in {"show", "reset"}
        if lowered[0] == "reset":
            return len(args) == 2
        return lowered[0] == "set" and len(args) >= 3
    if command == "workflow":
        if not args:
            return True
        action = lowered[0]
        if action in {
            "list",
            "files",
            "status",
            "validate",
            "history",
            "path",
            "accept",
            "discard",
        }:
            return len(args) <= 2
        if action in {"create", "refine"}:
            return True
        if action == "implement":
            return all(
                value in {"implement", "codex", "claude", "--rerun", "--interactive"}
                for value in lowered
            )
        if action == "select":
            return len(args) <= 2
        if action == "edit":
            return len(args) <= 3
        if action == "show":
            if len(args) == 1:
                return True
            if len(args) == 2:
                return lowered[1] in {
                    "spec",
                    "pending",
                    "source",
                    "overview",
                    "protocol",
                    "communications",
                    "actions",
                    "full",
                }
            if len(args) == 3 and lowered[1] in {"agent", "source"}:
                return True
            return len(args) >= 3 and lowered[1] == "agents"
        return False
    if command == "models":
        if not args:
            return True
        if len(args) == 1:
            return lowered[0] in {"show", "list", "check", "configure"}
        if len(args) == 2:
            return lowered[0] in {
                "check",
                "default",
                "configure",
                "edit",
                "remove",
                "connect",
                "disconnect",
                "inherit",
            }
        return len(args) == 3 and (
            lowered[0] == "assign"
            or lowered[0] == "rename"
            or lowered[0] == "connect"
            and lowered[1] in {"local", "ollama"}
        )
    if command == "editor":
        return (
            len(args) == 1
            and lowered[0] in {"show", "reset"}
            or len(args) >= 2
            and lowered[0] == "set"
        )
    if command == "edit":
        return (
            len(args) == 2
            and lowered[0] == "file"
        )
    if command == "run":
        if not args:
            return True
        if len(args) == 1:
            return not args[0].startswith("-")
        if len(args) == 2:
            return (
                lowered[0] == "--assistant"
                and lowered[1] in {"codex", "claude"}
            )
        return (
            len(args) == 3
            and lowered[1] == "--assistant"
            and lowered[2] in {"codex", "claude"}
        )
    if command == "deploy":
        return (
            not args
            or len(args) == 1
            and (args[0] == "--no-start" or not args[0].startswith("-"))
            or len(args) == 2
            and args[1] == "--no-start"
            and not args[0].startswith("-")
        )
    if command in {"status", "doctor", "logs", "start", "restart", "stop"}:
        return len(args) <= 1
    return False


def _completion_words(text: str) -> tuple[list[str], str]:
    """Split the completed shell words from the word under the cursor."""

    boundary = -1
    quote: str | None = None
    escaped = False
    for index, character in enumerate(text):
        if escaped:
            escaped = False
            continue
        if character == "\\" and quote != "'":
            escaped = True
            continue
        if character in {"'", '"'}:
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
            continue
        if character.isspace() and quote is None:
            boundary = index
    prefix = text[: boundary + 1]
    fragment = text[boundary + 1 :]
    try:
        words = shlex.split(prefix)
    except ValueError:
        words = prefix.split()
    return words, fragment


def _unquote_completion_fragment(fragment: str) -> str:
    if not fragment:
        return ""
    try:
        parsed = shlex.split(fragment)
    except ValueError:
        parsed = []
    if len(parsed) == 1:
        return parsed[0]
    if fragment[0] in {"'", '"'}:
        fragment = fragment[1:]
    return fragment.replace("\\ ", " ").replace("\\\t", "\t")


class StudioCompleter(Completer):
    """Prompt-toolkit adapter for Studio's project-aware candidates."""

    def __init__(self, studio: Studio) -> None:
        self.studio = studio

    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent,
    ) -> Iterator[Completion]:
        del complete_event
        words, raw_fragment = _completion_words(document.text_before_cursor)
        fragment = _unquote_completion_fragment(raw_fragment)
        try:
            candidates = self.studio.completion_candidates(words, fragment)
        except (Exception, SystemExit):
            # Completion is assistive and must never make command entry fail
            # because project state or a workflow is temporarily invalid.
            return
        for value, description in candidates:
            if not value.lower().startswith(fragment.lower()):
                continue
            inserted = shlex.quote(value) if any(c.isspace() for c in value) else value
            yield Completion(
                inserted,
                start_position=-len(raw_fragment),
                display=value,
                display_meta=description,
            )


def _canonical_provider(value: str) -> str:
    provider = value.partition(":")[0].strip().lower()
    return _PROVIDER_ALIASES.get(provider, provider)


def _validate_model_spec(value: str) -> str:
    spec = value.strip()
    provider, separator, model = spec.partition(":")
    canonical = _canonical_provider(provider)
    if not spec or canonical not in _SUPPORTED_PROVIDERS:
        raise SystemExit(
            "Model provider must be mock, local/ollama, openai, "
            "anthropic/claude, or mistral."
        )
    if separator and not model.strip():
        raise SystemExit(f"Model spec {value!r} is missing a model after ':'.")
    if canonical == "mock":
        if separator:
            raise SystemExit("The built-in mock model is written simply as 'mock'.")
        return "mock"
    return f"{canonical}:{model.strip()}" if separator else canonical


_HELP = """Commands:
  NATURAL LANGUAGE                describe a Studio operation in ordinary text
  ask TEXT                        explicitly interpret and execute ordinary text
  plan TEXT                       interpret ordinary text without executing it
  settings                        show global Studio preferences
  settings set learning on|off    control learning for every local project
  settings set interpreter MODE   choose auto, codex, claude, or off
  settings set assistant TOOL     choose codex or claude
  settings set editor COMMAND     choose the terminal editor
  settings set output STYLE       choose banner or compact output framing
  settings reset [NAME|all]       restore one or every global default
  language                        show interpreter and project-local history
  language set auto|codex|claude|off
                                  alias for global interpreter setting
  language learning on|off        alias for global learning setting
  language history|learned        inspect interpretations and reusable examples
  language forget ID|all          remove learned private interpretations
  project init [NAME]            create the project manifest
  project rename NAME            change its logical name, not its directory
  project show                   show visible project configuration
  project reset                  choose fresh design or state-only reset
  project reset fresh [--yes]    archive manifest, spec, legacy prompts, state
  project reset state [--yes]    reset private state; keep all project files
  workflow                       show the design and implementation dashboard
  workflow create [DESCRIPTION]  write the initial accepted specification
  workflow create --file PATH    import the initial specification
  workflow refine [CHANGE]       create/reopen the one pending refinement
  workflow refine --file PATH    append a refinement from a file
  workflow edit [spec|code]      edit the specification or selected Python file
  workflow list                  list discovered workflow entry points
  workflow select [NUMBER|NAME|PATH.py:NAME]
                                 select an entry point for inspection
  workflow files                 list the selected workflow's known files
  workflow show                  choose a code-first semantic view
  workflow show spec|pending|source
                                 inspect requirements or authored Python source
  workflow show overview|protocol|communications|actions|full
  workflow show agent [NAME]     exact local projection
  workflow show agents [NAME...] selected-participant focus view
  workflow status                show the current implementation lifecycle
  workflow implement [codex|claude]
                                 run an assistant and return to Studio
  workflow implement TOOL --rerun
                                 deliberately rerun an implementation in review
  workflow implement codex --interactive
                                 open an interactive Codex implementation
  workflow validate              validate the workflow and every projection
  workflow accept [--yes]        accept the reviewed workflow implementation
  workflow discard [--yes]       archive an unwanted pending refinement
  workflow history               show design and implementation history
  workflow path                  print the automatic specification path
  editor [show|set CMD|reset]     inspect or remember the terminal editor
  edit file PATH                  edit another project file
  edit ... --editor CMD           choose an editor for this invocation only
  current                        show the complete project/workflow dashboard
  models                         show configurations, connections, assignments
  models configure [NAME]        create/reopen a guided model configuration
  models list                    list all named model configurations
  models check [NAME|all]        verify configurations without assigning them
  models assign LIFELINE NAME    assign a checked or saved configuration
  models default NAME            set the inherited default configuration
  models inherit LIFELINE        restore inheritance from the default
  models edit NAME               edit a saved configuration
  models rename OLD NEW          rename it and update every assignment
  models remove NAME             remove an unused configuration
  models connect NAME [URL]      advanced: configure a provider connection
  models disconnect NAME         advanced: remove a provider connection
  run [LLM] [--assistant TOOL]   start a run with optional one-run backends
  resume                         resume the current incomplete run
  runs                           list managed development runs
  deploy [NAME] [--no-start]     configure deployment; optionally defer startup
  status|doctor|logs [NAME]      inspect the remembered named deployment
  start|restart|stop [NAME]      operate the remembered named deployment
  help | ?                       show this help
  exit | quit                    leave Studio
"""


class Studio:
    def __init__(
        self,
        workspace: Workspace,
        *,
        input_func: InputFunc = input,
        output_func: OutputFunc = print,
        secret_input_func: SecretInputFunc | None = None,
        color: bool | None = None,
    ) -> None:
        self.workspace = workspace
        self.input = input_func
        self.output = output_func
        self._prompt_toolkit_enabled = (
            input_func is input
            and output_func is print
            and bool(getattr(sys.stdin, "isatty", lambda: False)())
            and bool(getattr(sys.stdout, "isatty", lambda: False)())
        )
        self._prompt_session: PromptSession[str] | None = None
        self.color = (
            output_func is print
            and bool(getattr(sys.stdout, "isatty", lambda: False)())
            and "NO_COLOR" not in os.environ
            and os.environ.get("TERM") != "dumb"
            if color is None
            else color
        )
        if secret_input_func is None:
            import getpass

            secret_input_func = getpass.getpass
        self.secret_input = secret_input_func

    def _emit(self, value: object = "") -> None:
        self.output(str(value))

    def _status(self, kind: StatusKind, message: str, *, indent: int = 0) -> None:
        """Emit one consistent, terminal-safe human status line."""

        mark = _STATUS_MARKS[kind]
        if self.color:
            mark = f"\033[{_STATUS_COLORS[kind]}m{mark}\033[0m"
        self._emit(f"{' ' * indent}{mark} {message}")

    def _status_mark(self, kind: StatusKind) -> str:
        mark = _STATUS_MARKS[kind]
        if self.color:
            return f"\033[{_STATUS_COLORS[kind]}m{mark}\033[0m"
        return mark

    def _emit_table(
        self,
        title: str,
        rows: list[tuple[str, object, StatusKind | None]],
    ) -> None:
        """Render a compact, grouped key/value table with a clear boundary."""

        self._emit(title)
        self._emit("─" * len(title))
        width = max((len(label) for label, _value, _kind in rows), default=0)
        for label, value, kind in rows:
            prefix = f"{self._status_mark(kind)} " if kind else ""
            self._emit(f"  {label:<{width}}  {prefix}{value}")
        self._emit()

    def _success(self, message: str, *, indent: int = 0) -> None:
        self._status("success", message, indent=indent)

    def _warning(self, message: str, *, indent: int = 0) -> None:
        self._status("warning", message, indent=indent)

    def _error(self, message: str, *, indent: int = 0) -> None:
        self._status("error", message, indent=indent)

    def _info(self, message: str, *, indent: int = 0) -> None:
        self._status("info", message, indent=indent)

    def _emit_output_boundary(self, command: str) -> None:
        """Separate one command's interaction from its echoed input line."""

        self._emit()
        settings = self.workspace.global_settings()
        if settings.get("output_style") == "compact":
            label = f" ZipperGen Studio · {command} "
            self._emit(f"──{label}{'─' * max(2, 58 - len(label))}")
            return
        content = f" ZipperGen Studio · {command} "
        width = max(58, len(content))
        self._emit(f"╭{'─' * width}╮")
        self._emit(f"│{content:<{width}}│")
        self._emit(f"╰{'─' * width}╯")

    @staticmethod
    def _output_boundary_label(parts: list[str]) -> str:
        """Name a command precisely without echoing user values or secrets."""

        command = parts[0].casefold()
        if len(parts) > 1 and command in {
            "workflow",
            "models",
            "project",
            "settings",
            "language",
            "editor",
            "edit",
        }:
            return f"{command} {parts[1].casefold()}"
        return command

    def _prompt(self) -> str:
        current = self.workspace.current_workflow
        label = current.rsplit(":", 1)[-1] if current else "no workflow"
        return f"zippergen [{label}]> "

    def welcome(self) -> None:
        self._emit("ZipperGen Studio")
        manifest = self.workspace.project_manifest()
        self._emit(f"Project: {manifest['name']}")
        self._emit(f"Root: {self.workspace.root}")
        current = self.workspace.current_workflow
        self._emit(f"Workflow: {current}" if current else "No workflow selected.")
        self._emit(
            "Type a command or describe what you want in ordinary language; "
            "press Tab to complete; 'help' shows the exact commands."
        )

    def _new_prompt_session(self) -> PromptSession[str]:
        self.workspace.directory.mkdir(parents=True, exist_ok=True)
        try:
            self.workspace.directory.chmod(0o700)
        except OSError:
            pass
        return PromptSession(
            history=FileHistory(str(self._studio_history_path())),
            completer=StudioCompleter(self),
            auto_suggest=AutoSuggestFromHistory(),
            bottom_toolbar=self._completion_toolbar,
            complete_while_typing=False,
            enable_history_search=True,
        )

    def completion_explanation(self, text: str) -> str:
        """Explain the sole completion match that Tab can insert."""

        words, raw_fragment = _completion_words(text)
        fragment = _unquote_completion_fragment(raw_fragment)
        if not fragment:
            return ""
        try:
            candidates = self.completion_candidates(words, fragment)
        except (Exception, SystemExit):
            return ""
        matches = [
            (value, description)
            for value, description in candidates
            if value.lower().startswith(fragment.lower())
        ]
        if len(matches) != 1:
            return ""
        value, description = matches[0]
        return f" Tab: {value} — {description} "

    def _completion_toolbar(self) -> str:
        """Render metadata even when prompt-toolkit has no menu to display."""

        try:
            text = get_app().current_buffer.document.text_before_cursor
        except Exception:
            # Completion help is optional and must never disrupt command input.
            return ""
        return self.completion_explanation(text)

    def _studio_history_path(self) -> Path:
        return self.workspace.directory / "studio.history"

    def _protect_studio_history(self) -> None:
        try:
            self._studio_history_path().chmod(0o600)
        except FileNotFoundError:
            pass
        except OSError:
            # The containing workspace is already owner-only. Failure to make
            # the file stricter must not make the interactive shell unusable.
            pass

    def _read_command(self) -> str:
        if not self._prompt_toolkit_enabled:
            return self.input(self._prompt())
        if self._prompt_session is None:
            self._prompt_session = self._new_prompt_session()
        try:
            return self._prompt_session.prompt(
                self._prompt(),
                complete_in_thread=True,
            )
        finally:
            self._protect_studio_history()

    def run(self) -> int:
        self.welcome()
        while True:
            try:
                line = self._read_command()
            except EOFError:
                self._emit()
                return 0
            except KeyboardInterrupt:
                self._warning("Use 'exit' to leave Studio.")
                continue
            try:
                if not self.execute(line, show_boundary=True):
                    return 0
            except KeyboardInterrupt:
                request_record = self.workspace.current_request()
                if (
                    request_record is not None
                    and request_record.get("status") == "assistant_interrupted"
                ):
                    self._warning(
                        "Assistant interrupted. Its request and any project "
                        "changes were preserved; use 'workflow status' to "
                        "inspect them and 'workflow implement codex --rerun' "
                        "only when another pass is intentional."
                    )
                else:
                    self._warning(
                        "Command interrupted. Use 'current' to inspect context; "
                        "use 'resume' for an incomplete managed run."
                    )
            except (SystemExit, WorkspaceError, ValueError) as exc:
                self._error(str(exc))

    def _completion_lifelines(self, *, llm_only: bool = False) -> list[str]:
        if self.workspace.current_workflow is None:
            return []
        try:
            _current, workflow, module = self._current_context()
            if llm_only:
                return list(self._llm_action_lifelines(workflow, module))
            return self._agent_names(workflow)
        except (Exception, SystemExit):
            return []

    def _completion_model_configurations(self) -> list[tuple[str, str]]:
        try:
            configurations = self.workspace.model_configurations()
        except (WorkspaceError, OSError):
            return []
        return [
            (
                name,
                f"{configuration.get('spec', 'unknown')} model configuration",
            )
            for name, configuration in configurations.items()
        ]

    def _path_completion_candidates(
        self,
        fragment: str,
    ) -> list[tuple[str, str]]:
        """Complete paths while presenting project-relative values by default."""

        expanded = Path(fragment).expanduser() if fragment else Path()
        absolute = expanded.is_absolute()
        target = expanded if absolute else self.workspace.root / expanded
        directory = target if fragment.endswith(("/", os.sep)) else target.parent
        name_prefix = "" if fragment.endswith(("/", os.sep)) else target.name
        try:
            children = sorted(
                directory.iterdir(),
                key=lambda path: (not path.is_dir(), path.name.lower()),
            )
        except OSError:
            return []
        candidates: list[tuple[str, str]] = []
        for child in children:
            if not child.name.startswith(name_prefix):
                continue
            if child.name.startswith(".") and not name_prefix.startswith("."):
                continue
            if fragment.startswith("~"):
                try:
                    value = "~/" + child.relative_to(Path.home()).as_posix()
                except ValueError:
                    value = str(child)
            elif absolute:
                value = str(child)
            else:
                try:
                    value = child.relative_to(self.workspace.root).as_posix()
                except ValueError:
                    value = str(child)
            if child.is_dir():
                value += "/"
            candidates.append((value, "directory" if child.is_dir() else "file"))
            if len(candidates) >= 100:
                break
        return candidates

    def _editor_completion_candidates(self) -> list[tuple[str, str]]:
        return [
            (name, "available terminal editor")
            for name in ("micro", "nano", "vim", "vi")
            if shutil.which(name) is not None
        ]

    def completion_candidates(
        self,
        words: list[str],
        fragment: str = "",
    ) -> list[tuple[str, str]]:
        """Return context-sensitive candidates for the word under the cursor."""

        if not words:
            return list(_COMMAND_COMPLETIONS)
        command = words[0].lower()
        args = words[1:]
        if not args and command in _SUBCOMMAND_COMPLETIONS:
            return list(_SUBCOMMAND_COMPLETIONS[command])
        if command == "settings":
            if not args:
                return list(_SUBCOMMAND_COMPLETIONS["settings"])
            action = args[0].lower()
            rest = args[1:]
            setting_names = [
                ("learning", "natural-language learning policy"),
                ("interpreter", "natural-language CLI fallback"),
                ("assistant", "default workflow coding assistant"),
                ("editor", "terminal editor command"),
                ("output", "Studio output framing"),
            ]
            if action == "set":
                if not rest:
                    return setting_names
                if len(rest) == 1:
                    return {
                        "learning": [
                            ("on", "learn successful CLI interpretations"),
                            ("off", "do not learn new interpretations"),
                        ],
                        "interpreter": [
                            ("auto", "prefer Codex, then Claude"),
                            ("codex", "use Codex CLI"),
                            ("claude", "use Claude Code"),
                            ("off", "disable CLI interpretation"),
                        ],
                        "assistant": [
                            ("codex", "use Codex for workflow implementation"),
                            ("claude", "use Claude Code for workflow implementation"),
                        ],
                        "editor": self._editor_completion_candidates(),
                        "output": [
                            ("banner", "connected three-line Studio banner"),
                            ("compact", "single-line Studio boundary"),
                        ],
                    }.get(rest[0].lower(), [])
            if action == "reset" and not rest:
                return [("all", "restore every default"), *setting_names]
            return []
        if command == "workflow":
            if not args:
                return list(_SUBCOMMAND_COMPLETIONS["workflow"])
            action = args[0].lower()
            rest = args[1:]
            if action == "show":
                if not rest:
                    return [
                        ("spec", "accepted workflow specification"),
                        ("pending", "pending refinement"),
                        ("source", "authored Python source"),
                        *_SUBCOMMAND_COMPLETIONS["show"],
                    ]
                if rest[0].lower() == "source" and len(rest) == 1:
                    try:
                        return [
                            (path, role)
                            for path, role in self._workflow_file_records()
                        ]
                    except (SystemExit, WorkspaceError, OSError, ValueError):
                        return []
                if rest[0].lower() in {"agent", "agents"}:
                    used = {value.lower() for value in rest[1:]}
                    return [
                        (name, "workflow participant")
                        for name in self._completion_lifelines()
                        if name.lower() not in used
                    ]
                return []
            if action == "select":
                try:
                    workflows = self.workspace.discover_workflows()
                except (WorkspaceError, OSError):
                    workflows = []
                values = [
                    (str(index), value)
                    for index, value in enumerate(workflows, start=1)
                ]
                values.extend(
                    (value, "discovered workflow") for value in workflows
                )
                return values
            if action in {"create", "refine"}:
                if "--file" in rest and rest[-1] == "--file":
                    return self._path_completion_candidates(fragment)
                if "--editor" in rest and rest[-1] == "--editor":
                    return self._editor_completion_candidates()
                if not rest:
                    return [
                        ("--file", "import text from an existing file"),
                        ("--editor", "choose an editor for this invocation"),
                    ]
                return []
            if action == "edit":
                if "--editor" in rest and rest[-1] == "--editor":
                    return self._editor_completion_candidates()
                if not rest:
                    return [
                        ("spec", "edit the accepted specification"),
                        ("code", "edit the selected Python workflow"),
                    ]
                return [("--editor", "choose an editor for this invocation")]
            if action == "implement":
                if not rest:
                    return [
                        ("codex", "implement with Codex"),
                        ("claude", "implement with Claude Code"),
                    ]
                if rest[0].lower() == "codex":
                    values = []
                    if "--rerun" not in rest:
                        values.append(
                            ("--rerun", "deliberately rerun reviewed work")
                        )
                    if "--interactive" not in rest:
                        values.append(
                            ("--interactive", "open an interactive Codex session")
                        )
                    return values
                if rest[0].lower() == "claude" and "--rerun" not in rest:
                    return [("--rerun", "deliberately rerun reviewed work")]
                return []
            if action in {"accept", "discard"}:
                return [("--yes", "confirm without another prompt")]
            return []
        if command == "models":
            if not args:
                return list(_SUBCOMMAND_COMPLETIONS["models"])
            action = args[0].lower()
            if action == "default":
                return self._completion_model_configurations()
            if action == "assign":
                if len(args) == 1:
                    return [
                        (name, "LLM-active participant")
                        for name in self._completion_lifelines(llm_only=True)
                    ]
                return self._completion_model_configurations()
            if action == "configure" and len(args) == 1:
                return self._completion_model_configurations()
            if action in {"edit", "rename", "remove"} and len(args) == 1:
                return [
                    candidate
                    for candidate in self._completion_model_configurations()
                    if candidate[0] != "mock"
                ]
            if action in {"connect", "disconnect"} and len(args) == 1:
                return [
                    (name, "model provider")
                    for name in _SUPPORTED_PROVIDERS
                    if name != "mock"
                ]
            if action == "check" and len(args) == 1:
                return [
                    ("all", "all saved model configurations"),
                ] + self._completion_model_configurations()
            if action == "inherit" and len(args) == 1:
                return [
                    (name, "LLM-active participant")
                    for name in self._completion_lifelines(llm_only=True)
                ]
            return []
        if command in {"run"}:
            if args and args[-1] == "--assistant":
                return [
                    ("codex", "run @assistant actions with Codex CLI"),
                    ("claude", "run @assistant actions with Claude Code"),
                ]
            return [
                *_MODEL_COMPLETIONS,
                ("--assistant", "select the coding-assistant action backend"),
            ]
        if command == "project" and args and args[0].lower() == "reset":
            if len(args) == 1:
                return [
                    ("fresh", "start a fresh design cycle"),
                    ("state", "reset private Studio state only"),
                ]
            if args[1].lower() in {"fresh", "state"}:
                return [("--yes", "confirm without another prompt")]
            return []
        if command == "edit":
            if "--editor" in args and args[-1] == "--editor":
                return self._editor_completion_candidates()
            if args and args[0].lower() == "file":
                return self._path_completion_candidates(fragment)
            return []
        if command == "editor" and args and args[0].lower() == "set":
            return self._editor_completion_candidates()
        if command == "language":
            if not args:
                return list(_SUBCOMMAND_COMPLETIONS["language"])
            action = args[0].lower()
            if action == "set":
                return [
                    ("auto", "prefer Codex, then Claude, when interpretation is needed"),
                    ("codex", "use the repository-aware Codex CLI"),
                    ("claude", "use the repository-aware Claude Code CLI"),
                    ("off", "use deterministic and learned interpretations only"),
                ]
            if action == "learning":
                return [
                    ("on", "remember successful CLI interpretations privately"),
                    ("off", "do not add learned interpretations"),
                ]
            if action == "forget":
                learned = self._language_store().learned()
                return [("all", "forget every learned interpretation")] + [
                    (
                        str(record.get("id") or ""),
                        str(record.get("example") or "learned interpretation"),
                    )
                    for record in learned
                    if record.get("id")
                ]
            return []
        if command in {"deploy", "status", "doctor", "logs", "start", "restart", "stop"}:
            values: list[tuple[str, str]] = []
            deployment = self.workspace.load().get("last_deployment")
            if deployment:
                values.append((str(deployment), "remembered deployment"))
            if command == "deploy":
                values.append(("--no-start", "prepare without starting"))
            return values
        return []

    def _show_workflow_dashboard(self) -> None:
        specification = self.workspace.specification()
        pending = self.workspace.pending_refinement()
        record = self._ensure_current_task_fresh(announce=False)
        state = "none"
        state_kind: StatusKind | None = None
        next_action = (
            "workflow create"
            if specification is None
            else "workflow refine"
        )
        if record is not None:
            record = self._normalize_task_lifecycle(record)
            state, state_kind = self._task_state(record)
            next_action = self._task_next(record)
        self._emit_table(
            "Workflow development",
            [
                (
                    "Specification",
                    (
                        self.workspace.specification_path.relative_to(
                            self.workspace.root
                        )
                        if specification is not None
                        else "not written"
                    ),
                    "success" if specification is not None else "warning",
                ),
                (
                    "Refinement",
                    "pending" if pending is not None else "none",
                    "warning" if pending is not None else None,
                ),
                (
                    "Selected",
                    self.workspace.current_workflow or "none",
                    None if self.workspace.current_workflow else "warning",
                ),
                ("Implementation", state, state_kind),
                ("Next", next_action, None),
            ],
        )

    def manage_workflow(self, args: list[str]) -> None:
        """Present specification, implementation, and inspection as one lifecycle."""

        if not args:
            self._show_workflow_dashboard()
            return
        action, *rest = args
        action = action.casefold()
        if action == "create":
            self.create_from_command(rest)
            return
        if action == "refine":
            self.manage_spec(["refine", *rest])
            return
        if action == "edit":
            target = rest[0].casefold() if rest and not rest[0].startswith("-") else "spec"
            options = rest[1:] if rest and not rest[0].startswith("-") else rest
            if target == "spec":
                self.manage_spec(["edit", *options])
                return
            if target == "code":
                self.edit_file(["workflow", *options])
                return
            raise SystemExit("Use workflow edit [spec|code] [--editor COMMAND].")
        if action == "list":
            if rest:
                raise SystemExit("Use workflow list.")
            self.list_workflows()
            return
        if action == "select":
            self.select_workflow(rest)
            return
        if action == "files":
            if rest:
                raise SystemExit("Use workflow files.")
            self.show_workflow_files()
            return
        if action == "show":
            if rest and rest[0].casefold() == "spec":
                if len(rest) != 1:
                    raise SystemExit("Use workflow show spec.")
                self.manage_spec(["show"])
                return
            if rest and rest[0].casefold() == "pending":
                if len(rest) != 1:
                    raise SystemExit("Use workflow show pending.")
                self.manage_spec(["pending"])
                return
            self.show_workflow(rest)
            return
        if action == "status":
            if rest:
                raise SystemExit("Use workflow status.")
            self.manage_task([])
            return
        if action == "implement":
            self.run_assistant(rest)
            return
        if action == "validate":
            if rest:
                raise SystemExit("Use workflow validate.")
            self.validate()
            return
        if action == "accept":
            if rest not in ([], ["--yes"]):
                raise SystemExit("Use workflow accept [--yes].")
            if self.workspace.pending_refinement() is not None:
                self.manage_spec(["reconcile", *rest])
            else:
                self.manage_task(["close", *rest])
            return
        if action == "discard":
            self.manage_spec(["discard", *rest])
            return
        if action == "history":
            if rest:
                raise SystemExit("Use workflow history.")
            self.manage_spec(["history"])
            self._emit()
            self.manage_task(["history"])
            return
        if action == "path":
            if rest:
                raise SystemExit("Use workflow path.")
            self.manage_spec(["path"])
            return
        if action == "prompts":
            self.manage_prompts(rest)
            return
        raise SystemExit(
            "Use workflow create, refine, edit, list, select, files, show, "
            "status, implement, validate, accept, discard, history, or path."
        )

    def execute(
        self,
        line: str,
        *,
        show_boundary: bool = False,
        _allow_natural: bool = True,
    ) -> bool:
        try:
            parts = shlex.split(line)
        except ValueError as exc:
            rough_parts = line.strip().split()
            if _allow_natural and not _is_explicit_studio_syntax(rough_parts):
                if show_boundary:
                    self._emit_output_boundary("language")
                self.interpret_natural_language(line)
                return True
            if show_boundary:
                self._emit_output_boundary("input")
            self._error(f"Could not parse command: {exc}")
            return True
        if not parts:
            return True
        if parts[0].casefold() == "providers":
            if show_boundary:
                self._emit_output_boundary("models")
            raise SystemExit(
                "`providers` is not a Studio command. Provider connections are "
                "managed with `models connect NAME`; use `models` to inspect them."
            )
        explicit = _is_explicit_studio_syntax(parts)
        if _allow_natural and not explicit:
            if show_boundary:
                self._emit_output_boundary("language")
            self.interpret_natural_language(line)
            return True
        if not explicit:
            raise SystemExit(
                f"Invalid planned Studio command: {line}. "
                "The interpreter may only use documented Studio syntax."
            )
        command, *args = parts
        command = command.lower()
        if command in {"exit", "quit"}:
            return False
        if show_boundary:
            self._emit_output_boundary(self._output_boundary_label(parts))
        if command in {"help", "?"}:
            self._emit(_HELP.rstrip())
        elif command == "ask":
            if not args:
                raise SystemExit("Use ask TEXT.")
            self.interpret_natural_language(" ".join(args))
        elif command == "plan":
            if not args:
                raise SystemExit("Use plan TEXT.")
            self.interpret_natural_language(" ".join(args), preview_only=True)
        elif command == "settings":
            self.configure_settings(args)
        elif command == "language":
            self.manage_language(args)
        elif command == "project":
            self.configure_project(args)
        elif command == "workflow":
            self.manage_workflow(args)
        elif command == "editor":
            self.configure_editor(args)
        elif command == "edit":
            self.edit_file(args)
        elif command == "current":
            self.show_current()
        elif command == "models":
            self.configure_models(args)
        elif command == "run":
            assistant_backend = str(self._global_settings()["assistant"])
            run_args = list(args)
            if "--assistant" in run_args:
                index = run_args.index("--assistant")
                if index + 1 >= len(run_args):
                    raise SystemExit(
                        "Use run [LLM_SPEC] --assistant codex|claude."
                    )
                assistant_backend = run_args[index + 1].lower()
                del run_args[index:index + 2]
                if assistant_backend not in {"codex", "claude"}:
                    raise SystemExit(
                        "Assistant backend must be codex or claude."
                    )
            if len(run_args) > 1:
                raise SystemExit(
                    "Use run [LLM_SPEC] [--assistant codex|claude]."
                )
            profile = self._run_model_profile()
            default_model = profile.get("default")
            run_dev(
                self.workspace,
                llm=(
                    run_args[0]
                    if run_args
                    else str(default_model) if default_model else None
                ),
                llms=normalize_llm_overrides(profile.get("lifelines")),
                assistant=assistant_backend,
                interactive=True,
                input_func=self.input,
                output_func=self.output,
            )
        elif command == "resume":
            if args:
                raise SystemExit("Studio 'resume' takes no arguments.")
            run_dev(
                self.workspace,
                resume=True,
                interactive=True,
                input_func=self.input,
                output_func=self.output,
            )
        elif command == "runs":
            self.show_runs()
        elif command == "deploy":
            self.deploy_workflow(args)
        elif command in {"status", "doctor", "logs", "start", "restart", "stop"}:
            self.deployment_action(command, args)
        else:
            raise SystemExit(
                f"Unknown Studio command: {command}. "
                "Type 'help' for available commands."
            )
        return True

    def _language_store(self) -> NaturalLanguageStore:
        return NaturalLanguageStore(self.workspace.natural_language_path)

    def _global_settings(self) -> dict[str, object]:
        """Load global preferences, migrating old project preferences once."""

        if self.workspace.global_settings_path.exists():
            return self.workspace.global_settings()
        settings = self.workspace.default_global_settings()
        migrated = False
        if self.workspace.natural_language_path.exists():
            legacy_language = self._language_store().load()
            for name in ("learning", "interpreter"):
                if name in legacy_language:
                    settings[name] = legacy_language[name]
                    migrated = True
        legacy_editor = self.workspace.load().get("editor_command")
        if legacy_editor:
            settings["editor_command"] = legacy_editor
            migrated = True
        if migrated:
            return self.workspace.update_global_settings(**settings)
        return settings

    def configure_settings(self, args: list[str]) -> None:
        """Inspect and edit owner-private preferences shared by all projects."""

        settings = self._global_settings()
        if not args or args == ["show"]:
            editor = settings.get("editor_command")
            self._emit_table(
                "Global Studio settings",
                [
                    (
                        "Learning",
                        "on" if settings["learning"] else "off",
                        "success" if settings["learning"] else "warning",
                    ),
                    ("Interpreter", settings["interpreter"], None),
                    ("Assistant", settings["assistant"], None),
                    (
                        "Editor",
                        shlex.join(self._parse_editor_command(editor))
                        if editor
                        else "automatic discovery",
                        None,
                    ),
                    ("Output", settings["output_style"], None),
                    ("Scope", "all local ZipperGen projects", "success"),
                    ("Storage", self.workspace.global_settings_path, None),
                    (
                        "Project data",
                        "learned interpretations and history remain project-local",
                        None,
                    ),
                ],
            )
            return

        action, *rest = args
        action = action.casefold()
        if action == "set" and len(rest) >= 2:
            name = rest[0].casefold()
            values = rest[1:]
            changes: dict[str, object]
            shown: str
            if name == "learning" and len(values) == 1:
                value = values[0].casefold()
                if value not in {"on", "off"}:
                    raise SystemExit("Use settings set learning on|off.")
                changes = {"learning": value == "on"}
                shown = value
            elif name == "interpreter" and len(values) == 1:
                value = values[0].casefold()
                if value not in {"auto", "codex", "claude", "off"}:
                    raise SystemExit(
                        "Use settings set interpreter auto|codex|claude|off."
                    )
                if value in {"codex", "claude"}:
                    self._language_backend(value, required=True)
                changes = {"interpreter": value}
                shown = value
            elif name == "assistant" and len(values) == 1:
                value = values[0].casefold()
                if value not in {"codex", "claude"}:
                    raise SystemExit(
                        "Use settings set assistant codex|claude."
                    )
                executable = shutil.which(value)
                if executable is None:
                    raise SystemExit(
                        f"Assistant executable was not found: {value}."
                    )
                changes = {"assistant": value}
                shown = value
            elif name == "editor":
                command = self._parse_editor_command(values)
                if shutil.which(command[0]) is None:
                    raise SystemExit(
                        f"Editor executable was not found: {command[0]}."
                    )
                changes = {"editor_command": command}
                shown = shlex.join(command)
            elif name == "output" and len(values) == 1:
                value = values[0].casefold()
                if value not in {"banner", "compact"}:
                    raise SystemExit(
                        "Use settings set output banner|compact."
                    )
                changes = {"output_style": value}
                shown = value
            else:
                raise SystemExit(
                    "Use settings set learning|interpreter|assistant|editor|output VALUE."
                )
            self.workspace.update_global_settings(**changes)
            self._emit_table(
                "Global setting updated",
                [
                    ("Setting", name, None),
                    ("Value", shown, "success"),
                    ("Scope", "all local ZipperGen projects", None),
                    ("Next", "settings", None),
                ],
            )
            return

        if action == "reset" and len(rest) <= 1:
            public_name = rest[0].casefold() if rest else "all"
            mapping = {
                "learning": "learning",
                "interpreter": "interpreter",
                "assistant": "assistant",
                "editor": "editor_command",
                "output": "output_style",
            }
            if public_name != "all" and public_name not in mapping:
                raise SystemExit(
                    "Use settings reset [learning|interpreter|assistant|editor|output|all]."
                )
            self.workspace.reset_global_settings(
                None if public_name == "all" else mapping[public_name]
            )
            self._emit_table(
                "Global settings reset",
                [
                    ("Setting", public_name, None),
                    ("Status", "default restored", "success"),
                    ("Next", "settings", None),
                ],
            )
            return

        raise SystemExit(
            "Use settings, settings set NAME VALUE, or settings reset [NAME|all]."
        )

    def _language_participants(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        try:
            _current, workflow, module = self._current_context()
            participants = tuple(self._agent_names(workflow))
            active = tuple(self._llm_action_lifelines(workflow, module))
            return participants, active
        except (Exception, SystemExit):
            return (), ()

    def _language_backend(
        self,
        configured: str,
        *,
        required: bool,
    ) -> tuple[Literal["codex", "claude"], str] | None:
        mode = configured.casefold()
        if mode == "off":
            if required:
                raise SystemExit(
                    "I could not map this request to a deterministic or learned "
                    "Studio command, and the repository-aware CLI fallback is off. "
                    "Use an exact command or 'language set auto|codex|claude'."
                )
            return None
        choices = ("codex", "claude") if mode == "auto" else (mode,)
        for choice in choices:
            if choice not in {"codex", "claude"}:
                break
            executable = shutil.which(choice)
            if executable:
                return (
                    ("codex", executable)
                    if choice == "codex"
                    else ("claude", executable)
                )
        if not required:
            return None
        if mode == "auto":
            raise SystemExit(
                "This request needs repository-aware interpretation, but neither "
                "Codex nor Claude Code is installed. Install and authenticate one, "
                "use an exact Studio command, or use 'language set off'."
            )
        label = "Codex CLI" if mode == "codex" else "Claude Code"
        raise SystemExit(
            f"{label} is selected for natural-language interpretation but was "
            f"not found. Install it or use 'language set auto|off'."
        )

    def _language_context(self) -> str:
        state = self.workspace.load()
        participants, active = self._language_participants()
        manifest = self.workspace.project_manifest()
        profile: dict[str, object] = {"default": None, "lifelines": {}}
        if self.workspace.current_workflow:
            try:
                profile = self._run_model_profile()
            except (Exception, SystemExit):
                pass
        current_request: dict[str, object] | None = None
        try:
            value = self.workspace.current_request()
            if value is not None:
                current_request = {
                    "kind": value.get("kind"),
                    "status": value.get("status"),
                    "verification": value.get("assistant_verification"),
                    "workflow_spec": value.get("workflow_spec"),
                }
        except WorkspaceError:
            pass
        context = {
            "project_root": str(self.workspace.root),
            "project_name": manifest.get("name"),
            "manifest_exists": manifest.get("exists"),
            "specification_file": str(
                self.workspace.specification_path.relative_to(self.workspace.root)
            ),
            "specification_exists": self.workspace.specification() is not None,
            "pending_refinement": self.workspace.pending_refinement() is not None,
            "selected_workflow": self.workspace.current_workflow,
            "discovered_workflows": self.workspace.discover_workflows(),
            "participants": participants,
            "llm_active_participants": active,
            "model_profile": profile,
            "model_configurations": {
                name: {
                    "spec": configuration.get("spec"),
                    "status": configuration.get("check_status"),
                }
                for name, configuration
                in self.workspace.model_configurations().items()
            },
            "current_run": state.get("current_run"),
            "last_deployment": state.get("last_deployment"),
            "current_task": current_request,
        }
        return json.dumps(context, indent=2, sort_keys=True, default=str)

    def _interpret_with_cli(
        self,
        request_text: str,
        *,
        configured: str,
    ) -> NaturalCommandPlan:
        selected = self._language_backend(configured, required=True)
        assert selected is not None
        backend, executable = selected
        label = "Codex" if backend == "codex" else "Claude Code"
        self._info(
            f"Interpreting with {label}; repository access is read-only."
        )
        prompt = interpreter_prompt(
            request_text,
            context=self._language_context(),
        )
        if backend == "codex":
            command = [
                executable,
                "exec",
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "--cd",
                str(self.workspace.root),
                "-",
            ]
            stdin = prompt
        else:
            command = [
                executable,
                "--print",
                "--permission-mode",
                "plan",
                prompt,
            ]
            stdin = None
        completed = subprocess.run(
            command,
            cwd=self.workspace.root,
            input=stdin,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            if len(detail) > 500:
                detail = detail[-500:]
            raise SystemExit(
                f"{label} could not interpret the request"
                + (f": {detail}" if detail else ".")
            )
        try:
            return parse_cli_plan(completed.stdout, source=backend)
        except ValueError as exc:
            raise SystemExit(f"{label} returned an invalid command plan: {exc}") from exc

    def _canonical_natural_command(self, command_line: str) -> str:
        try:
            parts = shlex.split(command_line)
        except ValueError as exc:
            raise SystemExit(
                f"Invalid planned Studio command {command_line!r}: {exc}"
            ) from exc
        if not parts or not _is_allowed_natural_plan_command(parts):
            raise SystemExit(
                f"The interpreter proposed unsupported Studio syntax: "
                f"{command_line}"
            )
        top = parts[0].casefold()
        allowed = {
            "current",
            "deploy",
            "doctor",
            "edit",
            "editor",
            "logs",
            "models",
            "project",
            "restart",
            "resume",
            "run",
            "runs",
            "settings",
            "start",
            "status",
            "stop",
            "workflow",
        }
        if top not in allowed:
            raise SystemExit(
                f"The interpreter may not invoke the Studio command {parts[0]!r}."
            )

        participants, _active = self._language_participants()
        canonical = {name.casefold(): name for name in participants}
        configuration_names = {
            name.casefold(): name
            for name in self.workspace.model_configurations()
        }

        def replace(index: int) -> None:
            if index < len(parts):
                parts[index] = canonical.get(parts[index].casefold(), parts[index])

        if top == "workflow" and len(parts) >= 4:
            if parts[1].casefold() == "show" and parts[2].casefold() == "agent":
                replace(3)
            elif (
                parts[1].casefold() == "show"
                and parts[2].casefold() == "agents"
            ):
                for index in range(3, len(parts)):
                    replace(index)
        elif top == "models" and len(parts) >= 2:
            action = parts[1].casefold()
            if action in {"assign", "inherit"}:
                replace(2)
            if action == "assign" and len(parts) >= 4:
                parts[3] = configuration_names.get(
                    parts[3].casefold(), parts[3]
                )
            elif action in {
                "check",
                "edit",
                "rename",
                "remove",
                "default",
            } and len(parts) >= 3:
                parts[2] = configuration_names.get(
                    parts[2].casefold(), parts[2]
                )
        return shlex.join(parts)

    def _natural_command_risk(self, command_line: str) -> CommandRisk:
        parts = shlex.split(command_line)
        command = parts[0].casefold()
        args = [value.casefold() for value in parts[1:]]
        if command in {"current", "runs", "status", "doctor", "logs"}:
            return "read-only"
        if command == "settings" and (
            not args or args[0] == "show"
        ):
            return "read-only"
        if command == "workflow":
            if not args or args[0] in {
                "show",
                "list",
                "files",
                "status",
                "validate",
                "history",
                "path",
            }:
                return "read-only"
            if args[0] in {"discard", "accept"}:
                return "destructive"
            if args[0] == "implement":
                return "execution"
            return "configuration"
        if command == "project" and args[:1] == ["show"]:
            return "read-only"
        if command == "models" and (
            not args or args[0] in {"show", "list", "check"}
        ):
            return "read-only"
        if command == "editor" and (not args or args[0] == "show"):
            return "read-only"
        if command == "project" and args[:1] == ["reset"]:
            return "destructive"
        if command == "models" and args[:1] == ["remove"]:
            return "destructive"
        if command in {
            "deploy",
            "restart",
            "resume",
            "run",
            "start",
            "stop",
        }:
            return "execution"
        return "configuration"

    def _natural_preview_requested(self, request_text: str) -> bool:
        text = " ".join(request_text.strip().casefold().split())
        return bool(
            re.match(
                r"^(?:how (?:do|can|would) i|what command|which command|"
                r"tell me how|show me how|how should i)\b",
                text,
            )
        )

    def _natural_command_after_confirmation(self, command_line: str) -> str:
        """Avoid repeating an equivalent confirmation inside a planned command."""

        parts = shlex.split(command_line)
        lowered = [value.casefold() for value in parts]
        if (
            tuple(lowered[:2]) in {("spec", "discard"), ("task", "close")}
            and "--yes" not in lowered
        ):
            parts.append("--yes")
        return shlex.join(parts)

    def _emit_natural_plan(
        self,
        request_text: str,
        plan: NaturalCommandPlan,
        *,
        risk: CommandRisk | None,
        preview: bool,
    ) -> None:
        source = {
            "deterministic": "built-in interpreter",
            "learned": (
                "private learned interpretation"
                + (f" {plan.learned_id}" if plan.learned_id else "")
            ),
            "codex": "Codex CLI (read-only)",
            "claude": "Claude Code CLI (read-only)",
        }[plan.source]
        rows: list[tuple[str, object, StatusKind | None]] = [
            ("Request", request_text.strip(), None),
            ("Interpreter", source, None),
            ("Meaning", plan.summary, None),
        ]
        if plan.clarification:
            rows.append(("Status", "clarification required", "warning"))
        else:
            rows.extend(
                [
                    ("Safety", risk or "not classified", None),
                    (
                        "Mode",
                        "preview only" if preview else "execute",
                        "warning" if preview else "success",
                    ),
                ]
            )
        self._emit_table("Natural-language request", rows)
        if plan.commands:
            self._emit("Command plan")
            self._emit("────────────")
            for index, command in enumerate(plan.commands, start=1):
                self._emit(f"  {index}  {command}")
            self._emit()
        if plan.clarification:
            self._emit_table(
                "Clarification needed",
                [("Question", plan.clarification, "warning")],
            )

    def interpret_natural_language(
        self,
        request_text: str,
        *,
        preview_only: bool = False,
    ) -> None:
        request_text = request_text.strip()
        if not request_text:
            raise SystemExit("Describe the Studio operation you want.")
        if looks_sensitive(request_text):
            raise SystemExit(
                "The request appears to contain a secret value and was not sent "
                "to an interpreter or stored. Use 'models configure' so "
                "Studio can collect the key privately."
            )

        store = self._language_store()
        settings = self._global_settings()
        participants, active = self._language_participants()
        configurations = {
            name: configuration["spec"]
            for name, configuration
            in self.workspace.model_configurations().items()
        }
        plan = deterministic_plan(
            request_text,
            participants=participants,
            llm_participants=active,
            model_configurations=configurations,
        )
        if plan is None:
            plan = store.match(request_text)
        if plan is None:
            try:
                plan = self._interpret_with_cli(
                    request_text,
                    configured=str(settings.get("interpreter") or "auto"),
                )
            except (SystemExit, KeyboardInterrupt) as exc:
                store.record(
                    request_text,
                    None,
                    status="failed",
                    detail=str(exc),
                )
                raise

        proposed_text = "\n".join(
            (
                plan.summary,
                plan.clarification or "",
                *plan.commands,
            )
        )
        if looks_sensitive(proposed_text):
            store.record(
                request_text,
                None,
                status="failed",
                detail="interpreter output contained secret-looking text",
            )
            raise SystemExit(
                "The interpreter returned secret-looking text. Studio discarded "
                "the plan without displaying, executing, or learning it."
            )

        try:
            canonical_commands = tuple(
                self._canonical_natural_command(command)
                for command in plan.commands
            )
        except SystemExit:
            store.record(
                request_text,
                None,
                status="failed",
                detail="interpreter proposed unsupported Studio syntax",
            )
            raise
        plan = NaturalCommandPlan(
            plan.summary,
            canonical_commands,
            plan.source,
            clarification=plan.clarification,
            learned_id=plan.learned_id,
        )
        if plan.clarification:
            self._emit_natural_plan(
                request_text,
                plan,
                risk=None,
                preview=True,
            )
            store.record(request_text, plan, status="clarification")
            return

        risks = [self._natural_command_risk(command) for command in plan.commands]
        order: tuple[CommandRisk, ...] = (
            "read-only",
            "configuration",
            "execution",
            "destructive",
        )
        risk: CommandRisk = "read-only"
        for command_risk in risks:
            if order.index(command_risk) > order.index(risk):
                risk = cast(CommandRisk, command_risk)
        preview = preview_only or self._natural_preview_requested(request_text)
        self._emit_natural_plan(
            request_text,
            plan,
            risk=risk,
            preview=preview,
        )
        if preview:
            self._info("Plan shown without execution.")
            store.record(request_text, plan, status="previewed")
            return

        confirmed = False
        if risk in {"execution", "destructive"}:
            if not self._confirm_action(
                f"Execute this {risk} plan? [y/n]: ",
                cancel_message=(
                    "Natural-language plan cancelled; nothing was executed."
                ),
                default=False,
            ):
                store.record(request_text, plan, status="cancelled")
                return
            confirmed = True

        try:
            for index, command in enumerate(plan.commands, start=1):
                if len(plan.commands) > 1:
                    self._info(
                        f"Executing {index}/{len(plan.commands)}: {command}"
                    )
                execution_command = (
                    self._natural_command_after_confirmation(command)
                    if confirmed
                    else command
                )
                result = self.execute(
                    execution_command,
                    _allow_natural=False,
                )
                if not result:
                    raise SystemExit(
                        "A natural-language plan may not leave Studio."
                    )
        except (SystemExit, WorkspaceError, ValueError) as exc:
            store.record(
                request_text,
                plan,
                status="failed",
                detail=str(exc),
            )
            raise

        learned = None
        if plan.source in {"codex", "claude"}:
            learned = store.remember(
                request_text,
                plan,
                enabled=bool(self._global_settings().get("learning", True)),
            )
        store.record(request_text, plan, status="executed")
        self._success("Natural-language command plan completed.")
        if learned is not None:
            self._success(
                "Learned private interpretation "
                f"{learned['id']}; inspect it with 'language learned'."
            )

    def manage_language(self, args: list[str]) -> None:
        store = self._language_store()
        global_settings = self._global_settings()
        if not args or args == ["show"]:
            state = global_settings
            configured = str(state.get("interpreter") or "auto")
            selected = self._language_backend(configured, required=False)
            if configured == "off":
                effective = "off; deterministic and learned requests only"
                effective_kind: StatusKind = "warning"
            elif selected is None:
                effective = (
                    f"{configured}; no supported interpreter CLI was found"
                )
                effective_kind = "warning"
            else:
                backend = "Codex CLI" if selected[0] == "codex" else "Claude Code"
                effective = (
                    f"{configured} → {backend}"
                    if configured == "auto"
                    else backend
                )
                effective_kind = "success"
            learned = store.learned()
            history = store.history()
            self._emit_table(
                "Natural-language commands",
                [
                    ("Interpreter", effective, effective_kind),
                    (
                        "Learning",
                        "on; successful CLI plans are remembered privately"
                        if state.get("learning", True)
                        else "off",
                        "success" if state.get("learning", True) else "warning",
                    ),
                    ("Learned", len(learned), None),
                    ("History", len(history), None),
                    ("Settings", self.workspace.global_settings_path, None),
                    ("Project data", store.path, None),
                    (
                        "Privacy",
                        "owner-private; secret-looking requests are rejected",
                        "success",
                    ),
                ],
            )
            return

        action, *rest = args
        action = action.casefold()
        if action == "set" and len(rest) == 1:
            mode = rest[0].casefold()
            if mode not in {"auto", "codex", "claude", "off"}:
                raise SystemExit("Use language set auto|codex|claude|off.")
            if mode in {"codex", "claude"}:
                self._language_backend(mode, required=True)
            self.workspace.update_global_settings(interpreter=mode)
            self._success(
                f"Global natural-language CLI fallback: {mode}"
            )
            return
        if action == "learning" and len(rest) == 1:
            value = rest[0].casefold()
            if value not in {"on", "off"}:
                raise SystemExit("Use language learning on|off.")
            self.workspace.update_global_settings(learning=value == "on")
            self._success(f"Global natural-language learning: {value}")
            return
        if action == "history" and not rest:
            history = list(reversed(store.history()))
            if not history:
                self._emit_table(
                    "Natural-language history",
                    [("Status", "no interpreted requests yet", "warning")],
                )
                return
            self._emit("Natural-language history")
            self._emit("────────────────────────")
            for record in history[:25]:
                status = str(record.get("status") or "unknown")
                kind: StatusKind = (
                    "success"
                    if status == "executed"
                    else "warning"
                    if status in {"previewed", "clarification", "cancelled"}
                    else "error"
                )
                commands = record.get("commands") or []
                self._emit(
                    f"  {self._status_mark(kind)} {record.get('id')}  "
                    f"{status}  {record.get('request')}"
                )
                if commands:
                    self._emit("      " + " · ".join(map(str, commands)))
            self._emit()
            return
        if action == "learned" and not rest:
            learned = store.learned()
            if not learned:
                self._emit_table(
                    "Learned interpretations",
                    [
                        (
                            "Status",
                            "none; CLI fallback results will appear here",
                            "warning",
                        )
                    ],
                )
                return
            self._emit("Learned interpretations")
            self._emit("───────────────────────")
            for record in learned:
                self._emit(
                    f"  {record.get('id')}  {record.get('request_template')}"
                )
                commands = record.get("commands") or []
                self._emit("      " + " · ".join(map(str, commands)))
                self._emit(
                    f"      source: {record.get('source')}; "
                    f"uses: {record.get('uses', 0)}"
                )
            self._emit()
            return
        if action == "forget" and len(rest) == 1:
            identifier = rest[0]
            if identifier.casefold() == "all" and not self._confirm_action(
                "Forget all learned natural-language interpretations? [y/n]: ",
                cancel_message="Nothing was forgotten.",
                default=False,
            ):
                return
            removed = store.forget(identifier)
            if not removed:
                raise SystemExit(
                    f"No learned interpretation matches {identifier!r}."
                )
            self._success(
                f"Forgot {removed} learned interpretation"
                f"{'s' if removed != 1 else ''}."
            )
            return
        raise SystemExit(
            "Use language, language set auto|codex|claude|off, "
            "language learning on|off, language history, language learned, "
            "or language forget ID|all."
        )

    def _request_prompt(
        self,
        args: list[str],
        *,
        command: str,
        draft_content: str | None = None,
    ) -> _PromptInput:
        if not args:
            return _PromptInput("")
        if args[0] == "--file":
            if len(args) != 2:
                raise SystemExit(f"Use {command} --file PATH.")
            entered = Path(args[1]).expanduser()
            prompt_file = (
                entered
                if entered.is_absolute()
                else self.workspace.root / entered
            ).resolve()
            return _PromptInput(
                self._read_prompt_file(prompt_file),
                source_path=prompt_file,
            )
        if args[0] == "--edit":
            edit_args = args[1:]
            editor_override = None
            if "--editor" in edit_args:
                index = edit_args.index("--editor")
                if index != len(edit_args) - 2:
                    raise SystemExit(
                        f"Use {command} --edit [PATH] [--editor COMMAND]."
                    )
                editor_override = edit_args[-1]
                edit_args = edit_args[:index]
            if len(edit_args) > 1:
                raise SystemExit(
                    f"Use {command} --edit [PATH] [--editor COMMAND]."
                )
            self.workspace.initialize_project()
            if edit_args:
                prompt_file = self._project_path(edit_args[0], label="Prompt")
                draft_path = None
            else:
                prompt_file = self._new_prompt_draft(
                    command,
                    content=draft_content,
                )
                draft_path = prompt_file
            if prompt_file == self.workspace.prompt_index_path.resolve():
                raise SystemExit(
                    f"Prompt path is reserved for the managed ledger: "
                    f"{prompt_file}"
                )
            prompt_file.parent.mkdir(parents=True, exist_ok=True)
            self._launch_editor(prompt_file, override=editor_override)
            return _PromptInput(
                self._read_prompt_file(prompt_file),
                source_path=prompt_file,
                draft_path=draft_path,
            )
        if "--file" in args or "--edit" in args:
            raise SystemExit(
                f"Use {command} --file PATH or {command} --edit [PATH]."
            )
        return _PromptInput(" ".join(args).strip())

    def _new_prompt_draft(
        self,
        purpose: str,
        *,
        content: str | None = None,
    ) -> Path:
        draft_directory = self.workspace.root / ".zippergen" / "prompt-drafts"
        draft_directory.mkdir(parents=True, exist_ok=True)
        label = "".join(
            character.lower() if character.isalnum() else "-"
            for character in purpose
        ).strip("-")
        while "--" in label:
            label = label.replace("--", "-")
        label = label or "prompt"
        identifier = (
            f"{time.strftime('%Y%m%d-%H%M%S')}-"
            f"{time.time_ns() % 1_000_000_000:09d}"
        )
        draft = draft_directory / f"{identifier}-{label}.md"
        if content is None:
            draft.touch(exist_ok=False)
        else:
            draft.write_text(content.rstrip() + "\n", encoding="utf-8")
        return draft

    def _finish_prompt_input(self, prompt_input: _PromptInput) -> None:
        draft = prompt_input.draft_path
        if draft is None:
            return
        try:
            draft.unlink()
        except FileNotFoundError:
            return
        except OSError as exc:
            self._warning(
                f"Registered prompt, but could not remove draft {draft}: {exc}"
            )

    def _read_prompt_file(self, prompt_file: Path) -> str:
        try:
            prompt = prompt_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            raise SystemExit(
                f"Prompt file does not exist: {prompt_file}"
            ) from None
        except IsADirectoryError:
            raise SystemExit(
                f"Prompt path is a directory: {prompt_file}"
            ) from None
        except UnicodeDecodeError:
            raise SystemExit(
                f"Prompt file must contain UTF-8 text: {prompt_file}"
            ) from None
        except OSError as exc:
            raise SystemExit(
                f"Could not read prompt file {prompt_file}: {exc}"
            ) from exc
        if not prompt:
            raise SystemExit(f"Prompt file is empty: {prompt_file}")
        return prompt

    def _prepare_specification_editor(self, target: Path) -> None:
        """Place a removable guide in a specification that has no intent yet."""

        if self.workspace.specification() is not None:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(SPECIFICATION_GUIDE, encoding="utf-8")

    def _finish_specification_editor(self, target: Path) -> str:
        """Read user intent, remove the guide, and retain it after empty edits."""

        prompt = self.workspace.specification()
        if prompt is None:
            self._prepare_specification_editor(target)
            raise SystemExit(
                "No application requirements were written. The specification "
                "guide was kept; enter 'workflow create' and write below its "
                "comment."
            )
        self.workspace.save_specification(prompt)
        return prompt

    def _editor_override(
        self,
        args: list[str],
        *,
        usage: str,
    ) -> tuple[list[str], str | None]:
        values = list(args)
        if "--editor" not in values:
            return values, None
        index = values.index("--editor")
        if index != len(values) - 2:
            raise SystemExit(usage)
        return values[:index], values[-1]

    def create_from_command(self, args: list[str]) -> None:
        """Create a handoff while keeping specification filenames automatic."""

        values, editor_override = self._editor_override(
            args,
            usage="Use workflow create [DESCRIPTION], workflow create "
            "--file PATH, or workflow create [--edit] [--editor COMMAND].",
        )
        if not values or values == ["--edit"]:
            self.workspace.initialize_project()
            ensured = self.workspace.ensure_specification()
            target = self.workspace.specification_path
            self._prepare_specification_editor(target)
            self._launch_editor(target, override=editor_override)
            prompt = self._finish_specification_editor(target)
            self.create_request(prompt, specification_already_saved=True)
            if ensured["migrated"]:
                self._info(
                    "The former active prompt ledger was migrated into the "
                    "canonical specification; its original files were kept."
                )
            return
        if editor_override is not None:
            raise SystemExit(
                "--editor is only used when workflow create opens the "
                "specification editor."
            )
        if values[0] == "--file":
            if len(values) != 2:
                raise SystemExit("Use workflow create --file PATH.")
            entered = Path(values[1]).expanduser()
            source = (
                entered
                if entered.is_absolute()
                else self.workspace.root / entered
            ).resolve()
            prompt = self._read_prompt_file(source)
        elif "--file" in values or "--edit" in values:
            raise SystemExit(
                "Use workflow create [DESCRIPTION], workflow create --file "
                "PATH, or plain workflow create to open the automatic "
                "specification file."
            )
        else:
            prompt = " ".join(values).strip()
        self.create_request(prompt)

    def _show_specification(self) -> None:
        ensured = self.workspace.ensure_specification()
        content = ensured["content"]
        if content is None:
            self._emit_table(
                "Workflow specification",
                [
                    (
                        "Status",
                        "not written; use workflow create or workflow edit spec",
                        "warning",
                    ),
                    ("File", self.workspace.specification_path, None),
                ],
            )
            return
        self._emit_table(
            "Workflow specification",
            [
                ("Status", "canonical", "success"),
                ("File", self.workspace.specification_path, None),
                (
                    "Pending",
                    "yes; use workflow show pending"
                    if self.workspace.pending_refinement() is not None
                    else "none",
                    "warning"
                    if self.workspace.pending_refinement() is not None
                    else None,
                ),
            ],
        )
        self._emit("Requirements")
        self._emit("────────────")
        self._emit(str(content))
        self._emit()
        if ensured["migrated"]:
            self._info(
                "Migrated the former active prompt ledger into specification.md; "
                "the original prompt files were kept."
            )

    def _confirm_spec_action(self, question: str) -> bool:
        return self._confirm_action(
            question,
            cancel_message="Specification action cancelled; nothing was changed.",
        )

    def _confirm_action(
        self,
        question: str,
        *,
        cancel_message: str,
        default: bool | None = None,
    ) -> bool:
        while True:
            try:
                answer = self.input(question).strip().lower()
            except (EOFError, KeyboardInterrupt):
                self._warning(cancel_message)
                return False
            if not answer and default is not None:
                return default
            if answer in {"y", "yes"}:
                return True
            if answer in {"n", "no"}:
                self._warning(cancel_message)
                return False
            self._warning("Please enter 'y' or 'n'.")

    def manage_spec(self, args: list[str]) -> None:
        """Manage one canonical specification and one pending refinement."""

        if not args or args == ["show"]:
            self._show_specification()
            return
        action, *rest = args
        action = action.lower()
        if action == "path" and not rest:
            self.workspace.initialize_project()
            self._emit(self.workspace.specification_path)
            return
        if action == "edit":
            values, editor_override = self._editor_override(
                rest,
                usage="Use workflow edit spec [--editor COMMAND].",
            )
            if values:
                raise SystemExit(
                    "Use workflow edit spec [--editor COMMAND]."
                )
            self.workspace.initialize_project()
            ensured = self.workspace.ensure_specification()
            target = self.workspace.specification_path
            self._prepare_specification_editor(target)
            self._launch_editor(target, override=editor_override)
            self._finish_specification_editor(target)
            self._emit_table(
                "Specification updated",
                [
                    ("File", target, "success"),
                    ("Pending", "unchanged", None),
                    (
                        "Next",
                        "workflow status · workflow implement · "
                        "workflow validate",
                        None,
                    ),
                ],
            )
            if ensured["migrated"]:
                self._info(
                    "The former prompt ledger was migrated before editing; "
                    "its original files were kept."
                )
            return
        if action == "pending" and not rest:
            pending = self.workspace.pending_refinement()
            if pending is None:
                self._emit_table(
                    "Pending refinement",
                    [("Status", "none; use workflow refine", None)],
                )
                return
            request_record = self._ensure_current_task_fresh(announce=False)
            task_record = (
                request_record
                if request_record and request_record.get("kind") == "refine"
                else None
            )
            task_status = (
                str(task_record.get("status") or "prepared")
                if task_record
                else "prepared"
            )
            if task_status == "awaiting_review":
                pending_status = "assistant returned; awaiting human review"
                pending_kind: StatusKind = "warning"
                assert task_record is not None
                next_action = self._task_next(task_record)
            elif task_status == "assistant_running":
                pending_status = "assistant is integrating the change"
                pending_kind = "info"
                next_action = "wait for the assistant session to return"
            elif task_status in {"assistant_failed", "assistant_interrupted"}:
                pending_status = "assistant did not finish; refinement remains open"
                pending_kind = "error"
                next_action = (
                    "workflow status · workflow implement codex · "
                    "workflow implement claude"
                )
            else:
                pending_status = "waiting to be integrated"
                pending_kind = "warning"
                next_action = (
                    "workflow implement codex · workflow implement claude"
                )
            self._emit_table(
                "Pending refinement",
                [
                    ("Status", pending_status, pending_kind),
                    *(
                        [("Verification", *self._task_verification(task_record))]
                        if task_record
                        else []
                    ),
                    ("File", ".zippergen/pending-refinement.md", None),
                    ("Edit", "workflow refine", None),
                    ("Next", next_action, None),
                ],
            )
            self._emit("Requested change")
            self._emit("────────────────")
            self._emit(pending)
            self._emit()
            return
        if action == "refine":
            if self.workspace.current_workflow is None:
                raise SystemExit(
                    "No workflow selected. Use 'workflow select' before preparing "
                    "a refinement."
                )
            ensured = self.workspace.ensure_specification()
            if ensured["content"] is None:
                raise SystemExit(
                    "No workflow specification exists. Use 'workflow create' "
                    "or 'workflow edit spec' first."
                )
            values, editor_override = self._editor_override(
                rest,
                usage="Use workflow refine [CHANGE|--file PATH] "
                "[--editor COMMAND].",
            )
            existing = self.workspace.pending_refinement()
            if not values or values == ["--edit"]:
                target = self.workspace.begin_pending_refinement()
                self._launch_editor(target, override=editor_override)
                refinement = self._read_prompt_file(target)
                append = False
            elif values[0] == "--file":
                if len(values) != 2 or editor_override is not None:
                    raise SystemExit("Use workflow refine --file PATH.")
                entered = Path(values[1]).expanduser()
                source = (
                    entered
                    if entered.is_absolute()
                    else self.workspace.root / entered
                ).resolve()
                refinement = self._read_prompt_file(source)
                append = existing is not None
            elif "--file" in values or "--edit" in values or editor_override is not None:
                raise SystemExit(
                    "Use workflow refine [CHANGE|--file PATH] "
                    "[--editor COMMAND]."
                )
            else:
                refinement = " ".join(values).strip()
                append = existing is not None
            self.refine_request(refinement, append=append)
            if ensured["migrated"]:
                self._info(
                    "The former active prompt ledger was migrated into the "
                    "canonical specification; its original files were kept."
                )
            return
        if action in {"reconcile", "discard"}:
            if rest not in ([], ["--yes"]):
                public_action = "accept" if action == "reconcile" else "discard"
                raise SystemExit(f"Use workflow {public_action} [--yes].")
            pending = self.workspace.pending_refinement()
            if pending is None:
                raise SystemExit("There is no pending refinement.")
            if action == "reconcile":
                baseline = self.workspace.load().get(
                    "pending_specification_fingerprint"
                )
                current = self.workspace.specification_fingerprint(
                    include_pending=False
                )
                if baseline == current:
                    raise SystemExit(
                        "The canonical specification has not changed since this "
                        "refinement began. Run 'workflow implement' or use "
                        "'workflow edit spec' to integrate the change before "
                        "accepting it."
                    )
            if rest != ["--yes"]:
                verb = "Accept and clear" if action == "reconcile" else "Discard"
                if not self._confirm_spec_action(
                    f"{verb} the pending refinement? [y/n]: "
                ):
                    return
            result = self.workspace.archive_pending_refinement(
                status="reconciled" if action == "reconcile" else "discarded"
            )
            self._emit_table(
                "Specification refinement",
                [
                    (
                        "Status",
                        result["status"],
                        "success" if action == "reconcile" else "warning",
                    ),
                    (
                        "Canonical",
                        (
                            "existing integration accepted; no automatic merge "
                            "was performed"
                            if action == "reconcile"
                            else "unchanged by discard"
                        ),
                        "success" if action == "reconcile" else None,
                    ),
                    ("Pending", "cleared", "success"),
                    (
                        "Implementation",
                        "accepted; private history retained",
                        "success",
                    ),
                    ("History", result["history_path"], None),
                    ("Next", "workflow show spec · current", None),
                ],
            )
            return
        if action == "history" and not rest:
            records = self.workspace.list_spec_history()
            if not records:
                self._emit_table(
                    "Specification history",
                    [
                        (
                            "Status",
                            "none; accepted specification history lives in Git",
                            None,
                        )
                    ],
                )
                return
            self._emit("Specification refinement history")
            self._emit("────────────────────────────────")
            self._emit("  Status       Created                    Archived")
            for record in records:
                self._emit(
                    f"  {str(record.get('status') or 'unknown'):<12} "
                    f"{str(record.get('created_at') or '—'):<26} "
                    f"{record.get('archived_at') or '—'}"
                )
            self._emit()
            self._emit("Canonical specification history is versioned by Git.")
            return
        raise SystemExit(
            "Use workflow show spec, workflow edit spec, workflow path, "
            "workflow refine, workflow show pending, workflow accept [--yes], "
            "workflow discard [--yes], or workflow history."
        )

    def _project_path(self, value: str | Path, *, label: str = "File") -> Path:
        entered = Path(value).expanduser()
        path = (
            entered if entered.is_absolute() else self.workspace.root / entered
        ).resolve()
        if not path.is_relative_to(self.workspace.root):
            raise SystemExit(
                f"{label} must be inside the project root: {self.workspace.root}"
            )
        if path.is_dir():
            raise SystemExit(f"{label} path is a directory: {path}")
        return path

    def _parse_editor_command(self, value: object) -> list[str]:
        if isinstance(value, list):
            command = [str(part) for part in value if str(part)]
        else:
            try:
                command = shlex.split(str(value))
            except ValueError as exc:
                raise SystemExit(f"Could not parse editor command: {exc}") from exc
        if not command:
            raise SystemExit("Editor command must not be empty.")
        return command

    def _effective_editor(
        self,
        override: str | None = None,
    ) -> tuple[list[str], str]:
        if override is not None:
            candidates = [(self._parse_editor_command(override), "one-off")]
        else:
            configured = self._global_settings().get("editor_command")
            if configured:
                candidates = [
                    (self._parse_editor_command(configured), "global preference")
                ]
            else:
                candidates = []
                for variable in ("VISUAL", "EDITOR"):
                    value = os.environ.get(variable)
                    if value:
                        candidates.append(
                            (self._parse_editor_command(value), f"${variable}")
                        )
                candidates.extend(
                    ([name], "automatic")
                    for name in ("micro", "nano", "vim", "vi")
                )
        for command, source in candidates:
            executable = shutil.which(command[0])
            if executable is not None:
                return [executable, *command[1:]], source
            if source in {"one-off", "global preference"}:
                raise SystemExit(
                    f"Editor executable was not found: {command[0]}. "
                    "Use 'editor set COMMAND' or 'editor reset'."
                )
        raise SystemExit(
            "No terminal editor was found. Install micro/nano/vim, set $VISUAL "
            "or $EDITOR, or use 'editor set COMMAND'."
        )

    def configure_editor(self, args: list[str]) -> None:
        self._global_settings()
        if not args or args == ["show"]:
            command, source = self._effective_editor()
            preference = self._global_settings().get("editor_command")
            self._emit_table(
                "Editor",
                [
                    (
                        "Preference",
                        shlex.join(self._parse_editor_command(preference))
                        if preference
                        else "automatic",
                        None,
                    ),
                    ("Effective", shlex.join(command), "success"),
                    ("Source", source, None),
                ],
            )
            return
        action, *rest = args
        if action == "set" and rest:
            command = self._parse_editor_command(rest)
            executable = shutil.which(command[0])
            if executable is None:
                raise SystemExit(f"Editor executable was not found: {command[0]}.")
            self.workspace.update_global_settings(editor_command=command)
            self._success(
                f"Global editor preference: {shlex.join(command)}"
            )
            return
        if action == "reset" and not rest:
            self.workspace.reset_global_settings("editor_command")
            self._success(
                "Global editor preference reset to automatic discovery."
            )
            return
        raise SystemExit("Use editor, editor show, editor set COMMAND, or editor reset.")

    def _launch_editor(
        self,
        target: Path,
        *,
        override: str | None = None,
    ) -> None:
        command, source = self._effective_editor(override)
        try:
            displayed = target.relative_to(self.workspace.root)
        except ValueError:
            displayed = target
        self._emit_table(
            "Editor",
            [
                ("Command", shlex.join(command), None),
                ("Source", source, None),
                ("File", displayed, None),
            ],
        )
        try:
            completed = subprocess.run(
                [*command, str(target)],
                cwd=self.workspace.root,
                check=False,
            )
        except OSError as exc:
            raise SystemExit(f"Could not start editor: {exc}") from exc
        if completed.returncode != 0:
            raise SystemExit(
                f"Editor exited with status {completed.returncode}: {displayed}"
            )
        self._success(f"Editor closed: {displayed}")

    def edit_file(self, args: list[str]) -> None:
        editor_override = None
        if "--editor" in args:
            index = args.index("--editor")
            if index != len(args) - 2:
                raise SystemExit(
                    "Use workflow edit code [--editor COMMAND] or "
                    "edit file PATH [--editor COMMAND]."
                )
            editor_override = args[-1]
            args = args[:index]
        if not args or args == ["workflow"]:
            current = self._ensure_workflow_selected("edit its source")
            module_ref = self.workspace.absolute_spec(current).partition(":")[0]
            target = Path(module_ref)
            if target.suffix != ".py" or not target.is_file():
                raise SystemExit(
                    f"The selected workflow is not backed by a Python file: {current}"
                )
            target = self._project_path(target, label="Workflow")
            next_steps = "workflow validate · workflow show · run"
        elif len(args) == 2 and args[0] == "file":
            target = self._project_path(args[1])
            next_steps = "inspect the change; this generic edit was not registered"
        elif len(args) == 1:
            target = self._project_path(args[0])
            next_steps = "inspect the change; this generic edit was not registered"
        else:
            raise SystemExit(
                "Use workflow edit code [--editor COMMAND] or "
                "edit file PATH [--editor COMMAND]."
            )
        self._launch_editor(target, override=editor_override)
        self._emit(f"Next: {next_steps}")

    def _ensure_workflow_selected(self, purpose: str) -> str:
        current = self.workspace.current_workflow
        if current:
            return current
        candidates = self.workspace.discover_workflows()
        if not candidates:
            raise SystemExit(
                "No workflow entry points were discovered. Use 'workflow list' "
                "to confirm discovery, then inspect the generated Python files "
                "for a top-level @workflow definition."
            )
        if len(candidates) == 1:
            selected = candidates[0]
            automatic = True
        else:
            selected = self._select(
                f"Choose a workflow to {purpose}",
                candidates,
            )
            automatic = False
        assert isinstance(selected, str)
        canonical, name = self._select_workflow_spec(selected)
        message = (
            f"Automatically selected {canonical} ({name}) to {purpose}"
            if automatic
            else f"Selected {canonical} ({name}) to {purpose}"
        )
        self._info(f"{message}; validation has not run.")
        return canonical

    def _current_context(self, *, purpose: str = "inspect it"):
        from zippergen.serve import load_workflow_spec

        current = self._ensure_workflow_selected(purpose)
        workflow, module = load_workflow_spec(self.workspace.absolute_spec(current))
        return current, workflow, module

    def configure_project(self, args: list[str]) -> None:
        if not args or args == ["show"]:
            manifest = self.workspace.project_manifest()
            self._emit_table(
                "Project",
                [
                    ("Name", manifest["name"], None),
                    ("Root", self.workspace.root, None),
                    (
                        "Manifest",
                        f"{self.workspace.manifest_path} "
                        f"({'present' if manifest['exists'] else 'not created'})",
                        "success" if manifest["exists"] else "warning",
                    ),
                    (
                        "Specification",
                        self.workspace.specification_path,
                        "success"
                        if self.workspace.specification() is not None
                        else "warning",
                    ),
                    (
                        "Pending",
                        ".zippergen/pending-refinement.md"
                        if self.workspace.pending_refinement() is not None
                        else "none",
                        "warning"
                        if self.workspace.pending_refinement() is not None
                        else None,
                    ),
                    (
                        "Framework checkout",
                        manifest.get("framework_directory") or "none",
                        None,
                    ),
                ],
            )
            return
        if args[0] == "rename":
            if len(args) != 2:
                raise SystemExit("Use project rename NAME.")
            try:
                result = self.workspace.rename_project(args[1])
            except WorkspaceError as exc:
                raise SystemExit(str(exc)) from exc
            self._emit_table(
                "Project renamed",
                [
                    ("From", result["old_name"], None),
                    ("To", result["new_name"], "success"),
                    ("Manifest", result["manifest"], None),
                    (
                        "Root",
                        f"{result['root']} (unchanged)",
                        "success",
                    ),
                    (
                        "Scope",
                        "logical project name only; workflows and deployments unchanged",
                        None,
                    ),
                ],
            )
            return
        if args[0] == "reset":
            rest = args[1:]
            if not rest:
                selected = self._select(
                    "Choose reset scope",
                    [
                        "Fresh design cycle — archive manifest, specification, "
                        "legacy prompts, and private Studio state",
                        "Studio state only — keep manifest, specification, "
                        "source, tests, and Git",
                        "Cancel — change nothing",
                    ],
                )
                choice = str(selected)
                if choice.startswith("Cancel"):
                    self._warning("Project reset cancelled; nothing was changed.")
                    return
                mode = "fresh" if choice.startswith("Fresh") else "state"
                self.reset_project(mode=mode, confirm=True)
                return
            if rest[0] not in {"fresh", "state"} or rest[1:] not in (
                [],
                ["--yes"],
            ):
                raise SystemExit(
                    "Use project reset, project reset fresh [--yes], or "
                    "project reset state [--yes]."
                )
            explicit_mode: Literal["fresh", "state"] = (
                "fresh" if rest[0] == "fresh" else "state"
            )
            self.reset_project(
                mode=explicit_mode,
                confirm=rest[1:] != ["--yes"],
            )
            return
        if args[0] != "init" or len(args) > 2:
            raise SystemExit(
                "Use project show, project init [NAME], project rename NAME, "
                "project reset, project reset fresh [--yes], or "
                "project reset state [--yes]."
            )
        existed = self.workspace.manifest_path.exists()
        manifest = self.workspace.initialize_project(
            name=args[1] if len(args) == 2 else None
        )
        result = "already exists" if existed else "created"
        self._success(
            f"Project manifest {result}: {self.workspace.manifest_path}"
        )
        self._emit(f"Project: {manifest['name']}")
        self._emit(f"Specification: {self.workspace.specification_path}")

    def reset_project(
        self,
        *,
        mode: Literal["fresh", "state"],
        confirm: bool = True,
    ) -> None:
        summary = self.workspace.private_state_summary()
        project_exists = self.workspace.root.is_dir()
        manifest_exists = self.workspace.manifest_path.exists()
        specification_exists = self.workspace.specification_path.exists()
        legacy_prompts_exist = (
            self.workspace.prompts_directory != self.workspace.root
            and self.workspace.prompts_directory.exists()
        )
        private_exists = bool(
            summary["workspace_exists"] or summary["project_local_exists"]
        )
        visible_design_exists = bool(
            manifest_exists or specification_exists or legacy_prompts_exist
        )
        if mode == "state" and not private_exists:
            self._warning(
                "Private Studio state is already empty. The manifest, "
                "specification, source, tests, and Git were not changed."
            )
            return
        if mode == "fresh" and not private_exists and not visible_design_exists:
            self._warning(
                "This project already has no manifest, specification, legacy "
                "prompts, or private Studio state. Source, tests, and Git were "
                "not changed."
            )
            return

        git_exists = (self.workspace.root / ".git").exists()
        fresh = mode == "fresh"
        self._emit_table(
            "Project reset preview",
            [
                (
                    "Mode",
                    "fresh design cycle" if fresh else "Studio state only",
                    "warning" if fresh else "info",
                ),
                (
                    "Project",
                    (
                        self.workspace.root
                        if project_exists
                        else f"{self.workspace.root} (missing)"
                    ),
                    "success" if project_exists else "warning",
                ),
                ("Workflow source and tests", "kept", "success"),
                (
                    "Manifest",
                    "archive" if fresh and manifest_exists else "kept",
                    "warning" if fresh and manifest_exists else "success",
                ),
                (
                    "Specification",
                    "archive" if fresh and specification_exists else "kept",
                    "warning" if fresh and specification_exists else "success",
                ),
                (
                    "Legacy prompts",
                    "archive" if fresh and legacy_prompts_exist else "kept/none",
                    "warning" if fresh and legacy_prompts_exist else None,
                ),
                (
                    "Git history",
                    "kept" if git_exists else "not present",
                    "success" if git_exists else None,
                ),
                (
                    "Private Studio state",
                    "archive" if private_exists else "already empty",
                    "warning" if private_exists else None,
                ),
                ("Managed runs", summary["runs"], None),
                ("Implementation requests", summary["requests"], None),
                ("Language history", summary["language_history"], None),
                ("Learned language", summary["language_learned"], None),
                ("Development secrets", summary["development_secrets"], None),
                (
                    "Deployments",
                    "kept and not stopped; remembered name is cleared",
                    "warning" if summary["last_deployment"] else None,
                ),
                (
                    "Next",
                    (
                        "project init · workflow create"
                        if fresh
                        else "workflow list · workflow select · workflow create · current"
                    ),
                    None,
                ),
            ],
        )
        if confirm:
            action = (
                "Start a fresh design cycle"
                if fresh
                else "Reset only private Studio state"
            )
            if not self._confirm_action(
                f"{action}? [y/n]: ",
                cancel_message="Project reset cancelled; nothing was changed.",
            ):
                return

        result = (
            self.workspace.reset_fresh_design()
            if fresh
            else self.workspace.reset_private_state()
        )
        # Command history was moved with the private workspace. Recreate the
        # prompt session on the next loop iteration.
        self._prompt_session = None
        backup = result["backup_directory"]
        self._emit_table(
            "Project reset",
            [
                (
                    "Mode",
                    "fresh design cycle" if fresh else "Studio state only",
                    "success",
                ),
                ("Status", "complete", "success"),
                ("Backup", backup or "none needed", "success"),
                ("Manifest", "not created" if fresh else "kept", None),
                ("Specification", "not written" if fresh else "kept", None),
                ("Source and tests", "kept", "success"),
                ("Workflow", "none selected", None),
                ("Run", "none selected", None),
                ("Implementation request", "none", None),
                (
                    "Next",
                    (
                        "project init · workflow create"
                        if fresh and project_exists
                        else "workflow list · workflow select · workflow create · current"
                        if project_exists
                        else "exit and recreate the project directory"
                    ),
                    None,
                ),
            ],
        )

    def _emit_prompt_list(self) -> None:
        records = self.workspace.list_prompts()
        if not records:
            self._emit_table(
                "Prompts",
                [
                    (
                        "Status",
                        "none; use workflow create or workflow refine",
                        "warning",
                    )
                ],
            )
            return
        active = sum(bool(record["active"]) for record in records)
        self._emit_table(
            "Prompt summary",
            [
                ("Active", active, "success"),
                ("Archived", len(records) - active, None),
                ("Total", len(records), None),
                ("Precedence", "later rows override only explicit conflicts", None),
            ],
        )
        self._emit("Prompt ledger")
        self._emit("─" * len("Prompt ledger"))
        self._emit(
            f"  {'#':>2}  {'ID':<5}  {'Kind':<10}  {'Status':<10}  "
            f"{'Title':<48}  File"
        )
        for position, record in enumerate(records, start=1):
            status = "active" if record["active"] else "archived"
            mark = self._status_mark(
                "success" if record["active"] else "warning"
            )
            title = str(record["title"])
            if len(title) > 48:
                title = title[:47] + "…"
            self._emit(
                f"  {position:>2}  {str(record['id']):<5}  "
                f"{str(record['kind']):<10}  {mark} {status:<8}  "
                f"{title:<48}  {record['file']}"
            )
        self._emit()

    def manage_prompts(self, args: list[str]) -> None:
        if not args or args == ["list"]:
            self._emit_prompt_list()
            return
        action, *rest = args
        action = action.lower()
        if (
            action
            not in {"show", "inspect", "path", "context"}
            and self.workspace.specification() is not None
        ):
            raise SystemExit(
                "This project now uses the canonical specification. Legacy "
                "prompts remain inspectable, but cannot be changed. Use "
                "'workflow edit spec' for the accepted specification or "
                "'workflow refine' for a pending change."
            )
        if action in {"show", "inspect"} and len(rest) == 1:
            record = self.workspace.prompt(rest[0])
            position = next(
                index
                for index, candidate in enumerate(
                    self.workspace.list_prompts(),
                    start=1,
                )
                if candidate["id"] == record["id"]
            )
            self._emit_table(
                f"Prompt {record['id']}",
                [
                    ("Position", position, None),
                    ("Kind", record["kind"], None),
                    (
                        "Status",
                        "active" if record["active"] else "archived",
                        "success" if record["active"] else "warning",
                    ),
                    ("Title", record["title"], None),
                    ("File", record["file"], None),
                    ("Replaces", record.get("replaces") or "none", None),
                ],
            )
            self._emit("Requirement")
            self._emit("─" * len("Requirement"))
            self._emit(str(record["content"]))
            self._emit()
            return
        if action == "path" and len(rest) == 1:
            record = self.workspace.prompt(rest[0])
            self._emit(self.workspace.root / str(record["file"]))
            return
        if action == "context" and not rest:
            self._emit(self.workspace.prompt_context())
            return
        if action == "add":
            prompt_input = self._request_prompt(rest, command="prompts add")
            prompt = prompt_input.content
            if not prompt:
                prompt = self.input("Describe the requirement: ").strip()
            if not prompt:
                raise SystemExit("The prompt must not be empty.")
            kind = "refinement"
            if (
                not self.workspace.list_prompts()
                and self.workspace.current_workflow is None
            ):
                kind = "initial"
            record = self.workspace.add_prompt(
                kind=kind,
                content=prompt,
                source_path=prompt_input.source_path,
                workflow_spec=self.workspace.current_workflow,
            )
            self._finish_prompt_input(prompt_input)
            status = "Registered" if record["created"] else "Already registered"
            self._success(
                f"{status}: {record['id']} [{record['kind']}] {record['file']}"
            )
            return
        if action == "edit" and len(rest) in {1, 3}:
            if len(rest) == 3 and rest[1] == "--editor":
                editor_override = rest[2]
            elif len(rest) == 1:
                editor_override = None
            else:
                raise SystemExit("Use prompts edit ID [--editor COMMAND].")
            original = self.workspace.prompt(rest[0])
            draft = self._new_prompt_draft(
                f"edit-{original['id']}",
                content=str(original["content"]),
            )
            self._launch_editor(draft, override=editor_override)
            content = self._read_prompt_file(draft)
            record = self.workspace.update_prompt_content(
                str(original["id"]),
                content=content,
            )
            self._finish_prompt_input(_PromptInput(content, draft_path=draft))
            self._emit_table(
                "Prompt updated",
                [
                    ("ID", record["id"], "success"),
                    ("Kind", record["kind"], None),
                    ("Title", record["title"], None),
                    ("File", record["file"], None),
                    ("Next", f"prompts show {record['id']}", None),
                ],
            )
            return
        if action in {
            "enable",
            "restore",
            "disable",
            "remove",
            "archive",
        } and len(rest) == 1:
            active = action in {"enable", "restore"}
            record = self.workspace.set_prompt_active(rest[0], active=active)
            verb = "Restored" if active else "Archived"
            self._success(f"{verb}: {record['id']} — {record['title']}")
            return
        if action == "move" and len(rest) == 3:
            self.workspace.move_prompt(
                rest[0],
                relation=rest[1].lower(),
                other_id=rest[2],
            )
            self._success(
                f"Moved {rest[0].upper()} {rest[1]} {rest[2].upper()}."
            )
            self._emit_prompt_list()
            return
        if action == "replace" and len(rest) >= 1:
            original = self.workspace.prompt(rest[0])
            prompt_input = self._request_prompt(
                rest[1:],
                command=f"prompts replace {rest[0]}",
                draft_content=str(original["content"]),
            )
            prompt = prompt_input.content
            if not prompt:
                prompt = self.input("Describe the replacement requirement: ").strip()
            if not prompt:
                raise SystemExit("The replacement prompt must not be empty.")
            record = self.workspace.replace_prompt(
                rest[0],
                content=prompt,
                source_path=prompt_input.source_path,
            )
            self._finish_prompt_input(prompt_input)
            self._success(
                f"Replaced {rest[0].upper()} with {record['id']}: "
                f"{record['file']}"
            )
            return
        raise SystemExit(
            "Use prompts; prompts show|inspect|path|edit ID; prompts add "
            "[--file PATH|--edit [PATH]|PROMPT]; prompts context; prompts "
            "archive|restore ID; prompts replace ID "
            "[--file PATH|--edit [PATH]|PROMPT]; or prompts move ID "
            "before|after ID."
        )

    def _normalize_task_lifecycle(
        self,
        record: dict[str, object],
    ) -> dict[str, object]:
        """Add lifecycle meaning to tasks created before lifecycle tracking."""

        if record.get("status") == "assistant_running":
            raw_pid = record.get("studio_process_id")
            process_is_live = False
            if isinstance(raw_pid, int) and raw_pid > 0:
                try:
                    os.kill(raw_pid, 0)
                except ProcessLookupError:
                    process_is_live = False
                except PermissionError:
                    process_is_live = True
                else:
                    process_is_live = True
            if not process_is_live:
                return self.workspace.update_request(
                    str(record["request_id"]),
                    status="assistant_interrupted",
                    assistant_finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    assistant_error=(
                        "the Studio process ended before the assistant returned"
                    ),
                    result_specification_fingerprint=(
                        self.workspace.specification_fingerprint()
                    ),
                )
            return record
        if record.get("status"):
            return record
        changes: dict[str, object] = {"status": "prepared"}
        if (
            record.get("kind") == "refine"
            and self.workspace.pending_refinement() is not None
        ):
            baseline = self.workspace.load().get(
                "pending_specification_fingerprint"
            )
            current = self.workspace.specification_fingerprint(
                include_pending=False
            )
            if baseline and baseline != current:
                changes = {
                    "status": "awaiting_review",
                    "lifecycle_inferred": True,
                    "result_specification_fingerprint": (
                        self.workspace.specification_fingerprint()
                    ),
                }
        return self.workspace.update_request(
            str(record["request_id"]),
            **changes,
        )

    def _task_state(
        self,
        record: dict[str, object],
    ) -> tuple[str, StatusKind]:
        status = str(record.get("status") or "prepared")
        states: dict[str, tuple[str, StatusKind]] = {
            "prepared": ("ready for assistant", "success"),
            "assistant_running": ("assistant running", "info"),
            "awaiting_review": ("awaiting human review", "warning"),
            "assistant_failed": ("assistant failed", "error"),
            "assistant_interrupted": ("assistant interrupted", "warning"),
            "reconciled": ("reconciled", "success"),
            "discarded": ("discarded", "warning"),
            "closed": ("closed", "success"),
        }
        return states.get(status, (status.replace("_", " "), "warning"))

    def _task_verification(
        self,
        record: dict[str, object],
    ) -> tuple[str, StatusKind]:
        verification = str(record.get("assistant_verification") or "")
        checks = record.get("assistant_verification_checks")
        counts = {"passed": 0, "failed": 0, "not_run": 0}
        if isinstance(checks, list):
            for check in checks:
                if not isinstance(check, dict):
                    continue
                status = str(check.get("status") or "")
                if status in counts:
                    counts[status] += 1
        total = sum(counts.values())
        if verification == "passed":
            suffix = f" — {total} check{'s' if total != 1 else ''}"
        else:
            parts = [
                f"{count} {label}"
                for label, count in (
                    ("passed", counts["passed"]),
                    ("failed", counts["failed"]),
                    ("not run", counts["not_run"]),
                )
                if count
            ]
            suffix = f" — {', '.join(parts)}" if parts else " — no checks reported"
        if verification == "passed":
            return f"passed{suffix}", "success"
        if verification == "failed":
            return f"failed{suffix}", "error"
        if verification == "incomplete":
            return f"incomplete{suffix}", "warning"
        status = str(record.get("status") or "prepared")
        if status in {"prepared", "assistant_running"}:
            return "not available yet", "info"
        if record.get("manual_integration") and not record.get("assistant"):
            return "not reported — manual integration", "warning"
        return "not reported by this older implementation request", "warning"

    def _task_verification_summary(self, record: dict[str, object]) -> str | None:
        value = record.get("assistant_verification_summary")
        return str(value) if value else None

    def _emit_task_verification_checks(
        self,
        record: dict[str, object],
        *,
        problems_only: bool = False,
    ) -> None:
        checks = record.get("assistant_verification_checks")
        if not isinstance(checks, list) or not checks:
            return
        rows: list[tuple[str, object, StatusKind | None]] = []
        kinds: dict[str, StatusKind] = {
            "passed": "success",
            "failed": "error",
            "not_run": "warning",
        }
        for index, value in enumerate(checks, start=1):
            if not isinstance(value, dict):
                continue
            status = str(value.get("status") or "not_run")
            if problems_only and status == "passed":
                continue
            command = str(value.get("command") or "unspecified command")
            detail = str(value.get("detail") or "")
            outcome = command if not detail else f"{command} — {detail}"
            rows.append(
                (
                    f"{index}. {status}",
                    outcome,
                    kinds.get(status, "warning"),
                )
            )
        if rows:
            title = (
                "Failed or incomplete assistant checks"
                if problems_only
                else "Assistant verification checks"
            )
            self._emit_table(title, rows)

    def _task_next(self, record: dict[str, object]) -> str:
        status = str(record.get("status") or "prepared")
        kind = str(record.get("kind") or "")
        if status == "awaiting_review":
            verification = str(record.get("assistant_verification") or "")
            if verification != "passed" and record.get("assistant"):
                backend = (
                    "claude"
                    if str(record.get("assistant")).lower().startswith("claude")
                    else "codex"
                )
                review = (
                    "workflow list · workflow select · workflow show source · "
                    "workflow show protocol · workflow validate"
                    if kind == "create"
                    else "current · workflow validate · workflow show"
                )
                return (
                    f"{review} · workflow implement {backend} --rerun"
                )
            if kind == "refine":
                if record.get("specification_context_changed") is False:
                    return (
                        "workflow edit spec · workflow implement codex --rerun · "
                        "workflow implement claude --rerun"
                    )
                return (
                    "current · workflow validate · workflow show · run · "
                    "workflow accept"
                )
            return (
                "workflow list · workflow select · workflow show source · "
                "workflow show protocol · workflow validate · workflow accept"
            )
        if status == "assistant_running":
            return "wait for the assistant session to return"
        if status in {"assistant_failed", "assistant_interrupted"}:
            return (
                "workflow show · workflow implement codex · "
                "workflow implement claude"
            )
        return "workflow implement codex · workflow implement claude"

    def _task_execution(self, record: dict[str, object]) -> str:
        status = str(record.get("status") or "prepared")
        if status == "assistant_running":
            return "running synchronously now; nothing is queued"
        if status == "prepared":
            return "not started; nothing is scheduled"
        if record.get("manual_integration") and not record.get("assistant"):
            return "assistant not run; nothing is scheduled"
        return "assistant session ended; nothing is scheduled"

    def _task_assistant(self, record: dict[str, object]) -> str:
        assistant = record.get("assistant")
        status = str(record.get("status") or "prepared")
        if assistant:
            finished = record.get("assistant_finished_at")
            if status == "assistant_running":
                return (
                    f"{assistant} — started "
                    f"{record.get('assistant_started_at') or 'now'}"
                )
            if finished:
                return f"{assistant} — returned {finished}"
            return str(assistant)
        if record.get("lifecycle_inferred"):
            return "not recorded; review inferred from specification change"
        if record.get("manual_integration"):
            return "not used; canonical specification was edited manually"
        return "not started"

    def _task_context(
        self,
        record: dict[str, object],
    ) -> tuple[str, StatusKind]:
        status = str(record.get("status") or "prepared")
        current = self.workspace.specification_fingerprint()
        if status == "awaiting_review":
            result = record.get("result_specification_fingerprint")
            if result and result != current:
                if record.get("manual_integration"):
                    return "changed again after manual integration", "warning"
                return "changed again after the assistant returned", "warning"
            if record.get("manual_integration"):
                return "manual integration is preserved for review", "success"
            return "assistant result is preserved for review", "success"
        if record.get("specification_fingerprint") == current:
            return "matches the current specification", "success"
        return "changed since this implementation was prepared", "warning"

    def manage_task(self, args: list[str]) -> None:
        if len(args) > 2 or (
            args and args[0].lower() not in {"show", "path", "history", "close"}
        ):
            raise SystemExit(
                "Use workflow status, workflow history, or "
                "workflow accept [--yes]."
            )
        action = args[0].lower() if args else "summary"
        rest = args[1:]
        if action != "close" and rest:
            raise SystemExit(
                "Use workflow status, workflow history, or "
                "workflow accept [--yes]."
            )
        if action == "history":
            records = self.workspace.list_requests()
            if not records:
                self._emit_table(
                    "Implementation history",
                    [
                        (
                            "Status",
                            "none; use workflow create or workflow refine",
                            "warning",
                        )
                    ],
                )
                return
            self._emit("Implementation history")
            self._emit("──────────────────────")
            self._emit(
                "  Request                  Kind        State              "
                "Refreshes                 Created"
            )
            for record in records:
                state, _state_kind = self._task_state(record)
                self._emit(
                    f"  {str(record['request_id']):24} "
                    f"{str(record['kind']):11} "
                    f"{state:18} "
                    f"{str(record.get('refreshes_request') or '—'):24}  "
                    f"{record.get('created_at') or '—'}"
                )
            self._emit()
            return

        record = self._ensure_current_task_fresh()
        if record is None:
            if action in {"show", "path", "close"}:
                raise SystemExit(
                    "No current implementation. Use workflow create or "
                    "workflow refine to prepare one."
                )
            self._emit_table(
                "Workflow implementation",
                [
                    (
                        "Status",
                        "none; use workflow create or workflow refine",
                        "warning",
                    )
                ],
            )
            return
        record = self._normalize_task_lifecycle(record)
        if action == "close":
            if rest not in ([], ["--yes"]):
                raise SystemExit("Use workflow accept [--yes].")
            if self.workspace.pending_refinement() is not None:
                raise SystemExit(
                    "A refinement is still pending. Review it, then use "
                    "'workflow accept' to accept it or 'workflow discard' "
                    "to reject it."
                )
            if rest != ["--yes"] and not self._confirm_action(
                "Accept and close the reviewed workflow implementation? [y/n]: ",
                cancel_message=(
                    "Workflow acceptance cancelled; nothing was changed."
                ),
            ):
                return
            closed = self.workspace.clear_current_task()
            self._emit_table(
                "Workflow implementation accepted",
                [
                    ("Status", "closed", "success"),
                    ("Request", closed["request_id"], None),
                    ("History", "retained; use workflow history", None),
                    ("Next", "current", None),
                ],
            )
            return
        if action == "path":
            self._emit(self.workspace.current_task_path)
            return
        if action == "show":
            self._emit(
                self.workspace.current_task_path.read_text(encoding="utf-8").rstrip()
            )
            return
        state, state_kind = self._task_state(record)
        context, context_kind = self._task_context(record)
        self._emit_table(
            "Workflow implementation",
            [
                ("Status", state, state_kind),
                ("Kind", record["kind"], None),
                ("Request", record["request_id"], None),
                ("Workflow", record.get("workflow_spec") or "new workflow", None),
                ("Assistant", self._task_assistant(record), None),
                ("Execution", self._task_execution(record), None),
                ("Verification", *self._task_verification(record)),
                *(
                    [
                        (
                            "Verification note",
                            self._task_verification_summary(record),
                            None,
                        )
                    ]
                    if self._task_verification_summary(record)
                    else []
                ),
                ("Refreshes", record.get("refreshes_request") or "—", None),
                ("Context", context, context_kind),
                ("Record", ".zippergen/current-task.md", None),
                ("Next", self._task_next(record), None),
            ],
        )
        self._emit_task_verification_checks(record)

    def _consume_assistant_result(self) -> _AssistantResult:
        """Read and remove the assistant's bounded, project-local handoff."""

        path = self.workspace.assistant_result_path
        if not path.exists() and not path.is_symlink():
            return _AssistantResult(
                "incomplete",
                "The assistant returned without writing its required verification record.",
                error="assistant result file was not written",
            )
        try:
            if path.is_symlink():
                raise ValueError("the assistant result path must not be a symlink")
            if path.stat().st_size > 256 * 1024:
                raise ValueError("the assistant result is larger than 256 KiB")
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("the assistant result must be a JSON object")
            if value.get("schema_version") != 1:
                raise ValueError("schema_version must be 1")
            declared = value.get("verification")
            if declared not in {"passed", "failed", "incomplete"}:
                raise ValueError(
                    "verification must be passed, failed, or incomplete"
                )
            raw_summary = value.get("summary")
            if not isinstance(raw_summary, str) or not raw_summary.strip():
                raise ValueError("summary must be a non-empty string")
            summary = " ".join(raw_summary.split())[:1000]
            raw_checks = value.get("checks")
            if not isinstance(raw_checks, list):
                raise ValueError("checks must be a JSON array")
            if len(raw_checks) > 100:
                raise ValueError("checks must contain at most 100 entries")
            checks: list[dict[str, str]] = []
            for index, raw_check in enumerate(raw_checks, start=1):
                if not isinstance(raw_check, dict):
                    raise ValueError(f"check {index} must be a JSON object")
                command = raw_check.get("command")
                check_status = raw_check.get("status")
                detail = raw_check.get("detail", "")
                if not isinstance(command, str) or not command.strip():
                    raise ValueError(f"check {index} command must not be empty")
                if check_status not in {"passed", "failed", "not_run"}:
                    raise ValueError(
                        f"check {index} status must be passed, failed, or not_run"
                    )
                if not isinstance(detail, str):
                    raise ValueError(f"check {index} detail must be a string")
                checks.append(
                    {
                        "command": " ".join(command.split())[:1000],
                        "status": str(check_status),
                        "detail": " ".join(detail.split())[:1000],
                    }
                )

            verification = cast(AssistantVerification, declared)
            check_statuses = {check["status"] for check in checks}
            if "failed" in check_statuses:
                if verification != "failed":
                    summary = (
                        f"{summary} Studio corrected the overall result because "
                        "a reported check failed."
                    )[:1000]
                verification = "failed"
            elif "not_run" in check_statuses:
                if verification == "passed":
                    summary = (
                        f"{summary} Studio corrected the overall result because "
                        "a requested check was not run."
                    )[:1000]
                if verification != "failed":
                    verification = "incomplete"
            elif not checks and verification == "passed":
                verification = "incomplete"
                summary = (
                    f"{summary} Studio requires at least one reported check "
                    "before verification can be marked passed."
                )[:1000]
            return _AssistantResult(
                verification,
                summary,
                tuple(checks),
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            detail = str(exc)
            return _AssistantResult(
                "incomplete",
                f"Studio could not accept the assistant verification record: {detail}.",
                error=detail[:1000],
            )
        finally:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def _parse_codex_output(
        self,
        stdout: object,
        stderr: object,
    ) -> _CodexOutput:
        """Condense Codex JSONL while dropping known internal cache noise."""

        report: str | None = None
        diagnostics: list[str] = []
        suppressed = 0
        stderr_text = stderr if isinstance(stderr, str) else ""
        for raw_line in stderr_text.splitlines():
            line = " ".join(raw_line.split())
            if not line:
                continue
            if (
                "codex_models_manager::manager: failed to renew cache TTL"
                in line
                and "supports_reasoning_summaries" in line
            ) or "could not create PATH aliases: Operation not permitted" in line:
                suppressed += 1
                continue
            diagnostics.append(line[:1000])

        stdout_text = stdout if isinstance(stdout, str) else ""
        for raw_line in stdout_text.splitlines():
            if not raw_line.strip():
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                diagnostics.append(" ".join(raw_line.split())[:1000])
                continue
            if not isinstance(event, dict):
                continue
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message":
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    report = text.strip()[:20_000]
            event_type = str(event.get("type") or "")
            if event_type in {"error", "turn.failed"}:
                error = event.get("error")
                message = (
                    error.get("message")
                    if isinstance(error, dict)
                    else event.get("message") or error
                )
                if message:
                    diagnostics.append(" ".join(str(message).split())[:1000])
        return _CodexOutput(
            report=report,
            diagnostics=tuple(diagnostics[:20]),
            suppressed_diagnostics=suppressed,
        )

    def _emit_codex_output(self, output: _CodexOutput) -> None:
        if output.report:
            self._emit("Assistant report")
            self._emit("────────────────")
            self._emit(output.report)
            self._emit()
        for diagnostic in output.diagnostics[:3]:
            self._warning(f"Codex diagnostic: {diagnostic}")
        if len(output.diagnostics) > 3:
            self._warning(
                f"Codex emitted {len(output.diagnostics) - 3} additional "
                "diagnostic lines; the assistant verification record is "
                "authoritative for reported checks."
            )

    def _ensure_assistant_test_environment(self) -> None:
        """Fail before spending an assistant run when nested tests cannot run."""

        framework = self.workspace.project_manifest().get("framework_directory")
        if not framework:
            return
        uv = shutil.which("uv")
        if uv is None:
            project = shlex.quote(str(framework))
            raise SystemExit(
                "The nested ZipperGen development environment requires uv, but "
                "the uv command was not found. Install uv, then run "
                f"'uv sync --project {project}' once before starting the "
                "assistant. No assistant was started."
            )
        command = [
            uv,
            "run",
            "--offline",
            "--project",
            str(framework),
            "python",
            "-c",
            "import pytest",
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=self.workspace.root,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise SystemExit(
                f"Could not inspect the nested development environment: {exc}"
            ) from exc
        if completed.returncode != 0:
            project = shlex.quote(str(framework))
            raise SystemExit(
                "The nested ZipperGen development environment is not synchronized, "
                "so the assistant could not verify application tests offline. "
                f"In an ordinary terminal run 'uv sync --project {project}' "
                "once, then return and run the assistant again. No assistant "
                "was started."
            )

    def _start_assistant_heartbeat(
        self,
        tool: str,
    ) -> tuple[threading.Event, threading.Thread]:
        """Keep a condensed one-shot assistant visibly alive."""

        stopped = threading.Event()
        started = time.monotonic()
        interval = max(0.01, _ASSISTANT_HEARTBEAT_SECONDS)

        def report_progress() -> None:
            while not stopped.wait(interval):
                elapsed = max(0, int(time.monotonic() - started))
                hours, remainder = divmod(elapsed, 3600)
                minutes, seconds = divmod(remainder, 60)
                duration = (
                    f"{hours}:{minutes:02d}:{seconds:02d}"
                    if hours
                    else f"{minutes:02d}:{seconds:02d}"
                )
                self._info(
                    f"{tool} is still working — {duration} elapsed. "
                    "Press Control-C to interrupt safely."
                )

        thread = threading.Thread(
            target=report_progress,
            name="zippergen-assistant-progress",
            daemon=True,
        )
        thread.start()
        return stopped, thread

    def run_assistant(self, args: list[str]) -> None:
        rerun = "--rerun" in args
        interactive = "--interactive" in args
        values = [
            value
            for value in args
            if value not in {"--rerun", "--interactive"}
        ]
        if len(values) > 1 or any(
            value.lower() not in {"codex", "claude"} for value in values
        ) or args.count("--rerun") > 1 or args.count("--interactive") > 1:
            raise SystemExit(
                "Use workflow implement, workflow implement codex, "
                "workflow implement claude, or workflow implement "
                "[codex|claude] --rerun. Use workflow implement codex "
                "--interactive only for an interactive session."
            )
        assistant = (
            values[0].lower()
            if values
            else str(self._global_settings()["assistant"])
        )
        if interactive and assistant != "codex":
            raise SystemExit(
                "--interactive is supported only with workflow implement codex."
            )
        record = self._ensure_current_task_fresh(for_assistant=True)
        if record is None:
            raise SystemExit(
                "No current implementation request. Use workflow create or "
                "workflow refine before starting the assistant."
            )
        status = str(record.get("status") or "prepared")
        if status == "awaiting_review":
            manual_first_pass = bool(record.get("manual_integration")) and not bool(
                record.get("assistant")
            )
            if not rerun and not manual_first_pass:
                raise SystemExit(
                    "The assistant has already returned and this implementation is awaiting "
                    "human review. Use current, workflow validate, workflow show, "
                    "and then workflow accept; use 'workflow implement "
                    f"{assistant} --rerun' only to run it deliberately again."
                )
            record = self._ensure_current_task_fresh(
                for_assistant=True,
                force=True,
            )
            assert record is not None
            status = str(record.get("status") or "prepared")
        if status == "assistant_running":
            raise SystemExit(
                "This implementation is already marked as running. Wait for the assistant "
                "session to return; after an interrupted Studio process, prepare "
                "or refine the workflow again before retrying."
            )
        tool = "Claude Code" if assistant == "claude" else "Codex CLI"
        executable = shutil.which(assistant)
        if executable is None:
            if assistant == "claude":
                setup = (
                    "Install Claude Code and complete its first-run authentication"
                )
            else:
                setup = "Install Codex CLI and run 'codex login'"
            raise SystemExit(
                f"{tool} was not found. {setup} once; "
                "the current implementation request remains available at "
                f"{self.workspace.current_task_path}."
            )
        self._ensure_assistant_test_environment()
        try:
            self.workspace.assistant_result_path.unlink(missing_ok=True)
        except OSError as exc:
            raise SystemExit(
                "Could not clear the previous assistant verification handoff at "
                f"{self.workspace.assistant_result_path}: {exc}"
            ) from exc
        started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        record = self.workspace.update_request(
            str(record["request_id"]),
            status="assistant_running",
            assistant=("Claude Code" if assistant == "claude" else "Codex"),
            assistant_started_at=started_at,
            assistant_finished_at=None,
            assistant_exit_code=None,
            assistant_mode=("interactive" if interactive else "one_shot"),
            assistant_verification=None,
            assistant_verification_summary=None,
            assistant_verification_checks=[],
            assistant_result_error=None,
            assistant_report=None,
            assistant_cli_diagnostics=[],
            assistant_suppressed_diagnostics=0,
            studio_process_id=os.getpid(),
            lifecycle_inferred=False,
        )
        relative_task = self.workspace.current_task_path.relative_to(
            self.workspace.root
        ).as_posix()
        self._emit_table(
            "Assistant",
            [
                (
                    "Tool",
                    tool,
                    None,
                ),
                (
                    "Mode",
                    (
                        "interactive implementation session"
                        if interactive
                        else "one-shot implementation; returns to Studio automatically"
                    ),
                    None,
                ),
                ("Request", relative_task, "success"),
                ("Project", self.workspace.root, None),
                (
                    "MCP",
                    "not required; the assistant keeps its own configured tools",
                    None,
                ),
                (
                    "Output",
                    (
                        "condensed; final report and failed checks appear on return"
                        if assistant == "codex" and not interactive
                        else "live assistant terminal output"
                    ),
                    None,
                ),
            ],
        )
        instruction = (
            f"Read and execute {relative_task}. Follow the repository instructions, "
            "keep all generated code visible, run the requested verification, and "
            "do not deploy. Before exiting, write the required structured result "
            "to .zippergen/assistant-result.json."
        )
        if assistant == "codex" and interactive:
            command = [
                executable,
                "--cd",
                str(self.workspace.root),
                instruction,
            ]
        elif assistant == "codex":
            command = [
                executable,
                "exec",
                "--json",
                "--skip-git-repo-check",
                "--cd",
                str(self.workspace.root),
                instruction,
            ]
        else:
            # Claude's explicit print/agent mode executes the supplied task and
            # returns. acceptEdits permits project-local source changes while
            # retaining Claude Code's permission boundary for other commands.
            command = [
                executable,
                "--print",
                "--permission-mode",
                "acceptEdits",
                instruction,
            ]
        capture_codex = assistant == "codex" and not interactive
        heartbeat_stop: threading.Event | None = None
        heartbeat_thread: threading.Thread | None = None
        if capture_codex:
            self._info(
                f"{tool} is working. Detailed output is condensed until it "
                "returns; press Control-C to interrupt safely."
            )
            heartbeat_stop, heartbeat_thread = self._start_assistant_heartbeat(
                tool
            )
        elif not interactive:
            self._info(
                f"{tool} is working with live output. Press Control-C to "
                "interrupt safely."
            )
        try:
            try:
                if capture_codex:
                    completed = subprocess.run(
                        command,
                        cwd=self.workspace.root,
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                else:
                    completed = subprocess.run(
                        command,
                        cwd=self.workspace.root,
                        check=False,
                    )
            finally:
                if heartbeat_stop is not None:
                    heartbeat_stop.set()
                if heartbeat_thread is not None:
                    heartbeat_thread.join(timeout=0.5)
        except KeyboardInterrupt:
            self.workspace.update_request(
                str(record["request_id"]),
                status="assistant_interrupted",
                assistant_finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                result_specification_fingerprint=(
                    self.workspace.specification_fingerprint()
                ),
            )
            raise
        except OSError as exc:
            self.workspace.update_request(
                str(record["request_id"]),
                status="assistant_failed",
                assistant_finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                assistant_error=str(exc)[:240],
                result_specification_fingerprint=(
                    self.workspace.specification_fingerprint()
                ),
            )
            raise SystemExit(
                f"Could not start {tool}: {exc}"
            ) from exc
        codex_output = (
            self._parse_codex_output(completed.stdout, completed.stderr)
            if assistant == "codex" and not interactive
            else _CodexOutput()
        )
        self._emit_codex_output(codex_output)
        if completed.returncode != 0:
            assistant_result = self._consume_assistant_result()
            self.workspace.update_request(
                str(record["request_id"]),
                status="assistant_failed",
                assistant_finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                assistant_exit_code=completed.returncode,
                assistant_verification=assistant_result.verification,
                assistant_verification_summary=assistant_result.summary,
                assistant_verification_checks=[
                    dict(check) for check in assistant_result.checks
                ],
                assistant_result_error=assistant_result.error,
                assistant_report=codex_output.report,
                assistant_cli_diagnostics=list(codex_output.diagnostics),
                assistant_suppressed_diagnostics=(
                    codex_output.suppressed_diagnostics
                ),
                result_specification_fingerprint=(
                    self.workspace.specification_fingerprint()
                ),
            )
            raise SystemExit(
                f"{assistant.capitalize()} exited with status "
                f"{completed.returncode}; the implementation request remains at "
                f"{self.workspace.current_task_path}."
            )
        assistant_result = self._consume_assistant_result()
        result_fingerprint = self.workspace.specification_fingerprint()
        changed = record.get("specification_fingerprint") != result_fingerprint
        record = self.workspace.update_request(
            str(record["request_id"]),
            status="awaiting_review",
            assistant_finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            assistant_exit_code=0,
            assistant_verification=assistant_result.verification,
            assistant_verification_summary=assistant_result.summary,
            assistant_verification_checks=[
                dict(check) for check in assistant_result.checks
            ],
            assistant_result_error=assistant_result.error,
            assistant_report=codex_output.report,
            assistant_cli_diagnostics=list(codex_output.diagnostics),
            assistant_suppressed_diagnostics=codex_output.suppressed_diagnostics,
            result_specification_fingerprint=result_fingerprint,
            specification_context_changed=changed,
        )
        self._info(
            f"{'Claude Code' if assistant == 'claude' else 'Codex'} session "
            "returned to Studio."
        )
        if assistant_result.verification == "passed":
            self._success("Assistant reported that verification passed.")
        elif assistant_result.verification == "failed":
            self._error(
                "Assistant reported verification failures; do not accept the "
                "change until they are resolved."
            )
        else:
            self._warning(
                "Assistant verification is incomplete; a normal assistant exit "
                "does not mean the checks passed."
            )
        kind = str(record.get("kind") or "")
        specification_result = (
            "changed since implementation preparation"
            if changed
            else (
                "unchanged; reconciliation will refuse until it is integrated"
                if kind == "refine"
                else "unchanged; review the generated workflow files"
            )
        )
        self._emit_table(
            "Review required",
            [
                ("Status", "awaiting human review", "warning"),
                (
                    "Specification",
                    specification_result,
                    "success" if changed else "warning",
                ),
                (
                    "Refinement",
                    "still open; Studio never accepts it automatically"
                    if kind == "refine"
                    else "not applicable",
                    "warning" if kind == "refine" else None,
                ),
                ("Verification", *self._task_verification(record)),
                ("Verification note", assistant_result.summary, None),
                ("Next", self._task_next(record), None),
            ],
        )
        if assistant_result.verification != "passed":
            self._emit_task_verification_checks(record, problems_only=True)

    def show_current(self) -> None:
        from zippergen.serve import _validate_workflow

        state = self.workspace.load()
        global_settings = self._global_settings()
        manifest = self.workspace.project_manifest()
        request = self._ensure_current_task_fresh(announce=False)
        specification = self.workspace.specification()
        pending = self.workspace.pending_refinement()
        task_state, task_state_kind = (
            self._task_state(request)
            if request
            else ("none; use workflow create or workflow refine", "warning")
        )
        refinement_status = (
            (
                "pending — awaiting human review; use workflow show pending"
                if request
                and request.get("kind") == "refine"
                and request.get("status") == "awaiting_review"
                else "pending — use workflow show pending or workflow refine"
            )
            if pending is not None
            else "none"
        )
        self._emit("Current")
        self._emit("═══════")
        self._emit()
        self._emit_table(
            "Project",
            [
                ("Name", manifest["name"], None),
                ("Root", self.workspace.root, None),
                (
                    "Manifest",
                    (
                        f"present — {self.workspace.manifest_path}"
                        if manifest["exists"]
                        else f"not created — {self.workspace.manifest_path}"
                    ),
                    "success" if manifest["exists"] else "warning",
                ),
                (
                    "Specification",
                    (
                        f"ready — {self.workspace.specification_path.name}"
                        if specification is not None
                        else "not written; use workflow create or "
                        "workflow edit spec"
                    ),
                    "success" if specification is not None else "warning",
                ),
                (
                    "Refinement",
                    refinement_status,
                    "warning" if pending is not None else None,
                ),
                (
                    "Implementation",
                    (
                        f"{request['request_id']} ({request['kind']}) — "
                        f"{task_state}; .zippergen/current-task.md"
                        if request
                        else task_state
                    ),
                    task_state_kind,
                ),
                *(
                    [("Task next", self._task_next(request), None)]
                    if request
                    else []
                ),
                *(
                    [("Task verification", *self._task_verification(request))]
                    if request
                    else []
                ),
                (
                    "Editor",
                    (
                        shlex.join(
                            self._parse_editor_command(
                                global_settings["editor_command"]
                            )
                        )
                        if global_settings.get("editor_command")
                        else "automatic; use editor show or editor set COMMAND"
                    ),
                    None,
                ),
            ],
        )
        language_store = self._language_store()
        language_state = global_settings
        configured_interpreter = str(
            language_state.get("interpreter") or "auto"
        )
        selected_interpreter = self._language_backend(
            configured_interpreter,
            required=False,
        )
        if configured_interpreter == "off":
            interpreter_status = "off; deterministic and learned requests only"
            interpreter_kind: StatusKind = "warning"
        elif selected_interpreter is None:
            interpreter_status = (
                f"{configured_interpreter}; no supported CLI was found"
            )
            interpreter_kind = "warning"
        else:
            label = (
                "Codex CLI"
                if selected_interpreter[0] == "codex"
                else "Claude Code"
            )
            interpreter_status = (
                f"{configured_interpreter} → {label}"
                if configured_interpreter == "auto"
                else label
            )
            interpreter_kind = "success"
        self._emit_table(
            "Natural language",
            [
                ("Interpreter", interpreter_status, interpreter_kind),
                (
                    "Learning",
                    "on" if language_state.get("learning", True) else "off",
                    "success"
                    if language_state.get("learning", True)
                    else "warning",
                ),
                ("Learned", len(language_store.learned()), None),
                ("History", len(language_store.history()), None),
                (
                    "Scope",
                    "global policy; learned data is project-local",
                    None,
                ),
            ],
        )
        if state.get("current_workflow"):
            _current, workflow, module = self._current_context()
            model = workflow_semantics(workflow, module)
            raw_lifelines = model.get("lifelines")
            lifelines = (
                [str(name) for name in raw_lifelines]
                if isinstance(raw_lifelines, list)
                else []
            )
            raw_action_sites = model.get("action_sites")
            action_sites = (
                raw_action_sites if isinstance(raw_action_sites, list) else []
            )
            human_actions = [
                str(site.get("action"))
                for site in action_sites
                if isinstance(site, dict) and site.get("kind") == "human"
            ]
            effect_actions = [
                str(site.get("action"))
                for site in action_sites
                if isinstance(site, dict) and site.get("kind") == "effect"
            ]
            assistant_actions = [
                f"{site.get('lifeline')}.{site.get('action')}"
                for site in action_sites
                if isinstance(site, dict) and site.get("kind") == "assistant"
            ]
            active_models = self._llm_action_lifelines(workflow, module)
            llm_participants = list(active_models)
            validation = _validate_workflow(workflow, module)
            self._emit_table(
                "Workflow",
                [
                    ("Selected", state["current_workflow"], "success"),
                    ("Name", workflow.name, None),
                    (
                        "Participants",
                        f"{len(lifelines)} — "
                        + (", ".join(lifelines) if lifelines else "none"),
                        None,
                    ),
                    (
                        "LLM-active participants",
                        f"{len(llm_participants)} — "
                        + (
                            ", ".join(llm_participants)
                            if llm_participants
                            else "none"
                        ),
                        None,
                    ),
                    (
                        "Human actions",
                        f"{len(human_actions)} — "
                        + (", ".join(human_actions) if human_actions else "none"),
                        None,
                    ),
                    (
                        "Effects",
                        f"{len(effect_actions)} — "
                        + (", ".join(effect_actions) if effect_actions else "none"),
                        None,
                    ),
                    (
                        "Assistant actions",
                        f"{len(assistant_actions)} — "
                        + (
                            ", ".join(assistant_actions)
                            if assistant_actions
                            else "none"
                        ),
                        None,
                    ),
                    ("Connectors", "none", None),
                    (
                        "Validation",
                        "valid" if validation["valid"] else "invalid",
                        "success" if validation["valid"] else "error",
                    ),
                ],
            )
            assignments = self.workspace.model_assignment_profile(
                str(state["current_workflow"]),
                default=default_llm_spec(module),
            )
            configurations = self.workspace.model_configurations()
            default_configuration = str(assignments["default"])
            overrides = assignments.get("lifelines") or {}
            assert isinstance(overrides, dict)
            model_rows: list[tuple[str, object, StatusKind | None]] = [
                (
                    "Default",
                    f"{default_configuration} → "
                    f"{configurations[default_configuration]['spec']}",
                    None,
                )
            ]
            if active_models:
                for lifeline, actions in active_models.items():
                    explicit = overrides.get(lifeline)
                    effective = str(explicit or default_configuration)
                    source = "override" if explicit else "default"
                    spec = configurations.get(effective, {}).get(
                        "spec", "missing"
                    )
                    model_rows.append(
                        (
                            lifeline,
                            f"{effective} → {spec} "
                            f"({source}; actions: {', '.join(actions)})",
                            None,
                        )
                    )
            else:
                model_rows.append(("Assignments", "none", None))
            selected_configurations = {default_configuration} | {
                str(value) for value in overrides.values()
            }
            selected_specs = {
                configurations.get(name, {}).get("spec", "mock")
                for name in selected_configurations
            }
            providers = sorted({_canonical_provider(spec) for spec in selected_specs})
            for provider in providers:
                kind, provider_status = self._provider_configuration_status(
                    provider
                )
                model_rows.append(
                    (f"Provider {provider}", provider_status, kind)
                )
            self._emit_table("Models", model_rows)
        else:
            self._emit_table(
                "Workflow",
                [
                    ("Selected", "none", "warning"),
                    ("Name", "—", None),
                    ("Participants", "0 — none", None),
                    ("LLM-active participants", "0 — none", None),
                    ("Human actions", "0 — none", None),
                    ("Effects", "0 — none", None),
                    ("Assistant actions", "0 — none", None),
                    ("Connectors", "none", None),
                    ("Validation", "not available", "warning"),
                ],
            )
            self._emit_table(
                "Models",
                [
                    ("Default", "—", None),
                    ("Assignments", "none", None),
                    ("Providers", "none", None),
                ],
            )
        run = self.workspace.current_run()
        runtime_rows: list[tuple[str, object, StatusKind | None]] = []
        if run is None:
            runtime_rows.append(("Run", "none", None))
        else:
            run_status = str(run["status"])
            if run_status == "done":
                run_kind: StatusKind = "success"
            elif run_status == "failed":
                run_kind = "error"
            elif run_status in {"waiting", "interrupted"}:
                run_kind = "warning"
            else:
                run_kind = "info"
            runtime_rows.extend(
                [
                    ("Run", f"{run['run_id']} ({run['status']})", run_kind),
                    ("Store", run["store"], None),
                    (
                        "Assistant",
                        run.get("assistant") or "none selected",
                        None,
                    ),
                ]
            )
        runtime_rows.append(
            ("Deployment", state.get("last_deployment") or "none", None)
        )
        self._emit_table("Runtime", runtime_rows)

    def _select(self, heading: str, choices: list[str], *, allow_many: bool = False):
        if not choices:
            raise SystemExit("No choices are available.")
        self._emit(heading)
        for index, choice in enumerate(choices, 1):
            self._emit(f"  {index}. {choice}")
        suffix = " (comma-separated)" if allow_many else ""
        raw = self.input(f"Select{suffix}: ").strip()
        if allow_many:
            values = []
            for item in raw.split(","):
                try:
                    index = int(item.strip())
                except ValueError as exc:
                    raise SystemExit(f"Invalid selection: {item!r}") from exc
                if index < 1 or index > len(choices):
                    raise SystemExit(f"Selection must be between 1 and {len(choices)}.")
                values.append(choices[index - 1])
            return values
        try:
            index = int(raw)
        except ValueError as exc:
            raise SystemExit(f"Invalid selection: {raw!r}") from exc
        if index < 1 or index > len(choices):
            raise SystemExit(f"Selection must be between 1 and {len(choices)}.")
        return choices[index - 1]

    def list_workflows(self) -> None:
        candidates = self.workspace.discover_workflows()
        selected = self.workspace.current_workflow
        if not candidates:
            self._emit_table(
                "Available workflows",
                [
                    (
                        "Status",
                        "none discovered; no validation was run",
                        "warning",
                    ),
                    (
                        "Next",
                        "inspect generated Python for a top-level @workflow",
                        None,
                    ),
                ],
            )
            return
        rows: list[tuple[str, object, StatusKind | None]] = []
        for index, spec in enumerate(candidates, start=1):
            name = spec.rpartition(":")[2] or spec
            state = " — selected" if spec == selected else ""
            rows.append(
                (
                    str(index),
                    f"{name} — {spec}{state}",
                    "success" if spec == selected else None,
                )
            )
        rows.extend(
            [
                ("Discovery", "source scan only; validation was not run", "info"),
                ("Next", "workflow select NUMBER|NAME", None),
            ]
        )
        self._emit_table("Available workflows", rows)

    def _resolve_workflow_choice(
        self,
        value: str,
        candidates: list[str],
    ) -> str:
        if value.isdecimal():
            index = int(value)
            if 1 <= index <= len(candidates):
                return candidates[index - 1]
            raise SystemExit(
                f"Workflow number must be between 1 and {len(candidates)}."
            )
        canonical = self.workspace.canonical_spec(value, cwd=self.workspace.root)
        if canonical in candidates:
            return canonical
        by_name = [
            spec
            for spec in candidates
            if spec.rpartition(":")[2].casefold() == value.casefold()
        ]
        if len(by_name) == 1:
            return by_name[0]
        if len(by_name) > 1:
            raise SystemExit(
                f"Workflow name {value!r} is ambiguous. Use 'workflow list' "
                "and select its number or complete PATH.py:NAME."
            )
        raise SystemExit(
            f"Workflow was not discovered: {value}. Use 'workflow list' first."
        )

    def _select_workflow_spec(self, selected: str) -> tuple[str, str]:
        from zippergen.serve import load_workflow_spec

        canonical = self.workspace.canonical_spec(selected)
        workflow, _module = load_workflow_spec(self.workspace.absolute_spec(canonical))
        self.workspace.select_workflow(canonical, cwd=self.workspace.root)
        return canonical, workflow.name

    def select_workflow(self, args: list[str]) -> None:
        if len(args) > 1:
            raise SystemExit(
                "Use workflow select [NUMBER|NAME|PATH.py:WORKFLOW]."
            )
        candidates = self.workspace.discover_workflows()
        if not candidates:
            raise SystemExit(
                "No workflow entry points were discovered. Inspect the "
                "generated Python for a top-level @workflow definition."
            )
        if args:
            selected = self._resolve_workflow_choice(args[0], candidates)
        else:
            selected = self._select("Select workflow", candidates)
            assert isinstance(selected, str)
        canonical, name = self._select_workflow_spec(selected)
        self._emit_table(
            "Workflow selected",
            [
                ("Workflow", name, None),
                ("Entry point", canonical, None),
                ("Load", "succeeded; entry point is available", "success"),
                ("Purpose", "inspection, configuration, and execution", None),
                ("Validation", "not run; use workflow validate", "warning"),
                (
                    "Next",
                    "workflow show source · workflow show protocol · "
                    "workflow validate",
                    None,
                ),
            ],
        )

    def _resolve_local_module(
        self,
        base: Path,
    ) -> list[Path]:
        paths = [base.with_suffix(".py"), base / "__init__.py"]
        return [path.resolve() for path in paths if path.is_file()]

    def _local_python_dependencies(self, entry: Path) -> list[Path]:
        root = self.workspace.root
        discovered: list[Path] = []
        queued = [entry.resolve()]
        visited: set[Path] = set()
        while queued:
            path = queued.pop(0)
            if path in visited or not path.is_relative_to(root):
                continue
            visited.add(path)
            try:
                tree = ast.parse(
                    path.read_text(encoding="utf-8"),
                    filename=str(path),
                )
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            imports: list[Path] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.extend(
                            self._resolve_local_module(
                                root.joinpath(*alias.name.split("."))
                            )
                        )
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        module_base = path.parent
                        for _ in range(max(0, node.level - 1)):
                            module_base = module_base.parent
                    else:
                        module_base = root
                    if node.module:
                        module_base = module_base.joinpath(
                            *node.module.split(".")
                        )
                    imports.extend(self._resolve_local_module(module_base))
                    if node.module is None or module_base.is_dir():
                        for alias in node.names:
                            imports.extend(
                                self._resolve_local_module(
                                    module_base.joinpath(*alias.name.split("."))
                                )
                            )
            for imported in imports:
                if (
                    imported != entry
                    and imported.is_relative_to(root)
                    and imported not in discovered
                ):
                    discovered.append(imported)
                    queued.append(imported)
        return discovered

    def _workflow_file_records(self) -> list[tuple[str, str]]:
        current, workflow, module = self._current_context(
            purpose="inspect its files"
        )
        module_path = Path(
            self.workspace.absolute_spec(current).partition(":")[0]
        ).resolve()
        records: list[tuple[Path, str]] = [(module_path, "entry point")]
        records.extend(
            (path, "local Python import")
            for path in self._local_python_dependencies(module_path)
        )
        semantics = workflow_semantics(workflow, module)
        deployment = semantics.get("deployment")
        if isinstance(deployment, dict):
            files = deployment.get("files")
            if isinstance(files, list):
                for value in files:
                    candidate = (self.workspace.root / str(value)).resolve()
                    if candidate.is_file() and candidate.is_relative_to(
                        self.workspace.root
                    ):
                        records.append((candidate, "declared deployment file"))
        definitions = semantics.get("action_definitions")
        if isinstance(definitions, dict):
            for definition in definitions.values():
                if not isinstance(definition, dict):
                    continue
                value = definition.get("instructions_file")
                if value:
                    candidate = (self.workspace.root / str(value)).resolve()
                    if candidate.is_file() and candidate.is_relative_to(
                        self.workspace.root
                    ):
                        records.append((candidate, "assistant instructions"))
        unique: list[tuple[str, str]] = []
        seen: set[Path] = set()
        for path, role in records:
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            try:
                display = path.relative_to(self.workspace.root).as_posix()
            except ValueError:
                continue
            unique.append((display, role))
        return unique

    def show_workflow_files(self) -> None:
        records = self._workflow_file_records()
        rows: list[tuple[str, object, StatusKind | None]] = [
            (
                str(index),
                f"{path} — {role}",
                "success" if index == 1 else None,
            )
            for index, (path, role) in enumerate(records, start=1)
        ]
        rows.extend(
            [
                (
                    "Scope",
                    "entry point, statically imported local modules, and "
                    "declared resources",
                    "info",
                ),
                ("Next", "workflow show source [NUMBER|PATH]", None),
            ]
        )
        self._emit_table(
            "Workflow files",
            rows,
        )

    def show_workflow_source(self, args: list[str]) -> None:
        if len(args) > 1:
            raise SystemExit("Use workflow show source [NUMBER|PATH].")
        records = self._workflow_file_records()
        if not records:
            raise SystemExit("No source files were found for the selected workflow.")
        if not args:
            selected = records[0]
        elif args[0].isdecimal():
            index = int(args[0])
            if index < 1 or index > len(records):
                raise SystemExit(
                    f"Source number must be between 1 and {len(records)}."
                )
            selected = records[index - 1]
        else:
            matches = [record for record in records if record[0] == args[0]]
            if len(matches) != 1:
                raise SystemExit(
                    f"File is not part of the selected workflow: {args[0]}. "
                    "Use 'workflow files' first."
                )
            selected = matches[0]
        path, role = selected
        target = self.workspace.root / path
        try:
            source = target.read_text(encoding="utf-8").rstrip()
        except (OSError, UnicodeDecodeError) as exc:
            raise SystemExit(f"Could not read workflow file {path}: {exc}") from exc
        self._emit(f"Source: {path} ({role})")
        self._emit("─" * min(72, len(path) + len(role) + 11))
        self._emit(source)
        self._emit()

    def _agent_names(self, workflow) -> list[str]:
        from zippergen.serve import _workflow_lifelines

        return [lifeline.name for lifeline in _workflow_lifelines(workflow)]

    def show_workflow(self, args: list[str]) -> None:
        view = args[0].lower() if args else ""
        rest = args[1:]
        if view == "source":
            self.show_workflow_source(rest)
            return
        current, workflow, module = self._current_context(
            purpose="inspect it"
        )
        if not view:
            choices = [
                "Authored source",
                "Overview",
                "Protocol",
                "Communications only",
                "Actions and prompts",
                "Complete workflow",
                "One participant",
                "Selected participants",
            ]
            view = str(self._select(f"Inspect {workflow.name}", choices)).lower()

        if view in {"authored source"}:
            self.show_workflow_source([])
            return
        if view in {"overview"}:
            options = ViewOptions(detail="overview")
            remembered = "overview"
        elif view in {"protocol"}:
            options = ViewOptions(detail="protocol")
            remembered = "protocol"
        elif view in {"communications", "communication", "communications only"}:
            options = ViewOptions(detail="protocol", communications_only=True)
            remembered = "communications"
        elif view in {"actions", "actions and prompts"}:
            options = ViewOptions(detail="actions")
            remembered = "actions"
        elif view in {"full", "complete", "complete workflow"}:
            options = ViewOptions(detail="full")
            remembered = "full"
        elif view in {"agent", "one participant"}:
            names = self._agent_names(workflow)
            agent = rest[0] if rest else self._select("Participants", names)
            options = ViewOptions(agent=str(agent))
            remembered = f"agent {agent}"
        elif view in {"agents", "selected participants"}:
            names = self._agent_names(workflow)
            selected = rest or self._select("Participants", names, allow_many=True)
            assert isinstance(selected, list)
            options = ViewOptions(agents=tuple(selected))
            remembered = "agents " + " ".join(selected)
        else:
            raise SystemExit(
                "View must be overview, protocol, communications, actions, full, "
                "agent, or agents."
            )
        try:
            data = workflow_view_data(workflow, module, options=options)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        self.workspace.update(current_workflow=current, last_view=remembered)
        self._emit(data["code"])

    def validate(self) -> None:
        from zippergen.serve import _validate_workflow

        _current, workflow, module = self._current_context(
            purpose="validate it"
        )
        result = _validate_workflow(workflow, module)
        verdict = "valid" if result["valid"] else "invalid"
        summary = self._success if result["valid"] else self._error
        summary(f"Workflow {workflow.name}: {verdict}")
        for check in result["checks"]:  # type: ignore[index]
            status = str(check["status"]).lower()
            emit = {
                "ok": self._success,
                "warn": self._warning,
                "fail": self._error,
            }.get(status, self._info)
            emit(
                f"{check['name']}: {check['detail']}",
                indent=2,
            )

    def _llm_action_lifelines(self, workflow, module) -> dict[str, list[str]]:
        model = workflow_semantics(workflow, module)
        actions: dict[str, list[str]] = {}
        sites = model.get("action_sites") or []
        if isinstance(sites, list):
            for site in sites:
                if not isinstance(site, dict) or site.get("kind") != "llm":
                    continue
                name = str(site.get("lifeline"))
                action = str(site.get("action"))
                actions.setdefault(name, [])
                if action not in actions[name]:
                    actions[name].append(action)
        ordered = self._agent_names(workflow)
        return {name: actions[name] for name in ordered if name in actions}

    def _run_model_profile(self) -> dict[str, object]:
        current = self.workspace.current_workflow
        if not current:
            return {"default": None, "lifelines": {}}
        _current, _workflow, module = self._current_context()
        return self.workspace.model_profile(
            current,
            default=default_llm_spec(module),
        )

    def _configuration_status_kind(
        self,
        configuration: dict[str, str],
    ) -> StatusKind:
        status = configuration.get("check_status", "not_checked")
        if status == "available":
            return "success"
        if status == "unavailable":
            return "error"
        return "warning"

    def _emit_model_configurations(self) -> None:
        configurations = self.workspace.model_configurations()
        self._emit("Configurations")
        self._emit("──────────────")
        for name, configuration in configurations.items():
            status = configuration.get("check_status", "not_checked")
            detail = configuration.get("check_detail", "not checked")
            checked_at = configuration.get("checked_at")
            suffix = f"; checked {checked_at}" if checked_at else ""
            self._status(
                self._configuration_status_kind(configuration),
                f"{name}: {configuration['spec']}; "
                f"{status.replace('_', ' ')} — {detail}{suffix}",
                indent=2,
            )
        self._emit(
            "Configure → check → assign. Names are generated automatically "
            "unless you provide one."
        )

    def _emit_model_assignments(
        self,
        *,
        workflow,
        module,
        assignments: dict[str, object],
    ) -> None:
        active = self._llm_action_lifelines(workflow, module)
        configurations = self.workspace.model_configurations()
        default = str(assignments["default"])
        overrides = assignments.get("lifelines") or {}
        assert isinstance(overrides, dict)
        self._emit("Assignments")
        self._emit("───────────")
        self._emit(f"  Workflow      {workflow.name}")
        self._emit(
            f"  Default       {default} "
            f"({configurations.get(default, {}).get('spec', 'missing')})"
        )
        if not active:
            self._emit("  Assignments   none — no LLM actions in this workflow")
            return
        for lifeline, actions in active.items():
            explicit = overrides.get(lifeline)
            effective = str(explicit or default)
            source = "override" if explicit else "inherits default"
            spec = configurations.get(effective, {}).get("spec", "missing")
            self._emit(
                f"  {lifeline:<13} {effective} → {spec} "
                f"({source}; actions: "
                + ", ".join(actions)
                + ")"
            )

    def _model_configuration_name(self, requested: str) -> str:
        configurations = self.workspace.model_configurations()
        canonical = {
            name.casefold(): name for name in configurations
        }.get(requested.casefold())
        if canonical is None:
            available = ", ".join(configurations) or "none"
            raise SystemExit(
                f"Unknown model configuration {requested!r}. Available: "
                f"{available}. Use 'models configure' to create one."
            )
        return canonical

    def _configure_model_configuration(
        self,
        args: list[str],
        *,
        edit_only: bool = False,
    ) -> None:
        if len(args) > 1:
            command = "edit NAME" if edit_only else "configure [NAME]"
            raise SystemExit(f"Use models {command}.")
        configurations = self.workspace.model_configurations()
        requested = args[0] if args else None
        existing_name = None
        if requested:
            existing_name = {
                name.casefold(): name for name in configurations
            }.get(requested.casefold())
        if edit_only and existing_name is None:
            raise SystemExit(
                f"Unknown model configuration {requested!r}. "
                "Use 'models list' to see available names."
            )
        if existing_name == "mock":
            raise SystemExit("The built-in mock configuration cannot be edited.")

        existing = configurations.get(existing_name or "", {})
        provider_hint = (
            _canonical_provider(requested)
            if requested and requested.casefold() in {
                "local",
                "ollama",
                "openai",
                "anthropic",
                "claude",
                "mistral",
            }
            else None
        )
        default_provider = existing.get("provider") or provider_hint or "local"
        if provider_hint and not existing_name:
            provider = provider_hint
        else:
            entered_provider = self.input(
                f"Provider [{default_provider}]: "
            ).strip()
            provider = _canonical_provider(entered_provider or default_provider)
        if provider not in _SUPPORTED_PROVIDERS:
            raise SystemExit(
                "Provider must be mock, local/ollama, openai, "
                "anthropic/claude, or mistral."
            )
        if provider == "mock":
            self._success("mock is built in and already configured.")
            return
        if not self._provider_is_connected(provider):
            self._info(
                f"{provider} needs a connection before its models can be checked."
            )
            self._connect_model_provider([provider])

        _environment_name, fallback_model = _PROVIDER_DEFAULT_MODELS[provider]
        default_model = (
            existing.get("model")
            if existing.get("provider") == provider
            else None
        ) or fallback_model
        model = self.input(
            f"Model identifier [{default_model}]: "
        ).strip() or default_model
        spec = _validate_model_spec(f"{provider}:{model}")

        if existing_name:
            name = existing_name
        elif requested and provider_hint is None:
            name = requested
        else:
            name = self.workspace.automatic_model_configuration_name(spec)
            if name in configurations:
                self._success(
                    f"Model configuration already exists: {name} ({spec})"
                )
                self._emit(
                    f"Next: models check {name} · "
                    f"models assign LIFELINE {name}"
                )
                return
        try:
            self.workspace.save_model_configuration(
                name,
                {
                    "provider": provider,
                    "model": model,
                    "spec": spec,
                    "check_status": "not_checked",
                    "check_detail": "run 'models check' before assignment",
                },
            )
        except WorkspaceError as exc:
            raise SystemExit(str(exc)) from exc
        verb = "Updated" if existing_name else "Created"
        self._success(f"{verb} model configuration: {name} ({spec})")
        self._emit(
            f"Next: models check {name} · models assign LIFELINE {name}"
        )

    def _check_model_configurations(self, target: str) -> None:
        configurations = self.workspace.model_configurations()
        if target.casefold() == "all":
            selected = list(configurations)
        else:
            selected = [self._model_configuration_name(target)]
        self._emit("Configuration checks")
        self._emit("────────────────────")
        failures: list[str] = []
        for name in selected:
            configuration = configurations[name]
            if name == "mock":
                self._success("mock: built in and available.", indent=2)
                continue
            try:
                verification = self._verify_model_spec(
                    name,
                    configuration["spec"],
                    for_save=False,
                )
            except SystemExit as exc:
                detail = str(exc)
                self.workspace.save_model_configuration(
                    name,
                    {
                        **configuration,
                        "check_status": "unavailable",
                        "check_detail": detail[:240],
                        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    },
                )
                self._error(f"{name}: {detail}", indent=2)
                failures.append(name)
                continue
            check_status = (
                "available"
                if verification.kind == "success"
                else "unverified"
            )
            self.workspace.save_model_configuration(
                name,
                {
                    **configuration,
                    "check_status": check_status,
                    "check_detail": verification.message,
                    "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                },
            )
            self._status(
                verification.kind,
                verification.message,
                indent=2,
            )
        if failures:
            raise SystemExit(
                "Model configuration check failed for "
                + ", ".join(failures)
                + ". Assignments were not changed."
            )
        self._emit()
        self._success("Configuration checks complete; assignments unchanged.")

    def configure_models(self, args: list[str]) -> None:
        action = args[0].lower() if args else "show"

        if action == "configure":
            self._configure_model_configuration(args[1:])
            return

        if action == "edit":
            self._configure_model_configuration(args[1:], edit_only=True)
            return

        if action == "rename":
            if len(args) != 3:
                raise SystemExit("Use models rename OLD_NAME NEW_NAME.")
            old_name = self._model_configuration_name(args[1])
            try:
                result = self.workspace.rename_model_configuration(
                    old_name,
                    args[2],
                )
            except WorkspaceError as exc:
                raise SystemExit(str(exc)) from exc
            new_name = str(result["new_name"])
            configuration = result.get("configuration")
            spec = (
                str(configuration.get("spec") or "unknown")
                if isinstance(configuration, dict)
                else "unknown"
            )
            rows: list[tuple[str, object, StatusKind | None]] = [
                ("From", old_name, None),
                ("To", new_name, "success"),
                ("Model", spec, None),
                (
                    "Check",
                    (
                        str(configuration.get("check_status") or "not checked")
                        if isinstance(configuration, dict)
                        else "not checked"
                    )
                    + "; preserved",
                    None,
                ),
            ]
            references = result.get("references")
            reference_count = 0
            if isinstance(references, (tuple, list)):
                for reference in references:
                    if not isinstance(reference, dict):
                        continue
                    reference_count += 1
                    locations: list[str] = []
                    if reference.get("default"):
                        locations.append("default")
                    lifelines = reference.get("lifelines")
                    if isinstance(lifelines, (tuple, list)):
                        locations.extend(str(value) for value in lifelines)
                    rows.append(
                        (
                            f"Reference {reference_count}",
                            f"{reference.get('workflow')} — "
                            + ", ".join(locations),
                            "success",
                        )
                    )
            if reference_count == 0:
                rows.append(("References", "none assigned", None))
            rows.append(("Next", "models", None))
            self._emit_table("Model configuration renamed", rows)
            return

        if action == "remove":
            if len(args) != 2:
                raise SystemExit("Use models remove NAME.")
            name = self._model_configuration_name(args[1])
            try:
                self.workspace.remove_model_configuration(name)
            except WorkspaceError as exc:
                raise SystemExit(str(exc)) from exc
            self._success(f"Removed model configuration: {name}")
            return

        if action == "connect":
            if len(args) not in {2, 3}:
                raise SystemExit("Use models connect NAME [URL].")
            self._connect_model_provider(args[1:])
            self._emit()
            self._emit_model_connections()
            return

        if action == "disconnect":
            if len(args) != 2:
                raise SystemExit("Use models disconnect NAME.")
            self._disconnect_model_provider(args[1])
            self._emit()
            self._emit_model_connections()
            return

        if action in {"show", "list"}:
            if len(args) > 1:
                raise SystemExit("Use models, models show, or models list.")
            if action == "list":
                self._emit_model_configurations()
                return
            self._emit("Model configuration")
            self._emit("───────────────────")
            self._emit_model_connections()
            self._emit()
            self._emit_model_configurations()
            if not self.workspace.current_workflow:
                self._emit()
                self._emit("Assignments")
                self._emit("───────────")
                self._warning(
                    "No workflow is selected; use 'workflow select' to select one.",
                    indent=2,
                )
                return
            current, workflow, module = self._current_context()
            assignments = self.workspace.model_assignment_profile(
                current,
                default=default_llm_spec(module),
            )
            self._emit()
            self._emit_model_assignments(
                workflow=workflow,
                module=module,
                assignments=assignments,
            )
            return

        if action == "check":
            if len(args) > 2:
                raise SystemExit("Use models check [NAME|all].")
            self._check_model_configurations(
                args[1] if len(args) == 2 else "all"
            )
            return

        current, workflow, module = self._current_context()
        assignments = self.workspace.model_assignment_profile(
            current,
            default=default_llm_spec(module),
        )
        default = str(assignments["default"])
        overrides = dict(assignments.get("lifelines") or {})
        active = self._llm_action_lifelines(workflow, module)
        changed_configuration: str | None = None

        if action == "default" and len(args) == 2:
            default = self._model_configuration_name(args[1])
            changed_configuration = default
        elif action == "assign" and len(args) == 3:
            entered_lifeline, entered_configuration = args[1:]
            lifeline = {
                name.casefold(): name for name in active
            }.get(entered_lifeline.casefold())
            if lifeline is None:
                available = ", ".join(active) or "none"
                raise SystemExit(
                    f"{entered_lifeline!r} has no LLM actions. "
                    f"LLM-active lifelines: "
                    f"{available}."
                )
            configuration = self._model_configuration_name(
                entered_configuration
            )
            overrides[lifeline] = configuration
            changed_configuration = configuration
        elif action == "inherit" and len(args) == 2:
            entered_lifeline = args[1]
            lifeline = {
                name.casefold(): name for name in active
            }.get(entered_lifeline.casefold())
            if lifeline is None:
                available = ", ".join(active) or "none"
                raise SystemExit(
                    f"{entered_lifeline!r} has no LLM actions. "
                    f"LLM-active lifelines: "
                    f"{available}."
                )
            overrides.pop(lifeline, None)
        else:
            raise SystemExit(
                "Use models, models configure [NAME], models check [NAME|all], "
                "models assign LIFELINE NAME, models default NAME, "
                "models inherit LIFELINE, models edit NAME, "
                "models rename OLD_NAME NEW_NAME, or models remove NAME."
            )

        if changed_configuration is not None:
            configuration = self.workspace.model_configurations()[
                changed_configuration
            ]
            status = configuration.get("check_status")
            if status == "unavailable":
                raise SystemExit(
                    f"{changed_configuration} is unavailable. Run "
                    f"'models check {changed_configuration}' again, or edit "
                    "the configuration before assigning it."
                )
            if status != "available":
                self._warning(
                    f"{changed_configuration} is "
                    f"{status or 'not checked'}; "
                    f"use 'models check {changed_configuration}'."
                )
        saved = self.workspace.save_model_assignment_profile(
            current,
            default=default,
            lifelines=overrides,
        )
        self._success(f"Saved model assignments for {workflow.name}.")
        self._emit()
        self._emit_model_assignments(
            workflow=workflow,
            module=module,
            assignments=saved,
        )

    def _resolved_model(self, spec: str) -> tuple[str, str | None]:
        provider = _canonical_provider(spec)
        _prefix, separator, entered_model = spec.partition(":")
        if provider == "mock":
            return provider, None
        if separator:
            return provider, entered_model.strip()
        environment_name, fallback = _PROVIDER_DEFAULT_MODELS[provider]
        return provider, os.environ.get(environment_name, fallback)

    def _provider_api_key(self, provider: str) -> str | None:
        secret_name = _PROVIDER_SECRETS.get(provider)
        if secret_name is None:
            return None
        return os.environ.get(secret_name) or self.workspace.load_secrets().get(
            secret_name
        )

    def _api_model_request(
        self,
        provider: str,
        model: str,
        api_key: str,
    ) -> request.Request:
        encoded_model = quote(model, safe="")
        if provider == "openai":
            base_url = os.environ.get(
                "OPENAI_BASE_URL",
                "https://api.openai.com/v1",
            )
            models_url = self._local_models_url(base_url)
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "ZipperGen-Studio/0.1",
            }
        elif provider == "anthropic":
            models_url = "https://api.anthropic.com/v1/models"
            headers = {
                "Accept": "application/json",
                "anthropic-version": "2023-06-01",
                "x-api-key": api_key,
                "User-Agent": "ZipperGen-Studio/0.1",
            }
        else:
            assert provider == "mistral"
            models_url = "https://api.mistral.ai/v1/models"
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "ZipperGen-Studio/0.1",
            }
        return request.Request(
            f"{models_url}/{encoded_model}",
            headers=headers,
            method="GET",
        )

    def _remote_model_available(
        self,
        provider: str,
        model: str,
        api_key: str,
    ) -> tuple[bool | None, str | None]:
        req = self._api_model_request(provider, model, api_key)
        try:
            with request.urlopen(req, timeout=3.0) as response:
                raw = response.read(1_048_577)
        except HTTPError as exc:
            raw_detail = exc.read(512).decode("utf-8", errors="replace")
            detail = self._local_check_error(raw_detail) if raw_detail.strip() else ""
            suffix = f": {detail}" if detail else ""
            if exc.code in {400, 404, 422}:
                return False, f"HTTP {exc.code}{suffix}"
            if exc.code in {408, 429} or exc.code >= 500:
                return None, f"HTTP {exc.code}{suffix}"
            raise SystemExit(
                f"Could not verify model {model!r} with {provider}: "
                f"HTTP {exc.code}{suffix}. The configuration was not changed."
            ) from exc
        except URLError as exc:
            return None, self._local_check_error(exc.reason)
        except (TimeoutError, OSError) as exc:
            return None, self._local_check_error(exc)
        if len(raw) > 1_048_576:
            return None, "the model response exceeded the 1 MiB safety limit"
        return True, None

    def _verify_model_spec(
        self,
        label: str,
        spec: str,
        *,
        for_save: bool = True,
    ) -> _ModelVerification:
        provider, model = self._resolved_model(spec)
        if provider == "mock":
            return _ModelVerification(
                "success",
                f"{label}: mock is built in and available.",
            )
        assert model is not None
        resolved = f"{spec} (resolved model: {model})" if ":" not in spec else spec
        if provider == "local":
            profile = self.workspace.provider_profiles().get("local", {})
            base_url = profile.get(
                "base_url",
                os.environ.get(
                    "OLLAMA_BASE_URL",
                    "http://127.0.0.1:11434/v1",
                ),
            )
            try:
                result = self._check_local_provider(base_url)
            except _LocalProviderError as exc:
                message = (
                    f"saved {resolved}, but availability could not be checked"
                    if for_save
                    else f"{resolved} could not be checked"
                )
                return _ModelVerification(
                    "warning",
                    f"{label}: {message} at {base_url}: {exc}.",
                )
            if model not in result.model_ids:
                available = ", ".join(result.model_ids[:8]) or "none"
                raise SystemExit(
                    f"Model {model!r} is not available from the local provider "
                    f"at {base_url}. Available models: {available}. "
                    "The configuration was not changed."
                )
            return _ModelVerification(
                "success",
                f"{label}: {resolved} is available from the local provider.",
            )

        api_key = self._provider_api_key(provider)
        if not api_key:
            message = (
                f"saved {resolved}, but it was not checked"
                if for_save
                else f"{resolved} could not be checked"
            )
            return _ModelVerification(
                "warning",
                f"{label}: {message} because "
                f"{provider} is not connected. Use 'models connect {provider}'.",
            )
        available, detail = self._remote_model_available(provider, model, api_key)
        if available is False:
            suffix = f" ({detail})" if detail else ""
            raise SystemExit(
                f"Model {model!r} is not available with the configured "
                f"{provider} API key{suffix}. The configuration was not changed."
            )
        if available is None:
            suffix = f": {detail}" if detail else ""
            message = (
                f"saved {resolved}, but {provider} availability could not be checked"
                if for_save
                else f"{resolved} could not be checked with {provider}"
            )
            return _ModelVerification(
                "warning",
                f"{label}: {message}{suffix}.",
            )
        return _ModelVerification(
            "success",
            f"{label}: {resolved} is available with the configured "
            f"{provider} API key.",
        )

    def _provider_configuration_status(
        self,
        provider: str,
    ) -> tuple[StatusKind, str]:
        canonical = _canonical_provider(provider)
        if canonical == "mock":
            return "success", "available; built in"
        profiles = self.workspace.provider_profiles()
        if canonical == "local":
            profile = profiles.get("local", {})
            base_url = profile.get(
                "base_url",
                os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"),
            )
            check_status = profile.get("check_status")
            checked_at = profile.get("checked_at")
            if check_status == "reachable" and checked_at:
                count = profile.get("model_count", "0")
                noun = "model" if count == "1" else "models"
                kind: StatusKind = "success" if count != "0" else "warning"
                state = (
                    "last check succeeded"
                    if count != "0"
                    else "last check reached the endpoint but found no models"
                )
                return (
                    kind,
                    f"{state}; endpoint {base_url}; {count} {noun}; "
                    f"checked {checked_at}",
                )
            if check_status == "unreachable" and checked_at:
                detail = profile.get("check_error", "connection failed")
                return (
                    "error",
                    f"last check failed; endpoint {base_url}; "
                    f"checked {checked_at}: {detail}",
                )
            return (
                "warning",
                f"not connected; use 'models configure local' for the guided setup",
            )
        secret_name = _PROVIDER_SECRETS.get(canonical)
        if secret_name is None:
            return "error", "unsupported"
        if os.environ.get(secret_name):
            return (
                "success",
                f"connected; {secret_name} is in the environment; not tested here",
            )
        if self.workspace.load_secrets().get(secret_name):
            return (
                "success",
                f"connected; {secret_name} is in private Studio storage; "
                "not tested here",
            )
        return "warning", f"not connected; use 'models connect {canonical}'"

    def _provider_status(self, provider: str) -> str:
        return self._provider_configuration_status(provider)[1]

    def _emit_model_connections(self) -> None:
        self._emit("Connections")
        self._emit("───────────")
        for provider in _SUPPORTED_PROVIDERS:
            kind, status = self._provider_configuration_status(provider)
            self._status(kind, f"{provider}: {status}", indent=2)
        self._emit("API-key values are never displayed or written to the project.")
        self._emit(
            "Normal path: 'models configure' creates a model configuration; "
            "'models check NAME' verifies it."
        )

    def _local_models_url(self, base_url: str) -> str:
        parsed = urlsplit(base_url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise SystemExit(
                "A local provider URL must be a complete http:// or https:// URL."
            )
        if parsed.username or parsed.password:
            raise SystemExit(
                "Do not embed credentials in a local provider URL."
            )
        path = parsed.path.rstrip("/")
        return urlunsplit(
            (parsed.scheme, parsed.netloc, f"{path}/models", "", "")
        )

    def _local_check_error(self, value: object) -> str:
        printable = "".join(
            character if character.isprintable() else " "
            for character in str(value)
        )
        return " ".join(printable.split())[:240] or "connection failed"

    def _check_local_provider(self, base_url: str) -> _LocalProviderCheck:
        models_url = self._local_models_url(base_url)
        req = request.Request(
            models_url,
            headers={
                "Accept": "application/json",
                "User-Agent": "ZipperGen-Studio/0.1",
            },
            method="GET",
        )
        try:
            with request.urlopen(req, timeout=3.0) as response:
                raw = response.read(1_048_577)
        except HTTPError as exc:
            raw_detail = exc.read(512).decode("utf-8", errors="replace")
            detail = self._local_check_error(raw_detail) if raw_detail.strip() else ""
            suffix = f": {detail}" if detail else ""
            raise _LocalProviderError(f"HTTP {exc.code}{suffix}") from exc
        except URLError as exc:
            raise _LocalProviderError(
                self._local_check_error(exc.reason)
            ) from exc
        except (TimeoutError, OSError) as exc:
            raise _LocalProviderError(self._local_check_error(exc)) from exc
        if len(raw) > 1_048_576:
            raise _LocalProviderError(
                "the /models response exceeded the 1 MiB safety limit"
            )
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _LocalProviderError(
                "the /models endpoint did not return valid UTF-8 JSON"
            ) from exc
        models = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            raise _LocalProviderError(
                "the /models response is not OpenAI-compatible "
                "(expected a JSON 'data' list)"
            )
        return _LocalProviderCheck(
            checked_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            model_count=len(models),
            model_ids=tuple(
                str(item["id"])
                for item in models
                if isinstance(item, dict)
                and isinstance(item.get("id"), str)
            ),
        )

    def _save_local_provider_check(
        self,
        base_url: str,
        result: _LocalProviderCheck,
    ) -> None:
        self.workspace.save_provider_profile(
            "local",
            {
                "kind": "local",
                "base_url": base_url,
                "check_status": "reachable",
                "checked_at": result.checked_at,
                "model_count": str(result.model_count),
            },
        )

    def _provider_is_connected(self, provider: str) -> bool:
        canonical = _canonical_provider(provider)
        if canonical == "mock":
            return True
        if canonical == "local":
            profile = self.workspace.provider_profiles().get("local", {})
            return bool(profile.get("base_url"))
        return bool(self._provider_api_key(canonical))

    def _connect_model_provider(self, args: list[str]) -> None:
        if not args:
            raise SystemExit("Use models connect NAME [URL].")
        provider = _canonical_provider(args[0])
        if provider not in _SUPPORTED_PROVIDERS:
            raise SystemExit(
                "Provider must be local/ollama, openai, anthropic/claude, "
                "or mistral."
            )
        if provider == "mock":
            if len(args) != 1:
                raise SystemExit("The built-in mock model takes no settings.")
            self._success("mock is built in and always available.")
            return
        if provider == "local":
            if len(args) > 2:
                raise SystemExit("Use models connect local [BASE_URL].")
            existing = self.workspace.provider_profiles().get("local", {}).get(
                "base_url"
            )
            base_url = (
                args[1]
                if len(args) == 2
                else self.input(
                    "Local OpenAI-compatible base URL "
                    f"[{existing or 'http://127.0.0.1:11434/v1'}]: "
                ).strip()
                or existing
                or "http://127.0.0.1:11434/v1"
            )
            # Prove OpenAI compatibility before replacing a working endpoint.
            self._local_models_url(base_url)
            try:
                result = self._check_local_provider(base_url)
            except _LocalProviderError as exc:
                raise SystemExit(
                    f"Could not connect to the local provider at {base_url}: "
                    f"{exc}. Check that the model server and any SSH tunnel are "
                    "running; the connection was not saved."
                ) from exc
            self._save_local_provider_check(base_url, result)
            noun = "model" if result.model_count == 1 else "models"
            message = (
                f"Connected local provider: reachable; "
                f"{result.model_count} {noun}; endpoint {base_url}"
            )
            if result.model_count:
                self._success(message)
            else:
                self._warning(message + "; install or load a model before running")
            return
        if len(args) != 1:
            raise SystemExit(f"Use models connect {provider}.")
        secret_name = _PROVIDER_SECRETS[provider]
        secrets = self.workspace.load_secrets()
        from_environment = bool(os.environ.get(secret_name))
        if from_environment:
            self._success(
                f"Using {secret_name} from the current environment; "
                "its value was not copied."
            )
        else:
            existing = secrets.get(secret_name)
            suffix = " (press Enter to keep the saved value)" if existing else ""
            entered = self.secret_input(f"{secret_name}{suffix}: ").strip()
            if entered:
                secrets[secret_name] = entered
                self.workspace.save_secrets(secrets)
            elif not existing:
                raise SystemExit(f"{secret_name} must not be empty.")
        self.workspace.save_provider_profile(
            provider,
            {"kind": "api", "key_env": secret_name},
        )
        self._success(f"Connected {provider}: {self._provider_status(provider)}")

    def _disconnect_model_provider(self, name: str) -> None:
        provider = _canonical_provider(name)
        if provider not in _SUPPORTED_PROVIDERS or provider == "mock":
            raise SystemExit(
                "Disconnect must name local, openai, anthropic, or mistral."
            )
        self.workspace.remove_provider_profile(provider)
        secret_name = _PROVIDER_SECRETS.get(provider)
        removed_secret = False
        if secret_name:
            secrets = self.workspace.load_secrets()
            if secret_name in secrets:
                secrets.pop(secret_name)
                self.workspace.save_secrets(secrets)
                removed_secret = True
        detail = (
            " and removed its privately stored API key"
            if removed_secret
            else ""
        )
        self._success(f"Disconnected {provider}{detail}.")
        if secret_name and os.environ.get(secret_name):
            self._warning(
                f"{secret_name} is still present in the current environment."
            )

    def show_runs(self) -> None:
        runs = self.workspace.list_runs()
        if not runs:
            self._emit("No managed development runs.")
            return
        current = self.workspace.current_run_id
        for record in runs:
            marker = "*" if record["run_id"] == current else " "
            self._emit(
                f"{marker} {record['run_id']}  {record['status']}  "
                f"{record['workflow_spec']}"
            )

    def _run_project_cli(self, arguments: list[str]) -> int:
        from zippergen.serve import main

        previous = Path.cwd()
        try:
            os.chdir(self.workspace.root)
            return main(arguments)
        finally:
            os.chdir(previous)

    def _deployment_secret_reuse_arguments(
        self,
        *,
        name: str,
        spec,
        model_specs: tuple[str, ...],
    ) -> list[str]:
        """Offer selected development provider keys as deployment field values."""

        from zippergen.serve import (
            _deployment_profile_path,
            _load_deployment_profile,
            _load_deployment_secrets,
        )

        selected_secret_names = {
            secret_name
            for model_spec in model_specs
            if (
                secret_name := _PROVIDER_SECRETS.get(
                    _canonical_provider(model_spec)
                )
            )
        }
        if not selected_secret_names:
            return []

        available = self.workspace.development_provider_environment(model_specs)
        existing: dict[str, str] = {}
        if _deployment_profile_path(name).exists():
            existing = _load_deployment_secrets(_load_deployment_profile(name))

        selected_fields = [
            field
            for field in spec.fields
            if field.secret
            and field.target_name in selected_secret_names
        ]
        retained_fields = [
            field for field in selected_fields if field.target_name in existing
        ]
        reusable_fields = [
            field
            for field in selected_fields
            if field.target_name in available
            and field.target_name not in existing
        ]
        if not retained_fields and not reusable_fields:
            return []

        arguments: list[str] = []
        if retained_fields:
            retained_names = sorted(
                {field.target_name for field in retained_fields}
            )
            for field in retained_fields:
                arguments.extend(
                    ["--set", f"{field.name}={existing[field.target_name]}"]
                )
            noun = "credential" if len(retained_names) == 1 else "credentials"
            self._success(
                f"Keeping {len(retained_names)} existing deployment {noun}; "
                "values remain hidden."
            )

        if not reusable_fields:
            return arguments

        secret_names = sorted(
            {field.target_name for field in reusable_fields}
        )
        self._emit_table(
            "Deployment credentials",
            [
                (
                    "Available",
                    ", ".join(secret_names) + " in private Studio storage",
                    "success",
                ),
                ("Deployment", name, None),
                ("Storage", "separate private deployment secret file", None),
            ],
        )
        if not self._confirm_action(
            f"Reuse the configured credential"
            f"{'s' if len(secret_names) != 1 else ''} for deployment {name}? "
            "[Y/n]: ",
            cancel_message=(
                "Credential reuse declined; the deployer will request separate "
                "values."
            ),
            default=True,
        ):
            return arguments

        for field in reusable_fields:
            # Studio calls serve.main() in-process. This is not an OS command
            # line, and neither the argument nor its value is rendered.
            arguments.extend(
                ["--set", f"{field.name}={available[field.target_name]}"]
            )
        noun = "credential" if len(secret_names) == 1 else "credentials"
        self._success(
            f"Reusing {len(secret_names)} configured {noun}; values remain "
            "hidden and deployment-scoped."
        )
        return arguments

    def deploy_workflow(self, args: list[str]) -> None:
        from zippergen.deployment import deployment_spec_from_module
        from zippergen.serve import _deployment_name_from_workflow, _slug

        no_start = False
        names: list[str] = []
        for argument in args:
            if argument == "--no-start":
                no_start = True
            elif argument.startswith("--"):
                raise SystemExit(
                    "Use deploy [NAME] [--no-start]; unknown option "
                    f"{argument!r}."
                )
            else:
                names.append(argument)
        if len(names) > 1:
            raise SystemExit("Use deploy [NAME] [--no-start].")
        current, workflow, module = self._current_context()
        target = self.workspace.absolute_spec(current)
        spec = deployment_spec_from_module(module)
        name = _slug(
            names[0]
            if names
            else spec.name or _deployment_name_from_workflow(target, workflow)
        )
        self._emit(f"Guided deployment: {name}")
        arguments = ["deploy", target]
        if names:
            arguments.extend(["--name", name])
        if no_start:
            arguments.append("--no-start")
        profile = self.workspace.model_profile(
            current,
            default=default_llm_spec(module),
        )
        arguments.extend(["--llm", str(profile["default"])])
        overrides = profile.get("lifelines") or {}
        selected_specs = [str(profile["default"])]
        if isinstance(overrides, dict):
            for lifeline, model in sorted(overrides.items()):
                arguments.extend(["--llm-for", f"{lifeline}={model}"])
                selected_specs.append(str(model))
        arguments.extend(
            self._deployment_secret_reuse_arguments(
                name=name,
                spec=spec,
                model_specs=tuple(selected_specs),
            )
        )
        rc = self._run_project_cli(arguments)
        if rc != 0:
            raise SystemExit(f"Deployment {name} did not complete successfully.")
        self.workspace.update(last_deployment=name)
        outcome = "prepared" if no_start else "completed"
        self._success(f"Deployment {outcome}: {name}")

    def deployment_action(self, action: str, args: list[str]) -> None:
        if len(args) > 1:
            raise SystemExit(f"Use {action} or {action} NAME.")
        state = self.workspace.load()
        name = args[0] if args else state.get("last_deployment")
        if not name:
            raise SystemExit(
                "No deployment is remembered. Use 'deploy' or include a name."
            )
        rc = self._run_project_cli([action, str(name)])
        if rc != 0:
            raise SystemExit(f"{action} failed for deployment {name}.")
        self.workspace.update(last_deployment=str(name))
        self._success(f"Deployment {action} completed: {name}")

    def _assistant_skill_instructions(self) -> str:
        manifest = self.workspace.project_manifest()
        framework = manifest.get("framework_directory")
        if not framework:
            return (
                "Use $zippergen-workflows if it is available. Otherwise, if "
                "present, read and follow AGENTS.md and "
                ".agents/skills/zippergen-workflows/SKILL.md completely before "
                "editing workflow code."
            )
        base = Path(str(framework)).as_posix().rstrip("/")
        return (
            "Use $zippergen-workflows if it is available. Otherwise read and "
            f"follow {base}/AGENTS.md, "
            f"{base}/.agents/skills/zippergen-workflows/SKILL.md, and its linked "
            "DSL/CLI reference completely before editing workflow code."
        )

    def _assistant_environment_instructions(self) -> str:
        """Describe the project/runtime boundary without guessing shell state."""

        manifest = self.workspace.project_manifest()
        framework = manifest.get("framework_directory")
        if not framework:
            return (
                "Run application checks from the project root. Use the Python "
                "environment declared by this project's pyproject.toml when it "
                "exists. Run focused application tests by explicit path before "
                "considering a broader suite."
            )
        base = Path(str(framework)).as_posix().rstrip("/")
        quoted = shlex.quote(base)
        return f"""The application project root is the current directory. The
ZipperGen framework checkout is the separate nested project `{base}/`; its
`pyproject.toml` owns the framework dependencies. From the application root,
invoke the framework with `uv run --offline --project {quoted} zippergen ...`
and run
application tests explicitly with
`uv run --offline --project {quoted} pytest tests`. Do not run bare
`uv run pytest` from the application root: it may recursively collect
`{base}/tests` under the wrong environment. The nested framework suite is not
the application's broader suite; run it separately only if this task actually
changes framework source. Pytest is a declared framework development
dependency installed by the tutorial's initial `uv sync`. Do not use
`--with pytest`, install packages, or request network access during the
assistant run.
If pytest is missing, report verification as incomplete and tell the user to
run `uv sync --project {quoted}` once in an ordinary terminal."""

    def _assistant_completion_instructions(self) -> str:
        result = self.workspace.assistant_result_path.relative_to(
            self.workspace.root
        ).as_posix()
        return f"""Before ending, write `{result}` as plain JSON with this shape:

```json
{{
  "schema_version": 1,
  "verification": "passed",
  "summary": "One concise factual sentence.",
  "checks": [
    {{
      "command": "the exact command that was run",
      "status": "passed",
      "detail": "the relevant outcome"
    }}
  ]
}}
```

`verification` must be `passed`, `failed`, or `incomplete`; each check status
must be `passed`, `failed`, or `not_run`. Report `failed` when any requested
check fails. Report `incomplete` when a requested check could not be run or
when the available environment did not test the intended scope. Claim
`passed` only when every requested validation, semantic inspection/diff, and
relevant test completed successfully. A coding-assistant session returning
normally does not turn a failed command into a successful verification."""

    def _task_refresh_instruction(self, refreshes_request: str | None) -> str:
        if refreshes_request is None:
            return ""
        return (
            f"This task refreshes {refreshes_request} because the canonical "
            "specification or pending refinement changed. The documents below "
            "were captured immediately before this task was written."
        )

    def _creation_task_content(
        self,
        *,
        refreshes_request: str | None = None,
    ) -> str:
        context = self.workspace.specification_context()
        refresh_instruction = self._task_refresh_instruction(refreshes_request)
        return f"""# Current ZipperGen task

This generated task is the complete instruction for the coding assistant.
Work in the project root {self.workspace.root}. Keep workflow source and tests
visible in the repository. Do not deploy or start a service.

## Repository guidance

{self._assistant_skill_instructions()}

## Project environment

{self._assistant_environment_instructions()}

## Task

Create a new ZipperGen Python workflow in this project from the requirements
below. Choose a clear module and workflow name under workflows/ unless the
project has a more appropriate established location.

The canonical workflow specification is the durable source of truth.
{refresh_instruction}

{context}

Before editing, summarize participants, owned inputs and outputs, messages,
action kinds, owned decisions and loops, deployment requirements, retry and
safety assumptions, and acceptance examples. Then create visible Python source
and focused mock/fake tests. When deployment metadata is present, keep its
bundle self-contained by including the workflow source and any required
project assets. Run validation, show the communication-only and full code
views, and inspect every new participant's exact local projection. Do not
deploy or start a service. Report generated files, assumptions, and
verification results.

## Required completion record

{self._assistant_completion_instructions()}
"""

    def _refinement_task_content(
        self,
        *,
        workflow_spec: str,
        baseline_file: str | Path,
        refreshes_request: str | None = None,
    ) -> str:
        context = self.workspace.specification_context()
        refresh_instruction = self._task_refresh_instruction(refreshes_request)
        specification_file = self.workspace.specification_path.relative_to(
            self.workspace.root
        ).as_posix()
        return f"""# Current ZipperGen task

This generated task is the complete instruction for the coding assistant.
Work in the project root {self.workspace.root}. Keep workflow source and tests
visible in the repository. Do not deploy or start a service.

## Repository guidance

{self._assistant_skill_instructions()}

## Project environment

{self._assistant_environment_instructions()}

## Task

Refine {workflow_spec} using the canonical specification and the single pending
refinement below. The pending refinement changes only what it says explicitly;
preserve every unaffected requirement and behavior. {refresh_instruction}

{context}

Integrate the requested change coherently into {specification_file} itself so that
the canonical specification remains a clean description of the current
application, not a chronological change log. Do not delete or clear the pending
refinement; the user will reconcile it in Studio after reviewing your changes.

The semantic baseline is {baseline_file}.
Preserve all behavior not explicitly changed.
Update source, deployment metadata, and focused tests together when needed.
Keep any deployment bundle self-contained by including the workflow source and
required project assets.
Validate the result, show communication-only and full code views,
inspect every changed participant's exact local projection, and compare the
result with the baseline using `zippergen diff`. Do not deploy or start a
service. Report assumptions, intended semantic changes, preserved behavior,
and verification results.

## Required completion record

{self._assistant_completion_instructions()}
"""

    def _ensure_current_task_fresh(
        self,
        *,
        announce: bool = True,
        for_assistant: bool = False,
        force: bool = False,
    ) -> dict[str, object] | None:
        record = self.workspace.current_request()
        if record is None:
            return None
        record = self._normalize_task_lifecycle(record)
        if (
            not force
            and record.get("status") == "prepared"
            and record.get("kind") == "refine"
            and self.workspace.pending_refinement() is not None
        ):
            baseline = self.workspace.load().get(
                "pending_specification_fingerprint"
            )
            current_canonical = self.workspace.specification_fingerprint(
                include_pending=False
            )
            if baseline and baseline != current_canonical:
                record = self.workspace.update_request(
                    str(record["request_id"]),
                    status="awaiting_review",
                    manual_integration=True,
                    result_specification_fingerprint=(
                        self.workspace.specification_fingerprint()
                    ),
                    specification_context_changed=True,
                )
                if announce:
                    self._info(
                        "Canonical specification changed while the refinement "
                        "was open; preserving the task for human review."
                    )
                return record
        ensured = self.workspace.ensure_specification()
        if ensured["content"] is None:
            return record
        fingerprint = self.workspace.specification_fingerprint()
        status = str(record.get("status") or "prepared")
        may_refresh = status == "prepared" or (
            for_assistant
            and status in {"assistant_failed", "assistant_interrupted"}
        )
        if not force and not may_refresh:
            return record
        if (
            not force
            and record.get("specification_fingerprint") == fingerprint
            and record.get("task_contract_version")
            == ASSISTANT_TASK_CONTRACT_VERSION
        ):
            return record
        kind = str(record.get("kind") or "")
        workflow_spec = str(record.get("workflow_spec") or "") or None
        baseline_file = str(record.get("baseline_file") or "") or None
        refreshes_request = str(record["request_id"])
        if kind == "create":
            content = self._creation_task_content(
                refreshes_request=refreshes_request,
            )
        elif kind == "refine":
            if workflow_spec is None or baseline_file is None:
                raise WorkspaceError(
                    f"Refinement task {refreshes_request} is missing its workflow "
                    "or semantic baseline. Prepare a new refinement."
                )
            content = self._refinement_task_content(
                workflow_spec=workflow_spec,
                baseline_file=baseline_file,
                refreshes_request=refreshes_request,
            )
        else:
            raise WorkspaceError(
                f"Cannot refresh unsupported task kind {kind!r}. "
                "Use workflow create or workflow refine."
            )
        prompt = (
            self.workspace.pending_refinement()
            if kind == "refine"
            else self.workspace.specification()
        ) or str(record.get("prompt") or "")
        refreshed = self.workspace.save_request(
            kind=kind,
            prompt=prompt,
            content=content,
            workflow_spec=workflow_spec,
            specification_fingerprint=fingerprint,
            baseline_file=baseline_file,
            refreshes_request=refreshes_request,
        )
        if announce:
            self._success(
                "Implementation request refreshed from the current specification context."
            )
        return refreshed

    def create_request(
        self,
        prompt: str,
        *,
        source_path: str | Path | None = None,
        specification_already_saved: bool = False,
    ) -> None:
        if not prompt:
            prompt = self.input("Describe the workflow: ").strip()
        if not prompt:
            raise SystemExit("The workflow description must not be empty.")
        del source_path  # imported content is normalized into Studio's fixed path
        if not specification_already_saved:
            existing = self.workspace.specification()
            if existing is not None and existing != prompt.strip():
                raise SystemExit(
                    "A canonical specification already exists. Use "
                    "'workflow create' or 'workflow edit spec' to reopen it "
                    "instead of replacing it from the command line."
                )
            self.workspace.save_specification(prompt)
        prompt_fingerprint = self.workspace.specification_fingerprint()
        content = self._creation_task_content()
        self.workspace.save_request(
            kind="create",
            prompt=prompt,
            content=content,
            specification_fingerprint=prompt_fingerprint,
        )
        self._emit_table(
            "Creation",
            [
                (
                    "Specification",
                    self.workspace.specification_path.name,
                    "success",
                ),
                ("Implementation", "prepared", "success"),
                (
                    "Next",
                    "workflow implement codex · workflow implement claude",
                    None,
                ),
                ("Inspect", "workflow status · workflow history", None),
            ],
        )

    def refine_request(
        self,
        prompt: str,
        *,
        source_path: str | Path | None = None,
        append: bool = False,
    ) -> None:
        current, workflow, module = self._current_context()
        if not prompt:
            prompt = self.input("Describe the change: ").strip()
        if not prompt:
            raise SystemExit("The refinement description must not be empty.")
        del source_path  # pending refinement always uses Studio's fixed path
        ensured = self.workspace.ensure_specification()
        if ensured["content"] is None:
            raise SystemExit(
                "No workflow specification exists. Use 'workflow create' or "
                "'workflow edit spec' first."
            )
        pending = self.workspace.save_pending_refinement(prompt, append=append)
        self.workspace.requests_directory.mkdir(parents=True, exist_ok=True)
        state = self.workspace.load()
        stored_baseline = state.get("pending_semantic_baseline")
        baseline = Path(str(stored_baseline)) if stored_baseline else None
        if baseline is None or not baseline.exists():
            baseline = self.workspace.requests_directory / (
                f"{time.strftime('%Y%m%d-%H%M%S')}-"
                f"{time.time_ns() % 1_000_000_000:09d}-semantic-before.json"
            )
            baseline.write_text(
                json.dumps(semantic_snapshot(workflow, module), indent=2, default=str)
                + "\n"
            )
            self.workspace.update(pending_semantic_baseline=str(baseline))
        prompt_fingerprint = self.workspace.specification_fingerprint()
        content = self._refinement_task_content(
            workflow_spec=current,
            baseline_file=baseline,
        )
        self.workspace.save_request(
            kind="refine",
            prompt=str(pending["content"]),
            content=content,
            workflow_spec=current,
            specification_fingerprint=prompt_fingerprint,
            baseline_file=baseline,
        )
        self._emit_table(
            "Refinement",
            [
                (
                    "Pending",
                    (
                        "created — .zippergen/pending-refinement.md"
                        if pending["created"]
                        else "updated — .zippergen/pending-refinement.md"
                    ),
                    "success",
                ),
                ("Workflow", current, None),
                ("Baseline", baseline, "success"),
                ("Implementation", "prepared", "success"),
                (
                    "Next",
                    "workflow implement codex · workflow implement claude",
                    None,
                ),
                ("Inspect", "workflow status · workflow history", None),
            ],
        )
