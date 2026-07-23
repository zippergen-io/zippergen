"""A lightweight, discoverable project shell for ZipperGen development."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
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
from zippergen.semantic import semantic_snapshot, workflow_semantics
from zippergen.view import ViewOptions, workflow_view_data
from zippergen.workspace import SPECIFICATION_GUIDE, Workspace, WorkspaceError


InputFunc = Callable[[str], str]
OutputFunc = Callable[[str], object]
SecretInputFunc = Callable[[str], str]
StatusKind = Literal["success", "warning", "error", "info"]


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
    "assistant",
    "create",
    "current",
    "deploy",
    "doctor",
    "edit",
    "editor",
    "help",
    "inspect",
    "logs",
    "models",
    "project",
    "prompts",
    "providers",
    "quit",
    "refine",
    "restart",
    "resume",
    "run",
    "runs",
    "show",
    "spec",
    "start",
    "status",
    "stop",
    "task",
    "use",
    "validate",
    "workflow",
}

_COMMAND_COMPLETIONS = (
    ("project", "initialize, inspect, or reset the project"),
    ("create", "write or reopen the canonical specification"),
    ("spec", "inspect or refine the workflow specification"),
    ("task", "inspect the current coding-assistant task"),
    ("assistant", "run Codex or Claude on the current task"),
    ("editor", "inspect or configure the terminal editor"),
    ("edit", "edit the selected workflow or another project file"),
    ("use", "select a discovered workflow"),
    ("current", "show project, workflow, model, and runtime context"),
    ("show", "inspect code-first workflow views"),
    ("inspect", "alias for show"),
    ("workflow", "alias for current"),
    ("validate", "validate the selected workflow"),
    ("models", "configure default and participant-specific models"),
    ("providers", "show or configure model-provider settings"),
    ("run", "start a managed development run"),
    ("resume", "resume the current incomplete run"),
    ("runs", "list managed development runs"),
    ("refine", "short alias for spec refine"),
    ("deploy", "prepare or start a named deployment"),
    ("status", "show deployment status"),
    ("doctor", "check deployment readiness"),
    ("logs", "show deployment logs"),
    ("start", "start a deployment"),
    ("restart", "restart a deployment"),
    ("stop", "stop a deployment"),
    ("help", "show all Studio commands"),
    ("exit", "leave Studio"),
    ("quit", "alias for exit"),
)

_SUBCOMMAND_COMPLETIONS = {
    "project": (
        ("init", "create the visible project manifest"),
        ("show", "show visible project configuration"),
        ("reset", "back up and reset private project state"),
    ),
    "spec": (
        ("show", "show the canonical specification"),
        ("edit", "edit the canonical specification"),
        ("path", "print the automatic specification path"),
        ("refine", "create or reopen the pending refinement"),
        ("pending", "show the pending refinement"),
        ("reconcile", "accept an integrated refinement"),
        ("discard", "archive an unwanted refinement"),
        ("history", "list reconciled and discarded refinements"),
    ),
    "task": (
        ("show", "show the complete assistant task"),
        ("path", "print the generated task path"),
        ("history", "list previous assistant tasks"),
        ("close", "close a reviewed creation task"),
    ),
    "assistant": (
        ("codex", "run Codex once on the current task"),
        ("claude", "run Claude Code once on the current task"),
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
        ("show", "show effective model routing"),
        ("check", "check configured models without changing routing"),
        ("default", "set the inherited default model"),
        ("set", "override one LLM-active participant"),
        ("reset", "restore inheritance or reset all routing"),
    ),
    "providers": (
        ("show", "show configuration and the last local check"),
        ("set", "configure a provider"),
        ("check", "recheck local-provider connectivity"),
        ("reset", "remove provider configuration"),
    ),
}

_MODEL_COMPLETIONS = (
    ("mock", "deterministic built-in model"),
    ("local:", "local OpenAI-compatible model"),
    ("openai:", "OpenAI model"),
    ("anthropic:", "Anthropic model"),
    ("mistral:", "Mistral model"),
)


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
    return spec


_HELP = """Commands:
  project init [NAME]            create the project manifest
  project show                   show visible project configuration
  project reset                  choose fresh design or state-only reset
  project reset fresh [--yes]    archive manifest, spec, legacy prompts, state
  project reset state [--yes]    reset private state; keep all project files
  create                         write/reopen the canonical specification
  create [DESCRIPTION]           set the canonical specification without an editor
  create --file PATH             import a specification; its name is not retained
  spec show                      show the canonical workflow specification
  spec edit                      edit it at Studio's automatic path
  spec path                      print that automatic path
  spec refine                    create/reopen the one pending refinement
  refine                         short alias for spec refine
  spec pending                   inspect the pending refinement
  spec reconcile [--yes]         accept an integrated refinement and clear it
  spec discard [--yes]           archive an unwanted pending refinement
  spec history                   list reconciled/discarded private history
  task                            show the current freshness-checked task
  task show|path|history          inspect the synchronized task or its history
  task close [--yes]              close a reviewed creation task
  assistant [codex|claude]       sync the spec, then run a coding assistant
  assistant [codex|claude] --rerun
                                 deliberately rerun a task awaiting review
  assistant codex --interactive  open an interactive Codex session instead
  editor [show|set CMD|reset]     inspect or remember the terminal editor
  edit [workflow|file PATH]       edit with the remembered/default editor
  edit ... --editor CMD           choose an editor for this invocation only
  prompts                        legacy prompt-ledger migration/compatibility
  use [PATH.py:WORKFLOW]         select a workflow; no argument opens a selector
  current                        show the complete project/workflow dashboard
  show | inspect                 choose a code-first semantic view
  show overview|protocol|communications|actions|full
  show agent [NAME]              exact local projection (selector if omitted)
  show agents [NAME ...]         selected-participant focus view
  validate                       validate the current workflow
  models                         configure default/per-lifeline LLMs interactively
  models show                    show the current workflow's model routing
  models check [all|default|LIFELINE]
                                 verify effective models without changing them
  models default SPEC            set and verify the inherited default LLM
  models set LIFELINE SPEC       override and verify one LLM-active lifeline
  models reset LIFELINE|all      restore inheritance or reset the whole profile
  providers                      show configuration and the last local check
  providers set openai|anthropic|mistral
  providers set local [URL]      configure a local OpenAI-compatible endpoint
  providers check local          recheck the saved local endpoint
  providers reset NAME           remove a saved provider configuration
  run [LLM] [--assistant TOOL]   start a run with optional one-run backends
  resume                         resume the current incomplete run
  runs                           list managed development runs
  refine [CHANGE]                append to the one pending refinement
  refine --file PATH             import text into that pending refinement
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

        label = f" Output: {command} "
        self._emit()
        self._emit(f"──{label}{'─' * max(2, 58 - len(label))}")

    def _prompt(self) -> str:
        current = self.workspace.current_workflow
        label = current.rsplit(":", 1)[-1] if current else "no workflow"
        return f"zippergen [{label}]> "

    def welcome(self) -> None:
        self._emit("ZipperGen Studio")
        self._emit(f"Project: {self.workspace.root}")
        current = self.workspace.current_workflow
        self._emit(f"Workflow: {current}" if current else "No workflow selected.")
        self._emit(
            "Type 'help' for commands; press Tab to complete; "
            "'show' opens the inspection menu."
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
        if command == "inspect":
            command = "show"
        if command == "show":
            if not args:
                return list(_SUBCOMMAND_COMPLETIONS["show"])
            if args[0].lower() in {"agent", "agents"}:
                used = {value.lower() for value in args[1:]}
                return [
                    (name, "workflow participant")
                    for name in self._completion_lifelines()
                    if name.lower() not in used
                ]
            return []
        if command == "use":
            try:
                workflows = self.workspace.discover_workflows()
            except (WorkspaceError, OSError):
                workflows = []
            return [(value, "discovered workflow") for value in workflows]
        if command == "models":
            if not args:
                return list(_SUBCOMMAND_COMPLETIONS["models"])
            action = args[0].lower()
            if action == "default":
                return list(_MODEL_COMPLETIONS)
            if action == "set":
                if len(args) == 1:
                    return [
                        (name, "LLM-active participant")
                        for name in self._completion_lifelines(llm_only=True)
                    ]
                return list(_MODEL_COMPLETIONS)
            if action == "check" and len(args) == 1:
                return [
                    ("all", "default and all LLM-active participants"),
                    ("default", "inherited default model"),
                ] + [
                    (name, "effective model for this LLM-active participant")
                    for name in self._completion_lifelines(llm_only=True)
                ]
            if action == "reset" and len(args) == 1:
                return [("all", "reset the complete model profile")] + [
                    (name, "LLM-active participant")
                    for name in self._completion_lifelines(llm_only=True)
                ]
            return []
        if command == "providers":
            if not args:
                return list(_SUBCOMMAND_COMPLETIONS["providers"])
            if args[0].lower() == "check":
                return [("local", "saved local OpenAI-compatible endpoint")]
            if args[0].lower() in {"set", "reset"}:
                return [
                    (name, "model provider") for name in _SUPPORTED_PROVIDERS
                    if name != "mock"
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
        if command in {"create", "refine"}:
            if "--file" in args and args[-1] == "--file":
                return self._path_completion_candidates(fragment)
            if "--editor" in args and args[-1] == "--editor":
                return self._editor_completion_candidates()
            if not args:
                return [
                    ("--file", "import text from an existing file"),
                    ("--editor", "choose an editor for this invocation"),
                ]
            return []
        if command == "spec" and args and args[0].lower() == "refine":
            if "--file" in args and args[-1] == "--file":
                return self._path_completion_candidates(fragment)
            if "--editor" in args and args[-1] == "--editor":
                return self._editor_completion_candidates()
            return [
                ("--file", "import text into the pending refinement"),
                ("--editor", "choose an editor for this invocation"),
            ]
        if command == "spec" and args and args[0].lower() == "edit":
            if "--editor" in args and args[-1] == "--editor":
                return self._editor_completion_candidates()
            return [("--editor", "choose an editor for this invocation")]
        if command == "spec" and args and args[0].lower() in {
            "reconcile",
            "discard",
        }:
            return [("--yes", "confirm without another prompt")]
        if command == "task" and args and args[0].lower() == "close":
            return [("--yes", "confirm without another prompt")]
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
        if command == "assistant":
            if not args:
                return list(_SUBCOMMAND_COMPLETIONS["assistant"])
            if args[0].lower() == "codex":
                values = []
                if "--rerun" not in args:
                    values.append(
                        ("--rerun", "deliberately rerun a task awaiting review")
                    )
                if "--interactive" not in args:
                    values.append(
                        ("--interactive", "open an interactive Codex session")
                    )
                return values
            if args[0].lower() == "claude":
                if "--rerun" not in args:
                    return [
                        ("--rerun", "deliberately rerun a task awaiting review")
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
        if command == "prompts":
            legacy = [("list", "list legacy prompt entries")]
            if self.workspace.list_prompts():
                legacy.extend(
                    (action, "inspect legacy prompt history")
                    for action in ("show", "inspect", "path", "context")
                )
            if self.workspace.specification() is None:
                legacy.append(("add", "legacy compatibility only"))
            return legacy
        return []

    def execute(self, line: str, *, show_boundary: bool = False) -> bool:
        try:
            parts = shlex.split(line)
        except ValueError as exc:
            if show_boundary:
                self._emit_output_boundary("input")
            self._error(f"Could not parse command: {exc}")
            return True
        if not parts:
            return True
        command, *args = parts
        command = command.lower()
        if command in {"exit", "quit"}:
            return False
        if show_boundary:
            self._emit_output_boundary(
                command if command in _STUDIO_COMMANDS else "command"
            )
        if command in {"help", "?"}:
            self._emit(_HELP.rstrip())
        elif command == "project":
            self.configure_project(args)
        elif command == "spec":
            self.manage_spec(args)
        elif command == "prompts":
            self.manage_prompts(args)
        elif command == "task":
            self.manage_task(args)
        elif command == "assistant":
            self.run_assistant(args)
        elif command == "editor":
            self.configure_editor(args)
        elif command == "edit":
            self.edit_file(args)
        elif command in {"current", "workflow"}:
            self.show_current()
        elif command == "use":
            self.use_workflow(args)
        elif command in {"show", "inspect"}:
            self.show_workflow(args)
        elif command == "validate":
            self.validate()
        elif command == "models":
            self.configure_models(args)
        elif command == "providers":
            self.configure_providers(args)
        elif command == "run":
            assistant_backend = None
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
        elif command == "create":
            self.create_from_command(args)
        elif command == "refine":
            self.manage_spec(["refine", *args], alias=True)
        elif command == "deploy":
            self.deploy_workflow(args)
        elif command in {"status", "doctor", "logs", "start", "restart", "stop"}:
            self.deployment_action(command, args)
        else:
            self._error(
                f"Unknown command: {command}. Type 'help' for available commands."
            )
        return True

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
                "guide was kept; enter 'create' and write below its comment."
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
            usage="Use create [DESCRIPTION], create --file PATH, or "
            "create [--edit] [--editor COMMAND].",
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
                "--editor is only used when create opens the specification editor."
            )
        if values[0] == "--file":
            if len(values) != 2:
                raise SystemExit("Use create --file PATH.")
            entered = Path(values[1]).expanduser()
            source = (
                entered
                if entered.is_absolute()
                else self.workspace.root / entered
            ).resolve()
            prompt = self._read_prompt_file(source)
        elif "--file" in values or "--edit" in values:
            raise SystemExit(
                "Use create [DESCRIPTION], create --file PATH, or plain create "
                "to open the automatic specification file."
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
                    ("Status", "not written; use create or spec edit", "warning"),
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
                    "yes; use spec pending"
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

    def manage_spec(self, args: list[str], *, alias: bool = False) -> None:
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
                usage="Use spec edit [--editor COMMAND].",
            )
            if values:
                raise SystemExit("Use spec edit [--editor COMMAND].")
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
                    ("Next", "task · assistant · validate", None),
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
                    [("Status", "none; use spec refine", None)],
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
                next_action = "task · assistant codex · assistant claude"
            else:
                pending_status = "waiting to be integrated"
                pending_kind = "warning"
                next_action = "assistant codex · assistant claude"
            self._emit_table(
                "Pending refinement",
                [
                    ("Status", pending_status, pending_kind),
                    ("File", ".zippergen/pending-refinement.md", None),
                    ("Edit", "spec refine", None),
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
                    "No workflow selected. Use 'use' before preparing a refinement."
                )
            ensured = self.workspace.ensure_specification()
            if ensured["content"] is None:
                raise SystemExit(
                    "No workflow specification exists. Use 'create' or "
                    "'spec edit' first."
                )
            values, editor_override = self._editor_override(
                rest,
                usage="Use spec refine [CHANGE|--file PATH] "
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
                    raise SystemExit("Use spec refine --file PATH.")
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
                    "Use spec refine [CHANGE|--file PATH] [--editor COMMAND]."
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
            if alias:
                self._info("'refine' is the short alias for 'spec refine'.")
            return
        if action in {"reconcile", "discard"}:
            if rest not in ([], ["--yes"]):
                raise SystemExit(f"Use spec {action} [--yes].")
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
                        "refinement began. Run the assistant or use 'spec edit' to "
                        "integrate the change before reconciling it."
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
                    ("Task", "closed; private history retained", "success"),
                    ("History", result["history_path"], None),
                    ("Next", "spec show · current", None),
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
            "Use spec show, spec edit, spec path, spec refine, spec pending, "
            "spec reconcile [--yes], spec discard [--yes], or spec history."
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
            configured = self.workspace.load().get("editor_command")
            if configured:
                candidates = [
                    (self._parse_editor_command(configured), "project preference")
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
            if source in {"one-off", "project preference"}:
                raise SystemExit(
                    f"Editor executable was not found: {command[0]}. "
                    "Use 'editor set COMMAND' or 'editor reset'."
                )
        raise SystemExit(
            "No terminal editor was found. Install micro/nano/vim, set $VISUAL "
            "or $EDITOR, or use 'editor set COMMAND'."
        )

    def configure_editor(self, args: list[str]) -> None:
        if not args or args == ["show"]:
            command, source = self._effective_editor()
            preference = self.workspace.load().get("editor_command")
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
            self.workspace.update(editor_command=command)
            self._success(f"Editor preference: {shlex.join(command)}")
            return
        if action == "reset" and not rest:
            self.workspace.update(editor_command=None)
            self._success("Editor preference reset to automatic discovery.")
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
                    "Use edit [workflow|file PATH] [--editor COMMAND]."
                )
            editor_override = args[-1]
            args = args[:index]
        if not args or args == ["workflow"]:
            current = self.workspace.current_workflow
            if current is None:
                raise SystemExit("No workflow selected. Use 'use' first.")
            module_ref = self.workspace.absolute_spec(current).partition(":")[0]
            target = Path(module_ref)
            if target.suffix != ".py" or not target.is_file():
                raise SystemExit(
                    f"The selected workflow is not backed by a Python file: {current}"
                )
            target = self._project_path(target, label="Workflow")
            next_steps = "validate · show · run"
        elif len(args) == 2 and args[0] == "file":
            target = self._project_path(args[1])
            next_steps = "inspect the change; this generic edit was not registered"
        elif len(args) == 1:
            target = self._project_path(args[0])
            next_steps = "inspect the change; this generic edit was not registered"
        else:
            raise SystemExit(
                "Use edit, edit workflow, or edit file PATH [--editor COMMAND]."
            )
        self._launch_editor(target, override=editor_override)
        self._emit(f"Next: {next_steps}")

    def _current_context(self):
        from zippergen.serve import load_workflow_spec

        current = self.workspace.current_workflow
        if not current:
            raise SystemExit("No workflow selected. Use 'use' or 'create' first.")
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
                "Use project show, project init [NAME], project reset, "
                "project reset fresh [--yes], or project reset state [--yes]."
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
                ("Assistant tasks", summary["requests"], None),
                ("Development secrets", summary["development_secrets"], None),
                (
                    "Deployments",
                    "kept and not stopped; remembered name is cleared",
                    "warning" if summary["last_deployment"] else None,
                ),
                (
                    "Next",
                    (
                        "project init · create"
                        if fresh
                        else "use · create · current"
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
                ("Assistant task", "none", None),
                (
                    "Next",
                    (
                        "project init · create"
                        if fresh and project_exists
                        else "use · create · current"
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
                [("Status", "none; use create, refine, or prompts add", "warning")],
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
                "prompts remain inspectable, but cannot be changed. Use 'spec "
                "edit' for the accepted specification or 'spec refine' for a "
                "pending change."
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

    def _task_next(self, record: dict[str, object]) -> str:
        status = str(record.get("status") or "prepared")
        kind = str(record.get("kind") or "")
        if status == "awaiting_review":
            if kind == "refine":
                if record.get("specification_context_changed") is False:
                    return (
                        "spec edit · assistant codex --rerun · "
                        "assistant claude --rerun"
                    )
                return "current · validate · show · run · spec reconcile"
            return "use · current · validate · show · task close"
        if status == "assistant_running":
            return "wait for the assistant session to return"
        if status in {"assistant_failed", "assistant_interrupted"}:
            return "inspect changes · assistant codex · assistant claude"
        return "assistant codex · assistant claude"

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
        return "changed since this task was prepared", "warning"

    def manage_task(self, args: list[str]) -> None:
        if len(args) > 2 or (
            args and args[0].lower() not in {"show", "path", "history", "close"}
        ):
            raise SystemExit(
                "Use task, task show, task path, task history, or "
                "task close [--yes]."
            )
        action = args[0].lower() if args else "summary"
        rest = args[1:]
        if action != "close" and rest:
            raise SystemExit(
                "Use task, task show, task path, task history, or "
                "task close [--yes]."
            )
        if action == "history":
            records = self.workspace.list_requests()
            if not records:
                self._emit_table(
                    "Task history",
                    [("Status", "none; use create or spec refine", "warning")],
                )
                return
            self._emit("Task history")
            self._emit("────────────")
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
                    "No current task. Use create or spec refine to prepare one."
                )
            self._emit_table(
                "Current task",
                [("Status", "none; use create or spec refine", "warning")],
            )
            return
        record = self._normalize_task_lifecycle(record)
        if action == "close":
            if rest not in ([], ["--yes"]):
                raise SystemExit("Use task close [--yes].")
            if self.workspace.pending_refinement() is not None:
                raise SystemExit(
                    "A refinement is still pending. Review it, then use "
                    "'spec reconcile' to accept it or 'spec discard' to reject it."
                )
            if rest != ["--yes"] and not self._confirm_action(
                "Close the current reviewed task? [y/n]: ",
                cancel_message="Task close cancelled; nothing was changed.",
            ):
                return
            closed = self.workspace.clear_current_task()
            self._emit_table(
                "Task closed",
                [
                    ("Status", "closed", "success"),
                    ("Request", closed["request_id"], None),
                    ("History", "retained; use task history", None),
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
            "Current task",
            [
                ("Status", state, state_kind),
                ("Kind", record["kind"], None),
                ("Request", record["request_id"], None),
                ("Workflow", record.get("workflow_spec") or "new workflow", None),
                ("Assistant", self._task_assistant(record), None),
                ("Execution", self._task_execution(record), None),
                ("Refreshes", record.get("refreshes_request") or "—", None),
                ("Context", context, context_kind),
                ("File", ".zippergen/current-task.md", None),
                ("Next", self._task_next(record), None),
            ],
        )

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
                "Use assistant, assistant codex, assistant claude, or "
                "assistant [codex|claude] --rerun. Use "
                "assistant codex --interactive only for an interactive session."
            )
        assistant = values[0].lower() if values else "codex"
        if interactive and assistant != "codex":
            raise SystemExit(
                "--interactive is supported only with assistant codex."
            )
        record = self._ensure_current_task_fresh(for_assistant=True)
        if record is None:
            raise SystemExit(
                "No current task. Use create or spec refine before starting the "
                "assistant."
            )
        status = str(record.get("status") or "prepared")
        if status == "awaiting_review":
            manual_first_pass = bool(record.get("manual_integration")) and not bool(
                record.get("assistant")
            )
            if not rerun and not manual_first_pass:
                raise SystemExit(
                    "The assistant has already returned and this task is awaiting "
                    "human review. Use current, validate, show, and then "
                    "'spec reconcile'; use 'assistant "
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
                "This task is already marked as running. Wait for the assistant "
                "session to return; after an interrupted Studio process, prepare "
                "or refine the task again before retrying."
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
                "the current task remains available at "
                f"{self.workspace.current_task_path}."
            )
        started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        record = self.workspace.update_request(
            str(record["request_id"]),
            status="assistant_running",
            assistant=("Claude Code" if assistant == "claude" else "Codex"),
            assistant_started_at=started_at,
            assistant_finished_at=None,
            assistant_exit_code=None,
            assistant_mode=("interactive" if interactive else "one_shot"),
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
                        "interactive task session"
                        if interactive
                        else "one-shot task; returns to Studio automatically"
                    ),
                    None,
                ),
                ("Task", relative_task, "success"),
                ("Project", self.workspace.root, None),
                (
                    "MCP",
                    "not required; the assistant keeps its own configured tools",
                    None,
                ),
            ],
        )
        instruction = (
            f"Read and execute {relative_task}. Follow the repository instructions, "
            "keep all generated code visible, run the requested verification, and "
            "do not deploy."
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
        try:
            completed = subprocess.run(
                command,
                cwd=self.workspace.root,
                check=False,
            )
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
        if completed.returncode != 0:
            self.workspace.update_request(
                str(record["request_id"]),
                status="assistant_failed",
                assistant_finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                assistant_exit_code=completed.returncode,
                result_specification_fingerprint=(
                    self.workspace.specification_fingerprint()
                ),
            )
            raise SystemExit(
                f"{assistant.capitalize()} exited with status "
                f"{completed.returncode}; the task remains at "
                f"{self.workspace.current_task_path}."
            )
        result_fingerprint = self.workspace.specification_fingerprint()
        changed = record.get("specification_fingerprint") != result_fingerprint
        record = self.workspace.update_request(
            str(record["request_id"]),
            status="awaiting_review",
            assistant_finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            assistant_exit_code=0,
            result_specification_fingerprint=result_fingerprint,
            specification_context_changed=changed,
        )
        self._success(
            f"{'Claude Code' if assistant == 'claude' else 'Codex'} session "
            "ended successfully."
        )
        kind = str(record.get("kind") or "")
        specification_result = (
            "changed since task preparation"
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
                ("Next", self._task_next(record), None),
            ],
        )

    def show_current(self) -> None:
        from zippergen.serve import _validate_workflow

        state = self.workspace.load()
        manifest = self.workspace.project_manifest()
        request = self._ensure_current_task_fresh(announce=False)
        specification = self.workspace.specification()
        pending = self.workspace.pending_refinement()
        task_state, task_state_kind = (
            self._task_state(request)
            if request
            else ("none; use create or spec refine", "warning")
        )
        refinement_status = (
            (
                "pending — awaiting human review; use spec pending"
                if request
                and request.get("kind") == "refine"
                and request.get("status") == "awaiting_review"
                else "pending — use spec pending or spec refine"
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
                        else "not written; use create or spec edit"
                    ),
                    "success" if specification is not None else "warning",
                ),
                (
                    "Refinement",
                    refinement_status,
                    "warning" if pending is not None else None,
                ),
                (
                    "Task",
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
                (
                    "Editor",
                    (
                        shlex.join(
                            self._parse_editor_command(state["editor_command"])
                        )
                        if state.get("editor_command")
                        else "automatic; use editor show or editor set COMMAND"
                    ),
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
            profile = self.workspace.model_profile(
                str(state["current_workflow"]),
                default=default_llm_spec(module),
            )
            overrides = profile.get("lifelines") or {}
            assert isinstance(overrides, dict)
            model_rows: list[tuple[str, object, StatusKind | None]] = [
                ("Default", profile["default"], None)
            ]
            if active_models:
                for lifeline, actions in active_models.items():
                    explicit = overrides.get(lifeline)
                    effective = str(explicit or profile["default"])
                    source = "override" if explicit else "default"
                    model_rows.append(
                        (
                            lifeline,
                            f"{effective} ({source}; actions: {', '.join(actions)})",
                            None,
                        )
                    )
            else:
                model_rows.append(("Assignments", "none", None))
            selected_specs = {str(profile["default"])} | {
                str(value) for value in overrides.values()
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

    def use_workflow(self, args: list[str]) -> None:
        from zippergen.serve import load_workflow_spec

        if len(args) > 1:
            raise SystemExit("Use one workflow spec: use PATH.py:WORKFLOW")
        selected = args[0] if args else self._select(
            "Workflows", self.workspace.discover_workflows()
        )
        if not isinstance(selected, str):
            raise SystemExit("Select one workflow.")
        canonical = self.workspace.canonical_spec(selected)
        workflow, _module = load_workflow_spec(self.workspace.absolute_spec(canonical))
        self.workspace.select_workflow(canonical, cwd=self.workspace.root)
        self._success(f"Current workflow: {canonical} ({workflow.name})")

    def _agent_names(self, workflow) -> list[str]:
        from zippergen.serve import _workflow_lifelines

        return [lifeline.name for lifeline in _workflow_lifelines(workflow)]

    def show_workflow(self, args: list[str]) -> None:
        current, workflow, module = self._current_context()
        view = args[0].lower() if args else ""
        rest = args[1:]
        if not view:
            choices = [
                "Overview",
                "Protocol",
                "Communications only",
                "Actions and prompts",
                "Complete workflow",
                "One participant",
                "Selected participants",
            ]
            view = str(self._select(f"Inspect {workflow.name}", choices)).lower()

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

        _current, workflow, module = self._current_context()
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

    def _emit_models(
        self,
        *,
        workflow,
        module,
        profile: dict[str, object],
    ) -> None:
        active = self._llm_action_lifelines(workflow, module)
        default = str(profile["default"])
        overrides = profile.get("lifelines") or {}
        assert isinstance(overrides, dict)
        self._emit(f"Models for {workflow.name}")
        self._emit(f"  Default: {default}")
        if not active:
            self._emit("  No LLM actions are present in this workflow.")
            return
        for lifeline, actions in active.items():
            explicit = overrides.get(lifeline)
            effective = str(explicit or default)
            source = "override" if explicit else "inherits default"
            self._emit(
                f"  {lifeline}: {effective} ({source}; actions: "
                + ", ".join(actions)
                + ")"
            )
        selected = {default} | {str(value) for value in overrides.values()}
        for provider in sorted({_canonical_provider(spec) for spec in selected}):
            kind, status = self._provider_configuration_status(provider)
            self._status(kind, f"Provider {provider}: {status}", indent=2)

    def configure_models(self, args: list[str]) -> None:
        current, workflow, module = self._current_context()
        profile = self.workspace.model_profile(
            current,
            default=default_llm_spec(module),
        )
        default = str(profile["default"])
        overrides = dict(profile.get("lifelines") or {})
        active = self._llm_action_lifelines(workflow, module)
        changed: tuple[str, str] | None = None

        if args and args[0].lower() == "check":
            if len(args) > 2:
                raise SystemExit(
                    "Use models check, models check all, models check default, "
                    "or models check LIFELINE."
                )
            self._check_model_connectivity(
                workflow=workflow,
                profile=profile,
                active=active,
                target=args[1] if len(args) == 2 else "all",
            )
            return

        if not args or args == ["show"]:
            if not args:
                choices = ["Default for all unassigned lifelines"] + [
                    f"{name} ({', '.join(actions)})"
                    for name, actions in active.items()
                ]
                check_choice = "Check effective model connectivity (read-only)"
                choices.append(check_choice)
                selected = str(self._select("Configure models", choices))
                if selected == check_choice:
                    self._check_model_connectivity(
                        workflow=workflow,
                        profile=profile,
                        active=active,
                        target="all",
                    )
                    return
                if selected == choices[0]:
                    entered = self.input(f"Default model [{default}]: ").strip()
                    if entered:
                        default = _validate_model_spec(entered)
                        changed = ("Default", default)
                else:
                    index = choices.index(selected) - 1
                    lifeline = list(active)[index]
                    effective = overrides.get(lifeline, default)
                    entered = self.input(
                        f"Model for {lifeline} [{effective}] "
                        "(type 'inherit' to use the default): "
                    ).strip()
                    if entered.lower() in {"inherit", "default"}:
                        overrides.pop(lifeline, None)
                    elif entered:
                        overrides[lifeline] = _validate_model_spec(entered)
                        changed = (lifeline, overrides[lifeline])
                verification = (
                    self._verify_model_spec(*changed) if changed is not None else None
                )
                profile = self.workspace.save_model_profile(
                    current,
                    default=default,
                    lifelines=overrides,
                )
                if verification is not None:
                    self._status(verification.kind, verification.message)
            self._emit_models(
                workflow=workflow,
                module=module,
                profile=profile,
            )
            return

        action = args[0].lower()
        if action == "default" and len(args) == 2:
            default = _validate_model_spec(args[1])
            changed = ("Default", default)
        elif action == "set" and len(args) == 3:
            lifeline, spec = args[1:]
            if lifeline not in active:
                available = ", ".join(active) or "none"
                raise SystemExit(
                    f"{lifeline!r} has no LLM actions. LLM-active lifelines: "
                    f"{available}."
                )
            overrides[lifeline] = _validate_model_spec(spec)
            changed = (lifeline, overrides[lifeline])
        elif action == "reset" and len(args) == 2:
            if args[1].lower() == "all":
                default = default_llm_spec(module)
                overrides = {}
            else:
                overrides.pop(args[1], None)
        else:
            raise SystemExit(
                "Use models, models show, models check [all|default|LIFELINE], "
                "models default SPEC, models set LIFELINE SPEC, or models reset "
                "LIFELINE|all."
            )

        verification = (
            self._verify_model_spec(*changed) if changed is not None else None
        )
        saved = self.workspace.save_model_profile(
            current,
            default=default,
            lifelines=overrides,
        )
        if verification is not None:
            self._status(verification.kind, verification.message)
        self._success(f"Saved model routing for {workflow.name}.")
        self._emit_models(workflow=workflow, module=module, profile=saved)

    def _check_model_connectivity(
        self,
        *,
        workflow,
        profile: dict[str, object],
        active: dict[str, list[str]],
        target: str,
    ) -> None:
        """Verify effective model routes without modifying their profile."""

        default = str(profile["default"])
        raw_overrides = profile.get("lifelines") or {}
        assert isinstance(raw_overrides, dict)
        overrides = {str(name): str(spec) for name, spec in raw_overrides.items()}
        requested = target.strip()
        normalized = requested.lower()

        if normalized == "all":
            assignments = [("Default", default)] + [
                (lifeline, overrides.get(lifeline, default))
                for lifeline in active
            ]
            scope = "default and all LLM-active participants"
        elif normalized == "default":
            assignments = [("Default", default)]
            scope = "default"
        else:
            matches = {lifeline.lower(): lifeline for lifeline in active}
            lifeline = matches.get(normalized)
            if lifeline is None:
                available = ", ".join(active) or "none"
                raise SystemExit(
                    f"{requested!r} has no LLM actions. Use default, all, or an "
                    f"LLM-active lifeline: {available}."
                )
            assignments = [(lifeline, overrides.get(lifeline, default))]
            scope = lifeline

        routes: dict[str, list[str]] = {}
        for label, spec in assignments:
            routes.setdefault(spec, []).append(label)

        self._emit_table(
            "Model connectivity",
            [
                ("Workflow", workflow.name, None),
                ("Scope", scope, None),
                (
                    "Routes",
                    f"{len(routes)} unique across {len(assignments)} assignment"
                    f"{'s' if len(assignments) != 1 else ''}",
                    None,
                ),
                ("Mode", "read-only; routing will not be changed", "success"),
            ],
        )
        self._emit("Checks")
        self._emit("──────")
        available_count = 0
        warning_count = 0
        failure_count = 0
        failed_labels: list[str] = []
        for spec, labels in routes.items():
            label = ", ".join(labels)
            try:
                verification = self._verify_model_spec(
                    label,
                    spec,
                    for_save=False,
                )
            except SystemExit as exc:
                failure_count += 1
                failed_labels.extend(labels)
                self._error(f"{label}: {exc}", indent=2)
                continue
            if verification.kind == "success":
                available_count += 1
            elif verification.kind == "warning":
                warning_count += 1
            else:
                failure_count += 1
                failed_labels.extend(labels)
            self._status(
                verification.kind,
                verification.message,
                indent=2,
            )
        self._emit()

        if failure_count:
            result = "one or more selected models are unavailable"
            result_kind: StatusKind = "error"
        elif warning_count:
            result = "one or more selected models could not be verified"
            result_kind = "warning"
        else:
            result = "all selected models are available"
            result_kind = "success"
        self._emit_table(
            "Connectivity result",
            [
                ("Status", result, result_kind),
                ("Available routes", available_count, None),
                (
                    "Unverified routes",
                    warning_count,
                    "warning" if warning_count else None,
                ),
                (
                    "Unavailable routes",
                    failure_count,
                    "error" if failure_count else None,
                ),
                ("Routing", "unchanged", "success"),
            ],
        )
        if failure_count:
            raise SystemExit(
                "Model connectivity check failed for "
                + ", ".join(failed_labels)
                + ". Routing was not changed."
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
                f"HTTP {exc.code}{suffix}. Model routing was not changed."
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
                    "Model routing was not changed."
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
                f"{provider} is not configured. Use 'providers set {provider}'.",
            )
        available, detail = self._remote_model_available(provider, model, api_key)
        if available is False:
            suffix = f" ({detail})" if detail else ""
            raise SystemExit(
                f"Model {model!r} is not available with the configured "
                f"{provider} API key{suffix}. Model routing was not changed."
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
                f"not checked; endpoint {base_url}; use 'providers check local'",
            )
        secret_name = _PROVIDER_SECRETS.get(canonical)
        if secret_name is None:
            return "error", "unsupported"
        if os.environ.get(secret_name):
            return (
                "success",
                f"configured; {secret_name} is in the environment; not tested here",
            )
        if self.workspace.load_secrets().get(secret_name):
            return (
                "success",
                f"configured; {secret_name} is in private Studio storage; "
                "not tested here",
            )
        return "warning", f"not configured; use 'providers set {canonical}'"

    def _provider_status(self, provider: str) -> str:
        return self._provider_configuration_status(provider)[1]

    def _emit_providers(self) -> None:
        self._emit("Provider configuration")
        for provider in _SUPPORTED_PROVIDERS:
            kind, status = self._provider_configuration_status(provider)
            self._status(kind, f"{provider}: {status}", indent=2)
        self._emit("API-key values are never displayed or written to the project.")
        self._emit(
            "Use 'models check' for current configured-model availability; "
            "use 'providers check local' to refresh the local endpoint."
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

    def configure_providers(self, args: list[str]) -> None:
        if not args or args == ["show"]:
            self._emit_providers()
            return
        action, *rest = args
        action = action.lower()
        if action == "set" and rest:
            provider = _canonical_provider(rest[0])
            if provider not in _SUPPORTED_PROVIDERS:
                raise SystemExit(
                    "Provider must be mock, local/ollama, openai, "
                    "anthropic/claude, or mistral."
                )
            if provider == "mock":
                if len(rest) != 1:
                    raise SystemExit("The built-in mock provider takes no settings.")
                self._success("mock is built in and already ready.")
                return
            if provider == "local":
                if len(rest) > 2:
                    raise SystemExit("Use providers set local [BASE_URL].")
                existing = self.workspace.provider_profiles().get("local", {}).get(
                    "base_url"
                )
                base_url = (
                    rest[1]
                    if len(rest) == 2
                    else self.input(
                        "Local OpenAI-compatible base URL "
                        f"[{existing or 'http://127.0.0.1:11434/v1'}]: "
                    ).strip()
                    or existing
                    or "http://127.0.0.1:11434/v1"
                )
                # Validate the URL and prove OpenAI compatibility before
                # replacing any previously working endpoint.
                self._local_models_url(base_url)
                try:
                    result = self._check_local_provider(base_url)
                except _LocalProviderError as exc:
                    raise SystemExit(
                        f"Could not verify local provider at {base_url}: {exc}. "
                        "Check that the model server and any SSH tunnel are "
                        "running; the endpoint was not saved."
                    ) from exc
                self._save_local_provider_check(base_url, result)
                noun = "model" if result.model_count == 1 else "models"
                message = (
                    f"Configured local provider: reachable; "
                    f"{result.model_count} {noun}; endpoint {base_url}"
                )
                if result.model_count:
                    self._success(message)
                else:
                    self._warning(message + "; install or load a model before running")
                return
            if len(rest) != 1:
                raise SystemExit(f"Use providers set {provider}.")
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
            self._success(
                f"Configured {provider}: {self._provider_status(provider)}"
            )
            return
        if action == "check" and len(rest) == 1:
            provider = _canonical_provider(rest[0])
            if provider != "local":
                raise SystemExit("Only the local provider has an endpoint check.")
            profile = self.workspace.provider_profiles().get("local")
            if not profile or not profile.get("base_url"):
                raise SystemExit(
                    "No local endpoint is configured. Use 'providers set local'."
                )
            base_url = profile["base_url"]
            try:
                result = self._check_local_provider(base_url)
            except _LocalProviderError as exc:
                checked_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                self.workspace.save_provider_profile(
                    "local",
                    {
                        "kind": "local",
                        "base_url": base_url,
                        "check_status": "unreachable",
                        "checked_at": checked_at,
                        "check_error": str(exc)[:240],
                    },
                )
                raise SystemExit(
                    f"Local provider is unreachable at {base_url}: {exc}. "
                    "Check that the model server and any SSH tunnel are running."
                ) from exc
            self._save_local_provider_check(base_url, result)
            noun = "model" if result.model_count == 1 else "models"
            message = (
                f"Local provider is reachable: {result.model_count} {noun}; "
                f"endpoint {base_url}"
            )
            if result.model_count:
                self._success(message)
            else:
                self._warning(message + "; install or load a model before running")
            return
        if action == "reset" and len(rest) == 1:
            provider = _canonical_provider(rest[0])
            if provider not in _SUPPORTED_PROVIDERS or provider == "mock":
                raise SystemExit(
                    "Reset provider must be local, openai, anthropic, or mistral."
                )
            self.workspace.remove_provider_profile(provider)
            secret_name = _PROVIDER_SECRETS.get(provider)
            if secret_name:
                secrets = self.workspace.load_secrets()
                if secret_name in secrets:
                    secrets.pop(secret_name)
                    self.workspace.save_secrets(secrets)
            self._success(f"Reset provider configuration: {provider}")
            return
        raise SystemExit(
            "Use providers, providers set openai|anthropic|mistral, "
            "providers set local [URL], providers check local, or "
            "providers reset NAME."
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
        if not force and record.get("specification_fingerprint") == fingerprint:
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
                "Use create or spec refine."
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
                "Task refreshed from the current specification context."
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
                    "A canonical specification already exists. Use 'create' or "
                    "'spec edit' to reopen it instead of replacing it from the "
                    "command line."
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
                ("Task", ".zippergen/current-task.md", "success"),
                ("Next", "assistant codex · assistant claude", None),
                ("Inspect", "task · task show · task history", None),
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
                "No workflow specification exists. Use 'create' or 'spec edit' first."
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
                ("Task", ".zippergen/current-task.md", "success"),
                ("Next", "assistant codex · assistant claude", None),
                ("Inspect", "task · task show · task history", None),
            ],
        )
