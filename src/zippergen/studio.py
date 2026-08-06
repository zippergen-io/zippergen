"""A lightweight, discoverable project shell for ZipperGen development."""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast
from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion, CompleteEvent
from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl

from zippergen.dev import default_llm_spec, run_dev
from zippergen.models import normalize_llm_overrides
from zippergen.natural_language import (
    NaturalCommandPlan,
    NaturalLanguageStore,
    deterministic_plan,
    interpreter_prompt,
    looks_sensitive,
    parse_cli_plan,
    requirement_proposal,
)
from zippergen.rendering import StatusKind, TerminalRenderer
from zippergen.semantic import (
    read_semantic_snapshot,
    render_semantic_diff,
    semantic_diff_models,
    semantic_snapshot,
    workflow_semantics,
)
from zippergen.studio_commands import (
    CommandRisk,
    WORKFLOW_VIEWS,
    command_spec,
    concise_help,
    full_help,
    subcommand_completions,
    top_level_completions,
    workflow_view_completions,
    workflow_view_spec,
)
from zippergen.studio_connectors import StudioConnectorsMixin
from zippergen.studio_git import (
    GitCommitError,
    commit_implementation_unit,
    implementation_commit_unit,
    implementation_status_unit,
    uncommitted_commit_unit_paths,
)
from zippergen.studio_models import (
    StudioModelsMixin,
    _PROVIDER_SECRETS,
    _SUPPORTED_PROVIDERS,
    _canonical_provider,
    _validate_model_spec,
)
from zippergen.studio_storage import StudioStorageMixin
from zippergen.view import ViewOptions, workflow_view_data
from zippergen.workspace import (
    ASSISTANT_TASK_CONTRACT_VERSION,
    SPECIFICATION_GUIDE,
    Workspace,
    WorkspaceError,
    discover_workflow_specs,
)


InputFunc = Callable[[str], str]
OutputFunc = Callable[[str], object]
SecretInputFunc = Callable[[str], str]
AssistantVerification = Literal["passed", "failed", "incomplete"]
_ASSISTANT_HEARTBEAT_SECONDS = 10.0
_INSPECTION_WATCH_SECONDS = 1.0


def _valid_inspection_syntax(
    args: list[str],
    *,
    max_positionals: int,
) -> bool:
    """Recognize an inspection command with one optional watch flag."""

    lowered = [value.casefold() for value in args]
    if lowered.count("--watch") > 1:
        return False
    if any(value.startswith("-") and value != "--watch" for value in lowered):
        return False
    return len([value for value in lowered if value != "--watch"]) <= max_positionals


def _valid_storage_compact_syntax(args: list[str]) -> bool:
    positionals = 0
    yes = False
    for raw_value in args:
        value = raw_value.casefold()
        if value == "--yes":
            if yes:
                return False
            yes = True
        elif value.startswith("-"):
            return False
        else:
            positionals += 1
            if positionals > 1:
                return False
    return True


@dataclass(frozen=True)
class _PromptInput:
    content: str
    source_path: Path | None = None
    draft_path: Path | None = None


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


_COMMAND_COMPLETIONS = top_level_completions()

_SUBCOMMAND_COMPLETIONS = {
    parent: subcommand_completions(parent)
    for parent in (
        "studio",
        "project",
        "settings",
        "workflow",
        "language",
        "editor",
        "edit",
        "model",
        "connector",
        "run",
        "deploy",
    )
}
_SUBCOMMAND_COMPLETIONS.update({"show": workflow_view_completions()})

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
    if command_spec([command]) is None:
        return False
    if command in {"exit", "quit", "?", "current"}:
        return not args
    if command == "help":
        return not args or args == ["all"]
    if command in {
        "ask",
        "plan",
    }:
        return True
    if command == "run":
        if not args:
            return True
        action = args[0].casefold()
        if action == "inspect":
            return _valid_inspection_syntax(
                args[1:],
                max_positionals=1,
            )
        if action in {"tasks", "trace"}:
            return len(args) == 1
        if action == "approve":
            return 1 <= len(args) <= 3
        if len(args) == 1:
            return not args[0].startswith("-")
        if len(args) == 2:
            return args[0] == "--assistant"
        return len(args) == 3 and args[1] == "--assistant"
    if command in {"resume", "runs"}:
        return not args
    if command == "deploy":
        if args and args[0].casefold() in {"of", "please", "the"}:
            return False
        if not args:
            return True
        action = args[0].casefold()
        if action == "inspect":
            return _valid_inspection_syntax(
                args[1:],
                max_positionals=2,
            )
        if action == "storage":
            rest = args[1:]
            if not rest:
                return True
            if rest[0].casefold() != "compact":
                return len(rest) == 1 and not rest[0].startswith("-")
            return _valid_storage_compact_syntax(args[2:])
        if action == "remove":
            return len(args) <= 4
        if action == "logs" and len(args) >= 2:
            return (
                args[1].casefold() == "reset"
                and len(args) <= 4
                and all(
                    not value.startswith("-") or value == "--yes"
                    for value in args[2:]
                )
            )
        if action in {
            "list",
            "show",
            "doctor",
            "logs",
            "tasks",
            "approve",
            "trace",
            "start",
            "restart",
            "stop",
        }:
            return len(args) <= 2
        return True
    if command in _SUBCOMMAND_COMPLETIONS:
        allowed = {
            name.casefold()
            for name, _description in _SUBCOMMAND_COMPLETIONS[command]
        }
        return not args or args[0].casefold() in allowed
    return True


def _is_allowed_natural_plan_command(parts: list[str]) -> bool:
    """Validate the strict command subset exposed to repository interpreters."""

    if not parts:
        return False
    command = parts[0].casefold()
    args = parts[1:]
    lowered = [value.casefold() for value in args]
    declared = command_spec(parts)
    if declared is None or not declared.natural:
        return False
    if command == "help":
        return not args
    if command in {"current", "resume", "runs"}:
        return not args
    if command == "project":
        return (
            not args
            or len(args) == 1
            and lowered[0] == "reset"
            or 1 <= len(args) <= 2
            and lowered[0] in {"init", "rename"}
            or len(args) == 2
            and lowered == ["reset", "fresh"]
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
            "diff",
            "validate",
            "history",
            "path",
        }:
            return len(args) <= 2
        if action in {"edit-spec", "edit-refinement"}:
            return True
        if action == "refine-spec":
            return len(args) <= 2
        if action == "implement":
            return all(
                value
                in {
                    "implement",
                    "codex",
                    "claude",
                    "--rerun",
                    "--interactive",
                }
                for value in lowered
            )
        if action == "show":
            if len(args) == 1:
                return True
            if len(args) == 2:
                return lowered[1] in {
                    "spec",
                            "refinement",
                    "source",
                    *(view.command for view in WORKFLOW_VIEWS),
                }
            if len(args) == 3:
                if lowered[1] == "source":
                    return True
                one_view = workflow_view_spec(lowered[1])
                return (
                    one_view is not None
                    and one_view.participants == "one"
                )
            view = workflow_view_spec(lowered[1])
            return (
                len(args) >= 3
                and view is not None
                and view.participants == "many"
            )
        return False
    if command == "studio":
        return len(args) == 1 and lowered[0] in {"doctor", "restart"}
    if command == "model":
        if not args:
            return True
        if lowered[0] == "provider":
            if len(args) == 1:
                return True
            if lowered[1] in {"list", "check"}:
                return len(args) <= 3
            if lowered[1] == "configure":
                return 3 <= len(args) <= 4
            return lowered[1] == "remove" and len(args) == 3
        if lowered[0] == "config":
            if len(args) == 1:
                return True
            if lowered[1] in {"list", "show", "check", "create"}:
                return len(args) <= 3
            if lowered[1] in {"edit", "remove"}:
                return len(args) == 3
            return lowered[1] == "rename" and len(args) == 4
        if lowered[0] == "assignments":
            return len(args) == 1 or (
                len(args) == 2 and lowered[1] == "check"
            )
        if lowered[0] == "setup":
            return len(args) == 1
        if lowered[0] in {"default", "inherit"}:
            return len(args) == 2
        return lowered[0] == "assign" and len(args) == 3
    if command == "connector":
        if not args:
            return True
        if lowered[0] == "provider":
            return len(args) <= 3
        if lowered[0] == "config":
            return len(args) <= 4
        if lowered[0] == "setup":
            return len(args) == 1
        if lowered[0] == "assignments":
            return len(args) in {1, 2} and (
                len(args) == 1 or lowered[1] == "check"
            )
        if lowered[0] in {"assign", "bind"}:
            return len(args) == 3
        return lowered[0] == "inherit" and len(args) == 2
    if command == "deploy":
        if not args:
            return True
        if lowered[0] == "inspect":
            return len(args) <= 3
        if lowered[0] == "storage":
            if len(args) <= 2 and (
                len(args) == 1 or lowered[1] != "compact"
            ):
                return True
            return (
                len(args) >= 2
                and lowered[1] == "compact"
                and _valid_storage_compact_syntax(args[2:])
            )
        if lowered[0] == "remove":
            return (
                (
                    len(args) == 2
                    and not args[1].startswith("-")
                )
                or (
                    len(args) == 3
                    and not args[1].startswith("-")
                    and lowered[2] == "--purge"
                )
            )
        if lowered[:2] == ["logs", "reset"]:
            return (
                len(args) == 2
                or len(args) == 3
                and not args[2].startswith("-")
            )
        if (
            lowered[0]
            in {
                "list",
                "show",
                "doctor",
                "logs",
                "tasks",
                "approve",
                "trace",
                "storage",
                "start",
                "restart",
                "stop",
            }
        ):
            if lowered[0] == "list":
                return len(args) == 1
            return len(args) <= 2
        return (
            len(args) == 1
            and (args[0] == "--no-start" or not args[0].startswith("-"))
            or len(args) == 2
            and args[1] == "--no-start"
            and not args[0].startswith("-")
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
        if lowered[0] == "inspect":
            return len(args) <= 2
        if lowered[0] in {"tasks", "trace"}:
            return len(args) == 1
        if lowered[0] == "approve":
            return 1 <= len(args) <= 3
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


class Studio(StudioModelsMixin, StudioConnectorsMixin, StudioStorageMixin):
    def __init__(
        self,
        workspace: Workspace,
        *,
        input_func: InputFunc = input,
        output_func: OutputFunc = print,
        secret_input_func: SecretInputFunc | None = None,
        color: bool | None = None,
        command_mode: bool = False,
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
        self._interactive_offers_enabled = (
            self._prompt_toolkit_enabled and not command_mode
        )
        self._renderer = TerminalRenderer(
            output_func,
            color=color,
            columns=lambda: self._output_columns(),
        )
        self.color = self._renderer.color
        if secret_input_func is None:
            import getpass

            secret_input_func = getpass.getpass
        self.secret_input = secret_input_func
        archived_acceptance = self.workspace.retire_acceptance_state()
        if archived_acceptance is not None:
            self._info(
                "Archived retired workflow source snapshots at "
                f"{archived_acceptance}."
            )

    def _emit(self, value: object = "") -> None:
        self._renderer.emit(value)

    def _status(self, kind: StatusKind, message: str, *, indent: int = 0) -> None:
        """Emit one consistent, terminal-safe human status line."""

        self._renderer.status(kind, message, indent=indent)

    def _status_mark(self, kind: StatusKind) -> str:
        return self._renderer.status_mark(kind)

    def _emit_section_title(self, title: str, *, major: bool = True) -> None:
        """Render the shared boundary between a section heading and its data."""

        self._renderer.section(title, major=major)

    @staticmethod
    def _visible_width(value: str) -> int:
        return TerminalRenderer.visible_width(value)

    def _output_columns(self) -> int:
        if self.output is print and bool(
            getattr(sys.stdout, "isatty", lambda: False)()
        ):
            return shutil.get_terminal_size(fallback=(100, 24)).columns
        return 100

    @staticmethod
    def _wrapped_lines(value: object, width: int) -> list[str]:
        return TerminalRenderer.wrapped_lines(value, width)

    def _emit_wrapped_field(self, label: str, value: object) -> None:
        self._renderer.wrapped_field(label, value)

    def _pad_cell(self, value: object, width: int, *, right: bool = False) -> str:
        return self._renderer.pad_cell(value, width, right=right)

    def _emit_columns(
        self,
        title: str,
        headers: tuple[str, ...],
        rows: list[tuple[object, ...]],
        *,
        right_aligned: frozenset[int] = frozenset(),
    ) -> None:
        """Render a real column table whose header is distinct from its rows."""

        self._renderer.columns(
            title,
            headers,
            rows,
            right_aligned=right_aligned,
        )

    def _emit_next(self, value: object) -> None:
        self._renderer.next(value)

    def _emit_table(
        self,
        title: str,
        rows: list[tuple[str, object, StatusKind | None]],
    ) -> None:
        """Render key/value data with explicit headings and separate guidance."""

        self._renderer.table(title, rows)

    def _success(self, message: str, *, indent: int = 0) -> None:
        self._status("success", message, indent=indent)

    def _warning(self, message: str, *, indent: int = 0) -> None:
        self._status("warning", message, indent=indent)

    def _error(self, message: str, *, indent: int = 0) -> None:
        self._status("error", message, indent=indent)

    def _info(self, message: str, *, indent: int = 0) -> None:
        self._status("info", message, indent=indent)

    def _emit_studio_banner(
        self,
        command: str | None = None,
        *,
        leading_blank: bool = False,
    ) -> None:
        """Render the common Studio identity banner."""

        if leading_blank:
            self._emit()
        settings = self.workspace.global_settings()
        suffix = f" · {command}" if command else ""
        if settings.get("output_style") == "compact":
            label = f" ZipperGen Studio{suffix} "
            self._emit(f"──{label}{'─' * max(2, 58 - len(label))}")
            return
        content = f" ZipperGen Studio{suffix} "
        width = max(58, len(content))
        self._emit(f"╭{'─' * width}╮")
        self._emit(f"│{content:<{width}}│")
        self._emit(f"╰{'─' * width}╯")

    def _emit_output_boundary(self, command: str) -> None:
        """Separate one command's interaction from its echoed input line."""

        self._emit_studio_banner(command, leading_blank=True)

    @staticmethod
    def _output_boundary_label(parts: list[str]) -> str:
        """Name a command precisely without echoing user values or secrets."""

        command = parts[0].casefold()
        if (
            command == "run"
            and len(parts) > 1
            and parts[1].casefold() in {"inspect", "tasks", "approve", "trace"}
        ):
            return f"run {parts[1].casefold()}"
        if (
            command == "deploy"
            and len(parts) > 1
            and parts[1].casefold()
            in {
                "list",
                "show",
                "inspect",
                "doctor",
                "logs",
                "tasks",
                "approve",
                "trace",
                "storage",
                "start",
                "restart",
                "stop",
                "remove",
            }
        ):
            if (
                parts[1].casefold() == "storage"
                and len(parts) > 2
                and parts[2].casefold() == "compact"
            ):
                return "deploy storage compact"
            return f"deploy {parts[1].casefold()}"
        if len(parts) > 1 and command in {
            "workflow",
            "model",
            "connector",
            "project",
            "settings",
            "language",
            "editor",
            "edit",
            "studio",
        }:
            return f"{command} {parts[1].casefold()}"
        return command

    def _prompt(self) -> str:
        entry = self.workspace.workflow_entry
        label = entry.rsplit(":", 1)[-1] if entry else "no workflow"
        return f"zippergen [{label}]> "

    def welcome(self) -> None:
        self._migrate_workflow_entry()
        self._migrate_project_configuration()
        self._migrate_portable_implementation_lock()
        self._report_legacy_refinement()
        self._emit_studio_banner()
        manifest = self.workspace.project_manifest()
        entry = self.workspace.workflow_entry
        assistant, assistant_kind = self._coding_assistant_readiness()
        rows: list[tuple[str, object, StatusKind | None]] = [
            ("Project", manifest["name"], None),
            ("Root", self.workspace.root, None),
            (
                "Workflow",
                entry if entry else "not configured",
                "success" if entry else "warning",
            ),
            ("Assistant", assistant, assistant_kind),
        ]
        if not manifest["exists"]:
            rows.append(
                (
                    "Start",
                    "Type a command or describe what you want in ordinary "
                    "language. You can press Tab to complete. Use 'help' for "
                    "the short path.",
                    "info",
                )
            )
        self._emit_table(
            "Session context",
            rows,
        )
        self._emit_next(self._welcome_next_action())

    def _migrate_portable_implementation_lock(self) -> None:
        """Convert a trustworthy private Pass-1 record into project metadata."""

        if (
            self.workspace.implementation_lock_path.exists()
            or self.workspace.workflow_entry is None
        ):
            return
        record = self._implementation_record(self.workspace.workflow_entry)
        if record is None:
            return
        fingerprint = record.get("result_specification_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            return
        raw_accepted_files = record.get("accepted_source_files")
        if not isinstance(raw_accepted_files, list) or not raw_accepted_files:
            return
        try:
            files = [path for path, _role in self._workflow_file_records()]
            accepted_files = {
                str(item["path"]): str(item["sha256"])
                for item in raw_accepted_files
                if isinstance(item, dict)
                and isinstance(item.get("path"), str)
                and isinstance(item.get("sha256"), str)
            }
            if set(accepted_files) != set(files):
                return
            for relative in files:
                candidate = (self.workspace.root / relative).resolve()
                if (
                    not candidate.is_file()
                    or not candidate.is_relative_to(self.workspace.root)
                    or hashlib.sha256(candidate.read_bytes()).hexdigest()
                    != accepted_files[relative]
                ):
                    return
            self.workspace.write_implementation_lock(
                files,
                specification_fingerprint=fingerprint,
            )
        except (WorkspaceError, OSError, ValueError, SystemExit):
            return
        self._info(
            "Recorded the former local implementation relationship in "
            "zippergen.lock; commit this file with the project."
        )

    def _report_legacy_refinement(self) -> None:
        """Leave retired user prose untouched and explain its explicit paths."""

        if self.workspace.legacy_pending_refinement_path.is_file():
            self._warning(
                "A former pending refinement was left untouched at "
                f"{self.workspace.legacy_pending_refinement_path}. Copy it with "
                "'workflow edit-refinement --file PATH' or remove it only after "
                "you decide it is no longer needed."
            )

    def _coding_assistant_readiness(self) -> tuple[str, StatusKind]:
        configured = str(self._global_settings().get("assistant") or "codex")
        if shutil.which(configured):
            label = "Codex CLI" if configured == "codex" else "Claude Code"
            return f"{label} found", "success"
        alternative = "claude" if configured == "codex" else "codex"
        if shutil.which(alternative):
            label = "Claude Code" if alternative == "claude" else "Codex CLI"
            return (
                f"{configured} not found; {label} is available; use "
                f"settings set assistant {alternative}",
                "warning",
            )
        return (
            "no Codex or Claude CLI found; specifications and deterministic "
            "inspection still work",
            "warning",
        )

    def _welcome_next_action(self) -> str:
        manifest = self.workspace.project_manifest()
        if not manifest["exists"]:
            return "project init"
        specification = self.workspace.specification()
        implementation = self.workspace.implementation_state()
        return self._workflow_next_action(
            specification_present=specification is not None,
            refinement_present=self.workspace.refinement_buffer() is not None,
            implementation=str(implementation["state"]),
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
        if self.workspace.workflow_entry is None:
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
        if command == "run" and not args:
            return [
                *_SUBCOMMAND_COMPLETIONS["run"],
                *_MODEL_COMPLETIONS,
                ("--assistant", "select the coding-assistant action backend"),
            ]
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
                        ("spec", "canonical workflow specification"),
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
            if action == "import":
                return self._path_completion_candidates(fragment)
            if action == "edit-refinement":
                if "--file" in rest and rest[-1] == "--file":
                    return self._path_completion_candidates(fragment)
                if "--editor" in rest and rest[-1] == "--editor":
                    return self._editor_completion_candidates()
                if not rest:
                    choices = [
                        ("--file", "import text from an existing file"),
                        ("--editor", "choose an editor for this invocation"),
                    ]
                    return choices
                return []
            if action == "edit-spec":
                if "--file" in rest and rest[-1] == "--file":
                    return self._path_completion_candidates(fragment)
                if "--editor" in rest and rest[-1] == "--editor":
                    return self._editor_completion_candidates()
                return [
                    ("--file", "read the specification from a file"),
                    ("--editor", "choose an editor for this invocation"),
                ]
            if action == "refine-spec":
                return [
                    ("codex", "rewrite the specification with Codex"),
                    ("claude", "rewrite the specification with Claude Code"),
                ]
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
                if rest[0].lower() == "claude":
                    values = []
                    if "--rerun" not in rest:
                        values.append(
                            ("--rerun", "deliberately rerun reviewed work")
                        )
                    return values
                return []
            if action == "status":
                return [
                    (
                        "--details",
                        "include task IDs, refresh links, and internal record path",
                    )
                ]
            return []
        if command == "model":
            if not args:
                return list(_SUBCOMMAND_COMPLETIONS["model"])
            action = args[0].lower()
            if action == "provider":
                provider_actions = [
                    ("list", "list provider connection status"),
                    ("configure", "configure a key or local endpoint"),
                    ("check", "test provider connectivity"),
                    ("remove", "remove a provider connection"),
                ]
                if len(args) == 1:
                    return provider_actions
                if (
                    args[1].lower() in {"configure", "check", "remove"}
                    and len(args) == 2
                ):
                    values = [
                        (name, "model provider")
                        for name in _SUPPORTED_PROVIDERS
                        if args[1].lower() != "remove" or name != "mock"
                    ]
                    if args[1].lower() == "check":
                        values.insert(0, ("all", "all configured providers"))
                    return values
                return []
            if action == "config":
                config_actions = [
                    ("list", "list reusable configurations"),
                    (
                        "create",
                        "create a provider/model configuration with optional "
                        "local idle release",
                    ),
                    ("show", "inspect a saved configuration"),
                    ("check", "verify exact model availability"),
                    ("edit", "change provider, model, or local idle release"),
                    ("rename", "rename and migrate assignments"),
                    ("remove", "remove an unused configuration"),
                ]
                if len(args) == 1:
                    return config_actions
                subaction = args[1].lower()
                if subaction in {"show", "check"} and len(args) == 2:
                    return [
                        ("all", "all saved model configurations"),
                        *self._completion_model_configurations(),
                    ]
                if (
                    subaction in {"edit", "rename", "remove"}
                    and len(args) == 2
                ):
                    return [
                        candidate
                        for candidate in self._completion_model_configurations()
                        if candidate[0] != "mock"
                    ]
                return []
            if action == "assignments" and len(args) == 1:
                return [
                    (
                        "check",
                        "check models used by the project workflow",
                    )
                ]
            if action == "default":
                return self._completion_model_configurations()
            if action == "assign":
                if len(args) == 1:
                    participants = [
                        (name, "LLM-active participant")
                        for name in self._completion_lifelines(llm_only=True)
                    ]
                    try:
                        _current, workflow, module = self._current_context()
                        actions = [
                            (target, "LLM action override")
                            for target in self._llm_action_targets(
                                workflow, module
                            )
                        ]
                    except SystemExit:
                        actions = []
                    return [*participants, *actions]
                return self._completion_model_configurations()
            if action == "inherit" and len(args) == 1:
                participants = [
                    (name, "LLM-active participant")
                    for name in self._completion_lifelines(llm_only=True)
                ]
                try:
                    _current, workflow, module = self._current_context()
                    actions = [
                        (target, "LLM action override")
                        for target in self._llm_action_targets(
                            workflow, module
                        )
                    ]
                except SystemExit:
                    actions = []
                return [*participants, *actions]
            return []
        if command == "connector":
            if not args:
                return list(_SUBCOMMAND_COMPLETIONS["connector"])
            action = args[0].lower()
            if action == "provider":
                if len(args) == 1:
                    return [
                        ("list", "list provider status"),
                        ("configure", "configure private credentials"),
                        ("check", "test provider connectivity"),
                        ("remove", "remove an unused provider"),
                    ]
                if len(args) == 2 and args[1].lower() in {
                    "configure", "check", "remove"
                }:
                    values = [
                        ("telegram", "Telegram Bot API"),
                        ("google", "Google Workspace OAuth"),
                    ]
                    if args[1].lower() == "check":
                        values.insert(0, ("all", "all connector providers"))
                    return values
                return []
            if action == "config":
                if len(args) == 1:
                    return [
                        ("list", "list reusable configurations"),
                        ("create", "create a reusable resource configuration"),
                        ("show", "inspect one configuration"),
                        ("edit", "change a configuration"),
                        ("check", "check its destination"),
                        ("rename", "rename and migrate assignments"),
                        ("remove", "remove an unused configuration"),
                    ]
                if len(args) == 2 and args[1].lower() in {
                    "show", "edit", "check", "rename", "remove"
                }:
                    values = [
                        (name, value.get("provider", "connector"))
                        for name, value
                        in self.workspace.connector_configurations().items()
                    ]
                    if args[1].lower() == "check":
                        values.insert(0, ("all", "all configurations"))
                    return values
                return []
            if action in {"assign", "inherit"} and len(args) == 1:
                try:
                    _current, workflow, module = self._current_context()
                    participants = [
                        (name, "human-active participant")
                        for name in self._human_action_lifelines(
                            workflow, module
                        )
                    ]
                    actions = [
                        (name, "human action override")
                        for name in self._human_action_targets(
                            workflow, module
                        )
                    ]
                    return [*participants, *actions]
                except SystemExit:
                    return []
            if action == "assignments" and len(args) == 1:
                return [
                    ("check", "check effective connector routes")
                ]
            if action == "assign" and len(args) == 2:
                return [
                    (name, value.get("provider", "connector"))
                    for name, value
                    in self.workspace.connector_configurations().items()
                ]
            return []
        if command == "deploy":
            if not args:
                return [
                    *_SUBCOMMAND_COMPLETIONS["deploy"],
                    *self._deployment_completion_candidates(),
                    ("--no-start", "prepare without starting"),
                ]
            if (
                args[0].lower()
                in {
                    "show",
                    "inspect",
                    "doctor",
                    "logs",
                    "tasks",
                    "approve",
                    "trace",
                    "storage",
                    "start",
                    "restart",
                    "stop",
                }
                and len(args) == 1
            ):
                values = self._deployment_completion_candidates()
                if args[0].lower() == "logs":
                    values.insert(
                        0,
                        ("reset", "archive and reset visible log history"),
                    )
                elif args[0].lower() == "inspect":
                    values.insert(
                        0,
                        ("--watch", "refresh participant positions every second"),
                    )
                elif args[0].lower() == "storage":
                    values.insert(
                        0,
                        ("compact", "show or apply a safe compaction plan"),
                    )
                return values
            if (
                len(args) >= 2
                and args[0].lower() == "logs"
                and args[1].lower() == "reset"
            ):
                if len(args) == 2:
                    return self._deployment_completion_candidates()
                if len(args) == 3:
                    return [("--yes", "confirm log-history reset")]
                return []
            if args[0].lower() == "inspect":
                if "--watch" in {value.lower() for value in args[1:]}:
                    return []
                positionals = [
                    value for value in args[1:] if value.lower() != "--watch"
                ]
                if len(positionals) <= 1:
                    return [
                        ("--watch", "refresh participant positions every second"),
                        *[
                            (name, "workflow participant")
                            for name in self._completion_lifelines()
                        ],
                    ]
                if len(positionals) == 2:
                    return [
                        ("--watch", "refresh participant positions every second")
                    ]
                return []
            if args[0].lower() == "storage":
                if len(args) == 2 and args[1].lower() == "compact":
                    return [
                        *self._deployment_completion_candidates(),
                        ("--yes", "confirm safe compaction"),
                    ]
                if "compact" in {value.lower() for value in args[1:]}:
                    return [("--yes", "confirm safe compaction")]
                return []
            if args[0].lower() == "remove":
                if len(args) == 1:
                    return self._deployment_completion_candidates()
                if len(args) == 2:
                    return [
                        ("--purge", "permanently delete instead of archiving"),
                        ("--yes", "confirm recoverable removal"),
                    ]
                if len(args) == 3 and args[2].lower() == "--purge":
                    return [("--yes", "confirm permanent deletion")]
                return []
            return []
        if command == "run":
            action = args[0].lower()
            if action == "inspect":
                if "--watch" in {value.lower() for value in args[1:]}:
                    return []
                if len(args) > 1:
                    return [
                        ("--watch", "refresh participant positions every second")
                    ]
                return [
                    ("--watch", "refresh participant positions every second"),
                    *[
                        (name, "workflow participant")
                        for name in self._completion_lifelines()
                    ],
                ]
            if action == "approve":
                if len(args) == 1:
                    return self._run_task_completion_candidates()
                if len(args) == 2:
                    return [
                        ("yes", "approve a boolean human decision"),
                        ("no", "reject a boolean human decision"),
                    ]
                return []
            if action in {"tasks", "trace"}:
                return []
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
        return []

    def _deployment_completion_candidates(self) -> list[tuple[str, str]]:
        from zippergen.serve import _deployments_dir

        values: list[tuple[str, str]] = []
        directory = _deployments_dir()
        if directory.exists():
            for path in sorted(directory.glob("*.json")):
                if path.name.endswith(".secrets.json"):
                    continue
                try:
                    profile = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(profile, dict) and profile.get("name"):
                    values.append((str(profile["name"]), "deployment"))
        return values

    def _run_task_completion_candidates(self) -> list[tuple[str, str]]:
        try:
            record = self._run_store_record()
        except SystemExit:
            return []
        if not record.exists:
            return []
        from zippergen.serve import _store_status

        tasks = _store_status(str(record.path)).get("pending_human_tasks")
        if not isinstance(tasks, list):
            return []
        return [
            (
                str(task.get("task_id")),
                f"{task.get('role')}.{task.get('action')}",
            )
            for task in tasks
            if isinstance(task, dict) and task.get("task_id")
        ]

    def _show_workflow_dashboard(self) -> None:
        specification = self.workspace.specification()
        refinement = self.workspace.refinement_buffer() is not None
        implementation = self.workspace.implementation_state()
        state = str(implementation["state"])
        state_kind: StatusKind = "success" if state == "current" else "warning"
        next_action = self._workflow_next_action(
            specification_present=specification is not None,
            refinement_present=refinement,
            implementation=state,
        )
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
                    "ready to apply" if refinement else "none",
                    "warning" if refinement else None,
                ),
                (
                    "Workflow",
                    self.workspace.workflow_entry or "not configured",
                    None if self.workspace.workflow_entry else "warning",
                ),
                ("Implementation", state, state_kind),
                ("Reason", implementation.get("reason") or "—", None),
                ("Next", next_action, None),
            ],
        )

    @staticmethod
    def _workflow_next_action(
        *,
        specification_present: bool,
        refinement_present: bool,
        implementation: str,
    ) -> str:
        """Return the first matching row of the normative Next table."""

        if not specification_present:
            return "workflow edit-spec"
        if refinement_present:
            return "workflow refine-spec"
        if implementation == "external":
            return "workflow edit-spec · workflow implement"
        if implementation in {"absent", "stale"}:
            return "workflow implement"
        return "run · deploy"

    def _specification_diff(self) -> tuple[str, str]:
        current = self.workspace.specification()
        if current is None:
            return "none", "# No canonical specification is available."
        baseline: str | None = None
        for record in self.workspace.list_requests():
            if str(record.get("status") or "") in {
                "implemented",
                "awaiting_review",
                "reconciled",
                "closed",
            } and isinstance(record.get("prompt"), str):
                baseline = str(record["prompt"])
                break
        baseline_name = "last implemented specification"
        if baseline is None:
            return (
                "initial",
                "# No earlier specification baseline exists for this creation.",
            )
        lines = list(
            difflib.unified_diff(
                baseline.splitlines(),
                current.splitlines(),
                fromfile=baseline_name,
                tofile="current specification.md",
                lineterm="",
            )
        )
        return (
            baseline_name,
            "\n".join(lines) if lines else "# No specification changes.",
        )

    def _semantic_diff(self) -> tuple[str, str]:
        _current, workflow, module = self._current_context(
            purpose="compare its semantics"
        )
        baseline_file: object = None
        for record in self.workspace.list_requests():
            if record.get("baseline_file"):
                baseline_file = record.get("baseline_file")
                break
        baseline_snapshot: dict[str, object] | None = None
        baseline_name = "pre-refinement semantics"
        if baseline_file:
            try:
                raw = json.loads(
                    Path(str(baseline_file)).read_text(encoding="utf-8")
                )
                baseline_snapshot = {
                    "schema": raw.get("schema"),
                    "workflow": read_semantic_snapshot(raw),
                }
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                ValueError,
            ):
                baseline_snapshot = None
        if baseline_snapshot is None:
            return (
                "initial",
                "# No earlier semantic baseline exists for this creation.",
            )
        before = read_semantic_snapshot(baseline_snapshot)
        after = read_semantic_snapshot(semantic_snapshot(workflow, module))
        return (
            baseline_name,
            render_semantic_diff(semantic_diff_models(before, after)),
        )

    def show_workflow_changes(self) -> None:
        specification_baseline, specification_diff = self._specification_diff()
        semantic_baseline, semantic_diff = self._semantic_diff()
        self._emit_table(
            "Change comparison",
            [
                ("Intent baseline", specification_baseline, None),
                ("Behavior baseline", semantic_baseline, None),
                (
                    "Meaning",
                    "comparison only; validation is a separate technical check",
                    "info",
                ),
            ],
        )
        self._emit_section_title("Specification diff")
        self._emit(specification_diff)
        self._emit()
        self._emit_section_title("Semantic workflow diff")
        self._emit(semantic_diff)
        self._emit()

    def manage_workflow(
        self,
        args: list[str],
    ) -> None:
        """Present specification, implementation, and inspection as one lifecycle."""

        if not args:
            self._show_workflow_dashboard()
            return
        action, *rest = args
        action = action.casefold()
        if action == "edit-spec":
            self.manage_spec(["edit-spec", *rest])
            return
        if action == "edit-refinement":
            self.manage_spec(["edit-refinement", *rest])
            return
        if action == "refine-spec":
            self.refine_specification(rest)
            return
        if action == "list":
            if rest:
                raise SystemExit("Use workflow list.")
            self.list_workflows()
            return
        if action == "import":
            self.import_workflow(rest)
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
            self.show_workflow(rest)
            return
        if action == "diff":
            if rest:
                raise SystemExit("Use workflow diff.")
            self.show_workflow_changes()
            return
        if action == "status":
            if rest not in ([], ["--details"]):
                raise SystemExit("Use workflow status [--details].")
            self.manage_task(["details"] if rest else [])
            return
        if action == "implement":
            self.run_assistant(rest)
            return
        if action == "validate":
            if rest:
                raise SystemExit("Use workflow validate.")
            self.validate()
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
        raise SystemExit(
            "Use workflow edit-spec, edit-refinement, refine-spec, list, import, "
            "files, show, diff, status, implement, validate, history, or path."
        )

    def studio_doctor(self) -> None:
        """Report local development readiness without contacting providers."""

        manifest = self.workspace.project_manifest()
        try:
            editor, editor_source = self._effective_editor()
            editor_status = f"{shlex.join(editor)} — {editor_source}"
            editor_kind: StatusKind = "success"
        except SystemExit as exc:
            editor_status = str(exc)
            editor_kind = "error"
        assistant, assistant_kind = self._coding_assistant_readiness()
        interpreter = self._language_backend(
            str(self._global_settings().get("interpreter") or "auto"),
            required=False,
        )
        self._emit_table(
            "Studio readiness",
            [
                (
                    "Project",
                    (
                        f"manifest present — {self.workspace.manifest_path.name}"
                        if manifest["exists"]
                        else "manifest not created; use project init"
                    ),
                    "success" if manifest["exists"] else "warning",
                ),
                (
                    "Editor",
                    editor_status,
                    editor_kind,
                ),
                ("Assistant", assistant, assistant_kind),
                (
                    "Interpreter",
                    (
                        (
                            "Codex CLI"
                            if interpreter[0] == "codex"
                            else "Claude Code"
                        )
                        if interpreter
                        else "deterministic commands only; no CLI fallback found"
                    ),
                    "success" if interpreter else "warning",
                ),
                (
                    "Model setup",
                    "optional until a non-mock run; inspect with model",
                    "success",
                ),
                ("Next", self._welcome_next_action(), None),
            ],
        )

    def _studio_restart_command(self) -> tuple[str, list[str]]:
        argv = list(sys.argv)
        if not argv or not argv[0].strip():
            raise SystemExit(
                "Studio cannot determine its original launcher. Exit and run "
                "'zippergen' again from the project root."
            )
        launcher = argv[0]
        resolved: str | None = None
        if "/" not in launcher and os.sep not in launcher:
            resolved = shutil.which(launcher)
        if resolved is not None:
            return resolved, [resolved, *argv[1:]]

        candidate = Path(launcher).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            executable = str(candidate)
            return executable, [executable, *argv[1:]]
        if candidate.is_file() and candidate.suffix == ".py":
            return sys.executable, [sys.executable, str(candidate), *argv[1:]]
        raise SystemExit(
            f"Studio cannot restart because its launcher is unavailable: "
            f"{launcher!r}. Exit and run 'zippergen' again."
        )

    def restart_studio(self) -> None:
        """Replace this process so updated installed source is imported cleanly."""

        executable, arguments = self._studio_restart_command()
        self._emit_table(
            "Studio restart",
            [
                ("Project", self.workspace.root, None),
                ("Launcher", executable, None),
                ("State", "saved project context will be reloaded", "success"),
            ],
        )
        self._success("Restarting ZipperGen Studio.")
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.execv(executable, arguments)
        except OSError as exc:
            raise SystemExit(
                f"Studio restart failed: {exc}. The current Studio process "
                "is still running."
            ) from exc

    @staticmethod
    def _studio_source_checkout() -> Path:
        """Locate the Git checkout that supplies this imported Studio module."""

        source = Path(__file__).resolve()
        for candidate in source.parents:
            checkout_source = candidate / "src" / "zippergen" / "studio.py"
            try:
                same_source = checkout_source.samefile(source)
            except OSError:
                same_source = False
            if (
                same_source
                and (candidate / ".git").exists()
                and (candidate / "pyproject.toml").is_file()
            ):
                return candidate
        raise SystemExit(
            "Studio update is available only when ZipperGen is running from "
            "its Git source checkout. Update a packaged installation with the "
            "package manager that installed it."
        )

    @staticmethod
    def _update_subprocess(
        arguments: list[str],
        *,
        operation: str,
    ) -> subprocess.CompletedProcess[str]:
        try:
            completed = subprocess.run(
                arguments,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            raise SystemExit(f"{operation} could not start: {exc}.") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise SystemExit(
                f"{operation} failed"
                + (f": {detail}" if detail else ".")
            )
        return completed

    def update_studio(self) -> None:
        """Safely fast-forward an editable source checkout and restart Studio."""

        checkout = self._studio_source_checkout()
        git = shutil.which("git")
        if git is None:
            raise SystemExit(
                "Studio update needs Git, but 'git' was not found. Install Git "
                "or update the checkout in another terminal."
            )

        def git_output(*arguments: str, operation: str) -> str:
            completed = self._update_subprocess(
                [git, "-C", str(checkout), *arguments],
                operation=operation,
            )
            return completed.stdout.strip()

        changes = git_output(
            "status",
            "--porcelain",
            "--untracked-files=no",
            operation="Checking the ZipperGen working tree",
        )
        if changes:
            preview = ", ".join(
                line[3:].strip() or line.strip()
                for line in changes.splitlines()[:5]
            )
            raise SystemExit(
                "Studio update stopped because the ZipperGen checkout has "
                f"tracked local changes: {preview}. Commit, restore, or move "
                "those changes before updating. Project files and deployments "
                "were not touched."
            )

        branch = git_output(
            "rev-parse",
            "--abbrev-ref",
            "HEAD",
            operation="Reading the ZipperGen branch",
        )
        if branch == "HEAD":
            raise SystemExit(
                "Studio update stopped because the ZipperGen checkout is in "
                "detached-HEAD state. Check out a branch before updating."
            )
        upstream = git_output(
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
            operation="Finding the ZipperGen upstream branch",
        )
        before = git_output(
            "rev-parse",
            "HEAD",
            operation="Reading the current ZipperGen revision",
        )
        self._emit_table(
            "Studio update",
            [
                ("Checkout", checkout, None),
                ("Branch", f"{branch} → {upstream}", None),
                ("Current", before[:12], None),
                ("Project", f"preserved — {self.workspace.root}", "success"),
                (
                    "Deployments",
                    "unchanged; installed bundles remain immutable",
                    "success",
                ),
            ],
        )
        self._info("Pulling a fast-forward update from the configured upstream.")
        pull = self._update_subprocess(
            [git, "-C", str(checkout), "pull", "--ff-only"],
            operation="Updating the ZipperGen checkout",
        )
        after = git_output(
            "rev-parse",
            "HEAD",
            operation="Reading the updated ZipperGen revision",
        )
        metadata_changed = False
        if before != after:
            changed_metadata = git_output(
                "diff",
                "--name-only",
                before,
                after,
                "--",
                "pyproject.toml",
                "uv.lock",
                operation="Checking updated dependency metadata",
            )
            metadata_changed = bool(changed_metadata)

        environment_status = "unchanged; synchronization not needed"
        if metadata_changed:
            uv = shutil.which("uv")
            if uv is None:
                raise SystemExit(
                    "ZipperGen source was updated, but dependency metadata also "
                    "changed and 'uv' was not found. Run 'uv sync --project "
                    f"{checkout}' and then restart Studio."
                )
            sync_arguments = [uv, "sync", "--project", str(checkout)]
            if (checkout / "uv.lock").is_file():
                sync_arguments.insert(2, "--locked")
            self._info("Synchronizing the updated ZipperGen environment.")
            self._update_subprocess(
                sync_arguments,
                operation="Synchronizing the ZipperGen environment",
            )
            environment_status = "synchronized with uv"

        pull_detail = (pull.stdout or pull.stderr).strip().splitlines()
        self._emit_table(
            "Update complete",
            [
                ("Previous", before[:12], None),
                (
                    "Current",
                    after[:12],
                    "success",
                ),
                (
                    "Source",
                    (
                        "already up to date"
                        if before == after
                        else "fast-forwarded successfully"
                    ),
                    "success",
                ),
                ("Environment", environment_status, "success"),
                (
                    "Git",
                    pull_detail[-1] if pull_detail else "completed",
                    None,
                ),
            ],
        )
        self.restart_studio()

    def _is_explicit_command(self, parts: list[str]) -> bool:
        """Resolve ambiguous short verbs against known project objects."""

        if not _is_explicit_studio_syntax(parts):
            return False
        command = parts[0].casefold()
        args = parts[1:]
        if command == "run" and len(args) == 1:
            entered = args[0]
            if entered.casefold() in {"inspect", "tasks", "approve", "trace"}:
                return True
            configurations = {
                name.casefold()
                for name in self.workspace.model_configurations()
            }
            if entered.casefold() in configurations:
                return True
            try:
                _validate_model_spec(entered)
            except SystemExit:
                return False
            return True
        return True

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
            if _allow_natural and not self._is_explicit_command(rough_parts):
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
                self._emit_output_boundary("model")
            raise SystemExit(
                "`providers` is not a Studio command. Provider connections are "
                "managed with `model provider configure NAME`; use `model` "
                "to inspect them."
            )
        if parts[0].casefold() == "models":
            if show_boundary:
                self._emit_output_boundary("model")
            raise SystemExit(
                "`models` was renamed to `model`. Use `model`, "
                "`model setup`, or another `model ...` command."
            )
        if parts[0].casefold() == "deployment":
            if show_boundary:
                self._emit_output_boundary("deploy")
            raise SystemExit(
                "`deployment` was replaced by the single `deploy` namespace. "
                "Use `deploy list`, `deploy show`, or another `deploy ...` "
                "command."
            )
        legacy_deploy_commands = {
            "status": "show",
            "doctor": "doctor",
            "logs": "logs",
            "start": "start",
            "restart": "restart",
            "stop": "stop",
        }
        legacy_command = parts[0].casefold()
        legacy_target_known = False
        if legacy_command in legacy_deploy_commands and len(parts) == 2:
            remembered = self.workspace.load().get("last_deployment")
            legacy_target_known = bool(
                remembered
                and str(remembered).casefold() == parts[1].casefold()
            )
            if not legacy_target_known:
                from zippergen.serve import _deployment_profile_path

                legacy_target_known = _deployment_profile_path(parts[1]).exists()
        if (
            legacy_command in legacy_deploy_commands
            and (len(parts) == 1 or legacy_target_known)
        ):
            replacement = legacy_deploy_commands[legacy_command]
            if show_boundary:
                self._emit_output_boundary(f"deploy {replacement}")
            raise SystemExit(
                f"`{legacy_command}` is no longer a Studio command. "
                f"Use `deploy {replacement} [NAME]`."
            )
        if parts[0].casefold() == "store":
            if show_boundary:
                self._emit_output_boundary("run")
            raise SystemExit(
                "`store` is not a Studio command. Durable state belongs to "
                "a development run or deployment. Use run inspect, run tasks, "
                "run approve, run trace, or the corresponding deploy "
                "command."
            )
        if (
            len(parts) >= 2
            and parts[0].casefold() == "workflow"
            and parts[1].casefold() == "prompts"
        ):
            if show_boundary:
                self._emit_output_boundary("workflow")
            raise SystemExit(
                "`workflow prompts` was retired. Studio now maintains one "
                "canonical specification and one optional scratch refinement. "
                "Use workflow show spec, workflow edit-spec, workflow "
                "edit-refinement, or workflow refine-spec."
            )
        if (
            len(parts) >= 2
            and parts[0].casefold() == "workflow"
            and parts[1].casefold() == "select"
        ):
            if show_boundary:
                self._emit_output_boundary("workflow import")
            raise SystemExit(
                "`workflow select` was removed. Use `workflow import "
                "PATH.py:WORKFLOW`; a file already inside this project is "
                "adopted without being copied."
            )
        if (
            len(parts) >= 2
            and parts[0].casefold() == "workflow"
            and parts[1].casefold() in {"create", "refine", "discard"}
        ):
            if show_boundary:
                self._emit_output_boundary("workflow")
            raise SystemExit(
                f"`workflow {parts[1]}` was removed. Use workflow edit-spec, "
                "workflow edit-refinement, or workflow refine-spec."
            )
        explicit = self._is_explicit_command(parts)
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
            if args not in ([], ["all"]):
                raise SystemExit("Use help or help all.")
            self._emit(full_help() if args == ["all"] else concise_help())
        elif command == "ask":
            if not args:
                raise SystemExit("Use ask TEXT.")
            self.interpret_natural_language(
                " ".join(args),
                _allow_requirement_proposal=False,
            )
        elif command == "plan":
            if not args:
                raise SystemExit("Use plan TEXT.")
            self.interpret_natural_language(
                " ".join(args),
                preview_only=True,
                _allow_requirement_proposal=False,
            )
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
        elif command == "studio":
            if args == ["doctor"]:
                self.studio_doctor()
            elif args == ["restart"]:
                self.restart_studio()
            elif args == ["update"]:
                self.update_studio()
            else:
                raise SystemExit(
                    "Use studio doctor, studio restart, or studio update."
                )
        elif command == "model":
            self.configure_models(args)
        elif command == "connector":
            self.manage_connectors(args)
        elif command == "run":
            if args:
                run_action = args[0].casefold()
                if run_action == "inspect":
                    self.inspect_run(args[1:])
                    return True
                if run_action in {"tasks", "approve", "trace"}:
                    self.manage_run_state(run_action, args[1:])
                    return True
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
            current, workflow, module = self._current_context(
                purpose="run it"
            )
            profile = self._run_model_profile()
            default_model = profile.get("default")
            selected_default = (
                run_args[0]
                if run_args
                else str(default_model) if default_model else None
            )
            self._check_workflow_models(
                current,
                workflow,
                module,
                default_override=run_args[0] if run_args else None,
                for_run=True,
            )
            self._check_workflow_connectors(
                current,
                workflow,
                module,
                for_run=True,
            )
            idle_timeouts = self._model_idle_timeout_routes(
                current,
                workflow,
                module,
                default_override=run_args[0] if run_args else None,
            )
            connector_snapshot, connector_environment = (
                self._workflow_connector_runtime(
                    workflow_spec=current,
                    workflow=workflow,
                    module=module,
                )
            )
            human_connector_factory = (
                self._human_connector_factory_from_snapshot(
                    connector_snapshot,
                    connector_environment,
                )
            )
            try:
                run_dev(
                    self.workspace,
                    llm=selected_default,
                    llms=normalize_llm_overrides(profile.get("lifelines")),
                    llm_idle_timeouts=idle_timeouts,
                    assistant=assistant_backend,
                    interactive=True,
                    input_func=self.input,
                    output_func=self.output,
                    renderer=self._renderer,
                    human_connector_factory=human_connector_factory,
                    connector_environment=connector_environment,
                    connector_snapshot=connector_snapshot,
                )
            except RuntimeError as exc:
                raise SystemExit(
                    f"Run failed: {exc}. The durable store was preserved; "
                    "restore the failed dependency, then use 'resume'."
                ) from exc
        elif command == "resume":
            if args:
                raise SystemExit("Studio 'resume' takes no arguments.")
            current, workflow, module = self._current_context()
            run_record = self.workspace.current_run()
            if run_record is None:
                raise SystemExit("There is no current development run to resume.")
            raw_snapshot = run_record.get("connectors")
            connector_snapshot: dict[str, object]
            if isinstance(raw_snapshot, dict):
                connector_snapshot = {
                    str(name): cast(object, dict(record))
                    for name, record in raw_snapshot.items()
                    if isinstance(record, dict)
                }
                connector_environment = (
                    self._connector_environment_from_snapshot(
                        connector_snapshot
                    )
                )
            else:
                self._warning(
                    "This run predates connector snapshots. Studio will use "
                    "the current connector assignments once. Start a new run "
                    "to make future resumes independent of project changes."
                )
                self._check_workflow_connectors(
                    current,
                    workflow,
                    module,
                    for_run=True,
                )
                connector_snapshot, connector_environment = (
                    self._workflow_connector_runtime(
                        workflow_spec=current,
                        workflow=workflow,
                        module=module,
                    )
                )
            run_dev(
                self.workspace,
                resume=True,
                interactive=True,
                input_func=self.input,
                output_func=self.output,
                renderer=self._renderer,
                human_connector_factory=(
                    self._human_connector_factory_from_snapshot(
                        connector_snapshot,
                        connector_environment,
                    )
                ),
                connector_environment=connector_environment,
            )
        elif command == "runs":
            self.show_runs()
        elif command == "deploy":
            if args and args[0].casefold() in {
                "list",
                "show",
                "inspect",
                "doctor",
                "logs",
                "tasks",
                "approve",
                "trace",
                "storage",
                "start",
                "restart",
                "stop",
                "remove",
            }:
                self.manage_deploy(args)
            else:
                self.deploy_workflow(args)
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
        if self.workspace.workflow_entry:
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
            "refinement_present": self.workspace.refinement_buffer() is not None,
            "implementation": self.workspace.implementation_state()["state"],
            "workflow_entry": self.workspace.workflow_entry,
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
        elif top == "model" and len(parts) >= 2:
            action = parts[1].casefold()
            if action in {"assign", "inherit"}:
                replace(2)
            if action == "assign" and len(parts) >= 4:
                parts[3] = configuration_names.get(
                    parts[3].casefold(), parts[3]
                )
            elif action == "default" and len(parts) >= 3:
                parts[2] = configuration_names.get(
                    parts[2].casefold(), parts[2]
                )
            elif action == "config" and len(parts) >= 4:
                subaction = parts[2].casefold()
                if subaction in {"show", "check", "edit", "rename", "remove"}:
                    parts[3] = configuration_names.get(
                        parts[3].casefold(), parts[3]
                    )
            elif action == "provider" and len(parts) >= 4:
                if parts[3].casefold() != "all":
                    parts[3] = _canonical_provider(parts[3])
        return shlex.join(parts)

    def _natural_command_risk(self, command_line: str) -> CommandRisk:
        parts = shlex.split(command_line)
        declared = command_spec(parts)
        if declared is None:
            raise SystemExit(
                f"Cannot classify unsupported Studio command: {command_line}"
            )
        lowered = [value.casefold() for value in parts]
        if (
            lowered[:2] in (["model", "provider"], ["model", "config"])
            and len(lowered) >= 3
        ):
            if lowered[2] in {"list", "show", "check"}:
                return "read-only"
            if lowered[2] == "remove":
                return "destructive"
        return declared.risk

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
            tuple(lowered[:2]) == ("workflow", "discard")
            and "--yes" not in lowered
        ):
            parts.append("--yes")
        if (
            tuple(lowered[:3]) == ("project", "reset", "fresh")
            and "--yes" not in lowered
        ):
            parts.append("--yes")
        if (
            tuple(lowered[:2]) == ("deploy", "remove")
            and "--purge" not in lowered
            and "--yes" not in lowered
        ):
            parts.append("--yes")
        if (
            tuple(lowered[:3]) == ("deploy", "logs", "reset")
            and "--yes" not in lowered
        ):
            parts.append("--yes")
        if (
            tuple(lowered[:3]) == ("deploy", "storage", "compact")
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
            self._emit_columns(
                "Command plan",
                ("Step", "Command"),
                [
                    (index, command)
                    for index, command in enumerate(plan.commands, start=1)
                ],
                right_aligned=frozenset({0}),
            )
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
        _allow_requirement_proposal: bool = True,
    ) -> None:
        request_text = request_text.strip()
        if not request_text:
            raise SystemExit("Describe the Studio operation you want.")
        if looks_sensitive(request_text):
            raise SystemExit(
                "The request appears to contain a secret value and was not sent "
                "to an interpreter or stored. Use "
                "'model provider configure NAME' so Studio can collect the "
                "key privately."
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
        if plan is None and _allow_requirement_proposal:
            plan = requirement_proposal(
                request_text,
                has_specification=self.workspace.specification() is not None,
            )
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
            requires_confirmation=plan.requires_confirmation,
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
        if plan.requires_confirmation:
            while True:
                choice = self.input(
                    "Treat this as a workflow requirement? "
                    "[y/n/command]: "
                ).strip().casefold()
                if choice in {"y", "yes"}:
                    confirmed = True
                    break
                if choice in {"c", "command"}:
                    store.record(
                        request_text,
                        plan,
                        status="redirected",
                        detail="user requested Studio-command interpretation",
                    )
                    self._info(
                        "No specification was changed; interpreting the same "
                        "text as a Studio operation."
                    )
                    self.interpret_natural_language(
                        request_text,
                        preview_only=preview_only,
                        _allow_requirement_proposal=False,
                    )
                    return
                if choice in {"n", "no", "x", "cancel", ""}:
                    self._warning(
                        "Natural-language request cancelled; no specification "
                        "or command was changed."
                    )
                    store.record(request_text, plan, status="cancelled")
                    return
                self._warning("Enter 'y', 'n', or 'command'.")
        elif risk in {"execution", "destructive"}:
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
            rows: list[tuple[object, ...]] = []
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
                rows.append(
                    (
                        record.get("id"),
                        f"{self._status_mark(kind)} {status}",
                        record.get("request"),
                        " · ".join(map(str, commands)) if commands else "—",
                    )
                )
            self._emit_columns(
                "Natural-language history",
                ("ID", "Status", "Request", "Commands"),
                rows,
            )
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
            self._emit_columns(
                "Learned interpretations",
                ("ID", "Request template", "Commands", "Source", "Uses"),
                [
                    (
                        record.get("id"),
                        record.get("request_template"),
                        " · ".join(map(str, record.get("commands") or [])),
                        record.get("source"),
                        record.get("uses", 0),
                    )
                    for record in learned
                ],
                right_aligned=frozenset({4}),
            )
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
                "guide was kept; enter 'workflow edit-spec' and write below its "
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

    def _show_specification(self) -> None:
        ensured = self.workspace.ensure_specification()
        content = ensured["content"]
        if content is None:
            self._emit_table(
                "Workflow specification",
                [
                    (
                        "Status",
                        "not written; use workflow edit-spec",
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
                    "Refinement",
                    "ready to apply"
                    if self.workspace.refinement_buffer() is not None
                    else "none",
                    "warning"
                    if self.workspace.refinement_buffer() is not None
                    else None,
                ),
            ],
        )
        self._emit_section_title("Requirements")
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
        """Manage the canonical specification and its scratch refinement."""

        if not args or args == ["show"]:
            self._show_specification()
            return
        action, *rest = args
        action = action.lower()
        if action == "path" and not rest:
            self.workspace.initialize_project()
            self._emit(self.workspace.specification_path)
            return
        if action == "edit-spec":
            values, editor_override = self._editor_override(
                rest,
                usage="Use workflow edit-spec [DESCRIPTION|--file PATH] "
                "[--editor COMMAND].",
            )
            self.workspace.initialize_project()
            ensured = self.workspace.ensure_specification()
            target = self.workspace.specification_path
            if not values:
                self._prepare_specification_editor(target)
                self._launch_editor(target, override=editor_override)
                self._finish_specification_editor(target)
            elif editor_override is not None:
                raise SystemExit(
                    "--editor is only used when workflow edit-spec opens the "
                    "specification editor."
                )
            elif values[0] == "--file":
                if len(values) != 2:
                    raise SystemExit("Use workflow edit-spec --file PATH.")
                entered = Path(values[1]).expanduser()
                source = (
                    entered
                    if entered.is_absolute()
                    else self.workspace.root / entered
                ).resolve()
                self.workspace.save_specification(self._read_prompt_file(source))
            elif "--file" in values:
                raise SystemExit(
                    "Use workflow edit-spec [DESCRIPTION] or workflow "
                    "edit-spec --file PATH."
                )
            else:
                content = " ".join(values).strip()
                if not content:
                    raise SystemExit("The specification must not be empty.")
                self.workspace.save_specification(content)
            self._emit_table(
                "Specification updated",
                [
                    ("File", target, "success"),
                    (
                        "Implementation",
                        self.workspace.implementation_state()["state"],
                        None,
                    ),
                    (
                        "Next",
                        self._workflow_next_action(
                            specification_present=True,
                            refinement_present=(
                                self.workspace.refinement_buffer() is not None
                            ),
                            implementation=str(
                                self.workspace.implementation_state()["state"]
                            ),
                        ),
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
        if action == "edit-refinement":
            ensured = self.workspace.ensure_specification()
            if ensured["content"] is None:
                raise SystemExit(
                    "No workflow specification exists. Use 'workflow edit-spec' "
                    "first."
                )
            values, editor_override = self._editor_override(
                rest,
                usage="Use workflow edit-refinement [CHANGE|--file PATH] "
                "[--editor COMMAND].",
            )
            target = self.workspace.refinement_buffer_path
            if not values:
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    target.touch()
                self._launch_editor(target, override=editor_override)
                try:
                    refinement = target.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError) as exc:
                    raise SystemExit(
                        f"Could not read refinement buffer {target}: {exc}"
                    ) from exc
            elif values[0] == "--file":
                if len(values) != 2 or editor_override is not None:
                    raise SystemExit(
                        "Use workflow edit-refinement --file PATH."
                    )
                entered = Path(values[1]).expanduser()
                source = (
                    entered
                    if entered.is_absolute()
                    else self.workspace.root / entered
                ).resolve()
                try:
                    refinement = source.read_text(encoding="utf-8")
                except FileNotFoundError:
                    raise SystemExit(
                        f"Refinement file does not exist: {source}"
                    ) from None
                except (OSError, UnicodeDecodeError) as exc:
                    raise SystemExit(
                        f"Could not read refinement file {source}: {exc}"
                    ) from exc
            elif "--file" in values or editor_override is not None:
                raise SystemExit(
                    "Use workflow edit-refinement [CHANGE|--file PATH] "
                    "[--editor COMMAND]."
                )
            else:
                refinement = " ".join(values).strip()
            result = self.workspace.save_refinement_buffer(refinement)
            present = result["content"] is not None
            implementation = self.workspace.implementation_state()
            self._emit_table(
                "Refinement buffer",
                [
                    (
                        "Status",
                        "ready to apply" if present else "discarded",
                        "success" if present else None,
                    ),
                    ("Storage", "ignored scratch file", None),
                    (
                        "Implementation",
                        implementation["state"],
                        None,
                    ),
                    (
                        "Next",
                        "workflow refine-spec" if present else "workflow status",
                        None,
                    ),
                ],
            )
            if ensured["migrated"]:
                self._info(
                    "The former active prompt ledger was migrated into the "
                    "canonical specification; its original files were kept."
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
                            "none; specification history lives in Git",
                            None,
                        )
                    ],
                )
                return
            self._emit_columns(
                "Specification refinement history",
                ("Status", "Created", "Archived"),
                [
                    (
                        record.get("status") or "unknown",
                        record.get("created_at") or "—",
                        record.get("archived_at") or "—",
                    )
                    for record in records
                ],
            )
            self._emit("Canonical specification history is versioned by Git.")
            return
        raise SystemExit(
            "Use workflow show spec, workflow edit-spec, workflow "
            "edit-refinement, workflow path, or workflow history."
        )

    def refine_specification(self, args: list[str]) -> None:
        """Rewrite the specification in an isolated assistant workspace."""

        if len(args) > 1 or any(
            value.casefold() not in {"codex", "claude"} for value in args
        ):
            raise SystemExit("Use workflow refine-spec [codex|claude].")
        specification = self.workspace.specification()
        if specification is None:
            raise SystemExit(
                "No workflow specification exists. Use 'workflow edit-spec' first."
            )
        refinement = self.workspace.refinement_buffer()
        if refinement is None:
            raise SystemExit(
                "There is no refinement. Use 'workflow edit-refinement' first."
            )
        assistant = (
            args[0].casefold()
            if args
            else str(self._global_settings()["assistant"])
        )
        executable = shutil.which(assistant)
        if executable is None:
            label = "Codex CLI" if assistant == "codex" else "Claude Code"
            raise SystemExit(
                f"{label} was not found. Install and authenticate it, then run "
                "workflow refine-spec again; the refinement remains available."
            )
        tool = "Codex CLI" if assistant == "codex" else "Claude Code"
        with tempfile.TemporaryDirectory(prefix="zippergen-refine-spec-") as raw:
            isolated = Path(raw)
            isolated_specification = isolated / "specification.md"
            isolated_refinement = isolated / "refinement.md"
            isolated_task = isolated / "TASK.md"
            isolated_specification.write_text(
                specification.rstrip() + "\n",
                encoding="utf-8",
            )
            isolated_refinement.write_text(
                refinement.rstrip() + "\n",
                encoding="utf-8",
            )
            isolated_task.write_text(
                "# Specification revision\n\n"
                "This isolated directory contains only the current "
                "specification and the requested refinement. Read "
                "specification.md and refinement.md. Rewrite "
                "specification.md so it incorporates the requested change "
                "as a clean, complete specification. Preserve every "
                "unaffected requirement. Do not create implementation code, "
                "tests, configuration, or additional files.\n",
                encoding="utf-8",
            )
            instruction = (
                "Read TASK.md and perform the specification revision. Edit "
                "specification.md directly and do nothing else."
            )
            if assistant == "codex":
                command = [
                    executable,
                    "exec",
                    "--json",
                    "--strict-config",
                    "--skip-git-repo-check",
                    "--cd",
                    str(isolated),
                    "--sandbox",
                    "workspace-write",
                    "--ignore-user-config",
                    "--config",
                    "mcp_servers={}",
                    "--config",
                    'web_search="disabled"',
                    "--config",
                    "agents.enabled=false",
                    "--config",
                    "sandbox_workspace_write.network_access=false",
                    "-",
                ]
                stdin = instruction
            else:
                command = [
                    executable,
                    "--print",
                    "--permission-mode",
                    "acceptEdits",
                    "--tools",
                    "Read,Edit,Write",
                    "--safe-mode",
                    "--no-chrome",
                    "--disable-slash-commands",
                    "--strict-mcp-config",
                    instruction,
                ]
                stdin = None
            self._emit_table(
                "Specification refinement",
                [
                    ("Tool", tool, None),
                    ("Input", "specification and refinement only", "success"),
                    ("Implementation", "not exposed to the assistant", "success"),
                ],
            )
            try:
                completed = subprocess.run(
                    command,
                    cwd=isolated,
                    input=stdin,
                    text=True,
                    capture_output=True,
                    check=False,
                )
            except OSError as exc:
                raise SystemExit(
                    f"Could not start {tool}: {exc}; the refinement remains "
                    "available."
                ) from exc
            if assistant == "codex":
                self._emit_codex_output(
                    self._parse_codex_output(completed.stdout, completed.stderr)
                )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout).strip()
                suffix = f": {detail[:300]}" if detail else ""
                raise SystemExit(
                    f"{tool} exited with status {completed.returncode}{suffix}; "
                    "the specification and refinement were not changed."
                )
            try:
                revised = isolated_specification.read_text(
                    encoding="utf-8"
                ).strip()
            except (OSError, UnicodeDecodeError) as exc:
                raise SystemExit(
                    f"Could not read the revised specification: {exc}; the "
                    "refinement remains available."
                ) from exc
            if not revised:
                raise SystemExit(
                    "The assistant returned an empty specification; the current "
                    "specification and refinement were preserved."
                )
        previous_fingerprint = self.workspace.specification_fingerprint()
        self.workspace.save_specification(revised)
        self.workspace.consume_refinement_buffer()
        implementation = self.workspace.implementation_state()
        changed = previous_fingerprint != self.workspace.specification_fingerprint()
        self._emit_table(
            "Specification revised",
            [
                ("File", self.workspace.specification_path, "success"),
                ("Change", "updated" if changed else "no textual change", None),
                ("Refinement", "consumed", "success"),
                ("Implementation", implementation["state"], None),
                (
                    "Next",
                    self._workflow_next_action(
                        specification_present=True,
                        refinement_present=False,
                        implementation=str(implementation["state"]),
                    ),
                    None,
                ),
            ],
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
        title: str = "Editor",
        return_hint: str | None = None,
    ) -> None:
        command, source = self._effective_editor(override)
        try:
            displayed = target.relative_to(self.workspace.root)
        except ValueError:
            displayed = target
        rows: list[tuple[str, object, StatusKind | None]] = [
            ("File", displayed, None),
            ("Command", shlex.join(command), "success"),
            ("Source", source, None),
        ]
        if return_hint is not None:
            rows.append(("Return", return_hint, None))
        self._emit_table(title, rows)
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
            current = self._ensure_workflow_configured("edit its source")
            module_ref = self.workspace.absolute_spec(current).partition(":")[0]
            target = Path(module_ref)
            if target.suffix != ".py" or not target.is_file():
                raise SystemExit(
                    f"The project workflow is not backed by a Python file: {current}"
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
        self._emit_next(next_steps)

    def _migrate_workflow_entry(
        self,
        *,
        prompt_for_multiple: bool = True,
    ) -> str | None:
        """Move a former private workflow pointer into the project manifest."""

        result = self.workspace.migrate_workflow_entry()
        raw_candidates = result.get("candidates")
        candidates = (
            [str(value) for value in raw_candidates]
            if isinstance(raw_candidates, (tuple, list))
            else []
        )
        if candidates and prompt_for_multiple:
            selected = self._select(
                "Choose this project's workflow",
                candidates,
                prompt="Select workflow",
            )
            assert isinstance(selected, str)
            result = self.workspace.migrate_workflow_entry(selected=selected)
        entry = result.get("workflow_entry")
        source = result.get("source")
        if entry and source:
            self._associate_implementation_workflow(str(entry))
            description = {
                "private workspace": "former private Studio state",
                "discovery": "the only discovered workflow",
                "selection": "your selection",
            }.get(str(source), str(source))
            self._info(
                f"Recorded {entry} in zippergen.toml from {description}; "
                "validation has not run."
            )
        return str(entry) if entry else None

    def _migrate_project_configuration(self) -> None:
        """Move portable legacy model and connector settings into the manifest."""

        result = self.workspace.migrate_project_configuration()
        if result.get("migrated"):
            self._info(
                "Moved portable model and connector configuration into "
                "zippergen.toml; machine-specific settings and secrets remain "
                "private on this site."
            )

    def _ensure_workflow_configured(self, purpose: str) -> str:
        entry = self.workspace.workflow_entry
        if entry:
            return entry
        if self.workspace.manifest_path.exists():
            entry = self._migrate_workflow_entry()
            if entry:
                return entry
        candidates = self.workspace.discover_workflows()
        if not candidates:
            raise SystemExit(
                "No workflow entry points were discovered. Use 'workflow list' "
                "to confirm discovery, then inspect the generated Python files "
                "for a top-level @workflow definition."
            )
        self.workspace.initialize_project()
        entry = self._migrate_workflow_entry()
        if entry is None:
            raise SystemExit(f"No project workflow is configured to {purpose}.")
        return entry

    def _current_context(self, *, purpose: str = "inspect it"):
        from zippergen.serve import load_workflow_spec

        current = self._ensure_workflow_configured(purpose)
        workflow, module = load_workflow_spec(self.workspace.absolute_spec(current))
        return current, workflow, module

    def configure_project(self, args: list[str]) -> None:
        if not args:
            self._show_project_inventory()
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
                    prompt="Select reset scope",
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
                "Use project, project init [NAME], project rename NAME, "
                "project reset, project reset fresh [--yes], or "
                "project reset state [--yes]."
            )
        existed = self.workspace.manifest_path.exists()
        manifest = self.workspace.initialize_project(
            name=args[1] if len(args) == 2 else None
        )
        self._migrate_workflow_entry()
        manifest = self.workspace.project_manifest()
        result = "already exists" if existed else "created"
        self._success(
            f"Project manifest {result}: {self.workspace.manifest_path}"
        )
        self._emit_table(
            "Project initialized",
            [
                ("Name", manifest["name"], None),
                ("Manifest", self.workspace.manifest_path, None),
                ("Specification", self.workspace.specification_path, None),
                (
                    "Workflow",
                    manifest.get("workflow_entry") or "not configured",
                    "success" if manifest.get("workflow_entry") else "warning",
                ),
                (
                    "Next",
                    "workflow show · current"
                    if manifest.get("workflow_entry")
                    else "workflow edit-spec · workflow import PATH.py · "
                    "workflow list · current",
                    None,
                ),
            ],
        )

    def _show_project_inventory(self) -> None:
        """Show what belongs to this project without changing any selection."""

        from zippergen.serve import _deployment_service_status, _validate_workflow
        from zippergen.studio_stores import deployment_profiles

        manifest = self.workspace.project_manifest()
        specification = self.workspace.specification()
        refinement = self.workspace.refinement_buffer() is not None
        implementation = self.workspace.implementation_state()
        current = self.workspace.workflow_entry
        workflow_name = "not configured"
        validation = "not available"
        validation_kind: StatusKind = "warning"
        llm_action_count = 0
        if current:
            try:
                _current, workflow, module = self._current_context()
                workflow_name = f"{workflow.name} · {current}"
                raw_action_sites = workflow_semantics(
                    workflow,
                    module,
                ).get("action_sites")
                if isinstance(raw_action_sites, list):
                    llm_action_count = sum(
                        1
                        for site in raw_action_sites
                        if isinstance(site, dict)
                        and site.get("kind") == "llm"
                    )
                result = _validate_workflow(workflow, module)
                validation = "valid" if result["valid"] else "invalid"
                validation_kind = "success" if result["valid"] else "error"
            except (Exception, SystemExit) as exc:
                workflow_name = f"{current} · cannot load"
                validation = str(exc)
                validation_kind = "error"

        runs = self.workspace.list_runs()
        newest_run = runs[0] if runs else None
        deployments = deployment_profiles(self.workspace)

        self._emit_section_title(
            f"Project · {manifest['name']}",
            major=True,
        )
        self._emit()
        self._emit(
            f"Root  {self.workspace.root}"
        )
        self._emit(
            "Manifest  "
            + (
                f"{self._status_mark('success')} {self.workspace.manifest_path}"
                if manifest["exists"]
                else f"{self._status_mark('warning')} not created"
            )
        )
        self._emit()
        self._emit(f"├── Workflow · {workflow_name}")
        self._emit(
            "│   ├── Specification · "
            + (
                f"{self._status_mark('success')} "
                f"{self.workspace.specification_path.name}"
                if specification is not None
                else f"{self._status_mark('warning')} not written"
            )
        )
        self._emit(
            "│   ├── Refinement · "
            + (
                f"{self._status_mark('warning')} pending"
                if refinement
                else "none"
            )
        )
        self._emit(
            "│   ├── Implementation · "
            f"{self._status_mark('success' if implementation['state'] == 'current' else 'warning')} "
            f"{implementation['state']}"
        )
        self._emit(
            f"│   └── Validation · {self._status_mark(validation_kind)} "
            f"{validation}"
        )

        model_default = "none"
        model_override_count = 0
        connector_count = 0
        if current:
            model_assignments = self.workspace.model_assignment_profile(current)
            model_default = str(model_assignments.get("default") or "none")
            model_override_count = len(
                model_assignments.get("lifelines") or {}
            ) + len(
                model_assignments.get("actions") or {}
            )
            connector_assignments = (
                self.workspace.connector_assignment_profile(current)
            )
            connector_count = len(
                connector_assignments.get("lifelines") or {}
            ) + len(connector_assignments.get("actions") or {}) + len(
                self.workspace.connector_binding_profile(current)
            )
        self._emit(
            f"├── Models · default {model_default} · "
            f"{model_override_count} override"
            f"{'' if model_override_count == 1 else 's'} · "
            f"{llm_action_count} LLM action"
            f"{'' if llm_action_count == 1 else 's'}"
        )
        self._emit(
            f"├── Connectors · {connector_count} assignment"
            f"{'' if connector_count == 1 else 's'}"
        )
        missing_site = self.workspace.missing_site_requirements()
        if missing_site:
            self._emit(
                f"├── Site setup · {self._status_mark('warning')} "
                f"{len(missing_site)} missing"
            )
            for index, item in enumerate(missing_site):
                branch = "│   └──" if index == len(missing_site) - 1 else "│   ├──"
                self._emit(
                    f"{branch} {item['name']} · {item['command']}"
                )
        else:
            self._emit(
                f"├── Site setup · {self._status_mark('success')} ready"
            )
        run_summary = "none"
        if newest_run is not None:
            run_summary = (
                f"{len(runs)} · newest {newest_run['run_id']} "
                f"({newest_run['status']})"
            )
        self._emit(f"├── Runs · {run_summary}")
        if not deployments:
            self._emit("└── Deployments · none")
        else:
            self._emit(
                f"└── Deployments · {len(deployments)}"
            )
            for index, (_path, profile) in enumerate(deployments):
                name = str(profile.get("name") or "unknown")
                service = _deployment_service_status(name)
                alignment, alignment_kind, changed = (
                    self._deployment_project_alignment(profile)
                )
                tree_alignment = (
                    "differs from current project"
                    if changed
                    else "matches current project"
                    if alignment_kind == "success"
                    else alignment
                )
                branch = "    └──" if index == len(deployments) - 1 else "    ├──"
                self._emit(
                    f"{branch} {name} · {service['state']} · "
                    f"{self._status_mark(alignment_kind)} {tree_alignment}"
                )
        self._emit_next(
            "workflow status · model · connector · runs · deploy list"
        )

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
        implementation_lock_exists = self.workspace.implementation_lock_path.exists()
        legacy_prompts_exist = (
            self.workspace.prompts_directory != self.workspace.root
            and self.workspace.prompts_directory.exists()
        )
        private_exists = bool(
            summary["workspace_exists"] or summary["project_local_exists"]
        )
        visible_design_exists = bool(
            manifest_exists
            or specification_exists
            or implementation_lock_exists
            or legacy_prompts_exist
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
                    "Implementation record",
                    "archive"
                    if fresh and implementation_lock_exists
                    else "kept/none",
                    "warning" if fresh and implementation_lock_exists else None,
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
                        "project init · workflow edit-spec"
                        if fresh
                        else "workflow list · workflow import PATH.py · "
                        "workflow edit-spec · current"
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
                (
                    "Workflow",
                    "not configured"
                    if fresh
                    else self.workspace.workflow_entry or "not configured",
                    None,
                ),
                ("Run", "none selected", None),
                ("Implementation request", "none", None),
                (
                    "Next",
                    (
                        "project init · workflow edit-spec"
                        if fresh and project_exists
                        else "workflow show · current"
                        if project_exists and self.workspace.workflow_entry
                        else "workflow list · workflow import PATH.py · "
                        "workflow edit-spec · current"
                        if project_exists
                        else "exit and recreate the project directory"
                    ),
                    None,
                ),
            ],
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
                return {
                    **record,
                    "status": "assistant_interrupted",
                    "assistant_error": (
                        "the Studio process ended before the assistant returned"
                    ),
                }
            return record
        if record.get("status") == "awaiting_review":
            return {**record, "status": "implemented"}
        if record.get("status"):
            return record
        return {**record, "status": "prepared"}

    def _task_state(
        self,
        record: dict[str, object],
    ) -> tuple[str, StatusKind]:
        status = str(record.get("status") or "prepared")
        states: dict[str, tuple[str, StatusKind]] = {
            "prepared": ("ready for assistant", "success"),
            "assistant_running": ("assistant running", "info"),
            "awaiting_review": ("implemented", "success"),
            "implemented": ("implemented", "success"),
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
        kinds: dict[str, StatusKind] = {
            "passed": "success",
            "failed": "error",
            "not_run": "warning",
        }
        items: list[tuple[int, str, str, str]] = []
        for index, value in enumerate(checks, start=1):
            if not isinstance(value, dict):
                continue
            status = str(value.get("status") or "not_run")
            command = str(value.get("command") or "unspecified command")
            detail = str(value.get("detail") or "No result detail reported.")
            items.append((index, status, command, detail))
        selected = [
            item
            for item in items
            if not problems_only or item[1] != "passed"
        ]
        if not selected:
            return
        if str(record.get("assistant_verification") or "") != "passed":
            priority = {"failed": 0, "not_run": 1, "passed": 2}
            selected.sort(
                key=lambda item: (priority.get(item[1], 1), item[0])
            )

        counts = {"passed": 0, "failed": 0, "not_run": 0}
        for _index, status, _command, _detail in items:
            if status in counts:
                counts[status] += 1
        total = len(items)
        title = (
            "Failed or incomplete assistant checks"
            if problems_only
            else "Assistant checks"
        )
        self._emit_section_title(title)
        self._emit(
            f"{total} check{'s' if total != 1 else ''} · "
            f"{counts['passed']} passed · {counts['failed']} failed · "
            f"{counts['not_run']} not run"
        )
        self._emit()
        for index, status, command, detail in selected:
            self._status(
                kinds.get(status, "warning"),
                f"{index}. {status.replace('_', ' ')}",
            )
            self._emit_wrapped_field("Command", command)
            self._emit_wrapped_field("Result", detail)
            self._emit()

    def _task_next(self, record: dict[str, object]) -> str:
        status = str(record.get("status") or "prepared")
        if status == "assistant_running":
            return "wait for the assistant session to return"
        if status in {"assistant_failed", "assistant_interrupted"}:
            return "workflow implement codex · workflow implement claude"
        implementation = str(self.workspace.implementation_state()["state"])
        return self._workflow_next_action(
            specification_present=self.workspace.specification() is not None,
            refinement_present=self.workspace.refinement_buffer() is not None,
            implementation=implementation,
        )

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
            return "not recorded; implementation inferred from specification change"
        if record.get("manual_integration"):
            return "not used; canonical specification was edited manually"
        return "not started"

    def _task_context(
        self,
        record: dict[str, object],
    ) -> tuple[str, StatusKind]:
        del record
        implementation = self.workspace.implementation_state()
        state = str(implementation["state"])
        if state == "current":
            return "matches the current specification", "success"
        if state == "stale":
            return "specification changed after implementation", "warning"
        if state == "external":
            return "not known to have been generated from this specification", "warning"
        return "no implementation is present", "warning"

    def manage_task(self, args: list[str]) -> None:
        if len(args) > 1 or (
            args
            and args[0].lower()
            not in {"show", "path", "history", "details"}
        ):
            raise SystemExit(
                "Use workflow status [--details] or workflow history."
            )
        action = args[0].lower() if args else "summary"
        if action == "history":
            records = self.workspace.list_requests()
            if not records:
                self._emit_table(
                    "Implementation history",
                    [
                        (
                            "Status",
                            "none; use workflow implement",
                            "warning",
                        )
                    ],
                )
                return
            history_rows: list[tuple[object, ...]] = []
            for record in records:
                state, state_kind = self._task_state(record)
                history_rows.append(
                    (
                        record["request_id"],
                        record["kind"],
                        f"{self._status_mark(state_kind)} {state}",
                        record.get("refreshes_request") or "—",
                        record.get("created_at") or "—",
                    )
                )
            self._emit_columns(
                "Implementation history",
                ("Request", "Kind", "State", "Refreshes", "Created"),
                history_rows,
            )
            return

        record = self.workspace.current_request()
        if record is not None:
            record = self._normalize_task_lifecycle(record)
        if record is None:
            if action in {"show", "path"}:
                raise SystemExit(
                    "No current implementation task is open. A selected "
                    "workflow may still exist; use 'current' to inspect it."
                )
            implementation = self.workspace.implementation_state()
            implementation_state = str(implementation["state"])
            self._emit_table(
                "Workflow implementation task",
                [
                    (
                        "Status",
                        "none; use workflow implement",
                        "warning",
                    ),
                    ("Implementation", implementation_state, None),
                    *self._git_status_rows(),
                    (
                        "Next",
                        self._workflow_next_action(
                            specification_present=(
                                self.workspace.specification() is not None
                            ),
                            refinement_present=(
                                self.workspace.refinement_buffer() is not None
                            ),
                            implementation=implementation_state,
                        ),
                        None,
                    ),
                ],
            )
            return
        record = self._normalize_task_lifecycle(record)
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
        implementation = self.workspace.implementation_state()
        self._emit_table(
            "Workflow implementation task",
            [
                ("Status", state, state_kind),
                ("Implementation", implementation["state"], None),
                ("Kind", record["kind"], None),
                ("Workflow", record.get("workflow_spec") or "new workflow", None),
                ("Assistant", self._task_assistant(record), None),
                ("Execution", self._task_execution(record), None),
                ("Assistant checks", *self._task_verification(record)),
                *(
                    [
                        (
                            "Check summary",
                            self._task_verification_summary(record),
                            None,
                        )
                    ]
                    if self._task_verification_summary(record)
                    else []
                ),
                ("Context", context, context_kind),
                *self._git_status_rows(),
                *(
                    [
                        ("Request", record["request_id"], None),
                        (
                            "Refreshes",
                            record.get("refreshes_request") or "—",
                            None,
                        ),
                        ("Record", ".zippergen/current-task.md", None),
                    ]
                    if action == "details"
                    else []
                ),
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
                "The assistant returned without writing its required check report.",
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
                    "assistant result field 'verification' must be passed, "
                    "failed, or incomplete"
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
                    "before assistant checks can be marked passed."
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
                f"Studio could not accept the assistant check report: {detail}.",
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
            self._emit_section_title("Assistant report")
            self._emit(output.report)
            self._emit()
        for diagnostic in output.diagnostics[:3]:
            self._warning(f"Codex diagnostic: {diagnostic}")
        if len(output.diagnostics) > 3:
            self._warning(
                f"Codex emitted {len(output.diagnostics) - 3} additional "
                "diagnostic lines; the assistant check report is "
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

    def _implementation_file_snapshot(self) -> dict[str, str] | None:
        """Fingerprint visible project files for the implementation summary."""

        ignored = {
            ".git",
            ".hg",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            ".tox",
            ".venv",
            ".zippergen",
            "__pycache__",
            "build",
            "dist",
            "node_modules",
            "venv",
        }
        snapshot: dict[str, str] = {}
        try:
            for directory, directory_names, file_names in os.walk(
                self.workspace.root,
                followlinks=False,
            ):
                directory_names[:] = [
                    name for name in directory_names if name not in ignored
                ]
                base = Path(directory)
                for name in file_names:
                    path = base / name
                    relative = path.relative_to(self.workspace.root)
                    try:
                        if path.is_symlink():
                            digest = "symlink:" + os.readlink(path)
                        elif path.is_file():
                            digest = hashlib.sha256(path.read_bytes()).hexdigest()
                        else:
                            digest = "missing"
                    except OSError:
                        digest = "unreadable"
                    snapshot[relative.as_posix()] = digest
        except OSError:
            return None
        return snapshot

    @staticmethod
    def _implementation_changed_files(
        before: dict[str, str] | None,
        after: dict[str, str] | None,
    ) -> tuple[str, ...] | None:
        if before is None or after is None:
            return None
        return tuple(
            sorted(
                path
                for path in before.keys() | after.keys()
                if before.get(path) != after.get(path)
            )
        )

    @staticmethod
    def _implementation_files_summary(files: tuple[str, ...] | None) -> str:
        if files is None:
            return "not available"
        if not files:
            return "none detected"
        shown = files[:8]
        suffix = f" · +{len(files) - len(shown)} more" if len(files) > 8 else ""
        return f"{len(files)} · " + " · ".join(shown) + suffix

    def _implementation_validation_summary(
        self,
        workflow_spec: str | None,
    ) -> tuple[str, StatusKind]:
        """Validate the produced workflow without prompting for a selection."""

        if workflow_spec is None:
            return ("not available; select the produced workflow", "warning")
        try:
            from zippergen.serve import _validate_workflow, load_workflow_spec

            workflow, module = load_workflow_spec(
                self.workspace.absolute_spec(workflow_spec)
            )
            result = _validate_workflow(workflow, module)
        except (Exception, SystemExit) as exc:
            return (f"failed to load or validate: {exc}", "error")
        if result["valid"]:
            return ("passed", "success")
        failed = sum(
            1
            for check in result["checks"]  # type: ignore[index]
            if str(check.get("status") or "").casefold() == "fail"
        )
        detail = f"failed · {failed} check{'s' if failed != 1 else ''}"
        return (detail, "error")

    @staticmethod
    def _git_path_summary(paths: tuple[str, ...]) -> str:
        shown = paths[:5]
        suffix = f" · +{len(paths) - len(shown)} more" if len(paths) > 5 else ""
        return " · ".join(shown) + suffix

    def _uncommitted_implementation_paths(self) -> tuple[str, ...] | None:
        """Inspect the portable commit unit without making Git mandatory."""

        unit = implementation_status_unit(self.workspace)
        if unit is None:
            return None
        return uncommitted_commit_unit_paths(unit)

    def _git_status_rows(self) -> list[tuple[str, object, StatusKind | None]]:
        changed = self._uncommitted_implementation_paths()
        if not changed:
            return []
        return [
            (
                "Git",
                "uncommitted implementation files: "
                f"{self._git_path_summary(changed)}; a fresh clone may derive "
                "a different implementation state",
                "warning",
            )
        ]

    def _warn_if_git_state_differs(self) -> None:
        changed = self._uncommitted_implementation_paths()
        if not changed:
            return
        self._warning(
            "The implementation has uncommitted project files: "
            f"{self._git_path_summary(changed)}. A fresh clone may derive a "
            "different implementation state. Deployment will still proceed."
        )

    def _offer_implementation_commit(
        self,
        workflow_spec: str | None,
        *,
        include_manifest: bool,
    ) -> None:
        """Offer one explicit, path-limited commit after implementation."""

        if not self._interactive_offers_enabled:
            return
        unit = implementation_commit_unit(
            self.workspace,
            include_manifest=include_manifest,
        )
        if unit is None:
            return
        changed = uncommitted_commit_unit_paths(unit)
        if not changed:
            return
        self._emit_table(
            "Commit implementation",
            [
                (
                    "Files",
                    self._git_path_summary(unit.project_paths),
                    None,
                ),
                (
                    "Scope",
                    (
                        "specification, implementation, portable lock, and "
                        "manifest only"
                        if include_manifest
                        else "specification, implementation, and portable lock only"
                    ),
                    "info",
                ),
            ],
        )
        if not self._confirm_action(
            "Commit these files together? [Y/n]: ",
            cancel_message=(
                "Git commit skipped; project changes remain in the working tree."
            ),
            default=True,
        ):
            return
        workflow_name = (
            workflow_spec.rpartition(":")[2]
            if workflow_spec
            else "workflow"
        )
        default_message = f"Implement {workflow_name} workflow"
        try:
            entered = self.input(
                f"Commit message [{default_message}]: "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            self._warning(
                "Git commit skipped; project changes remain in the working tree."
            )
            return
        try:
            revision = commit_implementation_unit(
                unit,
                entered or default_message,
            )
        except GitCommitError as exc:
            self._warning(
                f"Git commit was not created: {exc} Project changes were preserved."
            )
            return
        self._success(
            f"Committed the specification and implementation together ({revision})."
        )

    def _record_portable_implementation(
        self,
        workflow_spec: str | None,
    ) -> dict[str, object]:
        """Write the committed relationship between spec and generated files."""

        if workflow_spec is None:
            raise WorkspaceError(
                "The assistant did not produce exactly one discoverable workflow."
            )
        canonical = self.workspace.canonical_spec(
            workflow_spec,
            cwd=self.workspace.root,
        )
        configured = self.workspace.workflow_entry
        if configured is None:
            self.workspace.select_workflow(canonical, cwd=self.workspace.root)
        elif configured != canonical:
            raise WorkspaceError(
                f"The assistant produced {canonical}, but this project is "
                f"configured for {configured}."
            )
        files = [path for path, _role in self._workflow_file_records()]
        return self.workspace.write_implementation_lock(files)

    def run_assistant(self, args: list[str]) -> None:
        if self.workspace.refinement_buffer() is not None:
            raise SystemExit(
                "The refinement must be applied before implementation. Use "
                "'workflow refine-spec' first."
            )
        rerun = "--rerun" in args
        interactive = "--interactive" in args
        values = [
            value
            for value in args
            if value not in {"--rerun", "--interactive"}
        ]
        if len(values) > 1 or any(
            value.lower() not in {"codex", "claude"} for value in values
        ) or any(
            args.count(option) > 1
            for option in ("--rerun", "--interactive")
        ):
            raise SystemExit(
                "Use workflow implement, workflow implement codex, "
                "workflow implement claude, or workflow implement "
                "[codex|claude] [--rerun]. Use workflow implement "
                "codex --interactive only for an interactive session."
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
                "No implementation request could be prepared. Use "
                "'workflow edit-spec' first."
            )
        status = str(record.get("status") or "prepared")
        if status in {"awaiting_review", "implemented"}:
            manual_first_pass = bool(record.get("manual_integration")) and not bool(
                record.get("assistant")
            )
            if not rerun and not manual_first_pass:
                raise SystemExit(
                    "The assistant has already returned for this implementation. "
                    "Use 'workflow diff' to inspect it; use 'workflow implement "
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
                "the current implementation request remains available; use "
                "'workflow status --details' to inspect its internal record."
            )
        self._ensure_assistant_test_environment()
        try:
            self.workspace.assistant_result_path.unlink(missing_ok=True)
        except OSError as exc:
            raise SystemExit(
                "Could not clear the previous assistant-result handoff at "
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
                ("Implementation task", "prepared", "success"),
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
            "keep all generated code visible, run the requested checks, and "
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
        files_before = self._implementation_file_snapshot()
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
                f"{completed.returncode}; the implementation request remains "
                "available through 'workflow status'."
            )
        assistant_result = self._consume_assistant_result()
        result_fingerprint = self.workspace.specification_fingerprint()
        changed = record.get("specification_fingerprint") != result_fingerprint
        workflow_spec = str(record.get("workflow_spec") or "") or None
        if workflow_spec is None:
            candidates = self.workspace.discover_workflows()
            if len(candidates) == 1:
                workflow_spec = self.workspace.canonical_spec(
                    candidates[0], cwd=self.workspace.root
                )
        try:
            self._record_portable_implementation(workflow_spec)
        except (WorkspaceError, OSError, ValueError) as exc:
            self.workspace.update_request(
                str(record["request_id"]),
                status="assistant_failed",
                assistant_finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                assistant_exit_code=0,
                assistant_verification=assistant_result.verification,
                assistant_verification_summary=assistant_result.summary,
                assistant_verification_checks=[
                    dict(check) for check in assistant_result.checks
                ],
                assistant_result_error=assistant_result.error,
                assistant_report=codex_output.report,
                result_specification_fingerprint=result_fingerprint,
            )
            raise SystemExit(
                "The assistant returned, but Studio could not record a portable "
                f"implementation: {exc}"
            ) from exc
        record = self.workspace.update_request(
            str(record["request_id"]),
            status="implemented",
            workflow_spec=workflow_spec,
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
            self._success("Assistant checks passed.")
        elif assistant_result.verification == "failed":
            self._error(
                "Assistant checks failed; resolve them before using the change."
            )
        else:
            self._warning(
                "Assistant checks are incomplete; a normal assistant exit "
                "does not mean the checks passed."
            )
        specification_result = (
            "changed since implementation preparation"
            if changed
            else "unchanged"
        )
        files_after = self._implementation_file_snapshot()
        changed_files = self._implementation_changed_files(
            files_before,
            files_after,
        )
        validation = self._implementation_validation_summary(workflow_spec)
        assistant_report = (
            str(record.get("assistant_report") or "").strip()
            or assistant_result.summary
            or "none provided"
        )
        self._emit_table(
            "Implementation summary",
            [
                ("Status", "implemented", "success"),
                (
                    "Files changed",
                    self._implementation_files_summary(changed_files),
                    None,
                ),
                (
                    "Specification",
                    specification_result,
                    "success" if changed else "warning",
                ),
                ("Portable record", self.workspace.implementation_lock_path, "success"),
                ("Validation", *validation),
                ("Assistant checks", *self._task_verification(record)),
                ("Assistant report", assistant_report, None),
                ("Next", self._task_next(record), None),
            ],
        )
        if assistant_result.verification != "passed":
            self._emit_task_verification_checks(record, problems_only=True)
        manifest_relative = self.workspace.manifest_path.relative_to(
            self.workspace.root
        ).as_posix()
        self._offer_implementation_commit(
            workflow_spec,
            include_manifest=(
                changed_files is not None
                and manifest_relative in changed_files
            ),
        )

    def show_current(self) -> None:
        from zippergen.serve import _validate_workflow

        state = self.workspace.load()
        global_settings = self._global_settings()
        manifest = self.workspace.project_manifest()
        request = self.workspace.current_request()
        if request is not None:
            request = self._normalize_task_lifecycle(request)
        specification = self.workspace.specification()
        refinement = self.workspace.refinement_buffer() is not None
        implementation = self.workspace.implementation_state()
        task_state, task_state_kind = (
            self._task_state(request)
            if request
            else ("none", None)
        )
        refinement_status = "ready to apply" if refinement else "none"
        self._emit_section_title("Current", major=True)
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
                        else "not written; use workflow edit-spec"
                    ),
                    "success" if specification is not None else "warning",
                ),
                (
                    "Refinement",
                    refinement_status,
                    "warning" if refinement else None,
                ),
                (
                    "Implementation",
                    implementation["state"],
                    "success"
                    if implementation["state"] == "current"
                    else "warning",
                ),
                (
                    "Last assistant task",
                    (
                        f"{request['kind']} — {task_state}"
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
                    [
                        (
                            "Assistant checks",
                            *self._task_verification(request),
                        )
                    ]
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
        workflow_entry = self.workspace.workflow_entry
        if workflow_entry:
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
            raw_definitions = model.get("action_definitions")
            action_definitions = (
                raw_definitions if isinstance(raw_definitions, dict) else {}
            )
            assistant_actions = []
            for site in action_sites:
                if not isinstance(site, dict) or site.get("kind") != "assistant":
                    continue
                action_name = str(site.get("action") or "")
                raw_definition = action_definitions.get(action_name)
                definition = (
                    raw_definition
                    if isinstance(raw_definition, dict)
                    else {}
                )
                assistant_actions.append(
                    f"{site.get('lifeline')}.{action_name} "
                    f"({definition.get('backend') or 'runtime'}; "
                    f"{definition.get('access') or 'write'})"
                )
            from zippergen.connectors import connector_requirements_from_module

            connector_requirements = connector_requirements_from_module(module)
            connector_bindings = self.workspace.connector_binding_profile(
                workflow_entry
            )
            human_connector_assignments = (
                self.workspace.connector_assignment_profile(
                    workflow_entry
                )
            )
            connector_parts = [
                *[
                    f"{target}={configuration}"
                    for target, configuration in
                    human_connector_assignments["lifelines"].items()
                ],
                *[
                    f"{target}={configuration}"
                    for target, configuration in
                    human_connector_assignments["actions"].items()
                ],
                *[
                    f"{item.name}={connector_bindings.get(item.name, 'not bound')}"
                    for item in connector_requirements
                ],
            ]
            connector_summary = " · ".join(connector_parts) or "none"
            active_models = self._llm_action_lifelines(workflow, module)
            llm_participants = list(active_models)
            validation = _validate_workflow(workflow, module)
            self._emit_table(
                "Workflow",
                [
                    ("Entry point", workflow_entry, "success"),
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
                    (
                        "Connectors",
                        connector_summary,
                        (
                            "warning"
                            if any(
                                item.required
                                and item.name not in connector_bindings
                                for item in connector_requirements
                            )
                            else None
                        ),
                    ),
                    (
                        "Validation",
                        "valid" if validation["valid"] else "invalid",
                        "success" if validation["valid"] else "error",
                    ),
                ],
            )
            assignments = self.workspace.model_assignment_profile(
                workflow_entry,
                default=default_llm_spec(module),
            )
            configurations = self.workspace.model_configurations()
            default_configuration = str(assignments["default"])
            overrides = assignments.get("lifelines") or {}
            action_overrides = assignments.get("actions") or {}
            assert isinstance(overrides, dict)
            assert isinstance(action_overrides, dict)
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
                    for action_name in actions:
                        target = f"{lifeline}.{action_name}"
                        action_explicit = action_overrides.get(target)
                        participant_explicit = overrides.get(lifeline)
                        effective = str(
                            action_explicit
                            or participant_explicit
                            or default_configuration
                        )
                        source = (
                            "action override"
                            if action_explicit
                            else "participant"
                            if participant_explicit
                            else "default"
                        )
                        spec = configurations.get(effective, {}).get(
                            "spec", "missing"
                        )
                        model_rows.append(
                            (
                                target,
                                f"{effective} → {spec} ({source})",
                                None,
                            )
                        )
            else:
                model_rows.append(("Assignments", "none", None))
            selected_configurations = {default_configuration} | {
                str(value) for value in overrides.values()
            } | {str(value) for value in action_overrides.values()}
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
            idle_routes = self._model_idle_timeout_routes(
                workflow_entry,
                workflow,
                module,
            )
            model_rows.append(
                (
                    "Local idle release",
                    self._model_idle_routes_summary(idle_routes),
                    "success" if idle_routes else None,
                )
            )
            self._emit_table("Models", model_rows)
        else:
            self._emit_table(
                "Workflow",
                [
                    ("Entry point", "not configured", "warning"),
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
                    (
                        "Assistant",
                        run.get("assistant") or "none selected",
                        None,
                    ),
                    (
                        "Run inspection",
                        "run inspect [PARTICIPANT] [--watch]",
                        None,
                    ),
                ]
            )
        deployment = state.get("last_deployment")
        if deployment:
            from zippergen.serve import _deployment_service_status

            service = _deployment_service_status(str(deployment))
            runtime_rows.append(
                (
                    "Deployment",
                    f"{deployment} — service {service['state']}",
                    (
                        "success"
                        if service["state"] in {"running", "completed"}
                        else "error"
                        if service["state"] == "restarting"
                        else "warning"
                    ),
                )
            )
            runtime_rows.append(
                (
                    "Deployment inspection",
                    f"deploy inspect {deployment} [PARTICIPANT]",
                    None,
                )
            )
        else:
            runtime_rows.append(("Deployment", "none", None))
        self._emit_table("Runtime", runtime_rows)

    def _select(
        self,
        heading: str,
        choices: list[str],
        *,
        prompt: str,
        allow_many: bool = False,
    ):
        if not choices:
            raise SystemExit("No choices are available.")
        self._emit_columns(
            heading,
            ("Choice", "Value"),
            [
                (index, choice)
                for index, choice in enumerate(choices, 1)
            ],
            right_aligned=frozenset({0}),
        )
        range_hint = f"1-{len(choices)}"
        if allow_many:
            range_hint += ", comma-separated"
        raw = self.input(f"{prompt} [{range_hint}]: ").strip()
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
        configured = self.workspace.workflow_entry
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
        rows: list[tuple[object, ...]] = []
        for index, spec in enumerate(candidates, start=1):
            name = spec.rpartition(":")[2] or spec
            rows.append(
                (
                    index,
                    name,
                    spec,
                    (
                        f"{self._status_mark('success')} project workflow"
                        if spec == configured
                        else "available"
                    ),
                )
            )
        self._emit_columns(
            "Available workflows",
            ("#", "Workflow", "Entry point", "Status"),
            rows,
            right_aligned=frozenset({0}),
        )
        self._emit_table(
            "Discovery",
            [
                ("Method", "source scan only", "info"),
                ("Validation", "not run", "warning"),
                (
                    "Next",
                    "workflow import PATH.py:WORKFLOW"
                    if configured is None
                    else "workflow show · workflow validate",
                    None,
                ),
            ],
        )

    @staticmethod
    def _imported_resource_paths(entry: Path, root: Path) -> list[Path]:
        """Find literal local files named by common workflow declarations."""

        try:
            tree = ast.parse(
                entry.read_text(encoding="utf-8"),
                filename=str(entry),
            )
        except (OSError, SyntaxError, UnicodeDecodeError):
            return []
        values: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.keyword) or node.arg not in {
                "files",
                "instructions_file",
            }:
                continue
            candidates = (
                node.value.elts
                if isinstance(node.value, (ast.List, ast.Tuple))
                else [node.value]
            )
            values.extend(
                str(candidate.value)
                for candidate in candidates
                if isinstance(candidate, ast.Constant)
                and isinstance(candidate.value, str)
            )
        resources: list[Path] = []
        for value in values:
            candidate = (root / value).resolve()
            if (
                candidate.is_file()
                and candidate.is_relative_to(root)
                and candidate not in resources
            ):
                resources.append(candidate)
        return resources

    @staticmethod
    def _external_workflow_root(source: Path) -> Path:
        """Find a source project root, or treat the file's folder as its root."""

        for parent in source.parents:
            if any(
                (parent / marker).exists()
                for marker in ("zippergen.toml", "pyproject.toml", ".git")
            ):
                return parent
        return source.parent

    @staticmethod
    def _package_initializers(path: Path, root: Path) -> list[Path]:
        initializers: list[Path] = []
        parent = path.parent
        while parent.is_relative_to(root):
            initializer = parent / "__init__.py"
            if initializer.is_file():
                initializers.append(initializer.resolve())
            if parent == root:
                break
            parent = parent.parent
        return list(reversed(initializers))

    def import_workflow(self, args: list[str]) -> None:
        """Copy an external workflow into the project and discover its entry."""

        if len(args) != 1:
            raise SystemExit("Use workflow import PATH.py[:WORKFLOW].")
        entered = args[0]
        module_text, separator, requested_name = entered.rpartition(":")
        if not separator or not module_text.casefold().endswith(".py"):
            module_text = entered
            requested_name = ""
        source = Path(module_text).expanduser()
        if not source.is_absolute():
            source = (self.workspace.root / source).resolve()
        else:
            source = source.resolve()
        if not source.is_file():
            raise SystemExit(f"Workflow source file does not exist: {source}")
        if source.suffix.casefold() != ".py":
            raise SystemExit("Workflow import requires a Python source file.")
        self.workspace.initialize_project()
        source_root = self._external_workflow_root(source).resolve()
        destination_root = self.workspace.root.resolve()
        if source_root == destination_root:
            candidates = [
                candidate
                for candidate in self.workspace.discover_workflows()
                if (self.workspace.root / candidate.partition(":")[0]).resolve()
                == source.resolve()
            ]
            if not candidates:
                raise SystemExit(
                    "The file is already in this project, but it contains no "
                    "discoverable top-level @workflow."
                )
            selected = next(
                (
                    candidate
                    for candidate in candidates
                    if not requested_name
                    or candidate.rpartition(":")[2] == requested_name
                ),
                None,
            )
            if selected is None or (len(candidates) > 1 and not requested_name):
                self.list_workflows()
                self._info(
                    "The source is already in this project. Repeat workflow "
                    "import with PATH.py:WORKFLOW."
                )
                return
            existing = self.workspace.workflow_entry
            replace = existing is not None and existing != selected
            if replace and not self._confirm_action(
                f"Replace project workflow {existing} with {selected}? [y/N]: ",
                cancel_message="Workflow import cancelled; nothing was changed.",
                default=False,
            ):
                return
            self._configure_workflow_spec(selected, replace=replace)
            if replace:
                self.workspace.remove_implementation_lock()
            self._success(f"Configured existing project workflow: {selected}")
            return

        source_entries = [
            candidate
            for candidate in discover_workflow_specs(source_root)
            if (source_root / candidate.partition(":")[0]).resolve() == source
        ]
        if not source_entries:
            raise SystemExit(
                "The source contains no discoverable top-level @workflow."
            )
        source_selected = next(
            (
                candidate
                for candidate in source_entries
                if requested_name
                and candidate.rpartition(":")[2] == requested_name
            ),
            None,
        )
        if requested_name and source_selected is None:
            raise SystemExit(
                f"Workflow {requested_name!r} was not found. Available: "
                f"{', '.join(source_entries)}."
            )
        if source_selected is None and len(source_entries) == 1:
            source_selected = source_entries[0]
        if source_selected is None:
            self._emit_columns(
                "Workflow entry points in source",
                ("Workflow", "Entry point"),
                [
                    (candidate.rpartition(":")[2], candidate)
                    for candidate in source_entries
                ],
            )
            self._emit_next("workflow import PATH.py:WORKFLOW")
            return
        incoming_entry = (
            f"{source.relative_to(source_root).as_posix()}:"
            f"{source_selected.rpartition(':')[2]}"
        )

        dependencies = self._local_python_dependencies(
            source,
            root=source_root,
        )
        resources = self._imported_resource_paths(source, source_root)
        sources: list[Path] = []
        for item in [source, *dependencies]:
            for candidate in [
                *self._package_initializers(item, source_root),
                item,
            ]:
                if candidate not in sources:
                    sources.append(candidate)
        sources.extend(
            resource for resource in resources if resource not in sources
        )
        planned: list[tuple[Path, Path, str]] = []
        for item in sources:
            relative = item.relative_to(source_root)
            destination = destination_root / relative
            role = (
                "entry point"
                if item == source
                else "local Python import"
                if item.suffix == ".py"
                else "declared resource"
            )
            planned.append((item, destination, role))

        conflicts = [
            destination
            for item, destination, _role in planned
            if destination.exists()
            and (
                not destination.is_file()
                or destination.read_bytes() != item.read_bytes()
            )
        ]
        existing = self.workspace.workflow_entry
        replacing = bool(conflicts) or (
            existing is not None and existing != incoming_entry
        )
        if replacing:
            self._emit_table(
                "Workflow replacement",
                [
                    ("Current", existing or "no configured workflow", "warning"),
                    ("Incoming", incoming_entry, None),
                    (
                        "Files overwritten",
                        len(conflicts),
                        "warning" if conflicts else None,
                    ),
                    (
                        "Meaning",
                        "the project manifest will identify the incoming workflow",
                        "warning",
                    ),
                ],
            )
            if not self._confirm_action(
                "Replace the project workflow? [y/N]: ",
                cancel_message="Workflow import cancelled; nothing was changed.",
                default=False,
            ):
                return

        copied: list[tuple[Path, str]] = []
        created: list[Path] = []
        overwritten: dict[Path, tuple[bytes, int]] = {}

        def restore_files() -> None:
            for path in reversed(created):
                try:
                    path.unlink()
                except OSError:
                    pass
            for path, (content, mode) in overwritten.items():
                try:
                    path.write_bytes(content)
                    path.chmod(mode)
                except OSError:
                    pass

        try:
            for item, destination, role in planned:
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    shutil.copy2(item, destination)
                    created.append(destination)
                elif destination.read_bytes() != item.read_bytes():
                    overwritten[destination] = (
                        destination.read_bytes(),
                        destination.stat().st_mode & 0o777,
                    )
                    shutil.copy2(item, destination)
                copied.append((destination, role))
        except OSError as exc:
            restore_files()
            raise SystemExit(
                f"Workflow import failed while copying files: {exc}"
            ) from exc

        imported_entry = (
            destination_root / source.relative_to(source_root)
        ).resolve()
        discovered = [
            candidate
            for candidate in self.workspace.discover_workflows()
            if (self.workspace.root / candidate.partition(":")[0]).resolve()
            == imported_entry
        ]
        self._emit_columns(
            "Imported workflow files",
            ("File", "Role"),
            [
                (
                    path.relative_to(self.workspace.root).as_posix(),
                    role,
                )
                for path, role in copied
            ],
        )
        if not discovered:
            restore_files()
            raise SystemExit(
                "Files were copied, but no top-level @workflow entry point "
                "was discovered in the imported source."
            )
        selected = next(
            (
                candidate
                for candidate in discovered
                if requested_name
                and candidate.rpartition(":")[2] == requested_name
            ),
            None,
        )
        if requested_name and selected is None:
            restore_files()
            raise SystemExit(
                f"Files were copied, but workflow {requested_name!r} was not "
                f"found. Available: {', '.join(discovered)}."
            )
        if selected is None and len(discovered) == 1:
            selected = discovered[0]
        if selected is None:
            restore_files()
            self._emit_columns(
                "Imported workflow entry points",
                ("Workflow", "Entry point"),
                [
                    (candidate.rpartition(":")[2], candidate)
                    for candidate in discovered
                ],
            )
            self._emit_next("workflow import PATH.py:WORKFLOW")
            return
        try:
            canonical, name = self._configure_workflow_spec(
                selected,
                replace=replacing,
            )
            if replacing:
                self.workspace.remove_implementation_lock()
        except Exception as exc:
            restore_files()
            raise SystemExit(
                "The files were copied, but the imported workflow could not "
                f"be loaded: {exc}"
            ) from exc
        self._emit_table(
            "Workflow imported",
            [
                ("Workflow", name, None),
                ("Entry point", canonical, None),
                ("Files", len(copied), "success"),
                ("Validation", "not run", "warning"),
                (
                    "Next",
                    "workflow show source · workflow show protocol · "
                    "workflow validate",
                    None,
                ),
            ],
        )

    def _configure_workflow_spec(
        self,
        selected: str,
        *,
        replace: bool = False,
    ) -> tuple[str, str]:
        from zippergen.serve import load_workflow_spec

        canonical = self.workspace.canonical_spec(
            selected,
            cwd=self.workspace.root,
        )
        project_path = str(self.workspace.root)
        sys.path.insert(0, project_path)
        try:
            workflow, _module = load_workflow_spec(
                self.workspace.absolute_spec(canonical)
            )
        finally:
            try:
                sys.path.remove(project_path)
            except ValueError:
                pass
        try:
            self.workspace.select_workflow(
                canonical,
                cwd=self.workspace.root,
                replace=replace,
            )
        except WorkspaceError as exc:
            raise SystemExit(str(exc)) from exc
        self._associate_implementation_workflow(canonical)
        return canonical, workflow.name

    def _resolve_local_module(
        self,
        base: Path,
    ) -> list[Path]:
        paths = [base.with_suffix(".py"), base / "__init__.py"]
        return [path.resolve() for path in paths if path.is_file()]

    def _local_python_dependencies(
        self,
        entry: Path,
        *,
        root: Path | None = None,
    ) -> list[Path]:
        root = (root or self.workspace.root).resolve()
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
        rows: list[tuple[object, ...]] = [
            (
                index,
                path,
                role,
            )
            for index, (path, role) in enumerate(records, start=1)
        ]
        self._emit_columns(
            "Workflow files",
            ("#", "File", "Role"),
            rows,
            right_aligned=frozenset({0}),
        )
        self._emit_table(
            "File discovery",
            [
                (
                    "Scope",
                    "entry point, statically imported local modules, and "
                    "declared resources",
                    "info",
                ),
                ("Next", "workflow show source [NUMBER|PATH]", None),
            ],
        )

    def show_workflow_source(self, args: list[str]) -> None:
        if len(args) > 1:
            raise SystemExit("Use workflow show source [NUMBER|PATH].")
        records = self._workflow_file_records()
        if not records:
            raise SystemExit("No source files were found for the project workflow.")
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
                    f"File is not part of the project workflow: {args[0]}. "
                    "Use 'workflow files' first."
                )
            selected = matches[0]
        path, role = selected
        target = self.workspace.root / path
        try:
            source = target.read_text(encoding="utf-8").rstrip()
        except (OSError, UnicodeDecodeError) as exc:
            raise SystemExit(f"Could not read workflow file {path}: {exc}") from exc
        self._emit_section_title(f"Source: {path} ({role})")
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
                *(item.label for item in WORKFLOW_VIEWS),
            ]
            view = str(
                self._select(
                    f"Inspect {workflow.name}",
                    choices,
                    prompt="Select workflow view",
                )
            ).lower()

        if view in {"authored source"}:
            self.show_workflow_source([])
            return
        selected_view = workflow_view_spec(view)
        if selected_view is None:
            available = ", ".join(item.command for item in WORKFLOW_VIEWS)
            raise SystemExit(f"View must be {available}.")
        if selected_view.participants == "one":
            names = self._agent_names(workflow)
            agent = (
                rest[0]
                if rest
                else self._select(
                    "Participants",
                    names,
                    prompt="Select participant",
                )
            )
            options = ViewOptions(
                detail=selected_view.detail,
                communications_only=selected_view.communications_only,
                agent=str(agent),
            )
            remembered = f"{selected_view.command} {agent}"
        elif selected_view.participants == "many":
            names = self._agent_names(workflow)
            selected = rest or self._select(
                "Participants",
                names,
                prompt="Select participants",
                allow_many=True,
            )
            assert isinstance(selected, list)
            options = ViewOptions(
                detail=selected_view.detail,
                communications_only=selected_view.communications_only,
                agents=tuple(selected),
            )
            remembered = selected_view.command + " " + " ".join(selected)
        else:
            options = ViewOptions(
                detail=selected_view.detail,
                communications_only=selected_view.communications_only,
            )
            remembered = selected_view.command
        try:
            data = workflow_view_data(workflow, module, options=options)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        self.workspace.update(last_view=remembered)
        self._emit_section_title(
            f"Workflow view · {workflow.name} · {remembered}"
        )
        self._emit(data["code"])
        self._emit()

    def validate(self) -> None:
        from zippergen.serve import _validate_workflow

        _current, workflow, module = self._current_context(
            purpose="validate it"
        )
        result = _validate_workflow(workflow, module)
        verdict = "valid" if result["valid"] else "invalid"
        summary = self._success if result["valid"] else self._error
        self._emit_section_title("Workflow validation")
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
        self._emit_table(
            "Validation result",
            [
                (
                    "Technical validation",
                    (
                        "passed for the current code"
                        if result["valid"]
                        else "failed for the current code"
                    ),
                    "success" if result["valid"] else "error",
                ),
                (
                    "Meaning",
                    "validation checks workflow structure and every local "
                    "projection",
                    "info",
                ),
                (
                    "Next",
                    (
                        "workflow diff · run · deploy"
                        if result["valid"]
                        else "workflow diff · workflow implement"
                    ),
                    None,
                ),
            ],
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

    def _assistant_action_lifelines(self, workflow, module) -> dict[str, list[str]]:
        """Return coding-assistant actions per participant, in protocol order."""

        model = workflow_semantics(workflow, module)
        actions: dict[str, list[str]] = {}
        sites = model.get("action_sites") or []
        if isinstance(sites, list):
            for site in sites:
                if not isinstance(site, dict) or site.get("kind") != "assistant":
                    continue
                name = str(site.get("lifeline"))
                action = str(site.get("action"))
                actions.setdefault(name, [])
                if action not in actions[name]:
                    actions[name].append(action)
        ordered = self._agent_names(workflow)
        return {name: actions[name] for name in ordered if name in actions}

    def _check_workflow_assistants(self, workflow, module) -> None:
        """Refuse to deploy assistant actions this machine cannot execute.

        A deployment runs on the machine that installs it, so a missing
        executable is a known failure rather than an unknown one.
        """

        active = self._assistant_action_lifelines(workflow, module)
        if not active:
            return
        configured = str(self._global_settings().get("assistant") or "codex")
        if shutil.which(configured) is not None:
            self._emit_table(
                "Assistant checks",
                [("Status", f"{configured} found", "success")],
            )
            return
        targets = ", ".join(
            f"{participant}.{action}"
            for participant, actions in active.items()
            for action in actions
        )
        self._emit_table(
            "Assistant checks",
            [
                ("Status", "blocked", "error"),
                ("Reason", f"{configured!r} was not found on PATH", "error"),
                ("Assistant actions", targets, None),
                (
                    "Next",
                    f"install {configured}, or use settings set assistant "
                    f"{'claude' if configured == 'codex' else 'codex'}",
                    None,
                ),
            ],
        )
        raise SystemExit(
            f"This workflow has coding-assistant actions, but {configured!r} "
            "is not installed on this machine."
        )

    def _llm_action_targets(self, workflow, module) -> dict[str, tuple[str, str]]:
        """Return canonical ``Participant.action`` targets in protocol order."""

        return {
            f"{participant}.{action}": (participant, action)
            for participant, actions in self._llm_action_lifelines(
                workflow, module
            ).items()
            for action in actions
        }

    def _human_action_lifelines(self, workflow, module) -> dict[str, list[str]]:
        model = workflow_semantics(workflow, module)
        actions: dict[str, list[str]] = {}
        sites = model.get("action_sites") or []
        if isinstance(sites, list):
            for site in sites:
                if (
                    not isinstance(site, dict)
                    or site.get("kind") != "human"
                ):
                    continue
                participant = str(site.get("lifeline"))
                action = str(site.get("action"))
                actions.setdefault(participant, [])
                if action not in actions[participant]:
                    actions[participant].append(action)
        ordered = self._agent_names(workflow)
        return {
            name: actions[name] for name in ordered if name in actions
        }

    def _human_action_targets(
        self, workflow, module
    ) -> dict[str, tuple[str, str]]:
        return {
            f"{participant}.{action}": (participant, action)
            for participant, actions in self._human_action_lifelines(
                workflow, module
            ).items()
            for action in actions
        }

    def _human_connector_factory(
        self,
        current: str,
        workflow,
        module,
    ):
        snapshot, environment = self._workflow_connector_runtime(
            workflow_spec=current,
            workflow=workflow,
            module=module,
        )
        return self._human_connector_factory_from_snapshot(
            snapshot,
            environment,
        )

    def _human_connector_factory_from_snapshot(
        self,
        snapshot: dict[str, object],
        environment: dict[str, str],
    ):
        human_records = [
            value
            for value in snapshot.values()
            if isinstance(value, dict) and value.get("type") == "human"
        ]
        if not human_records:
            return None
        routes: dict[str, dict[str, object]] = {}
        route_assignments: dict[str, str] = {}
        token: str | None = None
        for record in human_records:
            configuration = str(record.get("configuration") or "")
            target = str(record.get("target") or "")
            token_env = str(record.get("token_env") or "")
            if not configuration or not target or not token_env:
                raise SystemExit(
                    "Recorded human connector routing is incomplete. Start a "
                    "new run after checking connector assignments."
                )
            candidate = environment.get(token_env)
            if not candidate:
                raise SystemExit(
                    f"Private credential for connector {configuration} is "
                    "unavailable. Configure its provider, then resume."
                )
            if token is not None and candidate != token:
                raise SystemExit(
                    "A development run cannot use multiple Telegram provider "
                    "credentials for human delivery."
                )
            token = candidate
            routes[configuration] = dict(record)
            route_assignments[target] = configuration
        assert token is not None

        def factory(store_path: str):
            from zippergen.telegram_notify import (
                TelegramBotClient,
                TelegramDeploymentNotifier,
            )

            return TelegramDeploymentNotifier(
                store_path=store_path,
                client=TelegramBotClient(token),
                routes=routes,
                assignments=route_assignments,
            )

        return factory

    def _check_workflow_connectors(
        self,
        current: str,
        workflow,
        module,
        *,
        for_run: bool = False,
    ) -> None:
        from zippergen.connectors import connector_requirements_from_module

        assignments = self.workspace.connector_assignment_profile(current)
        requirements = connector_requirements_from_module(module)
        bindings = self.workspace.connector_binding_profile(current)
        missing = [
            requirement.name
            for requirement in requirements
            if requirement.required and requirement.name not in bindings
        ]
        if missing:
            raise SystemExit(
                "Required connector bindings are missing: "
                + ", ".join(missing)
                + ". Use 'connector setup'."
            )
        configurations = self.workspace.connector_configurations()
        google_requirements: list[tuple[str, str]] = []
        for requirement in requirements:
            configuration_name = bindings.get(requirement.name)
            if configuration_name is None:
                continue
            configuration = configurations.get(configuration_name)
            if configuration is None:
                raise SystemExit(
                    f"Connector {requirement.name} references missing "
                    f"configuration {configuration_name}."
                )
            if configuration.get("kind") != requirement.kind:
                raise SystemExit(
                    f"Connector {requirement.name} requires "
                    f"{requirement.kind}, but {configuration_name} is "
                    f"{configuration.get('kind') or 'unknown'}."
                )
            if requirement.kind in {"gmail", "google-sheets"}:
                google_requirements.append(
                    (requirement.kind, requirement.access)
                )
        if google_requirements:
            from zippergen.google_auth import google_scope_names

            required_scopes = self._google_scopes_for_requirements(
                google_requirements
            )
            profile = self.workspace.connector_provider_profiles().get(
                "google"
            )
            granted_scopes = self._google_profile_granted_scopes(profile)
            if not granted_scopes:
                raise SystemExit(
                    "The stored Google credential predates scope recording "
                    "and must be re-created with "
                    "'connector provider configure google'."
                )
            if not self._google_scopes_cover(
                granted_scopes, required_scopes
            ):
                missing = [
                    name
                    for scope, name in zip(
                        required_scopes,
                        google_scope_names(required_scopes),
                        strict=True,
                    )
                    if not self._google_scopes_cover(
                        granted_scopes, (scope,)
                    )
                ]
                raise SystemExit(
                    "Google authorization does not cover this workflow: "
                    + ", ".join(missing)
                    + ". Use 'connector provider configure google'."
                )
        names = list(dict.fromkeys([
            *assignments["lifelines"].values(),
            *assignments["actions"].values(),
            *bindings.values(),
        ]))
        if not names:
            return
        failed = [
            name for name in names
            if not self._check_connector_configuration(name)
        ]
        if failed:
            context = "Run" if for_run else "Connector assignment"
            raise SystemExit(
                f"{context} stopped because these connector configurations "
                f"are unavailable: {', '.join(failed)}."
            )
        self._success(
            "All connector configurations used by this workflow are reachable."
        )


    def show_runs(self) -> None:
        runs = self.workspace.list_runs()
        if not runs:
            self._emit_table(
                "Development runs",
                [("Status", "none", "warning")],
            )
            return
        current = self.workspace.current_run_id
        self._emit_columns(
            "Development runs",
            ("Current", "Run", "Status", "Workflow"),
            [
                (
                    "●" if record["run_id"] == current else "",
                    record["run_id"],
                    record["status"],
                    record["workflow_spec"],
                )
                for record in runs
            ],
        )
        self._emit_next("run inspect · resume · run")

    @staticmethod
    def _execution_age(updated_at: float | None) -> str:
        if updated_at is None:
            return "—"
        seconds = max(0, int(time.time() - updated_at))
        if seconds < 2:
            return "just now"
        if seconds < 60:
            return f"{seconds}s"
        minutes = seconds // 60
        if minutes < 60:
            return f"{minutes}m"
        hours = minutes // 60
        if hours < 48:
            return f"{hours}h"
        return f"{hours // 24}d"

    def _inspect_execution(
        self,
        *,
        workflow,
        store: str | Path,
        source_rows: list[tuple[str, object, StatusKind | None]],
        participant: str | None,
        next_commands: str,
        projection_indent: int = 0,
    ) -> None:
        from zippergen.execution_inspection import (
            default_focus,
            participant_positions,
            read_execution_states,
            state_label,
        )
        from zippergen.view import render_local_projection_with_pointers

        raw_states = read_execution_states(store)
        positions = participant_positions(workflow, raw_states)
        names = [position.participant for position in positions]
        if participant is not None:
            matched = next(
                (
                    name
                    for name in names
                    if name.casefold() == participant.casefold()
                ),
                None,
            )
            if matched is None:
                raise SystemExit(
                    f"Unknown participant {participant!r}. Available: "
                    f"{', '.join(names) or 'none'}."
                )
            focus = matched
        else:
            focus = default_focus(positions)
        observation_kind: StatusKind = "success" if raw_states else "warning"
        self._emit_table(
            "Execution context",
            [
                *source_rows,
                ("Store", store, None),
                (
                    "Observation",
                    (
                        f"{len(raw_states)} participant position(s) recorded"
                        if raw_states
                        else "no position data; the run may not have started "
                        "or may predate execution inspection"
                    ),
                    observation_kind,
                ),
                (
                    "Privacy",
                    "workflow variables and action inputs are not displayed",
                    "info",
                ),
            ],
        )
        status_kind = {
            "done": "success",
            "failed": "error",
            "cancelled": "warning",
            "waiting_human": "warning",
            "waiting_receive": "info",
            "running_model": "info",
            "running_assistant": "info",
            "running_effect": "info",
            "running_action": "info",
            "running": "info",
            "blocked": "warning",
            "not_started": "warning",
        }
        rows = []
        for position in positions:
            kind = cast(
                StatusKind,
                status_kind.get(position.state, "info"),
            )
            rows.append(
                (
                    "▶" if position.participant == focus else "",
                    position.participant,
                    f"{self._status_mark(kind)} {state_label(position.state)}",
                    position.location,
                    self._execution_age(position.updated_at),
                )
            )
        self._emit_columns(
            "Participant positions",
            ("Focus", "Participant", "State", "Current position", "Elapsed"),
            rows,
        )
        if focus is not None:
            selected = next(
                position
                for position in positions
                if position.participant == focus
            )
            self._emit()
            self._emit_section_title(f"{focus} · live local projection")
            self._emit(
                render_local_projection_with_pointers(
                    workflow,
                    focus,
                    selected.locators,
                    indent=projection_indent,
                )
            )
        self._emit_next(next_commands)

    @staticmethod
    def _inspection_options(
        args: list[str],
        *,
        max_positionals: int,
        usage: str,
    ) -> tuple[list[str], bool]:
        positionals: list[str] = []
        watch = False
        for value in args:
            if value.casefold() == "--watch":
                if watch:
                    raise SystemExit(f"Use {usage}.")
                watch = True
            elif value.startswith("-"):
                raise SystemExit(
                    f"Unknown inspection option {value!r}. Use {usage}."
                )
            else:
                positionals.append(value)
        if len(positionals) > max_positionals:
            raise SystemExit(f"Use {usage}.")
        return positionals, watch

    def _capture_watch_frame(
        self,
        render_once: Callable[[], None],
        *,
        command: str,
        subject: str,
    ) -> str:
        """Capture one ordinary Studio rendering for a full-screen display."""

        lines: list[str] = []
        original_output = self._renderer.output

        def capture(value: str) -> None:
            text = str(value)
            lines.extend(text.splitlines() if text else [""])

        self._renderer.output = capture
        try:
            self._emit_studio_banner(command)
            self._info(
                "Refreshing once per second. Press Ctrl-C to close this "
                f"view. The {subject} will keep running."
            )
            render_once()
        finally:
            self._renderer.output = original_output
        return "\n".join(lines)

    @staticmethod
    def _run_watch_display(frame_provider: Callable[[], str]) -> bool:
        """Run a differential full-screen display until Ctrl-C."""

        state = {"frame": frame_provider()}
        failure: list[BaseException] = []
        last_refresh = time.monotonic()
        bindings = KeyBindings()

        @bindings.add("c-c")
        def close_watch(event) -> None:
            event.app.exit(result="interrupted")

        def pointer_position() -> Point:
            pointer_lines = [
                index
                for index, line in enumerate(state["frame"].splitlines())
                if "▶" in line
            ]
            return Point(x=0, y=pointer_lines[-1] if pointer_lines else 0)

        control = FormattedTextControl(
            text=lambda: ANSI(state["frame"]),
            focusable=True,
            show_cursor=False,
            get_cursor_position=pointer_position,
        )

        def refresh_frame(application: Application[str]) -> None:
            nonlocal last_refresh
            now = time.monotonic()
            if now - last_refresh < _INSPECTION_WATCH_SECONDS * 0.9:
                return
            try:
                state["frame"] = frame_provider()
            except BaseException as exc:
                failure.append(exc)
                application.exit(result="failed")
                return
            last_refresh = now

        application: Application[str] = Application(
            layout=Layout(
                Window(
                    content=control,
                    wrap_lines=True,
                    always_hide_cursor=True,
                )
            ),
            key_bindings=bindings,
            full_screen=True,
            mouse_support=False,
            refresh_interval=_INSPECTION_WATCH_SECONDS,
            before_render=refresh_frame,
        )
        result = application.run()
        if failure:
            raise failure[0]
        return result == "interrupted"

    def _watch_execution(
        self,
        render_once: Callable[[], None],
        *,
        command: str,
        subject: str,
    ) -> None:
        if not self._prompt_toolkit_enabled:
            raise SystemExit(
                "--watch requires an interactive terminal. Open Studio normally "
                f"and use '{command}'."
            )
        interrupted = self._run_watch_display(
            lambda: self._capture_watch_frame(
                render_once,
                command=command,
                subject=subject,
            )
        )
        if interrupted:
            self._info(
                f"Stopped watching. The {subject} was not interrupted."
            )

    def inspect_run(self, args: list[str]) -> None:
        positionals, watch = self._inspection_options(
            args,
            max_positionals=1,
            usage="run inspect [PARTICIPANT] [--watch]",
        )
        record = self.workspace.current_run()
        if record is None:
            raise SystemExit(
                "There is no current development run. Start one with 'run'."
            )
        run_id = str(record.get("run_id") or "")
        participant = positionals[0] if positionals else None

        def render_once() -> None:
            current = self.workspace.load_run(run_id)
            workflow_spec = str(current.get("workflow_spec") or "")
            if not workflow_spec:
                raise SystemExit("The current run has no recorded workflow.")
            from zippergen.serve import load_workflow_spec

            workflow, _module = load_workflow_spec(
                self.workspace.absolute_spec(workflow_spec)
            )
            self._inspect_execution(
                workflow=workflow,
                store=str(current.get("store") or ""),
                source_rows=[
                    ("Run", current.get("run_id") or "unknown", None),
                    ("Workflow", workflow_spec, None),
                    ("Run status", current.get("status") or "unknown", None),
                ],
                participant=participant,
                next_commands=(
                    f"run inspect {participant} · runs · resume"
                    if participant
                    else "run inspect PARTICIPANT · runs · resume"
                ),
                projection_indent=1 if watch else 0,
            )

        if watch:
            command = "run inspect"
            if participant:
                command += f" {participant}"
            command += " --watch"
            self._watch_execution(
                render_once,
                command=command,
                subject="development run",
            )
        else:
            render_once()

    @staticmethod
    def _store_updated(value: float | None) -> str:
        if value is None:
            return "—"
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(value))

    def _store_record(self, selector: str | None = None):
        from zippergen.studio_stores import resolve_store

        try:
            return resolve_store(self.workspace, selector)
        except WorkspaceError as exc:
            raise SystemExit(str(exc)) from exc

    def _run_store_record(self):
        run = self.workspace.current_run()
        if run is None:
            raise SystemExit(
                "There is no current development run. Start one with 'run'."
            )
        store = str(run.get("store") or "")
        if not store:
            raise SystemExit("The current development run has no durable state.")
        record = self._store_record(store)
        return record

    def _emit_human_task_detail(self, task: dict[str, object]) -> None:
        """Render the evidence and expected response before a human decision."""

        raw_spec = task.get("spec")
        spec = raw_spec if isinstance(raw_spec, dict) else {}
        raw_rendered = spec.get("rendered")
        rendered = raw_rendered if isinstance(raw_rendered, dict) else {}
        output_type = str(spec.get("output_type") or "str")
        submit_label = str(spec.get("submit_label") or "Approve")
        cancel_label = str(spec.get("cancel_label") or "Reject")
        decision = (
            f"{submit_label} (yes) / {cancel_label} (no)"
            if output_type == "bool"
            else f"provide {spec.get('output') or 'a response'} ({output_type})"
        )
        self._emit_table(
            "Human decision",
            [
                ("Task", task.get("task_id") or "unknown", None),
                ("Participant", task.get("role") or "unknown", None),
                ("Action", task.get("action") or "unknown", None),
                ("Decision", decision, "warning"),
            ],
        )

        content_found = False
        for title, key in (
            ("Instruction", "instruction"),
            ("Context", "context"),
            ("Suggested response", "prefill"),
        ):
            value = rendered.get(key)
            if value is None or str(value).strip() == "":
                continue
            content_found = True
            self._emit_section_title(title)
            for line in self._wrapped_lines(
                value,
                max(1, self._renderer.output_columns() - 2),
            ):
                self._emit(f"  {line}")
            self._emit()

        if not content_found:
            inputs = task.get("inputs")
            if isinstance(inputs, dict) and inputs:
                self._emit_section_title("Inputs")
                rendered_inputs = json.dumps(
                    inputs,
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
                for line in rendered_inputs.splitlines():
                    self._emit(f"  {line}")
                self._emit()
            else:
                self._warning(
                    "This task contains no rendered instruction, context, or inputs."
                )
                self._emit()

    def _show_pending_tasks(
        self,
        record,
        *,
        deployment_name: str | None = None,
    ) -> None:
        if not record.exists:
            owner = (
                f"deployment {deployment_name}"
                if deployment_name
                else "the current development run"
            )
            raise SystemExit(f"Durable state does not exist yet for {owner}.")
        from zippergen.serve import _load_human_tasks

        tasks = cast(
            list[dict[str, object]],
            _load_human_tasks(str(record.path), status="pending"),
        )
        if not tasks:
            self._emit_table(
                "Pending human tasks",
                [
                    ("Store", record.name, None),
                    ("Status", "none", "success"),
                ],
            )
            return
        self._emit_columns(
            "Pending human tasks",
            ("Task", "Participant", "Action", "Updated"),
            [
                (
                    task["task_id"],
                    task["role"],
                    task["action"],
                    self._store_updated(float(str(task["updated_at"]))),
                )
                for task in tasks
            ],
        )
        for task in tasks:
            self._emit_human_task_detail(task)
        self._emit_next(
            f"deploy approve {deployment_name}"
            if deployment_name
            else "run approve TASK_ID"
        )

    def _approve_pending_task(
        self,
        record,
        args: list[str],
        *,
        deployment_name: str | None = None,
    ) -> None:
        if len(args) > 2:
            raise SystemExit(
                (
                    "Use deploy approve [NAME]."
                    if deployment_name
                    else "Use run approve [TASK_ID] [yes|no|VALUE]."
                )
            )
        if not record.exists:
            owner = (
                f"deployment {deployment_name}"
                if deployment_name
                else "the current development run"
            )
            raise SystemExit(f"Durable state does not exist yet for {owner}.")
        from zippergen.store import load_human_task, open_store
        from zippergen.serve import _store_status

        pending = _store_status(str(record.path)).get(
            "pending_human_tasks"
        )
        pending_ids = [
            str(item.get("task_id"))
            for item in pending
            if isinstance(item, dict) and item.get("task_id")
        ] if isinstance(pending, list) else []
        if not args:
            if not pending_ids:
                raise SystemExit(
                    f"No pending human tasks exist in {record.name}."
                )
            task_id = (
                pending_ids[0]
                if len(pending_ids) == 1
                else str(
                    self._select(
                        "Pending human tasks",
                        pending_ids,
                        prompt="Select a task",
                    )
                )
            )
            supplied_value = ""
        else:
            task_id = args[0]
            supplied_value = args[1] if len(args) == 2 else ""
        connection = open_store(str(record.path))
        try:
            task = load_human_task(connection, task_id)
        finally:
            connection.close()
        if task is None or task.get("status") != "pending":
            tasks_command = (
                f"deploy tasks {deployment_name}"
                if deployment_name
                else "run tasks"
            )
            raise SystemExit(
                f"Pending human task not found: {task_id}. "
                f"Use '{tasks_command}'."
            )
        self._emit_human_task_detail(cast(dict[str, object], task))
        spec = task.get("spec") or {}
        output_type = (
            str(spec.get("output_type") or "str")
            if isinstance(spec, dict)
            else "str"
        )
        value = supplied_value
        if not value:
            if output_type == "bool":
                submit_label = str(spec.get("submit_label") or "Approve")
                cancel_label = str(spec.get("cancel_label") or "Reject")
                value = self.input(
                    f"Decision — {submit_label} or {cancel_label}? [y/n]: "
                ).strip()
            else:
                value = self.input(
                    f"Response for {task['role']}.{task['action']}: "
                ).strip()
        lowered = value.casefold()
        arguments = [
            "approve",
            "--store",
            str(record.path),
            "--task",
            str(task["task_id"]),
        ]
        if output_type == "bool":
            if lowered in {"y", "yes", "true", "approve", "approved"}:
                arguments.append("--yes")
            elif lowered in {"n", "no", "false", "reject", "rejected"}:
                arguments.append("--no")
            else:
                raise SystemExit("Enter yes or no for this human decision.")
        else:
            if not value:
                raise SystemExit("A response value is required.")
            arguments.extend(["--value", value])
        rc = self._run_project_cli(arguments)
        if rc != 0:
            raise SystemExit(
                f"Could not complete human task {task['task_id']}."
            )
        self._success(f"Completed human task {task['task_id']}.")
        self._show_pending_tasks(
            record,
            deployment_name=deployment_name,
        )

    def _show_recent_trace(self, record, *, owner: str) -> None:
        if not record.exists:
            raise SystemExit(f"Durable state does not exist yet for {owner}.")
        from zippergen.serve import _load_trace_events, _trace_summary

        events = _load_trace_events(str(record.path), after_rowid=0, limit=30)
        if not events:
            self._emit_table(
                "Recent durable events",
                [("Execution", owner, None), ("Events", "none", "warning")],
            )
            return
        self._emit_columns(
            "Recent durable events",
            ("#", "Event"),
            [
                (
                    event.get("rowid"),
                    _trace_summary(
                        str(event.get("role") or "—"),
                        event.get("event"),
                    ),
                )
                for event in events
            ],
            right_aligned=frozenset({0}),
        )

    def manage_run_state(self, action: str, args: list[str]) -> None:
        record = self._run_store_record()
        if action == "tasks":
            if args:
                raise SystemExit("Use run tasks.")
            self._show_pending_tasks(record)
            return
        if action == "approve":
            self._approve_pending_task(record, args)
            return
        if action == "trace":
            if args:
                raise SystemExit("Use run trace.")
            run = self.workspace.current_run()
            run_id = str(run.get("run_id") or "current run") if run else "current run"
            self._show_recent_trace(record, owner=f"run {run_id}")
            return
        raise SystemExit("Use run inspect, run tasks, run approve, or run trace.")

    def _run_project_cli(
        self,
        arguments: list[str],
        *,
        cwd: Path | None = None,
    ) -> int:
        from zippergen.serve import main

        previous = Path.cwd()
        try:
            os.chdir(cwd or self.workspace.root)
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
        """Offer selected development provider keys to the deployment."""

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
        available = self.workspace.development_provider_environment(model_specs)
        arguments: list[str] = []
        local_base_url = available.get("OLLAMA_BASE_URL")
        if local_base_url:
            arguments.extend(
                ["--provider-env", f"OLLAMA_BASE_URL={local_base_url}"]
            )

        if not selected_secret_names:
            return arguments

        existing: dict[str, str] = {}
        if _deployment_profile_path(name).exists():
            existing = _load_deployment_secrets(_load_deployment_profile(name))

        declared_fields = {
            field.target_name: field
            for field in spec.fields
            if field.secret and field.target_name in selected_secret_names
        }
        retained_names = selected_secret_names & existing.keys()
        reusable_names = (
            selected_secret_names & available.keys()
        ) - existing.keys()
        if not retained_names and not reusable_names:
            return arguments

        if retained_names:
            for secret_name in sorted(retained_names):
                field = declared_fields.get(secret_name)
                if field is None:
                    continue
                arguments.extend(
                    ["--set", f"{field.name}={existing[secret_name]}"]
                )
            noun = "credential" if len(retained_names) == 1 else "credentials"
            self._success(
                f"Keeping {len(retained_names)} existing deployment {noun}; "
                "values remain hidden."
            )

        if not reusable_names:
            return arguments

        secret_names = sorted(reusable_names)
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
            undeclared = [
                secret_name
                for secret_name in secret_names
                if secret_name not in declared_fields
            ]
            for secret_name in undeclared:
                value = self.secret_input(
                    f"{secret_name} for deployment {name}: "
                ).strip()
                if not value:
                    raise SystemExit(
                        f"{secret_name} is required by the selected model; "
                        "deployment was not started."
                    )
                arguments.extend(
                    ["--provider-secret", f"{secret_name}={value}"]
                )
            return arguments

        for secret_name in secret_names:
            # Studio calls serve.main() in-process. This is not an OS command
            # line, and neither the argument nor its value is rendered.
            arguments.extend(
                [
                    "--provider-secret",
                    f"{secret_name}={available[secret_name]}",
                ]
            )
            field = declared_fields.get(secret_name)
            if field is not None:
                arguments.extend(
                    ["--set", f"{field.name}={available[secret_name]}"]
                )
        noun = "credential" if len(secret_names) == 1 else "credentials"
        self._success(
            f"Reusing {len(secret_names)} configured {noun}; values remain "
            "hidden and deployment-scoped."
        )
        return arguments

    def _deployment_connector_arguments(
        self,
        *,
        workflow_spec: str,
        workflow,
        module,
    ) -> list[str]:
        """Validate bindings and serialize connector references for deployment."""

        from zippergen.connectors import connector_requirements_from_module
        from zippergen.serve import _slug

        requirements = connector_requirements_from_module(module)
        human_assignments = self.workspace.connector_assignment_profile(
            workflow_spec
        )
        if (
            not requirements
            and not human_assignments["lifelines"]
            and not human_assignments["actions"]
        ):
            return []
        bindings = self.workspace.connector_binding_profile(workflow_spec)
        configurations = self.workspace.connector_configurations()
        missing = [
            item.name
            for item in requirements
            if item.required and item.name not in bindings
        ]
        if missing:
            raise SystemExit(
                "Required connector bindings are missing: "
                + ", ".join(missing)
                + ". Use connector bind."
            )
        google_requirements = [
            (item.kind, item.access)
            for item in requirements
            if (
                item.name in bindings
                and item.kind in {"gmail", "google-sheets"}
            )
        ]
        if google_requirements:
            from zippergen.google_auth import google_scope_names

            required_scopes = self._google_scopes_for_requirements(
                google_requirements
            )
            profile = self.workspace.connector_provider_profiles().get(
                "google"
            )
            granted_scopes = self._google_profile_granted_scopes(profile)
            if not granted_scopes:
                raise SystemExit(
                    "Google authorization has no verified granted-scope "
                    "record. Use 'connector provider configure google' before "
                    "running or deploying."
                )
            if not self._google_scopes_cover(
                granted_scopes, required_scopes
            ):
                missing_scopes = [
                    name
                    for scope, name in zip(
                        required_scopes,
                        google_scope_names(required_scopes),
                        strict=True,
                    )
                    if not self._google_scopes_cover(
                        granted_scopes, (scope,)
                    )
                ]
                raise SystemExit(
                    "Google authorization is missing "
                    + ", ".join(missing_scopes)
                    + ". Use 'connector provider configure google' before "
                    "running or deploying."
                )

        snapshot: dict[str, dict[str, object]] = {}
        arguments: list[str] = []
        connector_secrets: dict[str, str] = {}

        def configuration_record(
            configuration_name: str,
        ) -> tuple[dict[str, str], str, dict[str, str]]:
            configuration = configurations.get(configuration_name)
            if configuration is None:
                raise SystemExit(
                    f"Connector assignment references missing configuration "
                    f"{configuration_name}."
                )
            if configuration.get("check_status") != "available":
                raise SystemExit(
                    f"Connector {configuration_name} has not passed its latest "
                    f"check. Use 'connector config check "
                    f"{configuration_name}'."
                )
            provider = str(
                configuration.get("provider")
                or configuration.get("kind")
                or ""
            )
            secrets: dict[str, str] = {}
            if provider == "telegram":
                token = self.workspace.connector_provider_secret(
                    provider, "bot_token"
                ) or self.workspace.connector_secret(
                    configuration_name, "bot_token"
                )
                if not token:
                    raise SystemExit(
                        "Telegram provider token is missing. Use "
                        "'connector provider configure telegram'."
                    )
                secrets["bot_token"] = token
            elif provider == "google":
                authorized_user = (
                    self.workspace.connector_provider_secret(
                        provider, "authorized_user_json"
                    )
                )
                if not authorized_user:
                    raise SystemExit(
                        "Google authorization is missing. Use "
                        "'connector provider configure google'."
                    )
                secrets["authorized_user_json"] = authorized_user
            return configuration, provider, secrets

        human_sites = self._human_action_lifelines(workflow, module)
        human_targets = self._human_action_targets(workflow, module)
        for target, configuration_name in [
            *human_assignments["lifelines"].items(),
            *human_assignments["actions"].items(),
        ]:
            if target not in human_sites and target not in human_targets:
                raise SystemExit(
                    f"Connector assignment target no longer exists: {target}."
                )
            configuration, provider, secrets = configuration_record(
                configuration_name
            )
            if provider != "telegram":
                raise SystemExit(
                    f"Human target {target} needs a human-delivery connector, "
                    f"but {configuration_name} uses {provider or 'none'}."
                )
            token_env = (
                "ZIPPERGEN_CONNECTOR_"
                + _slug(provider).replace("-", "_").upper()
                + "_TOKEN"
            )
            connector_secrets[token_env] = secrets["bot_token"]
            snapshot[f"human:{target}"] = {
                "type": "human",
                "target": target,
                "participant": target.partition(".")[0],
                "action": target.partition(".")[2] or None,
                "kind": provider,
                "provider": provider,
                "configuration": configuration_name,
                "chat_id": configuration.get("chat_id"),
                "channel": configuration.get("channel")
                or f"telegram:{configuration_name}",
                "token_env": token_env,
            }

        for requirement in requirements:
            configuration_name = bindings.get(requirement.name)
            if configuration_name is None:
                continue
            configuration, provider, secrets = configuration_record(
                configuration_name
            )
            configuration_kind = str(configuration.get("kind") or "")
            if configuration_kind != requirement.kind:
                raise SystemExit(
                    f"Connector binding {requirement.name} requires "
                    f"{requirement.kind}, but {configuration_name} is "
                    f"{configuration_kind or 'unknown'}."
                )
            record: dict[str, object] = {
                **requirement.as_dict(),
                "provider": provider,
                "configuration": configuration_name,
                "channel": configuration.get("channel") or requirement.name,
            }
            if requirement.kind == "telegram":
                token = secrets.get("bot_token")
                if not token:
                    raise SystemExit(
                        f"Telegram token is missing for {configuration_name}."
                    )
                token_env = (
                    "ZIPPERGEN_CONNECTOR_"
                    + _slug(configuration_name).replace("-", "_").upper()
                    + "_TOKEN"
                )
                record.update(
                    {
                        "chat_id": configuration.get("chat_id"),
                        "token_env": token_env,
                    }
                )
                connector_secrets[token_env] = token
            elif requirement.kind == "google-sheets":
                credential = secrets.get("authorized_user_json")
                if provider != "google" or not credential:
                    raise SystemExit(
                        f"{configuration_name} needs a configured Google "
                        "provider."
                    )
                credential_env = (
                    "ZIPPERGEN_CONNECTOR_"
                    + _slug(configuration_name).replace("-", "_").upper()
                    + "_GOOGLE_CREDENTIAL"
                )
                record.update(
                    {
                        "spreadsheet_id": configuration.get(
                            "spreadsheet_id"
                        ),
                        "tab": configuration.get("tab"),
                        "credential_env": credential_env,
                    }
                )
                connector_secrets[credential_env] = credential
            elif requirement.kind == "gmail":
                credential = secrets.get("authorized_user_json")
                if provider != "google" or not credential:
                    raise SystemExit(
                        f"{configuration_name} needs a configured Google "
                        "provider."
                    )
                credential_env = (
                    "ZIPPERGEN_CONNECTOR_"
                    + _slug(configuration_name).replace("-", "_").upper()
                    + "_GOOGLE_CREDENTIAL"
                )
                record.update(
                    {
                        "account": configuration.get("account") or "me",
                        "query": configuration.get("query")
                        or "is:unread in:inbox",
                        "credential_env": credential_env,
                    }
                )
                connector_secrets[credential_env] = credential
            snapshot[f"requirement:{requirement.name}"] = record
        for secret_name, secret in sorted(connector_secrets.items()):
            arguments.extend(
                ["--connector-secret", f"{secret_name}={secret}"]
            )
        arguments.extend(
            [
                "--connectors-json",
                json.dumps(snapshot, sort_keys=True),
            ]
        )
        return arguments

    def _workflow_connector_environment(
        self,
        *,
        workflow_spec: str,
        workflow,
        module,
    ) -> dict[str, str]:
        """Build the same private connector context used by deployments."""

        _snapshot, environment = self._workflow_connector_runtime(
            workflow_spec=workflow_spec,
            workflow=workflow,
            module=module,
        )
        return environment

    def _workflow_connector_runtime(
        self,
        *,
        workflow_spec: str,
        workflow,
        module,
    ) -> tuple[dict[str, object], dict[str, str]]:
        """Return durable non-secret routing plus its current private values."""

        arguments = self._deployment_connector_arguments(
            workflow_spec=workflow_spec,
            workflow=workflow,
            module=module,
        )
        environment: dict[str, str] = {}
        snapshot: dict[str, object] = {}
        index = 0
        while index < len(arguments):
            option = arguments[index]
            index += 1
            if index >= len(arguments):
                break
            value = arguments[index]
            index += 1
            if option == "--connectors-json":
                environment["ZIPPERGEN_CONNECTORS_JSON"] = value
                raw = json.loads(value)
                if not isinstance(raw, dict):
                    raise SystemExit("Connector routing snapshot is invalid.")
                snapshot = {
                    str(name): dict(record)
                    for name, record in raw.items()
                    if isinstance(record, dict)
                }
            elif option == "--connector-secret":
                name, separator, secret = value.partition("=")
                if separator and name:
                    environment[name] = secret
        return snapshot, environment

    def _connector_environment_from_snapshot(
        self,
        snapshot: dict[str, object],
    ) -> dict[str, str]:
        """Resolve current private credentials for recorded connector routing."""

        if not snapshot:
            return {}
        environment = {
            "ZIPPERGEN_CONNECTORS_JSON": json.dumps(
                snapshot,
                sort_keys=True,
            )
        }
        for value in snapshot.values():
            if not isinstance(value, dict):
                continue
            provider = str(value.get("provider") or value.get("kind") or "")
            configuration = str(value.get("configuration") or "")
            token_env = str(value.get("token_env") or "")
            credential_env = str(value.get("credential_env") or "")
            if token_env:
                secret = self.workspace.connector_provider_secret(
                    provider,
                    "bot_token",
                ) or self.workspace.connector_secret(
                    configuration,
                    "bot_token",
                )
                if not secret:
                    raise SystemExit(
                        f"Private credential for connector {configuration} is "
                        "unavailable. Configure its provider, then resume."
                    )
                environment[token_env] = secret
            if credential_env:
                secret = self.workspace.connector_provider_secret(
                    provider,
                    "authorized_user_json",
                )
                if not secret:
                    raise SystemExit(
                        f"Private Google authorization for connector "
                        f"{configuration} is unavailable. Use 'connector "
                        "provider configure google', then resume."
                    )
                environment[credential_env] = secret
        return environment

    def _implementation_record(
        self,
        workflow_spec: str,
    ) -> dict[str, object] | None:
        """Return the newest implementation tied to this workflow."""

        canonical = self.workspace.canonical_spec(
            workflow_spec,
            cwd=self.workspace.root,
        )
        for record in self.workspace.list_requests():
            status = str(record.get("status") or "")
            if status not in {
                "implemented",
                "awaiting_review",
                "reconciled",
                "closed",
            }:
                continue
            recorded_workflow = record.get("workflow_spec")
            if not recorded_workflow:
                continue
            try:
                recorded_canonical = self.workspace.canonical_spec(
                    str(recorded_workflow),
                    cwd=self.workspace.root,
                )
            except WorkspaceError:
                continue
            if (
                recorded_canonical == canonical
                and record.get("result_specification_fingerprint")
            ):
                return record
        return None

    def _associate_implementation_workflow(self, workflow_spec: str) -> None:
        """Tie a completed creation request to the configured project workflow."""

        record = self.workspace.current_request()
        if (
            record is None
            or record.get("kind") != "create"
            or record.get("workflow_spec")
            or not record.get("result_specification_fingerprint")
            or str(record.get("status") or "")
            not in {"implemented", "awaiting_review", "closed"}
        ):
            return
        self.workspace.update_request(
            str(record["request_id"]),
            workflow_spec=self.workspace.canonical_spec(
                workflow_spec,
                cwd=self.workspace.root,
            ),
        )

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

        self._warn_if_git_state_differs()
        implementation = self.workspace.implementation_state()
        implementation_state = str(implementation["state"])
        if implementation_state in {"absent", "stale"}:
            absent = implementation_state == "absent"
            self._emit_table(
                "Deployment precondition",
                [
                    ("Status", "blocked", "error"),
                    (
                        "Reason",
                        implementation.get("reason") or "implementation is not current",
                        "error",
                    ),
                    (
                        "Required",
                        (
                            "an implementation must exist"
                            if absent
                            else "implementation must match the current specification"
                        ),
                        "info",
                    ),
                    (
                        "Next",
                        "workflow edit-spec · workflow import · workflow implement"
                        if absent
                        else "workflow implement",
                        None,
                    ),
                ],
            )
            raise SystemExit(
                "Deployment requires an implementation."
                if absent
                else "Deployment requires an implementation of the current "
                "specification."
            )
        if self.workspace.specification() is None:
            raise SystemExit(
                "Deployment requires a workflow specification. Use "
                "'workflow edit-spec' first."
            )

        current, workflow, module = self._current_context()
        current_target = self.workspace.absolute_spec(current)
        current_spec = deployment_spec_from_module(module)
        name = _slug(
            names[0]
            if names
            else current_spec.name
            or _deployment_name_from_workflow(current_target, workflow)
        )
        current_fingerprint = self.workspace.specification_fingerprint()

        target = current_target
        deployment_cwd = self.workspace.root
        deployment_workflow = workflow
        deployment_module = module
        deployment_source = "current working tree"
        if implementation_state == "external":
            self._warning(
                "This implementation was not generated from the specification. "
                "Deployment will proceed after technical validation."
            )
            implementation_summary = (
                "external; not generated from the specification"
            )
            implementation_kind: StatusKind = "warning"
        else:
            implementation_summary = "matches the current specification"
            implementation_kind = "success"
        # The routes `run` resolves, minus the live probe.  A deployment may be
        # prepared now and started later, so an unreachable endpoint is not a
        # reason to refuse.  What must hold is that every participant has a
        # model configuration, and that this machine can run the assistant
        # actions the workflow declares.  Connectors are validated separately
        # by _deployment_connector_arguments.
        self._check_workflow_models(
            current,
            deployment_workflow,
            deployment_module,
            verify=False,
        )
        self._check_workflow_assistants(deployment_workflow, deployment_module)
        spec = deployment_spec_from_module(deployment_module)
        self._emit_table(
            "Guided deployment",
            [
                ("Deployment", name, None),
                (
                    "Implementation",
                    implementation_summary,
                    implementation_kind,
                ),
                ("Source", deployment_source, None),
                (
                    "Safety boundary",
                    "deployment validates the selected source again; an "
                    "existing running bundle remains isolated and unchanged",
                    "info",
                ),
            ],
        )
        arguments = ["deploy", target]
        if names:
            arguments.extend(["--name", name])
        arguments.extend(["--project-root", str(self.workspace.root)])
        deployed_semantics = semantic_snapshot(
            deployment_workflow,
            deployment_module,
        )
        deployed_semantic_fingerprint = (
            self._semantic_snapshot_fingerprint(deployed_semantics)
        )
        alignment_metadata = {
            "schema_version": 1,
            "workflow_spec": current,
            "specification_fingerprint": current_fingerprint,
            "semantic_fingerprint": deployed_semantic_fingerprint,
        }
        arguments.extend(
            [
                "--project-alignment-json",
                json.dumps(alignment_metadata, sort_keys=True),
            ]
        )
        arguments.append("--concise")
        if no_start:
            arguments.append("--no-start")
        profile = self.workspace.model_profile(
            current,
            default=default_llm_spec(deployment_module),
        )
        arguments.extend(["--llm", str(profile["default"])])
        arguments.extend(
            [
                "--assistant",
                str(self._global_settings()["assistant"]),
            ]
        )
        overrides = profile.get("lifelines") or {}
        action_overrides = profile.get("actions") or {}
        selected_specs = [str(profile["default"])]
        if isinstance(overrides, dict):
            for lifeline, model in sorted(overrides.items()):
                arguments.extend(["--llm-for", f"{lifeline}={model}"])
                selected_specs.append(str(model))
        if isinstance(action_overrides, dict):
            for target, model in sorted(action_overrides.items()):
                arguments.extend(["--llm-for", f"{target}={model}"])
                selected_specs.append(str(model))
        idle_timeouts = self._model_idle_timeout_routes(
            current,
            deployment_workflow,
            deployment_module,
        )
        arguments.extend(
            [
                "--llm-idle-timeouts-json",
                json.dumps(idle_timeouts, sort_keys=True),
            ]
        )
        arguments.extend(
            self._deployment_secret_reuse_arguments(
                name=name,
                spec=spec,
                model_specs=tuple(selected_specs),
            )
        )
        arguments.extend(
            self._deployment_connector_arguments(
                workflow_spec=current,
                workflow=deployment_workflow,
                module=deployment_module,
            )
        )
        rc = self._run_project_cli(arguments, cwd=deployment_cwd)
        if rc != 0:
            raise SystemExit(f"Deployment {name} did not complete successfully.")
        from zippergen.serve import _deployment_profile_path, _load_deployment_profile

        deployed_profile: dict[str, object] | None = None
        if _deployment_profile_path(name).exists():
            deployed_profile = _load_deployment_profile(name)
            self.workspace.update(last_deployment=name)
        else:
            self.workspace.update(last_deployment=name)
        outcome = "prepared" if no_start else "completed"
        self._success(f"Deployment {outcome}: {name}")
        if (
            _deployment_profile_path(name).exists()
            and deployed_profile is not None
            and deployed_profile.get("workflow")
            and deployed_profile.get("store")
        ):
            self.show_deployment([name])


    def _deployment_name(self, selector: str | None = None) -> str:
        from zippergen.studio_stores import deployment_profiles

        profiles = deployment_profiles(self.workspace)
        if selector:
            matches = [
                str(profile["name"])
                for _path, profile in profiles
                if str(profile["name"]).casefold() == selector.casefold()
            ]
            if len(matches) == 1:
                return matches[0]
            raise SystemExit(
                f"Deployment not found: {selector}. Use 'deploy list'."
            )
        remembered = self.workspace.load().get("last_deployment")
        if remembered:
            return str(remembered)
        if len(profiles) == 1:
            return str(profiles[0][1]["name"])
        if not profiles:
            raise SystemExit("No deployments exist. Use deploy first.")
        raise SystemExit(
            "No deployment is selected. Use 'deploy list', then "
            "'deploy show NAME'."
        )

    def _deployment_store_record(self, name: str):
        from zippergen.serve import _load_deployment_profile

        profile = _load_deployment_profile(name)
        store = profile.get("store")
        if not store:
            raise SystemExit(
                f"Deployment {name} has no durable state configured."
            )
        record = self._store_record(str(store))
        self.workspace.update(last_deployment=name)
        return record

    def show_deployments(self) -> None:
        from zippergen.serve import _deployment_service_status, _store_status
        from zippergen.studio_stores import deployment_profiles

        profiles = deployment_profiles(self.workspace)
        if not profiles:
            self._emit_table(
                "Deployments",
                [
                    ("Status", "none", "warning"),
                    ("Next", "workflow status · deploy", None),
                ],
            )
            return
        remembered = str(self.workspace.load().get("last_deployment") or "")
        rows = []
        for _path, profile in profiles:
            name = str(profile["name"])
            service = _deployment_service_status(name)
            store = _store_status(str(profile["store"]))
            pending = store.get("pending_human_tasks")
            pending_count = len(pending) if isinstance(pending, list) else 0
            store_state = str(store["state"])
            if pending_count:
                noun = "task" if pending_count == 1 else "tasks"
                store_state += f" · {pending_count} {noun}"
            rows.append(
                (
                    "●" if name == remembered else "",
                    name,
                    service["state"],
                    store_state,
                    profile.get("workflow") or "—",
                )
            )
        self._emit_columns(
            "Deployments",
            ("Selected", "Deployment", "Service", "Store", "Workflow"),
            rows,
        )
        self._emit_next(
            "deploy show NAME · deploy tasks NAME · deploy logs NAME"
        )

    @staticmethod
    def _deployment_log_cause(profile: dict[str, object]) -> str | None:
        path = Path(str(profile.get("log") or "")).expanduser()
        if not path.is_file():
            return None
        try:
            content = path.read_bytes()
        except OSError:
            return None
        raw_offset = profile.get("log_generation_offset")
        if isinstance(raw_offset, int) and 0 <= raw_offset <= len(content):
            content = content[raw_offset:]
        lines = content.decode(errors="replace").splitlines()
        for line in reversed(lines[-200:]):
            stripped = line.strip()
            if (
                stripped.startswith(
                    (
                        "RuntimeError:",
                        "SystemExit:",
                        "ValueError:",
                        "TypeError:",
                        "ModuleNotFoundError:",
                        "ConnectionError:",
                    )
                )
                or "Error:" in stripped
            ):
                return stripped[:300]
        return None

    @staticmethod
    def _deployment_runtime_summary(profile: dict[str, object]) -> str:
        raw = profile.get("zippergen_runtime")
        if not isinstance(raw, dict):
            return "not recorded; redeploy to record the runtime revision"
        version = str(raw.get("version") or "unknown version")
        revision = str(raw.get("revision") or "").strip()
        source_hash = str(raw.get("source_sha256") or "").strip()
        kind = str(raw.get("kind") or "runtime")
        if revision:
            summary = f"{version} · {kind} {revision[:12]}"
            if source_hash:
                summary += f" · source {source_hash[:12]}"
            return summary
        return f"{version} · {kind}"

    def _deployment_model_routes(self, profile: dict[str, object]) -> str:
        """Describe effective LLM-active routes, omitting unused fallbacks."""

        default = str(profile.get("llm") or "mock")
        overrides = normalize_llm_overrides(profile.get("llms"))
        workflow_ref = str(profile.get("workflow") or "")
        cwd = Path(str(profile.get("cwd") or profile.get("bundle") or ""))
        try:
            from zippergen.serve import load_workflow_spec

            module_ref, separator, workflow_name = workflow_ref.partition(":")
            module_path = Path(module_ref).expanduser()
            if not module_path.is_absolute():
                module_path = cwd / module_path
            target = str(module_path)
            if separator:
                target += f":{workflow_name}"
            workflow, module = load_workflow_spec(target)
            active = self._llm_action_lifelines(workflow, module)
            if not active:
                return "none; no LLM actions"
            return " · ".join(
                f"{participant}.{action}="
                f"{overrides.get(f'{participant}.{action}', overrides.get(participant, default))}"
                for participant, actions in active.items()
                for action in actions
            )
        except (Exception, SystemExit):
            routes = [
                f"{participant}={model}"
                for participant, model in sorted(overrides.items())
            ]
            if routes:
                routes.append(f"default fallback={default}")
                return " · ".join(routes)
            return f"default={default}"

    @staticmethod
    def _deployment_connector_routes(profile: dict[str, object]) -> str:
        raw = profile.get("connectors") or {}
        if not isinstance(raw, dict) or not raw:
            return "none"
        routes = []
        for route_name, value in sorted(raw.items()):
            if not isinstance(value, dict):
                routes.append(f"{route_name}=invalid")
                continue
            label = (
                str(value.get("target"))
                if value.get("type") == "human"
                else route_name.removeprefix("requirement:")
            )
            routes.append(
                f"{label}="
                f"{value.get('configuration') or value.get('kind') or 'unknown'}"
            )
        return " · ".join(routes)

    @staticmethod
    def _normalized_connector_snapshot(
        raw: object,
    ) -> dict[str, dict[str, object]]:
        """Keep only non-secret routing and target values for comparison."""

        if not isinstance(raw, dict):
            return {}
        fields = {
            "type",
            "target",
            "participant",
            "action",
            "name",
            "kind",
            "provider",
            "configuration",
            "access",
            "capabilities",
            "channel",
            "chat_id",
            "spreadsheet_id",
            "tab",
            "account",
            "query",
        }
        return {
            str(name): {
                str(key): value
                for key, value in value.items()
                if key in fields
            }
            for name, value in raw.items()
            if isinstance(value, dict)
        }

    def _current_connector_snapshot(
        self,
        workflow_spec: str,
        workflow,
        module,
    ) -> dict[str, dict[str, object]]:
        """Build current non-secret connector routing without readiness checks."""

        from zippergen.connectors import connector_requirements_from_module

        configurations = self.workspace.connector_configurations()
        human_assignments = self.workspace.connector_assignment_profile(
            workflow_spec
        )
        bindings = self.workspace.connector_binding_profile(workflow_spec)
        snapshot: dict[str, dict[str, object]] = {}
        for target, configuration_name in [
            *human_assignments["lifelines"].items(),
            *human_assignments["actions"].items(),
        ]:
            configuration = configurations.get(configuration_name) or {}
            provider = str(
                configuration.get("provider")
                or configuration.get("kind")
                or ""
            )
            snapshot[f"human:{target}"] = {
                "type": "human",
                "target": target,
                "participant": target.partition(".")[0],
                "action": target.partition(".")[2] or None,
                "kind": provider,
                "provider": provider,
                "configuration": configuration_name,
                "chat_id": configuration.get("chat_id"),
                "channel": configuration.get("channel")
                or f"telegram:{configuration_name}",
            }
        for requirement in connector_requirements_from_module(module):
            configuration_name = bindings.get(requirement.name)
            if configuration_name is None:
                continue
            configuration = configurations.get(configuration_name) or {}
            provider = str(
                configuration.get("provider")
                or configuration.get("kind")
                or ""
            )
            record: dict[str, object] = {
                **requirement.as_dict(),
                "provider": provider,
                "configuration": configuration_name,
                "channel": configuration.get("channel") or requirement.name,
            }
            if requirement.kind == "telegram":
                record["chat_id"] = configuration.get("chat_id")
            elif requirement.kind == "google-sheets":
                record.update(
                    {
                        "spreadsheet_id": configuration.get("spreadsheet_id"),
                        "tab": configuration.get("tab"),
                    }
                )
            elif requirement.kind == "gmail":
                record.update(
                    {
                        "account": configuration.get("account") or "me",
                        "query": configuration.get("query")
                        or "is:unread in:inbox",
                    }
                )
            snapshot[f"requirement:{requirement.name}"] = record
        return self._normalized_connector_snapshot(snapshot)

    @staticmethod
    def _semantic_snapshot_fingerprint(value: dict[str, object]) -> str:
        return hashlib.sha256(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    def _deployment_project_alignment(
        self,
        profile: dict[str, object],
    ) -> tuple[str, StatusKind, tuple[str, ...]]:
        """Compare an immutable deployment snapshot with the current project."""

        baseline = profile.get("project_alignment")
        if not isinstance(baseline, dict):
            return (
                "not recorded; redeploy once to enable comparison",
                "warning",
                (),
            )
        if str(profile.get("project_root") or "") != str(self.workspace.root):
            return (
                "belongs to a different project root",
                "warning",
                ("project root",),
            )
        workflow_spec = str(baseline.get("workflow_spec") or "")
        if not workflow_spec:
            return (
                "record is incomplete; redeploy to refresh it",
                "warning",
                (),
            )
        try:
            from zippergen.serve import load_workflow_spec

            workflow, module = load_workflow_spec(
                self.workspace.absolute_spec(workflow_spec)
            )
        except (Exception, SystemExit):
            return (
                "project workflow can no longer be loaded",
                "warning",
                ("workflow",),
            )

        changed: list[str] = []
        current_semantic_fingerprint = self._semantic_snapshot_fingerprint(
            semantic_snapshot(workflow, module)
        )
        if (
            baseline.get("semantic_fingerprint")
            != current_semantic_fingerprint
        ):
            changed.append("workflow")
        if (
            baseline.get("specification_fingerprint")
            != self.workspace.specification_fingerprint()
        ):
            changed.append("specification")

        current_models = self.workspace.model_profile(
            workflow_spec,
            default=default_llm_spec(module),
        )
        deployed_models = {
            "default": str(profile.get("llm") or "mock"),
            "lifelines": normalize_llm_overrides(profile.get("llms")),
        }
        deployed_lifelines = {
            key: value
            for key, value in deployed_models["lifelines"].items()
            if "." not in key
        }
        deployed_actions = {
            key: value
            for key, value in deployed_models["lifelines"].items()
            if "." in key
        }
        deployed_models["lifelines"] = deployed_lifelines
        if deployed_actions:
            deployed_models["actions"] = deployed_actions
        normalized_current_models: dict[str, object] = {
            "default": str(current_models.get("default") or "mock"),
            "lifelines": dict(current_models.get("lifelines") or {}),
        }
        if current_models.get("actions"):
            normalized_current_models["actions"] = dict(
                current_models["actions"]
            )
        if deployed_models != normalized_current_models:
            changed.append("models")

        current_idle = self._model_idle_timeout_routes(
            workflow_spec,
            workflow,
            module,
        )
        raw_deployed_idle = profile.get("llm_idle_timeouts")
        deployed_idle = (
            {
                str(target): float(value)
                for target, value in raw_deployed_idle.items()
            }
            if isinstance(raw_deployed_idle, dict)
            else {}
        )
        if current_idle != deployed_idle:
            changed.append("model idle policy")
        if str(profile.get("assistant") or "codex") != str(
            self._global_settings().get("assistant") or "codex"
        ):
            changed.append("assistant")

        current_connectors = self._current_connector_snapshot(
            workflow_spec,
            workflow,
            module,
        )
        deployed_connectors = self._normalized_connector_snapshot(
            profile.get("connectors")
        )
        if current_connectors != deployed_connectors:
            changed.append("connectors")

        selected_specs = {
            str(normalized_current_models["default"]),
            *(
                str(value)
                for value in cast(
                    dict[str, object],
                    normalized_current_models["lifelines"],
                ).values()
            ),
            *(
                str(value)
                for value in cast(
                    dict[str, object],
                    normalized_current_models.get("actions") or {},
                ).values()
            ),
        }
        if any(_canonical_provider(spec) == "local" for spec in selected_specs):
            current_endpoint = self.workspace.provider_profiles().get(
                "local", {}
            ).get("base_url", "http://127.0.0.1:11434/v1")
            deployed_environment = profile.get("environment") or {}
            deployed_endpoint = (
                deployed_environment.get("OLLAMA_BASE_URL")
                if isinstance(deployed_environment, dict)
                else None
            )
            if str(deployed_endpoint or "") != str(current_endpoint):
                changed.append("local provider endpoint")

        unique_changed = tuple(dict.fromkeys(changed))
        if unique_changed:
            return (
                "differs from current project: "
                + ", ".join(unique_changed)
                + ". Redeploy to apply project changes",
                "warning",
                unique_changed,
            )
        return (
            "matches current specification, workflow, and configuration",
            "success",
            (),
        )

    def show_deployment(self, args: list[str]) -> None:
        if len(args) > 1:
            raise SystemExit("Use deploy show [NAME].")
        name = self._deployment_name(args[0] if args else None)
        from zippergen.serve import (
            _deployment_boot_status,
            _deployment_profile_path,
            _deployment_service_status,
            _doctor_checks,
            _load_deployment_profile,
            _store_status,
        )

        profile = _load_deployment_profile(name)
        service = _deployment_service_status(name)
        boot = _deployment_boot_status(name)
        store = _store_status(str(profile["store"]))
        bundle = Path(str(profile.get("bundle") or profile.get("cwd") or ""))
        checks = _doctor_checks(
            name,
            include_systemd=False,
            live_connectors=False,
        )
        failures = [
            check for check in checks if check.get("status") == "fail"
        ]
        log_cause = self._deployment_log_cause(profile)
        cause = (
            f"{failures[0]['name']}: {failures[0]['detail']}"
            if failures
            else log_cause
            if service["state"] == "restarting"
            else None
        )
        service_kind: StatusKind = (
            "success"
            if service["state"] in {"running", "completed"}
            else "error"
            if service["state"] == "restarting"
            else "warning"
        )
        run_kind: StatusKind = (
            "success"
            if store["state"] == "done"
            else "error"
            if store["state"] == "invalid"
            else "warning"
            if store["state"] in {"missing", "waiting"}
            else "info"
        )
        store_kind: StatusKind = (
            "error"
            if store["state"] == "invalid"
            else "warning"
            if store["state"] == "missing"
            else "success"
        )
        run_state = {
            "missing": "deployment store is missing",
            "waiting": "waiting for human action",
            "done": "completed result recorded",
            "active": "events recorded; no result yet",
            "empty": (
                "starting, no durable events recorded yet"
                if service["state"] == "running"
                else "service is restarting before durable execution"
                if service["state"] == "restarting"
                else "service completed without durable events"
                if service["state"] == "completed"
                else "not started, no durable events recorded"
            ),
            "invalid": "store cannot be read",
        }.get(str(store["state"]), str(store["summary"]))
        pending = store.get("pending_human_tasks")
        pending_count = len(pending) if isinstance(pending, list) else 0
        if store["state"] == "missing":
            store_description = f"missing — {profile['store']}"
        elif pending_count:
            noun = "task" if pending_count == 1 else "tasks"
            store_description = (
                f"ready — {pending_count} pending human {noun} — "
                f"{profile['store']}"
            )
        elif store["state"] == "empty":
            store_description = (
                f"ready — no run data yet — {profile['store']}"
            )
        else:
            store_description = (
                f"ready — {store['state']} — {profile['store']}"
            )
        selected_models = self._deployment_model_routes(profile)
        selected_connectors = self._deployment_connector_routes(profile)
        alignment, alignment_kind, _alignment_changed = (
            self._deployment_project_alignment(profile)
        )
        missing_provider = next(
            (
                provider
                for provider, secret_name in _PROVIDER_SECRETS.items()
                if any(
                    str(check.get("name") or "")
                    == f"model credential {secret_name}"
                    for check in failures
                )
            ),
            None,
        )
        next_action = (
            "deploy inspect · deploy stop · "
            + (
                f"model provider check {missing_provider} · deploy"
                if missing_provider
                else "deploy logs"
            )
            if service["state"] == "restarting"
            else "deploy inspect · deploy tasks"
            if store["state"] == "waiting"
            else "deploy inspect · deploy logs"
            if failures
            else "deploy inspect · deploy doctor · deploy logs"
        )
        self._emit_table(
            "Deployment state",
            [
                ("Deployment", name, None),
                (
                    "Bundle",
                    f"installed — {bundle}" if bundle.is_dir() else f"missing — {bundle}",
                    "success" if bundle.is_dir() else "error",
                ),
                (
                    "Runtime",
                    self._deployment_runtime_summary(profile),
                    (
                        "success"
                        if isinstance(profile.get("zippergen_runtime"), dict)
                        else "warning"
                    ),
                ),
                ("Service", service["detail"], service_kind),
                (
                    "Boot",
                    boot["detail"],
                    cast(StatusKind, boot["kind"]),
                ),
                ("Run", run_state, run_kind),
                (
                    "Store",
                    store_description,
                    store_kind,
                ),
                ("Models", selected_models, None),
                (
                    "Local idle release",
                    self._model_idle_routes_summary(
                        profile.get("llm_idle_timeouts")
                    ),
                    None,
                ),
                ("Connectors", selected_connectors, None),
                ("Project alignment", alignment, alignment_kind),
                (
                    "Cause",
                    cause or "no immediate failure detected",
                    "error" if cause else "success",
                ),
                *(
                    [
                        (
                            "Previous failure",
                            f"historical log entry — {log_cause}",
                            "warning",
                        )
                    ]
                    if log_cause and not cause
                    else []
                ),
                ("Profile", _deployment_profile_path(name), None),
                ("Log", profile.get("log") or "not configured", None),
                ("Next", next_action, None),
            ],
        )

    def inspect_deployment(self, args: list[str]) -> None:
        positionals, watch = self._inspection_options(
            args,
            max_positionals=2,
            usage="deploy inspect [NAME] [PARTICIPANT] [--watch]",
        )
        from zippergen.serve import (
            _deployment_profile_path,
            _deployment_service_status,
            _load_deployment_profile,
            load_workflow_spec,
        )

        participant: str | None = None
        if len(positionals) == 2:
            name = self._deployment_name(positionals[0])
            participant = positionals[1]
        elif (
            len(positionals) == 1
            and _deployment_profile_path(positionals[0]).exists()
        ):
            name = self._deployment_name(positionals[0])
        elif len(positionals) == 1:
            name = self._deployment_name(None)
            participant = positionals[0]
        else:
            name = self._deployment_name(None)

        def render_once() -> None:
            profile = _load_deployment_profile(name)
            workflow_ref = str(profile.get("workflow") or "")
            cwd = Path(str(profile.get("cwd") or profile.get("bundle") or ""))
            module_ref, separator, workflow_name = workflow_ref.partition(":")
            module_path = Path(module_ref).expanduser()
            if not module_path.is_absolute():
                module_path = cwd / module_path
            target = str(module_path)
            if separator:
                target += f":{workflow_name}"
            workflow, _module = load_workflow_spec(target)
            service = _deployment_service_status(name)
            service_kind: StatusKind = (
                "success"
                if service["state"] in {"running", "completed"}
                else "error"
                if service["state"] == "restarting"
                else "warning"
            )
            self._inspect_execution(
                workflow=workflow,
                store=str(profile.get("store") or ""),
                source_rows=[
                    ("Deployment", name, None),
                    ("Workflow", workflow_ref, None),
                    ("Service", service["detail"], service_kind),
                    (
                        "Bundle",
                        cwd,
                        "success" if cwd.is_dir() else "error",
                    ),
                ],
                participant=participant,
                next_commands=(
                    f"deploy inspect {name} {participant} · "
                    f"deploy tasks {name} · deploy logs {name}"
                    if participant
                    else f"deploy inspect {name} PARTICIPANT · "
                    f"deploy tasks {name} · deploy logs {name}"
                ),
                projection_indent=1 if watch else 0,
            )

        if watch:
            command = f"deploy inspect {name}"
            if participant:
                command += f" {participant}"
            command += " --watch"
            self._watch_execution(
                render_once,
                command=command,
                subject="deployment",
            )
        else:
            render_once()

    def manage_deploy(self, args: list[str]) -> None:
        if not args:
            self.show_deployments()
            return
        action, *rest = args
        action = action.casefold()
        if action == "list":
            if rest:
                raise SystemExit("Use deploy list.")
            self.show_deployments()
            return
        if action == "show":
            self.show_deployment(rest)
            return
        if action == "inspect":
            self.inspect_deployment(rest)
            return
        if action == "storage":
            if rest and rest[0].casefold() == "compact":
                self.compact_deployment_storage(rest[1:])
            else:
                self.show_deployment_storage(rest)
            return
        if action == "remove":
            self.remove_deployment(rest)
            return
        if action == "logs" and rest and rest[0].casefold() == "reset":
            self.reset_deployment_logs(rest[1:])
            return
        if action in {"tasks", "approve", "trace"}:
            if len(rest) > 1:
                raise SystemExit(
                    f"Use deploy {action} or deploy {action} NAME."
                )
            name = self._deployment_name(rest[0] if rest else None)
            record = self._deployment_store_record(name)
            if action == "tasks":
                self._show_pending_tasks(record, deployment_name=name)
            elif action == "approve":
                self._approve_pending_task(
                    record,
                    [],
                    deployment_name=name,
                )
            else:
                self._show_recent_trace(
                    record,
                    owner=f"deployment {name}",
                )
            return
        if action in {"doctor", "logs", "start", "restart", "stop"}:
            self.deployment_action(action, rest)
            return
        raise SystemExit(
            "Use deploy list, show, inspect, doctor, logs, tasks, approve, "
            "trace, storage, start, restart, stop, or remove."
        )

    def reset_deployment_logs(self, args: list[str]) -> None:
        yes = False
        names: list[str] = []
        for argument in args:
            if argument == "--yes":
                yes = True
            elif argument.startswith("--"):
                raise SystemExit(
                    "Use deploy logs reset [NAME] [--yes]."
                )
            else:
                names.append(argument)
        if len(names) > 1:
            raise SystemExit("Use deploy logs reset [NAME] [--yes].")

        name = self._deployment_name(names[0] if names else None)
        from zippergen.serve import _load_deployment_profile
        from zippergen.studio_deployments import (
            DeploymentRemovalError,
            reset_deployment_log,
        )

        profile = _load_deployment_profile(name)
        log = Path(str(profile.get("log") or "")).expanduser()
        size = log.stat().st_size if log.is_file() else 0
        raw_offset = profile.get("log_generation_offset")
        visible_size = (
            size - raw_offset
            if isinstance(raw_offset, int) and 0 <= raw_offset <= size
            else size
        )
        self._emit_table(
            "Deployment log reset",
            [
                ("Deployment", name, None),
                ("Log", log, None),
                ("Visible history", f"{visible_size:,} byte(s)", None),
                (
                    "Effect",
                    "archive existing history; keep the service, workflow "
                    "run, and durable store unchanged",
                    "info",
                ),
            ],
        )
        if not yes and not self._confirm_action(
            f"Archive and reset visible logs for {name}? [y/N]: ",
            cancel_message=(
                "Deployment log reset cancelled; nothing was changed."
            ),
            default=False,
        ):
            return
        try:
            result = reset_deployment_log(name, profile)
        except DeploymentRemovalError as exc:
            raise SystemExit(str(exc)) from exc

        self.workspace.update(last_deployment=name)
        self._success(f"Deployment log history reset: {name}")
        self._emit_table(
            "Log reset result",
            [
                ("Active log", result.log, None),
                (
                    "Archived",
                    (
                        f"{result.archived_bytes:,} byte(s) — "
                        f"{result.archive}"
                        if result.archive is not None
                        else "none; the log was already empty"
                    ),
                    "success",
                ),
                (
                    "Current history",
                    "empty; future entries appear normally",
                    "success",
                ),
                (
                    "Service",
                    "unchanged; no stop or restart was performed",
                    "success",
                ),
            ],
        )
        self._emit_next(f"deploy logs {name} · deploy show {name}")

    def remove_deployment(self, args: list[str]) -> None:
        purge = False
        yes = False
        names: list[str] = []
        for argument in args:
            if argument == "--purge":
                purge = True
            elif argument == "--yes":
                yes = True
            elif argument.startswith("--"):
                raise SystemExit(
                    "Use deploy remove [NAME] [--purge] [--yes]."
                )
            else:
                names.append(argument)
        if len(names) > 1:
            raise SystemExit(
                "Use deploy remove [NAME] [--purge] [--yes]."
            )
        if purge and not names:
            raise SystemExit(
                "Permanent removal requires an explicit deployment name. "
                "Use deploy remove NAME --purge."
            )

        name = self._deployment_name(names[0] if names else None)
        from zippergen.serve import (
            _deployment_service_status,
            _load_deployment_profile,
        )
        from zippergen.studio_deployments import (
            DeploymentRemovalError,
            present_deployment_artifacts,
            remove_deployment_artifacts,
            unregister_deployment_service,
        )

        profile = _load_deployment_profile(name)
        try:
            artifacts = present_deployment_artifacts(name, profile)
        except DeploymentRemovalError as exc:
            raise SystemExit(str(exc)) from exc
        service = _deployment_service_status(name)
        mode = (
            "permanent purge; no recovery archive"
            if purge
            else "recoverable archive"
        )
        self._emit_table(
            "Deployment removal",
            [
                ("Deployment", name, None),
                ("Mode", mode, "error" if purge else "warning"),
                (
                    "Service",
                    str(service.get("detail") or "state unavailable"),
                    (
                        "success"
                        if service.get("state") in {"not-loaded", "completed"}
                        else "warning"
                    ),
                ),
                (
                    "Project state",
                    "workflow code, implementation history, models, and connector "
                    "configurations remain",
                    "info",
                ),
            ],
        )
        self._emit_columns(
            "Deployment-owned artifacts",
            ("Artifact", "Path"),
            [(artifact.label, artifact.path) for artifact in artifacts],
        )

        if not yes:
            if purge:
                try:
                    confirmation = self.input(
                        f"Type {name} to permanently purge this deployment: "
                    ).strip()
                except (EOFError, KeyboardInterrupt):
                    self._warning(
                        "Deployment purge cancelled; nothing was changed."
                    )
                    return
                if confirmation != name:
                    self._warning(
                        "Deployment purge cancelled; the name did not match."
                    )
                    return
            elif not self._confirm_action(
                f"Archive deployment {name} and remove it from active use? "
                "[y/N]: ",
                cancel_message=(
                    "Deployment removal cancelled; nothing was changed."
                ),
                default=False,
            ):
                return

        service_result: str | None = None
        try:
            service_result = unregister_deployment_service(name)
            result = remove_deployment_artifacts(
                name,
                profile,
                purge=purge,
            )
        except DeploymentRemovalError as exc:
            if service_result is not None:
                raise SystemExit(
                    f"{exc} The service was safely unregistered, but the "
                    "deployment artifacts remain in active storage."
                ) from exc
            raise SystemExit(str(exc)) from exc

        state = self.workspace.load()
        updates: dict[str, object] = {}
        if str(state.get("last_deployment") or "").casefold() == name.casefold():
            updates["last_deployment"] = None
        if updates:
            self.workspace.update(**updates)

        self._success(
            (
                f"Deployment permanently purged: {name}"
                if result.purged
                else f"Deployment removed from active use: {name}"
            )
        )
        rows: list[tuple[str, object, StatusKind | None]] = [
            ("Deployment", name, None),
            ("Service", service_result, "success"),
            ("Artifacts", result.artifact_count, None),
        ]
        if result.archive is not None:
            rows.append(("Archive", result.archive, "success"))
        else:
            rows.append(("Archive", "none; deletion was permanent", "warning"))
        self._emit_table("Removal result", rows)
        self._emit_next("deploy list · deploy")

    def deployment_action(self, action: str, args: list[str]) -> None:
        if len(args) > 1:
            raise SystemExit(
                f"Use deploy {action} or deploy {action} NAME."
            )
        name = self._deployment_name(args[0] if args else None)
        arguments = [action, str(name)]
        if action == "start":
            arguments.append("--enable")
        rc = self._run_project_cli(arguments)
        if rc != 0:
            raise SystemExit(f"{action} failed for deployment {name}.")
        self.workspace.update(last_deployment=str(name))
        self._success(f"Deployment {action} completed: {name}")
        if action in {"start", "restart"}:
            from zippergen.serve import _deployment_profile_path

            if _deployment_profile_path(str(name)).exists():
                self.show_deployment([name])

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
If pytest is missing, report the assistant checks as incomplete and tell the user to
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
normally does not turn a failed command into successful assistant checks."""

    def _task_refresh_instruction(self, refreshes_request: str | None) -> str:
        if refreshes_request is None:
            return ""
        return (
            f"This task refreshes {refreshes_request} because the canonical "
            "specification changed. The document below was captured "
            "immediately before this task was written."
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
action kinds, owned decisions and loops, non-human connector requirements,
deployment requirements, retry and safety assumptions, and success
examples. Then create visible Python source and focused mock/fake tests.
Human delivery is inferred from `@human` action sites and configured in
Studio, so do not add a redundant Telegram or email requirement for it. Every
named non-human connector capability in the specification must remain
explicit in workflow source.
For Google Sheets, declare a `google-sheets` requirement and use connector-aware
`@effect` actions with `read_json_rows` or stable-key `upsert_json_row`.
Keep the table columns and key field visible in code. Never place a spreadsheet
ID, OAuth token, or credentials path in workflow source.
When deployment metadata is present, keep its bundle self-contained by
including the workflow source and any required project assets. Run validation,
show the communication-only and full code views, confirm that every requested
non-human connector capability appears in the full view, and inspect every new
participant's exact local projection. Do not deploy or start a service. Report
generated files, assumptions, and assistant-check results.

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

Update {workflow_spec} so it implements the canonical specification below.
Preserve every unaffected requirement and behavior. {refresh_instruction}

{context}

Treat {specification_file} as read-only intent. Do not rewrite it to fit the
implementation. The specification-rewriting operation has already happened in
an isolated workspace and this task must make the implementation follow it.

The semantic baseline is {baseline_file}.
Preserve all behavior not explicitly changed.
Update source, deployment metadata, and focused tests together when needed.
Preserve and update explicit non-human connector capability declarations.
Human Telegram or email delivery is configured from `@human` action sites and
does not require a duplicate module-level declaration.
Keep any deployment bundle self-contained by including the workflow source and
required project assets.
Validate the result, show communication-only and full code views,
confirm that every requested non-human connector capability appears in the full view,
inspect every changed participant's exact local projection, and compare the
result with the baseline using `zippergen diff`. Do not deploy or start a
service. Report assumptions, intended semantic changes, preserved behavior,
and assistant-check results.

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
        ensured = self.workspace.ensure_specification()
        if ensured["content"] is None:
            return self.workspace.current_request()
        fingerprint = self.workspace.specification_fingerprint()
        record = self.workspace.current_request()
        implementation = str(self.workspace.implementation_state()["state"])
        if record is None:
            return self._prepare_implementation_request(
                fingerprint=fingerprint,
                announce=announce,
            )
        record = self._normalize_task_lifecycle(record)
        status = str(record.get("status") or "prepared")
        result = record.get("result_specification_fingerprint")
        if (
            not force
            and status in {"implemented", "awaiting_review"}
            and result == fingerprint
            and implementation == "current"
        ):
            return record
        may_refresh = status == "prepared" or (
            for_assistant
            and status in {
                "assistant_failed",
                "assistant_interrupted",
                "implemented",
                "awaiting_review",
            }
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
        refreshes_request = str(record["request_id"])
        if kind not in {"create", "refine"}:
            kind = "create" if implementation == "absent" else "refine"
        refreshed = self._prepare_implementation_request(
            fingerprint=fingerprint,
            announce=False,
            refreshes_request=refreshes_request,
            force_kind=kind,
        )
        if announce:
            self._success(
                "Implementation request refreshed from the current specification."
            )
        return refreshed

    def _prepare_implementation_request(
        self,
        *,
        fingerprint: str,
        announce: bool,
        refreshes_request: str | None = None,
        force_kind: str | None = None,
    ) -> dict[str, object]:
        """Prepare a private assistant handoff from portable project state."""

        specification = self.workspace.specification()
        if specification is None:
            raise WorkspaceError("No canonical specification is available.")
        implementation = str(self.workspace.implementation_state()["state"])
        workflow_spec = self.workspace.workflow_entry
        kind = force_kind or ("create" if implementation == "absent" else "refine")
        baseline: Path | None = None
        if kind == "refine" and workflow_spec is not None:
            try:
                _current, workflow, module = self._current_context()
                self.workspace.requests_directory.mkdir(parents=True, exist_ok=True)
                baseline = self.workspace.requests_directory / (
                    f"{time.strftime('%Y%m%d-%H%M%S')}-"
                    f"{time.time_ns() % 1_000_000_000:09d}-semantic-before.json"
                )
                baseline.write_text(
                    json.dumps(
                        semantic_snapshot(workflow, module),
                        indent=2,
                        default=str,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            except (Exception, SystemExit):
                baseline = None
        if kind == "create" or workflow_spec is None:
            kind = "create"
            content = self._creation_task_content(
                refreshes_request=refreshes_request,
            )
        else:
            baseline_label: str | Path = baseline or "not available"
            content = self._refinement_task_content(
                workflow_spec=workflow_spec,
                baseline_file=baseline_label,
                refreshes_request=refreshes_request,
            )
        record = self.workspace.save_request(
            kind=kind,
            prompt=specification,
            content=content,
            workflow_spec=workflow_spec,
            specification_fingerprint=fingerprint,
            baseline_file=baseline,
            refreshes_request=refreshes_request,
        )
        if announce:
            self._success("Implementation request prepared from specification.md.")
        return record

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
                    "'workflow edit-spec' to change it."
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
        if not prompt:
            prompt = self.input("Describe the change: ").strip()
        if not prompt:
            raise SystemExit("The refinement description must not be empty.")
        del source_path
        del append
        ensured = self.workspace.ensure_specification()
        if ensured["content"] is None:
            raise SystemExit(
                "No workflow specification exists. Use 'workflow edit-spec' "
                "first."
            )
        self.workspace.save_refinement_buffer(prompt)
        self._emit_table(
            "Refinement buffer",
            [
                ("Status", "ready to apply", "success"),
                ("Storage", "ignored scratch file", None),
                ("Next", "workflow refine-spec", None),
            ],
        )
