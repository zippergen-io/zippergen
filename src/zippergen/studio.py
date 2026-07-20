"""A lightweight, discoverable project shell for ZipperGen development."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from zippergen.dev import default_llm_spec, run_dev
from zippergen.models import normalize_llm_overrides
from zippergen.semantic import semantic_snapshot, workflow_semantics
from zippergen.view import ViewOptions, workflow_view_data
from zippergen.workspace import Workspace, WorkspaceError


InputFunc = Callable[[str], str]
OutputFunc = Callable[[str], object]
SecretInputFunc = Callable[[str], str]
StatusKind = Literal["success", "warning", "error", "info"]


@dataclass(frozen=True)
class _PromptInput:
    content: str
    source_path: Path | None = None
    draft_path: Path | None = None


_PROVIDER_ALIASES = {
    "claude": "anthropic",
    "ollama": "local",
}
_PROVIDER_SECRETS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "mistral": "MISTRAL_API_KEY",
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
  project init [NAME]            create zippergen.toml and the prompt ledger
  project show                   show visible project configuration
  create [PROMPT]                prepare a new-workflow coding-assistant brief
  create --file PATH             read a multiline workflow prompt from a file
  create --edit [PATH]           compose a prompt, then prepare the task
  task                            show the current freshness-checked task
  task show|path|history          inspect the synchronized task or its history
  assistant [codex|claude]       sync prompts, then open a coding assistant
  editor [show|set CMD|reset]     inspect or remember the terminal editor
  edit [workflow|file PATH]       edit with the remembered/default editor
  edit ... --editor CMD           choose an editor for this invocation only
  prompts                        list the ordered project prompt ledger
  prompts add [PROMPT]           add design intent without preparing a handoff
  prompts add --file PATH        import/register a UTF-8 prompt file
  prompts add --edit [PATH]      compose and register project design intent
  prompts show|inspect ID        show one prompt and its metadata verbatim
  prompts path ID                print the prompt's absolute file path
  prompts edit ID                edit one prompt without changing its stable ID
  prompts context                show all active prompts in precedence order
  prompts archive|restore ID     exclude/include a prompt in future handoffs
  prompts remove|disable ID      aliases for archive; enable aliases restore
  prompts replace ID [PROMPT]    supersede a prompt while preserving its history
  prompts replace ID --edit [PATH]
                                 compose and register a replacement prompt
  prompts move ID before|after ID
  use [PATH.py:WORKFLOW]         select a workflow; no argument opens a selector
  current                        show the complete project/workflow dashboard
  show | inspect                 choose a code-first semantic view
  show overview|protocol|communications|actions|full
  show agent [NAME]              exact local projection (selector if omitted)
  show agents [NAME ...]         selected-participant focus view
  validate                       validate the current workflow
  models                         configure default/per-lifeline LLMs interactively
  models show                    show the current workflow's model routing
  models default SPEC            set the inherited default LLM
  models set LIFELINE SPEC       override one LLM-active lifeline
  models reset LIFELINE|all      restore inheritance or reset the whole profile
  providers                      show model-provider readiness without secrets
  providers set openai|anthropic|mistral
  providers set local [URL]      configure a local OpenAI-compatible endpoint
  providers reset NAME           remove a saved provider configuration
  run [LLM]                      start a run; optional LLM overrides its default once
  resume                         resume the current incomplete run
  runs                           list managed development runs
  refine [PROMPT]                prepare a semantic refinement handoff
  refine --file PATH             read a multiline refinement prompt from a file
  refine --edit [PATH]           compose a refinement, then prepare the task
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

    def _prompt(self) -> str:
        current = self.workspace.current_workflow
        label = current.rsplit(":", 1)[-1] if current else "no workflow"
        return f"zippergen [{label}]> "

    def welcome(self) -> None:
        self._emit("ZipperGen Studio")
        self._emit(f"Project: {self.workspace.root}")
        current = self.workspace.current_workflow
        self._emit(f"Workflow: {current}" if current else "No workflow selected.")
        self._emit("Type 'help' for commands; 'show' opens the inspection menu.")

    def run(self) -> int:
        self.welcome()
        while True:
            try:
                line = self.input(self._prompt())
            except EOFError:
                self._emit()
                return 0
            except KeyboardInterrupt:
                self._warning("Use 'exit' to leave Studio.")
                continue
            try:
                if not self.execute(line):
                    return 0
            except KeyboardInterrupt:
                self._warning(
                    "Command interrupted. Use 'current' to inspect context; "
                    "use 'resume' for an incomplete managed run."
                )
            except (SystemExit, WorkspaceError, ValueError) as exc:
                self._error(str(exc))

    def execute(self, line: str) -> bool:
        try:
            parts = shlex.split(line)
        except ValueError as exc:
            self._error(f"Could not parse command: {exc}")
            return True
        if not parts:
            return True
        command, *args = parts
        command = command.lower()
        if command in {"exit", "quit"}:
            return False
        if command in {"help", "?"}:
            self._emit(_HELP.rstrip())
        elif command == "project":
            self.configure_project(args)
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
            if len(args) > 1:
                raise SystemExit("Use run or run LLM_SPEC.")
            profile = self._run_model_profile()
            default_model = profile.get("default")
            run_dev(
                self.workspace,
                llm=(
                    args[0]
                    if args
                    else str(default_model) if default_model else None
                ),
                llms=normalize_llm_overrides(profile.get("lifelines")),
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
            prompt_input = self._request_prompt(args, command="create")
            self.create_request(
                prompt_input.content,
                source_path=prompt_input.source_path,
            )
            self._finish_prompt_input(prompt_input)
        elif command == "refine":
            if self.workspace.current_workflow is None:
                raise SystemExit(
                    "No workflow selected. Use 'use' before preparing a refinement."
                )
            prompt_input = self._request_prompt(args, command="refine")
            self.refine_request(
                prompt_input.content,
                source_path=prompt_input.source_path,
            )
            self._finish_prompt_input(prompt_input)
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
            next_steps = "inspect the change or use create/refine --file"
        elif len(args) == 1:
            target = self._project_path(args[0])
            next_steps = "inspect the change or use create/refine --file"
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
            self._emit(f"Project: {manifest['name']}")
            self._emit(f"Root: {self.workspace.root}")
            self._emit(
                f"Manifest: {self.workspace.manifest_path} "
                f"({'present' if manifest['exists'] else 'not created'})"
            )
            self._emit(f"Prompts: {self.workspace.prompts_directory}")
            self._emit(
                f"Framework checkout: {manifest.get('framework_directory') or 'none'}"
            )
            return
        if args[0] != "init" or len(args) > 2:
            raise SystemExit("Use project show or project init [NAME].")
        existed = self.workspace.manifest_path.exists()
        manifest = self.workspace.initialize_project(
            name=args[1] if len(args) == 2 else None
        )
        result = "already exists" if existed else "created"
        self._success(
            f"Project manifest {result}: {self.workspace.manifest_path}"
        )
        self._emit(f"Project: {manifest['name']}")
        self._emit(f"Prompt ledger: {self.workspace.prompt_index_path}")

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

    def manage_task(self, args: list[str]) -> None:
        if len(args) > 1 or (
            args and args[0].lower() not in {"show", "path", "history"}
        ):
            raise SystemExit("Use task, task show, task path, or task history.")
        action = args[0].lower() if args else "summary"
        if action == "history":
            records = self.workspace.list_requests()
            if not records:
                self._emit_table(
                    "Task history",
                    [("Status", "none; use create or refine", "warning")],
                )
                return
            self._emit("Task history")
            self._emit("────────────")
            self._emit(
                "  Request                  Kind        Prompt  "
                "Refreshes                 Created"
            )
            for record in records:
                self._emit(
                    f"  {str(record['request_id']):24} "
                    f"{str(record['kind']):11} "
                    f"{str(record.get('prompt_id') or '—'):6}  "
                    f"{str(record.get('refreshes_request') or '—'):24}  "
                    f"{record.get('created_at') or '—'}"
                )
            self._emit()
            return

        record = self._ensure_current_task_fresh()
        if record is None:
            if action in {"show", "path"}:
                raise SystemExit(
                    "No current task. Use create or refine to prepare one."
                )
            self._emit_table(
                "Current task",
                [("Status", "none; use create or refine", "warning")],
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
        self._emit_table(
            "Current task",
            [
                ("Status", "ready", "success"),
                ("Kind", record["kind"], None),
                ("Request", record["request_id"], None),
                ("Prompt", record.get("prompt_id") or "—", None),
                ("Workflow", record.get("workflow_spec") or "new workflow", None),
                ("Refreshes", record.get("refreshes_request") or "—", None),
                ("Context", "matches the current ordered prompt ledger", "success"),
                ("File", ".zippergen/current-task.md", None),
                ("Next", "assistant codex · assistant claude", None),
            ],
        )

    def run_assistant(self, args: list[str]) -> None:
        if len(args) > 1 or (
            args and args[0].lower() not in {"codex", "claude"}
        ):
            raise SystemExit("Use assistant, assistant codex, or assistant claude.")
        assistant = args[0].lower() if args else "codex"
        record = self._ensure_current_task_fresh()
        if record is None:
            raise SystemExit(
                "No current task. Use create or refine before starting the assistant."
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
        if assistant == "codex":
            command = [
                executable,
                "--cd",
                str(self.workspace.root),
                instruction,
            ]
        else:
            command = [executable, instruction]
        try:
            completed = subprocess.run(
                command,
                cwd=self.workspace.root,
                check=False,
            )
        except OSError as exc:
            raise SystemExit(
                f"Could not start {tool}: {exc}"
            ) from exc
        if completed.returncode != 0:
            raise SystemExit(
                f"{assistant.capitalize()} exited with status "
                f"{completed.returncode}; the task remains at "
                f"{self.workspace.current_task_path}."
            )
        self._success(
            f"{'Claude Code' if assistant == 'claude' else 'Codex'} session ended. "
            "Inspect the generated files, then use current, validate, and show."
        )

    def show_current(self) -> None:
        from zippergen.serve import _validate_workflow

        state = self.workspace.load()
        manifest = self.workspace.project_manifest()
        records = self.workspace.list_prompts()
        active_prompts = sum(bool(record["active"]) for record in records)
        request = self._ensure_current_task_fresh(announce=False)
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
                    "Prompts",
                    f"{active_prompts} active · "
                    f"{len(records) - active_prompts} archived · "
                    f"{len(records)} total",
                    None,
                ),
                (
                    "Task",
                    (
                        f"{request['request_id']} ({request['kind']}) — "
                        ".zippergen/current-task.md"
                        if request
                        else "none; use create or refine"
                    ),
                    "success" if request else "warning",
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
            active_models = self._llm_action_lifelines(workflow, module)
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
                kind, provider_status = self._provider_readiness(provider)
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
                    ("Human actions", "0 — none", None),
                    ("Effects", "0 — none", None),
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
            kind, status = self._provider_readiness(provider)
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

        if not args or args == ["show"]:
            if not args:
                choices = ["Default for all unassigned lifelines"] + [
                    f"{name} ({', '.join(actions)})"
                    for name, actions in active.items()
                ]
                selected = str(self._select("Configure models", choices))
                if selected == choices[0]:
                    entered = self.input(f"Default model [{default}]: ").strip()
                    if entered:
                        default = _validate_model_spec(entered)
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
                profile = self.workspace.save_model_profile(
                    current,
                    default=default,
                    lifelines=overrides,
                )
            self._emit_models(
                workflow=workflow,
                module=module,
                profile=profile,
            )
            return

        action = args[0].lower()
        if action == "default" and len(args) == 2:
            default = _validate_model_spec(args[1])
        elif action == "set" and len(args) == 3:
            lifeline, spec = args[1:]
            if lifeline not in active:
                available = ", ".join(active) or "none"
                raise SystemExit(
                    f"{lifeline!r} has no LLM actions. LLM-active lifelines: "
                    f"{available}."
                )
            overrides[lifeline] = _validate_model_spec(spec)
        elif action == "reset" and len(args) == 2:
            if args[1].lower() == "all":
                default = default_llm_spec(module)
                overrides = {}
            else:
                overrides.pop(args[1], None)
        else:
            raise SystemExit(
                "Use models, models show, models default SPEC, models set "
                "LIFELINE SPEC, or models reset LIFELINE|all."
            )

        saved = self.workspace.save_model_profile(
            current,
            default=default,
            lifelines=overrides,
        )
        self._success(f"Saved model routing for {workflow.name}.")
        self._emit_models(workflow=workflow, module=module, profile=saved)

    def _provider_readiness(self, provider: str) -> tuple[StatusKind, str]:
        canonical = _canonical_provider(provider)
        if canonical == "mock":
            return "success", "ready; built in"
        profiles = self.workspace.provider_profiles()
        if canonical == "local":
            base_url = profiles.get("local", {}).get(
                "base_url",
                os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"),
            )
            return "warning", f"endpoint {base_url}; availability unchecked"
        secret_name = _PROVIDER_SECRETS.get(canonical)
        if secret_name is None:
            return "error", "unsupported"
        if os.environ.get(secret_name):
            return "success", f"ready; {secret_name} is in the environment"
        if self.workspace.load_secrets().get(secret_name):
            return "success", f"ready; {secret_name} is in private Studio storage"
        return "warning", f"not configured; use 'providers set {canonical}'"

    def _provider_status(self, provider: str) -> str:
        return self._provider_readiness(provider)[1]

    def _emit_providers(self) -> None:
        self._emit("Model providers")
        for provider in _SUPPORTED_PROVIDERS:
            kind, status = self._provider_readiness(provider)
            self._status(kind, f"{provider}: {status}", indent=2)
        self._emit("API-key values are never displayed or written to the project.")

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
                if not base_url.startswith(("http://", "https://")):
                    raise SystemExit("A local provider URL must begin with http:// or https://.")
                self.workspace.save_provider_profile(
                    "local",
                    {"kind": "local", "base_url": base_url},
                )
                self._success(f"Configured local provider endpoint: {base_url}")
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
            "providers set local [URL], or providers reset NAME."
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
        if isinstance(overrides, dict):
            for lifeline, model in sorted(overrides.items()):
                arguments.extend(["--llm-for", f"{lifeline}={model}"])
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

    def _active_prompt_ids(self) -> tuple[str, ...]:
        return tuple(
            str(record["id"])
            for record in self.workspace.list_prompts()
            if record["active"]
        )

    def _task_ledger_instruction(
        self,
        *,
        prompt_id: str | None,
        label: str,
        refreshes_request: str | None = None,
    ) -> str:
        if refreshes_request is None:
            return f"The immediate {label} is {prompt_id or 'not identified'}."
        return (
            f"This task refreshes {refreshes_request} because the project prompt "
            "ledger changed. The active ledger below is authoritative; its prompt "
            "IDs, order, status, and text were captured immediately before this "
            f"task was written. The original {label} was "
            f"{prompt_id or 'not identified'}."
        )

    def _creation_task_content(
        self,
        *,
        prompt_id: str | None,
        refreshes_request: str | None = None,
    ) -> str:
        context = self.workspace.prompt_context()
        ledger_instruction = self._task_ledger_instruction(
            prompt_id=prompt_id,
            label="request",
            refreshes_request=refreshes_request,
        )
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

The active project prompt ledger is the durable design context. Read it in
order. Later prompts take precedence only where they explicitly change an
earlier requirement; preserve every unaffected earlier requirement.
{ledger_instruction}

{context}

Before editing, summarize participants, owned inputs and outputs, messages,
action kinds, owned decisions and loops, deployment requirements, retry and
safety assumptions, and acceptance examples. Then create visible Python source
and focused mock/fake tests. Run validation, show the communication-only and
full code views, and inspect every new participant's exact local projection.
Do not deploy or start a service. Report generated files, assumptions, and
verification results.
"""

    def _refinement_task_content(
        self,
        *,
        workflow_spec: str,
        prompt_id: str | None,
        baseline_file: str | Path,
        refreshes_request: str | None = None,
    ) -> str:
        context = self.workspace.prompt_context()
        ledger_instruction = self._task_ledger_instruction(
            prompt_id=prompt_id,
            label="change",
            refreshes_request=refreshes_request,
        )
        return f"""# Current ZipperGen task

This generated task is the complete instruction for the coding assistant.
Work in the project root {self.workspace.root}. Keep workflow source and tests
visible in the repository. Do not deploy or start a service.

## Repository guidance

{self._assistant_skill_instructions()}

## Task

Refine {workflow_spec} using the active project prompt ledger below. Read it in
order. Later prompts take precedence only where they explicitly change or
contradict an earlier requirement; preserve every unaffected earlier
requirement. {ledger_instruction}

{context}

The semantic baseline is {baseline_file}.
Preserve all behavior not explicitly changed.
Update source, deployment metadata, and focused tests together when needed.
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
    ) -> dict[str, object] | None:
        record = self.workspace.current_request()
        if record is None:
            return None
        fingerprint = self.workspace.prompt_ledger_fingerprint()
        if record.get("prompt_ledger_fingerprint") == fingerprint:
            return record
        kind = str(record.get("kind") or "")
        prompt_id = str(record.get("prompt_id") or "") or None
        workflow_spec = str(record.get("workflow_spec") or "") or None
        baseline_file = str(record.get("baseline_file") or "") or None
        refreshes_request = str(record["request_id"])
        if kind == "create":
            content = self._creation_task_content(
                prompt_id=prompt_id,
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
                prompt_id=prompt_id,
                baseline_file=baseline_file,
                refreshes_request=refreshes_request,
            )
        else:
            raise WorkspaceError(
                f"Cannot refresh unsupported task kind {kind!r}. "
                "Use create or refine."
            )
        prompt = str(record.get("prompt") or "")
        if prompt_id is not None:
            try:
                prompt = str(self.workspace.prompt(prompt_id)["content"])
            except WorkspaceError:
                pass
        active_prompt_ids = self._active_prompt_ids()
        refreshed = self.workspace.save_request(
            kind=kind,
            prompt=prompt,
            content=content,
            workflow_spec=workflow_spec,
            prompt_id=prompt_id,
            active_prompt_ids=active_prompt_ids,
            prompt_ledger_fingerprint=fingerprint,
            baseline_file=baseline_file,
            refreshes_request=refreshes_request,
        )
        if announce:
            ids = ", ".join(active_prompt_ids) or "none"
            self._success(
                f"Task refreshed from the current prompt ledger: {ids}."
            )
        return refreshed

    def create_request(
        self,
        prompt: str,
        *,
        source_path: str | Path | None = None,
    ) -> None:
        if not prompt:
            prompt = self.input("Describe the workflow: ").strip()
        if not prompt:
            raise SystemExit("The workflow description must not be empty.")
        design_prompt = self.workspace.add_prompt(
            kind="initial",
            content=prompt,
            source_path=source_path,
        )
        active_prompt_ids = self._active_prompt_ids()
        prompt_fingerprint = self.workspace.prompt_ledger_fingerprint()
        content = self._creation_task_content(
            prompt_id=str(design_prompt["id"]),
        )
        self.workspace.save_request(
            kind="create",
            prompt=prompt,
            content=content,
            prompt_id=str(design_prompt["id"]),
            active_prompt_ids=active_prompt_ids,
            prompt_ledger_fingerprint=prompt_fingerprint,
        )
        prompt_action = (
            "registered" if design_prompt["created"] else "already active"
        )
        self._emit_table(
            "Creation",
            [
                (
                    "Prompt",
                    f"{design_prompt['id']} {prompt_action} — "
                    f"{design_prompt['file']}",
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
    ) -> None:
        current, workflow, module = self._current_context()
        if not prompt:
            prompt = self.input("Describe the change: ").strip()
        if not prompt:
            raise SystemExit("The refinement description must not be empty.")
        self.workspace.requests_directory.mkdir(parents=True, exist_ok=True)
        baseline = self.workspace.requests_directory / (
            f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1_000_000_000:09d}"
            "-semantic-before.json"
        )
        baseline.write_text(
            json.dumps(semantic_snapshot(workflow, module), indent=2, default=str) + "\n"
        )
        design_prompt = self.workspace.add_prompt(
            kind="refinement",
            content=prompt,
            source_path=source_path,
            workflow_spec=current,
        )
        active_prompt_ids = self._active_prompt_ids()
        prompt_fingerprint = self.workspace.prompt_ledger_fingerprint()
        content = self._refinement_task_content(
            workflow_spec=current,
            prompt_id=str(design_prompt["id"]),
            baseline_file=baseline,
        )
        self.workspace.save_request(
            kind="refine",
            prompt=prompt,
            content=content,
            workflow_spec=current,
            prompt_id=str(design_prompt["id"]),
            active_prompt_ids=active_prompt_ids,
            prompt_ledger_fingerprint=prompt_fingerprint,
            baseline_file=baseline,
        )
        prompt_action = (
            "registered" if design_prompt["created"] else "already active"
        )
        self._emit_table(
            "Refinement",
            [
                (
                    "Prompt",
                    f"{design_prompt['id']} {prompt_action} — "
                    f"{design_prompt['file']}",
                    "success",
                ),
                ("Workflow", current, None),
                ("Baseline", baseline, "success"),
                ("Task", ".zippergen/current-task.md", "success"),
                ("Next", "assistant codex · assistant claude", None),
                ("Inspect", "task · task show · task history", None),
            ],
        )
