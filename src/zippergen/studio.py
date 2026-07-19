"""A lightweight, discoverable project shell for ZipperGen development."""

from __future__ import annotations

import json
import os
import shlex
import time
from collections.abc import Callable
from pathlib import Path

from zippergen.dev import default_llm_spec, run_dev
from zippergen.models import normalize_llm_overrides
from zippergen.semantic import semantic_snapshot, workflow_semantics
from zippergen.view import ViewOptions, workflow_view_data
from zippergen.workspace import Workspace, WorkspaceError


InputFunc = Callable[[str], str]
OutputFunc = Callable[[str], object]


_HELP = """Commands:
  create [PROMPT]                prepare a new-workflow coding-assistant brief
  create --file PATH             read a multiline workflow prompt from a file
  use [PATH.py:WORKFLOW]         select a workflow; no argument opens a selector
  current                        show project, workflow, and current run context
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
  run [LLM]                      start a run; optional LLM overrides its default once
  resume                         resume the current incomplete run
  runs                           list managed development runs
  refine [PROMPT]                prepare a semantic refinement handoff
  refine --file PATH             read a multiline refinement prompt from a file
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
    ) -> None:
        self.workspace = workspace
        self.input = input_func
        self.output = output_func

    def _emit(self, value: object = "") -> None:
        self.output(str(value))

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
                self._emit("\nUse 'exit' to leave Studio.")
                continue
            try:
                if not self.execute(line):
                    return 0
            except KeyboardInterrupt:
                self._emit(
                    "\nCommand interrupted. Use 'current' to inspect context; "
                    "use 'resume' for an incomplete managed run."
                )
            except (SystemExit, WorkspaceError, ValueError) as exc:
                self._emit(f"Error: {exc}")

    def execute(self, line: str) -> bool:
        try:
            parts = shlex.split(line)
        except ValueError as exc:
            self._emit(f"Could not parse command: {exc}")
            return True
        if not parts:
            return True
        command, *args = parts
        command = command.lower()
        if command in {"exit", "quit"}:
            return False
        if command in {"help", "?"}:
            self._emit(_HELP.rstrip())
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
            self.create_request(self._request_prompt(args, command="create"))
        elif command == "refine":
            self.refine_request(self._request_prompt(args, command="refine"))
        elif command == "deploy":
            self.deploy_workflow(args)
        elif command in {"status", "doctor", "logs", "start", "restart", "stop"}:
            self.deployment_action(command, args)
        else:
            self._emit(
                f"Unknown command: {command}. Type 'help' for available commands."
            )
        return True

    def _request_prompt(self, args: list[str], *, command: str) -> str:
        if not args:
            return ""
        if args[0] == "--file":
            if len(args) != 2:
                raise SystemExit(f"Use {command} --file PATH.")
            entered = Path(args[1]).expanduser()
            prompt_file = (
                entered
                if entered.is_absolute()
                else self.workspace.root / entered
            ).resolve()
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
            try:
                displayed = prompt_file.relative_to(self.workspace.root)
            except ValueError:
                displayed = prompt_file
            self._emit(f"Prompt file: {displayed}")
            return prompt
        if "--file" in args:
            raise SystemExit(f"Use {command} --file PATH.")
        return " ".join(args).strip()

    def _current_context(self):
        from zippergen.serve import load_workflow_spec

        current = self.workspace.current_workflow
        if not current:
            raise SystemExit("No workflow selected. Use 'use' or 'create' first.")
        workflow, module = load_workflow_spec(self.workspace.absolute_spec(current))
        return current, workflow, module

    def show_current(self) -> None:
        state = self.workspace.load()
        self._emit(f"Project: {self.workspace.root}")
        self._emit(f"Workspace: {self.workspace.directory}")
        self._emit(f"Workflow: {state.get('current_workflow') or 'none'}")
        run = self.workspace.current_run()
        if run is None:
            self._emit("Run: none")
        else:
            self._emit(f"Run: {run['run_id']} ({run['status']})")
            self._emit(f"Store: managed at {run['store']}")
        self._emit(f"Deployment: {state.get('last_deployment') or 'none'}")
        if state.get("current_workflow"):
            profile = self._run_model_profile()
            self._emit(f"Default LLM: {profile['default']}")
            overrides = profile.get("lifelines") or {}
            if isinstance(overrides, dict) and overrides:
                rendered = ", ".join(
                    f"{name}={spec}" for name, spec in sorted(overrides.items())
                )
                self._emit(f"LLM overrides: {rendered}")

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
        self._emit(f"Current workflow: {canonical} ({workflow.name})")

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
        self._emit(f"Workflow {workflow.name}: {verdict}")
        for check in result["checks"]:  # type: ignore[index]
            self._emit(
                f"{str(check['status']).upper():4} {check['name']}: {check['detail']}"
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
                        default = entered
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
                        overrides[lifeline] = entered
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
            default = args[1]
        elif action == "set" and len(args) == 3:
            lifeline, spec = args[1:]
            if lifeline not in active:
                available = ", ".join(active) or "none"
                raise SystemExit(
                    f"{lifeline!r} has no LLM actions. LLM-active lifelines: "
                    f"{available}."
                )
            overrides[lifeline] = spec
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
        self._emit_models(workflow=workflow, module=module, profile=saved)

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

    def create_request(self, prompt: str) -> None:
        if not prompt:
            prompt = self.input("Describe the workflow: ").strip()
        if not prompt:
            raise SystemExit("The workflow description must not be empty.")
        content = f"""Use $zippergen-workflows.

Create a new ZipperGen Python workflow in this project from the requirements
below. Choose a clear module and workflow name under workflows/ unless the
project has a more appropriate established location.

Requirements:
{prompt}

Before editing, summarize participants, owned inputs and outputs, messages,
action kinds, owned decisions and loops, deployment requirements, retry and
safety assumptions, and acceptance examples. Then create visible Python source
and focused mock/fake tests. Run validation, show the communication-only and
full code views, and inspect every new participant's exact local projection.
Do not deploy or start a service. Report generated files, assumptions, and
verification results.
"""
        record = self.workspace.save_request(
            kind="create",
            prompt=prompt,
            content=content,
        )
        self._emit(f"Creation brief: {record['content_file']}")
        self._emit("Pass this brief to a repository-aware coding assistant:")
        self._emit(content.rstrip())

    def refine_request(self, prompt: str) -> None:
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
        content = f"""Use $zippergen-workflows.

Refine {current} from the requirements below.

Requested change:
{prompt}

The semantic baseline is {baseline}.
Preserve all behavior not explicitly changed.
Update source, deployment metadata, and focused tests together when needed.
Validate the result, show communication-only and full code views,
inspect every changed participant's exact local projection, and compare the
result with the baseline using `zippergen diff`. Do not deploy or start a
service. Report assumptions, intended semantic changes, preserved behavior,
and verification results.
"""
        record = self.workspace.save_request(
            kind="refine",
            prompt=prompt,
            content=content,
            workflow_spec=current,
        )
        self._emit(f"Refinement brief: {record['content_file']}")
        self._emit("Pass this brief to a repository-aware coding assistant:")
        self._emit(content.rstrip())
