"""A lightweight, discoverable project shell for ZipperGen development."""

from __future__ import annotations

import json
import os
import shlex
import time
from collections.abc import Callable
from pathlib import Path

from zippergen.dev import run_dev
from zippergen.semantic import semantic_snapshot
from zippergen.view import ViewOptions, workflow_view_data
from zippergen.workspace import Workspace, WorkspaceError


InputFunc = Callable[[str], str]
OutputFunc = Callable[[str], object]


_HELP = """Commands:
  create                         describe a new workflow for a coding assistant
  use [PATH.py:WORKFLOW]         select a workflow; no argument opens a selector
  current                        show project, workflow, and current run context
  show | inspect                 choose a code-first semantic view
  show overview|protocol|communications|actions|full
  show agent [NAME]              exact local projection (selector if omitted)
  show agents [NAME ...]         selected-participant focus view
  validate                       validate the current workflow
  run [LLM]                      collect inputs and start a fresh durable run
  resume                         resume the current incomplete run
  runs                           list managed development runs
  refine [PROMPT]                save a semantic refinement handoff
  deploy [NAME]                  guided deployment of the current workflow
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
        elif command == "run":
            if len(args) > 1:
                raise SystemExit("Use run or run LLM_SPEC.")
            run_dev(
                self.workspace,
                llm=args[0] if args else None,
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
            self.create_request(" ".join(args).strip())
        elif command == "refine":
            self.refine_request(" ".join(args).strip())
        elif command == "deploy":
            self.deploy_workflow(args)
        elif command in {"status", "doctor", "logs", "start", "restart", "stop"}:
            self.deployment_action(command, args)
        else:
            self._emit(f"Unknown command: {command}. Type 'help' for available commands.")
        return True

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

        if len(args) > 1:
            raise SystemExit("Use deploy or deploy NAME.")
        current, workflow, module = self._current_context()
        target = self.workspace.absolute_spec(current)
        spec = deployment_spec_from_module(module)
        name = _slug(
            args[0]
            if args
            else spec.name or _deployment_name_from_workflow(target, workflow)
        )
        self._emit(f"Guided deployment: {name}")
        arguments = ["deploy", target]
        if args:
            arguments.extend(["--name", name])
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
