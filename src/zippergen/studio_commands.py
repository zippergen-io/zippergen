"""Declarative metadata for the ZipperGen Studio command surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CommandRisk = Literal["read-only", "configuration", "execution", "destructive"]
ParticipantScope = Literal["none", "one", "many"]


@dataclass(frozen=True)
class CommandSpec:
    path: tuple[str, ...]
    usage: str
    summary: str
    risk: CommandRisk
    natural: bool = True
    primary: bool = False
    hidden: bool = False


@dataclass(frozen=True)
class WorkflowViewSpec:
    command: str
    label: str
    description: str
    detail: str = "protocol"
    aliases: tuple[str, ...] = ()
    communications_only: bool = False
    participants: ParticipantScope = "none"


WORKFLOW_VIEWS: tuple[WorkflowViewSpec, ...] = (
    WorkflowViewSpec(
        "overview",
        "Overview",
        "compact workflow summary",
        detail="overview",
    ),
    WorkflowViewSpec(
        "protocol",
        "Protocol",
        "global protocol code",
    ),
    WorkflowViewSpec(
        "communications",
        "Communications only",
        "communications only",
        aliases=("communication", "communications only"),
        communications_only=True,
    ),
    WorkflowViewSpec(
        "actions",
        "Actions and prompts",
        "actions and prompts",
        detail="actions",
        aliases=("actions and prompts",),
    ),
    WorkflowViewSpec(
        "full",
        "Complete workflow",
        "complete workflow code",
        detail="full",
        aliases=("complete", "complete workflow"),
    ),
    WorkflowViewSpec(
        "agent",
        "One participant",
        "one exact local projection",
        aliases=("one participant",),
        participants="one",
    ),
    WorkflowViewSpec(
        "agents",
        "Selected participants",
        "selected-participant focus view",
        aliases=("selected participants",),
        participants="many",
    ),
)


def workflow_view_spec(value: str) -> WorkflowViewSpec | None:
    normalized = value.strip().casefold()
    return next(
        (
            view
            for view in WORKFLOW_VIEWS
            if normalized == view.command or normalized in view.aliases
        ),
        None,
    )


def workflow_view_completions() -> tuple[tuple[str, str], ...]:
    return tuple((view.command, view.description) for view in WORKFLOW_VIEWS)


COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec(
        ("workflow",),
        "workflow",
        "show development status and next step",
        "read-only",
        primary=True,
    ),
    CommandSpec(
        ("workflow", "edit-spec"),
        "workflow edit-spec [DESCRIPTION|--file PATH] [--editor COMMAND]",
        "write or edit the canonical specification",
        "configuration",
        primary=True,
    ),
    CommandSpec(
        ("workflow", "edit-refinement"),
        "workflow edit-refinement [CHANGE|--file PATH] [--editor COMMAND]",
        "write or discard the scratch refinement",
        "configuration",
    ),
    CommandSpec(
        ("workflow", "refine-spec"),
        "workflow refine-spec [codex|claude]",
        "apply the refinement to the specification",
        "execution",
    ),
    CommandSpec(
        ("workflow", "adopt"),
        "workflow adopt [codex|claude]",
        "write a specification for an implementation Studio did not generate",
        "execution",
    ),
    CommandSpec(
        ("workflow", "list"),
        "workflow list",
        "list discovered workflow entry points",
        "read-only",
    ),
    CommandSpec(
        ("workflow", "import"),
        "workflow import PATH.py[:WORKFLOW]",
        "import or replace this project's workflow and local files",
        "configuration",
    ),
    CommandSpec(
        ("workflow", "files"),
        "workflow files",
        "list files used by the project workflow",
        "read-only",
    ),
    CommandSpec(
        ("workflow", "show"),
        "workflow show [VIEW]",
        "inspect source or a semantic workflow view",
        "read-only",
        primary=True,
    ),
    CommandSpec(
        ("workflow", "diff"),
        "workflow diff",
        "compare current intent and semantics with the available baseline",
        "read-only",
        primary=True,
    ),
    CommandSpec(
        ("workflow", "status"),
        "workflow status [--details]",
        "show implementation status and next step",
        "read-only",
    ),
    CommandSpec(
        ("workflow", "implement"),
        "workflow implement [codex|claude] [--rerun]",
        "implement, summarize, and offer one coherent Git commit",
        "execution",
        primary=True,
    ),
    CommandSpec(
        ("workflow", "validate"),
        "workflow validate",
        "check structural validity and every projection",
        "read-only",
    ),
    CommandSpec(
        ("workflow", "history"),
        "workflow history",
        "show specification and implementation history",
        "read-only",
    ),
    CommandSpec(
        ("workflow", "path"),
        "workflow path",
        "print the canonical specification path",
        "read-only",
    ),
    CommandSpec(
        ("model",),
        "model",
        "show providers, configurations, and assignments",
        "read-only",
    ),
    CommandSpec(
        ("model", "setup"),
        "model setup",
        "guide provider, configuration, and assignment setup",
        "configuration",
    ),
    CommandSpec(
        ("model", "provider"),
        "model provider list|configure|check|remove ...",
        "manage provider connections",
        "configuration",
    ),
    CommandSpec(
        ("model", "config"),
        "model config list|create|show|check|edit|rename|remove ...",
        "manage reusable model configurations",
        "configuration",
    ),
    CommandSpec(
        ("model", "assignments"),
        "model assignments",
        "show participant routes and their last check",
        "read-only",
    ),
    CommandSpec(
        ("model", "assignments", "check"),
        "model assignments check",
        "check configurations used by the project workflow",
        "read-only",
    ),
    CommandSpec(
        ("model", "assign"),
        "model assign PARTICIPANT_OR_ACTION NAME [--site]",
        "assign a project configuration, optionally only on this site",
        "configuration",
    ),
    CommandSpec(
        ("model", "default"),
        "model default NAME [--site]",
        "set the project or site default configuration",
        "configuration",
    ),
    CommandSpec(
        ("model", "inherit"),
        "model inherit PARTICIPANT_OR_ACTION [--site]",
        "restore project or site inheritance",
        "configuration",
    ),
    CommandSpec(
        ("connector",),
        "connector",
        "show providers, configurations, human routes, and service bindings",
        "read-only",
    ),
    CommandSpec(
        ("connector", "setup"),
        "connector setup",
        "configure and bind the connectors required by this workflow",
        "configuration",
    ),
    CommandSpec(
        ("connector", "provider"),
        "connector provider list|configure|check|remove ...",
        "manage connector provider credentials",
        "configuration",
    ),
    CommandSpec(
        ("connector", "config"),
        "connector config list|create|show|edit|check|rename|remove ...",
        "manage reusable connector configurations",
        "configuration",
    ),
    CommandSpec(
        ("connector", "assignments"),
        "connector assignments",
        "show effective human-action routes",
        "read-only",
    ),
    CommandSpec(
        ("connector", "assignments", "check"),
        "connector assignments check",
        "check configurations used by human actions",
        "read-only",
    ),
    CommandSpec(
        ("connector", "assign"),
        "connector assign PARTICIPANT_OR_ACTION CONFIGURATION",
        "route human actions through a configuration",
        "configuration",
    ),
    CommandSpec(
        ("connector", "inherit"),
        "connector inherit PARTICIPANT_OR_ACTION",
        "restore the inherited or local human route",
        "configuration",
    ),
    CommandSpec(
        ("connector", "bind"),
        "connector bind REQUIREMENT CONFIGURATION",
        "bind an explicit non-human capability requirement",
        "configuration",
    ),
    CommandSpec(
        ("run",),
        "run [MODEL] [--assistant TOOL]",
        "start a managed development run",
        "execution",
        primary=True,
    ),
    CommandSpec(
        ("run", "inspect"),
        "run inspect [PARTICIPANT] [--watch]",
        "show participant positions once or refresh them every second",
        "read-only",
    ),
    CommandSpec(
        ("run", "tasks"),
        "run tasks",
        "show pending decisions for the current development run",
        "read-only",
    ),
    CommandSpec(
        ("run", "approve"),
        "run approve [TASK_ID] [yes|no|VALUE]",
        "review and answer a pending development-run decision",
        "execution",
    ),
    CommandSpec(
        ("run", "trace"),
        "run trace",
        "show recent events for the current development run",
        "read-only",
    ),
    CommandSpec(
        ("resume",), "resume", "resume the current incomplete run", "execution"
    ),
    CommandSpec(("runs",), "runs", "list managed development runs", "read-only"),
    CommandSpec(
        ("deploy",),
        "deploy [NAME] [--no-start]",
        "deploy an implementation matching the current specification",
        "execution",
        primary=True,
    ),
    CommandSpec(
        ("deploy", "list"),
        "deploy list",
        "list deployment profiles",
        "read-only",
    ),
    CommandSpec(
        ("deploy", "show"),
        "deploy show [NAME]",
        "show bundle, service, run, and store health",
        "read-only",
    ),
    CommandSpec(
        ("deploy", "inspect"),
        "deploy inspect [NAME] [PARTICIPANT] [--watch]",
        "show participant positions once or refresh them every second",
        "read-only",
    ),
    CommandSpec(
        ("deploy", "doctor"),
        "deploy doctor [NAME]",
        "run detailed deployment readiness checks",
        "read-only",
    ),
    CommandSpec(
        ("deploy", "logs"),
        "deploy logs [NAME]",
        "show deployment logs",
        "read-only",
    ),
    CommandSpec(
        ("deploy", "logs", "reset"),
        "deploy logs reset [NAME] [--yes]",
        "archive and reset visible deployment log history",
        "destructive",
    ),
    CommandSpec(
        ("deploy", "tasks"),
        "deploy tasks [NAME]",
        "show pending decisions and their complete context",
        "read-only",
    ),
    CommandSpec(
        ("deploy", "approve"),
        "deploy approve [NAME]",
        "review and answer a pending deployment decision",
        "execution",
    ),
    CommandSpec(
        ("deploy", "trace"),
        "deploy trace [NAME]",
        "show recent events for a deployment",
        "read-only",
    ),
    CommandSpec(
        ("deploy", "storage"),
        "deploy storage [NAME]",
        "show durable-store, log, and recovery storage",
        "read-only",
    ),
    CommandSpec(
        ("deploy", "storage", "compact"),
        "deploy storage compact [NAME] [--yes]",
        "compact recovery-safe events and stopped-service logs",
        "destructive",
    ),
    CommandSpec(
        ("deploy", "start"),
        "deploy start [NAME]",
        "start a deployment",
        "execution",
    ),
    CommandSpec(
        ("deploy", "restart"),
        "deploy restart [NAME]",
        "restart a deployment",
        "execution",
    ),
    CommandSpec(
        ("deploy", "stop"),
        "deploy stop [NAME]",
        "stop a deployment",
        "execution",
    ),
    CommandSpec(
        ("deploy", "remove"),
        "deploy remove [NAME] [--purge] [--yes]",
        "remove a deployment, keeping its durable store unless purged",
        "destructive",
    ),
    CommandSpec(
        ("current",),
        "current",
        "show the active workflow and execution context",
        "read-only",
    ),
    CommandSpec(
        ("studio",),
        "studio",
        "inspect or operate the Studio process",
        "read-only",
        hidden=True,
    ),
    CommandSpec(
        ("studio", "doctor"),
        "studio doctor",
        "check local Studio readiness",
        "read-only",
    ),
    CommandSpec(
        ("studio", "restart"),
        "studio restart",
        "replace this process and reload installed source",
        "execution",
    ),
    CommandSpec(
        ("studio", "update"),
        "studio update",
        "fast-forward a source checkout, synchronize it, and restart Studio",
        "execution",
    ),
    CommandSpec(
        ("project",),
        "project",
        "show everything recorded in this project",
        "read-only",
    ),
    CommandSpec(
        ("project", "init"),
        "project init [NAME]",
        "create the project manifest",
        "configuration",
    ),
    CommandSpec(
        ("project", "rename"),
        "project rename NAME",
        "change the logical project name",
        "configuration",
    ),
    CommandSpec(
        ("project", "reset"),
        "project reset [fresh|state] [--yes]",
        "archive and reset project state",
        "destructive",
    ),
    CommandSpec(
        ("settings",), "settings", "show global Studio preferences", "read-only"
    ),
    CommandSpec(
        ("settings", "show"),
        "settings show",
        "show global Studio preferences",
        "read-only",
    ),
    CommandSpec(
        ("settings", "set"),
        "settings set NAME VALUE",
        "set a global preference",
        "configuration",
    ),
    CommandSpec(
        ("settings", "reset"),
        "settings reset [NAME|all]",
        "restore global defaults",
        "configuration",
    ),
    CommandSpec(
        ("language",), "language", "show interpreter and learning status", "read-only"
    ),
    CommandSpec(
        ("language", "show"),
        "language show",
        "show interpreter and learning status",
        "read-only",
    ),
    CommandSpec(
        ("language", "set"),
        "language set auto|codex|claude|off",
        "choose the interpreter fallback",
        "configuration",
    ),
    CommandSpec(
        ("language", "learning"),
        "language learning on|off",
        "control private interpretation learning",
        "configuration",
    ),
    CommandSpec(
        ("language", "history"),
        "language history",
        "show interpreted requests",
        "read-only",
    ),
    CommandSpec(
        ("language", "learned"),
        "language learned",
        "show reusable interpretations",
        "read-only",
    ),
    CommandSpec(
        ("language", "forget"),
        "language forget ID|all",
        "forget learned interpretations",
        "destructive",
    ),
    CommandSpec(
        ("editor",), "editor", "show the effective terminal editor", "read-only"
    ),
    CommandSpec(
        ("editor", "show"),
        "editor show",
        "show the effective terminal editor",
        "read-only",
    ),
    CommandSpec(
        ("editor", "set"),
        "editor set COMMAND",
        "remember a terminal editor",
        "configuration",
    ),
    CommandSpec(
        ("editor", "reset"),
        "editor reset",
        "restore automatic editor discovery",
        "configuration",
    ),
    CommandSpec(
        ("edit",),
        "edit file PATH",
        "edit another project file",
        "configuration",
        hidden=True,
    ),
    CommandSpec(
        ("edit", "file"),
        "edit file PATH [--editor COMMAND]",
        "edit a project file",
        "configuration",
    ),
    CommandSpec(
        ("ask",),
        "ask TEXT",
        "interpret and execute ordinary language",
        "configuration",
        natural=False,
    ),
    CommandSpec(
        ("plan",),
        "plan TEXT",
        "interpret ordinary language without executing",
        "read-only",
        natural=False,
    ),
    CommandSpec(
        ("help",),
        "help [all]",
        "show the happy path or complete command reference",
        "read-only",
    ),
    CommandSpec(("exit",), "exit", "leave Studio", "read-only", natural=False),
    CommandSpec(
        ("quit",), "quit", "alias for exit", "read-only", natural=False, hidden=True
    ),
    CommandSpec(("?",), "?", "alias for help", "read-only", natural=False, hidden=True),
)


# Pass-3 authoring lifecycle map. The root ``workflow`` command is the compact
# status observation; files, history, and path are narrower observation views.
WORKFLOW_COMMAND_MAP: dict[str, frozenset[tuple[str, ...]]] = {
    "transitions": frozenset(
        {
            ("workflow", "edit-spec"),
            ("workflow", "edit-refinement"),
            ("workflow", "refine-spec"),
            ("workflow", "adopt"),
            ("workflow", "implement"),
            ("workflow", "import"),
        }
    ),
    "observations": frozenset(
        {
            ("workflow",),
            ("workflow", "validate"),
            ("workflow", "show"),
            ("workflow", "diff"),
            ("workflow", "status"),
            ("workflow", "list"),
            ("workflow", "files"),
            ("workflow", "history"),
            ("workflow", "path"),
        }
    ),
}


def command_spec(parts: list[str] | tuple[str, ...]) -> CommandSpec | None:
    lowered = tuple(value.casefold() for value in parts)
    matches = [
        spec
        for spec in COMMANDS
        if len(spec.path) <= len(lowered) and lowered[: len(spec.path)] == spec.path
    ]
    return max(matches, key=lambda spec: len(spec.path), default=None)


def top_level_completions() -> tuple[tuple[str, str], ...]:
    seen: set[str] = set()
    values: list[tuple[str, str]] = []
    for spec in COMMANDS:
        name = spec.path[0]
        if spec.hidden or name in seen:
            continue
        seen.add(name)
        root = (
            next(candidate for candidate in COMMANDS if candidate.path == (name,))
            if any(candidate.path == (name,) for candidate in COMMANDS)
            else spec
        )
        values.append((name, root.summary))
    return tuple(values)


def subcommand_completions(parent: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        (spec.path[1], spec.summary)
        for spec in COMMANDS
        if len(spec.path) == 2 and spec.path[0] == parent and not spec.hidden
    )


def concise_help() -> str:
    return """Getting started:
  1. project init
  2. workflow edit-spec
  3. workflow implement
  4. workflow validate
  5. run
  6. deploy

Existing workflow:
  workflow import /path/to/workflow.py

Main areas:
  workflow    specify, implement, inspect, and validate
  model       configure, check, and assign models
  connector   configure and route human or service connections
  run         execute and inspect development runs
  deploy      create, inspect, and operate installed applications

Start actions:
  run         start a managed development run
  deploy      create or update an installed deployment

Useful now:
  current          show context and the next action
  run inspect      show where each participant currently is
  run tasks        show pending decisions for the current run
  studio doctor    check editor and assistant readiness
  help all         show every exact command
  exit             leave Studio

You may also describe what you want in ordinary language."""


def full_help() -> str:
    lines = [
        "All Studio commands:",
        "  NATURAL LANGUAGE                               describe an operation or workflow requirement",
    ]
    for spec in COMMANDS:
        if spec.hidden:
            continue
        lines.append(f"  {spec.usage:<46} {spec.summary}")
    return "\n".join(lines)


def natural_command_catalog() -> str:
    """Return the exact command vocabulary exposed to interpreter CLIs."""

    groups: dict[str, list[str]] = {
        "Read-only": [],
        "Local configuration and development": [],
        "Execution and deployment": [],
    }
    for spec in COMMANDS:
        if (
            not spec.natural
            or spec.hidden
            or len(spec.path) == 1
            and spec.path != ("deploy",)
            and any(
                candidate.path[:1] == spec.path and len(candidate.path) > 1
                for candidate in COMMANDS
            )
        ):
            continue
        if spec.risk == "read-only":
            group = "Read-only"
        elif spec.risk == "execution":
            group = "Execution and deployment"
        else:
            group = "Local configuration and development"
        groups[group].append(f"- {spec.usage}")
    lines: list[str] = []
    for title, commands in groups.items():
        lines.extend([f"{title}:", *commands, ""])
    lines.extend(
        [
            "Never produce exit, quit, arbitrary shell commands, chained shell ",
            "syntax, secret values, or a command not present in this catalogue.",
        ]
    )
    return "\n".join(lines)
