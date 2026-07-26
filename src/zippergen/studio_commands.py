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
        ("workflow", "create"),
        "workflow create [DESCRIPTION|--file PATH|--edit]",
        "write the initial accepted specification",
        "configuration",
        primary=True,
    ),
    CommandSpec(
        ("workflow", "refine"),
        "workflow refine [CHANGE|--file PATH|--edit] "
        "[--implement [--review]]",
        "create or reopen the pending refinement",
        "configuration",
    ),
    CommandSpec(
        ("workflow", "edit"),
        "workflow edit [spec|code]",
        "edit requirements or selected source",
        "configuration",
    ),
    CommandSpec(
        ("workflow", "list"),
        "workflow list",
        "list discovered workflow entry points",
        "read-only",
    ),
    CommandSpec(
        ("workflow", "select"),
        "workflow select [NUMBER|NAME|PATH.py:NAME]",
        "select a workflow entry point",
        "configuration",
    ),
    CommandSpec(
        ("workflow", "files"),
        "workflow files",
        "list files used by the selected workflow",
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
        "compare current intent and semantics with the review baseline",
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
        "workflow implement [codex|claude] [--review]",
        "run a coding assistant and return to Studio",
        "execution",
        primary=True,
    ),
    CommandSpec(
        ("workflow", "review"),
        "workflow review",
        "inspect, validate, run, and accept interactively",
        "read-only",
    ),
    CommandSpec(
        ("workflow", "validate"),
        "workflow validate",
        "check structural validity and every projection; does not accept",
        "read-only",
    ),
    CommandSpec(
        ("workflow", "accept"),
        "workflow accept [--yes]",
        "record human approval and the accepted intent/semantic baseline",
        "destructive",
    ),
    CommandSpec(
        ("workflow", "discard"),
        "workflow discard [--yes]",
        "archive a rejected refinement; does not revert files",
        "destructive",
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
        ("models",),
        "models",
        "show providers, configurations, and assignments",
        "read-only",
    ),
    CommandSpec(
        ("models", "setup"),
        "models setup",
        "guide provider, configuration, and assignment setup",
        "configuration",
    ),
    CommandSpec(
        ("models", "provider"),
        "models provider list|configure|check|remove ...",
        "manage provider connections",
        "configuration",
    ),
    CommandSpec(
        ("models", "config"),
        "models config list|create|show|check|edit|rename|remove ...",
        "manage reusable model configurations",
        "configuration",
    ),
    CommandSpec(
        ("models", "assignments"),
        "models assignments",
        "show participant routes and their last check",
        "read-only",
    ),
    CommandSpec(
        ("models", "assignments", "check"),
        "models assignments check",
        "check configurations used by the selected workflow",
        "read-only",
    ),
    CommandSpec(
        ("models", "assign"),
        "models assign LIFELINE NAME",
        "assign a configuration to a participant",
        "configuration",
    ),
    CommandSpec(
        ("models", "default"),
        "models default NAME",
        "set the inherited default configuration",
        "configuration",
    ),
    CommandSpec(
        ("models", "inherit"),
        "models inherit LIFELINE",
        "restore inheritance from the default",
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
        ("resume",), "resume", "resume the current incomplete run", "execution"
    ),
    CommandSpec(("runs",), "runs", "list managed development runs", "read-only"),
    CommandSpec(
        ("store",),
        "store",
        "list durable execution stores",
        "read-only",
    ),
    CommandSpec(
        ("store", "list"),
        "store list",
        "list run, deployment, and standalone stores",
        "read-only",
    ),
    CommandSpec(
        ("store", "show"),
        "store show [NUMBER|NAME]",
        "inspect durable state and ownership",
        "read-only",
    ),
    CommandSpec(
        ("store", "use"),
        "store use NUMBER|NAME",
        "select the store used by short store commands",
        "configuration",
    ),
    CommandSpec(
        ("store", "path"),
        "store path [NUMBER|NAME]",
        "print the exact SQLite path",
        "read-only",
    ),
    CommandSpec(
        ("store", "tasks"),
        "store tasks [NUMBER|NAME]",
        "show pending human tasks",
        "read-only",
    ),
    CommandSpec(
        ("store", "approve"),
        "store approve [TASK_ID] [yes|no|VALUE]",
        "complete a pending human task in the selected store",
        "execution",
    ),
    CommandSpec(
        ("store", "trace"),
        "store trace [NUMBER|NAME]",
        "show recent durable events",
        "read-only",
    ),
    CommandSpec(
        ("store", "create"),
        "store create NAME",
        "create an empty standalone store",
        "configuration",
    ),
    CommandSpec(
        ("store", "rename"),
        "store rename NUMBER|NAME NEW_NAME",
        "rename a stopped store and update its references",
        "configuration",
    ),
    CommandSpec(
        ("store", "delete"),
        "store delete NUMBER|NAME|all [--yes]",
        "archive store data recoverably",
        "destructive",
    ),
    CommandSpec(
        ("deploy",),
        "deploy [NAME] [--no-start] [--accepted|--unreviewed --reason TEXT]",
        "deploy accepted source by default with a review gate",
        "execution",
        primary=True,
    ),
    CommandSpec(
        ("deployment",),
        "deployment",
        "list named deployed applications",
        "read-only",
    ),
    CommandSpec(
        ("deployment", "list"),
        "deployment list",
        "list deployment profiles",
        "read-only",
    ),
    CommandSpec(
        ("deployment", "show"),
        "deployment show [NAME]",
        "show bundle, service, run, and store health",
        "read-only",
    ),
    CommandSpec(
        ("deployment", "doctor"),
        "deployment doctor [NAME]",
        "run detailed deployment readiness checks",
        "read-only",
    ),
    CommandSpec(
        ("deployment", "logs"),
        "deployment logs [NAME]",
        "show deployment logs",
        "read-only",
    ),
    CommandSpec(
        ("deployment", "tasks"),
        "deployment tasks [NAME]",
        "show pending decisions and their complete context",
        "read-only",
    ),
    CommandSpec(
        ("deployment", "approve"),
        "deployment approve [NAME]",
        "review and answer a pending deployment decision",
        "execution",
    ),
    CommandSpec(
        ("deployment", "start"),
        "deployment start [NAME]",
        "start a deployment",
        "execution",
    ),
    CommandSpec(
        ("deployment", "restart"),
        "deployment restart [NAME]",
        "restart a deployment",
        "execution",
    ),
    CommandSpec(
        ("deployment", "stop"),
        "deployment stop [NAME]",
        "stop a deployment",
        "execution",
    ),
    CommandSpec(
        ("status",), "status [NAME]", "show deployment status", "read-only", hidden=True
    ),
    CommandSpec(
        ("doctor",),
        "doctor [NAME]",
        "check deployment readiness",
        "read-only",
        hidden=True,
    ),
    CommandSpec(
        ("logs",), "logs [NAME]", "show deployment logs", "read-only", hidden=True
    ),
    CommandSpec(
        ("start",), "start [NAME]", "start a deployment", "execution", hidden=True
    ),
    CommandSpec(
        ("restart",),
        "restart [NAME]",
        "restart a deployment",
        "execution",
        hidden=True,
    ),
    CommandSpec(
        ("stop",), "stop [NAME]", "stop a deployment", "execution", hidden=True
    ),
    CommandSpec(("current",), "current", "show complete project context", "read-only"),
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
    CommandSpec(("project",), "project", "show project configuration", "read-only"),
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
        ("project", "show"),
        "project show",
        "show visible project configuration",
        "read-only",
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
  2. workflow create
  3. workflow implement
  4. workflow review
  5. run
  6. deploy

Three main areas:
  workflow    specify, implement, inspect, and validate
  models      configure, check, and assign models
  deployment  operate installed applications and their durable state

Start actions:
  run         start a managed development run
  deploy      create or update an installed deployment

Useful now:
  current          show context and the next action
  studio doctor    check editor and assistant readiness
  store             advanced development-state and recovery tools
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
