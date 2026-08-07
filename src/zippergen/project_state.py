"""Project state as properties and machines, for the agent-facing actions.

This module is deliberately independent of ZipperGen Studio. It reads a
`Workspace` and returns plain dataclasses; it renders nothing, prompts for
nothing, and holds no state of its own. Actions are built over it and the CLI
renders their results.

Two kinds of state, and the distinction matters:

*Properties* are observed. Whether a specification exists, whether the
workflow loads and projects, whether this machine is configured — none of
these are positions in a transition system, and nothing "moves through" them.

*Machines* are genuine transition systems with preconditions: runs and
deployments. Those live in their own modules; this one covers the properties
and the derived next action.

Deliberately absent: any notion of whether the implementation was *generated
from* the specification. Once a coding agent edits the workflow directly, a
byte fingerprint answers a question nobody asked, and correspondence between
prose and code is not machine-decidable. See AGENT-HARNESS-DESIGN.md §2.1a.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from zippergen.workspace import Workspace, WorkspaceError

SpecificationState = Literal["absent", "present"]
WorkflowState = Literal["absent", "present"]
ValidationState = Literal["valid", "invalid", "unknown"]


@dataclass(frozen=True)
class MissingRequirement:
    """One thing this machine needs before the project can run here."""

    what: str
    fix: str
    may_agent_run: bool = False


@dataclass(frozen=True)
class NextAction:
    """The single next step, derived rather than left to the caller.

    Callers must read this rather than re-deriving it. The ordering has
    changed more than once, and an agent reasoning from first principles
    reproduces whichever version it happens to remember.
    """

    command: str
    reason: str
    may_agent_run: bool


@dataclass(frozen=True)
class ProjectState:
    """Everything an action needs to decide what is true and what is next."""

    root: str
    name: str
    manifest_present: bool
    specification: SpecificationState
    workflow: WorkflowState
    workflow_entry: str | None
    validation: ValidationState
    validation_detail: str | None
    missing: tuple[MissingRequirement, ...] = field(default_factory=tuple)

    @property
    def configuration(self) -> Literal["complete", "incomplete"]:
        return "complete" if not self.missing else "incomplete"

    @property
    def next_action(self) -> NextAction:
        return next_action(self)


def next_action(state: ProjectState) -> NextAction:
    """Return the first matching row of the normative Next table.

    First match wins. Keep this the only place the ordering is written down.
    """

    if not state.manifest_present:
        return NextAction(
            "zippergen project init",
            "this directory is not a ZipperGen project",
            may_agent_run=True,
        )
    if state.workflow == "absent":
        return NextAction(
            "zippergen adopt",
            "no workflow is configured for this project",
            may_agent_run=True,
        )
    if state.validation == "invalid":
        return NextAction(
            "zippergen verify",
            str(state.validation_detail or "the workflow does not validate"),
            may_agent_run=True,
        )
    if state.specification == "absent":
        return NextAction(
            "write specification.md",
            "the project has no statement of intent",
            may_agent_run=True,
        )
    if state.missing:
        first = state.missing[0]
        return NextAction(
            first.fix,
            f"{first.what} is not available on this machine",
            may_agent_run=first.may_agent_run,
        )
    return NextAction(
        "zippergen run",
        "the project is valid and configured",
        may_agent_run=True,
    )


def read_project_state(workspace: Workspace) -> ProjectState:
    """Observe the project's properties without changing anything."""

    manifest = workspace.project_manifest()
    manifest_present = bool(manifest.get("exists"))
    entry = workspace.workflow_entry

    specification: SpecificationState = "absent"
    try:
        specification = (
            "present" if workspace.specification() is not None else "absent"
        )
    except WorkspaceError:
        specification = "absent"

    workflow: WorkflowState = "absent"
    validation: ValidationState = "unknown"
    detail: str | None = None
    if entry:
        workflow = "present"
        validation, detail = _validate(workspace, entry)

    return ProjectState(
        root=str(workspace.root),
        name=str(manifest.get("name") or workspace.root.name),
        manifest_present=manifest_present,
        specification=specification,
        workflow=workflow,
        workflow_entry=entry,
        validation=validation,
        validation_detail=detail,
        missing=_missing_requirements(workspace),
    )


def _validate(
    workspace: Workspace,
    entry: str,
) -> tuple[ValidationState, str | None]:
    """Load and project the workflow, reporting the first failure."""

    from zippergen.projection import project
    from zippergen.serve import load_workflow_spec
    from zippergen.syntax import _ordered_workflow_lifelines

    # The entry is relative to the project, not to wherever the caller
    # happens to be standing.
    path, separator, attribute = entry.partition(":")
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = workspace.root / resolved
    spec_text = f"{resolved}{separator}{attribute}" if separator else str(resolved)

    try:
        wf, _module = load_workflow_spec(spec_text)
    except FileNotFoundError:
        return "invalid", f"{entry} does not exist"
    except Exception as exc:  # the workflow module is arbitrary user code
        return "invalid", f"{type(exc).__name__}: {exc}"

    try:
        for lifeline in _ordered_workflow_lifelines(wf):
            project(wf, lifeline)
    except Exception as exc:
        return "invalid", f"projection failed: {exc}"
    return "valid", None


def _missing_requirements(
    workspace: Workspace,
) -> tuple[MissingRequirement, ...]:
    """Report what this machine still needs, never a credential value."""

    missing: list[MissingRequirement] = []
    try:
        secrets = workspace.load_secrets()
    except WorkspaceError:
        secrets = {}

    from zippergen.studio_models import _PROVIDER_SECRETS

    try:
        configurations = workspace.model_configurations()
    except WorkspaceError:
        configurations = {}

    seen: set[str] = set()
    for configuration in configurations.values():
        provider = str(configuration.get("provider") or "")
        variable = _PROVIDER_SECRETS.get(provider)
        if variable is None or variable in seen:
            continue
        seen.add(variable)
        if secrets.get(variable) or os.environ.get(variable):
            continue
        missing.append(
            MissingRequirement(
                what=variable,
                fix=f"zippergen provider configure {provider}",
                may_agent_run=False,
            )
        )
    return tuple(missing)
