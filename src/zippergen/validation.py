"""Structural validation for loaded ZipperGen workflows."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import ModuleType

from zippergen.availability import workflow_availability_error
from zippergen.connectors import connector_requirements_from_module
from zippergen.deployment import deployment_spec_from_module
from zippergen.projection import project, project_templates
from zippergen.pools import pool_operations
from zippergen.syntax import (
    ActStmt,
    AssistantAction,
    EffectAction,
    IfStmt,
    MsgStmt,
    ParallelStmt,
    SpawnStmt,
    SeqStmt,
    WhileStmt,
    VarExpr,
    Workflow,
    _ordered_workflow_lifelines,
    _all_workflow_lifelines,
    participation_set,
)
from zippergen.view import ViewOptions, render_workflow


def workflow_action_sites(
    workflow: Workflow,
) -> tuple[tuple[str, object], ...]:
    """Return each action site and its owning participant."""

    found: list[tuple[str, object]] = []

    def visit(stmt) -> None:
        if isinstance(stmt, ActStmt):
            found.append((stmt.lifeline.name, stmt.action))
        elif isinstance(stmt, SeqStmt):
            visit(stmt.first)
            visit(stmt.second)
        elif isinstance(stmt, IfStmt):
            visit(stmt.branch_true)
            visit(stmt.branch_false)
        elif isinstance(stmt, WhileStmt):
            visit(stmt.body)
            visit(stmt.exit_body)
        elif isinstance(stmt, SpawnStmt):
            visit(stmt.body)
        elif isinstance(stmt, ParallelStmt):
            for branch in stmt.branches:
                visit(branch)

    visit(workflow.body)
    return tuple(found)


def workflow_actions(workflow: Workflow) -> tuple[object, ...]:
    """Return each distinct first-class action used by a global workflow."""

    found: list[object] = []
    seen: set[int] = set()
    for _participant, action in workflow_action_sites(workflow):
        if id(action) in seen:
            continue
        seen.add(id(action))
        found.append(action)
    return tuple(found)


def assistant_actions(workflow: Workflow) -> tuple[AssistantAction, ...]:
    """Return each first-class assistant action used by a global workflow."""

    return tuple(
        action
        for action in workflow_actions(workflow)
        if isinstance(action, AssistantAction)
    )


def _spawn_sharing_checks(workflow: Workflow) -> list[dict[str, object]]:
    """Warn about writes through inherited cells, without forbidding sharing.

    Nested freshening stops a write from reaching the enclosing session's
    cell. Reads and mutations hidden inside Python actions are not analysed.
    """
    checks: list[dict[str, object]] = []

    def visit(stmt, site: str) -> set[tuple[str, str]]:
        if isinstance(stmt, ActStmt):
            return {(stmt.lifeline.name, value.name) for value in stmt.outputs}
        if isinstance(stmt, MsgStmt):
            return {(stmt.receiver.name, value.var.name) for value in stmt.bindings
                    if isinstance(value, VarExpr)}
        if isinstance(stmt, SpawnStmt):
            writes = visit(stmt.body, site + ".0")
            fresh = {(field.lifeline.name, field.name) for field in stmt.fresh}
            retained = (participation_set(stmt.body) if stmt.bindings is None
                        else frozenset(stmt.retain or ()))
            inherited = {(owner, name) for owner, name in writes - fresh
                         if any(role.name == owner for role in retained)}
            if inherited:
                fields = ", ".join(f"{owner}.{name}" for owner, name in sorted(inherited))
                checks.append({
                    "status": "warn",
                    "name": f"spawn shared writes at {site}",
                    "detail": (
                        f"{fields}: assigned inside spawn but omitted from fresh. "
                        "These cells are shared with the parent and may be shared "
                        "with other sessions. Add per-request fields to fresh. "
                        "Keep them omitted only when sharing is intentional, "
                        "for example a shared accumulator."
                    ),
                })
            return inherited
        children = ()
        if isinstance(stmt, SeqStmt):
            children = (stmt.first, stmt.second)
        elif isinstance(stmt, IfStmt):
            children = (stmt.branch_true, stmt.branch_false)
        elif isinstance(stmt, WhileStmt):
            children = (stmt.body, stmt.exit_body)
        elif isinstance(stmt, ParallelStmt):
            children = stmt.branches
        return set().union(*(visit(child, f"{site}.{i}") for i, child in enumerate(children)))

    visit(workflow.body, "0")
    return checks


def _spawn_interface_checks(workflow: Workflow) -> list[dict[str, object]]:
    """Check parent availability separately from each child's interface."""

    failures: list[str] = []

    def visit(node, available: frozenset, site: str) -> None:
        if isinstance(node, SpawnStmt):
            child = participation_set(node.body)
            if node.bindings is None:
                missing = child - available
                if missing:
                    failures.append(f"inherited spawn {site} needs parent roles "
                                    + ", ".join(sorted(role.name for role in missing)))
            else:
                if node.owner not in available:
                    failures.append(f"bound spawn {site} owner {node.owner.name} is unavailable in parent")
                missing = set(node.retain or ()) - available
                if missing:
                    failures.append(f"bound spawn {site} retains unavailable parent roles "
                                    + ", ".join(sorted(role.name for role in missing)))
            visit(node.body, child, site + ".0")
        elif isinstance(node, SeqStmt):
            visit(node.first, available, site + ".0")
            visit(node.second, available, site + ".1")
        elif isinstance(node, IfStmt):
            visit(node.branch_true, available, site + ".0")
            visit(node.branch_false, available, site + ".1")
        elif isinstance(node, WhileStmt):
            visit(node.body, available, site + ".0")
            visit(node.exit_body, available, site + ".1")
        elif isinstance(node, ParallelStmt):
            for index, branch in enumerate(node.branches):
                visit(branch, available, site + f".{index}")

    visit(workflow.body, participation_set(workflow.body), "0")
    return [{"status": "fail", "name": "spawn interface", "detail": detail}
            for detail in failures]


def validate_workflow(workflow: Workflow, module: ModuleType) -> dict[str, object]:
    lifelines = _ordered_workflow_lifelines(workflow)
    checks: list[dict[str, object]] = []
    names = [lifeline.name for lifeline in lifelines]
    all_roles = _all_workflow_lifelines(workflow)
    has_recruits = len(all_roles) > len(lifelines)
    if len(names) != len(set(names)):
        checks.append({
            "status": "fail",
            "name": "lifelines",
            "detail": "lifeline names are not unique",
        })
    else:
        checks.append({
            "status": "ok",
            "name": "lifelines",
            "detail": (f"{len(lifelines)} unique root lifeline(s): {', '.join(names)}"
                       if has_recruits else
                       f"{len(lifelines)} unique lifeline(s): {', '.join(names)}"),
        })

    try:
        templates = project_templates(workflow)
        from zippergen.control import program_fingerprint
        program_fingerprint({}, templates=templates)
    except Exception as exc:
        checks.append({
            "status": "fail", "name": "spawn templates",
            "detail": f"{type(exc).__name__}: {exc}",
        })
    else:
        if templates:
            roles = ", ".join(role.name for role in all_roles)
            checks.append({
                "status": "ok", "name": "spawn templates",
                "detail": f"{len(templates)} compiled role template(s); all static roles: {roles}",
            })

    availability_error = workflow_availability_error(workflow)
    checks.append({
        "status": "fail" if availability_error is not None else "ok",
        "name": "variable availability",
        "detail": (
            availability_error
            if availability_error is not None
            else "every participant reads only values definitely available locally"
        ),
    })
    checks.extend(_spawn_sharing_checks(workflow))
    checks.extend(_spawn_interface_checks(workflow))

    projections: dict[str, str] = {}
    for lifeline in lifelines:
        try:
            local = project(workflow, lifeline)
            code = render_workflow(
                workflow,
                module,
                options=ViewOptions(agent=lifeline.name),
            )
        except Exception as exc:
            checks.append({
                "status": "fail",
                "name": f"projection {lifeline.name}",
                "detail": f"{type(exc).__name__}: {exc}",
            })
        else:
            projections[lifeline.name] = code
            checks.append({
                "status": "ok",
                "name": f"projection {lifeline.name}",
                "detail": type(local).__name__,
            })

    declaration = None
    try:
        declaration = deployment_spec_from_module(module)
    except Exception as exc:
        checks.append({
            "status": "fail",
            "name": "deployment declaration",
            "detail": f"{type(exc).__name__}: {exc}",
        })
        deployment: dict[str, object] | None = None
    else:
        deployment = declaration.as_dict()
        checks.append({
            "status": "ok",
            "name": "deployment declaration",
            "detail": (
                f"{len(declaration.fields)} field(s), "
                f"{len(declaration.packages)} package(s), "
                f"{len(declaration.setup)} setup step(s)"
            ),
        })

    input_fields = {
        field.target_name: field
        for field in (declaration.fields if declaration is not None else ())
        if field.target == "input"
    }
    if workflow.inputs:
        rendered_inputs: list[str] = []
        for name, value_type, lifeline in workflow.inputs:
            type_name = getattr(value_type, "__name__", str(value_type))
            owner = f" @ {lifeline.name}" if lifeline is not None else ""
            field = input_fields.get(name)
            if field is not None and field.default is not None:
                default = f", default {field.default!r}"
            else:
                default = ", required"
            rendered_inputs.append(
                f"{name} ({type_name}){owner}{default}"
            )
        input_detail = ", ".join(rendered_inputs)
    else:
        input_detail = "none, the run starts without setup questions"
    checks.append({
        "status": "ok",
        "name": "workflow inputs",
        "detail": input_detail,
    })

    try:
        connector_requirements = connector_requirements_from_module(module)
    except (TypeError, ValueError) as exc:
        checks.append({
            "status": "fail",
            "name": "connector requirements",
            "detail": str(exc),
        })
    else:
        participants = {item.name for item in lifelines}
        requirement_names = {
            requirement.name for requirement in connector_requirements
        }
        requirements_by_name = {
            requirement.name: requirement
            for requirement in connector_requirements
        }
        for requirement in connector_requirements:
            if requirement.participant not in participants:
                checks.append({
                    "status": "fail",
                    "name": f"connector {requirement.name}",
                    "detail": (
                        f"participant {requirement.participant!r} does not "
                        "exist in the workflow"
                    ),
                })
            else:
                checks.append({
                    "status": "ok",
                    "name": f"connector {requirement.name}",
                    "detail": (
                        f"{requirement.kind}; {requirement.access}; participant "
                        f"{requirement.participant}"
                    ),
                })
        referenced_connectors = {
            action.connector
            for action in workflow_actions(workflow)
            if isinstance(action, EffectAction) and action.connector is not None
        }
        for connector_name in sorted(referenced_connectors):
            if connector_name not in requirement_names:
                checks.append({
                    "status": "fail",
                    "name": f"effect connector {connector_name}",
                    "detail": (
                        "the effect references an undeclared connector; add a "
                        "matching ConnectorRequirement"
                    ),
                })
            else:
                checks.append({
                    "status": "ok",
                    "name": f"effect connector {connector_name}",
                    "detail": "declared logical connector requirement",
                })
        for participant, action in workflow_action_sites(workflow):
            if not isinstance(action, EffectAction) or action.connector is None:
                continue
            requirement = requirements_by_name.get(action.connector)
            if (
                requirement is not None
                and requirement.participant != participant
            ):
                checks.append({
                    "status": "fail",
                    "name": f"effect connector owner {action.name}",
                    "detail": (
                        f"action runs on {participant}, but connector "
                        f"{requirement.name} belongs to "
                        f"{requirement.participant}"
                    ),
                })

    pools: dict[str, set[float]] = {}
    for operation in pool_operations(workflow):
        pools.setdefault(operation.pool, set()).add(operation.lease_seconds)
    for name, leases in sorted(pools.items()):
        consistent = len(leases) == 1
        checks.append({
            "status": "ok" if consistent else "fail",
            "name": f"work pool {name}",
            "detail": (
                f"execution-local FIFO pool; lease {next(iter(leases)):g}s; "
                "CPL context follows claimed jobs; no worker-fairness guarantee"
                if consistent else "conflicting lease declarations for the same pool"
            ),
        })

    workflow_assistant_actions = assistant_actions(workflow)
    for action in workflow_assistant_actions:
        checks.append({
            "status": "ok",
            "name": f"assistant access {action.name}",
            "detail": (
                f"{action.access}; enforced by the selected CLI permission mode"
            ),
        })
        checks.append({
            "status": (
                "warn" if action.external_tools == "configured" else "ok"
            ),
            "name": f"assistant external tools {action.name}",
            "detail": (
                "configured assistant MCP/tool integrations are permitted"
                if action.external_tools == "configured"
                else "configured MCP/tool integrations and web access are disabled"
            ),
        })
        # Workflow validation does not know the project's assistant routing.
        # Any shell-enabled action may therefore be assigned to Claude.
        shell_warning = action.shell == "enabled"
        checks.append({
            "status": "warn" if shell_warning else "ok",
            "name": f"assistant shell {action.name}",
            "detail": (
                "enabled; a Claude selection receives Bash without structural "
                "network isolation; use a separate fixed verification action "
                "when possible"
                if shell_warning
                else (
                    "enabled inside the Codex structural sandbox"
                    if action.shell == "enabled"
                    else (
                        "restricted; Claude receives no Bash and Codex commands "
                        "remain inside its structural sandbox"
                    )
                )
            ),
        })
        module_file = getattr(module, "__file__", None)
        if action.access == "write" and module_file:
            requested_workspace = Path(action.workspace or ".").expanduser()
            workspace_path = (
                requested_workspace.resolve()
                if requested_workspace.is_absolute()
                else (Path.cwd() / requested_workspace).resolve()
            )
            workflow_source = Path(module_file).resolve()
            if workflow_source.is_relative_to(workspace_path):
                checks.append({
                    "status": "warn",
                    "name": f"assistant self-modification {action.name}",
                    "detail": (
                        f"write workspace {workspace_path} contains the "
                        f"executing workflow source {workflow_source}; keep "
                        "self-modification, deployment, and Git boundaries "
                        "explicit in the reviewed instructions"
                    ),
                })
        if action.instructions_path is None:
            checks.append({
                "status": "ok",
                "name": f"assistant instructions {action.name}",
                "detail": (
                    f"inline Markdown; sha256 "
                    f"{action.instructions_sha256[:12]}"
                ),
            })
            continue
        path = Path(action.instructions_path)
        if not path.is_file():
            checks.append({
                "status": "fail",
                "name": f"assistant instructions {action.name}",
                "detail": f"file no longer exists: {path}",
            })
            continue
        current_text = path.read_text(encoding="utf-8")
        current_hash = hashlib.sha256(current_text.encode("utf-8")).hexdigest()
        if current_hash != action.instructions_sha256:
            checks.append({
                "status": "fail",
                "name": f"assistant instructions {action.name}",
                "detail": (
                    "file changed after the workflow was imported; reload the "
                    f"workflow: {path}"
                ),
            })
        else:
            checks.append({
                "status": "ok",
                "name": f"assistant instructions {action.name}",
                "detail": (
                    f"{action.instructions_file}; sha256 "
                    f"{action.instructions_sha256[:12]}; automatically bundled"
                ),
            })

    try:
        render_workflow(workflow, module, options=ViewOptions(detail="full"))
    except Exception as exc:
        checks.append({
            "status": "fail",
            "name": "canonical rendering",
            "detail": f"{type(exc).__name__}: {exc}",
        })
    else:
        checks.append({
            "status": "ok",
            "name": "canonical rendering",
            "detail": "global and local code views rendered successfully",
        })

    return {
        "workflow": workflow.name,
        "valid": not any(check["status"] == "fail" for check in checks),
        "lifelines": names,
        "inputs": [
            {
                "name": name,
                "type": getattr(value_type, "__name__", str(value_type)),
                "lifeline": lifeline.name if lifeline else None,
            }
            for name, value_type, lifeline in workflow.inputs
        ],
        "outputs": [
            {
                "name": value.name,
                "type": getattr(value.type, "__name__", str(value.type)),
                "lifeline": lifeline.name,
            }
            for value, lifeline in workflow.outputs
        ],
        "deployment": deployment,
        "checks": checks,
        "projections": projections,
    }
