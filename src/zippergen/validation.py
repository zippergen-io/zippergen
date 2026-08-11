"""Structural validation for loaded ZipperGen workflows."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import ModuleType

from zippergen.connectors import connector_requirements_from_module
from zippergen.deployment import deployment_spec_from_module
from zippergen.projection import project
from zippergen.syntax import (
    ActStmt,
    AssistantAction,
    CoregionStmt,
    EffectAction,
    IfStmt,
    ParallelStmt,
    SeqStmt,
    WhileStmt,
    Workflow,
    _ordered_workflow_lifelines,
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
        elif isinstance(stmt, ParallelStmt):
            for branch in stmt.branches:
                visit(branch)
        elif isinstance(stmt, CoregionStmt):
            return

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


def validate_workflow(workflow: Workflow, module: ModuleType) -> dict[str, object]:
    lifelines = _ordered_workflow_lifelines(workflow)
    checks: list[dict[str, object]] = []
    names = [lifeline.name for lifeline in lifelines]
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
            "detail": f"{len(lifelines)} unique lifeline(s): {', '.join(names)}",
        })

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
