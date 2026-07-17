"""Stable semantic summaries and diffs for ZipperGen workflows."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import textwrap
from collections.abc import Iterable, Mapping
from collections import Counter
from types import ModuleType

from zippergen.deployment import deployment_spec_from_module
from zippergen.syntax import (
    ActStmt,
    AnyStmt,
    CoregionStmt,
    EffectAction,
    EmptyStmt,
    HumanAction,
    IfStmt,
    LLMAction,
    LitExpr,
    MsgStmt,
    ParallelStmt,
    PlannerAction,
    PureAction,
    SeqStmt,
    SkipStmt,
    VarExpr,
    WhileStmt,
    Workflow,
    _ordered_workflow_lifelines,
)


SEMANTIC_SNAPSHOT_SCHEMA = "zippergen.workflow-semantics.v1"


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_value(item) for item in value), key=str)
    return str(value)


def _expr(value: object) -> str:
    if isinstance(value, VarExpr):
        return value.var.name
    if isinstance(value, LitExpr):
        return repr(value.value)
    return repr(value)


def _condition(value: object) -> str:
    return str(getattr(value, "_src", "<condition>"))


def _type_name(value: object) -> str:
    return str(getattr(value, "__name__", value))


def _pairs(values: object) -> list[dict[str, str]]:
    return [
        {"name": str(name), "type": _type_name(value_type)}
        for name, value_type in values  # type: ignore[union-attr]
    ]


def _implementation_hash(action: PureAction | EffectAction) -> str:
    try:
        source = textwrap.dedent(inspect.getsource(action.fn)).strip()
        fingerprint = ast.dump(ast.parse(source), include_attributes=False)
    except (OSError, SyntaxError, TypeError):
        code = getattr(action.fn, "__code__", None)
        fingerprint = repr((
            getattr(code, "co_code", action.fn),
            getattr(code, "co_consts", ()),
            getattr(code, "co_names", ()),
        ))
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]


def _action_definition(action: object) -> dict[str, object]:
    base: dict[str, object] = {
        "name": str(action.name),  # type: ignore[attr-defined]
        "inputs": _pairs(getattr(action, "inputs", ())),
    }
    if isinstance(action, HumanAction):
        base.update({
            "kind": "human",
            "outputs": [{"name": action.output, "type": _type_name(action.output_type)}],
            "human_kind": action.kind,
            "context": action.context,
            "instruction": action.instruction,
            "prefill": action.prefill,
            "submit_label": action.submit_label,
            "cancel_label": action.cancel_label,
        })
    else:
        base["outputs"] = _pairs(getattr(action, "outputs", ()))
    if isinstance(action, LLMAction):
        base.update({
            "kind": "llm",
            "system_prompt": action.system_prompt,
            "user_prompt": action.user_prompt,
            "parse_format": action.parse_format,
        })
    elif isinstance(action, PureAction):
        base.update({"kind": "pure", "implementation_hash": _implementation_hash(action)})
    elif isinstance(action, EffectAction):
        base.update({"kind": "effect", "implementation_hash": _implementation_hash(action)})
    elif isinstance(action, PlannerAction):
        base.update({
            "kind": "planner",
            "allow": list(action.allow),
            "instructions": action.instructions,
            "lifelines": [item.name for item in action.lifelines],
            "actions": [item.name for item in action.actions],
            "max_retries": action.max_retries,
        })
    return base


def workflow_semantics(
    workflow: Workflow,
    module: ModuleType | None = None,
) -> dict[str, object]:
    """Return a JSON-compatible semantic model independent of source layout."""

    messages: list[dict[str, object]] = []
    action_sites: list[dict[str, object]] = []
    controls: list[dict[str, object]] = []
    regions: list[dict[str, object]] = []
    skips: list[dict[str, object]] = []
    definitions: dict[str, dict[str, object]] = {}

    def sequence(*nodes: object) -> dict[str, object]:
        steps: list[object] = []
        for node in nodes:
            if isinstance(node, dict) and node.get("node") == "empty":
                continue
            if isinstance(node, dict) and node.get("node") == "sequence":
                nested = node.get("steps")
                if isinstance(nested, list):
                    steps.extend(nested)
                    continue
            steps.append(node)
        if not steps:
            return {"node": "empty"}
        if len(steps) == 1 and isinstance(steps[0], dict):
            return steps[0]
        return {"node": "sequence", "steps": steps}

    def canonical_unordered(nodes: Iterable[object]) -> list[object]:
        return sorted(nodes, key=lambda item: json.dumps(item, sort_keys=True, default=str))

    def walk(stmt: AnyStmt, context: tuple[str, ...] = ()) -> dict[str, object]:
        if isinstance(stmt, EmptyStmt):
            return {"node": "empty"}
        if isinstance(stmt, SeqStmt):
            return sequence(walk(stmt.first, context), walk(stmt.second, context))
        if isinstance(stmt, MsgStmt):
            payload = [_expr(value) for value in stmt.payload]
            bindings = [_expr(value) for value in stmt.bindings]
            record = {
                "sender": stmt.sender.name,
                "receiver": stmt.receiver.name,
                "payload": payload,
                "bindings": bindings,
                "self_message": stmt.sender == stmt.receiver,
                "context": list(context),
                "code": (
                    f"{stmt.sender.name}({', '.join(payload)}) >> "
                    f"{stmt.receiver.name}({', '.join(bindings)})"
                ),
            }
            messages.append(record)
            return {
                "node": "message",
                **{key: value for key, value in record.items() if key != "code"},
            }
        if isinstance(stmt, CoregionStmt):
            marker = f"coregion[{len(stmt.messages)}]"
            regions.append({"kind": "coregion", "size": len(stmt.messages), "context": list(context)})
            nodes = [walk(message, (*context, marker)) for message in stmt.messages]
            return {"node": "coregion", "messages": canonical_unordered(nodes)}
        if isinstance(stmt, ActStmt):
            definition = _action_definition(stmt.action)
            key = str(definition["name"])
            definitions[key] = definition
            inputs = [_expr(value) for value in stmt.inputs]
            outputs = [value.name for value in stmt.outputs]
            record = {
                "lifeline": stmt.lifeline.name,
                "action": stmt.action.name,
                "kind": definition["kind"],
                "inputs": inputs,
                "outputs": outputs,
                "context": list(context),
                "code": (
                    f"{stmt.lifeline.name}: {', '.join(outputs)} = "
                    f"{stmt.action.name}({', '.join(inputs)})"
                ),
            }
            action_sites.append(record)
            return {
                "node": "action",
                **{key: value for key, value in record.items() if key != "code"},
            }
        if isinstance(stmt, SkipStmt):
            record = {"lifeline": stmt.lifeline.name, "context": list(context)}
            skips.append(record)
            return {"node": "skip", **record}
        if isinstance(stmt, IfStmt):
            condition = _condition(stmt.condition)
            record = {
                "kind": "if",
                "owner": stmt.owner.name,
                "condition": condition,
                "context": list(context),
                "code": f"if ({condition}) @ {stmt.owner.name}",
            }
            controls.append(record)
            marker = f"if:{stmt.owner.name}:{condition}"
            return {
                "node": "if",
                "owner": stmt.owner.name,
                "condition": condition,
                "true": walk(stmt.branch_true, (*context, marker + "=true")),
                "false": walk(stmt.branch_false, (*context, marker + "=false")),
            }
        if isinstance(stmt, WhileStmt):
            condition = _condition(stmt.condition)
            record = {
                "kind": "while",
                "owner": stmt.owner.name,
                "condition": condition,
                "context": list(context),
                "code": f"while ({condition}) @ {stmt.owner.name}",
            }
            controls.append(record)
            marker = f"while:{stmt.owner.name}:{condition}"
            return {
                "node": "while",
                "owner": stmt.owner.name,
                "condition": condition,
                "body": walk(stmt.body, (*context, marker + "=body")),
                "exit": walk(stmt.exit_body, (*context, marker + "=exit")),
            }
        if isinstance(stmt, ParallelStmt):
            regions.append({
                "kind": "parallel",
                "branches": len(stmt.branches),
                "context": list(context),
            })
            branches = [
                walk(branch_stmt, (*context, f"parallel[{len(stmt.branches)}]"))
                for branch_stmt in stmt.branches
            ]
            return {"node": "parallel", "branches": canonical_unordered(branches)}
        raise TypeError(f"unsupported global statement: {type(stmt).__name__}")

    protocol = walk(workflow.body)
    result: dict[str, object] = {
        "name": workflow.name,
        "lifelines": [item.name for item in _ordered_workflow_lifelines(workflow)],
        "inputs": {
            name: {
                "type": _type_name(value_type),
                "lifeline": lifeline.name if lifeline else None,
            }
            for name, value_type, lifeline in workflow.inputs
        },
        "outputs": {
            f"{value.name}@{lifeline.name}": {
                "name": value.name,
                "type": _type_name(value.type),
                "lifeline": lifeline.name,
            }
            for value, lifeline in workflow.outputs
        },
        "messages": messages,
        "action_definitions": definitions,
        "action_sites": action_sites,
        "controls": controls,
        "regions": regions,
        "skips": skips,
        "protocol": protocol,
    }
    if module is not None:
        declaration = deployment_spec_from_module(module)
        result["deployment"] = {
            "name": declaration.name,
            "description": declaration.description,
            "fields": {
                field.name: _json_value(field.__dict__) for field in declaration.fields
            },
            "packages": {
                package.requirement: _json_value(package.__dict__)
                for package in declaration.packages
            },
            "setup": {
                step.name: _json_value(step.__dict__) for step in declaration.setup
            },
            "files": list(declaration.files),
        }
    normalized = _json_value(result)
    assert isinstance(normalized, dict)
    return normalized


def _map_changes(before: dict[str, object], after: dict[str, object]) -> dict[str, object]:
    before_names = set(before)
    after_names = set(after)
    changed = []
    for name in sorted(before_names & after_names):
        if before[name] == after[name]:
            continue
        old = before[name]
        new = after[name]
        fields: dict[str, object] = {}
        if isinstance(old, dict) and isinstance(new, dict):
            for field in sorted(set(old) | set(new)):
                if old.get(field) != new.get(field):
                    fields[field] = {"before": old.get(field), "after": new.get(field)}
        changed.append({"name": name, "before": old, "after": new, "fields": fields})
    return {
        "added": [{"name": name, "value": after[name]} for name in sorted(after_names - before_names)],
        "removed": [{"name": name, "value": before[name]} for name in sorted(before_names - after_names)],
        "changed": changed,
    }


def _list_changes(before: list[object], after: list[object]) -> dict[str, object]:
    def encoded(values: list[object]) -> tuple[Counter[str], dict[str, object]]:
        records: dict[str, object] = {}
        counter: Counter[str] = Counter()
        for value in values:
            key = json.dumps(value, sort_keys=True, default=str)
            records[key] = value
            counter[key] += 1
        return counter, records

    old_counter, old_records = encoded(before)
    new_counter, new_records = encoded(after)
    added: list[object] = []
    removed: list[object] = []
    for key, count in (new_counter - old_counter).items():
        added.extend([new_records[key]] * count)
    for key, count in (old_counter - new_counter).items():
        removed.extend([old_records[key]] * count)
    return {"added": added, "removed": removed}


def _deployment_changes(before: object, after: object) -> dict[str, object]:
    old = before if isinstance(before, dict) else {}
    new = after if isinstance(after, dict) else {}
    return {
        "metadata": _map_changes(
            {key: old.get(key) for key in ("name", "description")},
            {key: new.get(key) for key in ("name", "description")},
        ),
        "fields": _map_changes(old.get("fields", {}), new.get("fields", {})),
        "packages": _map_changes(old.get("packages", {}), new.get("packages", {})),
        "setup": _map_changes(old.get("setup", {}), new.get("setup", {})),
        "files": _list_changes(old.get("files", []), new.get("files", [])),
    }


def _has_changes(value: object) -> bool:
    if isinstance(value, list):
        return bool(value)
    if isinstance(value, dict):
        if value and set(value).issubset({"before", "after"}):
            return True
        return any(_has_changes(item) for item in value.values())
    return False


def semantic_snapshot(
    workflow: Workflow,
    module: ModuleType | None = None,
) -> dict[str, object]:
    """Capture a versioned, JSON-compatible semantic baseline."""

    return {
        "schema": SEMANTIC_SNAPSHOT_SCHEMA,
        "workflow": workflow_semantics(workflow, module),
    }


def read_semantic_snapshot(value: object) -> dict[str, object]:
    """Validate and return the workflow model from a semantic snapshot."""

    if not isinstance(value, dict):
        raise ValueError("semantic snapshot must be a JSON object")
    if value.get("schema") != SEMANTIC_SNAPSHOT_SCHEMA:
        raise ValueError(
            "unsupported semantic snapshot schema; expected "
            f"{SEMANTIC_SNAPSHOT_SCHEMA!r}"
        )
    workflow = value.get("workflow")
    if not isinstance(workflow, dict) or not isinstance(workflow.get("name"), str):
        raise ValueError("semantic snapshot is missing its workflow model")
    required = {
        "lifelines", "inputs", "outputs", "messages", "action_definitions",
        "action_sites", "controls", "regions", "skips", "protocol",
    }
    missing = sorted(required - set(workflow))
    if missing:
        raise ValueError(
            "semantic snapshot workflow model is incomplete; missing "
            + ", ".join(missing)
        )
    return workflow


def semantic_diff_models(
    before: dict[str, object],
    after: dict[str, object],
) -> dict[str, object]:
    """Compare two previously extracted semantic workflow models."""

    changes = {
        "name": (
            {"before": before["name"], "after": after["name"]}
            if before["name"] != after["name"]
            else {}
        ),
        "lifelines": _list_changes(before["lifelines"], after["lifelines"]),  # type: ignore[arg-type]
        "inputs": _map_changes(before["inputs"], after["inputs"]),  # type: ignore[arg-type]
        "outputs": _map_changes(before["outputs"], after["outputs"]),  # type: ignore[arg-type]
        "messages": _list_changes(before["messages"], after["messages"]),  # type: ignore[arg-type]
        "action_definitions": _map_changes(
            before["action_definitions"], after["action_definitions"]  # type: ignore[arg-type]
        ),
        "action_sites": _list_changes(before["action_sites"], after["action_sites"]),  # type: ignore[arg-type]
        "controls": _list_changes(before["controls"], after["controls"]),  # type: ignore[arg-type]
        "regions": _list_changes(before["regions"], after["regions"]),  # type: ignore[arg-type]
        "skips": _list_changes(before["skips"], after["skips"]),  # type: ignore[arg-type]
        "protocol": (
            {"before": before["protocol"], "after": after["protocol"]}
            if before["protocol"] != after["protocol"]
            else {}
        ),
        "deployment": _deployment_changes(before.get("deployment"), after.get("deployment")),
    }
    return {
        "before": str(before["name"]),
        "after": str(after["name"]),
        "changed": _has_changes(changes),
        "changes": changes,
    }


def semantic_diff(
    before_workflow: Workflow,
    after_workflow: Workflow,
    before_module: ModuleType | None = None,
    after_module: ModuleType | None = None,
) -> dict[str, object]:
    """Compare two workflows by meaning-bearing IR facts, not source lines."""

    return semantic_diff_models(
        workflow_semantics(before_workflow, before_module),
        workflow_semantics(after_workflow, after_module),
    )


def _fact_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if value.get("code"):
            text = str(value["code"])
            context = value.get("context") or []
            if context:
                text += f"  # {' / '.join(str(item) for item in context)}"
            return text
        if value.get("name") and len(value) <= 3:
            return str(value["name"])
    return json.dumps(value, sort_keys=True, default=str)


def _change_item_text(item: object) -> str:
    if isinstance(item, dict) and "name" in item and "value" in item:
        return f"{item['name']}: {_fact_text(item['value'])}"
    return _fact_text(item)


def render_semantic_diff(result: dict[str, object]) -> str:
    """Render a semantic diff as stable, source-like terminal text."""

    lines = [
        f"# Semantic workflow diff: {result['before']} -> {result['after']}",
    ]
    if not result["changed"]:
        return "\n".join([*lines, "# No semantic changes."])

    changes = result["changes"]
    assert isinstance(changes, dict)
    order = (
        "name", "lifelines", "inputs", "outputs", "messages",
        "action_definitions", "action_sites", "controls", "regions",
        "skips", "protocol",
    )
    for section in order:
        value = changes.get(section)
        if not _has_changes(value):
            continue
        lines.extend(["", f"# {section.replace('_', ' ').title()}"])
        if section == "name" and isinstance(value, dict):
            lines.append(f"~ {value.get('before')} -> {value.get('after')}")
            continue
        if section == "protocol":
            lines.append("~ execution order or control structure changed")
            continue
        if not isinstance(value, dict):
            continue
        for item in value.get("removed", []):
            lines.append(f"- {_change_item_text(item)}")
        for item in value.get("added", []):
            lines.append(f"+ {_change_item_text(item)}")
        for item in value.get("changed", []):
            if not isinstance(item, dict):
                continue
            lines.append(f"~ {item.get('name')}")
            for field, field_change in item.get("fields", {}).items():
                lines.append(
                    f"    {field}: {_fact_text(field_change.get('before'))} -> "
                    f"{_fact_text(field_change.get('after'))}"
                )

    deployment = changes.get("deployment")
    if _has_changes(deployment) and isinstance(deployment, dict):
        lines.extend(["", "# Deployment"])
        for subsection in ("metadata", "fields", "packages", "setup", "files"):
            value = deployment.get(subsection)
            if not _has_changes(value) or not isinstance(value, dict):
                continue
            for item in value.get("removed", []):
                lines.append(f"- {subsection}: {_change_item_text(item)}")
            for item in value.get("added", []):
                lines.append(f"+ {subsection}: {_change_item_text(item)}")
            for item in value.get("changed", []):
                if isinstance(item, dict):
                    lines.append(f"~ {subsection}: {item.get('name')}")
    return "\n".join(lines)


__all__ = [
    "SEMANTIC_SNAPSHOT_SCHEMA",
    "read_semantic_snapshot",
    "render_semantic_diff",
    "semantic_diff",
    "semantic_diff_models",
    "semantic_snapshot",
    "workflow_semantics",
]
