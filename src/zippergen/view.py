"""Code-first inspection views for ZipperGen workflows.

The renderer works from the workflow IR rather than the original source.  This
makes every view deterministic and lets the same machinery display a complete
global choreography, a communication-only lens, a selected group with explicit
external boundaries, or the exact local program produced by projection.

Rendered text is intentionally source-like and editor-friendly.  Global views
closely follow the public ZipperGen DSL.  Filtered and local views are marked as
generated read-only views because boundary/local operations are explanatory
syntax rather than authoring APIs.
"""

from __future__ import annotations

import inspect
import json
import pprint
import textwrap
from dataclasses import dataclass
from types import ModuleType

from zippergen.deployment import deployment_spec_from_module
from zippergen.projection import project
from zippergen.syntax import (
    ActStmt,
    AnyStmt,
    CoregionStmt,
    EffectAction,
    EmptyStmt,
    Expr,
    HumanAction,
    IfRecvStmt,
    IfStmt,
    LLMAction,
    Lifeline,
    LitExpr,
    MsgStmt,
    ParallelLocalStmt,
    ParallelStmt,
    PlannerAction,
    PureAction,
    ReceiveAnyStmt,
    RecvStmt,
    SelfAssignStmt,
    SendStmt,
    SeqStmt,
    SkipStmt,
    VarExpr,
    WhileRecvStmt,
    WhileStmt,
    Workflow,
    _ordered_workflow_lifelines,
    is_kappa_ctrl,
)


DETAILS = ("overview", "protocol", "actions", "full")


@dataclass(frozen=True)
class ViewOptions:
    detail: str = "protocol"
    communications_only: bool = False
    agent: str | None = None
    agents: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.detail not in DETAILS:
            raise ValueError(f"detail must be one of {', '.join(DETAILS)}")
        if self.agent and self.agents:
            raise ValueError("use either agent or agents, not both")


def _expr(expr: Expr) -> str:
    if isinstance(expr, VarExpr):
        return expr.var.name
    if isinstance(expr, LitExpr):
        return repr(expr.value)
    return repr(expr)


def _exprs(values: tuple[Expr, ...]) -> str:
    return ", ".join(_expr(value) for value in values)


def _type_name(value: type) -> str:
    return getattr(value, "__name__", repr(value))


def _return_type(outputs: tuple[tuple[str, type], ...]) -> str:
    if not outputs:
        return "None"
    if len(outputs) == 1:
        return _type_name(outputs[0][1])
    return "tuple[" + ", ".join(_type_name(value) for _name, value in outputs) + "]"


def _condition(condition: object) -> str:
    return str(getattr(condition, "_src", "<condition>"))


def _channel_suffix(channel: str) -> str:
    return "" if channel == "main" else f", channel={channel!r}"


def _indent(lines: list[str], level: int = 1) -> list[str]:
    prefix = "    " * level
    return [prefix + line if line else "" for line in lines]


def _flatten_seq(stmt: AnyStmt) -> list[AnyStmt]:
    if isinstance(stmt, SeqStmt):
        return [*_flatten_seq(stmt.first), *_flatten_seq(stmt.second)]
    return [stmt]


def _actions(stmt: AnyStmt) -> list[object]:
    found: list[object] = []
    seen: set[int] = set()

    def visit(node: AnyStmt) -> None:
        if isinstance(node, ActStmt):
            if id(node.action) not in seen:
                seen.add(id(node.action))
                found.append(node.action)
            return
        if isinstance(node, SeqStmt):
            visit(node.first)
            visit(node.second)
        elif isinstance(node, CoregionStmt):
            return
        elif isinstance(node, IfStmt):
            visit(node.branch_true)
            visit(node.branch_false)
        elif isinstance(node, WhileStmt):
            visit(node.body)
            visit(node.exit_body)
        elif isinstance(node, IfRecvStmt):
            visit(node.branch_true)
            visit(node.branch_false)
        elif isinstance(node, WhileRecvStmt):
            visit(node.body)
            visit(node.exit_body)
        elif isinstance(node, (ParallelStmt, ParallelLocalStmt)):
            for branch in node.branches:
                visit(branch)

    visit(stmt)
    return found


def _action_owners(stmt: AnyStmt) -> dict[int, set[str]]:
    owners: dict[int, set[str]] = {}

    def visit(node: AnyStmt) -> None:
        if isinstance(node, ActStmt):
            owners.setdefault(id(node.action), set()).add(node.lifeline.name)
        elif isinstance(node, SeqStmt):
            visit(node.first)
            visit(node.second)
        elif isinstance(node, (IfStmt, IfRecvStmt)):
            visit(node.branch_true)
            visit(node.branch_false)
        elif isinstance(node, (WhileStmt, WhileRecvStmt)):
            visit(node.body)
            visit(node.exit_body)
        elif isinstance(node, (ParallelStmt, ParallelLocalStmt)):
            for branch_stmt in node.branches:
                visit(branch_stmt)

    visit(stmt)
    return owners


def _action_kind(action: object) -> str:
    if isinstance(action, LLMAction):
        return "llm"
    if isinstance(action, PureAction):
        return "pure"
    if isinstance(action, EffectAction):
        return "effect"
    if isinstance(action, HumanAction):
        return "human"
    if isinstance(action, PlannerAction):
        return "planner"
    return type(action).__name__


def _action_signature(action: object) -> str:
    inputs = getattr(action, "inputs", ())
    outputs = getattr(action, "outputs", ())
    if isinstance(action, HumanAction):
        outputs = ((action.output, action.output_type),)
    params = ", ".join(f"{name}: {_type_name(value)}" for name, value in inputs)
    return f"{action.name}({params}) -> {_return_type(outputs)}"  # type: ignore[attr-defined]


def _render_action(action: object, *, full: bool) -> list[str]:
    signature = _action_signature(action)
    if isinstance(action, (PureAction, EffectAction)) and full:
        try:
            source = textwrap.dedent(inspect.getsource(action.fn)).strip()
        except (OSError, TypeError):
            source = ""
        if source:
            return source.splitlines()

    params = ", ".join(
        f"{name}: {_type_name(value)}" for name, value in getattr(action, "inputs", ())
    )
    outputs = getattr(action, "outputs", ())
    if isinstance(action, HumanAction):
        outputs = ((action.output, action.output_type),)
    result_type = _return_type(outputs)

    if isinstance(action, LLMAction):
        system = action.system_prompt if full else "<hidden at this detail level>"
        user = action.user_prompt if full else "<hidden at this detail level>"
        return [
            "@llm(",
            f"    system={system!r},",
            f"    user={user!r},",
            f"    parse={action.parse_format!r},",
            ")",
            f"def {action.name}({params}) -> {result_type}: ...",
        ]
    if isinstance(action, HumanAction):
        return [
            f"@human(kind={action.kind!r}, instruction={action.instruction!r})",
            f"def {action.name}({params}) -> {result_type}: ...",
        ]
    if isinstance(action, PlannerAction):
        instructions = action.instructions if full else "<hidden at this detail level>"
        return [
            f"@planner(allow={action.allow!r}, instructions={instructions!r})",
            f"def {action.name}({params}) -> {result_type}: ...",
        ]
    decorator = "effect" if isinstance(action, EffectAction) else "pure"
    return [f"@{decorator}", f"def {action.name}({params}) -> {result_type}: ..."]  # type: ignore[attr-defined]


class _GlobalRenderer:
    def __init__(
        self,
        *,
        selected: frozenset[str] | None,
        communications_only: bool,
        detail: str,
    ) -> None:
        self.selected = selected
        self.communications_only = communications_only
        self.detail = detail

    def _selected(self, lifeline: Lifeline) -> bool:
        return self.selected is None or lifeline.name in self.selected

    def _endpoint(self, lifeline: Lifeline, args: str) -> str:
        if self._selected(lifeline):
            return f"{lifeline.name}({args})"
        suffix = f", {args}" if args else ""
        return f"external({lifeline.name!r}{suffix})"

    def _relevant(self, stmt: AnyStmt) -> bool:
        if isinstance(stmt, EmptyStmt):
            return False
        if isinstance(stmt, MsgStmt):
            if self.communications_only and stmt.sender == stmt.receiver:
                return False
            return self.selected is None or self._selected(stmt.sender) or self._selected(stmt.receiver)
        if isinstance(stmt, CoregionStmt):
            return any(self._relevant(message) for message in stmt.messages)
        if isinstance(stmt, ActStmt):
            return not self.communications_only and self._selected(stmt.lifeline)
        if isinstance(stmt, SkipStmt):
            return self.detail == "full" and not self.communications_only and self._selected(stmt.lifeline)
        if isinstance(stmt, SeqStmt):
            return self._relevant(stmt.first) or self._relevant(stmt.second)
        if isinstance(stmt, IfStmt):
            return self._relevant(stmt.branch_true) or self._relevant(stmt.branch_false)
        if isinstance(stmt, WhileStmt):
            return self._relevant(stmt.body) or self._relevant(stmt.exit_body)
        if isinstance(stmt, ParallelStmt):
            return any(self._relevant(branch) for branch in stmt.branches)
        return False

    def render(self, stmt: AnyStmt, indent: int = 0) -> list[str]:
        pad = "    " * indent
        if not self._relevant(stmt):
            return []
        if isinstance(stmt, SeqStmt):
            lines: list[str] = []
            for part in _flatten_seq(stmt):
                lines.extend(self.render(part, indent))
            return lines
        if isinstance(stmt, MsgStmt):
            left = self._endpoint(stmt.sender, _exprs(stmt.payload))
            right = self._endpoint(stmt.receiver, _exprs(stmt.bindings))
            return [f"{pad}{left} >> {right}"]
        if isinstance(stmt, CoregionStmt):
            messages = [message for message in stmt.messages if self._relevant(message)]
            lines = [f"{pad}with coregion:"]
            for message in messages:
                lines.extend(self.render(message, indent + 1))
            return lines
        if isinstance(stmt, ActStmt):
            outputs = ", ".join(value.name for value in stmt.outputs)
            if len(stmt.outputs) != 1:
                outputs = f"({outputs})"
            return [
                f"{pad}{stmt.lifeline.name}: {outputs} = "
                f"{stmt.action.name}({_exprs(stmt.inputs)})"
            ]
        if isinstance(stmt, SkipStmt):
            return [f"{pad}skip({stmt.lifeline.name})"]
        if isinstance(stmt, IfStmt):
            owner = stmt.owner.name if self._selected(stmt.owner) else f"external({stmt.owner.name!r})"
            true_lines = self.render(stmt.branch_true, indent + 1) or ["    " * (indent + 1) + "pass"]
            false_lines = self.render(stmt.branch_false, indent + 1) or ["    " * (indent + 1) + "pass"]
            return [
                f"{pad}if ({_condition(stmt.condition)}) @ {owner}:",
                *true_lines,
                f"{pad}else:",
                *false_lines,
            ]
        if isinstance(stmt, WhileStmt):
            owner = stmt.owner.name if self._selected(stmt.owner) else f"external({stmt.owner.name!r})"
            body = self.render(stmt.body, indent + 1) or ["    " * (indent + 1) + "pass"]
            lines = [f"{pad}while ({_condition(stmt.condition)}) @ {owner}:", *body]
            exit_lines = self.render(stmt.exit_body, indent + 1)
            if exit_lines:
                lines.extend([f"{pad}else:", *exit_lines])
            return lines
        if isinstance(stmt, ParallelStmt):
            branches = [branch for branch in stmt.branches if self._relevant(branch)]
            lines = [f"{pad}with parallel:"]
            for branch in branches:
                lines.append(f"{pad}    with branch:")
                lines.extend(self.render(branch, indent + 2) or ["    " * (indent + 2) + "pass"])
            return lines
        raise TypeError(f"unsupported global statement: {type(stmt).__name__}")


class _LocalRenderer:
    def __init__(self, *, communications_only: bool, detail: str) -> None:
        self.communications_only = communications_only
        self.detail = detail

    def _relevant(self, stmt: AnyStmt) -> bool:
        if isinstance(stmt, EmptyStmt):
            return False
        if isinstance(stmt, ActStmt):
            return not self.communications_only
        if isinstance(stmt, SkipStmt):
            return self.detail == "full" and not self.communications_only
        if isinstance(stmt, SelfAssignStmt):
            return not self.communications_only
        if isinstance(stmt, SeqStmt):
            return self._relevant(stmt.first) or self._relevant(stmt.second)
        if isinstance(stmt, (IfStmt, IfRecvStmt)):
            return self._relevant(stmt.branch_true) or self._relevant(stmt.branch_false)
        if isinstance(stmt, (WhileStmt, WhileRecvStmt)):
            return self._relevant(stmt.body) or self._relevant(stmt.exit_body)
        if isinstance(stmt, ParallelLocalStmt):
            return any(self._relevant(branch) for branch in stmt.branches)
        return True

    def render(self, stmt: AnyStmt, indent: int = 0) -> list[str]:
        pad = "    " * indent
        if not self._relevant(stmt):
            return []
        if isinstance(stmt, SeqStmt):
            lines: list[str] = []
            for part in _flatten_seq(stmt):
                lines.extend(self.render(part, indent))
            return lines
        if isinstance(stmt, SendStmt):
            if len(stmt.payload) == 2 and is_kappa_ctrl(stmt.payload[1]):
                return [
                    f"{pad}send_decision({stmt.receiver.name!r}, "
                    f"{_expr(stmt.payload[0])}{_channel_suffix(stmt.channel)})"
                ]
            return [
                f"{pad}send({stmt.receiver.name!r}, {_exprs(stmt.payload)}"
                f"{_channel_suffix(stmt.channel)})"
            ]
        if isinstance(stmt, RecvStmt):
            bindings = _exprs(stmt.bindings)
            prefix = f"{bindings} = " if bindings else ""
            return [
                f"{pad}{prefix}recv({stmt.sender.name!r}{_channel_suffix(stmt.channel)})"
            ]
        if isinstance(stmt, ReceiveAnyStmt):
            options = ", ".join(
                f"{sender.name!r}: ({_exprs(bindings)})" for sender, bindings in stmt.receives
            )
            return [f"{pad}recv_any({{{options}}}{_channel_suffix(stmt.channel)})"]
        if isinstance(stmt, SelfAssignStmt):
            return [f"{pad}{_exprs(stmt.bindings)} = {_exprs(stmt.payload)}"]
        if isinstance(stmt, ActStmt):
            outputs = ", ".join(value.name for value in stmt.outputs)
            return [f"{pad}{outputs} = {stmt.action.name}({_exprs(stmt.inputs)})"]
        if isinstance(stmt, SkipStmt):
            return [f"{pad}pass  # skip {stmt.lifeline.name}"]
        if isinstance(stmt, IfStmt):
            true_lines = self.render(stmt.branch_true, indent + 1) or ["    " * (indent + 1) + "pass"]
            false_lines = self.render(stmt.branch_false, indent + 1) or ["    " * (indent + 1) + "pass"]
            return [
                f"{pad}if {_condition(stmt.condition)}:",
                *true_lines,
                f"{pad}else:",
                *false_lines,
            ]
        if isinstance(stmt, IfRecvStmt):
            true_lines = self.render(stmt.branch_true, indent + 1) or ["    " * (indent + 1) + "pass"]
            false_lines = self.render(stmt.branch_false, indent + 1) or ["    " * (indent + 1) + "pass"]
            return [
                f"{pad}if recv_decision({stmt.sender.name!r}{_channel_suffix(stmt.channel)}):",
                *true_lines,
                f"{pad}else:",
                *false_lines,
            ]
        if isinstance(stmt, WhileStmt):
            body = self.render(stmt.body, indent + 1) or ["    " * (indent + 1) + "pass"]
            lines = [f"{pad}while {_condition(stmt.condition)}:", *body]
            exit_lines = self.render(stmt.exit_body, indent + 1)
            if exit_lines:
                lines.extend([f"{pad}else:", *exit_lines])
            return lines
        if isinstance(stmt, WhileRecvStmt):
            body = self.render(stmt.body, indent + 1) or ["    " * (indent + 1) + "pass"]
            lines = [
                f"{pad}while recv_decision({stmt.sender.name!r}{_channel_suffix(stmt.channel)}):",
                *body,
            ]
            exit_lines = self.render(stmt.exit_body, indent + 1)
            if exit_lines:
                lines.extend([f"{pad}else:", *exit_lines])
            return lines
        if isinstance(stmt, ParallelLocalStmt):
            indices = stmt.branch_indices or tuple(range(len(stmt.branches)))
            lines = [f"{pad}with local_parallel:"]
            for index, branch_stmt in zip(indices, stmt.branches):
                if not self._relevant(branch_stmt):
                    continue
                lines.append(f"{pad}    with branch({index + 1}):")
                lines.extend(self.render(branch_stmt, indent + 2) or ["    " * (indent + 2) + "pass"])
            return lines
        raise TypeError(f"unsupported local statement: {type(stmt).__name__}")


def _workflow_signature(workflow: Workflow, *, agent: str | None = None) -> str:
    inputs = []
    for name, value_type, lifeline in workflow.inputs:
        if agent and (lifeline is None or lifeline.name != agent):
            continue
        annotation = _type_name(value_type)
        if lifeline is not None and not agent:
            annotation += f" @ {lifeline.name}"
        inputs.append(f"{name}: {annotation}")
    if agent and not any(lifeline.name == agent for _value, lifeline in workflow.outputs):
        result = "None"
    else:
        result = _type_name(workflow.output_type)
    name = workflow.name if not agent else f"{workflow.name}__{agent}"
    return f"def {name}({', '.join(inputs)}) -> {result}:"


def _overview(workflow: Workflow, module: ModuleType | None) -> str:
    lifelines = _ordered_workflow_lifelines(workflow)
    actions = _actions(workflow.body)
    lines = [
        f"# workflow: {workflow.name}",
        f"# lifelines: {', '.join(item.name for item in lifelines) or '(none)'}",
        "# inputs: " + (
            ", ".join(
                f"{name}: {_type_name(value)}" + (f" @ {lifeline.name}" if lifeline else "")
                for name, value, lifeline in workflow.inputs
            ) or "(none)"
        ),
        "# outputs: " + (
            ", ".join(f"{value.name}: {_type_name(value.type)} @ {lifeline.name}" for value, lifeline in workflow.outputs)
            or "(none)"
        ),
        "# actions: " + (", ".join(f"{_action_kind(action)}:{action.name}" for action in actions) or "(none)"),  # type: ignore[attr-defined]
    ]
    if module is not None:
        spec = deployment_spec_from_module(module)
        lines.append(
            f"# deployment: {len(spec.fields)} fields, {len(spec.packages)} packages, "
            f"{len(spec.setup)} setup steps"
        )
    lines.extend(["", "@workflow", _workflow_signature(workflow), "    ..."])
    return "\n".join(lines)


def workflow_view_data(
    workflow: Workflow,
    module: ModuleType | None = None,
    *,
    options: ViewOptions | None = None,
) -> dict[str, object]:
    options = options or ViewOptions()
    lifelines = _ordered_workflow_lifelines(workflow)
    names = {item.name for item in lifelines}
    requested = ({options.agent} if options.agent else set(options.agents)) - {None}
    unknown = sorted(str(item) for item in requested if item not in names)
    if unknown:
        raise ValueError(
            f"unknown agent(s): {', '.join(unknown)}; available: {', '.join(sorted(names))}"
        )

    code = render_workflow(workflow, module, options=options)
    actions = _actions(workflow.body)
    data: dict[str, object] = {
        "workflow": workflow.name,
        "detail": options.detail,
        "communications_only": options.communications_only,
        "agent": options.agent,
        "agents": list(options.agents),
        "lifelines": [item.name for item in lifelines],
        "inputs": [
            {
                "name": name,
                "type": _type_name(value),
                "lifeline": lifeline.name if lifeline else None,
            }
            for name, value, lifeline in workflow.inputs
        ],
        "outputs": [
            {"name": value.name, "type": _type_name(value.type), "lifeline": lifeline.name}
            for value, lifeline in workflow.outputs
        ],
        "actions": [
            {"name": action.name, "kind": _action_kind(action), "signature": _action_signature(action)}  # type: ignore[attr-defined]
            for action in actions
        ],
        "code": code,
    }
    if module is not None:
        data["deployment"] = deployment_spec_from_module(module).as_dict()
    return data


def render_workflow(
    workflow: Workflow,
    module: ModuleType | None = None,
    *,
    options: ViewOptions | None = None,
) -> str:
    options = options or ViewOptions()
    if options.detail == "overview":
        return _overview(workflow, module)

    lifelines = _ordered_workflow_lifelines(workflow)
    if options.agent:
        lifeline = next((item for item in lifelines if item.name == options.agent), None)
        if lifeline is None:
            raise ValueError(
                f"unknown agent {options.agent!r}; available: "
                f"{', '.join(item.name for item in lifelines)}"
            )
        local = project(workflow, lifeline)
        body = _LocalRenderer(
            communications_only=options.communications_only,
            detail=options.detail,
        ).render(local, 1)
        lines = [
            f"# Generated local projection for {lifeline.name}; read-only code view.",
            "# send/recv/recv_decision denote projected runtime operations.",
            "",
            f"@role({lifeline.name!r})",
            _workflow_signature(workflow, agent=lifeline.name),
            *(body or ["    pass"]),
        ]
        return "\n".join(lines)

    selected = frozenset(options.agents) if options.agents else None
    renderer = _GlobalRenderer(
        selected=selected,
        communications_only=options.communications_only,
        detail=options.detail,
    )
    body = renderer.render(workflow.body, 1)
    lines: list[str] = []
    if selected:
        lines.extend([
            f"# Generated focus view for: {', '.join(options.agents)}.",
            "# external(name, ...) preserves interactions with hidden lifelines.",
            "",
        ])

    if options.detail in {"actions", "full"} and not options.communications_only:
        action_owners = _action_owners(workflow.body)
        actions = [
            action
            for action in _actions(workflow.body)
            if selected is None
            or bool(action_owners.get(id(action), set()) & selected)
        ]
        if actions:
            lines.append("# Actions")
            for index, action in enumerate(actions):
                if index:
                    lines.append("")
                lines.extend(_render_action(action, full=options.detail == "full"))
            lines.append("")

    lines.extend(["@workflow", _workflow_signature(workflow), *(body or ["    pass"])])
    if workflow.outputs and not selected:
        output_lines = [
            f"    return {value.name} @ {lifeline.name}"
            for value, lifeline in workflow.outputs
        ]
        # The IR body does not contain the source return, so append it explicitly.
        lines.extend(output_lines)

    if options.detail == "full" and module is not None:
        declaration = deployment_spec_from_module(module)
        if declaration.fields or declaration.packages or declaration.setup or declaration.files:
            lines.extend([
                "",
                "# Deployment declaration (normalized data form)",
                "zippergen_deployment = "
                + pprint.pformat(declaration.as_dict(), sort_dicts=False, width=100),
            ])
    return "\n".join(lines)


def render_workflow_json(
    workflow: Workflow,
    module: ModuleType | None = None,
    *,
    options: ViewOptions | None = None,
) -> str:
    return json.dumps(workflow_view_data(workflow, module, options=options), indent=2, default=str)


__all__ = [
    "DETAILS",
    "ViewOptions",
    "render_workflow",
    "render_workflow_json",
    "workflow_view_data",
]
