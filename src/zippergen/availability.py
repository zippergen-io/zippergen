"""Definite per-lifeline variable availability.

The DSL's ownership rule is syntax-directed: a lifeline may read a variable
only after owning it as an input, starting with an explicit default, producing
it, or receiving it.  This module owns the control-flow state operations used
by both handwritten-workflow validation and planner-source validation.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from zippergen.syntax import (
    ActStmt,
    EmptyStmt,
    IfStmt,
    LitExpr,
    MsgStmt,
    ParallelStmt,
    SeqStmt,
    SkipStmt,
    Var,
    VarExpr,
    WhileStmt,
    Workflow,
    occurring_variable_declarations,
    occurring_variables,
    _ordered_workflow_lifelines,
)

__all__ = [
    "AvailabilityState",
    "AvailabilityViolation",
    "require_workflow_availability",
    "workflow_availability_error",
]


class AvailabilityViolation(ValueError):
    """A participant reads a variable absent on at least one control path."""

    @classmethod
    def for_read(
        cls,
        lifeline: str,
        variable: str,
        operation: str,
    ) -> "AvailabilityViolation":
        return cls(
            f"{operation} reads {variable!r}, but participant {lifeline!r} does "
            "not definitely have it. A participant may read only an owned "
            "workflow input, an explicit default, an action output, or a "
            "received value available on every control-flow path."
        )


@dataclass
class AvailabilityState:
    """Definitely available variable names, independently for each lifeline."""

    scopes: dict[str, set[str]]

    @classmethod
    def empty(cls) -> "AvailabilityState":
        return cls({})

    def copy(self) -> "AvailabilityState":
        return AvailabilityState({name: set(values) for name, values in self.scopes.items()})

    def bind(self, lifeline: str, names: set[str] | tuple[str, ...] | list[str]) -> None:
        self.scopes.setdefault(lifeline, set()).update(names)

    def require(self, lifeline: str, variable: str, operation: str) -> None:
        if variable in self.scopes.get(lifeline, set()):
            return
        raise AvailabilityViolation.for_read(lifeline, variable, operation)

    def replace_with(self, other: "AvailabilityState") -> None:
        self.scopes = {
            name: set(values) for name, values in other.scopes.items()
        }

    def merge_alternatives(
        self,
        left: "AvailabilityState",
        right: "AvailabilityState",
    ) -> None:
        """Keep only names available after either branch."""

        self.scopes = {
            lifeline: left.scopes.get(lifeline, set())
            & right.scopes.get(lifeline, set())
            for lifeline in set(left.scopes) | set(right.scopes)
        }

    def merge_parallel(self, branches: list["AvailabilityState"]) -> None:
        """After every parallel branch completes, all of their bindings exist."""

        merged: dict[str, set[str]] = {}
        for branch in branches:
            for lifeline, names in branch.scopes.items():
                merged.setdefault(lifeline, set()).update(names)
        self.scopes = merged


def _variable_names(exprs) -> tuple[str, ...]:
    return tuple(expr.var.name for expr in exprs if isinstance(expr, VarExpr))


class _GuardNameVisitor(ast.NodeVisitor):
    """Collect ordinary guard names while treating formula calls as opaque."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        # The builder permits only causal-past formula calls. They read monitor
        # state, not the participant's variable environment.
        return

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        self.names.add(node.id)


def _guard_variables(condition: object, known_variables: frozenset[str]) -> tuple[str, ...]:
    source = getattr(condition, "_src", None)
    if isinstance(source, str):
        try:
            tree = ast.parse(source, mode="eval")
        except SyntaxError:
            return ()
        visitor = _GuardNameVisitor()
        visitor.visit(tree)
        return tuple(sorted(visitor.names & known_variables))

    code = getattr(condition, "__code__", None)
    names = set(getattr(code, "co_names", ()))
    return tuple(sorted(names & known_variables))


def _initial_state(
    workflow: Workflow,
    initial_envs: dict[str, dict[str, object]] | None = None,
) -> AvailabilityState:
    state = AvailabilityState.empty()
    for name, _ztype, owner in workflow.inputs:
        if owner is not None:
            state.bind(owner.name, (name,))

    declared_variables = {
        *workflow.vars,
        *(value for value in workflow.ns.values() if isinstance(value, Var)),
        *occurring_variable_declarations(workflow.body),
        *(variable for variable, _lifeline in workflow.outputs),
    }
    defaults = {
        value.name
        for value in declared_variables
        if value.has_default
    }
    if defaults:
        for lifeline in _ordered_workflow_lifelines(workflow):
            state.bind(lifeline.name, defaults)
    for lifeline, values in (initial_envs or {}).items():
        state.bind(lifeline, tuple(values))
    return state


def _check_statement(
    stmt,
    state: AvailabilityState,
    known_variables: frozenset[str],
) -> None:
    match stmt:
        case EmptyStmt() | SkipStmt():
            return

        case MsgStmt(sender=sender, payload=payload, receiver=receiver, bindings=bindings):
            for name in _variable_names(payload):
                state.require(sender.name, name, f"message from {sender.name!r}")
            state.bind(receiver.name, _variable_names(bindings))

        case ActStmt(lifeline=lifeline, action=action, inputs=inputs, outputs=outputs):
            for name in _variable_names(inputs):
                state.require(
                    lifeline.name,
                    name,
                    f"action {action.name!r} on {lifeline.name!r}",
                )
            state.bind(lifeline.name, tuple(var.name for var in outputs))

        case SeqStmt(first=first, second=second):
            _check_statement(first, state, known_variables)
            _check_statement(second, state, known_variables)

        case IfStmt(
            condition=condition,
            owner=owner,
            branch_true=branch_true,
            branch_false=branch_false,
        ):
            for name in _guard_variables(condition, known_variables):
                state.require(owner.name, name, f"if guard on {owner.name!r}")
            true_state = state.copy()
            false_state = state.copy()
            _check_statement(branch_true, true_state, known_variables)
            _check_statement(branch_false, false_state, known_variables)
            state.merge_alternatives(true_state, false_state)

        case WhileStmt(condition=condition, owner=owner, body=body, exit_body=exit_body):
            for name in _guard_variables(condition, known_variables):
                state.require(owner.name, name, f"while guard on {owner.name!r}")
            body_state = state.copy()
            _check_statement(body, body_state, known_variables)
            # The body may run zero times. Only the entry state and the exit
            # branch therefore establish what is available after the loop.
            exit_state = state.copy()
            _check_statement(exit_body, exit_state, known_variables)
            state.replace_with(exit_state)

        case ParallelStmt(branches=branches):
            branch_states: list[AvailabilityState] = []
            for branch in branches:
                branch_state = state.copy()
                _check_statement(branch, branch_state, known_variables)
                branch_states.append(branch_state)
            state.merge_parallel(branch_states)

        case _:
            raise TypeError(
                f"Variable availability has no rule for {type(stmt).__name__}"
            )


def require_workflow_availability(
    workflow: Workflow,
    initial_envs: dict[str, dict[str, object]] | None = None,
) -> None:
    """Raise when a global workflow reads a variable absent at its lifeline."""

    state = _initial_state(workflow, initial_envs)
    known_variables = (
        occurring_variables(workflow.body)
        | frozenset(name for name, _ztype, _owner in workflow.inputs)
        | frozenset(var.name for var, _owner in workflow.outputs)
        | frozenset(
            value.name
            for value in workflow.ns.values()
            if isinstance(value, Var)
        )
    )
    _check_statement(workflow.body, state, known_variables)
    for variable, lifeline in workflow.outputs:
        state.require(
            lifeline.name,
            variable.name,
            f"return from {lifeline.name!r}",
        )


def workflow_availability_error(workflow: Workflow) -> str | None:
    """Return the first definite-availability error, or ``None``."""

    try:
        require_workflow_availability(workflow)
    except AvailabilityViolation as exc:
        return str(exc)
    return None
