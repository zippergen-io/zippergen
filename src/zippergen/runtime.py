"""
Layer 5: Runtime executor. Projects the workflow onto each lifeline, runs one
thread per lifeline, wires FIFO queues, and drives execution to completion.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast
import time
import textwrap

from zippergen.planner import _exec_planner, _validate_planner_spec
from zippergen.human_tasks import validate_human_action_result
from zippergen.errors import WorkflowCancelled

from zippergen.llm_policy import (
    attempt_llm_action,
    checked_llm_outputs,
    retry_reporter,
)
from zippergen.syntax import (
    EmptyStmt, SendStmt, RecvStmt, ReceiveAnyStmt, SelfAssignStmt, ActStmt, SkipStmt,
    SeqStmt, IfStmt, WhileStmt, IfRecvStmt, WhileRecvStmt,
    ParallelStmt, ParallelLocalStmt,
    VarExpr, LitExpr, Var, Json,
    LLMAction, PureAction, EffectAction, AssistantAction, PlannerAction, HumanAction,
    Lifeline, Workflow, LocalStmt, AnyStmt,
    is_kappa_ctrl,
    _clone_zvalue,
    _ordered_workflow_lifelines,
    seq,
    validate_zvalue,
)
from zippergen.control import PartialReceiveAny, Residual
from zippergen.projection import project
from zippergen.formula import Formula as _Formula, subformulas as _subformulas
from zippergen.monitor import MonitorState
from zippergen.channels import _SeqQueue, InProcessChannel  # noqa: F401

__all__ = ["run", "mock_llm", "console_trace", "tee_traces"]


@dataclass(frozen=True)
class PendingExternal:
    """Durable-mode signal: an external act must run with no transaction open.

    Carries the node (for its action and outputs) and the evaluated inputs.
    Returned in the residual slot with progressed=False; never produced outside
    durable mode."""
    node: object
    inputs: dict
    trace_start: dict | None = None
    trace_seq: int | None = None


@dataclass(frozen=True)
class _ResolvedExternal:
    """Result returned by the durable driver after an outside-world call."""

    outputs: dict[str, object]
    trace_seq: int | None


# ---------------------------------------------------------------------------
# Default LLM backend — simple values for built-in scalar types
# ---------------------------------------------------------------------------

def mock_llm(action: LLMAction, inputs: dict[str, object], *,
             min_delay: float = 0.0, max_delay: float = 0.0):
    """
    Trivial mock: Bool outputs → random True/False; Text outputs → sentinel;
    Int outputs → random integers; Float outputs → random floats.

    ``min_delay`` / ``max_delay`` add a random sleep to simulate LLM latency.
    Pass a backend via ``llm_backend=lambda a, i: mock_llm(a, i, min_delay=0.3, max_delay=1.2)``.
    """
    import random
    if max_delay > 0:
        time.sleep(random.uniform(min_delay, max_delay))
    result: dict[str, object] = {}
    for name, ztype in action.outputs:
        if ztype is bool:
            result[name] = random.choice([True, False])
        elif ztype is int:
            result[name] = random.randint(0, 10)
        elif ztype is float:
            result[name] = random.uniform(0.0, 10.0)
        elif ztype is str:
            result[name] = f"[{action.name}:{name}]"
        elif ztype is Json:
            result[name] = {}
        else:
            result[name] = None
    return result


# ---------------------------------------------------------------------------
# Default tracer — pretty-prints structured event dicts to stdout
# ---------------------------------------------------------------------------

_print_lock = threading.Lock()
_act_seq_lock = threading.Lock()
_act_seq = 0


def _next_act_seq() -> int:
    global _act_seq
    with _act_seq_lock:
        seq = _act_seq
        _act_seq += 1
        return seq


def _format_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, list):
        return "[" + ", ".join(_format_scalar(v) for v in value) + "]"
    return str(value)


def _format_mapping_lines(mapping: dict[str, object], *, width: int = 88) -> list[str]:
    lines: list[str] = []
    for key, value in mapping.items():
        rendered = _format_scalar(value)
        wrapped = textwrap.wrap(
            rendered,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
            replace_whitespace=False,
            drop_whitespace=False,
        ) or [""]
        lines.append(f"    {key} = {wrapped[0]}")
        for extra in wrapped[1:]:
            lines.append(f"      {extra}")
    return lines


def _format_sequence_lines(values: list[object], *, width: int = 88) -> list[str]:
    lines: list[str] = []
    for idx, value in enumerate(values, start=1):
        if isinstance(value, str) and value.startswith("κ_ctrl_"):
            continue
        rendered = _format_scalar(value)
        wrapped = textwrap.wrap(
            rendered,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
            replace_whitespace=False,
            drop_whitespace=False,
        ) or [""]
        lines.append(f"    [{idx}] = {wrapped[0]}")
        for extra in wrapped[1:]:
            lines.append(f"          {extra}")
    return lines


def console_trace(event: dict) -> None:
    lifeline = event.get("lifeline") or threading.current_thread().name
    t = event["type"]
    lines: list[str] | None = None

    if t == "send":
        is_ctrl = any(isinstance(v, str) and v.startswith("κ_ctrl_") for v in (event.get("values") or []))
        lines = [f"[{lifeline}] {'control' if is_ctrl else 'send'} -> {event['to']}"]
        payload_lines = _format_sequence_lines(event.get("values") or [])
        if payload_lines:
            lines.append("  payload")
            lines.extend(payload_lines)
        else:
            lines.append("  payload")
            lines.append("    (empty)")
    elif t == "recv":
        is_ctrl = bool(event.get("ctrl"))
        lines = [f"[{lifeline}] {'control' if is_ctrl else 'recv'} <- {event['from']}"]
        bindings = event.get("bindings") or {}
        if bindings:
            lines.append("  bindings")
            lines.extend(_format_mapping_lines(bindings))
        else:
            lines.append("  bindings")
            lines.append("    (none)")
    elif t == "llm_retry":
        lines = [f"[{lifeline}] --- {event['action']} retrying ---"]
        lines.append(f"  {event.get('detail') or 'retrying'}")
    elif t == "act_start":
        lines = [f"[{lifeline}] --- {event['action']} ---"]
        inputs = event.get("inputs") or {}
        if inputs:
            lines.append("  input")
            lines.extend(_format_mapping_lines(inputs))
        else:
            lines.append("  input")
            lines.append("    (none)")
    elif t == "act":
        lines = [f"[{lifeline}] --- {event['action']} done ---"]
        outputs = event.get("outputs") or {}
        if outputs:
            lines.append("  output")
            lines.extend(_format_mapping_lines(outputs))
        else:
            lines.append("  output")
            lines.append("    (none)")
    elif t == "decision":
        kind = event.get("kind", "if")
        val  = event.get("value")
        cond = event.get("formula") or event.get("condition")
        if kind == "if":
            label = "⊤ true" if val else "⊥ false"
        else:
            label = "↻ continue" if val else "⊥ exit"
        suffix = f" ({cond})" if cond else ""
        lines = [f"[{lifeline}] {kind}{suffix}: {label}"]

    if not lines:
        return

    with _print_lock:
        print("\n".join(lines))
        if t in {"act_start", "act"}:
            print()


def tee_traces(*traces):
    active = [trace for trace in traces if trace is not None]

    def _trace(event: dict) -> None:
        for trace in active:
            trace(event)

    return _trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

Env = dict[str, object]


class _CondEnv:
    """Attribute-access proxy for condition lambdas.

    Resolves ``_e.name`` by looking up ``name`` in the lifeline's local env
    first, then falling back to the workflow's global namespace (for constants
    like ``MAX_ROUNDS``).
    """
    __slots__ = ("_env", "_ns")

    def __init__(self, env: dict, ns: dict) -> None:
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "_ns", ns)

    def __getattr__(self, name: str) -> object:
        env = object.__getattribute__(self, "_env")
        ns  = object.__getattribute__(self, "_ns")
        if name in env:
            return env[name]
        if name in ns:
            return ns[name]
        raise AttributeError(
            f"Condition variable {name!r} not found in env or workflow namespace"
        )


class _ConditionValueUnavailable(LookupError):
    """Formula discovery reached a value that exists only during execution."""


class _FormulaProbeEnv(_CondEnv):
    """Resolve formula globals without pretending workflow variables have values."""

    def __getattr__(self, name: str) -> object:
        ns = object.__getattribute__(self, "_ns")
        if name not in ns:
            raise _ConditionValueUnavailable(name)
        value = ns[name]
        if isinstance(value, Var):
            raise _ConditionValueUnavailable(name)
        return value


def _eval(expr, env: Env) -> object:
    match expr:
        case VarExpr(var=v):
            return env.get(v.name, v.default)
        case LitExpr(value=val):
            return val
        case _:
            raise TypeError(f"Unknown expr: {type(expr).__name__}")


def _bind(bindings: tuple, values: tuple, env: Env) -> None:
    for binding, value in zip(bindings, values):
        if isinstance(binding, VarExpr):
            env[binding.var.name] = value
        elif isinstance(binding, LitExpr) and value != binding.value:
            raise RuntimeError(
                f"received value {value!r} does not match literal binding {binding.value!r}"
            )


def _jsonify(value: object) -> object:
    """Convert a Python runtime value to a JSON-safe object."""
    if isinstance(value, (bool, int, float, str, type(None))):
        return value
    if isinstance(value, tuple):
        return [_jsonify(v) for v in value]
    return str(value)


def _bound_dict(bindings: tuple, values: tuple) -> dict:
    """Build a {var_name: value} dict from a binding/value pair, skipping kappa."""
    return {
        b.var.name: _jsonify(v)
        for b, v in zip(bindings, values)
        if isinstance(b, VarExpr) and not is_kappa_ctrl(b)
    }


def _monitor_trace_fields(monitor) -> dict[str, object]:
    if not monitor:
        return {}
    return {"vc": monitor.snapshot_vc()}


def _recv_trace_fields(monitor, message_vc: dict | None) -> dict[str, object]:
    fields = _monitor_trace_fields(monitor)
    if monitor and message_vc is not None:
        fields["message_vc"] = dict(message_vc)
    return fields


def _action_kind(action: object) -> str:
    if isinstance(action, PureAction):
        return "pure"
    if isinstance(action, EffectAction):
        return "effect"
    if isinstance(action, AssistantAction):
        return "assistant"
    if isinstance(action, PlannerAction):
        return "planner"
    if isinstance(action, HumanAction):
        return "human"
    if isinstance(action, LLMAction):
        return "llm"
    return "act"


def _action_visible(action: object) -> bool:
    return (
        not isinstance(action, (PureAction, EffectAction, AssistantAction, HumanAction))
        or action.visible
    )


def _receive_any(
    ch: InProcessChannel,
    receiver: str,
    pending_senders: set[str],
    channel: str,
    *,
    stop: threading.Event | None = None,
) -> tuple[str, tuple[int, tuple, dict | None, dict | None, dict | None]]:
    while True:
        try_get_any = getattr(ch, "try_get_any", None)
        if try_get_any is not None:
            selected = try_get_any(receiver, pending_senders, channel)
            if selected is not None:
                return selected
        for sender in sorted(pending_senders):
            item = ch.try_get(sender, receiver, channel)
            if item is not None:
                return sender, item
        if stop is not None and stop.is_set():
            raise WorkflowCancelled("Workflow cancelled: another lifeline failed")
        time.sleep(0.01)



# ---------------------------------------------------------------------------
# Local-program interpreter
# ---------------------------------------------------------------------------

def _try_channel_get(ch, sender: str, receiver: str, channel: str):
    return ch.try_get(sender, receiver, channel)


def _try_channel_get_any(ch, receiver: str, pending_senders: set[str], channel: str):
    try_get_any = getattr(ch, "try_get_any", None)
    if try_get_any is not None:
        return try_get_any(receiver, pending_senders, channel)
    for sender in sorted(pending_senders):
        item = ch.try_get(sender, receiver, channel)
        if item is not None:
            return sender, item
    return None


def _with_parallel_branch(trace, label: str):
    if trace is None:
        return None

    def wrapped(event: dict) -> None:
        if "parallel_branch" not in event:
            event = {**event, "parallel_branch": label}
        trace(event)

    return wrapped


def _python_action_out_map(action, values: tuple, outs) -> dict:
    raw = action.fn(*values)
    result = {outs[0].name: raw} if len(outs) == 1 else {
        var.name: val for var, val in zip(outs, cast(tuple, raw))
    }
    return _validate_action_out_map(
        action,
        result,
        outs,
        source="Python action",
    )


def _validate_action_out_map(
    action,
    values: dict[str, object],
    outs,
    *,
    source: str,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for (action_name, expected_type), var in zip(action.outputs, outs):
        if var.name not in values:
            raise RuntimeError(
                f"{source} for '{action.name}' did not return required output "
                f"{action_name!r}."
            )
        result[var.name] = validate_zvalue(
            values[var.name],
            expected_type,
            context=(
                f"{source} for '{action.name}' output {action_name!r}"
            ),
        )
    return result


def _assistant_action_out_map(action, named_outputs, outs) -> dict:
    result: dict[str, object] = {}
    for (action_name, expected_type), var in zip(action.outputs, outs):
        if action_name not in named_outputs:
            raise RuntimeError(
                f"Assistant backend for '{action.name}' did not return "
                f"required output {action_name!r}."
            )
        value = named_outputs[action_name]
        result[var.name] = validate_zvalue(
            value,
            expected_type,
            context=(
                f"Assistant backend for '{action.name}' output "
                f"{action_name!r}"
            ),
        )
    return result


def llm_out_map(
    action,
    named_inputs: dict,
    outs,
    llm_backend,
    *,
    stop=None,
    trace=None,
) -> dict:
    """Call the model until the answer is usable, or the policy gives up.

    One attempt is transport, parse, coercion and output validation together,
    because a well-formed response carrying the wrong type has failed just as
    surely as a refused connection, and asking again may fix either.
    """

    def attempt() -> dict:
        named_outputs = llm_backend(action, named_inputs)
        return checked_llm_outputs(action, named_outputs)

    # One namespace throughout: a real answer and a declared fallback both
    # arrive keyed by the action's own output names, already checked against
    # the declared types. Guessing which namespace a result was in silently
    # mixed them up when a variable happened to share another output's name.
    outputs = attempt_llm_action(
        action,
        attempt,
        stop=stop,
        report=retry_reporter(trace, action),
        check=lambda values: checked_llm_outputs(action, values),
    )
    result = {
        var.name: outputs[aname] for (aname, _), var in zip(action.outputs, outs)
    }
    return _validate_action_out_map(action, result, outs, source="LLM backend")


def external_out_map(
    action,
    named_inputs,
    outs,
    llm_backend,
    human_backend,
    assistant_backend,
    *,
    stop=None,
    trace=None,
) -> dict:
    """Return the environment delta for a durable external action.

    This is the same LLM/Human/Planner/Effect/Assistant computation performed
    by ``_exec``, factored for the durable driver.
    """
    if isinstance(action, EffectAction):
        values = tuple(named_inputs[name] for name, _ in action.inputs)
        return _python_action_out_map(action, values, outs)
    if isinstance(action, PlannerAction):
        result = {
            outs[0].name: _exec_planner(
                action,
                named_inputs,
                llm_backend,
                trace=trace,
                parent_seq=_next_act_seq(),
                stop=stop,
            )
        }
        return _validate_action_out_map(
            action,
            result,
            outs,
            source="Planner action",
        )
    if isinstance(action, AssistantAction):
        named_outputs = assistant_backend(action, named_inputs)
        return _assistant_action_out_map(action, named_outputs, outs)
    if isinstance(action, HumanAction):
        if not action.visible:
            default = True if action.output_type is bool else ""
            return {outs[0].name: default}
        named_outputs = validate_human_action_result(
            action, named_inputs, human_backend(action, named_inputs)
        )
        return {outs[0].name: named_outputs[action.output]}
    return llm_out_map(  # LLMAction
        action, named_inputs, outs, llm_backend, stop=stop, trace=trace
    )


def _input_hash(named_inputs: dict) -> str | None:
    try:
        return hashlib.sha1(json.dumps(named_inputs, sort_keys=True).encode()).hexdigest()[:16]
    except (TypeError, ValueError):
        return None   # non-serializable inputs (incl. circular refs) -> skip hash (locator+kind still assert)


def _step(
    stmt: Residual,
    env: Env,
    ch: InProcessChannel,
    ns: dict,
    llm_backend,
    human_backend,
    monitor,
    trace,
    formula_conditions: dict[int, _Formula],
    stop: threading.Event | None,
    durable: bool = False,
    assistant_backend=None,
    resolved: dict[int, _ResolvedExternal] | None = None,
) -> tuple[Residual | PendingExternal, bool]:
    """Execute at most one enabled local step.

    Returns ``(residual, progressed)``. Blocking receives return the original
    residual with ``progressed=False`` so the local parallel scheduler can try
    another branch.

    In durable mode an external act (LLM/assistant/human/effect) is NOT run
    inline. A ``PendingExternal`` comes back instead, so the driver can call the
    outside world with no SQLite transaction open and then commit the result and
    the successor control state together. Pure acts are deterministic, so they
    run inline either way.
    """
    if assistant_backend is None:
        from zippergen.assistant_backends import make_cli_assistant_backend
        assistant_backend = make_cli_assistant_backend()

    match stmt:
        case EmptyStmt():
            return EmptyStmt(), False

        case SkipStmt():
            return EmptyStmt(), True

        case SendStmt() | SelfAssignStmt():
            _exec(
                stmt, env, ch, ns, llm_backend, human_backend, assistant_backend,
                monitor, trace, formula_conditions, stop,
            )
            return EmptyStmt(), True

        case ActStmt(lifeline=_, action=action, inputs=ins, outputs=outs):
            if not durable or isinstance(action, PureAction):
                # in-process, or a pure (deterministic) act that is cheap to redo
                _exec(
                    stmt, env, ch, ns, llm_backend, human_backend, assistant_backend,
                    monitor, trace, formula_conditions, stop,
                )
                return EmptyStmt(), True
            in_vals = tuple(_eval(x, env) for x in ins)
            named_inputs = {name: val for (name, _), val in zip(action.inputs, in_vals)}
            if resolved is not None and id(stmt) in resolved:
                # The driver already ran this action outside the transaction.
                # Applying the outputs and advancing control happen together, in
                # the caller's single commit.
                resolution = resolved[id(stmt)]
                out_map = resolution.outputs
                env.update(out_map)
                if monitor:
                    monitor.on_event("act", env)
                if trace and _action_visible(action):
                    act_seq = resolution.trace_seq
                    if act_seq is None:
                        raise RuntimeError(
                            "Resolved visible external action has no trace sequence."
                        )
                    trace({
                        "type": "act",
                        "lifeline": threading.current_thread().name,
                        "action": action.name,
                        "action_kind": _action_kind(action),
                        "inputs": {k: _jsonify(v) for k, v in named_inputs.items()},
                        "outputs": {k: _jsonify(v) for k, v in out_map.items()},
                        "seq": act_seq,
                        **_monitor_trace_fields(monitor),
                    })
                return EmptyStmt(), True
            trace_start = None
            trace_seq = None
            if trace and _action_visible(action):
                trace_seq = _next_act_seq()
                trace_start = {
                    "type": "act_start",
                    "lifeline": threading.current_thread().name,
                    "action": action.name,
                    "action_kind": _action_kind(action),
                    "inputs": {k: _jsonify(v) for k, v in named_inputs.items()},
                    "seq": trace_seq,
                }
            return PendingExternal(
                stmt,
                named_inputs,
                trace_start=trace_start,
                trace_seq=trace_seq,
            ), False

        case RecvStmt(lifeline=A, bindings=ys, sender=B, channel=channel):
            item = _try_channel_get(ch, B.name, A.name, channel)
            if item is None:
                return stmt, False
            seq_no, values, recv_vc, recv_view, recv_field_view = item
            _bind(ys, values, env)
            if monitor:
                monitor.on_event("recv", env, recv_vc=recv_vc, recv_view=recv_view, recv_field_view=recv_field_view)
            if trace:
                trace({
                    "type": "recv",
                    "to": A.name, "from": B.name,
                    "channel": channel,
                    "bindings": _bound_dict(ys, values),
                    "seq": seq_no,
                    **_recv_trace_fields(monitor, recv_vc),
                })
            return EmptyStmt(), True

        case ReceiveAnyStmt() | PartialReceiveAny():
            origin = stmt.origin if isinstance(stmt, PartialReceiveAny) else stmt
            A = stmt.lifeline
            channel = stmt.channel
            receives = stmt.receives
            pending = {sender.name: (sender, ys) for sender, ys in receives}
            selected = _try_channel_get_any(ch, A.name, set(pending), channel)
            if selected is None:
                return stmt, False
            sender_name, item = selected
            sender, ys = pending[sender_name]
            seq_no, values, recv_vc, recv_view, recv_field_view = item
            _bind(ys, values, env)
            if monitor:
                monitor.on_event("recv", env, recv_vc=recv_vc, recv_view=recv_view, recv_field_view=recv_field_view)
            if trace:
                trace({
                    "type": "recv",
                    "to": A.name, "from": sender.name,
                    "channel": channel,
                    "bindings": _bound_dict(ys, values),
                    "seq": seq_no,
                    **_recv_trace_fields(monitor, recv_vc),
                })
            # Keep pointing at the static node so the control state stays exactly
            # representable; only the outstanding sender set shrinks.
            remaining = tuple(name for name in pending if name != sender_name)
            if not remaining:
                return EmptyStmt(), True
            return PartialReceiveAny(origin, remaining), True

        case SeqStmt(first=p1, second=p2):
            first = cast(Residual, p1)
            second = cast(Residual, p2)
            if isinstance(first, EmptyStmt):
                return second, True
            new_first, progressed = _step(
                first, env, ch, ns, llm_backend, human_backend, monitor, trace,
                formula_conditions, stop, durable=durable,
                assistant_backend=assistant_backend, resolved=resolved,
            )
            if isinstance(new_first, PendingExternal):
                return new_first, False
            if not progressed:
                return stmt, False
            return cast(
                LocalStmt,
                seq(cast(AnyStmt, new_first), cast(AnyStmt, second)),
            ), True

        case IfStmt(condition=c, owner=B, branch_true=t, branch_false=f):
            cached_formula = formula_conditions.get(id(c))
            if cached_formula is not None:
                cond_formula = cached_formula
                cond_value = None
            elif isinstance(c, _Formula):
                cond_formula: _Formula | None = c
                cond_value = None
            else:
                raw = c(_CondEnv(env, ns))
                if isinstance(raw, _Formula):
                    cond_formula = raw
                    cond_value = None
                else:
                    cond_formula = None
                    cond_value = raw
            if cond_formula is not None and monitor is None:
                raise RuntimeError(
                    f"CPL Formula guard {cond_formula!r} on lifeline '{threading.current_thread().name}' "
                    "but no monitor was built. Make the Formula guard discoverable before execution."
                )
            if monitor:
                monitor.on_event("choice", env)
            if cond_formula is not None:
                assert monitor is not None
                flag = monitor.guard_value(cond_formula)
                formula_repr = repr(cond_formula)
            else:
                flag = bool(cond_value)
                formula_repr = None
            if trace:
                trace({"type": "decision", "lifeline": B.name, "kind": "if", "value": flag,
                       "condition": getattr(c, "_src", None), "formula": formula_repr,
                       **_monitor_trace_fields(monitor)})
            return cast(LocalStmt, t if flag else f), True

        case IfRecvStmt(lifeline=A, bindings=ys, sender=B, branch_true=t, branch_false=f, channel=channel):
            item = _try_channel_get(ch, B.name, A.name, channel)
            if item is None:
                return stmt, False
            seq_no, values, recv_vc, recv_view, recv_field_view = item
            _bind(ys, values, env)
            if monitor:
                monitor.on_event("recv", env, recv_vc=recv_vc, recv_view=recv_view, recv_field_view=recv_field_view)
            flag = _eval(ys[0], env) if isinstance(ys[0], VarExpr) else values[0]
            if trace:
                trace({
                    "type": "recv",
                    "to": A.name, "from": B.name,
                    "channel": channel,
                    "bindings": {"branch": "true" if flag else "false"},
                    "seq": seq_no, "ctrl": True,
                    **_recv_trace_fields(monitor, recv_vc),
                })
            return cast(LocalStmt, t if flag else f), True

        case WhileStmt(condition=c, owner=B, body=body, exit_body=exit_b):
            cached_formula = formula_conditions.get(id(c))
            if cached_formula is not None:
                wc_formula = cached_formula
                wc_value = None
            elif isinstance(c, _Formula):
                wc_formula: _Formula | None = c
                wc_value = None
            else:
                wraw = c(_CondEnv(env, ns))
                if isinstance(wraw, _Formula):
                    wc_formula = wraw
                    wc_value = None
                else:
                    wc_formula = None
                    wc_value = wraw
            if wc_formula is not None and monitor is None:
                raise RuntimeError(
                    f"CPL Formula guard {wc_formula!r} on lifeline '{threading.current_thread().name}' "
                    "but no monitor was built. Make the Formula guard discoverable before execution."
                )
            if monitor:
                monitor.on_event("choice", env)
            if wc_formula is not None:
                assert monitor is not None
                flag = monitor.guard_value(wc_formula)
                formula_repr = repr(wc_formula)
            else:
                flag = bool(wc_value)
                formula_repr = None
            if trace:
                trace({"type": "decision", "lifeline": B.name, "kind": "while", "value": flag,
                       "condition": getattr(c, "_src", None), "formula": formula_repr,
                       **_monitor_trace_fields(monitor)})
            if flag:
                return cast(LocalStmt, seq(body, stmt)), True
            return cast(LocalStmt, exit_b), True

        case WhileRecvStmt(lifeline=A, bindings=ys, sender=B, body=body, exit_body=exit_b, channel=channel):
            item = _try_channel_get(ch, B.name, A.name, channel)
            if item is None:
                return stmt, False
            seq_no, values, recv_vc, recv_view, recv_field_view = item
            _bind(ys, values, env)
            if monitor:
                monitor.on_event("recv", env, recv_vc=recv_vc, recv_view=recv_view, recv_field_view=recv_field_view)
            flag = _eval(ys[0], env) if isinstance(ys[0], VarExpr) else values[0]
            if trace:
                trace({
                    "type": "recv",
                    "to": A.name, "from": B.name,
                    "channel": channel,
                    "bindings": {"loop": "continue" if flag else "exit"},
                    "seq": seq_no, "ctrl": True,
                    **_recv_trace_fields(monitor, recv_vc),
                })
            if flag:
                return cast(LocalStmt, seq(body, stmt)), True
            return cast(LocalStmt, exit_b), True

        case ParallelLocalStmt(branches=branches, branch_indices=labels) if not durable:
            _exec(
                stmt, env, ch, ns, llm_backend, human_backend, assistant_backend,
                monitor, trace, formula_conditions, stop,
            )
            return EmptyStmt(), True

        case ParallelLocalStmt(branches=branches, branch_indices=labels):
            residuals: list[Residual] = list(branches)
            for i, branch in enumerate(residuals):
                if isinstance(branch, EmptyStmt):
                    continue
                new_branch, progressed = _step(
                    branch, env, ch, ns, llm_backend, human_backend, monitor, trace,
                    formula_conditions, stop, durable=durable,
                    assistant_backend=assistant_backend, resolved=resolved)
                if isinstance(new_branch, PendingExternal):
                    return new_branch, False           # propagate up; serve resolves
                residuals[i] = new_branch
                if progressed:
                    if all(isinstance(b, EmptyStmt) for b in residuals):
                        return EmptyStmt(), True
                    return ParallelLocalStmt(
                        cast(tuple[LocalStmt, ...], tuple(residuals)), labels
                    ), True
            if all(isinstance(b, EmptyStmt) for b in residuals):
                return EmptyStmt(), True
            return stmt, False

        case _:
            raise TypeError(f"Unknown local stmt: {type(stmt).__name__}")


def _exec(
    stmt: LocalStmt,
    env: Env,
    ch: InProcessChannel,
    ns: dict,
    llm_backend,
    human_backend,
    assistant_backend,
    monitor,
    trace,
    formula_conditions: dict[int, _Formula] | None = None,
    stop: threading.Event | None = None,
) -> None:
    """Execute a LocalStmt, updating env in place."""
    if formula_conditions is None:
        formula_conditions = {}
    match stmt:

        case EmptyStmt() | SkipStmt():
            return

        case SendStmt(lifeline=A, payload=xs, receiver=B, channel=channel):
            values = tuple(copy.deepcopy(_eval(x, env)) for x in xs)
            if monitor:
                monitor.on_event("send", env)
                seq = ch.put(A.name, B.name, channel, values,
                             monitor.snapshot_vc(), monitor.snapshot_view(), monitor.snapshot_field_view())
            else:
                seq = ch.put(A.name, B.name, channel, values)
            if trace:
                names = [x.var.name if isinstance(x, VarExpr) else f"_{i}" for i, x in enumerate(xs)]
                trace({
                    "type": "send",
                    "from": A.name, "to": B.name,
                    "channel": channel,
                    "values": [_jsonify(v) for v in values],
                    "bindings": {name: _jsonify(v) for name, v in zip(names, values)},
                    "seq": seq,
                    **_monitor_trace_fields(monitor),
                })

        case RecvStmt(lifeline=A, bindings=ys, sender=B, channel=channel):
            seq, values, recv_vc, recv_view, recv_field_view = ch.get(B.name, A.name, channel, stop=stop)
            _bind(ys, values, env)
            if monitor:
                monitor.on_event("recv", env, recv_vc=recv_vc, recv_view=recv_view, recv_field_view=recv_field_view)
            if trace:
                trace({
                    "type": "recv",
                    "to": A.name, "from": B.name,
                    "channel": channel,
                    "bindings": _bound_dict(ys, values),
                    "seq": seq,
                    **_recv_trace_fields(monitor, recv_vc),
                })

        case ReceiveAnyStmt(lifeline=A, receives=receives, channel=channel):
            pending = {
                sender.name: (sender, bindings)
                for sender, bindings in receives
            }
            while pending:
                sender_name, item = _receive_any(ch, A.name, set(pending), channel, stop=stop)
                seq, values, recv_vc, recv_view, recv_field_view = item
                sender, ys = pending.pop(sender_name)
                _bind(ys, values, env)
                if monitor:
                    monitor.on_event("recv", env, recv_vc=recv_vc, recv_view=recv_view, recv_field_view=recv_field_view)
                if trace:
                    trace({
                        "type": "recv",
                        "to": A.name, "from": sender.name,
                        "channel": channel,
                        "bindings": _bound_dict(ys, values),
                        "seq": seq,
                        **_recv_trace_fields(monitor, recv_vc),
                    })

        case SelfAssignStmt(lifeline=A, payload=xs, bindings=ys):
            values = tuple(_eval(x, env) for x in xs)
            _bind(ys, values, env)
            if monitor:
                monitor.on_event("act", env)
            if trace:
                x_names = [x.var.name if isinstance(x, VarExpr) else f"_{i}" for i, x in enumerate(xs)]
                y_names = [y.var.name if isinstance(y, VarExpr) else f"_{i}" for i, y in enumerate(ys)]
                seq = _next_act_seq()
                trace({
                    "type": "act_start",
                    "lifeline": A.name,
                    "action": "assign",
                    "action_kind": "pure",
                    "inputs": {k: _jsonify(v) for k, v in zip(x_names, values)},
                    "seq": seq,
                })
                trace({
                    "type": "act",
                    "lifeline": A.name,
                    "action": "assign",
                    "action_kind": "pure",
                    "inputs": {k: _jsonify(v) for k, v in zip(x_names, values)},
                    "outputs": {k: _jsonify(v) for k, v in zip(y_names, values)},
                    "seq": seq,
                    **_monitor_trace_fields(monitor),
                })

        case ActStmt(lifeline=_, action=action, inputs=ins, outputs=outs):
            in_vals = tuple(_eval(x, env) for x in ins)
            named_inputs = {name: val for (name, _), val in zip(action.inputs, in_vals)}
            # For display, prefer the argument variable name over the formal parameter name.
            display_inputs = {
                (expr.var.name if isinstance(expr, VarExpr) else formal): val
                for (formal, _), expr, val in zip(action.inputs, ins, in_vals)
            }
            seq = _next_act_seq()
            _show = _action_visible(action)
            if trace and _show:
                trace({
                    "type": "act_start",
                    "lifeline": threading.current_thread().name,
                    "action": action.name,
                    "action_kind": _action_kind(action),
                    "inputs": {k: _jsonify(v) for k, v in display_inputs.items()},
                    "seq": seq,
                })
            out_map: dict[str, object]
            if isinstance(action, (PureAction, EffectAction)):
                out_map = _python_action_out_map(action, in_vals, outs)
            elif isinstance(action, PlannerAction):
                out_map = {
                    outs[0].name: _exec_planner(
                        action,
                        named_inputs,
                        llm_backend,
                        trace=trace,
                        parent_seq=seq,
                        stop=stop,
                    )
                }
            elif isinstance(action, HumanAction):
                if not action.visible:
                    default: object = (
                        True if action.output_type is bool else ""
                    )
                    out_map = {outs[0].name: default}
                else:
                    named_outputs = validate_human_action_result(
                        action, named_inputs, human_backend(action, named_inputs)
                    )
                    out_map = {outs[0].name: named_outputs[action.output]}
            elif isinstance(action, AssistantAction):
                named_outputs = assistant_backend(action, named_inputs)
                out_map = _assistant_action_out_map(action, named_outputs, outs)
            else:
                out_map = llm_out_map(
                    action, named_inputs, outs, llm_backend,
                    stop=stop, trace=trace,
                )
            if isinstance(action, HumanAction):
                if not action.visible:
                    out_map[outs[0].name] = validate_zvalue(
                        out_map[outs[0].name],
                        action.output_type,
                        context=(
                            f"Human backend for '{action.name}' output "
                            f"{action.output!r}"
                        ),
                    )
            else:
                out_map = _validate_action_out_map(
                    action,
                    out_map,
                    outs,
                    source=(
                        "Python action"
                        if isinstance(action, (PureAction, EffectAction))
                        else "Planner action"
                        if isinstance(action, PlannerAction)
                        else "Assistant backend"
                        if isinstance(action, AssistantAction)
                        else "LLM backend"
                    ),
                )
            env.update(out_map)
            if monitor:
                monitor.on_event("act", env)
            if trace and _show:
                trace({
                    "type": "act",
                    "lifeline": threading.current_thread().name,
                    "action": action.name,
                    "action_kind": _action_kind(action),
                    "inputs": {k: _jsonify(v) for k, v in display_inputs.items()},
                    "outputs": {k: _jsonify(v) for k, v in out_map.items()},
                    "seq": seq,
                    **_monitor_trace_fields(monitor),
                })

        case SeqStmt(first=p1, second=p2):
            _exec(cast(LocalStmt, p1), env, ch, ns, llm_backend, human_backend, assistant_backend, monitor, trace, formula_conditions, stop)
            _exec(cast(LocalStmt, p2), env, ch, ns, llm_backend, human_backend, assistant_backend, monitor, trace, formula_conditions, stop)

        case ParallelLocalStmt(branches=branches, branch_indices=branch_indices):
            residuals = list(branches)
            labels = branch_indices or tuple(range(len(branches)))
            cursor = 0

            while any(not isinstance(branch, EmptyStmt) for branch in residuals):
                if stop is not None and stop.is_set():
                    raise WorkflowCancelled(
                        "Workflow cancelled: another lifeline failed"
                    )

                progressed = False
                for _ in range(len(residuals)):
                    i = cursor % len(residuals)
                    cursor = (i + 1) % len(residuals)
                    branch = residuals[i]
                    if isinstance(branch, EmptyStmt):
                        continue
                    branch_trace = _with_parallel_branch(trace, f"P{labels[i] + 1}")
                    next_branch, did_step = _step(
                        branch, env, ch, ns, llm_backend, human_backend,
                        monitor, branch_trace, formula_conditions, stop,
                        assistant_backend=assistant_backend,
                    )
                    if isinstance(next_branch, PendingExternal):
                        raise RuntimeError("Unexpected pending external action in in-memory parallel execution.")
                    residuals[i] = cast(LocalStmt, next_branch)
                    if did_step:
                        progressed = True
                        break

                if not progressed:
                    time.sleep(0.01)

        case IfStmt(condition=c, owner=B, branch_true=t, branch_false=f):
            # c may be a Formula (direct) or a lambda (builder-rewritten native syntax).
            # Formula-valued lambdas are resolved once before execution when possible.
            cached_formula = formula_conditions.get(id(c))
            if cached_formula is not None:
                cond_formula = cached_formula
                cond_value = None
            elif isinstance(c, _Formula):
                cond_formula: _Formula | None = c
                cond_value = None
            else:
                raw = c(_CondEnv(env, ns))
                if isinstance(raw, _Formula):
                    cond_formula = raw
                    cond_value = None
                else:
                    cond_formula = None
                    cond_value = raw
            if cond_formula is not None and monitor is None:
                raise RuntimeError(
                    f"CPL Formula guard {cond_formula!r} on lifeline '{threading.current_thread().name}' "
                    "but no monitor was built. Make the Formula guard discoverable before execution."
                )
            if monitor:
                monitor.on_event("choice", env)
            if cond_formula is not None:
                assert monitor is not None
                flag = monitor.guard_value(cond_formula)
                formula_repr = repr(cond_formula)
            else:
                flag = bool(cond_value)
                formula_repr = None
            if trace:
                trace({"type": "decision", "lifeline": B.name, "kind": "if", "value": flag,
                       "condition": getattr(c, "_src", None), "formula": formula_repr,
                       **_monitor_trace_fields(monitor)})
            _exec(cast(LocalStmt, t if flag else f), env, ch, ns, llm_backend, human_backend, assistant_backend, monitor, trace, formula_conditions, stop)

        case IfRecvStmt(lifeline=A, bindings=ys, sender=B, branch_true=t, branch_false=f, channel=channel):
            seq, values, recv_vc, recv_view, recv_field_view = ch.get(B.name, A.name, channel, stop=stop)
            _bind(ys, values, env)
            if monitor:
                monitor.on_event("recv", env, recv_vc=recv_vc, recv_view=recv_view, recv_field_view=recv_field_view)
            flag = _eval(ys[0], env) if isinstance(ys[0], VarExpr) else values[0]
            if trace:
                trace({
                    "type": "recv",
                    "to": A.name, "from": B.name,
                    "channel": channel,
                    "bindings": {"branch": "true" if flag else "false"},
                    "seq": seq, "ctrl": True,
                    **_recv_trace_fields(monitor, recv_vc),
                })
            _exec(cast(LocalStmt, t if flag else f), env, ch, ns, llm_backend, human_backend, assistant_backend, monitor, trace, formula_conditions, stop)

        case WhileStmt(condition=c, owner=B, body=body, exit_body=exit_b):
            # Same Formula-dispatch logic as IfStmt — see comment there.
            while True:
                cached_formula = formula_conditions.get(id(c))
                if cached_formula is not None:
                    wc_formula = cached_formula
                    wc_value = None
                elif isinstance(c, _Formula):
                    wc_formula: _Formula | None = c
                    wc_value = None
                else:
                    wraw = c(_CondEnv(env, ns))
                    if isinstance(wraw, _Formula):
                        wc_formula = wraw
                        wc_value = None
                    else:
                        wc_formula = None
                        wc_value = wraw
                if wc_formula is not None and monitor is None:
                    raise RuntimeError(
                        f"CPL Formula guard {wc_formula!r} on lifeline '{threading.current_thread().name}' "
                        "but no monitor was built. Make the Formula guard discoverable before execution."
                    )
                if monitor:
                    monitor.on_event("choice", env)
                if wc_formula is not None:
                    assert monitor is not None
                    flag = monitor.guard_value(wc_formula)
                    formula_repr = repr(wc_formula)
                else:
                    flag = bool(wc_value)
                    formula_repr = None
                if trace:
                    trace({"type": "decision", "lifeline": B.name, "kind": "while", "value": flag,
                           "condition": getattr(c, "_src", None), "formula": formula_repr,
                           **_monitor_trace_fields(monitor)})
                if not flag:
                    break
                _exec(cast(LocalStmt, body), env, ch, ns, llm_backend, human_backend, assistant_backend, monitor, trace, formula_conditions, stop)
            _exec(cast(LocalStmt, exit_b), env, ch, ns, llm_backend, human_backend, assistant_backend, monitor, trace, formula_conditions, stop)

        case WhileRecvStmt(lifeline=A, bindings=ys, sender=B, body=body, exit_body=exit_b, channel=channel):
            while True:
                seq, values, recv_vc, recv_view, recv_field_view = ch.get(B.name, A.name, channel, stop=stop)
                _bind(ys, values, env)
                if monitor:
                    monitor.on_event("recv", env, recv_vc=recv_vc, recv_view=recv_view, recv_field_view=recv_field_view)
                flag = _eval(ys[0], env) if isinstance(ys[0], VarExpr) else values[0]
                if trace:
                    trace({
                        "type": "recv",
                        "to": A.name, "from": B.name,
                        "channel": channel,
                        "bindings": {"loop": "continue" if flag else "exit"},
                        "seq": seq, "ctrl": True,
                        **_recv_trace_fields(monitor, recv_vc),
                    })
                if not flag:
                    break
                _exec(cast(LocalStmt, body), env, ch, ns, llm_backend, human_backend, assistant_backend, monitor, trace, formula_conditions, stop)
            _exec(cast(LocalStmt, exit_b), env, ch, ns, llm_backend, human_backend, assistant_backend, monitor, trace, formula_conditions, stop)

        case _:
            raise TypeError(f"Unknown local stmt: {type(stmt).__name__}")


# ---------------------------------------------------------------------------
# Per-lifeline thread body
# ---------------------------------------------------------------------------

def _thread_body(local_stmt, env, ch, ns, result_box, llm_backend, human_backend, assistant_backend,
                 monitor, trace, formula_conditions, stop):
    try:
        _exec(local_stmt, env, ch, ns, llm_backend, human_backend, assistant_backend, monitor, trace, formula_conditions, stop)
        result_box.append(env)
    except Exception as exc:
        stop.set()  # unblock any threads waiting on queue.get()
        result_box.append(exc)


# ---------------------------------------------------------------------------
# Formula guard collection
# ---------------------------------------------------------------------------

def _condition_formula(condition, ns: dict) -> _Formula | None:
    if isinstance(condition, _Formula):
        return condition
    if not callable(condition):
        return None
    try:
        raw = condition(_FormulaProbeEnv({}, ns))
    except _ConditionValueUnavailable:
        # Ordinary workflow guards depend on values that do not exist until
        # execution. Formula guards resolve from the workflow namespace and can
        # be discovered now. Do not hide other exceptions: they indicate a
        # broken guard or formula-building expression.
        return None
    return raw if isinstance(raw, _Formula) else None


def _collect_formula_guards(stmt, ns: dict) -> tuple[list, dict[int, _Formula]]:
    guards: list = []
    condition_formulas: dict[int, _Formula] = {}
    # Walks the global program only; IfRecvStmt/WhileRecvStmt never appear in wf.body.
    def walk(s) -> None:
        match s:
            case IfStmt(condition=c, branch_true=t, branch_false=f):
                formula = _condition_formula(c, ns)
                if formula is not None:
                    guards.append(formula)
                    condition_formulas[id(c)] = formula
                walk(t)
                walk(f)
            case WhileStmt(condition=c, body=b, exit_body=x):
                formula = _condition_formula(c, ns)
                if formula is not None:
                    guards.append(formula)
                    condition_formulas[id(c)] = formula
                walk(b)
                walk(x)
            case SeqStmt(first=p1, second=p2):
                walk(p1)
                walk(p2)
            case ParallelStmt(branches=branches):
                for branch in branches:
                    walk(branch)
            case _:
                pass
    walk(stmt)
    return guards, condition_formulas


def _build_formula_monitors(
    wf: Workflow,
    lifelines: list[Lifeline] | tuple[Lifeline, ...],
) -> tuple[dict[str, MonitorState], dict[int, _Formula]]:
    formula_guards, formula_conditions = _collect_formula_guards(wf.body, wf.ns)
    if not formula_guards:
        return {}, formula_conditions

    all_subs: list = []
    seen_ids: set[int] = set()
    for guard in formula_guards:
        for subformula in _subformulas(guard):
            if id(subformula) not in seen_ids:
                seen_ids.add(id(subformula))
                all_subs.append(subformula)
    names = [lifeline.name for lifeline in lifelines]
    monitors: dict[str, MonitorState] = {
        lifeline.name: MonitorState(lifeline.name, names, all_subs)
        for lifeline in lifelines
    }
    return monitors, formula_conditions


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    wf: Workflow,
    lifelines: list[Lifeline],
    initial_envs: dict[str, dict[str, object]],
    *,
    llm_backend=None,
    human_backend=None,
    assistant_backend=None,
    verbose: bool = False,
    trace=None,
    timeout: float = 60.0,
) -> object:
    """
    Project ``wf`` onto every lifeline and run all of them concurrently.

    Parameters
    ----------
    wf            : global Workflow to execute
    lifelines     : ordered list of Lifeline objects to participate
    initial_envs  : mapping lifeline_name → {var_name: value}
    llm_backend   : optional callable(action, inputs_dict) → outputs_dict
                    Defaults to ``mock_llm``.
    human_backend : optional callable(action, inputs_dict) → outputs_dict
                    Defaults to ``make_cli_human_backend()``.
    verbose       : if True, print each event to stdout as it happens
    trace         : custom trace callable(event_dict) — overrides verbose
    timeout       : seconds to wait for each thread (default 60 s);
                    use 0 to run without a deadline

    Returns
    -------
    dict lifeline_name → final env dict
    Raises RuntimeError if any lifeline thread raised an exception.
    """
    if llm_backend is None:
        llm_backend = mock_llm

    if human_backend is None:
        from zippergen.human_backends import make_cli_human_backend
        human_backend = make_cli_human_backend()
    if assistant_backend is None:
        from zippergen.assistant_backends import make_cli_assistant_backend
        assistant_backend = make_cli_assistant_backend()

    if trace is None and verbose:
        trace = console_trace

    stop = threading.Event()
    human_dispatcher = None
    if getattr(human_backend, "requires_main_thread", False):
        from zippergen.human_backends import _MainThreadHumanDispatcher

        human_dispatcher = _MainThreadHumanDispatcher(human_backend, stop)
        human_backend = human_dispatcher.worker_backend

    names = [l.name for l in lifelines]
    channels = InProcessChannel()

    threads: list[threading.Thread] = []
    result_boxes: dict[str, list] = {}

    monitors, formula_conditions = _build_formula_monitors(wf, lifelines)

    for ll in lifelines:
        local_stmt = project(wf, ll)
        # Seed env with Var defaults so conditions see proper values before
        # any assignment has run, then override with caller-supplied values.
        env = {
            k: _clone_zvalue(v.default, v.type)
            for k, v in wf.ns.items()
            if isinstance(v, Var)
        }
        supplied = initial_envs.get(ll.name, {})
        env.update(supplied)
        for name, ztype, owner in wf.inputs:
            if (
                owner is not None
                and owner.name == ll.name
                and name in supplied
            ):
                env[name] = _clone_zvalue(
                    validate_zvalue(
                        supplied[name],
                        ztype,
                        context=f"{wf.name} input {name!r}",
                    ),
                    ztype,
                )
        box: list = []
        result_boxes[ll.name] = box

        def make_target(stmt, e, b, mon):
            def target():
                _thread_body(stmt, e, channels, wf.ns, b, llm_backend, human_backend, assistant_backend,
                             mon, trace, formula_conditions, stop)
            return target

        t = threading.Thread(
            target=make_target(local_stmt, env, box, monitors.get(ll.name)),
            name=ll.name,
            daemon=True,
        )
        threads.append(t)

    for t in threads:
        t.start()

    if human_dispatcher is not None:
        deadline = None if timeout <= 0 else time.monotonic() + timeout
        try:
            while any(t.is_alive() for t in threads):
                if stop.is_set():
                    break
                wait = 0.05
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        stop.set()
                        human_dispatcher.cancel_pending()
                        for t in threads:
                            t.join(timeout=1.0)
                        alive = next((t for t in threads if t.is_alive()), None)
                        if alive is not None:
                            raise TimeoutError(
                                f"Lifeline '{alive.name}' did not finish within "
                                f"{timeout}s"
                            )
                        raise TimeoutError(
                            f"Workflow did not finish within {timeout}s"
                        )
                    wait = min(wait, remaining)
                human_dispatcher.service_next(timeout=wait)
        except BaseException:
            stop.set()
            human_dispatcher.cancel_pending()
            for t in threads:
                t.join(timeout=1.0)
            raise
        for t in threads:
            if t.is_alive():
                t.join(timeout=1.0)
    elif timeout <= 0:
        try:
            while any(t.is_alive() for t in threads):
                if stop.is_set():
                    break
                time.sleep(0.05)
        except KeyboardInterrupt:
            stop.set()
            for t in threads:
                t.join(timeout=1.0)
            raise
        for t in threads:
            if t.is_alive():
                t.join(timeout=1.0)
    else:
        deadline = time.monotonic() + timeout
        for t in threads:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                stop.set()
                raise TimeoutError(f"Workflow did not finish within {timeout}s")
            t.join(timeout=remaining)
            if t.is_alive():
                stop.set()
                raise TimeoutError(f"Lifeline '{t.name}' did not finish within {timeout}s")

    # Collect all exceptions, preferring root-cause errors over secondary
    # "Workflow cancelled" errors that are triggered by the stop event.
    root_cause: tuple[str, Exception] | None = None
    cancelled: tuple[str, Exception] | None = None
    final_envs: dict[str, dict] = {}
    missing: list[str] = []
    for ll in lifelines:
        box = result_boxes[ll.name]
        if not box:
            missing.append(ll.name)
            continue
        result = box[0]
        if isinstance(result, Exception):
            if isinstance(result, WorkflowCancelled):
                if cancelled is None:
                    cancelled = (ll.name, result)
            else:
                if root_cause is None:
                    root_cause = (ll.name, result)
        else:
            final_envs[ll.name] = result

    error = root_cause or cancelled
    if error is not None:
        name, exc = error
        raise RuntimeError(f"Lifeline '{name}' raised: {exc}") from exc
    if missing:
        names = ", ".join(repr(name) for name in missing)
        raise RuntimeError(f"Lifeline(s) produced no result: {names}.")

    if len(wf.outputs) == 0:
        return final_envs
    if len(wf.outputs) == 1:
        var, ll = wf.outputs[0]
        return final_envs[ll.name][var.name]
    return tuple(final_envs[ll.name][var.name] for var, ll in wf.outputs)


# ---------------------------------------------------------------------------
# Workflow execution helpers — called from Workflow methods via lazy import
# ---------------------------------------------------------------------------

def _workflow_configure(
    wf: Workflow,
    llm=None,
    *,
    backend: object = None,
    trace: object = None,
    timeout: float = 60.0,
    mock_delay: tuple[float, float] = (1.0, 2.0),
    llm_idle_timeout: float | None = None,
    llm_idle_timeouts: Mapping[str, float] | None = None,
    execution: str | None = None,
    store_path: str | None = None,
    human_backend: object | None = None,
    assistant: str | object | None = None,
    assistant_backend: object | None = None,
    assistant_root: str | None = None,
) -> Workflow:
    if llm_idle_timeout is not None and (
        not math.isfinite(llm_idle_timeout) or llm_idle_timeout < 0
    ):
        raise ValueError("llm_idle_timeout must be non-negative.")
    normalized_idle_timeouts = {
        str(target): float(value)
        for target, value in (llm_idle_timeouts or {}).items()
    }
    if any(
        not math.isfinite(value) or value < 0
        for value in normalized_idle_timeouts.values()
    ):
        raise ValueError("llm_idle_timeouts values must be non-negative.")
    if callable(llm):
        if backend is not None:
            raise ValueError("Use either positional backend/llm or 'backend=', not both.")
        backend = llm
        llm = None
    if execution is not None:
        if execution not in {"memory", "sqlite"}:
            raise ValueError("execution must be 'memory' or 'sqlite'")
        wf._rt._execution = execution
    if store_path is not None:
        wf._rt._store_path = store_path

    if llm is not None:
        from zippergen.backends import router_from_specs
        from zippergen.models import effective_llm_routes
        if llm == "mock":
            routes: dict = {}
        elif isinstance(llm, str):
            routes = effective_llm_routes(wf, llm)
        else:
            routes = {str(k): v for k, v in llm.items()}
        built_backend, _label = router_from_specs(
            routes,
            fallback=lambda a, i: mock_llm(a, i, min_delay=mock_delay[0], max_delay=mock_delay[1]),
            idle_timeout=llm_idle_timeout,
            idle_timeouts=normalized_idle_timeouts,
        )
        wf._rt._backend = built_backend
    if backend is not None:
        wf._rt._backend = backend
    if assistant is not None and assistant_backend is not None:
        raise ValueError(
            "Use either 'assistant=' or 'assistant_backend=', not both."
        )
    if callable(assistant):
        assistant_backend = assistant
        assistant = None
    if assistant is not None:
        from zippergen.assistant_backends import make_cli_assistant_backend
        assistant_backend = make_cli_assistant_backend(
            str(assistant),
            project_root=assistant_root,
        )
    elif assistant_backend is None and assistant_root is not None:
        from zippergen.assistant_backends import make_cli_assistant_backend
        assistant_backend = make_cli_assistant_backend(project_root=assistant_root)
    if assistant_backend is not None:
        wf._rt._assistant_backend = assistant_backend

    if trace is not None:
        wf._rt._trace = trace

    # An explicit backend is used by project-aware development surfaces that
    # keep SQLite durability while presenting human tasks in the terminal.
    if human_backend is not None:
        wf._rt._human_backend = human_backend
    elif wf._rt._execution == "sqlite":
        from zippergen.human_backends import make_sqlite_human_backend
        wf._rt._human_backend = make_sqlite_human_backend()
    else:
        from zippergen.human_backends import make_cli_human_backend
        wf._rt._human_backend = make_cli_human_backend()

    wf._rt._timeout = timeout
    return wf


def _workflow_run_once(wf: Workflow, kwargs: dict[str, object]) -> object:
    initial_envs: dict[str, dict[str, object]] = {}
    for name, ztype, lifeline in wf.inputs:
        if lifeline is None:
            raise TypeError(
                f"{wf.name}(): input '{name}' has no lifeline declared. "
                f"Use 'name: type @ Lifeline' in the @workflow signature."
            )
        if name not in kwargs:
            raise TypeError(f"{wf.name}() missing argument: '{name}'")
        value = validate_zvalue(
            kwargs[name],
            ztype,
            context=f"{wf.name}() input {name!r}",
        )
        initial_envs.setdefault(lifeline.name, {})[name] = _clone_zvalue(
            value,
            ztype,
        )

    lifelines = _ordered_workflow_lifelines(wf)
    backend = wf._rt._backend if wf._rt._backend is not None else mock_llm
    with wf._rt._run_lock:
        trace = wf._rt._trace
        human_backend = wf._rt._human_backend
        assistant_backend = wf._rt._assistant_backend
        if assistant_backend is None:
            from zippergen.assistant_backends import make_cli_assistant_backend
            assistant_backend = make_cli_assistant_backend()
        if wf._rt._execution == "sqlite":
            from zippergen.sqlite_runner import run_sqlite
            return run_sqlite(
                wf,
                list(lifelines),
                initial_envs,
                store_path=wf._rt._store_path,
                llm_backend=backend,
                human_backend=human_backend,
                assistant_backend=assistant_backend,
                trace=trace,
                timeout=wf._rt._timeout,
            )
        return run(
            wf,
            list(lifelines),
            initial_envs,
            llm_backend=backend,
            human_backend=human_backend,
            assistant_backend=assistant_backend,
            trace=trace,
            timeout=wf._rt._timeout,
        )


def _workflow_call(wf: Workflow, kwargs: dict[str, object]) -> object:
    return _workflow_run_once(wf, dict(kwargs))
