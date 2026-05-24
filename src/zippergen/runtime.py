"""
Layer 5: Runtime executor. Projects the workflow onto each lifeline, runs one
thread per lifeline, wires FIFO queues, and drives execution to completion.
"""

from __future__ import annotations

import copy
import queue
import threading
from collections import defaultdict
from typing import cast
import time
import textwrap

from zippergen.planner import _exec_planner, _validate_planner_spec

from zippergen.syntax import (
    EmptyStmt, SendStmt, RecvStmt, ReceiveAnyStmt, SelfAssignStmt, ActStmt, SkipStmt,
    SeqStmt, IfStmt, WhileStmt, IfRecvStmt, WhileRecvStmt,
    ParallelStmt, ParallelLocalStmt,
    VarExpr, LitExpr, Var,
    LLMAction, PureAction, PlannerAction, HumanAction,
    Lifeline, Workflow, LocalStmt, AnyStmt,
    is_kappa_ctrl,
    _ordered_workflow_lifelines,
    seq,
)
from zippergen.projection import project
from zippergen.formula import Formula as _Formula, subformulas as _subformulas
from zippergen.monitor import MonitorState

__all__ = ["run", "mock_llm", "console_trace", "tee_traces"]


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
# Seq-stamped queue — each message carries a monotone sequence number
# so send/recv events can be paired in the web viewer
# ---------------------------------------------------------------------------

class _SeqQueue:
    """FIFO queue that auto-stamps each item with a per-channel sequence number.

    Items are stored as (seq, values, vc, view, field_view). vc, view, and
    field_view are the sender's monitor snapshot, or None when monitoring is
    inactive.
    """

    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self._next = 0

    def put(
        self,
        values: tuple,
        vc: dict | None = None,
        view: dict | None = None,
        field_view: dict | None = None,
    ) -> int:
        seq = self._next
        self._next += 1
        self._q.put((seq, values, vc, view, field_view))
        return seq

    def get(
        self, *, stop: threading.Event | None = None
    ) -> tuple[int, tuple, dict | None, dict | None, dict | None]:
        if stop is None:
            return self._q.get()
        while True:
            try:
                return self._q.get(timeout=0.5)
            except queue.Empty:
                if stop.is_set():
                    raise RuntimeError("Workflow cancelled: another lifeline failed")

    def get_nowait(self) -> tuple[int, tuple, dict | None, dict | None, dict | None]:
        return self._q.get_nowait()


Channels = defaultdict[tuple[str, str, str], _SeqQueue]


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
    if isinstance(action, PlannerAction):
        return "planner"
    if isinstance(action, HumanAction):
        return "human"
    if isinstance(action, LLMAction):
        return "llm"
    return "act"


def _receive_any(
    ch: Channels,
    receiver: str,
    pending_senders: set[str],
    channel: str,
    *,
    stop: threading.Event | None = None,
) -> tuple[str, tuple[int, tuple, dict | None, dict | None, dict | None]]:
    while True:
        for sender in sorted(pending_senders):
            try:
                return sender, ch[(sender, receiver, channel)].get_nowait()
            except queue.Empty:
                pass
        if stop is not None and stop.is_set():
            raise RuntimeError("Workflow cancelled: another lifeline failed")
        time.sleep(0.01)



# ---------------------------------------------------------------------------
# Local-program interpreter
# ---------------------------------------------------------------------------

def _try_channel_get(
    ch: Channels,
    sender: str,
    receiver: str,
    channel: str,
) -> tuple[int, tuple, dict | None, dict | None, dict | None] | None:
    try:
        return ch[(sender, receiver, channel)].get_nowait()
    except queue.Empty:
        return None


def _with_parallel_branch(trace, label: str):
    if trace is None:
        return None

    def wrapped(event: dict) -> None:
        if "parallel_branch" not in event:
            event = {**event, "parallel_branch": label}
        trace(event)

    return wrapped


def _step(
    stmt: LocalStmt,
    env: Env,
    ch: Channels,
    ns: dict,
    llm_backend,
    human_backend,
    monitor,
    trace,
    formula_conditions: dict[int, _Formula],
    stop: threading.Event | None,
) -> tuple[LocalStmt, bool]:
    """Execute at most one enabled local step.

    Returns ``(residual, progressed)``. Blocking receives return the original
    residual with ``progressed=False`` so the local parallel scheduler can try
    another branch.
    """
    match stmt:
        case EmptyStmt():
            return EmptyStmt(), False

        case SkipStmt():
            return EmptyStmt(), True

        case SendStmt() | SelfAssignStmt() | ActStmt():
            _exec(stmt, env, ch, ns, llm_backend, human_backend, monitor, trace, formula_conditions, stop)
            return EmptyStmt(), True

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

        case ReceiveAnyStmt(lifeline=A, receives=receives, channel=channel):
            for sender, ys in receives:
                item = _try_channel_get(ch, sender.name, A.name, channel)
                if item is None:
                    continue
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
                remaining = tuple((s, b) for s, b in receives if s != sender)
                if not remaining:
                    return EmptyStmt(), True
                return ReceiveAnyStmt(A, remaining, channel), True
            return stmt, False

        case SeqStmt(first=p1, second=p2):
            first = cast(LocalStmt, p1)
            second = cast(LocalStmt, p2)
            if isinstance(first, EmptyStmt):
                return second, True
            new_first, progressed = _step(
                first, env, ch, ns, llm_backend, human_backend, monitor, trace,
                formula_conditions, stop,
            )
            if not progressed:
                return stmt, False
            return cast(LocalStmt, seq(new_first, second)), True

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

        case ParallelLocalStmt():
            _exec(stmt, env, ch, ns, llm_backend, human_backend, monitor, trace, formula_conditions, stop)
            return EmptyStmt(), True

        case _:
            raise TypeError(f"Unknown local stmt: {type(stmt).__name__}")


def _exec(stmt: LocalStmt, env: Env, ch: Channels, ns: dict, llm_backend, human_backend, monitor, trace,
          formula_conditions: dict[int, _Formula] | None = None,
          stop: threading.Event | None = None) -> None:
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
                seq = ch[(A.name, B.name, channel)].put(values, monitor.snapshot_vc(), monitor.snapshot_view(), monitor.snapshot_field_view())
            else:
                seq = ch[(A.name, B.name, channel)].put(values)
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
            seq, values, recv_vc, recv_view, recv_field_view = ch[(B.name, A.name, channel)].get(stop=stop)
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

            if not hasattr(action, 'inputs'):
                raise RuntimeError(
                    f"Action lookup failed: expected an action object but got "
                    f"{type(action).__name__} '{getattr(action, 'name', repr(action))}'. "
                    f"An output variable likely has the same name as an action — rename it."
                )
            named_inputs = {name: val for (name, _), val in zip(action.inputs, in_vals)}
            # For display, prefer the argument variable name over the formal parameter name.
            display_inputs = {
                (expr.var.name if isinstance(expr, VarExpr) else formal): val
                for (formal, _), expr, val in zip(action.inputs, ins, in_vals)
            }
            seq = _next_act_seq()
            if trace:
                trace({
                    "type": "act_start",
                    "lifeline": threading.current_thread().name,
                    "action": action.name,
                    "action_kind": _action_kind(action),
                    "inputs": {k: _jsonify(v) for k, v in display_inputs.items()},
                    "seq": seq,
                })
            if isinstance(action, PureAction):
                raw = action.fn(*in_vals)
                out_map = {outs[0].name: raw} if len(outs) == 1 else {
                    var.name: val for var, val in zip(outs, cast(tuple, raw))
                }
            elif isinstance(action, PlannerAction):
                out_map = {outs[0].name: _exec_planner(action, named_inputs, llm_backend, trace, seq)}
            elif isinstance(action, HumanAction):
                named_outputs = human_backend(action, named_inputs)
                out_map = {outs[0].name: named_outputs[action.output]}
            else:
                named_outputs = llm_backend(action, named_inputs)
                out_map = {
                    var.name: named_outputs.get(aname)
                    for (aname, _), var in zip(action.outputs, outs)
                }
            env.update(out_map)
            if monitor:
                monitor.on_event("act", env)
            if trace:
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
            _exec(cast(LocalStmt, p1), env, ch, ns, llm_backend, human_backend, monitor, trace, formula_conditions, stop)
            _exec(cast(LocalStmt, p2), env, ch, ns, llm_backend, human_backend, monitor, trace, formula_conditions, stop)

        case ParallelLocalStmt(branches=branches, branch_indices=branch_indices):
            residuals = list(branches)
            labels = branch_indices or tuple(range(len(branches)))
            cursor = 0

            while any(not isinstance(branch, EmptyStmt) for branch in residuals):
                if stop is not None and stop.is_set():
                    raise RuntimeError("Workflow cancelled: another lifeline failed")

                progressed = False
                for _ in range(len(residuals)):
                    i = cursor % len(residuals)
                    cursor = (i + 1) % len(residuals)
                    branch = residuals[i]
                    if isinstance(branch, EmptyStmt):
                        continue
                    branch_trace = _with_parallel_branch(trace, f"P{labels[i] + 1}")
                    next_branch, did_step = _step(
                        branch, env, ch, ns, llm_backend, human_backend, monitor,
                        branch_trace, formula_conditions, stop,
                    )
                    residuals[i] = next_branch
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
            _exec(cast(LocalStmt, t if flag else f), env, ch, ns, llm_backend, human_backend, monitor, trace, formula_conditions, stop)

        case IfRecvStmt(lifeline=A, bindings=ys, sender=B, branch_true=t, branch_false=f, channel=channel):
            seq, values, recv_vc, recv_view, recv_field_view = ch[(B.name, A.name, channel)].get(stop=stop)
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
            _exec(cast(LocalStmt, t if flag else f), env, ch, ns, llm_backend, human_backend, monitor, trace, formula_conditions, stop)

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
                _exec(cast(LocalStmt, body), env, ch, ns, llm_backend, human_backend, monitor, trace, formula_conditions, stop)
            _exec(cast(LocalStmt, exit_b), env, ch, ns, llm_backend, human_backend, monitor, trace, formula_conditions, stop)

        case WhileRecvStmt(lifeline=A, bindings=ys, sender=B, body=body, exit_body=exit_b, channel=channel):
            while True:
                seq, values, recv_vc, recv_view, recv_field_view = ch[(B.name, A.name, channel)].get(stop=stop)
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
                _exec(cast(LocalStmt, body), env, ch, ns, llm_backend, human_backend, monitor, trace, formula_conditions, stop)
            _exec(cast(LocalStmt, exit_b), env, ch, ns, llm_backend, human_backend, monitor, trace, formula_conditions, stop)

        case _:
            raise TypeError(f"Unknown local stmt: {type(stmt).__name__}")


# ---------------------------------------------------------------------------
# Per-lifeline thread body
# ---------------------------------------------------------------------------

def _thread_body(local_stmt, env, ch, ns, result_box, llm_backend, human_backend,
                 monitor, trace, formula_conditions, stop):
    try:
        _exec(local_stmt, env, ch, ns, llm_backend, human_backend, monitor, trace, formula_conditions, stop)
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
        raw = condition(_CondEnv({}, ns))
    except Exception:
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
    timeout       : seconds to wait for each thread (default 60 s)

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

    if trace is None and verbose:
        trace = console_trace

    stop = threading.Event()

    names = [l.name for l in lifelines]
    channels: Channels = defaultdict(_SeqQueue)

    threads: list[threading.Thread] = []
    result_boxes: dict[str, list] = {}

    formula_guards, formula_conditions = _collect_formula_guards(wf.body, wf.ns)
    if formula_guards:
        all_subs: list = []
        seen_ids: set[int] = set()
        for g in formula_guards:
            for sf in _subformulas(g):
                if id(sf) not in seen_ids:
                    seen_ids.add(id(sf))
                    all_subs.append(sf)
        monitors: dict[str, MonitorState] = {
            ll.name: MonitorState(ll.name, [l.name for l in lifelines], all_subs)
            for ll in lifelines
        }
    else:
        monitors = {}

    for ll in lifelines:
        local_stmt = project(wf, ll)
        # Seed env with Var defaults so conditions see proper values before
        # any assignment has run, then override with caller-supplied values.
        env = {k: v.default for k, v in wf.ns.items() if isinstance(v, Var)}
        env.update(initial_envs.get(ll.name, {}))
        box: list = []
        result_boxes[ll.name] = box

        def make_target(stmt, e, b, mon):
            def target():
                _thread_body(stmt, e, channels, wf.ns, b, llm_backend, human_backend,
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
    for ll in lifelines:
        box = result_boxes[ll.name]
        if not box:
            raise RuntimeError(f"Lifeline '{ll.name}' produced no result.")
        result = box[0]
        if isinstance(result, Exception):
            msg = str(result)
            if "Workflow cancelled" in msg:
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
    wf: Workflow, *,
    backend: object = None,
    trace: object = None,
    timeout: float = 60.0,
    llms=None,
    ui: bool | None = None,
    mock_delay: tuple[float, float] = (1.0, 2.0),
) -> Workflow:
    lifelines = _ordered_workflow_lifelines(wf)

    if llms is not None:
        from zippergen.backends import router_from_env
        if llms == "mock":
            routes: dict = {}
        elif isinstance(llms, str):
            routes = {lifeline.name: llms for lifeline in lifelines}
        else:
            routes = {str(k): v for k, v in llms.items()}
        built_backend, _label = router_from_env(
            routes,
            fallback=lambda a, i: mock_llm(a, i, min_delay=mock_delay[0], max_delay=mock_delay[1]),
        )
        wf._rt._backend = built_backend
    if backend is not None:
        wf._rt._backend = backend

    if ui is not None:
        wf._rt._ui_enabled = ui
    if wf._rt._ui_enabled:
        from zipperchat import WebTrace
        if isinstance(trace, WebTrace):
            wf._rt._webtrace = trace.start()
            wf._rt._trace = console_trace
        else:
            if wf._rt._webtrace is None:
                wf._rt._webtrace = WebTrace(lifelines, name=wf.name).start()
            base_trace = trace if trace is not None else console_trace
            wf._rt._trace = tee_traces(wf._rt._webtrace, base_trace)
    elif trace is not None:
        wf._rt._trace = trace

    # Human backend: web if UI is enabled, CLI otherwise.
    if (
        wf._rt._ui_enabled
        and wf._rt._webtrace is not None
        and not wf._rt._webtrace.is_dashboard
    ):
        wf._rt._human_backend = wf._rt._webtrace.make_human_backend()
    else:
        from zippergen.human_backends import make_cli_human_backend
        wf._rt._human_backend = make_cli_human_backend()

    wf._rt._timeout = timeout
    return wf


def _workflow_run_once(wf: Workflow, kwargs: dict[str, object]) -> object:
    initial_envs: dict[str, dict[str, object]] = {}
    for name, _ztype, lifeline in wf.inputs:
        if lifeline is None:
            raise TypeError(
                f"{wf.name}(): input '{name}' has no lifeline declared. "
                f"Use 'name: type @ Lifeline' in the @workflow signature."
            )
        if name not in kwargs:
            raise TypeError(f"{wf.name}() missing argument: '{name}'")
        initial_envs.setdefault(lifeline.name, {})[name] = kwargs[name]

    lifelines = _ordered_workflow_lifelines(wf)
    backend = wf._rt._backend if wf._rt._backend is not None else mock_llm
    with wf._rt._run_lock:
        run_trace = None
        trace = wf._rt._trace
        human_backend = wf._rt._human_backend
        if wf._rt._webtrace is not None and wf._rt._ui_enabled:
            if wf._rt._webtrace.is_dashboard:
                run_trace = wf._rt._webtrace.start_run(wf.name, lifelines)
                trace = tee_traces(run_trace, wf._rt._trace)
                human_backend = run_trace.make_human_backend()
            else:
                wf._rt._webtrace.reset()
        try:
            return run(wf, list(lifelines), initial_envs,
                       llm_backend=backend,
                       human_backend=human_backend,
                       trace=trace, timeout=wf._rt._timeout)
        finally:
            if wf._rt._webtrace is not None and wf._rt._ui_enabled:
                if run_trace is not None:
                    run_trace.done()
                else:
                    wf._rt._webtrace.done()


def _workflow_ensure_replay_loop(wf: Workflow) -> None:
    if not wf._rt._ui_enabled or wf._rt._webtrace is None or wf._rt._replay_thread is not None:
        return
    if wf._rt._webtrace.is_dashboard:
        return

    def _worker() -> None:
        assert wf._rt._webtrace is not None
        while True:
            wf._rt._webtrace.wait_for_replay()
            if not wf._rt._last_kwargs:
                continue
            try:
                result = _workflow_run_once(wf, dict(wf._rt._last_kwargs))
                print(f"\nResult → {result}")
            except Exception as exc:
                print(f"\nReplay failed: {exc}")

    wf._rt._replay_thread = threading.Thread(target=_worker, daemon=True)
    wf._rt._replay_thread.start()


def _workflow_call(wf: Workflow, kwargs: dict[str, object]) -> object:
    wf._rt._last_kwargs = dict(kwargs)
    _workflow_ensure_replay_loop(wf)
    return _workflow_run_once(wf, dict(kwargs))
