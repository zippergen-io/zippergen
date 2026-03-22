"""
ZipperGen — Layer 5: Runtime executor.

Workflow
--------
1. Call ``run(wf, lifelines, initial_envs)`` with a global Workflow, the list of
   lifelines to participate, and the initial variable bindings for each.
2. The runtime projects the workflow onto every lifeline, creates one thread per
   lifeline, wires FIFO queues between them, and runs everything to completion.
3. When all threads finish, ``run()`` returns the final env of every lifeline.

LLM actions
-----------
Pass ``llm_backend`` — a callable with signature::

    def backend(action: LLMAction, inputs: dict[str, object]) -> dict[str, object]:
        ...

If omitted, a simple random mock is used.

Tracing
-------
The ``trace`` parameter accepts a callable ``(event: dict) -> None``.
Pass ``verbose=True`` for built-in console printing, or pass a ``WebTrace``
instance from ``zippergen.web`` for browser-based real-time visualisation.

Each event dict has a ``"type"`` key with values:
  ``"send"``  — a lifeline put a message on a channel
  ``"recv"``  — a lifeline received a message from a channel
  ``"act"``   — a lifeline executed an action
"""

from __future__ import annotations

import queue
import inspect
import threading
import time
import textwrap
from contextlib import nullcontext

from zippergen.syntax import (
    EmptyStmt, SendStmt, RecvStmt, ActStmt, SkipStmt,
    SeqStmt, IfStmt, WhileStmt, IfRecvStmt, WhileRecvStmt,
    VarExpr, LitExpr, Var,
    LLMAction, PureAction,
    Lifeline, Workflow, LocalStmt,
    kappa_ctrl,
)
from zippergen.projection import project

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
        if value == "κ_ctrl":
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
        is_ctrl = "κ_ctrl" in (event.get("values") or [])
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

    if not lines:
        return

    with _print_lock:
        print("\n".join(lines))
        if t in {"act_start", "act"}:
            print()


def _default_trace(event: dict) -> None:
    console_trace(event)


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
    """FIFO queue that auto-stamps each item with a per-channel sequence number."""

    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self._next = 0

    def put(self, values: tuple) -> int:
        seq = self._next
        self._next += 1
        self._q.put((seq, values))
        return seq

    def get(self) -> tuple[int, tuple]:
        return self._q.get()


Channels = dict[tuple[str, str], _SeqQueue]


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


def _is_kappa(expr) -> bool:
    return expr == kappa_ctrl


def _bind(bindings: tuple, values: tuple, env: Env) -> None:
    for binding, value in zip(bindings, values):
        if _is_kappa(binding):
            continue
        if isinstance(binding, VarExpr):
            env[binding.var.name] = value


def _jsonify(value: object) -> object:
    """Convert a Python runtime value to a JSON-safe object."""
    if isinstance(value, (bool, int, float, str, type(None))):
        return value
    if isinstance(value, tuple):
        return [_jsonify(v) for v in value]
    return str(value)


def _fmt_vals(values) -> str:
    """Short string for console trace."""
    if isinstance(values, list):
        return str(tuple(values))
    return str(values)


def _bound_dict(bindings: tuple, values: tuple) -> dict:
    """Build a {var_name: value} dict from a binding/value pair, skipping kappa."""
    return {
        b.var.name: _jsonify(v)
        for b, v in zip(bindings, values)
        if isinstance(b, VarExpr) and not _is_kappa(b)
    }


def _call_llm_backend(llm_backend, action, inputs: dict[str, object], lifeline_name: str) -> dict[str, object]:
    accepts_lifeline = getattr(llm_backend, "_zippergen_accepts_lifeline", None)
    if accepts_lifeline is None:
        try:
            accepts_lifeline = len(inspect.signature(llm_backend).parameters) >= 3
        except (TypeError, ValueError):
            accepts_lifeline = False
        try:
            setattr(llm_backend, "_zippergen_accepts_lifeline", accepts_lifeline)
        except (AttributeError, TypeError):
            pass
    if accepts_lifeline:
        return llm_backend(action, inputs, lifeline_name)
    return llm_backend(action, inputs)


def _backend_lock(llm_backend, lifeline_name: str):
    lock_for = getattr(llm_backend, "_zippergen_lock_for", None)
    if callable(lock_for):
        return lock_for(lifeline_name)
    return getattr(llm_backend, "_zippergen_lock", None)


# ---------------------------------------------------------------------------
# Local-program interpreter
# ---------------------------------------------------------------------------

def _exec(stmt: LocalStmt, env: Env, ch: Channels, ns: dict, llm_backend, trace) -> None:
    """Execute a LocalStmt, updating env in place."""
    match stmt:

        case EmptyStmt() | SkipStmt():
            return

        case SendStmt(lifeline=A, payload=xs, receiver=B):
            values = tuple(_eval(x, env) for x in xs)
            seq = ch[(A.name, B.name)].put(values)
            if trace:
                names = [x.var.name if isinstance(x, VarExpr) else f"_{i}" for i, x in enumerate(xs)]
                trace({
                    "type": "send",
                    "from": A.name, "to": B.name,
                    "values": [_jsonify(v) for v in values],
                    "bindings": {name: _jsonify(v) for name, v in zip(names, values)},
                    "seq": seq,
                })

        case RecvStmt(lifeline=A, bindings=ys, sender=B):
            seq, values = ch[(B.name, A.name)].get()
            _bind(ys, values, env)
            if trace:
                trace({
                    "type": "recv",
                    "to": A.name, "from": B.name,
                    "bindings": _bound_dict(ys, values),
                    "seq": seq,
                })

        case ActStmt(lifeline=_, action=action, inputs=ins, outputs=outs):
            in_vals = tuple(_eval(x, env) for x in ins)
            named_inputs = {name: val for (name, _), val in zip(action.inputs, in_vals)}
            seq = _next_act_seq()
            lock = _backend_lock(llm_backend, threading.current_thread().name) if not isinstance(action, PureAction) else None
            if lock is None:
                lock_ctx = nullcontext()
            else:
                lock_ctx = lock
            with lock_ctx:
                if trace:
                    trace({
                        "type": "act_start",
                        "lifeline": threading.current_thread().name,
                        "action": action.name,
                        "inputs": {k: _jsonify(v) for k, v in named_inputs.items()},
                        "seq": seq,
                    })
                if isinstance(action, PureAction):
                    raw = action.fn(*in_vals)
                    if len(outs) == 1:
                        out_map = {outs[0].name: raw}
                    else:
                        out_map = {var.name: val for var, val in zip(outs, raw)}
                else:
                    named_outputs = _call_llm_backend(
                        llm_backend,
                        action,
                        named_inputs,
                        threading.current_thread().name,
                    )
                    out_map = {
                        var.name: named_outputs.get(aname)
                        for (aname, _), var in zip(action.outputs, outs)
                    }
            env.update(out_map)
            if trace:
                trace({
                    "type": "act",
                    "lifeline": threading.current_thread().name,
                    "action": action.name,
                    "inputs": {k: _jsonify(v) for k, v in named_inputs.items()},
                    "outputs": {k: _jsonify(v) for k, v in out_map.items()},
                    "seq": seq,
                })

        case SeqStmt(first=p1, second=p2):
            _exec(p1, env, ch, ns, llm_backend, trace)
            _exec(p2, env, ch, ns, llm_backend, trace)

        case IfStmt(condition=c, owner=_, branch_true=t, branch_false=f):
            _exec(t if c(_CondEnv(env, ns)) else f, env, ch, ns, llm_backend, trace)

        case IfRecvStmt(lifeline=A, bindings=ys, sender=B, branch_true=t, branch_false=f):
            seq, values = ch[(B.name, A.name)].get()
            _bind(ys, values, env)
            flag = _eval(ys[0], env) if isinstance(ys[0], VarExpr) else values[0]
            if trace:
                trace({
                    "type": "recv",
                    "to": A.name, "from": B.name,
                    "bindings": {"branch": "true" if flag else "false"},
                    "seq": seq, "ctrl": True,
                })
            _exec(t if flag else f, env, ch, ns, llm_backend, trace)

        case WhileStmt(condition=c, owner=_, body=body, exit_body=exit_b):
            while c(_CondEnv(env, ns)):
                _exec(body, env, ch, ns, llm_backend, trace)
            _exec(exit_b, env, ch, ns, llm_backend, trace)

        case WhileRecvStmt(lifeline=A, bindings=ys, sender=B, body=body, exit_body=exit_b):
            while True:
                seq, values = ch[(B.name, A.name)].get()
                _bind(ys, values, env)
                flag = _eval(ys[0], env) if isinstance(ys[0], VarExpr) else values[0]
                if trace:
                    trace({
                        "type": "recv",
                        "to": A.name, "from": B.name,
                        "bindings": {"loop": "continue" if flag else "exit"},
                        "seq": seq, "ctrl": True,
                    })
                if not flag:
                    break
                _exec(body, env, ch, ns, llm_backend, trace)
            _exec(exit_b, env, ch, ns, llm_backend, trace)

        case _:
            raise TypeError(f"Unknown local stmt: {type(stmt).__name__}")


# ---------------------------------------------------------------------------
# Per-lifeline thread body
# ---------------------------------------------------------------------------

def _thread_body(local_stmt, env, ch, ns, result_box, llm_backend, trace):
    try:
        _exec(local_stmt, env, ch, ns, llm_backend, trace)
        result_box.append(env)
    except Exception as exc:
        result_box.append(exc)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    wf: Workflow,
    lifelines: list[Lifeline],
    initial_envs: dict[str, dict[str, object]],
    *,
    llm_backend=None,
    verbose: bool = False,
    trace=None,
    timeout: float = 60.0,
) -> dict[str, dict[str, object]]:
    """
    Project ``wf`` onto every lifeline and run all of them concurrently.

    Parameters
    ----------
    wf            : global Workflow to execute
    lifelines     : ordered list of Lifeline objects to participate
    initial_envs  : mapping lifeline_name → {var_name: value}
    llm_backend   : optional callable(action, inputs_dict) → outputs_dict
                    Defaults to ``mock_llm``.
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

    if trace is None and verbose:
        trace = _default_trace

    names = [l.name for l in lifelines]
    channels: Channels = {
        (a, b): _SeqQueue()
        for a in names for b in names if a != b
    }

    threads: list[threading.Thread] = []
    result_boxes: dict[str, list] = {}

    for ll in lifelines:
        local_stmt = project(wf, ll)
        # Seed env with Var defaults so conditions see proper values before
        # any assignment has run, then override with caller-supplied values.
        env = {k: v.default for k, v in wf.ns.items() if isinstance(v, Var)}
        env.update(initial_envs.get(ll.name, {}))
        box: list = []
        result_boxes[ll.name] = box

        def make_target(stmt, e, b):
            def target():
                _thread_body(stmt, e, channels, wf.ns, b, llm_backend, trace)
            return target

        t = threading.Thread(
            target=make_target(local_stmt, env, box),
            name=ll.name,
            daemon=True,
        )
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=timeout)
        if t.is_alive():
            raise TimeoutError(f"Lifeline '{t.name}' did not finish within {timeout}s")

    final_envs: dict[str, dict] = {}
    for ll in lifelines:
        box = result_boxes[ll.name]
        if not box:
            raise RuntimeError(f"Lifeline '{ll.name}' produced no result.")
        result = box[0]
        if isinstance(result, Exception):
            raise RuntimeError(f"Lifeline '{ll.name}' raised: {result}") from result
        final_envs[ll.name] = result

    if wf.output_var is not None:
        return final_envs[wf.output_lifeline.name][wf.output_var.name]

    return final_envs
