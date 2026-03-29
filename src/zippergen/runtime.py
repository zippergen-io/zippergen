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
import threading
from typing import cast
import time
import textwrap

_planner_path: threading.local = threading.local()

from zippergen.syntax import (
    EmptyStmt, SendStmt, RecvStmt, ActStmt, SkipStmt,
    SeqStmt, IfStmt, WhileStmt, IfRecvStmt, WhileRecvStmt,
    VarExpr, LitExpr, Var,
    LLMAction, PureAction, PlannerAction,
    Lifeline, Workflow, LocalStmt, AnyStmt,
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

    def get(self, *, stop: threading.Event | None = None) -> tuple[int, tuple]:
        if stop is None:
            return self._q.get()
        while True:
            try:
                return self._q.get(timeout=0.5)
            except queue.Empty:
                if stop.is_set():
                    raise RuntimeError("Workflow cancelled: another lifeline failed")


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



# ---------------------------------------------------------------------------
# Planner helpers
# ---------------------------------------------------------------------------

_PLANNER_DSL_RULES = """\
- Name the function exactly `generated_workflow`.
- Declare inputs as `name: str @ {caller}`; return type is `-> str`.
- Send variables between lifelines: `A(x, y) >> B(x, y)`. Both sides MUST list
  the same variables in the same order — the left side sends, the right side binds.
- Single-output action: `A: var = action(arg1, arg2)`.
- Multi-output action: `A: (var1, var2) = action(arg1, arg2)` — use tuple unpacking,
  never subscript (`var["key"]` is invalid).
- REQUIRED: the last statement before `return` MUST send the final result back to {caller}:
  `LastWorker(result) >> {caller}(result)`.
- Return: `return var @ {caller}`.
- Do NOT include any import statements or Var/Lifeline declarations.
- A lifeline can only use variables it has explicitly received. To give a downstream
  worker access to original inputs, forward them: `Worker1(result, var) >> Worker2(result, var)`.

Example — sequential with handoff (Worker1 drafts, Worker2 refines):

@workflow
def generated_workflow(text: str @ {caller}, instructions: str @ {caller}) -> str:
    {caller}(text, instructions) >> Worker1(text, instructions)
    Worker1: draft = write(text, instructions)
    Worker1(draft) >> Worker2(draft)
    Worker2: final = refine(draft, instructions)
    Worker2(final) >> {caller}(final)
    return final @ {caller}

Example — three-worker chain (Worker1 drafts, Worker2 critiques, Worker3 polishes):

@workflow
def generated_workflow(text: str @ {caller}, instructions: str @ {caller}) -> str:
    {caller}(text, instructions) >> Worker1(text, instructions)
    Worker1: draft = write(text, instructions)
    Worker1(draft) >> Worker2(draft)
    Worker2: feedback = critique(draft, instructions)
    Worker2(draft, feedback) >> Worker3(draft, feedback)
    Worker3: final = polish(draft, feedback)
    Worker3(final) >> {caller}(final)
    return final @ {caller}

Example — back-and-forth (Worker1 drafts, Worker2 critiques, Worker1 revises):

@workflow
def generated_workflow(text: str @ {caller}, instructions: str @ {caller}) -> str:
    {caller}(text, instructions) >> Worker1(text, instructions)
    Worker1: draft = write(text, instructions)
    Worker1(draft) >> Worker2(draft)
    Worker2: feedback = critique(draft, instructions)
    Worker2(feedback) >> Worker1(feedback)
    Worker1: final = revise(draft, feedback)
    Worker1(final) >> {caller}(final)
    return final @ {caller}

Example — parallel then aggregate:

@workflow
def generated_workflow(text: str @ {caller}, focus: str @ {caller}, language: str @ {caller}) -> str:
    {caller}(text, focus, language) >> Worker1(text, focus)
    {caller}(text, focus, language) >> Worker2(text, language)
    Worker1: summary = summarise(text, focus)
    Worker2: result = translate(text, language)
    Worker1(summary) >> Aggregator(summary)
    Worker2(result) >> Aggregator(result)
    Aggregator: comparison = compare(summary, result)
    Aggregator(comparison) >> {caller}(comparison)
    return comparison @ {caller}
"""


_PLANNER_ALLOW_PURE = """\
You may define @pure helper functions before the workflow.  Pure functions are
plain Python — no LLM calls, no side effects, no imports.  Use them for
formatting, parsing, or combining strings.

    @pure
    def join_results(a: str, b: str) -> str:
        return a + "\\n\\n" + b

Supported parameter and return types: str, int, float, bool.
"""

_PLANNER_ALLOW_LLM = """\
You may define new @llm actions before the workflow.  Use custom prompts
when no pre-defined action fits the task.

Single output (parse="text"):

    @llm(
        system="Your system prompt here.",
        user="Prompt template with {var_name} placeholders.",
        parse="text",
        outputs=(("output_name", str),),
    )
    def my_action(input1: str, input2: str): ...

    # Used in workflow as:
    # A: result = my_action(input1, input2)

Multiple outputs (parse="json") — use tuple unpacking, never subscript:

    @llm(
        system="Your system prompt here.",
        user="...",
        parse="json",
        outputs=(("out1", str), ("out2", str)),
    )
    def split_action(input1: str): ...

    # Used in workflow as:
    # A: (out1, out2) = split_action(input1)
    # NOT: A: result = split_action(input1)  then result["out1"]  ← invalid

ALL four keyword arguments are REQUIRED for every @llm definition:
  system=  (str)   — the system prompt
  user=    (str)   — the user prompt template, with {var} placeholders
  parse=   (str)   — exactly one of: "text", "json", "bool"
  outputs= (tuple) — MUST be present; one or more (name, type) pairs

parse rules:
- parse="text" → outputs must have exactly one str entry.
- parse="json" → outputs has one or more entries; use tuple unpacking in the workflow.
- parse="bool" → outputs must have exactly one bool entry.
- The function body must be exactly `...`.
- Prefer fewer outputs per action — split responsibilities across actions rather
  than returning many fields from one.
"""

_PLANNER_ALLOW_IF = """\
You may use conditional branching. The owner lifeline evaluates the condition and
controls which branch executes; other lifelines receive the decision automatically.

Syntax:
    if condition @ Owner:
        # true branch
    else:
        # false branch — use `pass` if empty

The condition is a Python boolean expression over variables already bound on Owner.
Supported operators: ==, !=, <, >, <=, >=, not, and, or.

Example — route to a second worker only if the draft needs revision:

    Worker1: (draft, needs_revision) = write_and_assess(text, instructions)
    if needs_revision @ Worker1:
        Worker1(draft) >> Worker2(draft)
        Worker2: draft = revise(draft, instructions)
        Worker2(draft) >> {caller}(draft)
    else:
        Worker1(draft) >> {caller}(draft)
"""

_PLANNER_ALLOW_WHILE = """\
You may use loops. The owner lifeline evaluates the loop condition each iteration.

Syntax:
    while condition @ Owner:
        # loop body — executes while condition is True
    else:
        # exit body — executes once when condition becomes False (REQUIRED, use `pass` if empty)

The condition must reference variables bound on Owner. Loop state (a counter or
convergence flag) must be returned by an action and updated each iteration.
Use parse="json" on an @llm action to return both content and a bool control flag,
or use a @pure counter (if @pure is also allowed).

Example — iterate draft/critique until done or 3 rounds:

    Worker1: (draft, round, done) = start(text, instructions)
    while (not done and round < 3) @ Worker1:
        Worker1(draft) >> Worker2(draft)
        Worker2: feedback = critique(draft)
        Worker2(feedback) >> Worker1(feedback)
        Worker1: (draft, round, done) = revise_and_check(draft, feedback, round)
    else:
        Worker1(draft) >> {caller}(draft)
"""


def _llm_action_to_source(action: LLMAction) -> str:
    """Reconstruct @llm(...) source code from a LLMAction node."""
    params = ", ".join(f"{n}: {t.__name__}" for n, t in action.inputs)
    outputs_repr = (
        "(" + ", ".join(f'("{n}", {t.__name__})' for n, t in action.outputs) + ",)"
    )
    return (
        f"@llm(\n"
        f"    system={action.system_prompt!r},\n"
        f"    user={action.user_prompt!r},\n"
        f"    parse={action.parse_format!r},\n"
        f"    outputs={outputs_repr},\n"
        f")\n"
        f"def {action.name}({params}): ...\n"
    )


def _pure_action_to_source(action: PureAction) -> str:
    """Return the source of a PureAction's underlying function (with @pure decorator)."""
    import inspect as _inspect
    import textwrap as _textwrap
    try:
        return _textwrap.dedent(_inspect.getsource(action.fn))
    except (OSError, TypeError):
        # Fallback: stub with correct signature
        params = ", ".join(f"{n}: {t.__name__}" for n, t in action.inputs)
        _, out_type = action.outputs[0]
        return f"@pure\ndef {action.name}({params}) -> {out_type.__name__}: ...\n"


def _extract_intermediate_var_names(spec: str) -> set[str]:
    """Extract variable names used as outputs in annotated assignments.

    In ``Worker1: summary = summarise(text, focus)``, the annotation is
    ``summary`` — this is the intermediate variable that needs to be declared
    as a Var in the preamble so the builder can resolve it.
    """
    import ast as _ast
    var_names: set[str] = set()
    try:
        tree = _ast.parse(spec)
    except SyntaxError:
        return var_names
    for fn_node in _ast.walk(tree):
        if not (isinstance(fn_node, _ast.FunctionDef) and fn_node.name == "generated_workflow"):
            continue
        for node in _ast.walk(fn_node):
            if not isinstance(node, _ast.AnnAssign):
                continue
            ann = node.annotation
            if isinstance(ann, _ast.Tuple):
                for elt in ann.elts:
                    if isinstance(elt, _ast.Name):
                        var_names.add(elt.id)
            elif isinstance(ann, _ast.Name):
                var_names.add(ann.id)
    return var_names


def _validate_planner_spec(
    spec: str, caller: str, known_actions: set[str] | None = None
) -> str | None:
    """Check structural invariants on a generated workflow.

    Returns None if the spec is valid, or a human-readable error string
    describing the first violation found.

    Invariants:
    1. First statement sends FROM the caller:  ``caller(...) >> X(...)``
    2. Second-to-last statement sends TO the caller:  ``X(...) >> caller(...)``
    3. Last statement returns owned by the caller:  ``return var @ caller``

    The DSL is expressed as Python with custom ``>>`` and ``@`` operators, so
    we check the AST structure rather than evaluating the code.
    """
    import ast as _ast

    try:
        tree = _ast.parse(spec)
    except SyntaxError as exc:
        return f"SyntaxError in generated spec: {exc}"

    # Find the generated_workflow function
    fn_node = None
    for node in _ast.walk(tree):
        if isinstance(node, _ast.FunctionDef) and node.name == "generated_workflow":
            fn_node = node
            break
    if fn_node is None:
        return "No `generated_workflow` function found."

    body = fn_node.body
    # Filter to just expression/return statements (skip docstrings etc.)
    stmts = [s for s in body if isinstance(s, (_ast.Expr, _ast.Return))]
    if len(stmts) < 2:
        return "Workflow body is too short to contain both a send and a return."

    # Helper: does an AST expression represent `Name(...)` i.e. a Call on a Name?
    def call_name(node) -> str | None:
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name):
            return node.func.id
        return None

    # Helper: for `A >> B`, returns (left_name, right_name) or None
    def binop_rshift(node) -> tuple[str, str] | None:
        if (isinstance(node, _ast.BinOp)
                and isinstance(node.op, _ast.RShift)):
            l = call_name(node.left)
            r = call_name(node.right)
            if l and r:
                return (l, r)
        return None

    # Helper: for `return expr @ Name`, returns the Name or None
    def return_at_name(node) -> str | None:
        if not isinstance(node, _ast.Return):
            return None
        val = node.value
        if (isinstance(val, _ast.BinOp)
                and isinstance(val.op, _ast.MatMult)
                and isinstance(val.right, _ast.Name)):
            return val.right.id
        return None

    # --- Invariant 3: last statement is `return ... @ caller` ---
    last = stmts[-1]
    ret_owner = return_at_name(last)
    if ret_owner is None:
        return "Last statement must be `return var @ {caller}` (no `return` found or wrong form).".replace("{caller}", caller)
    if ret_owner != caller:
        return (
            f"Last statement returns to `{ret_owner}` but must return to `{caller}`. "
            f"Use `return var @ {caller}`."
        )

    # --- Invariant 1: first statement sends FROM caller ---
    first_expr = stmts[0]
    if not isinstance(first_expr, _ast.Expr):
        return f"First statement must be `{caller}(...) >> SomeWorker(...)`, got a non-expression."
    pair = binop_rshift(first_expr.value)
    if pair is None:
        return f"First statement must be `{caller}(...) >> SomeWorker(...)` (>> send), got something else."
    if pair[0] != caller:
        return (
            f"First statement sends from `{pair[0]}` but must send from `{caller}`. "
            f"Start with `{caller}(inputs) >> SomeWorker(inputs)`."
        )

    # --- Invariant 2: second-to-last statement sends TO caller ---
    # Use the full body (not just Expr/Return) so control flow nodes are visible.
    non_doc_body = [s for s in body if not (isinstance(s, _ast.Expr)
                                            and isinstance(s.value, _ast.Constant))]
    second_last = non_doc_body[-2] if len(non_doc_body) >= 2 else None
    if second_last is not None and not isinstance(second_last, (_ast.If, _ast.While)):
        # Only check for linear send when the preceding statement is not control flow.
        if not isinstance(second_last, _ast.Expr):
            return f"Statement before `return` must be `SomeWorker(...) >> {caller}(...)`, got a non-expression."
        pair2 = binop_rshift(second_last.value)
        if pair2 is None:
            return (
                f"Statement before `return` must be `SomeWorker(...) >> {caller}(...)` (>> send), "
                f"got something else."
            )
        if pair2[1] != caller:
            return (
                f"Statement before `return` sends to `{pair2[1]}` but must send to `{caller}`. "
                f"End with `SomeWorker(result) >> {caller}(result)` before the return."
            )

    # --- Invariant 4: all action calls are defined or pre-defined ---
    # Collect names called as actions: `A: var = name(...)` or `A: (v1, v2) = name(...)`
    import re as _re
    called_names = set(_re.findall(
        r':\s+(?:\w+|\([^)]+\))\s*=\s*(\w+)\s*\(', spec
    ))
    defined_names = set(_re.findall(r'^def\s+(\w+)', spec, _re.MULTILINE))
    undefined = called_names - defined_names - (known_actions or set())
    if undefined:
        return (
            f"The workflow calls actions that are not defined: {', '.join(sorted(undefined))}. "
            f"Each must be defined as an @llm or @pure action before the workflow function."
        )

    # --- Invariant 5: both sides of every >> send have the same argument count ---
    for node in _ast.walk(tree):
        if not (isinstance(node, _ast.Expr) and isinstance(node.value, _ast.BinOp)
                and isinstance(node.value.op, _ast.RShift)):
            continue
        lhs, rhs = node.value.left, node.value.right
        if (isinstance(lhs, _ast.Call) and isinstance(rhs, _ast.Call)):
            if len(lhs.args) != len(rhs.args):
                lname = lhs.func.id if isinstance(lhs.func, _ast.Name) else "?"
                rname = rhs.func.id if isinstance(rhs.func, _ast.Name) else "?"
                return (
                    f"`{lname}(...)  >> {rname}(...)` has mismatched argument counts "
                    f"({len(lhs.args)} vs {len(rhs.args)}). "
                    f"Both sides must list the same variables: `{lname}(x, y) >> {rname}(x, y)`."
                )

    return None  # all good


def _strip_fences(spec: str) -> str:
    """Remove markdown code fences if the LLM wrapped the output."""
    spec = spec.strip()
    if spec.startswith("```"):
        spec = "\n".join(spec.split("\n")[1:]).strip()
    if spec.endswith("```"):
        spec = "\n".join(spec.split("\n")[:-1]).strip()
    return spec


def _exec_planner(action: PlannerAction, named_inputs: dict, llm_backend, trace=None, parent_seq: int = 0) -> str:
    """Execute a PlannerAction: generate a workflow spec via LLM, then run it."""
    import ast as _ast
    import importlib.util as _ilu
    import os as _os
    import sys as _sys
    import tempfile as _tmpfile
    import threading as _threading

    # The outer lifeline name (e.g. "Planner") is this thread's name.
    # It becomes the "caller" in the inner workflow — supplying inputs and
    # receiving the final result — so we compute it before building any prompts.
    outer_lifeline_name = _threading.current_thread().name

    # --- 1. Build vocabulary descriptions ---
    action_lines: list[str] = []
    for a in action.actions:
        params = ", ".join(f"{n}: {t.__name__}" for n, t in a.inputs)
        out_parts = ", ".join(f"{n}: {t.__name__}" for n, t in a.outputs)
        action_lines.append(f"{a.name}({params}) -> {out_parts}")

    worker_names = [ll.name for ll in action.lifelines]
    worker_list  = ", ".join(worker_names)
    worker_desc  = (
        f"You have {len(worker_names)} worker{'s' if len(worker_names) != 1 else ''} "
        f"available: {worker_list}."
    )

    lifeline_lines = [f"{outer_lifeline_name}    provides inputs, receives the final result"]
    for ll in action.lifelines:
        lifeline_lines.append(ll.name)

    allow_sections: list[str] = []
    if "pure" in action.allow:
        allow_sections.append("Defining @pure actions:\n" + _PLANNER_ALLOW_PURE)
    if "llm" in action.allow:
        allow_sections.append("Defining @llm actions:\n" + _PLANNER_ALLOW_LLM)

    control_flow_sections: list[str] = []
    if "if" in action.allow:
        control_flow_sections.append(
            "Conditional branching:\n"
            + _PLANNER_ALLOW_IF.format(caller=outer_lifeline_name)
        )
    if "while" in action.allow:
        control_flow_sections.append(
            "Loops:\n"
            + _PLANNER_ALLOW_WHILE.format(caller=outer_lifeline_name)
        )

    system_parts = [
        action.system_prompt,
        worker_desc,
        "DSL rules:\n" + _PLANNER_DSL_RULES.format(caller=outer_lifeline_name),
    ]
    if control_flow_sections:
        system_parts.append("Control flow (available for use in the workflow):\n\n" + "\n\n".join(control_flow_sections))
    if action.actions and allow_sections:
        # Both pre-defined vocab and the ability to define new actions.
        system_parts.append("Pre-defined actions (ready to use):\n" + "\n".join(action_lines))
        system_parts.append(
            "In addition, you may define new actions before the workflow:\n\n"
            + "\n".join(allow_sections)
        )
    elif action.actions:
        # Fixed vocabulary only.
        system_parts.append("Pre-defined actions (ready to use):\n" + "\n".join(action_lines))
    elif allow_sections:
        # No pre-defined vocab — LLM must write everything.
        system_parts.append(
            "No pre-defined actions are provided. "
            "Define all actions you need before the workflow:\n\n"
            + "\n".join(allow_sections)
        )
    system_parts.append("Available lifelines:\n" + "\n".join(lifeline_lines))
    coordination_instruction = (
        action.instructions
        if action.instructions
        else "Use as many workers as reasonable, giving each a distinct role."
    )
    system_parts.append(f"Coordination requirement (follow exactly):\n{coordination_instruction}")
    system_parts.append(
        "Return only the Python code. No markdown fences, no explanations, no imports."
    )
    system = "\n\n".join(system_parts)

    # --- 2. Describe available input variables ---
    request_text = str(named_inputs.get("request", ""))
    # All named inputs except "request" are domain data variables passed to workers
    inputs_data = {k: v for k, v in named_inputs.items() if k != "request"}
    _PREVIEW_LEN = 120
    def _preview(v: object) -> str:
        s = str(v)
        return s if len(s) <= _PREVIEW_LEN else s[:_PREVIEW_LEN] + "…"
    inputs_desc = "\n".join(f"- {k}: {_preview(v)}" for k, v in inputs_data.items())

    # --- 3. Call LLM to generate workflow spec ---
    # Pre-format the user content so .format() receives no template variables
    # (avoids breakage if request_text or inputs_desc contains literal braces).
    user_content = (
        f"Request: {request_text}\n\n"
        f"Input variables available:\n{inputs_desc}\n\n"
        "Generate the workflow."
    )
    user_content_safe = user_content.replace("{", "{{").replace("}", "}}")

    spec_action = LLMAction(
        name="_generate_spec",
        inputs=(),
        outputs=(("workflow_spec", str),),
        system_prompt=system,
        user_prompt=user_content_safe,
        parse_format="text",
    )
    spec_result = llm_backend(spec_action, {})
    spec = _strip_fences(str(spec_result.get("workflow_spec", "")))

    # --- Validate structural invariants; re-prompt once on failure ---
    validation_error = _validate_planner_spec(spec, outer_lifeline_name, {a.name for a in action.actions})
    if validation_error:
        correction_prompt = (
            f"The workflow you generated has an error:\n\n"
            f"  {validation_error}\n\n"
            f"Return the complete corrected output — including all @llm and @pure "
            f"action definitions followed by the @workflow function. "
            f"Structural rules:\n"
            f"  1. First statement: `{outer_lifeline_name}(inputs) >> SomeWorker(inputs)`\n"
            f"  2. Second-to-last: `SomeWorker(result) >> {outer_lifeline_name}(result)`\n"
            f"  3. Last statement: `return result @ {outer_lifeline_name}`\n\n"
            f"Current (broken) workflow:\n{spec}"
        )
        correction_prompt_safe = correction_prompt.replace("{", "{{").replace("}", "}}")
        retry_action = LLMAction(
            name="_generate_spec_retry",
            inputs=(),
            outputs=(("workflow_spec", str),),
            system_prompt=system,
            user_prompt=correction_prompt_safe,
            parse_format="text",
        )
        spec_result2 = llm_backend(retry_action, {})
        spec = _strip_fences(str(spec_result2.get("workflow_spec", "")))
        validation_error2 = _validate_planner_spec(spec, outer_lifeline_name, {a.name for a in action.actions})
        if validation_error2:
            raise RuntimeError(
                f"Planner generated an invalid workflow after correction attempt.\n"
                f"Error: {validation_error2}\n\n"
                f"Workflow:\n{spec}"
            )

    print(f"\n{'='*60}")
    print("GENERATED WORKFLOW")
    print("=" * 60)
    print(spec)
    print("=" * 60 + "\n")

    # --- 4. Build preamble for temp file ---
    intermediate_vars = _extract_intermediate_var_names(spec)

    preamble_lines = [
        "from zippergen.syntax import Lifeline, Var",
        "from zippergen.actions import llm, pure",
        "from zippergen.builder import workflow",
        "",
        "# Lifelines",
        f'{outer_lifeline_name} = Lifeline("{outer_lifeline_name}")',
    ]
    for ll in action.lifelines:
        preamble_lines.append(f'{ll.name} = Lifeline("{ll.name}")')
    preamble_lines.append("")
    preamble_lines.append("# Intermediate variables")
    for var_name in sorted(intermediate_vars):
        preamble_lines.append(f'{var_name} = Var("{var_name}", str)')
    preamble_lines.append("")
    preamble_lines.append("# Action library")
    for a in action.actions:
        if isinstance(a, LLMAction):
            preamble_lines.append(_llm_action_to_source(a))
        elif isinstance(a, PureAction):
            preamble_lines.append(_pure_action_to_source(a))

    preamble = "\n".join(preamble_lines) + "\n"
    full_source = preamble + "\n" + spec + "\n"

    # --- 5. Write to temp file, import, and run ---
    fd, tmp_path = _tmpfile.mkstemp(suffix=".py", prefix="zippergen_planner_")
    mod_name = f"_zippergen_plan_{_os.getpid()}_{id(spec)}"
    try:
        with _os.fdopen(fd, "w") as f:
            f.write(full_source)

        spec_obj = _ilu.spec_from_file_location(mod_name, tmp_path)
        if spec_obj is None or spec_obj.loader is None:
            raise RuntimeError("Could not create module spec for generated workflow.")
        mod = _ilu.module_from_spec(spec_obj)
        _sys.modules[mod_name] = mod
        spec_obj.loader.exec_module(mod)

        from zippergen.syntax import Workflow as _Workflow
        wf = getattr(mod, "generated_workflow", None)
        if not isinstance(wf, _Workflow):
            raise RuntimeError("Planner did not produce a valid `generated_workflow`.")

        # Determine path for this planner level
        parent_path: list[str] = list(getattr(_planner_path, 'path', []))
        my_path = parent_path + [action.name]
        _planner_path.path = my_path

        # Collect inner lifeline names from the loaded workflow
        from zippergen.syntax import _ordered_workflow_lifelines
        inner_lifeline_names = [ll.name for ll in _ordered_workflow_lifelines(wf)]

        # Emit level_push so the frontend can create the child level view
        if trace:
            trace({
                "type": "level_push",
                "name": action.name,
                "path": my_path,
                "lifelines": inner_lifeline_names,
                "parent_seq": parent_seq,
            })

        # Wrap trace to tag all inner events with my_path
        def _inner_trace(event: dict) -> None:
            if trace:
                trace({**event, "path": my_path})
        wf._trace = _inner_trace

        # Extract inner workflow input names from the generated spec
        inner_param_names: list[str] = []
        try:
            tree = _ast.parse(spec)
            for fn_node in _ast.walk(tree):
                if isinstance(fn_node, _ast.FunctionDef) and fn_node.name == "generated_workflow":
                    inner_param_names = [arg.arg for arg in fn_node.args.args]
                    break
        except SyntaxError:
            pass

        inputs_for_wf = {name: named_inputs[name] for name in inner_param_names if name in named_inputs}

        # Wrap the outer backend so inner lifeline threads route through the
        # Planner's provider.  The outer router is keyed by thread name, and
        # _exec_planner is always called from the Planner thread, so we capture
        # that name here and temporarily restore it inside each inner-thread call.
        def _inner_backend(act_node, inp):
            t = _threading.current_thread()
            saved = t.name
            t.name = outer_lifeline_name
            try:
                return llm_backend(act_node, inp)
            finally:
                t.name = saved

        wf._backend = _inner_backend
        wf._timeout = 180
        try:
            result = str(wf._run_once(inputs_for_wf))
        finally:
            _planner_path.path = parent_path
            if trace:
                trace({"type": "level_pop", "path": my_path})
        return result

    finally:
        _sys.modules.pop(mod_name, None)
        try:
            _os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Local-program interpreter
# ---------------------------------------------------------------------------

def _exec(stmt: AnyStmt, env: Env, ch: Channels, ns: dict, llm_backend, trace,
          stop: threading.Event | None = None) -> None:
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
            seq, values = ch[(B.name, A.name)].get(stop=stop)
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
                out_map = {outs[0].name: raw} if len(outs) == 1 else {
                    var.name: val for var, val in zip(outs, cast(tuple, raw))
                }
            elif isinstance(action, PlannerAction):
                out_map = {outs[0].name: _exec_planner(action, named_inputs, llm_backend, trace, seq)}
            else:
                named_outputs = llm_backend(action, named_inputs)
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
            _exec(p1, env, ch, ns, llm_backend, trace, stop)
            _exec(p2, env, ch, ns, llm_backend, trace, stop)

        case IfStmt(condition=c, owner=_, branch_true=t, branch_false=f):
            _exec(t if c(_CondEnv(env, ns)) else f, env, ch, ns, llm_backend, trace, stop)

        case IfRecvStmt(lifeline=A, bindings=ys, sender=B, branch_true=t, branch_false=f):
            seq, values = ch[(B.name, A.name)].get(stop=stop)
            _bind(ys, values, env)
            flag = _eval(ys[0], env) if isinstance(ys[0], VarExpr) else values[0]
            if trace:
                trace({
                    "type": "recv",
                    "to": A.name, "from": B.name,
                    "bindings": {"branch": "true" if flag else "false"},
                    "seq": seq, "ctrl": True,
                })
            _exec(t if flag else f, env, ch, ns, llm_backend, trace, stop)

        case WhileStmt(condition=c, owner=_, body=body, exit_body=exit_b):
            while c(_CondEnv(env, ns)):
                _exec(body, env, ch, ns, llm_backend, trace, stop)
            _exec(exit_b, env, ch, ns, llm_backend, trace, stop)

        case WhileRecvStmt(lifeline=A, bindings=ys, sender=B, body=body, exit_body=exit_b):
            while True:
                seq, values = ch[(B.name, A.name)].get(stop=stop)
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
                _exec(body, env, ch, ns, llm_backend, trace, stop)
            _exec(exit_b, env, ch, ns, llm_backend, trace, stop)

        case _:
            raise TypeError(f"Unknown local stmt: {type(stmt).__name__}")


# ---------------------------------------------------------------------------
# Per-lifeline thread body
# ---------------------------------------------------------------------------

def _thread_body(local_stmt, env, ch, ns, result_box, llm_backend, trace, stop):
    try:
        _exec(local_stmt, env, ch, ns, llm_backend, trace, stop)
        result_box.append(env)
    except Exception as exc:
        stop.set()  # unblock any threads waiting on queue.get()
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

    stop = threading.Event()

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
                _thread_body(stmt, e, channels, wf.ns, b, llm_backend, trace, stop)
            return target

        t = threading.Thread(
            target=make_target(local_stmt, env, box),
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

    if wf.output_var is not None and wf.output_lifeline is not None:
        return final_envs[wf.output_lifeline.name][wf.output_var.name]

    return final_envs
