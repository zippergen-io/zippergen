"""
ZipperGen — Planner subsystem.

Handles LLM-driven dynamic workflow generation (PlannerAction). Given a
PlannerAction node, _exec_planner builds a system prompt from the action's
vocabulary and constraints, calls the LLM to generate a ZipperGen workflow
spec, validates it, writes the spec to a temp file, imports it as a module,
and runs the resulting Workflow.

Public surface:
  _validate_planner_spec  — structural validator (returns None or error string)
  _exec_planner           — full generation + validation + execution pipeline

Prompt templates (_PLANNER_DSL_RULES, _PLANNER_ALLOW_*) are plain string
constants so they can be studied, ablated, and extended independently.
"""

from __future__ import annotations

import threading

from zippergen.syntax import (
    LLMAction, PureAction, PlannerAction,
)

__all__ = ["_exec_planner", "_validate_planner_spec"]


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_PLANNER_DSL_RULES = """\
Syntax
------
- Name the function exactly `generated_workflow`.
- Signature: `def generated_workflow({input_sig}) -> str:`
  Annotate each parameter with the lifeline that holds it at workflow start:
  `name: type @ Lifeline`. Inputs may be distributed across different lifelines.
- Action call: `Lifeline: var = action(arg1, arg2)` (single output)
             or `Lifeline: (var1, var2) = action(arg1, arg2)` (multi-output, no subscript).
  Arguments may be variable names or literals (str, int, float):
  `A: r = add(x, 3.14)` — match literal type to the action's parameter type.
- Message:  `A(x, y) >> B(x, y)` — both sides list the same variables in the same order.
- Return:   `return var @ Lifeline` — the last (and only) return in the workflow.
- Self-send `A(x) >> A(y)` renames `x` to `y` in A's scope; names must differ.

Rules
-----
- A lifeline's own action outputs are immediately in scope — never self-forward:
  after `Worker1: x = f(a)` and `Worker1: y = g(b)`, write `Worker1: z = h(x, y)` directly.
- A lifeline can only use variables it produced or received via `>>`.
  Forward inputs explicitly: `Sender(var) >> Receiver(var)`.
- Every input parameter must appear in at least one action call or send.
- Output variable names must not match any action name
  (e.g. use `is_zero_result` not `is_zero` if `is_zero` is an action).
- Every action call must use `Lifeline: var = action(...)` — no bare `var = action(...)`.
- Do NOT include import statements or Var/Lifeline declarations.
- The RHS of `Lifeline: var = ...` must be a function call, never a plain variable.

Examples
--------
Sequential with handoff (Worker1 drafts, Worker2 refines):

@workflow
def generated_workflow(text: str @ Worker1, instructions: str @ Worker1) -> str:
    Worker1: draft = write(text, instructions)
    Worker1(draft) >> Worker2(draft)
    Worker2: final = refine(draft, instructions)
    return final @ Worker2

Three-worker chain (draft → critique → polish):

@workflow
def generated_workflow(text: str @ Worker1, instructions: str @ Worker1) -> str:
    Worker1: draft = write(text, instructions)
    Worker1(draft, instructions) >> Worker2(draft, instructions)
    Worker2: feedback = critique(draft, instructions)
    Worker2(draft, feedback) >> Worker3(draft, feedback)
    Worker3: final = polish(draft, feedback)
    return final @ Worker3

Back-and-forth (draft → critique → revise):

@workflow
def generated_workflow(text: str @ Worker1, instructions: str @ Worker1) -> str:
    Worker1: draft = write(text, instructions)
    Worker1(draft, instructions) >> Worker2(draft, instructions)
    Worker2: feedback = critique(draft, instructions)
    Worker2(feedback) >> Worker1(feedback)
    Worker1: final = revise(draft, feedback, instructions)
    return final @ Worker1

Parallel then aggregate (each worker gets its own input directly):

@workflow
def generated_workflow(doc1: str @ Worker1, doc2: str @ Worker2) -> str:
    Worker1: summary1 = summarise(doc1)
    Worker2: summary2 = summarise(doc2)
    Worker1(summary1) >> Aggregator(summary1)
    Worker2(summary2) >> Aggregator(summary2)
    Aggregator: merged = merge(summary1, summary2)
    return merged @ Aggregator
"""


_PLANNER_ALLOW_PURE = """\
You may define @pure helper functions before the workflow. Pure functions are
plain Python — no LLM calls, no side effects, no imports. Use them for
formatting, parsing, or combining strings.

    @pure
    def join_results(a: str, b: str) -> str:
        return a + "\\n\\n" + b

Supported parameter and return types: str, int, float, bool.
"""


_PLANNER_ALLOW_LLM = """\
You may define new @llm actions before the workflow. Use custom prompts
when no pre-defined action fits the task.

Single output (parse="text"):

    @llm(
        system="Your system prompt here.",
        user="Prompt template with {var_name} placeholders.",
        parse="text",
        outputs=(("output_name", str),),
    )
    def my_action(input1: str, input2: str): ...

Multiple outputs (parse="json") — use tuple unpacking in the workflow:

    @llm(
        system="Your system prompt here.",
        user="...",
        parse="json",
        outputs=(("out1", str), ("out2", str)),
    )
    def split_action(input1: str): ...

    # A: (out1, out2) = split_action(input1)   ← correct
    # A: result = split_action(input1)          ← wrong (cannot subscript)

ALL four keyword arguments are REQUIRED: system=, user=, parse=, outputs=.
parse="text" → one str output. parse="json" → one or more outputs.
parse="bool" → one bool output. Function body must be exactly `...`.
"""


_PLANNER_ALLOW_IF = """\
You may use conditional branching. The owner lifeline evaluates the condition;
all other participating lifelines receive the decision automatically.

Syntax:
    if condition @ Owner:
        # true branch
    else:
        # false branch — use `pass` if empty

The condition is a Python boolean expression over variables bound on Owner.
Supported operators: ==, !=, <, >, <=, >=, not, and, or.

No early return: `return` is the very last statement of the whole workflow —
never inside a branch. To skip remaining work on an error condition, put ALL
downstream computation in the `else` branch and produce a sentinel in `if`:

    Worker: denom = compute(x)
    Worker: denom_zero = is_zero(denom)
    if denom_zero @ Worker:
        Worker: result = error_value()        # sentinel; all further work skipped
    else:
        Worker: result = divide(y, denom)
        # ... any further computation goes here, inside else
    return result @ Worker

Both branches must produce the return variable under the SAME name on the
SAME lifeline:

    if cond @ Worker1:
        Worker1: result = option_a(draft)     # ← result on Worker1
    else:
        Worker1: result = option_b(draft)     # ← result on Worker1
    return result @ Worker1                   # ← OK: same name, same lifeline

Example — draft/assess/revise loop with optional feedback:

    Worker1: draft = write(text, instructions)
    Worker1(draft, instructions) >> Worker2(draft, instructions)
    Worker2: (needs_revision, feedback) = assess(draft, instructions)
    if needs_revision @ Worker2:
        Worker2(draft, feedback) >> Worker1(draft, feedback)
        Worker1: revised = revise(draft, feedback)
        Worker1(revised) >> Worker2(result)
    else:
        Worker2: result = refine(draft, instructions)
    return result @ Worker2
"""


_PLANNER_ALLOW_WHILE = """\
You may use loops. The owner lifeline evaluates the loop condition each iteration.

Syntax:
    while condition @ Owner:
        # loop body — executes while condition is True
    else:
        # exit body — runs once when condition becomes False (use `pass` if empty)

The condition must reference variables bound on Owner. Loop state must be
returned by an action and updated each iteration. Use parse="json" on an @llm
action to return both content and a bool control flag, or use a @pure counter.

Example — iterate draft/critique until done or 3 rounds:

    Worker1: (draft, round, done) = start(text, instructions)
    while (not done and round < 3) @ Worker1:
        Worker1(draft) >> Worker2(draft)
        Worker2: feedback = critique(draft)
        Worker2(feedback) >> Worker1(feedback)
        Worker1: (draft, round, done) = revise_and_check(draft, feedback, round)
    else:
        pass
    return draft @ Worker1
"""


# ---------------------------------------------------------------------------
# Spec helpers
# ---------------------------------------------------------------------------

def _extract_intermediate_var_names(spec: str) -> set[str]:
    """Return variable names that need Var declarations in the generated preamble.

    Collects output names from action calls (`Lifeline: var = f(...)`) and
    receive-side bindings of sends (`A(x) >> B(result)` → `result`).
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
            if isinstance(node, _ast.AnnAssign):
                ann = node.annotation
                if isinstance(ann, _ast.Tuple):
                    var_names.update(e.id for e in ann.elts if isinstance(e, _ast.Name))
                elif isinstance(ann, _ast.Name):
                    var_names.add(ann.id)
            elif (isinstance(node, _ast.Expr)
                  and isinstance(node.value, _ast.BinOp)
                  and isinstance(node.value.op, _ast.RShift)):
                rhs = node.value.right
                if isinstance(rhs, _ast.Call):
                    var_names.update(a.id for a in rhs.args if isinstance(a, _ast.Name))
    return var_names


def _validate_planner_spec(
    spec: str,
    caller: str,
    known_actions: dict[str, int] | None = None,
    expected_input_types: dict[str, type] | None = None,
) -> str | None:
    """Check structural invariants on a generated workflow spec.

    Returns None if valid, or a human-readable error string on the first
    violation. Checks, in order:

      1. Signature     — parameter names and types match expected_input_types
      2. Return form   — last visible statement is `return var @ Lifeline`
      3. Actions       — every called action is defined or pre-defined
      4. Name clash    — output variable names do not shadow action names
      5. Output counts — call-site unpacking matches declared output count
      6. Sends         — >> sends have matching arg counts; self-sends disjoint
      7. Assignments   — no bare assignments or non-call RHS
      8. Scope         — per-lifeline variable scope respected throughout
      9. Return scope  — return variable is in scope on all control-flow paths
     10. Inputs used   — every input parameter appears in at least one call/send
    """
    import ast as _ast
    import re as _re

    # ---- Parse ----
    try:
        tree = _ast.parse(spec)
    except SyntaxError as exc:
        return f"SyntaxError in generated spec: {exc}"

    fn_node = next(
        (n for n in _ast.walk(tree)
         if isinstance(n, _ast.FunctionDef) and n.name == "generated_workflow"),
        None,
    )
    if fn_node is None:
        return "No `generated_workflow` function found."

    has_workflow_deco = any(
        (isinstance(d, _ast.Name) and d.id == "workflow")
        for d in fn_node.decorator_list
    )
    if not has_workflow_deco:
        return "Function `generated_workflow` must be decorated with `@workflow`."

    # Visible expression/return statements in the function body (excludes AnnAssign).
    stmts = [s for s in fn_node.body if isinstance(s, (_ast.Expr, _ast.Return))]
    if not stmts:
        return "Workflow body is empty — must contain at least one action and a return."

    defined_names = set(_re.findall(r'^def\s+(\w+)', spec, _re.MULTILINE))
    all_action_names = set(known_actions or {}) | (defined_names - {"generated_workflow"})

    # ---- 1. Signature ----
    def check_signature() -> str | None:
        if not expected_input_types:
            return None
        _TYPE_MAP = {"str": str, "int": int, "bool": bool, "float": float, "tuple": tuple}
        declared = {arg.arg for arg in fn_node.args.args}
        expected = set(expected_input_types)
        sig = ", ".join(f"{n}: {t.__name__} @ <Lifeline>" for n, t in expected_input_types.items())

        missing = expected - declared
        if missing:
            return (f"Missing parameter(s): {', '.join(sorted(missing))}. "
                    f"Signature must include: def generated_workflow({sig}) -> str:")
        extra = declared - expected
        if extra:
            return (f"Unexpected parameter(s): {', '.join(sorted(extra))}. "
                    f"Signature must include: def generated_workflow({sig}) -> str:")

        for arg in fn_node.args.args:
            ann = arg.annotation
            type_node = (ann.left if isinstance(ann, _ast.BinOp)
                         and isinstance(ann.op, _ast.MatMult) else ann)
            if not isinstance(type_node, _ast.Name):
                continue
            actual_t = _TYPE_MAP.get(type_node.id)
            expected_t = expected_input_types[arg.arg]
            if actual_t is not None and actual_t is not expected_t:
                return (f"Parameter `{arg.arg}` declared as `{type_node.id}` "
                        f"but must be `{expected_t.__name__}`. "
                        f"Use `{arg.arg}: {expected_t.__name__} @ <Lifeline>`.")
        return None

    # ---- 2. Return form ----
    def check_return_form() -> str | None:
        last = stmts[-1]
        if not (isinstance(last, _ast.Return)
                and isinstance(last.value, _ast.BinOp)
                and isinstance(last.value.op, _ast.MatMult)
                and isinstance(last.value.right, _ast.Name)):
            return "Last statement must be `return var @ Lifeline`."
        return None

    # ---- 3. Actions defined ----
    def check_actions_defined() -> str | None:
        called = set(_re.findall(r':\s+(?:\w+|\([^)]+\))\s*=\s*(\w+)\s*\(', spec))
        undefined = called - defined_names - set(known_actions or {})
        if undefined:
            return (f"Undefined action(s): {', '.join(sorted(undefined))}. "
                    f"Define each as @llm or @pure before the workflow.")
        return None

    # ---- 4. Name clash (output var vs action name) ----
    def check_name_collisions() -> str | None:
        for node in _ast.walk(fn_node):
            if not isinstance(node, _ast.AnnAssign):
                continue
            ann = node.annotation
            if isinstance(ann, _ast.Name):
                names = [ann.id]
            elif isinstance(ann, _ast.Tuple):
                names = [e.id for e in ann.elts if isinstance(e, _ast.Name)]
            else:
                names = []
            for name in names:
                if name in all_action_names:
                    return (f"Output variable `{name}` has the same name as action `{name}`. "
                            f"Rename it — e.g. `{name}_result`.")
        return None

    # ---- 5. Output counts ----
    def check_output_counts() -> str | None:
        if not known_actions:
            return None
        for node in _ast.walk(tree):
            if not (isinstance(node, _ast.AnnAssign)
                    and isinstance(node.value, _ast.Call)
                    and isinstance(node.value.func, _ast.Name)
                    and node.value.func.id in known_actions):
                continue
            fn_name = node.value.func.id
            expected = known_actions[fn_name]
            ann = node.annotation
            actual = (1 if isinstance(ann, _ast.Name)
                      else len([e for e in ann.elts if isinstance(e, _ast.Name)])
                      if isinstance(ann, _ast.Tuple) else None)
            if actual is None or actual == expected:
                continue
            ll = node.target.id if isinstance(node.target, _ast.Name) else "?"
            lhs = ("result" if expected == 1
                   else "(" + ", ".join(f"out{i+1}" for i in range(expected)) + ")")
            return (f"`{ll}: {_ast.unparse(ann)} = {fn_name}(...)` unpacks {actual} "
                    f"output{'s' if actual != 1 else ''} but `{fn_name}` returns {expected}. "
                    f"Use: `{ll}: {lhs} = {fn_name}(...)`.")
        return None

    # ---- 6. Sends ----
    def check_sends() -> str | None:
        for node in _ast.walk(tree):
            if not (isinstance(node, _ast.Expr)
                    and isinstance(node.value, _ast.BinOp)
                    and isinstance(node.value.op, _ast.RShift)):
                continue
            lhs, rhs = node.value.left, node.value.right
            if not (isinstance(lhs, _ast.Call) and isinstance(rhs, _ast.Call)):
                continue
            ln = lhs.func.id if isinstance(lhs.func, _ast.Name) else "?"
            rn = rhs.func.id if isinstance(rhs.func, _ast.Name) else "?"
            if ln == rn:
                overlap = ({a.id for a in lhs.args if isinstance(a, _ast.Name)}
                           & {a.id for a in rhs.args if isinstance(a, _ast.Name)})
                if overlap:
                    vs = ", ".join(sorted(overlap))
                    return (f"`{ln}({vs}) >> {rn}({vs})` is a no-op self-send "
                            f"(same variable on both sides). Use distinct names: "
                            f"`{ln}(x) >> {ln}(y)`.")
            if len(lhs.args) != len(rhs.args):
                return (f"`{ln}(...) >> {rn}(...)` has mismatched argument counts "
                        f"({len(lhs.args)} vs {len(rhs.args)}). "
                        f"Both sides must list the same variables.")
        return None

    # ---- 7. Assignments ----
    def check_assignments() -> str | None:
        for node in _ast.walk(tree):
            if isinstance(node, _ast.AnnAssign) and node.value is not None:
                if not isinstance(node.value, _ast.Call):
                    tgt = node.target.id if isinstance(node.target, _ast.Name) else "?"
                    ann = node.annotation.id if isinstance(node.annotation, _ast.Name) else "?"
                    return (f"`{tgt}: {ann} = <non-call>` — RHS must be a function call. "
                            f"To rename a variable use: `Sender({ann}) >> Other(new_name)`.")
            elif isinstance(node, _ast.Assign) and isinstance(node.value, _ast.Call):
                fn_name = node.value.func.id if isinstance(node.value.func, _ast.Name) else "?"
                targets = ", ".join(t.id for t in node.targets if isinstance(t, _ast.Name))
                return (f"Bare assignment `{targets} = {fn_name}(...)` — "
                        f"use `Lifeline: {targets} = {fn_name}(...)`.")
        return None

    # ---- 8 + 9. Scope and return-variable scope ----
    def check_scope_and_return() -> str | None:
        # Seed each lifeline's scope from the signature annotations.
        scope: dict[str, set[str]] = {}
        for arg in fn_node.args.args:
            ann = arg.annotation
            owner = (ann.right.id
                     if isinstance(ann, _ast.BinOp) and isinstance(ann.op, _ast.MatMult)
                     and isinstance(ann.right, _ast.Name)
                     else caller)
            scope.setdefault(owner, set()).add(arg.arg)

        def _walk(body: list, sc: dict[str, set[str]]) -> str | None:
            for stmt in body:
                if isinstance(stmt, _ast.AnnAssign) and isinstance(stmt.target, _ast.Name):
                    ll = stmt.target.id
                    ann = stmt.annotation
                    outs = ([ann.id] if isinstance(ann, _ast.Name)
                            else [e.id for e in ann.elts if isinstance(e, _ast.Name)]
                            if isinstance(ann, _ast.Tuple) else [])
                    if isinstance(stmt.value, _ast.Call):
                        for arg in stmt.value.args:
                            if isinstance(arg, _ast.Name) and arg.id not in sc.get(ll, set()):
                                fn_n = (stmt.value.func.id
                                        if isinstance(stmt.value.func, _ast.Name) else "?")
                                return (f"`{ll}: ... = {fn_n}(...)` uses `{arg.id}` "
                                        f"but `{ll}` has not received it. "
                                        f"Forward it: `Sender({arg.id}) >> {ll}({arg.id})`.")
                    sc.setdefault(ll, set()).update(outs)

                elif (isinstance(stmt, _ast.Expr)
                      and isinstance(stmt.value, _ast.BinOp)
                      and isinstance(stmt.value.op, _ast.RShift)):
                    lhs2, rhs2 = stmt.value.left, stmt.value.right
                    if isinstance(lhs2, _ast.Call) and isinstance(lhs2.func, _ast.Name):
                        sender = lhs2.func.id
                        for arg in lhs2.args:
                            if isinstance(arg, _ast.Name) and arg.id not in sc.get(sender, set()):
                                rn = (rhs2.func.id if isinstance(rhs2, _ast.Call)
                                      and isinstance(rhs2.func, _ast.Name) else "?")
                                return (f"`{sender}({arg.id}) >> {rn}(...)` sends `{arg.id}` "
                                        f"but `{sender}` has not received it.")
                    if isinstance(rhs2, _ast.Call) and isinstance(rhs2.func, _ast.Name):
                        rcv = rhs2.func.id
                        for arg in rhs2.args:
                            if isinstance(arg, _ast.Name):
                                sc.setdefault(rcv, set()).add(arg.id)

                elif isinstance(stmt, _ast.If):
                    sc_t = {k: set(v) for k, v in sc.items()}
                    sc_f = {k: set(v) for k, v in sc.items()}
                    err = _walk(stmt.body, sc_t) or _walk(stmt.orelse, sc_f)
                    if err:
                        return err
                    for ll in set(sc_t) | set(sc_f):
                        sc[ll] = sc_t.get(ll, set()) & sc_f.get(ll, set())

                elif isinstance(stmt, _ast.While):
                    sc_body = {k: set(v) for k, v in sc.items()}
                    sc_exit = {k: set(v) for k, v in sc.items()}
                    err = _walk(stmt.body, sc_body) or _walk(stmt.orelse, sc_exit)
                    if err:
                        return err
                    for ll in set(sc_exit):
                        sc[ll] = sc_exit.get(ll, set())
            return None

        err = _walk(fn_node.body, scope)
        if err:
            return err

        # Check that the return variable is in scope on all paths.
        ret = stmts[-1]
        if (isinstance(ret, _ast.Return)
                and isinstance(ret.value, _ast.BinOp)
                and isinstance(ret.value.left, _ast.Name)
                and isinstance(ret.value.right, _ast.Name)):
            ret_var, ret_ll = ret.value.left.id, ret.value.right.id
            if ret_var not in scope.get(ret_ll, set()):
                return (f"`return {ret_var} @ {ret_ll}`: `{ret_var}` is not available on "
                        f"`{ret_ll}` on all control-flow paths. Every branch must produce "
                        f"or forward `{ret_var}` to `{ret_ll}` before the return.")
        return None

    # ---- 10. Inputs used ----
    def check_inputs_used() -> str | None:
        used: set[str] = set()
        for node in _ast.walk(fn_node):
            if isinstance(node, _ast.AnnAssign) and isinstance(node.value, _ast.Call):
                used.update(a.id for a in node.value.args if isinstance(a, _ast.Name))
            elif (isinstance(node, _ast.Expr)
                  and isinstance(node.value, _ast.BinOp)
                  and isinstance(node.value.op, _ast.RShift)):
                lhs = node.value.left
                if isinstance(lhs, _ast.Call):
                    used.update(a.id for a in lhs.args if isinstance(a, _ast.Name))
        for arg in fn_node.args.args:
            if arg.arg not in used:
                return (f"Input parameter `{arg.arg}` is never used. "
                        f"Every input must appear in at least one action call or send: "
                        f"`Lifeline: out = action({arg.arg}, ...)` "
                        f"or `Lifeline({arg.arg}) >> Other({arg.arg})`.")
        return None

    return (check_signature()
            or check_return_form()
            or check_actions_defined()
            or check_name_collisions()
            or check_output_counts()
            or check_sends()
            or check_assignments()
            or check_scope_and_return()
            or check_inputs_used())


def _strip_fences(spec: str) -> str:
    """Remove markdown code fences if the LLM wrapped the output."""
    spec = spec.strip()
    if spec.startswith("```"):
        spec = "\n".join(spec.split("\n")[1:]).strip()
    if spec.endswith("```"):
        spec = "\n".join(spec.split("\n")[:-1]).strip()
    return spec


# ---------------------------------------------------------------------------
# Module-level thread-local for nested planner path tracking
# ---------------------------------------------------------------------------

_planner_path: threading.local = threading.local()


# ---------------------------------------------------------------------------
# Main planner executor
# ---------------------------------------------------------------------------

def _exec_planner(action: PlannerAction, named_inputs: dict, llm_backend, trace=None, parent_seq: int = 0) -> str:
    """Execute a PlannerAction: generate a workflow spec via LLM, then run it."""
    import ast as _ast
    import importlib.util as _ilu
    import os as _os
    import sys as _sys
    import tempfile as _tmpfile
    import threading as _threading

    outer_lifeline_name = _threading.current_thread().name

    # --- 1. Build system prompt ---
    action_lines = [
        f"{a.name}({', '.join(f'{n}: {t.__name__}' for n, t in a.inputs)}) "
        f"-> {', '.join(f'{n}: {t.__name__}' for n, t in a.outputs)}"
        for a in action.actions
    ]
    worker_names = [ll.name for ll in action.lifelines]

    allow_sections: list[str] = []
    if "pure" in action.allow:
        allow_sections.append("Defining @pure actions:\n" + _PLANNER_ALLOW_PURE)
    if "llm" in action.allow:
        allow_sections.append("Defining @llm actions:\n" + _PLANNER_ALLOW_LLM)

    control_flow_sections: list[str] = []
    if "if" in action.allow:
        control_flow_sections.append("Conditional branching:\n" + _PLANNER_ALLOW_IF)
    if "while" in action.allow:
        control_flow_sections.append("Loops:\n" + _PLANNER_ALLOW_WHILE)

    input_sig = ", ".join(
        f"{name}: {t.__name__}"
        for name, t in action.inputs
        if name != "request"
    )
    system_parts = [
        action.system_prompt,
        (f"You have {len(worker_names)} worker{'s' if len(worker_names) != 1 else ''} "
         f"available: {', '.join(worker_names)}."),
        "DSL rules:\n" + _PLANNER_DSL_RULES.format(input_sig=input_sig),
    ]
    if control_flow_sections:
        system_parts.append(
            "Control flow (available for use in the workflow):\n\n"
            + "\n\n".join(control_flow_sections)
        )
    if action.actions and allow_sections:
        system_parts.append("Pre-defined actions (ready to use):\n" + "\n".join(action_lines))
        system_parts.append(
            "In addition, you may define new actions before the workflow:\n\n"
            + "\n".join(allow_sections)
        )
    elif action.actions:
        system_parts.append("Pre-defined actions (ready to use):\n" + "\n".join(action_lines))
    elif allow_sections:
        system_parts.append(
            "No pre-defined actions are provided. "
            "Define all actions you need before the workflow:\n\n"
            + "\n".join(allow_sections)
        )
    system_parts.append("Available lifelines:\n" + "\n".join(worker_names))
    system_parts.append(
        f"Coordination requirement (follow exactly):\n"
        + (action.instructions or "Use as many workers as reasonable, giving each a distinct role.")
    )
    system_parts.append("Return only the Python code. No markdown fences, no explanations, no imports.")
    system = "\n\n".join(system_parts)

    # --- 2. Build user prompt ---
    request_text = str(named_inputs.get("request", ""))
    inputs_data  = {k: v for k, v in named_inputs.items() if k != "request"}
    input_types  = {name: t for name, t in action.inputs if name != "request"}
    _PREVIEW_LEN = 120
    def _preview(v: object) -> str:
        s = str(v)
        return s if len(s) <= _PREVIEW_LEN else s[:_PREVIEW_LEN] + "…"
    inputs_section = (
        "Input variables available:\n"
        + "\n".join(f"- {k} ({input_types.get(k, type(v)).__name__}): {_preview(v)}"
                    for k, v in inputs_data.items())
        + "\n\n"
    ) if inputs_data else ""
    user_content = f"Request: {request_text}\n\n{inputs_section}Generate the workflow."
    user_content_safe = user_content.replace("{", "{{").replace("}", "}}")

    # --- 3. Call LLM ---
    spec_result = llm_backend(
        LLMAction(name="_generate_spec", inputs=(), outputs=(("workflow_spec", str),),
                  system_prompt=system, user_prompt=user_content_safe, parse_format="text"),
        {},
    )
    spec = _strip_fences(str(spec_result.get("workflow_spec", "")))

    # Guard: worker names must not clash with the calling lifeline.
    conflicts = [n for n in worker_names if n == outer_lifeline_name]
    if conflicts:
        raise ValueError(
            f"Planner worker lifeline(s) {conflicts} share the name of the calling "
            f"lifeline '{outer_lifeline_name}'. Worker names must be distinct."
        )

    # --- 4. Validate; re-prompt on failure ---
    known_actions = {a.name: len(a.outputs) for a in action.actions
                     if isinstance(a, (LLMAction, PureAction, PlannerAction))}
    attempts_used = 1
    for attempt in range(action.max_retries):
        error = _validate_planner_spec(spec, outer_lifeline_name, known_actions, input_types)
        if error is None:
            attempts_used = attempt + 1
            break
        print(f"[planner] attempt {attempt + 1} failed: {error}")
        hint = (
            "\nHint: a lifeline already has every variable it produced — "
            "do NOT self-forward those back. Only send variables from OTHER lifelines.\n"
            if "self-send" in error else ""
        )
        correction = (
            f"The workflow you generated has an error:\n\n  {error}\n{hint}\n"
            f"Return the complete corrected output (all @llm/@pure definitions + @workflow).\n"
            f"Key rules:\n"
            f"  - Each parameter annotated with its owner: `name: type @ Lifeline`\n"
            f"  - Last statement: `return var @ Lifeline` where var is in scope\n\n"
            f"Current (broken) workflow:\n{spec}"
        )
        spec_result = llm_backend(
            LLMAction(
                name=f"_generate_spec_retry{attempt + 1}",
                inputs=(), outputs=(("workflow_spec", str),),
                system_prompt=system,
                user_prompt=correction.replace("{", "{{").replace("}", "}}"),
                parse_format="text",
            ),
            {},
        )
        spec = _strip_fences(str(spec_result.get("workflow_spec", "")))
    else:
        error = _validate_planner_spec(spec, outer_lifeline_name, known_actions, input_types)
        if error:
            raise RuntimeError(
                f"Planner failed after {action.max_retries} attempts.\n"
                f"Error: {error}\n\nWorkflow:\n{spec}"
            )

    attempt_str = f"{attempts_used} attempt{'s' if attempts_used > 1 else ''}"
    print(f"\n{'='*60}\nGENERATED WORKFLOW  ({attempt_str})\n{'='*60}\n{spec}\n{'='*60}\n")

    # --- 5. Build preamble, write temp file, import ---
    intermediate_vars = _extract_intermediate_var_names(spec)
    preamble_lines = [
        "from zippergen.syntax import Lifeline, Var",
        "from zippergen.builder import workflow",
        "from zippergen.actions import llm, pure, planner",
        "",
        f'{outer_lifeline_name} = Lifeline("{outer_lifeline_name}")',
    ] + [f'{ll.name} = Lifeline("{ll.name}")' for ll in action.lifelines] + [
        "",
    ] + [f'{v} = Var("{v}", str)' for v in sorted(intermediate_vars)]

    full_source = "\n".join(preamble_lines) + "\n\n" + spec + "\n"

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
        for a in action.actions:
            mod.__dict__[a.name] = a
        spec_obj.loader.exec_module(mod)

        from zippergen.syntax import Workflow as _Workflow
        wf = getattr(mod, "generated_workflow", None)
        if not isinstance(wf, _Workflow):
            raise RuntimeError("Planner did not produce a valid `generated_workflow`.")

        # --- 6. Configure trace and run ---
        parent_path: list[str] = list(getattr(_planner_path, "path", []))
        my_path = parent_path + [action.name]
        _planner_path.path = my_path

        from zippergen.syntax import _ordered_workflow_lifelines
        if trace:
            trace({
                "type": "level_push",
                "name": action.name,
                "path": my_path,
                "lifelines": [ll.name for ll in _ordered_workflow_lifelines(wf)],
                "parent_seq": parent_seq,
            })

        def _inner_trace(event: dict) -> None:
            if trace:
                trace({**event, "path": my_path})
        wf._trace = _inner_trace

        # Extract inner workflow parameter names to build initial_envs.
        inner_params: list[str] = []
        try:
            for fn_node in _ast.walk(_ast.parse(spec)):
                if isinstance(fn_node, _ast.FunctionDef) and fn_node.name == "generated_workflow":
                    inner_params = [arg.arg for arg in fn_node.args.args]
                    break
        except SyntaxError:
            pass
        inputs_for_wf = {name: named_inputs[name] for name in inner_params if name in named_inputs}

        def _inner_backend(act_node, inp):
            t = _threading.current_thread()
            saved, t.name = t.name, outer_lifeline_name
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
