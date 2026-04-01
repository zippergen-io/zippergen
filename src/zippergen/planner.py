"""
ZipperGen — Planner subsystem.

Handles LLM-driven dynamic workflow generation (PlannerAction). Given a
PlannerAction node, _exec_planner builds a system prompt from the action's
vocabulary and constraints, calls the LLM to generate a ZipperGen workflow
spec, validates it with _validate_planner_spec, writes the spec to a temp
file, imports it as a module, and runs the resulting Workflow. All planner
prompt templates and structural validators live here.
"""

from __future__ import annotations

import threading

from zippergen.syntax import (
    LLMAction, PureAction, PlannerAction,
)

__all__ = ["_exec_planner", "_validate_planner_spec"]

# ---------------------------------------------------------------------------
# Planner DSL prompt templates
#
# Each constant is a plain string injected into the LLM system prompt.
# Use {caller} wherever the coordinating lifeline's name should appear —
# it is filled in at runtime via .format(caller=outer_lifeline_name).
# ---------------------------------------------------------------------------

_PLANNER_DSL_RULES = """\
- Name the function exactly `generated_workflow`.
- Declare inputs as `name: str @ {caller}`; return type is `-> str`.
- Send variables between lifelines: `A(x, y) >> B(x, y)`. Both sides MUST list
  the same variables in the same order — the left side sends, the right side binds.
- Single-output action: `A: var = action(arg1, arg2)`.
- Multi-output action: `A: (var1, var2) = action(arg1, arg2)` — use tuple unpacking,
  never subscript (`var["key"]` is invalid).
- Action arguments may be variable names OR literals (string, int, or float): `A: r = add(x, 3.14)`.
  Use literals to inject known constants directly without a preceding action call.
  Match the literal type to the action's parameter type: use `2` or `2.0` for float parameters, `"text"` for str parameters.
- REQUIRED: the last statement before `return` MUST send the final result back to {caller}:
  `LastWorker(result) >> {caller}(result)`.
- Return: `return var @ {caller}`.
- The right-hand side of every `Lifeline: var = ...` must be a function call.
  Never write `Lifeline: result = some_var` — use `identity(some_var)` to pass
  a variable through unchanged.
- A lifeline cannot send a message to itself. `A(...) >> A(...)` is invalid.
  When joining results from multiple lifelines onto one, only forward variables
  that the joining lifeline has NOT itself produced. Its own action outputs are
  already in scope. Example: if Worker2 ran `Worker2: y = action(...)`,
  then `Worker2(y) >> Worker2(y)` is wrong — `y` is already there.
  Only send variables from OTHER lifelines: `Worker1(x) >> Worker2(x)`.
- Every action call must use the `Lifeline: var = action(...)` syntax.
  Never write a bare assignment like `var = action(...)` without the lifeline prefix.
- Do NOT include any import statements or Var/Lifeline declarations.
- A lifeline can only use variables it has explicitly received or that an action
  on that lifeline produced. Plan action outputs first: if a lifeline needs to
  send `feedback`, it must run an action whose `outputs=` includes `("feedback", str)`
  before the send. Do not invent variable names — every variable must have a clear source.
- To give a downstream worker access to original inputs, forward them explicitly:
  `Worker1(result, var) >> Worker2(result, var)`.
- Every branch of an `if` must send the final result to {caller} under the SAME
  binding name. The name on the RIGHT side of `>>` is what {caller} will use in
  `return`. Example — CORRECT (both branches bind to `result`):
      if cond @ Worker1:
          Worker1(final) >> {caller}(result)   # binds as `result`
      else:
          Worker1(draft) >> {caller}(result)   # also binds as `result`
      return result @ {caller}
  WRONG (different names → `return` cannot refer to both):
      if cond @ Worker1:
          Worker1(final) >> {caller}(final)    # ← final
      else:
          Worker1(draft) >> {caller}(draft)    # ← draft (different!)
      return draft @ {caller}                  # ← fails: draft not on all paths

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
    Worker1(draft, instructions) >> Worker2(draft, instructions)
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
    Worker1(draft, instructions) >> Worker2(draft, instructions)
    Worker2: feedback = critique(draft, instructions)
    Worker2(feedback) >> Worker1(feedback)
    Worker1: final = revise(draft, feedback, instructions)
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

Example — Worker1 drafts, Worker2 assesses and optionally sends feedback for revision:
(Worker2 produces BOTH the bool flag AND the feedback string in one action.
 Both branches bind the final value to `result` on the {caller} side.)

    Worker1: draft = write(text, instructions)
    Worker1(draft, instructions) >> Worker2(draft, instructions)
    Worker2: (needs_revision, feedback) = assess(draft, instructions)
    if needs_revision @ Worker2:
        Worker2(draft, feedback) >> Worker1(draft, feedback)
        Worker1: revised = revise(draft, feedback)
        Worker1(revised) >> {caller}(result)   # bind as `result`
    else:
        Worker2(draft) >> {caller}(result)     # also bind as `result`
    return result @ {caller}
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


# ---------------------------------------------------------------------------
# Source reconstruction helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Spec helpers
# ---------------------------------------------------------------------------

def _extract_intermediate_var_names(spec: str) -> set[str]:
    """Extract variable names that need to be declared as Var in the preamble.

    Two sources:
    - AnnAssign annotations: ``Worker1: summary = f(x)`` → ``summary``
    - RHS bindings of >> sends: ``A(x) >> B(result)`` → ``result``
      These may differ from the sender's variable name and would not appear
      in any annotation, so they must be collected separately.
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
            # Action output annotations: Worker1: (a, b) = f(x)
            if isinstance(node, _ast.AnnAssign):
                ann = node.annotation
                if isinstance(ann, _ast.Tuple):
                    for elt in ann.elts:
                        if isinstance(elt, _ast.Name):
                            var_names.add(elt.id)
                elif isinstance(ann, _ast.Name):
                    var_names.add(ann.id)
            # RHS bindings of >> sends: A(x) >> B(result)
            elif (isinstance(node, _ast.Expr)
                  and isinstance(node.value, _ast.BinOp)
                  and isinstance(node.value.op, _ast.RShift)):
                rhs = node.value.right
                if isinstance(rhs, _ast.Call):
                    for arg in rhs.args:
                        if isinstance(arg, _ast.Name):
                            var_names.add(arg.id)
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

    # --- Invariant 5: both sides of every >> send have the same argument count,
    #                  and no lifeline sends to itself ---
    for node in _ast.walk(tree):
        if not (isinstance(node, _ast.Expr) and isinstance(node.value, _ast.BinOp)
                and isinstance(node.value.op, _ast.RShift)):
            continue
        lhs, rhs = node.value.left, node.value.right
        if (isinstance(lhs, _ast.Call) and isinstance(rhs, _ast.Call)):
            lname = lhs.func.id if isinstance(lhs.func, _ast.Name) else "?"
            rname = rhs.func.id if isinstance(rhs.func, _ast.Name) else "?"
            if lname == rname:
                return (
                    f"`{lname}(...) >> {rname}(...)` is a self-send — a lifeline cannot "
                    f"send messages to itself. Variables produced by actions on `{lname}` "
                    f"are already in scope and can be used directly in subsequent steps."
                )
            if len(lhs.args) != len(rhs.args):
                return (
                    f"`{lname}(...)  >> {rname}(...)` has mismatched argument counts "
                    f"({len(lhs.args)} vs {len(rhs.args)}). "
                    f"Both sides must list the same variables: `{lname}(x, y) >> {rname}(x, y)`."
                )

    # --- Invariant 5b: no bare or non-action assignments ---
    # AnnAssign with non-Call RHS (e.g. `Calculator: result = product`) compiles
    # as an assignment to the lifeline name, making it a local variable and
    # causing UnboundLocalError.  Plain Assign (e.g. `result = action(...)`)
    # silently drops the lifeline context.
    for node in _ast.walk(tree):
        if isinstance(node, _ast.AnnAssign) and node.value is not None:
            if not isinstance(node.value, _ast.Call):
                target = node.target.id if isinstance(node.target, _ast.Name) else "?"
                ann    = node.annotation.id if isinstance(node.annotation, _ast.Name) else "?"
                return (
                    f"`{target}: {ann} = <non-call>` is not a valid action call. "
                    f"The right-hand side must always be a function call like `action(args)`. "
                    f"To pass a variable under a different name use binding sides of >>: "
                    f"`{target}({ann}) >> OtherLifeline(new_name)`."
                )

    # --- Invariant 5d: no bare assignments (missing Lifeline: prefix) ---
    import ast as _ast2
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Assign):
            # Plain `x = expr(...)` inside the workflow body — missing lifeline prefix
            if isinstance(node.value, _ast.Call):
                fn = node.value.func
                fn_name = fn.id if isinstance(fn, _ast.Name) else "?"
                targets = ", ".join(
                    t.id for t in node.targets if isinstance(t, _ast.Name)
                )
                return (
                    f"Bare assignment `{targets} = {fn_name}(...)` found — "
                    f"every action call must use the `Lifeline: {targets} = {fn_name}(...)` syntax."
                )

    # --- Invariant 6: per-lifeline variable scope ---
    # Each lifeline may only use variables it has explicitly received (via >> or as
    # function outputs).  Seed caller's scope from the declared function parameters.
    scope: dict[str, set[str]] = {}
    for arg in fn_node.args.args:
        scope.setdefault(caller, set()).add(arg.arg)

    def _check_scopes(stmts: list, sc: dict[str, set[str]]) -> str | None:
        for stmt in stmts:
            # `LifelineName: output = action(args)`  →  AnnAssign
            if isinstance(stmt, _ast.AnnAssign) and isinstance(stmt.target, _ast.Name):
                ll = stmt.target.id
                ann = stmt.annotation
                if isinstance(ann, _ast.Name):
                    outputs: list[str] = [ann.id]
                elif isinstance(ann, _ast.Tuple):
                    outputs = [e.id for e in ann.elts if isinstance(e, _ast.Name)]
                else:
                    outputs = []
                if isinstance(stmt.value, _ast.Call):
                    for arg in stmt.value.args:
                        if isinstance(arg, _ast.Name) and arg.id not in sc.get(ll, set()):
                            fn_name = (stmt.value.func.id
                                       if isinstance(stmt.value.func, _ast.Name) else "?")
                            return (
                                f"`{ll}: ... = {fn_name}(...)` uses `{arg.id}` "
                                f"but `{ll}` has not received it. "
                                f"Forward it explicitly: "
                                f"`Sender({arg.id}, ...) >> {ll}({arg.id}, ...)`."
                            )
                sc.setdefault(ll, set()).update(outputs)

            # `A(vars) >> B(vars)`  →  Expr with RShift BinOp
            elif (isinstance(stmt, _ast.Expr)
                  and isinstance(stmt.value, _ast.BinOp)
                  and isinstance(stmt.value.op, _ast.RShift)):
                lhs2, rhs2 = stmt.value.left, stmt.value.right
                if isinstance(lhs2, _ast.Call) and isinstance(lhs2.func, _ast.Name):
                    sender = lhs2.func.id
                    for arg in lhs2.args:
                        if isinstance(arg, _ast.Name) and arg.id not in sc.get(sender, set()):
                            recv_name = (rhs2.func.id
                                         if isinstance(rhs2, _ast.Call)
                                         and isinstance(rhs2.func, _ast.Name) else "?")
                            return (
                                f"`{sender}({arg.id}) >> {recv_name}(...)` sends `{arg.id}` "
                                f"but `{sender}` has not received it."
                            )
                if isinstance(rhs2, _ast.Call) and isinstance(rhs2.func, _ast.Name):
                    receiver = rhs2.func.id
                    for arg in rhs2.args:
                        if isinstance(arg, _ast.Name):
                            sc.setdefault(receiver, set()).add(arg.id)

            # `if cond @ Owner:`
            elif isinstance(stmt, _ast.If):
                sc_true  = {k: set(v) for k, v in sc.items()}
                sc_false = {k: set(v) for k, v in sc.items()}
                err = _check_scopes(stmt.body,   sc_true)
                if err:
                    return err
                err = _check_scopes(stmt.orelse, sc_false)
                if err:
                    return err
                # Intersection: a var is only guaranteed available after the if
                # if it was sent to that lifeline in EVERY branch.
                for ll in set(sc_true) | set(sc_false):
                    sc[ll] = sc_true.get(ll, set()) & sc_false.get(ll, set())

            # `while cond @ Owner:`
            elif isinstance(stmt, _ast.While):
                # Check body for correctness (body vars are reachable from sc_before).
                sc_body = {k: set(v) for k, v in sc.items()}
                err = _check_scopes(stmt.body, sc_body)
                if err:
                    return err
                # Exit body always runs with scope-before-loop (loop may execute
                # 0 times), so body-only vars are NOT guaranteed available inside it.
                sc_exit = {k: set(v) for k, v in sc.items()}
                err = _check_scopes(stmt.orelse, sc_exit)
                if err:
                    return err
                # After the while: only exit-body additions are guaranteed.
                for ll in set(sc_exit):
                    sc[ll] = sc_exit.get(ll, set())

        return None

    scope_err = _check_scopes(fn_node.body, scope)
    if scope_err:
        return scope_err

    # --- Invariant 6b: return variable must be in caller's scope on ALL paths ---
    ret_node = stmts[-1]
    if (isinstance(ret_node, _ast.Return)
            and isinstance(ret_node.value, _ast.BinOp)
            and isinstance(ret_node.value.left, _ast.Name)):
        ret_var = ret_node.value.left.id
        if ret_var not in scope.get(caller, set()):
            return (
                f"`return {ret_var} @ {caller}`: `{ret_var}` is not available on "
                f"all control-flow paths. Every branch of every `if` must send the "
                f"same variable name to `{caller}` before the return, e.g.: "
                f"`Worker(result) >> {caller}(result)` in every branch."
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

    # --- Validate structural invariants; re-prompt up to max_retries times ---
    known_actions = {a.name for a in action.actions}
    for attempt in range(action.max_retries):
        validation_error = _validate_planner_spec(spec, outer_lifeline_name, known_actions)
        if validation_error is None:
            break
        self_send_hint = (
            "\nHint for self-send errors: a lifeline already has every variable it "
            "produced via its own actions — do NOT forward those back to itself. "
            "Only send variables that originated on a DIFFERENT lifeline.\n"
            if "self-send" in validation_error else ""
        )
        correction_prompt = (
            f"The workflow you generated has an error:\n\n"
            f"  {validation_error}\n"
            f"{self_send_hint}\n"
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
            name=f"_generate_spec_retry{attempt + 1}",
            inputs=(),
            outputs=(("workflow_spec", str),),
            system_prompt=system,
            user_prompt=correction_prompt_safe,
            parse_format="text",
        )
        spec_result = llm_backend(retry_action, {})
        spec = _strip_fences(str(spec_result.get("workflow_spec", "")))
    else:
        validation_error = _validate_planner_spec(spec, outer_lifeline_name, known_actions)
        if validation_error:
            raise RuntimeError(
                f"Planner generated an invalid workflow after {action.max_retries} correction attempts.\n"
                f"Error: {validation_error}\n\n"
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
