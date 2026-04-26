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
# _PLANNER_DSL_RULES uses {input_sig} (filled at runtime) for the signature template.
# ---------------------------------------------------------------------------

_PLANNER_DSL_RULES = """\
- Name the function exactly `generated_workflow`.
- Declare the function signature as: `def generated_workflow({input_sig}) -> str:`
  The parameter names and types are fixed. Annotate each parameter with the lifeline
  that should hold it at workflow start: `name: type @ Lifeline`. Inputs may be
  distributed across different lifelines — no initial forwarding step is needed.
- Send variables between lifelines: `A(x, y) >> B(x, y)`. Both sides MUST list
  the same variables in the same order — the left side sends, the right side binds.
- Single-output action: `A: var = action(arg1, arg2)`.
- Multi-output action: `A: (var1, var2) = action(arg1, arg2)` — use tuple unpacking,
  never subscript (`var["key"]` is invalid).
- Action arguments may be variable names OR literals (string, int, or float): `A: r = add(x, 3.14)`.
  Use literals to inject known constants directly without a preceding action call.
  Match the literal type to the action's parameter type: use `2` or `2.0` for float parameters, `"text"` for str parameters.
- Return: `return var @ Lifeline` where `var` is in `Lifeline`'s scope at that point.
  The result may live on any lifeline.
- The right-hand side of every `Lifeline: var = ...` must be a function call.
  Never write `Lifeline: result = some_var` — use `identity(some_var)` to pass
  a variable through unchanged.
- A lifeline's own action outputs are immediately available for subsequent actions on
  that same lifeline — no self-send needed. Example: after
      Calculator1: x = add(a, b)
      Calculator1: y = subtract(c, d)
  you can directly write `Calculator1: z = multiply(x, y)` — both `x` and `y` are
  already in Calculator1's scope. Never write `Calculator1(x, y) >> Calculator1(x, y)`
  to "collect" variables the lifeline already has.
- Self-sends `A(x) >> A(y)` are allowed and act as local variable assignments (y := x).
  The left-side and right-side variable names must be distinct — `A(x) >> A(x)` is
  a no-op and invalid. Use self-sends only to rename a variable from another lifeline:
  `Worker1(result) >> Worker1(renamed)`.
  When joining results from multiple workers, only forward variables from OTHER lifelines:
  if Worker2 already produced `y`, use `Worker1(x) >> Worker2(x)` not `Worker2(y) >> Worker2(y)`.
- Every action call must use the `Lifeline: var = action(...)` syntax.
  Never write a bare assignment like `var = action(...)` without the lifeline prefix.
- Do NOT include any import statements or Var/Lifeline declarations.
- A lifeline can only use variables it has explicitly received or that an action
  on that lifeline produced. Plan action outputs first: if a lifeline needs to
  send `feedback`, it must run an action whose `outputs=` includes `("feedback", str)`
  before the send. Do not invent variable names — every variable must have a clear source.
- To give a downstream worker access to original inputs, forward them explicitly:
  `Worker1(result, var) >> Worker2(result, var)`.
- Output variable names (left of `=`) must NOT match any action name. Rename to avoid
  the collision: e.g. `is_zero_result` instead of `is_zero`, `add_result` instead of `add`.
- Every input parameter MUST appear in at least one action call or send. A parameter
  that is never used is an error.
- There is NO early return. `return` may only appear as the very last statement of the
  workflow, never inside an `if` or `while` branch. If you want to short-circuit on an
  error condition, put ALL subsequent computation inside the `else` branch:
      if error_condition @ Owner:
          Owner: result = error_value()   # produce a sentinel result
      else:
          # ALL remaining work goes here
          ...
          # result must be forwarded/produced on the same lifeline as in the true branch
      return result @ Owner
- Every branch of an `if` must produce the return variable under the SAME name on the
  SAME lifeline, so that `return` can refer to it on all paths. Example — CORRECT:
      if cond @ Worker1:
          Worker1: result = option_a(draft)
      else:
          Worker1: result = option_b(draft)
      return result @ Worker1
  WRONG (different names — `return` cannot refer to both):
      if cond @ Worker1:
          Worker1: final = option_a(draft)   # ← final
      else:
          Worker1: summary = option_b(draft) # ← summary (different!)
      return ???                              # ← fails: no single name on all paths

Example — sequential with handoff (Worker1 drafts, Worker2 refines):

@workflow
def generated_workflow(text: str @ Worker1, instructions: str @ Worker1) -> str:
    Worker1: draft = write(text, instructions)
    Worker1(draft) >> Worker2(draft)
    Worker2: final = refine(draft, instructions)
    return final @ Worker2

Example — three-worker chain (Worker1 drafts, Worker2 critiques, Worker3 polishes):

@workflow
def generated_workflow(text: str @ Worker1, instructions: str @ Worker1) -> str:
    Worker1: draft = write(text, instructions)
    Worker1(draft, instructions) >> Worker2(draft, instructions)
    Worker2: feedback = critique(draft, instructions)
    Worker2(draft, feedback) >> Worker3(draft, feedback)
    Worker3: final = polish(draft, feedback)
    return final @ Worker3

Example — back-and-forth (Worker1 drafts, Worker2 critiques, Worker1 revises):

@workflow
def generated_workflow(text: str @ Worker1, instructions: str @ Worker1) -> str:
    Worker1: draft = write(text, instructions)
    Worker1(draft, instructions) >> Worker2(draft, instructions)
    Worker2: feedback = critique(draft, instructions)
    Worker2(feedback) >> Worker1(feedback)
    Worker1: final = revise(draft, feedback, instructions)
    return final @ Worker1

Example — parallel then aggregate (each worker gets its own input directly):

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

IMPORTANT — no early return: `return` is always the last statement of the whole workflow.
If a branch detects an error and you want to skip remaining computation, put ALL
downstream computation inside the `else` branch, and produce a sentinel result in the
`if` branch. Example — divide by zero guard:

    Worker: denominator = compute_denominator(x)
    Worker: is_zero = is_zero(denominator)
    if is_zero @ Worker:
        Worker: result = error_value()   # sentinel; all further work is skipped
    else:
        Worker: result = divide(y, denominator)
        # ... any further computation on other lifelines goes HERE inside else
    return result @ Worker

Example — Worker1 drafts, Worker2 assesses and optionally sends feedback for revision:
(Worker2 produces BOTH the bool flag AND the feedback string in one action.
 Both branches produce `result` on Worker2.)

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
        pass
    return draft @ Worker1
"""


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
    spec: str, caller: str, known_actions: dict[str, int] | None = None,
    expected_input_types: dict[str, type] | None = None,
) -> str | None:
    """Check structural invariants on a generated workflow.

    Returns None if the spec is valid, or a human-readable error string
    describing the first violation found.

    Invariants:
    1. Last statement is ``return var @ Lifeline``.
    2. All called actions are defined.
    2b. Output count of each call matches the action's declaration.
    3. Every >> send has matching argument counts; self-sends have disjoint variable names.
    4. No bare assignments; every action call uses ``Lifeline: var = action(...)`` syntax.
    5. Per-lifeline variable scope is respected (seeded from signature annotations).
    6. Parameter types in generated_workflow signature match expected input types.

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

    # --- Invariant 7: parameter names and types match expected inputs exactly ---
    if expected_input_types:
        _TYPE_MAP = {"str": str, "int": int, "bool": bool, "float": float, "tuple": tuple}
        declared_params = {arg.arg for arg in fn_node.args.args}
        expected_params = set(expected_input_types.keys())

        missing = expected_params - declared_params
        if missing:
            expected_sig = ", ".join(
                f"{n}: {t.__name__} @ <Lifeline>"
                for n, t in expected_input_types.items()
            )
            return (
                f"Generated workflow is missing parameter(s): {', '.join(sorted(missing))}. "
                f"The signature must include: def generated_workflow({expected_sig}) -> str:"
            )

        extra = declared_params - expected_params
        if extra:
            expected_sig = ", ".join(
                f"{n}: {t.__name__} @ <Lifeline>"
                for n, t in expected_input_types.items()
            )
            return (
                f"Generated workflow has unexpected parameter(s): {', '.join(sorted(extra))}. "
                f"The signature must include: def generated_workflow({expected_sig}) -> str:"
            )

        for arg in fn_node.args.args:
            name = arg.arg
            ann = arg.annotation
            # Annotations have the form `type @ Lifeline` (BinOp MatMult) or plain `type`
            if isinstance(ann, _ast.BinOp) and isinstance(ann.op, _ast.MatMult):
                type_node = ann.left
            else:
                type_node = ann
            if not isinstance(type_node, _ast.Name):
                continue
            declared = _TYPE_MAP.get(type_node.id)
            expected = expected_input_types[name]
            if declared is not None and declared is not expected:
                return (
                    f"Parameter `{name}` is declared as `{type_node.id}` but the actual "
                    f"input type is `{expected.__name__}`. "
                    f"Use `{name}: {expected.__name__} @ <Lifeline>`."
                )

    body = fn_node.body
    # Filter to just expression/return statements (skip docstrings etc.)
    stmts = [s for s in body if isinstance(s, (_ast.Expr, _ast.Return))]
    if not stmts:
        return "Workflow body is empty — must contain at least one action and a return statement."

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

    # --- Invariant: last statement is `return var @ SomeLifeline` ---
    last = stmts[-1]
    ret_owner = return_at_name(last)
    if ret_owner is None:
        return "Last statement must be `return var @ Lifeline` (no `return` found or wrong form)."

    # --- Invariant 4: all action calls are defined or pre-defined ---
    # Collect names called as actions: `A: var = name(...)` or `A: (v1, v2) = name(...)`
    import re as _re
    called_names = set(_re.findall(
        r':\s+(?:\w+|\([^)]+\))\s*=\s*(\w+)\s*\(', spec
    ))
    defined_names = set(_re.findall(r'^def\s+(\w+)', spec, _re.MULTILINE))
    undefined = called_names - defined_names - set(known_actions or {})
    if undefined:
        return (
            f"The workflow calls actions that are not defined: {', '.join(sorted(undefined))}. "
            f"Each must be defined as an @llm or @pure action before the workflow function."
        )

    # --- Invariant 4b: output variable names must not collide with action names ---
    # A collision causes the Var declaration in the preamble to shadow the action,
    # making the action reference resolve to a Var object at builder time.
    all_action_names = (set(known_actions or {})) | (defined_names - {"generated_workflow"})
    for node in _ast.walk(fn_node):
        if not isinstance(node, _ast.AnnAssign):
            continue
        ann = node.annotation
        clashing = []
        if isinstance(ann, _ast.Name) and ann.id in all_action_names:
            clashing.append(ann.id)
        elif isinstance(ann, _ast.Tuple):
            clashing = [e.id for e in ann.elts
                        if isinstance(e, _ast.Name) and e.id in all_action_names]
        for name in clashing:
            return (
                f"Output variable `{name}` has the same name as action `{name}`. "
                f"Rename the output variable to avoid the collision — "
                f"e.g. use `{name}_result` or `{name}_val` instead."
            )

    # --- Invariant 3b: output count matches action declaration ---
    if known_actions:
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.AnnAssign):
                continue
            if not isinstance(node.value, _ast.Call):
                continue
            fn = node.value.func
            if not isinstance(fn, _ast.Name) or fn.id not in known_actions:
                continue
            fn_name = fn.id
            expected = known_actions[fn_name]
            ann = node.annotation
            if isinstance(ann, _ast.Name):
                actual = 1
            elif isinstance(ann, _ast.Tuple):
                actual = len([e for e in ann.elts if isinstance(e, _ast.Name)])
            else:
                continue
            if actual != expected:
                lifeline = node.target.id if isinstance(node.target, _ast.Name) else "?"
                lhs_example = (
                    "result" if expected == 1
                    else "(" + ", ".join(f"out{i+1}" for i in range(expected)) + ")"
                )
                return (
                    f"`{lifeline}: {_ast.unparse(ann)} = {fn_name}(...)` "
                    f"unpacks {actual} output{'s' if actual != 1 else ''} "
                    f"but `{fn_name}` returns {expected}. "
                    f"Use: `{lifeline}: {lhs_example} = {fn_name}(...)`."
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
                # Self-send: check that left-side and right-side var names are disjoint.
                lhs_vars = {arg.id for arg in lhs.args if isinstance(arg, _ast.Name)}
                rhs_vars = {arg.id for arg in rhs.args if isinstance(arg, _ast.Name)}
                overlap = lhs_vars & rhs_vars
                if overlap:
                    return (
                        f"`{lname}({', '.join(sorted(overlap))}) >> {rname}({', '.join(sorted(overlap))})` "
                        f"is a no-op self-send: the same variable appears on both sides. "
                        f"Use distinct names — `{lname}(x) >> {lname}(y)` assigns x to y — "
                        f"or remove it if the variable is already in scope."
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
        ann = arg.annotation
        if (isinstance(ann, _ast.BinOp) and isinstance(ann.op, _ast.MatMult)
                and isinstance(ann.right, _ast.Name)):
            owner = ann.right.id
        else:
            owner = caller  # fallback if annotation has no lifeline
        scope.setdefault(owner, set()).add(arg.arg)

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

    # --- Invariant 6b: return variable must be in the return lifeline's scope on ALL paths ---
    ret_node = stmts[-1]
    if (isinstance(ret_node, _ast.Return)
            and isinstance(ret_node.value, _ast.BinOp)
            and isinstance(ret_node.value.left, _ast.Name)
            and isinstance(ret_node.value.right, _ast.Name)):
        ret_var = ret_node.value.left.id
        ret_ll  = ret_node.value.right.id
        if ret_var not in scope.get(ret_ll, set()):
            return (
                f"`return {ret_var} @ {ret_ll}`: `{ret_var}` is not available on "
                f"`{ret_ll}` on all control-flow paths. Every branch of every `if` "
                f"must produce or forward `{ret_var}` to `{ret_ll}` before the return."
            )

    # --- Invariant 7: every input parameter must be used at least once ---
    # Collect variable names that appear as arguments in action calls or sends.
    used_in_body: set[str] = set()
    for node in _ast.walk(fn_node):
        if isinstance(node, _ast.AnnAssign) and isinstance(node.value, _ast.Call):
            for arg in node.value.args:
                if isinstance(arg, _ast.Name):
                    used_in_body.add(arg.id)
        elif (isinstance(node, _ast.Expr)
              and isinstance(node.value, _ast.BinOp)
              and isinstance(node.value.op, _ast.RShift)):
            lhs = node.value.left
            if isinstance(lhs, _ast.Call):
                for arg in lhs.args:
                    if isinstance(arg, _ast.Name):
                        used_in_body.add(arg.id)
    for arg in fn_node.args.args:
        if arg.arg not in used_in_body:
            return (
                f"Input parameter `{arg.arg}` is declared but never used. "
                f"Every input must appear in at least one action call or send. "
                f"Pass it to an action: `Lifeline: out = action({arg.arg}, ...)`, "
                f"or forward it: `Lifeline({arg.arg}) >> Other({arg.arg})`."
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

    lifeline_lines = []
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
            "Conditional branching:\n" + _PLANNER_ALLOW_IF
        )
    if "while" in action.allow:
        control_flow_sections.append(
            "Loops:\n" + _PLANNER_ALLOW_WHILE
        )

    system_parts = [
        action.system_prompt,
        worker_desc,
        "DSL rules:\n" + _PLANNER_DSL_RULES.format(
            caller=outer_lifeline_name,
            input_sig=", ".join(
                f"{name}: {t.__name__}"
                for name, t in action.inputs
                if name != "request"
            ),
        ),
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
    input_types = {name: t for name, t in action.inputs if name != "request"}
    _PREVIEW_LEN = 120
    def _preview(v: object) -> str:
        s = str(v)
        return s if len(s) <= _PREVIEW_LEN else s[:_PREVIEW_LEN] + "…"
    inputs_desc = "\n".join(
        f"- {k} ({input_types.get(k, type(v)).__name__}): {_preview(v)}"
        for k, v in inputs_data.items()
    )

    # --- 3. Call LLM to generate workflow spec ---
    # Pre-format the user content so .format() receives no template variables
    # (avoids breakage if request_text or inputs_desc contains literal braces).
    inputs_section = f"Input variables available:\n{inputs_desc}\n\n" if inputs_data else ""
    user_content = (
        f"Request: {request_text}\n\n"
        f"{inputs_section}"
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

    # --- Check that no worker shares the caller's name ---
    caller_conflicts = [n for n in worker_names if n == outer_lifeline_name]
    if caller_conflicts:
        raise ValueError(
            f"Planner worker lifeline(s) {caller_conflicts} share the name of the "
            f"calling lifeline '{outer_lifeline_name}'. Worker names must be distinct "
            f"from the caller."
        )

    # --- Validate structural invariants; re-prompt up to max_retries times ---
    known_actions = {a.name: len(a.outputs) for a in action.actions
                     if isinstance(a, (LLMAction, PureAction, PlannerAction))}
    attempts_used = 1
    for attempt in range(action.max_retries):
        validation_error = _validate_planner_spec(spec, outer_lifeline_name, known_actions, input_types)
        if validation_error is None:
            attempts_used = attempt + 1
            break
        print(f"[planner] attempt {attempt + 1} failed: {validation_error}")
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
            f"  1. Each parameter annotated with its owner lifeline: `name: type @ Lifeline`\n"
            f"  2. Last statement: `return var @ Lifeline` where var is in Lifeline's scope\n\n"
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
        validation_error = _validate_planner_spec(spec, outer_lifeline_name, known_actions, input_types)
        if validation_error:
            raise RuntimeError(
                f"Planner generated an invalid workflow after {action.max_retries} correction attempts.\n"
                f"Error: {validation_error}\n\n"
                f"Workflow:\n{spec}"
            )

    attempt_str = f"{attempts_used} attempt{'s' if attempts_used > 1 else ''}"
    print(f"\n{'='*60}")
    print(f"GENERATED WORKFLOW  ({attempt_str})")
    print("=" * 60)
    print(spec)
    print("=" * 60 + "\n")

    # --- 4. Build preamble for temp file ---
    # Action objects are injected directly into the module namespace after
    # import, so the preamble only needs Lifeline/Var declarations.
    # This supports arbitrary action types (llm, pure, planner, workflow)
    # without any serialisation.
    intermediate_vars = _extract_intermediate_var_names(spec)

    preamble_lines = [
        "from zippergen.syntax import Lifeline, Var",
        "from zippergen.builder import workflow",
        "from zippergen.actions import llm, pure, planner",
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

    preamble = "\n".join(preamble_lines) + "\n"
    full_source = preamble + "\n" + spec + "\n"

    # --- 5. Write to temp file, import, and inject action objects ---
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
        # Inject action objects directly — no serialisation needed, any action
        # type (including @workflow and nested @planner) works automatically.
        for a in action.actions:
            mod.__dict__[a.name] = a
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
