"""
Layer 3: @workflow decorator and DSL helpers. Rewrites native if/while/>> syntax
into IR builder calls via AST transformation.
"""

from __future__ import annotations

import ast
import inspect
import sys
import textwrap
from collections.abc import Callable
from typing import cast

from zippergen.syntax import (
    ZType, Lifeline, Var,
    ZTypeAtLifeline,
    Expr, VarExpr, LitExpr,
    Stmt, AnyStmt, EmptyStmt, MsgStmt, CoregionStmt, ActStmt, SkipStmt, SeqStmt, IfStmt, WhileStmt,
    ParallelStmt,
    LLMAction, PureAction, PlannerAction,
    Workflow,
    seq, is_ztype,
)

__all__ = [
    # Workflow decorator
    "workflow",
    # Fragment decorator
    "fragment",
    # Statement builders
    "msg", "act", "skip", "coregion", "parallel", "branch",
    "if_", "while_",
]


# ---------------------------------------------------------------------------
# Recording context
# ---------------------------------------------------------------------------

_stack: list[list[Stmt]] = []


class _CoregionMarker:
    def __repr__(self) -> str:
        return "coregion"


class _ParallelMarker:
    def __repr__(self) -> str:
        return "parallel"


class _BranchMarker:
    def __repr__(self) -> str:
        return "branch"


coregion = _CoregionMarker()
parallel = _ParallelMarker()
branch = _BranchMarker()


def _record(stmt: Stmt) -> None:
    """Append stmt to the innermost open scope."""
    if not _stack:
        raise RuntimeError(
            "Statement builders (msg, act, etc.) must be called "
            "inside a @workflow body or a nested branch."
        )
    _stack[-1].append(stmt)


def _collect(fn: Callable) -> AnyStmt:
    """
    Open a new scope, run fn(), close the scope, and return the
    collected statements folded into a single Stmt via seq().
    """
    _stack.append([])
    fn()
    stmts = _stack.pop()
    return seq(*stmts)


def _flatten_seq(stmt: AnyStmt) -> list[AnyStmt]:
    if isinstance(stmt, EmptyStmt):
        return []
    if isinstance(stmt, SeqStmt):
        return [*_flatten_seq(stmt.first), *_flatten_seq(stmt.second)]
    return [stmt]


# ---------------------------------------------------------------------------
# Expression builders  (pure — no recording)
# ---------------------------------------------------------------------------

def _to_expr(x: object) -> Expr:
    """Coerce a Var or Python literal to an Expr; pass through existing Expr."""
    if isinstance(x, Var):
        return VarExpr(x)
    if isinstance(x, bool):
        return LitExpr(x, bool)
    if isinstance(x, int):
        return LitExpr(x, int)
    if isinstance(x, float):
        return LitExpr(x, float)
    if isinstance(x, str):
        return LitExpr(x, str)
    return x  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Statement builders  (record into current scope, return None)
# ---------------------------------------------------------------------------

def msg(
    sender: Lifeline,
    payload: tuple,
    receiver: Lifeline,
    bindings: tuple,
) -> None:
    """msg sender(payload) → receiver(bindings)

    Vars are automatically wrapped in VarExpr; Expr values pass through.
    """
    _record(MsgStmt(
        sender,
        tuple(_to_expr(x) for x in payload),
        receiver,
        tuple(_to_expr(x) for x in bindings),
    ))


def act(
    lifeline: Lifeline,
    action: LLMAction | PureAction | PlannerAction,
    inputs: tuple,
    outputs: tuple[Var, ...],
) -> None:
    """act lifeline: outputs := action(inputs)

    Vars in inputs are automatically wrapped in VarExpr.
    """
    _record(ActStmt(
        lifeline,
        action,
        tuple(_to_expr(x) for x in inputs),
        tuple(outputs),
    ))


def skip(lifeline: Lifeline | None = None) -> None:
    """skip — ε (empty program). Optional lifeline records explicit participation."""
    if lifeline is not None:
        _record(SkipStmt(lifeline))


def coregion_(body: Callable) -> None:
    """Record a restricted co-region of unordered messages."""
    collected = _collect(body)
    stmts = _flatten_seq(collected)
    messages: list[MsgStmt] = []
    for stmt in stmts:
        if not isinstance(stmt, MsgStmt):
            raise TypeError(
                "coregion body may contain only message statements "
                f"(Sender(...) >> Receiver(...)), got {type(stmt).__name__}"
            )
        messages.append(stmt)
    _record(CoregionStmt(tuple(messages)))


def parallel_(*branches: Callable) -> None:
    """Record a first-class parallel region.

    Each argument is a zero-argument function whose body contains ordinary
    ZipperGen statements. The branches start together and the continuation runs
    after all branches have finished.
    """
    if not branches:
        raise ValueError("parallel requires at least one branch")
    _record(ParallelStmt(tuple(_collect(branch) for branch in branches)))


def if_(
    condition: Callable[..., bool],
    owner: Lifeline,
    *,
    then: Callable,
    else_: Callable,
) -> None:
    """
    if condition@owner then { then() } else { else_() }

    Prefer native ``if cond @ owner:`` syntax inside ``@workflow`` bodies.
    This function is the explicit fallback (e.g. for programmatic construction).
    """
    _record(IfStmt(
        condition,
        owner,
        _collect(then),
        _collect(else_),
    ))


def while_(
    condition: Callable[..., bool],
    owner: Lifeline,
    *,
    body: Callable,
    exit_body: Callable,
) -> None:
    """
    while condition@owner do { body() } exit { exit_body() }

    Prefer native ``while cond @ owner: ... else: ...`` inside ``@workflow``.
    This function is the explicit fallback (e.g. for programmatic construction).
    """
    _record(WhileStmt(
        condition,
        owner,
        _collect(body),
        _collect(exit_body),
    ))


# ---------------------------------------------------------------------------
# AST rewriting — translating native if/while to if_()/while_() calls
# ---------------------------------------------------------------------------

_gen_counter: int = 0


def _fresh(prefix: str) -> str:
    global _gen_counter
    _gen_counter += 1
    return f"__{prefix}_{_gen_counter}"


def _cond_ast(node: ast.expr) -> ast.expr:
    """
    Recursively rewrite a condition AST so that every ``Name('x')`` becomes
    ``Attribute(Name('_e'), 'x')`` — i.e. ``_e.x``.

    The result is used as the body of a ``lambda _e: <body>`` expression.
    At runtime ``_e`` is a ``_CondEnv`` proxy that resolves names against
    the lifeline's local env first, then the workflow's global namespace.
    """
    if isinstance(node, ast.Name):
        return ast.Attribute(
            value=ast.Name(id="_e", ctx=ast.Load()),
            attr=node.id,
            ctx=ast.Load(),
        )
    if isinstance(node, ast.UnaryOp):
        return ast.UnaryOp(op=node.op, operand=_cond_ast(node.operand))
    if isinstance(node, ast.BoolOp):
        return ast.BoolOp(op=node.op, values=[_cond_ast(v) for v in node.values])
    if isinstance(node, ast.Compare):
        return ast.Compare(
            left=_cond_ast(node.left),
            ops=node.ops,
            comparators=[_cond_ast(c) for c in node.comparators],
        )
    # Constants and anything else pass through unchanged.
    return node


def _make_cond_lambda(cond_node: ast.expr) -> ast.expr:
    """Wrap a rewritten condition AST in ``lambda _e: <cond>``."""
    return ast.Lambda(
        args=ast.arguments(
            posonlyargs=[], args=[ast.arg(arg="_e")], vararg=None,
            kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[],
        ),
        body=_cond_ast(cond_node),
    )


def _tag_cond(fn: Callable, src: str) -> Callable:
    """Attach a human-readable source string to a condition lambda."""
    fn._src = src  # type: ignore[attr-defined]
    return fn


def _make_fn(name: str, body: list[ast.stmt]) -> ast.FunctionDef:
    """Build a no-argument FunctionDef AST node."""
    args = ast.arguments(
        posonlyargs=[], args=[], vararg=None,
        kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[],
    )
    if sys.version_info >= (3, 12):
        return ast.FunctionDef(
            name=name, args=args, body=body or [ast.Pass()],
            decorator_list=[], returns=None, type_params=[],
            lineno=0, col_offset=0,
        )
    return ast.FunctionDef(
        name=name, args=args, body=body or [ast.Pass()],
        decorator_list=[], returns=None,
        lineno=0, col_offset=0,
    )


class _ProcTransformer(ast.NodeTransformer):
    """
    Rewrites coordination-DSL patterns into their builder-function equivalents:

    - ``Lifeline: outputs = action(inputs)``  →  ``act(Lifeline, action, inputs, outputs)``
    - ``with Lifeline:\n    y = f(x)\n    ...``  →  one ``act(...)`` call per body line
    - ``with coregion:\n    A(x) >> R(x)\n    ...`` → unordered messages to one receiver
    - ``with parallel:\n    with branch: ...`` → first-class parallel region
    - ``Sender(x, y) >> Receiver(a, b)``      →  ``msg(Sender, (x, y), Receiver, (a, b))``
    - ``if cond @ owner: ...``                →  ``if_(cond, owner, then=..., else_=...)``
    - ``while cond @ owner: ...``             →  ``while_(cond, owner, body=..., exit_body=...)``
    - ``return var``                          →  removed; ``var`` stored in ``self.return_var_name``

    All other Python statements pass through unchanged.
    """

    def __init__(self) -> None:
        super().__init__()
        self.return_outputs: list[tuple[str, str]] = []  # [(var_name, lifeline_name), ...]

    @staticmethod
    def _is_at(node: ast.expr) -> bool:
        return isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult)

    @staticmethod
    def _parse_msg_side(node: ast.expr, side: str) -> tuple[ast.expr, list[ast.expr]]:
        """Return (lifeline_ast, args) from a Call node ``Lifeline(x, y, ...)``."""
        if not isinstance(node, ast.Call):
            raise SyntaxError(
                f"Both sides of '>>' must be a lifeline call, "
                f"e.g. Sender(x) >> Receiver(y). "
                f"Got a non-call on the {side} side. "
                f"Use empty brackets for no payload: Sender() >> Receiver()."
            )
        return node.func, list(node.args)

    @staticmethod
    def _make_act_call(lifeline_ast, action_ast, args, outputs):
        """Build an ``ast.Expr`` for ``act(lifeline, action, inputs, outputs)``."""
        return ast.Expr(ast.Call(
            func=ast.Name(id="act", ctx=ast.Load()),
            args=[
                lifeline_ast,
                action_ast,
                ast.Tuple(elts=list(args), ctx=ast.Load()),
                ast.Tuple(elts=list(outputs), ctx=ast.Load()),
            ],
            keywords=[],
        ))

    def visit_Return(self, node: ast.Return) -> ast.stmt:
        """Strip ``return var @ Lifeline`` or ``return (v1 @ L1, v2 @ L2, ...)``."""
        val = node.value
        if val is None:
            raise SyntaxError(
                "return in @workflow must have the form  return var @ Lifeline"
            )

        def _parse_one(elt: ast.expr) -> tuple[str, str]:
            if (isinstance(elt, ast.BinOp) and isinstance(elt.op, ast.MatMult)
                    and isinstance(elt.left, ast.Name) and isinstance(elt.right, ast.Name)):
                return elt.left.id, elt.right.id
            raise SyntaxError(
                "return in @workflow must have the form  return var @ Lifeline  "
                "or  return (var1 @ Lifeline1, var2 @ Lifeline2, ...)"
            )

        if isinstance(val, ast.Tuple):
            self.return_outputs = [_parse_one(elt) for elt in val.elts]
        else:
            self.return_outputs = [_parse_one(val)]
        return ast.Pass()

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.stmt:
        """Rewrite ``Lifeline: outputs = action(inputs)`` to ``act(...)``."""
        self.generic_visit(node)
        if node.value is None or not isinstance(node.value, ast.Call):
            return node
        # Target has Store context; rebuild as Load for use as a function arg.
        target = node.target
        lifeline_ast = (
            ast.Name(id=target.id, ctx=ast.Load())
            if isinstance(target, ast.Name)
            else target
        )
        action_call = node.value
        ann = node.annotation
        outputs = list(ann.elts) if isinstance(ann, ast.Tuple) else [ann]
        return self._make_act_call(lifeline_ast, action_call.func, action_call.args, outputs)

    def visit_With(self, node: ast.With) -> ast.stmt | list[ast.stmt]:
        """
        Rewrite a ``with Lifeline:`` block into one ``act(...)`` call per line.

        Rewrite ``with coregion:`` into a restricted unordered message block.

        Only single-item ``with`` blocks without an ``as`` clause whose every
        body statement has the form ``outputs = action(inputs)`` are matched.
        Anything else passes through unchanged (let Python handle it normally).
        """
        # Must be a bare `with Lifeline:` — no `as var`, no multiple items.
        if len(node.items) != 1 or node.items[0].optional_vars is not None:
            self.generic_visit(node)
            return node

        lifeline_ast = node.items[0].context_expr
        if isinstance(lifeline_ast, ast.Name) and lifeline_ast.id == "coregion":
            self.generic_visit(node)
            body_name = _fresh("coregion")
            body_fn = _make_fn(body_name, node.body)
            call = ast.Expr(ast.Call(
                func=ast.Name(id="coregion_", ctx=ast.Load()),
                args=[ast.Name(id=body_name, ctx=ast.Load())],
                keywords=[],
            ))
            return [body_fn, call]

        if isinstance(lifeline_ast, ast.Name) and lifeline_ast.id == "parallel":
            branch_fns: list[ast.FunctionDef] = []
            branch_names: list[str] = []
            for stmt in node.body:
                if not (
                    isinstance(stmt, ast.With)
                    and len(stmt.items) == 1
                    and stmt.items[0].optional_vars is None
                    and isinstance(stmt.items[0].context_expr, ast.Name)
                    and stmt.items[0].context_expr.id == "branch"
                ):
                    raise SyntaxError(
                        "with parallel: body may contain only with branch: blocks"
                    )
                branch_name = _fresh("branch")
                branch_names.append(branch_name)
                branch_fns.append(_make_fn(branch_name, stmt.body))

            for branch_fn in branch_fns:
                self.generic_visit(branch_fn)

            call = ast.Expr(ast.Call(
                func=ast.Name(id="parallel_", ctx=ast.Load()),
                args=[ast.Name(id=name, ctx=ast.Load()) for name in branch_names],
                keywords=[],
            ))
            return [*branch_fns, call]

        result: list[ast.stmt] = []

        for stmt in node.body:
            if not (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.value, ast.Call)
            ):
                # Unrecognised body statement — fall through to normal Python.
                self.generic_visit(node)
                return node

            target = stmt.targets[0]
            action_call = stmt.value
            # Assign targets have Store context; rewrite to Load for use as args.
            def _to_load(n: ast.expr) -> ast.expr:
                if isinstance(n, ast.Name):
                    return ast.Name(id=n.id, ctx=ast.Load())
                return n
            outputs = [_to_load(e) for e in target.elts] if isinstance(target, ast.Tuple) else [_to_load(target)]
            result.append(self._make_act_call(lifeline_ast, action_call.func, action_call.args, outputs))

        return result if result else [ast.Pass()]

    def visit_Expr(self, node: ast.Expr) -> ast.Expr:
        """Rewrite ``Sender(payload) >> Receiver(bindings)`` to ``msg(...)``."""
        self.generic_visit(node)
        val = node.value
        if not (isinstance(val, ast.BinOp) and isinstance(val.op, ast.RShift)):
            return node
        sender_ast, payload = self._parse_msg_side(val.left,  "left (sender)")
        receiver_ast, bindings = self._parse_msg_side(val.right, "right (receiver)")
        return ast.Expr(ast.Call(
            func=ast.Name(id="msg", ctx=ast.Load()),
            args=[
                sender_ast,
                ast.Tuple(elts=payload,   ctx=ast.Load()),
                receiver_ast,
                ast.Tuple(elts=bindings,  ctx=ast.Load()),
            ],
            keywords=[],
        ))

    def visit_If(self, node: ast.If) -> ast.stmt | list[ast.stmt]:
        # Recurse into children first (bottom-up).
        self.generic_visit(node)

        if not self._is_at(node.test):
            return node

        cond_src = ast.unparse(node.test.left)      # type: ignore[arg-type]
        cond = ast.Call(
            func=ast.Name(id="_tag_cond", ctx=ast.Load()),
            args=[_make_cond_lambda(node.test.left), ast.Constant(value=cond_src)],  # type: ignore[arg-type]
            keywords=[],
        )
        owner = node.test.right                     # type: ignore[union-attr]

        then_name = _fresh("then")
        else_name = _fresh("else")
        then_fn = _make_fn(then_name, node.body)
        else_fn = _make_fn(else_name, node.orelse or [ast.Pass()])

        call = ast.Expr(ast.Call(
            func=ast.Name(id="if_", ctx=ast.Load()),
            args=[cond, owner],
            keywords=[
                ast.keyword(arg="then",  value=ast.Name(id=then_name, ctx=ast.Load())),
                ast.keyword(arg="else_", value=ast.Name(id=else_name, ctx=ast.Load())),
            ],
        ))
        return [then_fn, else_fn, call]

    def visit_While(self, node: ast.While) -> ast.stmt | list[ast.stmt]:
        # Recurse into children first (bottom-up).
        self.generic_visit(node)

        if not self._is_at(node.test):
            return node

        cond_src = ast.unparse(node.test.left)      # type: ignore[arg-type]
        cond = ast.Call(
            func=ast.Name(id="_tag_cond", ctx=ast.Load()),
            args=[_make_cond_lambda(node.test.left), ast.Constant(value=cond_src)],  # type: ignore[arg-type]
            keywords=[],
        )
        owner = node.test.right                     # type: ignore[union-attr]

        body_name = _fresh("body")
        exit_name = _fresh("exit")
        body_fn = _make_fn(body_name, node.body)
        exit_fn = _make_fn(exit_name, node.orelse or [ast.Pass()])

        call = ast.Expr(ast.Call(
            func=ast.Name(id="while_", ctx=ast.Load()),
            args=[cond, owner],
            keywords=[
                ast.keyword(arg="body",      value=ast.Name(id=body_name, ctx=ast.Load())),
                ast.keyword(arg="exit_body", value=ast.Name(id=exit_name, ctx=ast.Load())),
            ],
        ))
        return [body_fn, exit_fn, call]


def _transform_proc_source(fn: Callable) -> tuple[Callable, list[tuple[str, str]]]:
    """
    Obtain the source of *fn*, apply ``_ProcTransformer``, compile and exec
    the result in *fn*'s global namespace, and return the rewritten function.

    The ``@workflow`` decorator is stripped from the AST before compilation to
    avoid re-entrant decoration.
    """
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError) as exc:
        raise RuntimeError(
            f"@workflow '{fn.__name__}': cannot retrieve source for AST rewriting. "
            f"Define the function in a .py file (not interactively). "
            f"Original error: {exc}"
        ) from exc

    source = textwrap.dedent(source)

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SyntaxError(
            f"@workflow '{fn.__name__}': failed to parse source."
        ) from exc

    # Strip all decorators from the target function so we don't recurse.
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn.__name__:
            node.decorator_list = []
            break

    transformer = _ProcTransformer()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)

    # Compile in the original file so tracebacks point to real lines.
    try:
        filename = inspect.getfile(fn)
    except (OSError, TypeError):
        filename = "<@workflow>"

    code = compile(new_tree, filename, "exec")

    # Inject builder helpers so users don't need to import them explicitly.
    exec_globals = fn.__globals__.copy()
    exec_globals.update({
        "msg":       msg,
        "act":       act,
        "if_":       if_,
        "while_":    while_,
        "coregion_": coregion_,
        "parallel_": parallel_,
        "_tag_cond": _tag_cond,
    })

    local_ns: dict = {}
    exec(code, exec_globals, local_ns)   # noqa: S102

    return local_ns[fn.__name__], transformer.return_outputs


# ---------------------------------------------------------------------------
# @workflow decorator
# ---------------------------------------------------------------------------

def _workflow_inputs(fn: Callable) -> tuple[tuple[str, ZType, Lifeline | None], ...]:
    """Extract (name, ZType, Lifeline | None) triples from parameter annotations.

    Accepts either plain ``ZType`` (lifeline is ``None``) or
    ``ZType @ Lifeline`` (a ``ZTypeAtLifeline`` object).
    """
    sig = inspect.signature(fn)
    inputs = []
    for name, param in sig.parameters.items():
        ann = param.annotation
        if ann is inspect.Parameter.empty:
            raise TypeError(
                f"@workflow '{fn.__name__}': parameter '{name}' "
                f"must have a type annotation (e.g. str @ Planner)."
            )
        if isinstance(ann, ZTypeAtLifeline):
            inputs.append((name, ann.type, ann.lifeline))
        elif is_ztype(ann):
            inputs.append((name, ann, None))
        else:
            raise TypeError(
                f"@workflow '{fn.__name__}': annotation for '{name}' "
                f"must be a supported coordination type or "
                f"type @ Lifeline, got {ann!r}."
            )
    return tuple(inputs)


def _workflow_output(fn: Callable) -> ZType:
    """Extract output ZType from return annotation.

    For single-output workflows use the concrete type (str, int, …).
    For multi-output workflows use ``tuple`` as the return annotation.
    Omit the annotation (or use -> None) for non-terminating workflows.
    """
    ret = fn.__annotations__.get("return")
    if ret is None or ret is type(None):
        return type(None)   # non-terminating / no output
    if not is_ztype(ret):
        raise TypeError(
            f"@workflow '{fn.__name__}': return annotation must be a supported "
            f"coordination type (e.g. -> str) or tuple for multi-output."
        )
    return ret


def workflow(fn: Callable) -> Workflow:
    """
    Decorator that records a coordination program and returns a ``Workflow`` IR node.

    The function parameters declare the workflow's input interface; each must have
    a ZipperGen type annotation. The return annotation declares the output type.

    Inside the body use:

    - ``Sender(x, y) >> Receiver(a, b)`` — message passing
    - ``with coregion: ...`` — unordered messages to one receiver
    - ``with parallel:\n        with branch: ...`` — first-class parallel region
    - ``Lifeline: outputs = action(inputs)`` — single local action
    - ``with Lifeline:\n        y1 = f1(x)\n        y2 = f2(y1)`` — block of consecutive local actions on one lifeline
    - ``skip()`` — local no-op
    - Native ``if cond @ owner: ... else: ...`` — conditional branching
    - Native ``while cond @ owner: ... else: ...`` — loops (else = exit body)

    The condition (left of ``@``) supports ``not``, ``and``, ``or``, ``<``, and
    boolean literals. Wrap compound conditions in parentheses when they start
    with ``not`` (Python precedence: ``@`` binds tighter than ``not``).
    """
    # Annotations are read from the original function before transformation.
    inputs = _workflow_inputs(fn)
    output_type = _workflow_output(fn)

    # Rewrite native if/while/>> into builder-function calls.
    transformed, return_outputs = _transform_proc_source(fn)

    # Execute the transformed body once to record all statements.
    kwargs = {name: Var(name, ztype) for name, ztype, _ll in inputs}
    body = _collect(lambda: transformed(**kwargs))

    # Resolve each return (var_name, lifeline_name) pair.
    resolved_outputs: list[tuple[Var, Lifeline]] = []
    if return_outputs:
        namespace = {**fn.__globals__, **kwargs}
        for var_name, ll_name in return_outputs:
            output_var = namespace.get(var_name)
            if not isinstance(output_var, Var):
                raise NameError(
                    f"@workflow '{fn.__name__}': return variable '{var_name}' "
                    f"is not a declared Var."
                )
            output_lifeline = namespace.get(ll_name)
            if not isinstance(output_lifeline, Lifeline):
                raise NameError(
                    f"@workflow '{fn.__name__}': return lifeline '{ll_name}' "
                    f"is not a declared Lifeline."
                )
            resolved_outputs.append((output_var, output_lifeline))

    return Workflow(
        name=fn.__name__,
        inputs=inputs,
        output_type=output_type,
        vars=(),    # variable declarations populated by verifier (Layer 5)
        body=cast(Stmt, body),
        outputs=tuple(resolved_outputs),
        ns=fn.__globals__,
    )


# ---------------------------------------------------------------------------
# @fragment decorator
# ---------------------------------------------------------------------------

def fragment(fn: Callable) -> Callable:
    """
    Decorator for reusable coordination sub-sequences.

    Apply the ZipperGen DSL transformation to a helper function.  When called
    inside a ``@workflow`` body, the fragment's statements are recorded
    directly into the surrounding scope — identical to writing them inline.

    Parameters are the ``Var`` objects that must already be in scope at the
    call site; all other ``Var`` objects may be referenced as module globals.

    Example::

        @fragment
        def task_branch(email):
            Dispatcher(email) >> Writer(email)
            Writer: (task_title, task_notes) = extract_task(email)
            Writer(email, task_title, task_notes) >> User(email, task_title, task_notes)
            ...

        @workflow
        def command_center():
            ...
            elif (route == "task") @ Dispatcher:
                task_branch(email)
    """
    transformed, _ = _transform_proc_source(fn)

    def wrapper(*args, **kwargs):
        if not _stack:
            raise RuntimeError(
                f"@fragment '{fn.__name__}' must be called inside a @workflow body."
            )
        transformed(*args, **kwargs)

    wrapper.__name__ = fn.__name__
    wrapper.__qualname__ = fn.__qualname__
    return wrapper
