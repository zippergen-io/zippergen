"""
ZipperGen — Layer 3: Program builder.

Design notes
------------
**What this module is.**
This module provides helper functions that let you write a coordination program
as a normal Python function (decorated with ``@proc``) instead of constructing
IR nodes by hand. The helpers ``msg()``, ``act()``, ``skip()`` are the primary
user-facing API. Control structures (``if``/``while``) are written as native
Python — no special builder functions needed.

**How the recording context works.**
There is a global stack of statement lists (``_stack``). Entering a scope
(a ``@proc`` body or a nested branch) pushes a new empty list onto the stack.
Every call to ``msg()``, ``act()``, etc. appends one IR node to the top list.
Leaving the scope pops the list and folds it into a ``Stmt`` tree using
``seq()``. This is the only "magic" in this module.

**Native if/while syntax.**
The ``@proc`` decorator rewrites the function's AST before executing it.
Any ``if`` or ``while`` whose condition is of the form ``expr @ Lifeline``
is rewritten into the equivalent ``if_()`` / ``while_()`` call, where the
``@ Lifeline`` part identifies the owner of the control decision.

  .. code-block:: python

      @proc
      def myProc(task: Text) -> Text:
          if planNeedsReview @ Planner:
              Planner(plan) >> Reviewer(tR)
          else:
              skip(Planner)

          while (not agreed) @ LLM1:
              LLM1(v1) >> LLM2(v2)
              ...
          else:              # while...else = exit body
              LLM1(v1) >> User(result)

The condition (left of ``@``) can use native Python ``not``, ``and``, ``or``
and boolean literals — they are translated to ``Expr`` IR at rewrite time, so
they are never executed as Python boolean operations.

**Operator precedence note.**
Because Python gives ``@`` higher precedence than ``not``/``and``/``or``,
wrap compound conditions in parentheses when they start with ``not``:

  - ``if (not agreed) @ LLM1:``   ✓  — condition is ``not agreed``
  - ``if not agreed @ LLM1:``     ✗  — parsed as ``not (agreed @ LLM1)``

**Backward compatibility.**
The old ``if_()`` and ``while_()`` builder functions still work; existing
programs need no changes.

**Thread safety.**
The stack is a plain module-level list. This works correctly for sequential
(single-threaded) use, which covers all normal cases. If concurrent proc
definitions are ever needed, replace ``_stack`` with a ``threading.local()``
object — a one-line change.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable

from zippergen.syntax import (
    ZType, Lifeline, Var,
    ZTypeAtLifeline,
    Expr, VarExpr, LitExpr, NotExpr, AndExpr, OrExpr,
    Stmt, MsgStmt, ActStmt, SkipStmt, IfStmt, WhileStmt,
    LLMAction, PureAction,
    Proc,
    seq, is_ztype,
)

__all__ = [
    # Proc decorator
    "proc",
    # Statement builders
    "msg", "act", "skip",
    "if_", "while_",
    # Expression builders (usable outside @proc, or as explicit style inside)
    "not_", "and_", "or_", "lit",
]


# ---------------------------------------------------------------------------
# Recording context
# ---------------------------------------------------------------------------

_stack: list[list[Stmt]] = []


def _record(stmt: Stmt) -> None:
    """Append stmt to the innermost open scope."""
    if not _stack:
        raise RuntimeError(
            "Statement builders (msg, act, etc.) must be called "
            "inside a @proc body or a nested branch."
        )
    _stack[-1].append(stmt)


def _collect(fn: Callable) -> Stmt:
    """
    Open a new scope, run fn(), close the scope, and return the
    collected statements folded into a single Stmt via seq().
    """
    _stack.append([])
    fn()
    stmts = _stack.pop()
    return seq(*stmts)


# ---------------------------------------------------------------------------
# Expression builders  (pure — no recording)
# ---------------------------------------------------------------------------

def _to_expr(x: object) -> Expr:
    """Coerce a Var to VarExpr; pass through anything already an Expr."""
    if isinstance(x, Var):
        return VarExpr(x)
    return x  # type: ignore[return-value]


def not_(expr: object) -> NotExpr:
    """Logical negation."""
    return NotExpr(_to_expr(expr))


def and_(left: object, right: object) -> AndExpr:
    """Logical conjunction."""
    return AndExpr(_to_expr(left), _to_expr(right))


def or_(left: object, right: object) -> OrExpr:
    """Logical disjunction."""
    return OrExpr(_to_expr(left), _to_expr(right))


def lit(value: object, type_: ZType) -> LitExpr:
    """A literal value: lit(True, Bool)  →  LitExpr(True, Bool)"""
    return LitExpr(value, type_)


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
    action: LLMAction | PureAction,
    inputs: tuple,
    outputs: tuple[Var, ...],
) -> None:
    """
    act lifeline: outputs := action(inputs)

    Vars in inputs are automatically wrapped in VarExpr.
    """
    _record(ActStmt(
        lifeline,
        action,
        tuple(_to_expr(x) for x in inputs),
        tuple(outputs),
    ))


def skip(lifeline: Lifeline) -> None:
    """skip lifeline — local no-op."""
    _record(SkipStmt(lifeline))


def if_(
    condition: object,
    owner: Lifeline,
    *,
    then: Callable,
    else_: Callable,
) -> None:
    """
    if condition@owner then { then() } else { else_() }

    Prefer native ``if cond @ owner:`` syntax inside ``@proc`` bodies.
    This function is the explicit fallback (e.g. for programmatic construction).
    """
    _record(IfStmt(
        _to_expr(condition),
        owner,
        _collect(then),
        _collect(else_),
    ))


def while_(
    condition: object,
    owner: Lifeline,
    *,
    body: Callable,
    exit_body: Callable,
) -> None:
    """
    while condition@owner do { body() } exit { exit_body() }

    Prefer native ``while cond @ owner: ... else: ...`` inside ``@proc``.
    This function is the explicit fallback (e.g. for programmatic construction).
    """
    _record(WhileStmt(
        _to_expr(condition),
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


def _ast_to_expr_ast(node: ast.expr) -> ast.expr:
    """
    Translate a Python condition AST node to an AST fragment that, when
    executed, produces a ZipperGen ``Expr`` object.

    - ``Name('x')``          → left as-is  (Var at runtime; _to_expr handles it)
    - ``UnaryOp(Not, ...)``  → ``Call(not_, [...])``
    - ``BoolOp(And, [...])`` → folded ``Call(and_, [..., ...])``
    - ``BoolOp(Or, [...])``  → folded ``Call(or_, [..., ...])``
    - ``Constant(bool)``     → ``Call(lit, [value, Bool])``
    - anything else          → left as-is  (e.g. explicit not_(), and_() calls)
    """
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return ast.Call(
            func=ast.Name(id="not_", ctx=ast.Load()),
            args=[_ast_to_expr_ast(node.operand)],
            keywords=[],
        )

    if isinstance(node, ast.BoolOp):
        fn_name = "and_" if isinstance(node.op, ast.And) else "or_"
        # Right-fold: a and b and c  →  and_(a, and_(b, c))
        result = _ast_to_expr_ast(node.values[-1])
        for val in reversed(node.values[:-1]):
            result = ast.Call(
                func=ast.Name(id=fn_name, ctx=ast.Load()),
                args=[_ast_to_expr_ast(val), result],
                keywords=[],
            )
        return result

    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return ast.Call(
            func=ast.Name(id="lit", ctx=ast.Load()),
            args=[node, ast.Name(id="bool", ctx=ast.Load())],
            keywords=[],
        )

    # Name, Call, Attribute, etc. — pass through unchanged.
    return node


def _make_fn(name: str, body: list[ast.stmt]) -> ast.FunctionDef:
    """Build a no-argument FunctionDef AST node."""
    return ast.FunctionDef(
        name=name,
        args=ast.arguments(
            posonlyargs=[], args=[], vararg=None,
            kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[],
        ),
        body=body or [ast.Pass()],
        decorator_list=[],
        returns=None,
        lineno=0, col_offset=0,
    )


class _ProcTransformer(ast.NodeTransformer):
    """
    Rewrites coordination-DSL patterns into their builder-function equivalents:

    - ``Lifeline: outputs = action(inputs)``  →  ``act(Lifeline, action, inputs, outputs)``
    - ``Sender(x, y) >> Receiver(a, b)``      →  ``msg(Sender, (x, y), Receiver, (a, b))``
    - ``if cond @ owner: ...``                →  ``if_(cond, owner, then=..., else_=...)``
    - ``while cond @ owner: ...``             →  ``while_(cond, owner, body=..., exit_body=...)``
    - ``return var``                          →  removed; ``var`` stored in ``self.return_var_name``

    All other Python statements pass through unchanged.
    """

    def __init__(self) -> None:
        super().__init__()
        self.return_var_name: str | None = None
        self.return_lifeline_name: str | None = None

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

    def visit_Return(self, node: ast.Return) -> ast.stmt:
        """Strip ``return var @ Lifeline`` and record both names as the proc output."""
        val = node.value
        if (isinstance(val, ast.BinOp) and isinstance(val.op, ast.MatMult)
                and isinstance(val.left, ast.Name) and isinstance(val.right, ast.Name)):
            self.return_var_name = val.left.id
            self.return_lifeline_name = val.right.id
        else:
            raise SyntaxError(
                "return in @proc must have the form  return var @ Lifeline"
            )
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
        return ast.Expr(ast.Call(
            func=ast.Name(id="act", ctx=ast.Load()),
            args=[
                lifeline_ast,
                action_call.func,
                ast.Tuple(elts=list(action_call.args), ctx=ast.Load()),
                ast.Tuple(elts=outputs,                ctx=ast.Load()),
            ],
            keywords=[],
        ))

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

        cond = _ast_to_expr_ast(node.test.left)   # type: ignore[arg-type]
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

        cond = _ast_to_expr_ast(node.test.left)   # type: ignore[arg-type]
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


def _transform_proc_source(fn: Callable) -> tuple[Callable, str | None, str | None]:
    """
    Obtain the source of *fn*, apply ``_ProcTransformer``, compile and exec
    the result in *fn*'s global namespace, and return the rewritten function.

    The ``@proc`` decorator is stripped from the AST before compilation to
    avoid re-entrant decoration.
    """
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError) as exc:
        raise RuntimeError(
            f"@proc '{fn.__name__}': cannot retrieve source for AST rewriting. "
            f"Define the function in a .py file (not interactively). "
            f"Original error: {exc}"
        ) from exc

    source = textwrap.dedent(source)

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SyntaxError(
            f"@proc '{fn.__name__}': failed to parse source."
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
        filename = "<@proc>"

    code = compile(new_tree, filename, "exec")

    # Inject builder helpers so users don't need to import them explicitly.
    exec_globals = fn.__globals__.copy()
    exec_globals.update({
        "msg":   msg,
        "act":   act,
        "if_":   if_,
        "while_": while_,
        "not_":  not_,
        "and_":  and_,
        "or_":   or_,
        "lit":   lit,
    })

    local_ns: dict = {}
    exec(code, exec_globals, local_ns)   # noqa: S102

    return local_ns[fn.__name__], transformer.return_var_name, transformer.return_lifeline_name


# ---------------------------------------------------------------------------
# @proc decorator
# ---------------------------------------------------------------------------

def _proc_inputs(fn: Callable) -> tuple[tuple[str, ZType, Lifeline | None], ...]:
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
                f"@proc '{fn.__name__}': parameter '{name}' "
                f"must have a type annotation (e.g. Text @ Planner)."
            )
        if isinstance(ann, ZTypeAtLifeline):
            inputs.append((name, ann.type, ann.lifeline))
        elif is_ztype(ann):
            inputs.append((name, ann, None))
        else:
            raise TypeError(
                f"@proc '{fn.__name__}': annotation for '{name}' "
                f"must be a ZType or ZType @ Lifeline, got {ann!r}."
            )
    return tuple(inputs)


def _proc_output(fn: Callable) -> ZType:
    """Extract output ZType from return annotation."""
    ret = fn.__annotations__.get("return")
    if ret is None or not is_ztype(ret):
        raise TypeError(
            f"@proc '{fn.__name__}': return annotation must be a ZType "
            f"(e.g. -> Text)."
        )
    return ret


def proc(fn: Callable) -> Proc:
    """
    Decorator that records a coordination program and returns a ``Proc`` IR node.

    The function parameters declare the proc's input interface; each must have
    a ZipperGen type annotation. The return annotation declares the output type.

    Inside the body use:

    - ``Sender(x, y) >> Receiver(a, b)`` — message passing
    - ``Lifeline: outputs = action(inputs)`` — local action (mirrors paper's ``act A : y := f(x)``)
    - ``skip()`` — local no-op
    - Native ``if cond @ owner: ... else: ...`` — conditional branching
    - Native ``while cond @ owner: ... else: ...`` — loops (else = exit body)

    The condition (left of ``@``) supports ``not``, ``and``, ``or``, and
    boolean literals. Wrap compound conditions in parentheses when they start
    with ``not`` (Python precedence: ``@`` binds tighter than ``not``).
    """
    # Annotations are read from the original function before transformation.
    inputs = _proc_inputs(fn)
    output_type = _proc_output(fn)

    # Rewrite native if/while/>> into builder-function calls.
    transformed, return_var_name, return_lifeline_name = _transform_proc_source(fn)

    # Execute the transformed body once to record all statements.
    kwargs = {name: Var(name, ztype) for name, ztype, _ll in inputs}
    body = _collect(lambda: transformed(**kwargs))

    # Resolve return var @ Lifeline if declared.
    output_var: Var | None = None
    output_lifeline: Lifeline | None = None
    if return_var_name is not None:
        if return_lifeline_name is None:
            raise RuntimeError(
                f"@proc '{fn.__name__}': internal error: return lifeline missing "
                f"for return variable '{return_var_name}'."
            )
        namespace = {**fn.__globals__, **kwargs}
        output_var = namespace.get(return_var_name)
        if not isinstance(output_var, Var):
            raise NameError(
                f"@proc '{fn.__name__}': return variable '{return_var_name}' "
                f"is not a declared Var."
            )
        output_lifeline = namespace.get(return_lifeline_name)
        if not isinstance(output_lifeline, Lifeline):
            raise NameError(
                f"@proc '{fn.__name__}': return lifeline '{return_lifeline_name}' "
                f"is not a declared Lifeline."
            )

    return Proc(
        name=fn.__name__,
        inputs=inputs,
        output_type=output_type,
        vars=(),    # variable declarations populated by verifier (Layer 5)
        body=body,
        output_var=output_var,
        output_lifeline=output_lifeline,
    )
