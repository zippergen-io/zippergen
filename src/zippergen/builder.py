"""
ZipperGen — Layer 3: Program builder.

Design notes
------------
**What this module is.**
This module provides helper functions that let you write a coordination program
as a normal Python function (decorated with ``@proc``) instead of constructing
IR nodes by hand. The helpers ``msg()``, ``act()``, ``while_()``, etc. are the
user-facing API for building programs.

**How the recording context works.**
There is a global stack of statement lists (``_stack``). Entering a scope
(a ``@proc`` body or a nested ``body``/``exit_body`` function) pushes a new
empty list onto the stack. Every call to ``msg()``, ``act()``, etc. appends
one IR node to the top list. Leaving the scope pops the list and folds it
into a ``Stmt`` tree using ``seq()``. This is the only "magic" in this module.

**Statement builders vs expression builders.**
``msg()``, ``act()``, ``skip()``, ``if_()``, ``while_()`` are statement
builders: they record a node into the current scope and return ``None``.
``not_()``, ``and_()``, ``or_()``, ``lit()`` are expression builders: they
are pure functions that construct and return an ``Expr`` object without
touching the recording stack at all.

**Why nested def for while_/if_ bodies?**
Python lambdas cannot contain multiple statements, so the natural way to pass
a block of code is a plain nested ``def``. The body function is called once
by ``_collect()``, which temporarily pushes a new scope, runs the function,
and returns the collected statements as a single ``Stmt``.

**Why @proc is a simple decorator (no parentheses)?**
It always takes exactly one argument — the function — so there is no need
for the double-parentheses pattern. Compare with ``@llm(...)`` which requires
configuration arguments.

**Thread safety.**
The stack is a plain module-level list. This works correctly for sequential
(single-threaded) use, which covers all normal cases. If concurrent proc
definitions are ever needed, replace ``_stack`` with a ``threading.local()``
object — a one-line change.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

from zippergen.syntax import (
    ZType, Lifeline, Var,
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
    # Expression builders
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
            "inside a @proc body or a nested body/exit_body function."
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
    """Logical negation: not_(agreed)  →  NotExpr(VarExpr(agreed))"""
    return NotExpr(_to_expr(expr))


def and_(left: object, right: object) -> AndExpr:
    """Logical conjunction."""
    return AndExpr(_to_expr(left), _to_expr(right))


def or_(left: object, right: object) -> OrExpr:
    """Logical disjunction."""
    return OrExpr(_to_expr(left), _to_expr(right))


def lit(value: object, type_: ZType) -> LitExpr:
    """A literal value: lit(42, Int)  →  LitExpr(42, Int)"""
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

    Pass the two branches as plain callables (nested defs or lambdas).
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

    Pass the two blocks as plain callables (nested defs).
    """
    _record(WhileStmt(
        _to_expr(condition),
        owner,
        _collect(body),
        _collect(exit_body),
    ))


# ---------------------------------------------------------------------------
# @proc decorator
# ---------------------------------------------------------------------------

def _proc_inputs(fn: Callable) -> tuple[tuple[str, ZType], ...]:
    """Extract (name, ZType) pairs from parameter annotations."""
    sig = inspect.signature(fn)
    inputs = []
    for name, param in sig.parameters.items():
        ann = param.annotation
        if ann is inspect.Parameter.empty:
            raise TypeError(
                f"@proc '{fn.__name__}': parameter '{name}' "
                f"must have a ZipperGen type annotation (e.g. Text, Bool)."
            )
        if not is_ztype(ann):
            raise TypeError(
                f"@proc '{fn.__name__}': annotation for '{name}' "
                f"must be a ZType (Text, Bool, Int, Float, or TTuple), "
                f"got {ann!r}."
            )
        inputs.append((name, ann))
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
    Decorator that runs the function body once (recording all statements)
    and returns a Proc IR node.

    The function parameters declare the proc's input interface; each must
    have a ZipperGen type annotation. The return annotation declares the
    output type.

    Inside the body, use msg(), act(), skip(), if_(), while_() to build
    the program. Vars passed as parameters are equal to same-named
    module-level Var objects (frozen dataclass equality), so either works.
    """
    inputs = _proc_inputs(fn)
    output_type = _proc_output(fn)
    # Pass Var objects for each declared input so the function can be called.
    kwargs = {name: Var(name, ztype) for name, ztype in inputs}
    body = _collect(lambda: fn(**kwargs))
    return Proc(
        name=fn.__name__,
        inputs=inputs,
        output_type=output_type,
        vars=(),    # variable declarations populated by verifier (Layer 5)
        body=body,
    )
