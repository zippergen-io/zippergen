"""
ZipperGen — Layer 2: Action decorators.

Design notes
------------
**What this module is.**
This module provides ``@llm`` and ``@pure``, two decorators that let you define
actions using ordinary Python function syntax. They read the function's
parameter annotations to extract input types, and produce a ``LLMAction`` or
``PureAction`` IR node (from Layer 1). The function itself is not called by
the decorator — it is stored inside the IR node for later use by the executor.

**Why decorators instead of constructing IR nodes directly?**
Decorators keep the action definition close to normal Python style and avoid
repeating the parameter names (once in the signature, once in an inputs tuple).
They also enforce that every parameter has a ZipperGen type annotation, which
prevents silent mistakes.

**Why do the decorators return an IR node, not a callable?**
An action in ZipperGen is a declaration, not a function call. Returning an IR
node makes this explicit: after ``@pure``, the name refers to a ``PureAction``
object that belongs to a program. If you need to call the underlying Python
function directly (e.g. in tests), use ``action.fn(...)``.

**Why does @llm always require outputs= explicitly?**
An LLM action's body is ``...`` — there is no Python code to inspect for
return information. Output names and types must therefore be stated explicitly.
For ``@pure``, the single output name and type are inferred from the function
name and return annotation.

**Type annotations use ordinary Python built-ins.**
ZipperGen coordination types are expressed with built-in Python types like
``str``, ``bool``, ``int``, and ``float``. The decorators read these runtime
annotations directly via ``inspect.signature``.

Usage
-----
    @llm(
        system="You are a medical expert ...",
        user="Notes: {notes}\\nDiagnosis: {diag}",
        parse="json",
        outputs=(("verdict", bool), ("reason", str)),
    )
    def assess(notes: str, diag: str) -> None: ...

    @pure
    def check_agreement(v1: bool, v2: bool) -> bool:
        return v1 == v2

Notes
-----
The decorated name becomes a LLMAction / PureAction IR node, not a callable.
To call the underlying Python function directly, use ``action.fn(...)``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

from zippergen.syntax import (
    ZType, LLMAction, PureAction, is_ztype,
)

__all__ = ["llm", "pure"]

# Type alias for an output spec list/tuple (used by @llm)
OutputSpec = list[tuple[str, ZType]] | tuple[tuple[str, ZType], ...]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_inputs(fn: Callable) -> tuple[tuple[str, ZType], ...]:
    """
    Read (name, ZType) pairs from a function's parameter annotations.
    Raises TypeError for missing or non-ZType annotations.
    """
    sig = inspect.signature(fn)
    inputs: list[tuple[str, ZType]] = []
    for name, param in sig.parameters.items():
        ann = param.annotation
        if ann is inspect.Parameter.empty:
            raise TypeError(
                f"@action '{fn.__name__}': parameter '{name}' "
                f"must have a ZipperGen type annotation "
                f"(e.g. str, bool, int, float)."
            )
        if not is_ztype(ann):
            raise TypeError(
                f"@action '{fn.__name__}': annotation for '{name}' "
                f"must be a supported coordination type "
                f"(str, bool, int, float, or tuple), "
                f"got {ann!r}."
            )
        inputs.append((name, ann))
    return tuple(inputs)


def _single_output_from_return(fn: Callable) -> tuple[tuple[str, ZType], ...]:
    """
    Derive a single output from the return annotation.
    The output name is taken from the function name.
    Raises TypeError if the annotation is missing or not a ZType.
    """
    ret = fn.__annotations__.get("return")
    if ret is None or not is_ztype(ret):
        raise TypeError(
            f"@pure '{fn.__name__}': return annotation must be a supported "
            f"coordination type (e.g. -> bool)."
        )
    return ((fn.__name__, ret),)


# ---------------------------------------------------------------------------
# @llm decorator
# ---------------------------------------------------------------------------

def llm(
    *,
    system: str,
    user: str,
    parse: str,
    outputs: OutputSpec,
) -> Callable[[Callable], LLMAction]:
    """
    Decorator that produces a LLMAction node.

    Parameters
    ----------
    system : str
        System prompt passed to the LLM.
    user : str
        User prompt template; may contain ``{var_name}`` placeholders.
    parse : str
        Expected output format: ``"json"``, ``"text"``, or ``"bool"``.
    outputs : sequence of (name, ZType) pairs
        Output variable names and their ZipperGen types.
    """
    def decorator(fn: Callable) -> LLMAction:
        inputs = _extract_inputs(fn)
        return LLMAction(
            name=fn.__name__,
            inputs=inputs,
            outputs=tuple(outputs),
            system_prompt=system,
            user_prompt=user,
            parse_format=parse,
        )
    return decorator


# ---------------------------------------------------------------------------
# @pure decorator
# ---------------------------------------------------------------------------

def pure(fn: Callable) -> PureAction:
    """
    Decorator that produces a PureAction node.

        @pure
        def check_agreement(v1: bool, v2: bool) -> bool:
            return v1 == v2

    The output name is taken from the function name and its type from the
    return annotation.
    """
    inputs = _extract_inputs(fn)
    outputs = _single_output_from_return(fn)
    return PureAction(
        name=fn.__name__,
        inputs=inputs,
        outputs=outputs,
        fn=fn,
    )
