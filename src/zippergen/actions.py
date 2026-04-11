"""
Layer 2: @llm, @pure, and @planner decorators. Read Python annotations to
produce LLMAction, PureAction, and PlannerAction IR nodes.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

from zippergen.syntax import (
    ZType, LLMAction, PureAction, PlannerAction, Lifeline, is_ztype,
)

__all__ = ["llm", "pure", "planner"]

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
# @planner decorator
# ---------------------------------------------------------------------------

def planner(
    *,
    description: str,
    actions: list,
    lifelines: list,
    allow: list[str] | None = None,
    instructions: str | None = None,
    max_retries: int = 3,
) -> Callable[[Callable], PlannerAction]:
    """
    Decorator that produces a PlannerAction node.

    The decorated function's body must be ``...``.  Its parameter annotations
    declare the inputs passed to the LLM planner.  The return annotation
    must be ``str``.

    At runtime the action builds a full hidden system prompt from ``description``,
    an auto-generated worker summary, optional ``instructions``, DSL rules, and
    any ``allow`` extensions, then calls the LLM to generate a sub-workflow,
    validates it, and runs it.

    Parameters
    ----------
    description : str
        One sentence describing the planner's task domain.
    actions : list
        Pre-defined action vocabulary (``LLMAction`` / ``PureAction`` nodes).
        The LLM may use these directly.  Pass ``[]`` to start from scratch.
    lifelines : list
        Workers available to the generated workflow.  Each entry may be a
        ``Lifeline`` object or a plain string; strings are converted to
        ``Lifeline`` objects automatically.  Worker names must be distinct
        from the calling lifeline's name.
    allow : list of str, optional
        What the LLM is permitted to use or define in the generated spec.
        ``"pure"`` — may define ``@pure`` Python helper functions.
        ``"llm"``  — may define new ``@llm`` actions with custom prompts.
        ``"if"``   — may use ``if cond @ Owner:`` conditional branching.
        ``"while"``— may use ``while cond @ Owner:`` loops.
        Defaults to no extensions (fixed vocabulary, linear workflows only).
    instructions : str, optional
        Additional coordination guidance, e.g. how to assign roles to workers.
        When omitted the runtime encourages the planner to use as many workers
        as reasonable.

    Example::

        @planner(
            description="A workflow planner for text processing tasks.",
            actions=[summarise, translate],
            lifelines=["Worker1", "Worker2", "Aggregator"],
            allow=["pure", "llm"],
            instructions="Use Worker1 and Worker2 in parallel, then Aggregator to combine.",
        )
        def run_task(request: str, inputs_json: str) -> str: ...
    """
    def decorator(fn: Callable) -> PlannerAction:
        inputs = _extract_inputs(fn)
        ret = fn.__annotations__.get("return")
        if ret is not str:
            raise TypeError(
                f"@planner '{fn.__name__}': return annotation must be str."
            )
        outputs = ((fn.__name__, str),)
        _allow = tuple(allow) if allow else ()
        _valid = {"pure", "llm", "if", "while"}
        for kind in _allow:
            if kind not in _valid:
                raise ValueError(
                    f"@planner '{fn.__name__}': unsupported allow value {kind!r}. "
                    f"Supported: {sorted(_valid)}"
                )
        # Normalise: accept Lifeline objects or plain strings.
        _lifelines = tuple(
            ll if isinstance(ll, Lifeline) else Lifeline(ll)
            for ll in lifelines
        )
        return PlannerAction(
            name=fn.__name__,
            inputs=inputs,
            outputs=outputs,
            system_prompt=description,
            actions=tuple(actions),
            lifelines=_lifelines,
            allow=_allow,
            instructions=instructions,
            max_retries=max_retries,
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
