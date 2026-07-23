"""Layer 2 action decorators.

``@llm``, ``@pure``, ``@effect``, ``@assistant``, ``@planner``, and ``@human``
read Python annotations to produce action IR nodes.
"""

from __future__ import annotations

import inspect
import hashlib
import re
from collections.abc import Callable
from pathlib import Path

from zippergen.syntax import (
    ZType, LLMAction, PureAction, EffectAction, AssistantAction, PlannerAction,
    HumanAction, Lifeline, is_ztype,
)

__all__ = ["llm", "pure", "effect", "assistant", "planner", "human"]

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
            f"@action '{fn.__name__}': return annotation must be a supported "
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
    declares the generated workflow's result type.

    At runtime the action builds a full hidden system prompt from ``description``,
    an auto-generated worker summary, optional ``instructions``, DSL rules, and
    any ``allow`` extensions, then calls the LLM to generate a sub-workflow,
    validates it, and runs it.

    Parameters
    ----------
    description : str
        One sentence describing the planner's task domain.
    actions : list
        Pre-defined action vocabulary (``LLMAction`` / ``PureAction`` /
        ``EffectAction`` / ``HumanAction`` / ``PlannerAction`` nodes).
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
    max_retries : int, optional
        Maximum number of generated workflow candidates to validate.

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
        if ret is None or not is_ztype(ret):
            raise TypeError(
                f"@planner '{fn.__name__}': return annotation must be a supported "
                f"coordination type (str, bool, int, float, or tuple)."
            )
        outputs = ((fn.__name__, ret),)
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

def pure(fn: Callable | None = None, *, visible: bool = True):
    """
    Decorator that produces a PureAction node.

        @pure
        def check_agreement(v1: bool, v2: bool) -> bool:
            return v1 == v2

    Pass ``visible=False`` to suppress ZipperChat trace events:

        @pure(visible=False)
        def wait_briefly() -> str: ...

    The output name is taken from the function name and its type from the
    return annotation.
    """
    def decorator(fn: Callable) -> PureAction:
        inputs = _extract_inputs(fn)
        outputs = _single_output_from_return(fn)
        return PureAction(
            name=fn.__name__,
            inputs=inputs,
            outputs=outputs,
            fn=fn,
            visible=visible,
        )
    if fn is not None:
        return decorator(fn)
    return decorator


# ---------------------------------------------------------------------------
# @effect decorator
# ---------------------------------------------------------------------------

def effect(fn: Callable | None = None, *, visible: bool = True):
    """
    Decorator for Python actions that are not deterministic/pure.

    In the in-memory runner, an effect action executes like ``@pure``. In the
    SQLite runner, it is resolved outside the write transaction and its output
    is journaled so replay returns the recorded result instead of performing the
    external operation again.
    """
    def decorator(fn: Callable) -> EffectAction:
        inputs = _extract_inputs(fn)
        outputs = _single_output_from_return(fn)
        return EffectAction(
            name=fn.__name__,
            inputs=inputs,
            outputs=outputs,
            fn=fn,
            visible=visible,
        )
    if fn is not None:
        return decorator(fn)
    return decorator


# ---------------------------------------------------------------------------
# @assistant decorator
# ---------------------------------------------------------------------------

def _assistant_instruction_path(fn: Callable, declared: str) -> Path:
    path = Path(declared).expanduser()
    if path.is_absolute():
        return path.resolve()

    # Action instruction files are project-relative.  This is stable for the
    # CLI and deployment runner, both of which run from the project/bundle
    # root.  A module-relative fallback keeps direct imports convenient.
    project_candidate = (Path.cwd() / path).resolve()
    if project_candidate.is_file():
        return project_candidate
    source = inspect.getsourcefile(fn)
    if source:
        module_candidate = (Path(source).resolve().parent / path).resolve()
        if module_candidate.is_file():
            return module_candidate
    return project_candidate


def assistant(
    *,
    instructions: str | None = None,
    instructions_file: str | None = None,
    backend: str | None = None,
    workspace: str | None = None,
    timeout: float | None = None,
    visible: bool = True,
):
    """Declare a repository-aware coding-assistant action.

    Exactly one of ``instructions`` and ``instructions_file`` is required.
    Markdown files are read when the workflow module is imported, fingerprinted
    as part of the semantic action definition, and automatically included in a
    guided deployment bundle.

    ``backend`` may request ``"codex"`` or ``"claude"`` for this action.  When
    omitted, the runtime default selected with
    ``workflow.configure(assistant="...")`` or ``ZIPPERGEN_ASSISTANT`` is used.
    ``workspace`` is a static path, relative to the configured project root.
    The decorated function's typed parameters become explicit dynamic inputs;
    its return annotation declares the single typed result.
    """
    if (instructions is None) == (instructions_file is None):
        raise TypeError(
            "@assistant requires exactly one of 'instructions' or "
            "'instructions_file'."
        )
    if backend is not None and backend not in {"codex", "claude"}:
        raise ValueError(
            f"@assistant backend must be 'codex' or 'claude', got {backend!r}."
        )
    if timeout is not None and timeout <= 0:
        raise ValueError("@assistant timeout must be greater than zero.")

    def decorator(fn: Callable) -> AssistantAction:
        inputs = _extract_inputs(fn)
        outputs = _single_output_from_return(fn)
        path: Path | None = None
        text = instructions
        if instructions_file is not None:
            path = _assistant_instruction_path(fn, instructions_file)
            if not path.is_file():
                raise FileNotFoundError(
                    f"@assistant '{fn.__name__}': instruction file does not "
                    f"exist: {path}"
                )
            text = path.read_text(encoding="utf-8")
        assert text is not None
        if not text.strip():
            source = f"file {path}" if path is not None else "inline instructions"
            raise ValueError(
                f"@assistant '{fn.__name__}': {source} must not be empty."
            )
        return AssistantAction(
            name=fn.__name__,
            inputs=inputs,
            outputs=outputs,
            instructions=text,
            instructions_file=instructions_file,
            instructions_path=str(path) if path is not None else None,
            instructions_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            backend=backend,
            workspace=workspace,
            timeout=timeout,
            visible=visible,
        )

    return decorator


# ---------------------------------------------------------------------------
# @human decorator
# ---------------------------------------------------------------------------

def _parse_human_output(spec: str, fn_name: str) -> tuple[str, type]:
    """Parse "name: type" output spec string into (name, type) pair."""
    parts = [p.strip() for p in spec.split(":")]
    if len(parts) != 2:
        raise TypeError(
            f"@human '{fn_name}': output spec must be 'name: type', got {spec!r}"
        )
    name, type_str = parts
    if not name:
        raise TypeError(
            f"@human '{fn_name}': output name must not be empty in spec {spec!r}"
        )
    type_map: dict[str, type] = {"bool": bool, "str": str}
    if type_str not in type_map:
        raise TypeError(
            f"@human '{fn_name}': output type must be 'bool' or 'str', "
            f"got {type_str!r}"
        )
    return name, type_map[type_str]


def human(
    *,
    kind: str,
    outputs: list[str],
    context: str | None = None,
    instruction: str | None = None,
    prefill: str | None = None,
    submit_label: str | None = None,
    cancel_label: str | None = None,
    visible: bool = True,
):
    """
    Decorator that produces a HumanAction node.

    Parameters
    ----------
    kind : str
        Interaction shape: ``"confirm"`` (yes/no), ``"ack"`` (acknowledge a
        completed event — single button, no cancel), ``"edit"`` (edit
        pre-filled text), ``"select"`` (choose from a list), ``"input"``
        (free-form text).
    outputs : list of str
        Single-element list with ``"name: type"`` spec, e.g. ``["approved: bool"]``.
        ``confirm`` and ``ack`` require ``bool``; ``edit``, ``select``, and
        ``input`` require ``str``.
    context : str, optional
        Template for the left-column context panel.  Use ``{var_name}``
        placeholders to embed input variable values; literal text is shown
        as-is.  Multiple variables can be included: ``"{email}\\n{notes}"``.
    instruction : str, optional
        Instruction text shown in the right column (or above the buttons for
        ``confirm``).  Supports ``{var_name}`` placeholders.
    prefill : str, optional
        For ``edit``: a ``{var_name}`` template whose resolved value
        pre-populates the textarea.
        For ``options``: either a ``{var_name}`` template or a literal
        newline-separated string of choices (e.g. ``"Send\\nSave as draft"``).
    submit_label : str, optional
        Label for the primary (approve/submit) button.
    cancel_label : str, optional
        Label for the secondary (decline/cancel) button.
    """
    _valid_kinds = {"confirm", "edit", "select", "input", "ack"}

    def decorator(fn: Callable) -> HumanAction:
        fn_name = fn.__name__
        inputs = _extract_inputs(fn)
        input_names = {name for name, _ in inputs}

        if kind not in _valid_kinds:
            raise ValueError(
                f"@human '{fn_name}': unsupported kind {kind!r}. "
                f"Supported: {sorted(_valid_kinds)}"
            )

        if len(outputs) != 1:
            raise TypeError(
                f"@human '{fn_name}': exactly one output required, "
                f"got {len(outputs)}"
            )
        output_name, output_type = _parse_human_output(outputs[0], fn_name)

        if kind in ("confirm", "ack") and output_type is not bool:
            raise TypeError(
                f"@human '{fn_name}': kind='{kind}' requires a bool output, "
                f"got '{output_type.__name__}'"
            )
        if kind in ("edit", "select", "input") and output_type is not str:
            raise TypeError(
                f"@human '{fn_name}': kind='{kind}' requires a str output, "
                f"got '{output_type.__name__}'"
            )

        # Validate {var} placeholders in template fields
        for field_name, value in (
            ("context", context),
            ("instruction", instruction),
            ("prefill", prefill),
        ):
            if value is None:
                continue
            unknown = set(re.findall(r'\{(\w+)\}', value)) - input_names
            if unknown:
                raise TypeError(
                    f"@human '{fn_name}': {field_name}= references unknown "
                    f"variables {unknown}. Declared inputs: {input_names}"
                )

        return HumanAction(
            name=fn_name,
            inputs=inputs,
            output=output_name,
            output_type=output_type,
            kind=kind,
            context=context,
            instruction=instruction,
            prefill=prefill,
            submit_label=submit_label,
            cancel_label=cancel_label,
            visible=visible,
        )

    return decorator
