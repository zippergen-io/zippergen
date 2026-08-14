"""Canonical durable contract for human actions and their responses.

Every adapter sees the same task specification.  This module owns the rules
that turn a :class:`HumanAction` into that specification and that decide which
responses are legal; terminal, CLI, and connector code should only render or
transport those values.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, TypedDict, cast

from zippergen.syntax import HumanAction, validate_zvalue


HumanTaskKind = Literal["confirm", "ack", "edit", "select", "input"]
_KINDS = {"confirm", "ack", "edit", "select", "input"}
_BOOL_KINDS = {"confirm", "ack"}
_STRING_KINDS = {"edit", "select", "input"}
_MISSING = object()


class RenderedHumanTask(TypedDict, total=False):
    context: str | None
    instruction: str | None
    prefill: str | None


class HumanTaskSpec(TypedDict):
    kind: HumanTaskKind
    output: str
    output_type: Literal["bool", "str"]
    rendered: RenderedHumanTask
    submit_label: str | None
    cancel_label: str | None


def _render(template: str | None, inputs: Mapping[str, object]) -> str | None:
    return template.format(**inputs) if template else None


def build_human_task_spec(
    action: HumanAction,
    inputs: Mapping[str, object],
) -> HumanTaskSpec:
    """Build the single durable representation of a visible human action."""

    return validate_human_task_spec(
        {
            "kind": action.kind,
            "output": action.output,
            "output_type": action.output_type.__name__,
            "rendered": {
                "context": _render(action.context, inputs),
                "instruction": _render(action.instruction, inputs),
                "prefill": _render(action.prefill, inputs),
            },
            "submit_label": action.submit_label,
            "cancel_label": action.cancel_label,
        },
        context=f"Human action {action.name!r}",
    )


def validate_human_task_spec(
    value: object,
    *,
    context: str = "Human task specification",
) -> HumanTaskSpec:
    """Validate and canonicalize a durable human-task specification."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object.")
    kind = value.get("kind")
    if not isinstance(kind, str) or kind not in _KINDS:
        raise ValueError(
            f"{context} has unsupported kind {kind!r}; "
            f"expected one of {sorted(_KINDS)}."
        )
    output = value.get("output")
    if not isinstance(output, str) or not output:
        raise ValueError(f"{context} must name its output.")
    output_type = value.get("output_type")
    if output_type not in {"bool", "str"}:
        raise ValueError(f"{context} output_type must be 'bool' or 'str'.")
    if kind in _BOOL_KINDS and output_type != "bool":
        raise ValueError(f"{context} kind {kind!r} requires output_type 'bool'.")
    if kind in _STRING_KINDS and output_type != "str":
        raise ValueError(f"{context} kind {kind!r} requires output_type 'str'.")

    rendered_value = value.get("rendered") or {}
    if not isinstance(rendered_value, Mapping):
        raise TypeError(f"{context} rendered content must be an object.")
    rendered: RenderedHumanTask = {}
    for field in ("context", "instruction", "prefill"):
        field_value = rendered_value.get(field)
        if field_value is not None and not isinstance(field_value, str):
            raise TypeError(f"{context} rendered {field} must be text or null.")
        rendered[field] = field_value

    labels: dict[str, str | None] = {}
    for field in ("submit_label", "cancel_label"):
        field_value = value.get(field)
        if field_value is not None and not isinstance(field_value, str):
            raise TypeError(f"{context} {field} must be text or null.")
        labels[field] = field_value

    return {
        "kind": cast(HumanTaskKind, kind),
        "output": output,
        "output_type": cast(Literal["bool", "str"], output_type),
        "rendered": rendered,
        "submit_label": labels["submit_label"],
        "cancel_label": labels["cancel_label"],
    }


def human_task_options(spec: Mapping[str, object]) -> tuple[str, ...]:
    """Return the rendered choices of a select task, in display order."""

    canonical = validate_human_task_spec(spec)
    if canonical["kind"] != "select":
        return ()
    return tuple(
        line.strip()
        for line in (canonical["rendered"].get("prefill") or "").splitlines()
        if line.strip()
    )


def _parse_bool(raw: object) -> bool:
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().casefold()
    if text in {"true", "yes", "1", "y", "approve", "approved", "ack"}:
        return True
    if text in {
        "false", "no", "0", "n", "decline", "declined", "reject", "rejected"
    }:
        return False
    raise ValueError(f"Cannot parse boolean human response: {raw!r}")


def human_task_result_from_value(
    spec: Mapping[str, object],
    value: object = _MISSING,
) -> dict[str, object]:
    """Parse one CLI/connector response according to the task contract."""

    canonical = validate_human_task_spec(spec)
    kind = canonical["kind"]
    output = canonical["output"]
    if canonical["output_type"] == "bool":
        result_value = (
            True if value is _MISSING or value is None else _parse_bool(value)
        )
    else:
        if value is _MISSING or value is None:
            raise ValueError(f"Human task requires a text value for {output!r}.")
        result_value = str(value)
        options = human_task_options(canonical)
        if options:
            raw = result_value.strip()
            if raw.isdigit() and 1 <= int(raw) <= len(options):
                result_value = options[int(raw) - 1]
            elif raw not in options:
                raise ValueError(f"Choose a number between 1 and {len(options)}.")

    result: dict[str, object] = {output: result_value}
    if kind == "ack" and result_value is not True:
        raise ValueError("An acknowledgement can only be completed affirmatively.")
    return result


def validate_human_task_result(
    spec: Mapping[str, object],
    result: object,
    *,
    context: str = "Human task result",
) -> dict[str, object]:
    """Validate a backend/store result without coercing its Python value."""

    canonical = validate_human_task_spec(spec)
    if not isinstance(result, Mapping):
        raise TypeError(f"{context} must be an object.")
    output = canonical["output"]
    if set(result) != {output}:
        raise ValueError(f"{context} must contain exactly output {output!r}.")
    expected_type = bool if canonical["output_type"] == "bool" else str
    result_value = validate_zvalue(
        result[output],
        expected_type,
        context=f"{context} output {output!r}",
    )
    if canonical["kind"] == "ack" and result_value is not True:
        raise ValueError("An acknowledgement can only be completed affirmatively.")
    options = human_task_options(canonical)
    if options and result_value not in options:
        raise ValueError(
            f"{context} output {output!r} must be one of: {', '.join(options)}."
        )
    return {output: result_value}


def validate_human_action_result(
    action: HumanAction,
    inputs: Mapping[str, object],
    result: object,
) -> dict[str, object]:
    """Apply the durable human-task policy to an in-memory backend result."""

    return validate_human_task_result(
        build_human_task_spec(action, inputs),
        result,
        context=f"Human backend for {action.name!r}",
    )
