"""Constrained natural-language interpretation for ZipperGen Studio.

The interpreter never returns shell commands.  It produces ordinary Studio
commands, which the Studio dispatcher validates, classifies, displays, and
executes.  Common requests are handled deterministically; a repository-aware
CLI may be used as a read-only fallback.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


NATURAL_LANGUAGE_SCHEMA_VERSION = 1
InterpretationSource = Literal["deterministic", "learned", "codex", "claude"]


@dataclass(frozen=True)
class NaturalCommandPlan:
    """A proposed sequence of Studio commands."""

    summary: str
    commands: tuple[str, ...]
    source: InterpretationSource
    clarification: str | None = None
    learned_id: str | None = None


def normalize_request(value: str) -> str:
    """Normalize prose for deterministic and learned matching."""

    normalized = " ".join(value.strip().casefold().split())
    return normalized.rstrip(" \t\r\n.?!")


_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(
        r"\b(?:api[- ]?key|access[- ]?token|secret)\s*(?:is|=|:)\s*\S+",
        re.IGNORECASE,
    ),
)


def looks_sensitive(value: str) -> bool:
    """Return whether prose appears to contain a secret value."""

    return any(pattern.search(value) for pattern in _SECRET_PATTERNS)


def _canonical_name(value: str, names: tuple[str, ...]) -> str | None:
    requested = value.strip().casefold()
    return next((name for name in names if name.casefold() == requested), None)


def _mentioned_names(value: str, names: tuple[str, ...]) -> list[str]:
    lowered = value.casefold()
    matches = [
        name
        for name in names
        if re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(name.casefold())}"
            r"(?![A-Za-z0-9_])",
            lowered,
        )
    ]
    return sorted(matches, key=len, reverse=True)


def _model_spec(value: str) -> str | None:
    explicit = re.search(
        r"\b(mock|local|ollama|openai|anthropic|claude|mistral)"
        r":[A-Za-z0-9][A-Za-z0-9_.:/-]*\b",
        value,
        re.IGNORECASE,
    )
    if explicit:
        return explicit.group(0)

    provider_then_model = re.search(
        r"\b(openai|anthropic|claude|mistral|local|ollama)"
        r"(?:\s+(?:model|model named))?\s+"
        r"([A-Za-z0-9][A-Za-z0-9_.:/-]{1,})\b",
        value,
        re.IGNORECASE,
    )
    if provider_then_model:
        provider = provider_then_model.group(1).casefold()
        provider = {"claude": "anthropic", "ollama": "local"}.get(
            provider, provider
        )
        model = provider_then_model.group(2)
        if model.casefold() not in {"model", "to", "for"}:
            return f"{provider}:{model}"

    model_then_provider = re.search(
        r"\b([A-Za-z0-9][A-Za-z0-9_.:/-]{1,})\s+"
        r"(?:from|using)\s+"
        r"(openai|anthropic|claude|mistral|local|ollama)\b",
        value,
        re.IGNORECASE,
    )
    if model_then_provider:
        provider = model_then_provider.group(2).casefold()
        provider = {"claude": "anthropic", "ollama": "local"}.get(
            provider, provider
        )
        return f"{provider}:{model_then_provider.group(1)}"
    if re.search(r"\bmock(?: model)?\b", value, re.IGNORECASE):
        return "mock"
    return None


def deterministic_plan(
    request: str,
    *,
    participants: tuple[str, ...] = (),
    llm_participants: tuple[str, ...] = (),
) -> NaturalCommandPlan | None:
    """Interpret common Studio requests without starting a model."""

    text = normalize_request(request)
    if not text:
        return None

    if re.search(r"\b(current|active)\s+task\b", text) or text in {
        "what is the task",
        "show the task",
    }:
        return NaturalCommandPlan(
            "Show the current coding-assistant task.",
            ("task",),
            "deterministic",
        )

    if (
        re.search(r"\b(current|project|workflow)\s+(state|status|context)\b", text)
        or text in {"where are we", "what is current", "show current"}
    ):
        return NaturalCommandPlan(
            "Show the complete current Studio context.",
            ("current",),
            "deterministic",
        )

    if re.search(r"\b(validate|validation|check)\b.*\bworkflow\b", text):
        return NaturalCommandPlan(
            "Validate the selected workflow.",
            ("validate",),
            "deterministic",
        )

    if re.fullmatch(
        r"(?:please\s+)?(?:run|start)(?:\s+the|\s+this|\s+current)?"
        r"\s+workflow",
        text,
    ):
        return NaturalCommandPlan(
            "Run the selected workflow.",
            ("run",),
            "deterministic",
        )

    if re.search(r"\b(?:show|display|inspect|view)\b", text):
        if re.search(r"\bcommunications?\b", text):
            return NaturalCommandPlan(
                "Show workflow communications only.",
                ("show communications",),
                "deterministic",
            )
        if re.search(r"\b(actions?|prompts?)\b", text):
            return NaturalCommandPlan(
                "Show workflow actions and prompts.",
                ("show actions",),
                "deterministic",
            )
        if re.search(r"\b(protocol|global protocol)\b", text):
            return NaturalCommandPlan(
                "Show the global workflow protocol.",
                ("show protocol",),
                "deterministic",
            )
        if re.search(r"\b(full|complete|whole)\s+workflow\b", text):
            return NaturalCommandPlan(
                "Show the complete workflow implementation.",
                ("show full",),
                "deterministic",
            )
        if re.search(r"\boverview\b", text):
            return NaturalCommandPlan(
                "Show the workflow overview.",
                ("show overview",),
                "deterministic",
            )

    mentioned = _mentioned_names(text, participants)
    if (
        len(mentioned) == 1
        and re.search(
            r"\b(local|projection|projected|sees|view|behaviou?r)\b", text
        )
    ):
        participant = mentioned[0]
        return NaturalCommandPlan(
            f"Show {participant}'s exact local projection.",
            (shlex.join(["show", "agent", participant]),),
            "deterministic",
        )

    if re.search(r"\b(show|display|list|what)\b.*\b(models?|routing)\b", text):
        return NaturalCommandPlan(
            "Show effective model routing.",
            ("models show",),
            "deterministic",
        )

    if re.search(r"\b(check|verify|test)\b.*\b(models?|routing)\b", text):
        active = _mentioned_names(text, llm_participants)
        target = active[0] if len(active) == 1 else "all"
        return NaturalCommandPlan(
            f"Check model connectivity for {target}.",
            (shlex.join(["models", "check", target]),),
            "deterministic",
        )

    if re.search(r"\b(assign|set|use|route)\b", text):
        active = _mentioned_names(text, llm_participants)
        spec = _model_spec(request)
        if len(active) == 1 and spec is not None:
            participant = active[0]
            return NaturalCommandPlan(
                f"Assign {spec} to {participant}.",
                (shlex.join(["models", "set", participant, spec]),),
                "deterministic",
            )

    if re.search(r"\b(show|display|list|what)\b.*\bproviders?\b", text):
        return NaturalCommandPlan(
            "Show configured model providers.",
            ("providers show",),
            "deterministic",
        )

    if re.search(r"\b(pending|current)\s+refinement\b", text):
        return NaturalCommandPlan(
            "Show the pending specification refinement.",
            ("spec pending",),
            "deterministic",
        )

    if re.search(r"\b(show|display|read)\b.*\bspec(?:ification)?\b", text):
        return NaturalCommandPlan(
            "Show the canonical workflow specification.",
            ("spec show",),
            "deterministic",
        )

    if re.search(r"\b(show|display|get)\b.*\b(?:deployment\s+)?logs?\b", text):
        return NaturalCommandPlan(
            "Show deployment logs.",
            ("logs",),
            "deterministic",
        )

    if re.search(r"\b(deployment|service)\s+(state|status)\b", text):
        return NaturalCommandPlan(
            "Show deployment status.",
            ("status",),
            "deterministic",
        )

    if re.search(r"\b(list|show)\b.*\bruns?\b", text):
        return NaturalCommandPlan(
            "List managed development runs.",
            ("runs",),
            "deterministic",
        )
    return None


def parse_cli_plan(value: str, *, source: Literal["codex", "claude"]) -> NaturalCommandPlan:
    """Extract and validate the last JSON object emitted by an interpreter CLI."""

    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for index, character in enumerate(value):
        if character != "{":
            continue
        try:
            decoded, _end = decoder.raw_decode(value[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            candidates.append(decoded)
    if not candidates:
        raise ValueError("The interpreter did not return a JSON command plan.")
    payload = candidates[-1]
    summary = payload.get("summary")
    commands = payload.get("commands")
    clarification = payload.get("clarification")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("The command plan is missing a summary.")
    if not isinstance(commands, list) or len(commands) > 8:
        raise ValueError("The command plan must contain at most eight commands.")
    validated: list[str] = []
    for command in commands:
        if (
            not isinstance(command, str)
            or not command.strip()
            or "\n" in command
            or "\r" in command
            or "\x00" in command
        ):
            raise ValueError("Every planned command must be one non-empty line.")
        validated.append(command.strip())
    if clarification is not None and (
        not isinstance(clarification, str) or not clarification.strip()
    ):
        raise ValueError("Plan clarification must be text or null.")
    if not validated and not clarification:
        raise ValueError("The interpreter returned neither commands nor a question.")
    return NaturalCommandPlan(
        summary.strip(),
        tuple(validated),
        source,
        clarification.strip() if isinstance(clarification, str) else None,
    )


COMMAND_CATALOG = """\
Read-only:
- current
- project show
- spec show | spec path | spec pending | spec history
- task | task show | task path | task history
- show overview | show protocol | show communications | show actions | show full
- show agent PARTICIPANT | show agents PARTICIPANT...
- validate
- models show | models check [all|default|PARTICIPANT]
- providers show | providers check local
- runs
- status [DEPLOYMENT] | doctor [DEPLOYMENT] | logs [DEPLOYMENT]

Local configuration and development:
- project init [NAME]
- use [PATH.py:WORKFLOW]
- models default PROVIDER:MODEL
- models set PARTICIPANT PROVIDER:MODEL
- models reset PARTICIPANT|all
- providers set local [BASE_URL]
- providers set openai|anthropic|mistral
- providers reset local|openai|anthropic|mistral
- create DESCRIPTION
- spec refine DESCRIPTION
- spec reconcile
- spec discard
- task close
- editor show | editor set COMMAND | editor reset
- edit workflow | edit file PATH

Execution and deployment:
- run [MODEL] [--assistant codex|claude]
- resume
- assistant [codex|claude]
- deploy [NAME] [--no-start]
- start [NAME] | restart [NAME] | stop [NAME]

Never produce exit, quit, arbitrary shell commands, chained shell syntax, secret
values, or a command not present in this catalogue.
"""


def interpreter_prompt(
    request: str,
    *,
    context: str,
) -> str:
    """Build the read-only repository-aware command-planning request."""

    return f"""\
You are the constrained natural-language interpreter for ZipperGen Studio.
Inspect the repository read-only when that is necessary to resolve workflow,
participant, file, or deployment names. Do not modify files and do not execute
state-changing commands.

Translate the user's request into zero or more commands from the exact catalogue
below. Preserve literal names and values with shell quoting when needed. If an
essential value is ambiguous or missing, return no commands and ask one concise
clarifying question. Never invent a model name, participant, workflow, provider,
path, or deployment.

Return exactly one JSON object with this shape:
{{
  "summary": "short interpretation",
  "commands": ["one Studio command per item"],
  "clarification": null
}}

Studio—not you—will validate, display, classify, confirm, and execute the plan.

## Studio command catalogue

{COMMAND_CATALOG}

## Current non-secret project context

{context}

## User request

{request}
"""


def _default_language_state() -> dict[str, Any]:
    return {
        "schema_version": NATURAL_LANGUAGE_SCHEMA_VERSION,
        "interpreter": "auto",
        "learning": True,
        "learned": [],
        "history": [],
    }


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def _next_id(records: list[dict[str, Any]], prefix: str) -> str:
    values: list[int] = []
    for record in records:
        identifier = str(record.get("id") or "")
        if identifier.startswith(prefix) and identifier[len(prefix) :].isdigit():
            values.append(int(identifier[len(prefix) :]))
    return f"{prefix}{max(values, default=0) + 1:03d}"


class NaturalLanguageStore:
    """Owner-private interpretation configuration, history, and learned examples."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return _default_language_state()
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid natural-language state {self.path}: {exc}"
            ) from exc
        if not isinstance(state, dict) or state.get(
            "schema_version"
        ) != NATURAL_LANGUAGE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported natural-language state in {self.path}."
            )
        state.setdefault("interpreter", "auto")
        state.setdefault("learning", True)
        state.setdefault("learned", [])
        state.setdefault("history", [])
        return state

    def update_settings(
        self,
        *,
        interpreter: str | None = None,
        learning: bool | None = None,
    ) -> dict[str, Any]:
        state = self.load()
        if interpreter is not None:
            state["interpreter"] = interpreter
        if learning is not None:
            state["learning"] = learning
        _atomic_write(self.path, state)
        return state

    def learned(self) -> list[dict[str, Any]]:
        value = self.load().get("learned")
        return list(value) if isinstance(value, list) else []

    def history(self) -> list[dict[str, Any]]:
        value = self.load().get("history")
        return list(value) if isinstance(value, list) else []

    def match(self, request: str) -> NaturalCommandPlan | None:
        normalized = normalize_request(request)
        state = self.load()
        records = state.get("learned")
        if not isinstance(records, list):
            return None
        for record in sorted(
            (item for item in records if isinstance(item, dict)),
            key=lambda item: (int(item.get("uses") or 0), str(item.get("id") or "")),
            reverse=True,
        ):
            template = str(record.get("request_template") or "")
            commands = record.get("commands")
            if not template or not isinstance(commands, list):
                continue
            values = _match_template(template, normalized)
            if values is None:
                continue
            try:
                rendered = tuple(
                    command.format_map(
                        {name: shlex.quote(value) for name, value in values.items()}
                    )
                    for command in commands
                    if isinstance(command, str)
                )
            except (KeyError, ValueError):
                continue
            if not rendered:
                continue
            record["uses"] = int(record.get("uses") or 0) + 1
            record["last_used_at"] = _timestamp()
            _atomic_write(self.path, state)
            return NaturalCommandPlan(
                str(record.get("summary") or "Use a learned Studio command."),
                rendered,
                "learned",
                learned_id=str(record.get("id") or ""),
            )
        return None

    def remember(
        self,
        request: str,
        plan: NaturalCommandPlan,
    ) -> dict[str, Any] | None:
        state = self.load()
        if not bool(state.get("learning", True)) or looks_sensitive(request):
            return None
        learned = state.get("learned")
        if not isinstance(learned, list):
            learned = []
            state["learned"] = learned
        request_template, command_templates = generalize_interpretation(
            request, plan.commands
        )
        for record in learned:
            if (
                isinstance(record, dict)
                and record.get("request_template") == request_template
                and record.get("commands") == list(command_templates)
            ):
                return record
        record = {
            "id": _next_id(
                [item for item in learned if isinstance(item, dict)], "L"
            ),
            "request_template": request_template,
            "commands": list(command_templates),
            "summary": plan.summary,
            "example": request.strip(),
            "source": plan.source,
            "created_at": _timestamp(),
            "uses": 0,
        }
        learned.append(record)
        state["learned"] = learned[-100:]
        _atomic_write(self.path, state)
        return record

    def record(
        self,
        request: str,
        plan: NaturalCommandPlan | None,
        *,
        status: str,
        detail: str | None = None,
    ) -> dict[str, Any] | None:
        if looks_sensitive(request):
            return None
        state = self.load()
        history = state.get("history")
        if not isinstance(history, list):
            history = []
        record = {
            "id": _next_id(
                [item for item in history if isinstance(item, dict)], "H"
            ),
            "request": request.strip(),
            "summary": plan.summary if plan else None,
            "commands": list(plan.commands) if plan else [],
            "source": plan.source if plan else None,
            "status": status,
            "detail": detail,
            "created_at": _timestamp(),
        }
        history.append(record)
        state["history"] = history[-200:]
        _atomic_write(self.path, state)
        return record

    def forget(self, identifier: str) -> int:
        state = self.load()
        learned = state.get("learned")
        if not isinstance(learned, list):
            return 0
        if identifier.casefold() == "all":
            removed = len(learned)
            state["learned"] = []
        else:
            retained = [
                record
                for record in learned
                if not isinstance(record, dict)
                or str(record.get("id") or "").casefold()
                != identifier.casefold()
            ]
            removed = len(learned) - len(retained)
            state["learned"] = retained
        if removed:
            _atomic_write(self.path, state)
        return removed


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _slot_positions(parts: list[str]) -> list[tuple[str, int]]:
    if not parts:
        return []
    command = parts[0].casefold()
    if command in {"show", "inspect"} and len(parts) == 3:
        if parts[1].casefold() == "agent":
            return [("participant", 2)]
    if command == "models" and len(parts) >= 3:
        action = parts[1].casefold()
        if action == "set" and len(parts) == 4:
            return [("participant", 2), ("model", 3)]
        if action in {"check", "reset"} and parts[2].casefold() not in {
            "all",
            "default",
        }:
            return [("participant", 2)]
    if command == "use" and len(parts) == 2:
        return [("workflow", 1)]
    if command in {"status", "doctor", "logs", "start", "restart", "stop"} and len(
        parts
    ) == 2:
        return [("deployment", 1)]
    return []


def generalize_interpretation(
    request: str,
    commands: tuple[str, ...],
) -> tuple[str, tuple[str, ...]]:
    """Generalize safely discoverable command arguments into named slots."""

    request_template = normalize_request(request)
    parsed = [shlex.split(command) for command in commands]
    command_templates = [list(parts) for parts in parsed]
    used_names: dict[str, int] = {}
    replacements: list[tuple[str, str, int, int, str | None]] = []
    for command_index, parts in enumerate(parsed):
        for base_name, part_index in _slot_positions(parts):
            value = parts[part_index]
            requested_value = value
            command_prefix: str | None = None
            if base_name == "model" and ":" in value:
                provider, model = value.split(":", 1)
                if model.casefold() in request_template:
                    requested_value = model
                    command_prefix = provider + ":"
            if requested_value.casefold() not in request_template:
                continue
            count = used_names.get(base_name, 0) + 1
            used_names[base_name] = count
            name = base_name if count == 1 else f"{base_name}{count}"
            replacements.append(
                (
                    name,
                    requested_value.casefold(),
                    command_index,
                    part_index,
                    command_prefix,
                )
            )
    for name, value, command_index, part_index, prefix in sorted(
        replacements, key=lambda item: len(item[1]), reverse=True
    ):
        placeholder = "{" + name + "}"
        updated, count = re.subn(
            re.escape(value),
            placeholder,
            request_template,
            count=1,
            flags=re.IGNORECASE,
        )
        if not count:
            continue
        request_template = updated
        sentinel = f"__ZG_{name.upper()}__"
        command_templates[command_index][part_index] = (prefix or "") + sentinel
    rendered_templates: list[str] = []
    for parts in command_templates:
        rendered = shlex.join(parts)
        for name, _value, _command_index, _part_index, _prefix in replacements:
            rendered = rendered.replace(
                f"__ZG_{name.upper()}__", "{" + name + "}"
            )
        rendered_templates.append(rendered)
    return request_template, tuple(rendered_templates)


def _match_template(template: str, request: str) -> dict[str, str] | None:
    cursor = 0
    expressions: list[str] = []
    names: list[str] = []
    for match in re.finditer(r"\{([a-z][a-z0-9_]*)\}", template):
        expressions.append(re.escape(template[cursor : match.start()]))
        name = match.group(1)
        if name in names:
            expressions.append(rf"(?P={name})")
        else:
            names.append(name)
            expressions.append(rf"(?P<{name}>.+?)")
        cursor = match.end()
    expressions.append(re.escape(template[cursor:]))
    matched = re.fullmatch("".join(expressions), request, re.IGNORECASE)
    if matched is None:
        return None
    return {
        name: value.strip()
        for name, value in matched.groupdict().items()
        if value.strip()
    }
