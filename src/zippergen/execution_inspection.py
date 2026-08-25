"""Read-only inspection of durable participant execution positions."""

from __future__ import annotations

from datetime import datetime
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from zippergen.control import ControlError, decode_control, frontier_paths
from zippergen.deployment_checks import _safe_json_loads
from zippergen.locator import resolve_path
from zippergen.projection import project
from zippergen.store import list_role_states, open_store_readonly
from zippergen.syntax import (
    Workflow,
    _ordered_workflow_lifelines,
)
from zippergen.view import describe_local_statement


@dataclass(frozen=True)
class ParticipantPosition:
    participant: str
    state: str
    locators: tuple[tuple[int, ...], ...]
    location: str
    updated_at: float | None
    detail: dict[str, object]


_STATE_LABELS = {
    "running": "running",
    "blocked": "blocked",
    "waiting_receive": "waiting to receive",
    "waiting_human": "waiting for human action",
    "running_model": "running model action",
    "running_assistant": "running assistant action",
    "running_effect": "running external effect",
    "running_action": "running action",
    "done": "completed",
    "failed": "failed",
    "cancelled": "cancelled",
    "not_started": "not started",
}

_FOCUS_PRIORITY = {
    "failed": 0,
    "waiting_human": 1,
    "running_assistant": 2,
    "running_model": 3,
    "running_effect": 4,
    "blocked": 5,
    "waiting_receive": 6,
    "running": 7,
    "cancelled": 8,
    "not_started": 9,
    "done": 10,
}


def read_execution_states(path: str | Path) -> list[dict]:
    """Read durable role state without creating or migrating a missing store."""

    store = Path(path).expanduser()
    if not store.is_file():
        return []
    connection = sqlite3.connect(f"file:{store.resolve()}?mode=ro", uri=True)
    try:
        try:
            return list_role_states(connection)
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                return []
            raise
    finally:
        connection.close()


def participant_positions(
    workflow: Workflow,
    rows: list[dict],
) -> list[ParticipantPosition]:
    """Combine persisted paths with freshly projected local programs."""

    by_role = {
        str(row.get("role")): row
        for row in rows
        if row.get("role")
    }
    positions: list[ParticipantPosition] = []
    for lifeline in _ordered_workflow_lifelines(workflow):
        row = by_role.get(lifeline.name)
        if row is None:
            positions.append(
                ParticipantPosition(
                    participant=lifeline.name,
                    state="not_started",
                    locators=(),
                    location="entry point not reached",
                    updated_at=None,
                    detail={},
                )
            )
            continue
        # The control state is the position, so derive the display from it
        # rather than storing the same fact twice.
        local = project(workflow, lifeline)
        try:
            residual = decode_control(local, row.get("control") or {})
            locators = tuple(
                tuple(path) for path in frontier_paths(local, residual)
            )
        except ControlError:
            locators = ()
        descriptions = []
        for locator in locators:
            node = resolve_path(local, list(locator))
            description = (
                describe_local_statement(node)
                if node is not None
                else f"unknown position {list(locator)}"
            )
            if description not in descriptions:
                descriptions.append(description)
        detail = row.get("detail")
        safe_detail = detail if isinstance(detail, dict) else {}
        if descriptions:
            location = " · ".join(descriptions)
        elif row.get("status") == "done":
            location = "end of local program"
        elif safe_detail.get("action"):
            location = f"action {safe_detail['action']}"
        else:
            location = "position unavailable"
        positions.append(
            ParticipantPosition(
                participant=lifeline.name,
                state=str(row.get("status") or "not_started"),
                locators=locators,
                location=location,
                updated_at=(
                    float(row["updated_at"])
                    if isinstance(row.get("updated_at"), (int, float))
                    else None
                ),
                detail={str(key): value for key, value in safe_detail.items()},
            )
        )
    return positions


def state_label(state: str) -> str:
    return _STATE_LABELS.get(state, state.replace("_", " "))


def default_focus(positions: list[ParticipantPosition]) -> str | None:
    if not positions:
        return None
    return min(
        enumerate(positions),
        key=lambda value: (
            _FOCUS_PRIORITY.get(value[1].state, 50),
            value[0],
        ),
    )[1].participant


# ---------------------------------------------------------------------------
# Trace interpretation
#
# Reading history rows and pairing each action's start with its end is
# interpretation, not argument parsing, so it lives beside the other durable
# observation code. `serve.py` renders what these return.
# ---------------------------------------------------------------------------

def _load_trace_events(
    store_path: str,
    *,
    after_rowid: int = 0,
    limit: int = 50,
    newest: bool = True,
) -> list[dict]:
    if limit <= 0:
        raise SystemExit("--tail must be greater than 0.")
    path = Path(store_path).expanduser()
    if not path.exists():
        raise SystemExit(f"Store does not exist: {store_path}")
    # A trace refresh is observational. In particular, do not use open_store()
    # here: its schema-claim transaction takes the WAL writer lock and can make
    # readers wait for gaps in a busy multi-lifeline workflow.
    conn = open_store_readonly(path)
    try:
        order = "DESC" if newest else "ASC"
        rows = conn.execute(
            "SELECT id, role, payload FROM history WHERE id>? "
            f"ORDER BY id {order} LIMIT ?",
            (after_rowid, limit),
        ).fetchall()
    finally:
        conn.close()
    if newest:
        rows = list(reversed(rows))
    return [
        {
            "rowid": row[0],
            "role": row[1],
            "event": _safe_json_loads(row[2]),
        }
        for row in rows
    ]


def _trace_seconds(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    timestamp = float(value)
    if not math.isfinite(timestamp):
        return None
    return timestamp


def _trace_value(value: object, *, limit: int = 120) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _control_values(event: object) -> tuple[bool, list[object]]:
    """Is this a control broadcast, and what did it carry that a person sent?

    The answer is read from the event, never inferred from the payload. The
    runtime knows which sends are control broadcasts -- it is executing the
    statement -- and records that, so no reader has to guess from a value's
    contents. Guessing is what once let an ordinary workflow string be
    classified as control and dropped from the trace.

    An event with no flag is an ordinary send, and its payload is shown.
    """

    if not isinstance(event, dict):
        return False, []
    values = event.get("values")
    return bool(event.get("control")), values if isinstance(values, list) else []


def _trace_row(
    item: dict,
    *,
    duration: float | None = None,
    incomplete: bool = False,
) -> tuple[str, str, str, str, str]:
    role = str(item.get("role") or "-")
    event = item.get("event")
    if not isinstance(event, dict):
        return "—", f"#{item['rowid']}", role, "event", _trace_value(event)

    timestamp = _trace_time(event.get("recorded_at"))
    event_type = event.get("type", "event")
    if event_type == "send":
        source = event.get("from", role)
        target = event.get("to", "?")
        channel = event.get("channel") or "-"
        control, visible_values = _control_values(event)
        if control:
            detail = f"{source} → {target} [{channel}]"
            if visible_values:
                detail += f" · value={_trace_value(visible_values[0])}"
            return timestamp, f"#{item['rowid']}", role, "control send", detail
        detail = f"{source} → {target} [{channel}]"
        bindings = event.get("bindings") or {}
        if bindings:
            detail += f" · {_trace_fields(bindings)}"
        return timestamp, f"#{item['rowid']}", role, "send", detail
    if event_type == "recv":
        source = event.get("from", "?")
        target = event.get("to", role)
        channel = event.get("channel") or "-"
        bindings = event.get("bindings") or {}
        detail = f"{source} → {target} [{channel}]"
        if bindings:
            detail += f" · {_trace_fields(bindings)}"
        kind = "control receive" if event.get("ctrl") else "receive"
        return timestamp, f"#{item['rowid']}", role, kind, detail
    if event_type in {"act_start", "act", "act_failed"}:
        action = event.get("action", "?")
        action_kind = event.get("action_kind") or "action"
        phase = (
            "incomplete"
            if event_type == "act_start" and incomplete
            else "start" if event_type == "act_start"
            else "failed" if event_type == "act_failed"
            else "done"
        )
        detail = str(action)
        seq = event.get("seq")
        if seq is not None:
            detail += f" · seq={seq}"
        payload = (
            event.get("inputs")
            if event_type == "act_start"
            else event.get("outputs")
            if event_type == "act"
            else {
                key: event[key]
                for key in ("error", "message")
                if event.get(key)
            }
        ) or {}
        if payload:
            detail += f" · {_trace_fields(payload)}"
        if duration is not None:
            detail += f" · {_trace_duration(duration)}"
        if incomplete:
            detail += " · no completion recorded"
        return (
            timestamp,
            f"#{item['rowid']}",
            role,
            f"{action_kind} {phase}",
            detail,
        )
    if event_type == "llm_retry":
        action = event.get("action", "?")
        detail = str(action)
        if event.get("detail"):
            detail += f" · {event['detail']}"
        return timestamp, f"#{item['rowid']}", role, "LLM retry", detail
    if event_type == "decision":
        decision_kind = event.get("kind", "if")
        value = event.get("value")
        label = (
            "continue" if value else "exit"
        ) if decision_kind == "while" else ("true" if value else "false")
        detail = f"{decision_kind} → {label}"
        condition = event.get("formula") or event.get("condition")
        if condition:
            detail += f" · {condition}"
        return timestamp, f"#{item['rowid']}", role, "decision", detail

    remainder = {
        key: value
        for key, value in event.items()
        if key not in {"type", "recorded_at"}
    }
    return (
        timestamp,
        f"#{item['rowid']}",
        role,
        str(event_type),
        _trace_fields(remainder),
    )


def _trace_rows(
    events: list[dict],
    *,
    mark_unmatched_incomplete: bool = True,
) -> list[tuple[object, ...]]:
    starts: dict[tuple[object, ...], list[tuple[int, float | None]]] = {}
    matched_starts: set[int] = set()
    durations: dict[int, float] = {}
    for item in events:
        role = str(item.get("role") or "-")
        event = item.get("event")
        if not isinstance(event, dict):
            continue
        timestamp = _trace_seconds(event.get("recorded_at"))
        attempt_id = event.get("attempt_id")
        seq = event.get("seq")
        key: tuple[object, ...] | None
        if isinstance(attempt_id, str) and attempt_id:
            key = ("attempt", attempt_id)
        elif type(seq) is int:
            key = ("sequence", role, seq)
        else:
            key = None
        if key is None:
            continue
        if event.get("type") == "act_start":
            starts.setdefault(key, []).append((int(item["rowid"]), timestamp))
        elif event.get("type") in {"act", "act_failed"} and starts.get(key):
            start_rowid, started_at = starts[key].pop()
            matched_starts.add(start_rowid)
            stored_ms = event.get("duration_ms")
            if isinstance(stored_ms, (int, float)) and stored_ms >= 0:
                durations[int(item["rowid"])] = stored_ms / 1000
            elif timestamp is not None and started_at is not None:
                measured = timestamp - started_at
                if measured >= 0:
                    durations[int(item["rowid"])] = measured

    rows: list[tuple[object, ...]] = []
    for item in events:
        event = item.get("event")
        rowid = int(item["rowid"])
        incomplete = (
            mark_unmatched_incomplete
            and isinstance(event, dict)
            and event.get("type") == "act_start"
            and rowid not in matched_starts
        )
        duration = durations.get(rowid)
        if duration is None and isinstance(event, dict):
            stored_ms = event.get("duration_ms")
            if isinstance(stored_ms, (int, float)) and stored_ms >= 0:
                duration = stored_ms / 1000
        rows.append(
            _trace_row(
                item,
                duration=duration,
                incomplete=incomplete,
            )
        )
    return rows


def _trace_time(value: object) -> str:
    timestamp = _trace_seconds(value)
    if timestamp is None:
        return "—"
    try:
        return datetime.fromtimestamp(timestamp).astimezone().isoformat(
            sep=" ", timespec="milliseconds"
        )
    except (OSError, OverflowError, ValueError):
        return "—"


def _trace_fields(value: object) -> str:
    if not isinstance(value, dict):
        return _trace_value(value)
    return " · ".join(
        f"{name}={_trace_value(field)}" for name, field in value.items()
    )


def _trace_duration(seconds: float) -> str:
    if seconds < 0.001:
        return "<1ms"
    if seconds < 1:
        return f"{round(seconds * 1000)}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)}m {remainder:.1f}s"
