"""Read-only inspection of durable participant execution positions."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from zippergen.locator import resolve_path
from zippergen.projection import project
from zippergen.store import list_execution_states
from zippergen.syntax import Workflow, _ordered_workflow_lifelines
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
    """Read observation rows without creating or migrating a missing store."""

    store = Path(path).expanduser()
    if not store.is_file():
        return []
    connection = sqlite3.connect(f"file:{store.resolve()}?mode=ro", uri=True)
    try:
        try:
            return list_execution_states(connection)
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
        raw_locators = row.get("locators")
        locators = tuple(
            tuple(int(index) for index in path)
            for path in raw_locators
            if isinstance(path, list)
        ) if isinstance(raw_locators, list) else ()
        local = project(workflow, lifeline)
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
        elif row.get("state") == "done":
            location = "end of local program"
        elif safe_detail.get("action"):
            location = f"action {safe_detail['action']}"
        else:
            location = "position unavailable"
        positions.append(
            ParticipantPosition(
                participant=lifeline.name,
                state=str(row.get("state") or "not_started"),
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
