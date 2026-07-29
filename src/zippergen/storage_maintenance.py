"""Recovery-aware inspection and compaction for durable SQLite stores."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from zippergen.store import backfill_recovery_high_water, open_store


@dataclass(frozen=True)
class StorageReport:
    path: Path
    database_bytes: int
    wal_bytes: int
    shm_bytes: int
    reusable_bytes: int
    event_counts: dict[str, int]
    completed_tasks: int
    pending_tasks: int
    snapshot_roles: tuple[str, ...]
    roles_without_snapshot: tuple[str, ...]

    @property
    def total_bytes(self) -> int:
        return self.database_bytes + self.wal_bytes + self.shm_bytes

    @property
    def total_events(self) -> int:
        return sum(self.event_counts.values())


@dataclass(frozen=True)
class CompactionPlan:
    trace_keep: int
    trace_events: int
    removable_traces: int
    removable_messages: int
    removable_journal: int
    roles_without_snapshot: tuple[str, ...]

    @property
    def removable_core(self) -> int:
        return self.removable_messages + self.removable_journal

    @property
    def removable_total(self) -> int:
        return self.removable_traces + self.removable_core


@dataclass(frozen=True)
class CompactionResult:
    plan: CompactionPlan
    deleted_traces: int
    deleted_messages: int
    deleted_journal: int
    before_bytes: int
    after_bytes: int
    reusable_before_bytes: int
    reusable_after_bytes: int

    @property
    def deleted_total(self) -> int:
        return (
            self.deleted_traces
            + self.deleted_messages
            + self.deleted_journal
        )


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size if path.is_file() else 0
    except OSError:
        return 0


def sqlite_family_size(path: str | Path) -> tuple[int, int, int]:
    database = Path(path).expanduser()
    return (
        _file_size(database),
        _file_size(Path(f"{database}-wal")),
        _file_size(Path(f"{database}-shm")),
    )


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        ).fetchone()
        is not None
    )


def inspect_store_storage(path: str | Path) -> StorageReport:
    store = Path(path).expanduser()
    database_bytes, wal_bytes, shm_bytes = sqlite_family_size(store)
    if not store.is_file():
        return StorageReport(
            path=store,
            database_bytes=database_bytes,
            wal_bytes=wal_bytes,
            shm_bytes=shm_bytes,
            reusable_bytes=0,
            event_counts={},
            completed_tasks=0,
            pending_tasks=0,
            snapshot_roles=(),
            roles_without_snapshot=(),
        )

    conn = sqlite3.connect(f"file:{store.resolve()}?mode=ro", uri=True)
    try:
        event_counts = (
            {
                str(kind): int(count)
                for kind, count in conn.execute(
                    "SELECT kind,COUNT(*) FROM events GROUP BY kind"
                ).fetchall()
            }
            if _table_exists(conn, "events")
            else {}
        )
        seed_roles = (
            {
                str(row[0])
                for row in conn.execute(
                    "SELECT DISTINCT sender FROM events WHERE kind='seed'"
                ).fetchall()
            }
            if _table_exists(conn, "events")
            else set()
        )
        snapshot_roles = (
            {
                str(row[0])
                for row in conn.execute(
                    "SELECT role FROM snapshots"
                ).fetchall()
            }
            if _table_exists(conn, "snapshots")
            else set()
        )
        task_counts = (
            {
                str(status): int(count)
                for status, count in conn.execute(
                    "SELECT status,COUNT(*) FROM human_tasks GROUP BY status"
                ).fetchall()
            }
            if _table_exists(conn, "human_tasks")
            else {}
        )
        page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
        free_pages = int(conn.execute("PRAGMA freelist_count").fetchone()[0])
    finally:
        conn.close()

    return StorageReport(
        path=store,
        database_bytes=database_bytes,
        wal_bytes=wal_bytes,
        shm_bytes=shm_bytes,
        reusable_bytes=page_size * free_pages,
        event_counts=event_counts,
        completed_tasks=task_counts.get("done", 0),
        pending_tasks=task_counts.get("pending", 0),
        snapshot_roles=tuple(sorted(snapshot_roles)),
        roles_without_snapshot=tuple(sorted(seed_roles - snapshot_roles)),
    )


def _snapshot_floors(conn: sqlite3.Connection) -> dict[str, dict]:
    if not _table_exists(conn, "snapshots"):
        return {}
    floors: dict[str, dict] = {}
    for role, raw_floor in conn.execute(
        "SELECT role,floor FROM snapshots"
    ).fetchall():
        try:
            floor = json.loads(raw_floor)
        except (TypeError, json.JSONDecodeError):
            continue
        if (
            isinstance(floor, dict)
            and isinstance(floor.get("out"), int)
            and isinstance(floor.get("cursors"), dict)
            and isinstance(floor.get("journal"), int)
        ):
            floors[str(role)] = floor
    return floors


def _collectable_counts(
    conn: sqlite3.Connection,
    *,
    trace_keep: int,
) -> CompactionPlan:
    if trace_keep < 0:
        raise ValueError("trace_keep must be zero or greater")
    floors = _snapshot_floors(conn)
    seed_roles = {
        str(row[0])
        for row in conn.execute(
            "SELECT DISTINCT sender FROM events WHERE kind='seed'"
        ).fetchall()
    }
    trace_events = int(
        conn.execute(
            "SELECT COUNT(*) FROM events WHERE kind='trace'"
        ).fetchone()[0]
    )
    removable_traces = max(0, trace_events - trace_keep)
    removable_messages = 0
    removable_journal = 0
    for rowid, sender, receiver, channel, kind in conn.execute(
        "SELECT rowid,sender,receiver,channel,kind FROM events "
        "WHERE kind IN ('msg','ctrl','act','decision','effect') "
        "ORDER BY rowid"
    ):
        rowid = int(rowid)
        sender_floor = floors.get(str(sender))
        if sender_floor is None:
            continue
        if kind in {"msg", "ctrl"}:
            receiver_floor = floors.get(str(receiver))
            if receiver_floor is None:
                continue
            channel_key = f"{sender}|{receiver}|{channel}"
            consumed = receiver_floor["cursors"].get(channel_key, 0)
            if (
                rowid <= int(sender_floor["out"])
                and rowid <= int(consumed)
            ):
                removable_messages += 1
        elif rowid <= int(sender_floor["journal"]):
            removable_journal += 1
    return CompactionPlan(
        trace_keep=trace_keep,
        trace_events=trace_events,
        removable_traces=removable_traces,
        removable_messages=removable_messages,
        removable_journal=removable_journal,
        roles_without_snapshot=tuple(sorted(seed_roles - set(floors))),
    )


def plan_store_compaction(
    path: str | Path,
    *,
    trace_keep: int = 10_000,
) -> CompactionPlan:
    store = Path(path).expanduser()
    if not store.is_file():
        raise FileNotFoundError(store)
    conn = sqlite3.connect(f"file:{store.resolve()}?mode=ro", uri=True)
    try:
        return _collectable_counts(conn, trace_keep=trace_keep)
    finally:
        conn.close()


def compact_store(
    path: str | Path,
    *,
    trace_keep: int = 10_000,
) -> CompactionResult:
    """Delete only diagnostic traces and events covered by durable floors."""

    store = Path(path).expanduser()
    if not store.is_file():
        raise FileNotFoundError(store)
    before_bytes = sum(sqlite_family_size(store))
    before_report = inspect_store_storage(store)
    conn = open_store(str(store))
    try:
        plan = _collectable_counts(conn, trace_keep=trace_keep)
        conn.execute("BEGIN IMMEDIATE")
        try:
            backfill_recovery_high_water(conn)
            if trace_keep == 0:
                trace_cursor = conn.execute(
                    "DELETE FROM events WHERE kind='trace'"
                )
            else:
                cutoff = conn.execute(
                    "SELECT rowid FROM events WHERE kind='trace' "
                    "ORDER BY rowid DESC LIMIT 1 OFFSET ?",
                    (trace_keep - 1,),
                ).fetchone()
                trace_cursor = (
                    conn.execute(
                        "DELETE FROM events WHERE kind='trace' AND rowid<?",
                        (int(cutoff[0]),),
                    )
                    if cutoff is not None
                    else None
                )

            floors = _snapshot_floors(conn)
            conn.execute(
                "CREATE TEMP TABLE collectable_events("
                "rowid INTEGER PRIMARY KEY, category TEXT NOT NULL)"
            )
            pending: list[tuple[int, str]] = []
            for rowid, sender, receiver, channel, kind in conn.execute(
                "SELECT rowid,sender,receiver,channel,kind FROM events "
                "WHERE kind IN ('msg','ctrl','act','decision','effect')"
            ):
                rowid = int(rowid)
                sender_floor = floors.get(str(sender))
                if sender_floor is None:
                    continue
                if kind in {"msg", "ctrl"}:
                    receiver_floor = floors.get(str(receiver))
                    if receiver_floor is None:
                        continue
                    channel_key = f"{sender}|{receiver}|{channel}"
                    if (
                        rowid <= int(sender_floor["out"])
                        and rowid
                        <= int(receiver_floor["cursors"].get(channel_key, 0))
                    ):
                        pending.append((rowid, "message"))
                elif rowid <= int(sender_floor["journal"]):
                    pending.append((rowid, "journal"))
                if len(pending) >= 1_000:
                    conn.executemany(
                        "INSERT INTO collectable_events(rowid,category) "
                        "VALUES(?,?)",
                        pending,
                    )
                    pending.clear()
            if pending:
                conn.executemany(
                    "INSERT INTO collectable_events(rowid,category) VALUES(?,?)",
                    pending,
                )
            deleted_messages = int(
                conn.execute(
                    "SELECT COUNT(*) FROM collectable_events "
                    "WHERE category='message'"
                ).fetchone()[0]
            )
            deleted_journal = int(
                conn.execute(
                    "SELECT COUNT(*) FROM collectable_events "
                    "WHERE category='journal'"
                ).fetchone()[0]
            )
            conn.execute(
                "DELETE FROM events WHERE rowid IN "
                "(SELECT rowid FROM collectable_events)"
            )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        # events.rowid is an explicit INTEGER PRIMARY KEY, so VACUUM preserves
        # the stable identifiers used by recovery floors and cursors.
        conn.execute("VACUUM")
    finally:
        conn.close()
    after_bytes = sum(sqlite_family_size(store))
    after_report = inspect_store_storage(store)
    return CompactionResult(
        plan=plan,
        deleted_traces=(
            int(trace_cursor.rowcount)
            if trace_cursor is not None and trace_cursor.rowcount >= 0
            else plan.removable_traces
        ),
        deleted_messages=deleted_messages,
        deleted_journal=deleted_journal,
        before_bytes=before_bytes,
        after_bytes=after_bytes,
        reusable_before_bytes=before_report.reusable_bytes,
        reusable_after_bytes=after_report.reusable_bytes,
    )
