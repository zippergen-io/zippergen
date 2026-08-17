"""Inspection and housekeeping for durable SQLite stores.

There is no compaction here, and nothing to prove. Durable state is the current
state of the computation, so it is already bounded by the size of that
computation: one row per role, plus messages nobody has absorbed yet.

The only thing that accumulates is ``history``, which recovery never reads.
Pruning it is a retention choice, not a correctness argument.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path

from zippergen.store import open_store, prune_history


@dataclass(frozen=True)
class StorageReport:
    path: Path
    database_bytes: int
    wal_bytes: int
    shm_bytes: int
    reusable_bytes: int
    roles: int
    outstanding_messages: int
    history_rows: int
    completed_tasks: int
    pending_tasks: int
    task_tokens: int
    task_notifications: int
    workflow_results: int
    connector_entries: int
    integrity_ok: bool | None
    integrity_detail: str


@dataclass(frozen=True)
class StoreIntegrity:
    ok: bool
    detail: str


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


def _count(conn: sqlite3.Connection, table: str) -> int:
    if not _table_exists(conn, table):
        return 0
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def check_store_integrity(path: str | Path) -> StoreIntegrity:
    """Run SQLite's bounded diagnostic check through a read-only connection."""

    store = Path(path).expanduser()
    if not store.is_file():
        return StoreIntegrity(False, f"store does not exist: {store}")
    try:
        conn = sqlite3.connect(
            f"file:{store.resolve()}?mode=ro",
            uri=True,
            timeout=5.0,
        )
        try:
            rows = conn.execute("PRAGMA quick_check(1)").fetchall()
        finally:
            conn.close()
    except (OSError, sqlite3.DatabaseError) as exc:
        return StoreIntegrity(False, f"{type(exc).__name__}: {exc}")
    details = [str(row[0]) for row in rows if row]
    if details == ["ok"]:
        return StoreIntegrity(True, "SQLite quick check passed")
    return StoreIntegrity(
        False,
        details[0] if details else "SQLite quick check returned no result",
    )


def inspect_store_storage(path: str | Path) -> StorageReport:
    store = Path(path).expanduser()
    database_bytes, wal_bytes, shm_bytes = sqlite_family_size(store)
    empty = StorageReport(
        path=store,
        database_bytes=database_bytes,
        wal_bytes=wal_bytes,
        shm_bytes=shm_bytes,
        reusable_bytes=0,
        roles=0,
        outstanding_messages=0,
        history_rows=0,
        completed_tasks=0,
        pending_tasks=0,
        task_tokens=0,
        task_notifications=0,
        workflow_results=0,
        connector_entries=0,
        integrity_ok=None,
        integrity_detail="store does not exist",
    )
    if not store.is_file():
        return empty

    integrity = check_store_integrity(store)
    if not integrity.ok:
        return replace(
            empty,
            integrity_ok=False,
            integrity_detail=integrity.detail,
        )

    conn = sqlite3.connect(f"file:{store.resolve()}?mode=ro", uri=True)
    try:
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
        report = StorageReport(
            path=store,
            database_bytes=database_bytes,
            wal_bytes=wal_bytes,
            shm_bytes=shm_bytes,
            reusable_bytes=(
                int(conn.execute("PRAGMA page_size").fetchone()[0])
                * int(conn.execute("PRAGMA freelist_count").fetchone()[0])
            ),
            roles=_count(conn, "role_state"),
            outstanding_messages=_count(conn, "outstanding_messages"),
            history_rows=_count(conn, "history"),
            completed_tasks=task_counts.get("done", 0),
            pending_tasks=task_counts.get("pending", 0),
            task_tokens=_count(conn, "human_task_tokens"),
            task_notifications=_count(conn, "human_task_notifications"),
            workflow_results=_count(conn, "workflow_results"),
            connector_entries=_count(conn, "adapter_state"),
            integrity_ok=True,
            integrity_detail=integrity.detail,
        )
    finally:
        conn.close()
    return report


@dataclass(frozen=True)
class HistoryPruneResult:
    removed_rows: int
    before_bytes: int
    after_bytes: int


def prune_store_history(
    path: str | Path,
    *,
    keep: int = 0,
) -> HistoryPruneResult:
    """Drop optional history and reclaim the space.

    Recovery never reads history, so this is safe at any moment, including
    while the deployment is running. ``keep=0`` removes all of it.
    """

    store = Path(path).expanduser()
    if not store.is_file():
        raise FileNotFoundError(store)
    before_bytes = sum(sqlite_family_size(store))
    conn = open_store(str(store))
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            removed = prune_history(conn, keep=keep)
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("VACUUM")
        except sqlite3.OperationalError:
            # Reclaiming space needs exclusive access. The rows are already
            # gone, so a running deployment just means the file shrinks later.
            pass
    finally:
        conn.close()
    return HistoryPruneResult(
        removed_rows=removed,
        before_bytes=before_bytes,
        after_bytes=sum(sqlite_family_size(store)),
    )
