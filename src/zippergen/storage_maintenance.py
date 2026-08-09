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
    task_tokens: int
    task_notifications: int
    snapshot_roles: tuple[str, ...]
    roles_without_snapshot: tuple[str, ...]
    integrity_ok: bool | None
    integrity_detail: str

@dataclass(frozen=True)
class CompactionPlan:
    removable_messages: int
    removable_journal: int
    roles_without_snapshot: tuple[str, ...]

    @property
    def removable_core(self) -> int:
        return self.removable_messages + self.removable_journal


@dataclass(frozen=True)
class CompactionResult:
    plan: CompactionPlan
    deleted_messages: int
    deleted_journal: int
    before_bytes: int
    after_bytes: int
    reusable_before_bytes: int
    reusable_after_bytes: int

    @property
    def deleted_total(self) -> int:
        return self.deleted_messages + self.deleted_journal


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
            task_tokens=0,
            task_notifications=0,
            snapshot_roles=(),
            roles_without_snapshot=(),
            integrity_ok=None,
            integrity_detail="store does not exist",
        )

    integrity = check_store_integrity(store)
    if not integrity.ok:
        return StorageReport(
            path=store,
            database_bytes=database_bytes,
            wal_bytes=wal_bytes,
            shm_bytes=shm_bytes,
            reusable_bytes=0,
            event_counts={},
            completed_tasks=0,
            pending_tasks=0,
            task_tokens=0,
            task_notifications=0,
            snapshot_roles=(),
            roles_without_snapshot=(),
            integrity_ok=False,
            integrity_detail=integrity.detail,
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
        task_tokens = (
            int(
                conn.execute(
                    "SELECT COUNT(*) FROM human_task_tokens"
                ).fetchone()[0]
            )
            if _table_exists(conn, "human_task_tokens")
            else 0
        )
        task_notifications = (
            int(
                conn.execute(
                    "SELECT COUNT(*) FROM human_task_notifications"
                ).fetchone()[0]
            )
            if _table_exists(conn, "human_task_notifications")
            else 0
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
        task_tokens=task_tokens,
        task_notifications=task_notifications,
        snapshot_roles=tuple(sorted(snapshot_roles)),
        roles_without_snapshot=tuple(sorted(seed_roles - snapshot_roles)),
        integrity_ok=True,
        integrity_detail=integrity.detail,
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


def _collectable_counts(conn: sqlite3.Connection) -> CompactionPlan:
    floors = _snapshot_floors(conn)
    seed_roles = {
        str(row[0])
        for row in conn.execute(
            "SELECT DISTINCT sender FROM events WHERE kind='seed'"
        ).fetchall()
    }
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
        removable_messages=removable_messages,
        removable_journal=removable_journal,
        roles_without_snapshot=tuple(sorted(seed_roles - set(floors))),
    )


def plan_store_compaction(path: str | Path) -> CompactionPlan:
    store = Path(path).expanduser()
    if not store.is_file():
        raise FileNotFoundError(store)
    conn = sqlite3.connect(f"file:{store.resolve()}?mode=ro", uri=True)
    try:
        return _collectable_counts(conn)
    finally:
        conn.close()


def compact_store(path: str | Path) -> CompactionResult:
    """Delete only recovery events covered by durable snapshot floors."""

    store = Path(path).expanduser()
    if not store.is_file():
        raise FileNotFoundError(store)
    integrity = check_store_integrity(store)
    if not integrity.ok:
        raise ValueError(
            f"Store integrity check failed before compaction: {integrity.detail}"
        )
    before_bytes = sum(sqlite_family_size(store))
    before_report = inspect_store_storage(store)
    conn = open_store(str(store))
    try:
        plan = _collectable_counts(conn)
        conn.execute("BEGIN IMMEDIATE")
        try:
            backfill_recovery_high_water(conn)
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
        deleted_messages=deleted_messages,
        deleted_journal=deleted_journal,
        before_bytes=before_bytes,
        after_bytes=after_bytes,
        reusable_before_bytes=before_report.reusable_bytes,
        reusable_after_bytes=after_report.reusable_bytes,
    )
