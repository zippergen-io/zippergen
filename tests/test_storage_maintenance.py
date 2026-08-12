"""Store inspection, and history pruning as a retention choice.

There is no compaction to test. Durable state is bounded by the computation, so
the only thing that grows is history, and history is not part of recovery.
"""

import json

import pytest

from zippergen.storage_maintenance import (
    check_store_integrity,
    inspect_store_storage,
    prune_store_history,
)
from zippergen.store import (
    ensure_human_task,
    ensure_human_task_token,
    open_store,
    record_history,
    record_human_task_notification,
    write_role_state,
)


def _populated_store(path: str, *, history_rows: int = 5) -> None:
    conn = open_store(path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        write_role_state(
            conn,
            "A",
            env={"x": 1},
            control={"k": "done"},
            monitor=None,
            steps=3,
            status="done",
        )
        conn.execute(
            "INSERT INTO outstanding_messages(sender,receiver,channel,payload) "
            "VALUES('A','B','main',?)",
            (json.dumps([1]),),
        )
        task, _created = ensure_human_task(
            conn,
            task_id="task-1",
            role="A",
            locator=[0],
            action="approve",
            input_hash=None,
            inputs={},
            spec={},
        )
        ensure_human_task_token(conn, "task-1")
        record_human_task_notification(
            conn, "task-1", channel="telegram", target="42"
        )
        conn.execute("COMMIT")
        for index in range(history_rows):
            record_history(conn, "A", {"type": "step", "index": index})
    finally:
        conn.close()


def test_the_report_counts_state_messages_and_history(tmp_path):
    path = str(tmp_path / "s.sqlite")
    _populated_store(path)

    report = inspect_store_storage(path)

    assert report.roles == 1
    assert report.outstanding_messages == 1
    assert report.history_rows == 5
    assert report.pending_tasks == 1
    assert report.completed_tasks == 0
    assert report.task_tokens == 1
    assert report.task_notifications == 1
    assert report.integrity_ok is True


def test_the_report_is_empty_for_a_missing_store(tmp_path):
    report = inspect_store_storage(tmp_path / "absent.sqlite")
    assert report.integrity_ok is None
    assert report.roles == 0


def test_integrity_check_reports_a_malformed_database(tmp_path):
    broken = tmp_path / "broken.sqlite"
    broken.write_bytes(b"this is definitely not a database")

    integrity = check_store_integrity(broken)

    assert integrity.ok is False
    assert integrity.detail


def test_pruning_history_leaves_recovery_state_untouched(tmp_path):
    """The invariant that makes history retention a free choice."""

    path = str(tmp_path / "s.sqlite")
    _populated_store(path, history_rows=40)

    outcome = prune_store_history(path, keep=0)

    assert outcome.removed_rows == 40
    report = inspect_store_storage(path)
    assert report.history_rows == 0
    assert report.roles == 1
    assert report.outstanding_messages == 1
    assert report.pending_tasks == 1


def test_pruning_can_keep_the_newest_rows(tmp_path):
    path = str(tmp_path / "s.sqlite")
    _populated_store(path, history_rows=40)

    prune_store_history(path, keep=10)

    assert inspect_store_storage(path).history_rows == 10


def test_pruning_a_missing_store_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        prune_store_history(tmp_path / "absent.sqlite")
