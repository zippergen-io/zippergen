"""Store inspection, and history pruning as a retention choice.

There is no compaction to test. Durable state is bounded by the computation, so
the only thing that grows is history, and history is not part of recovery.
"""

import json

import pytest

from zippergen.storage_maintenance import (
    check_store_integrity,
    initialize_store_history_keep,
    inspect_store_storage,
    prune_store_history,
    set_store_history_keep,
)
from zippergen.store import (
    HISTORY_KEEP_DEFAULT,
    StoreSchemaError,
    ensure_human_task,
    ensure_human_task_token,
    list_history,
    open_store,
    read_history_high_water,
    read_history_keep,
    record_history,
    record_human_task_notification,
    write_history_keep,
    write_meta,
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
            spec={
                "kind": "confirm",
                "output": "approved",
                "output_type": "bool",
            },
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


# ---------------------------------------------------------------------------
# The history budget: how much trace a store keeps
# ---------------------------------------------------------------------------


def test_a_store_keeps_the_default_budget_until_it_is_told_otherwise(tmp_path):
    path = str(tmp_path / "s.sqlite")
    _populated_store(path)

    assert inspect_store_storage(path).history_keep == HISTORY_KEEP_DEFAULT


def test_the_budget_bounds_what_a_running_store_accumulates(tmp_path):
    """The point of the budget: writing more does not grow the store."""

    path = str(tmp_path / "s.sqlite")
    conn = open_store(path)
    try:
        write_history_keep(conn, 20)
        for index in range(500):
            record_history(conn, "A", {"type": "step", "index": index})
    finally:
        conn.close()

    report = inspect_store_storage(path)
    assert report.history_keep == 20
    # Trimming happens in batches, so the table sits a little over budget in
    # between. A tenth is the documented headroom.
    assert 20 <= report.history_rows <= 22


def test_the_newest_events_are_the_ones_kept(tmp_path):
    path = str(tmp_path / "s.sqlite")
    conn = open_store(path)
    try:
        write_history_keep(conn, 5)
        for index in range(100):
            record_history(conn, "A", {"type": "step", "index": index})
        kept = [row["event"]["index"] for row in list_history(conn)]
    finally:
        conn.close()

    assert kept == sorted(kept)
    assert kept[-1] == 99


def test_a_budget_of_zero_records_nothing_at_all(tmp_path):
    path = str(tmp_path / "s.sqlite")
    conn = open_store(path)
    try:
        write_history_keep(conn, 0)
        assert record_history(conn, "A", {"type": "step"}) == 0
    finally:
        conn.close()

    assert inspect_store_storage(path).history_rows == 0


def test_lowering_the_budget_applies_immediately(tmp_path):
    """A budget nobody has reached yet would otherwise be a promise, not a fact.

    With a budget of zero nothing is ever written again, so no later write can
    trim what is already there. Setting the budget has to do it.
    """

    path = str(tmp_path / "s.sqlite")
    _populated_store(path, history_rows=40)

    outcome = set_store_history_keep(path, 0)

    assert outcome.removed_rows == 40
    report = inspect_store_storage(path)
    assert report.history_rows == 0
    assert report.history_keep == 0
    # The invariant that makes this safe.
    assert report.roles == 1
    assert report.outstanding_messages == 1
    assert report.pending_tasks == 1


def test_the_budget_survives_reopening_the_store(tmp_path):
    path = str(tmp_path / "s.sqlite")
    conn = open_store(path)
    try:
        write_history_keep(conn, 250)
    finally:
        conn.close()

    conn = open_store(path)
    try:
        assert read_history_keep(conn) == 250
    finally:
        conn.close()


def test_pruning_with_no_number_uses_the_stores_own_budget(tmp_path):
    """A bare 'tidy this up' trims to the budget, it does not empty the store."""

    path = str(tmp_path / "s.sqlite")
    _populated_store(path, history_rows=40)
    conn = open_store(path)
    try:
        write_history_keep(conn, 30)
    finally:
        conn.close()

    prune_store_history(path)

    assert inspect_store_storage(path).history_rows == 30


def test_a_negative_budget_is_refused(tmp_path):
    path = str(tmp_path / "s.sqlite")
    _populated_store(path)

    with pytest.raises(ValueError):
        set_store_history_keep(path, -1)


def test_a_hand_edited_budget_is_reported_not_ignored(tmp_path):
    path = str(tmp_path / "s.sqlite")
    _populated_store(path)
    conn = open_store(path)
    try:
        write_meta(conn, "history_keep", "lots")
    finally:
        conn.close()

    with pytest.raises(StoreSchemaError):
        inspect_store_storage(path)


def test_a_budget_can_be_set_before_the_store_exists(tmp_path):
    """A run records where its store will be before anything opens it.

    Setting the budget at that moment has to create the store rather than fail,
    which is the difference between this and ``set_store_history_keep``.
    """

    path = tmp_path / "not-yet.sqlite"
    assert not path.exists()

    initialize_store_history_keep(path, 25)

    assert path.is_file()
    assert inspect_store_storage(path).history_keep == 25


def test_initializing_a_budget_leaves_an_existing_store_alone(tmp_path):
    path = str(tmp_path / "s.sqlite")
    _populated_store(path, history_rows=5)

    initialize_store_history_keep(path, 3)

    report = inspect_store_storage(path)
    assert report.history_keep == 3
    assert report.roles == 1
    assert report.outstanding_messages == 1


def test_event_numbers_never_repeat_after_the_trace_is_turned_off(tmp_path):
    """Event numbers are the stored order that ``trace --after`` pages through.

    ``history`` uses a plain INTEGER PRIMARY KEY, so SQLite starts again from 1
    once the table is empty — and a budget of zero empties it. Reusing a number
    would make ``--after N`` skip every new event until the counter caught up.
    """

    path = str(tmp_path / "s.sqlite")
    conn = open_store(path)
    try:
        write_history_keep(conn, 25)
        for index in range(25):
            record_history(conn, "A", {"type": "step", "index": index})
        before = [row["rowid"] for row in list_history(conn)]

        write_history_keep(conn, 0)
        write_history_keep(conn, 25)
        for index in range(3):
            record_history(conn, "A", {"type": "step", "index": 100 + index})
        after = [row["rowid"] for row in list_history(conn)]
    finally:
        conn.close()

    assert before == list(range(1, 26))
    assert min(after) > max(before)
    conn = open_store(path)
    try:
        # The paging a trace viewer does still reaches the new events.
        assert [row["rowid"] for row in list_history(conn, after_id=max(before))] == after
    finally:
        conn.close()


def test_the_high_water_mark_survives_reopening(tmp_path):
    path = str(tmp_path / "s.sqlite")
    conn = open_store(path)
    try:
        write_history_keep(conn, 10)
        for index in range(10):
            record_history(conn, "A", {"type": "step", "index": index})
        write_history_keep(conn, 0)
    finally:
        conn.close()

    conn = open_store(path)
    try:
        assert read_history_high_water(conn) == 10
        write_history_keep(conn, 5)
        assert record_history(conn, "A", {"type": "step"}) == 11
    finally:
        conn.close()


def test_ordinary_trimming_does_not_disturb_event_numbers(tmp_path):
    """Trimming to a positive budget keeps the newest rows, so ids just continue."""

    path = str(tmp_path / "s.sqlite")
    conn = open_store(path)
    try:
        write_history_keep(conn, 5)
        for index in range(60):
            record_history(conn, "A", {"type": "step", "index": index})
        ids = [row["rowid"] for row in list_history(conn)]
    finally:
        conn.close()

    assert ids == sorted(ids)
    assert ids == list(range(ids[0], ids[0] + len(ids)))
    assert ids[-1] == 60
