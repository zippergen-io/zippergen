from zippergen.role_runner import _floor_coherent
from zippergen.storage_maintenance import (
    compact_store,
    inspect_store_storage,
    plan_store_compaction,
)
from zippergen.store import (
    DurableChannel,
    complete_human_task,
    ensure_human_task,
    ensure_human_task_token,
    load_snapshot,
    open_store,
    record_human_task_notification,
    record_trace_event,
    recovery_high_water,
    write_snapshot,
)


def _collectable_store(path: str) -> tuple[int, int]:
    conn = open_store(path)
    conn.execute(
        "INSERT INTO events(sender,receiver,channel,kind,payload) "
        "VALUES('A',NULL,NULL,'seed','{}')"
    )
    conn.execute(
        "INSERT INTO events(sender,receiver,channel,kind,payload) "
        "VALUES('B',NULL,NULL,'seed','{}')"
    )
    a = DurableChannel(conn, "A")
    conn.execute("BEGIN")
    message = a.put("A", "B", "main", ("hello",))
    a.commit_txn()
    b = DurableChannel(conn, "B")
    conn.execute("BEGIN")
    assert b.try_get("A", "B", "main") is not None
    b.commit_txn()
    conn.execute("BEGIN")
    journal = a.record_act(
        {
            "status": "done",
            "locator": [0],
            "action": "work",
            "outputs": {"value": "done"},
        }
    )
    a.commit_txn()
    assert a.consume_journal("act", [0]) is not None
    for index in range(5):
        record_trace_event(conn, "A", {"type": "step", "index": index})
    write_snapshot(
        conn,
        "A",
        {"value": "done"},
        [0],
        a.position(),
    )
    write_snapshot(
        conn,
        "B",
        {"value": "hello"},
        [0],
        b.position(),
    )
    conn.close()
    return message, journal


def test_storage_report_counts_files_events_and_snapshot_coverage(tmp_path):
    path = str(tmp_path / "deployment.sqlite")
    _collectable_store(path)
    conn = open_store(path)
    ensure_human_task(
        conn,
        task_id="approve-1",
        role="Human",
        locator=[0],
        action="approve",
        input_hash=None,
        inputs={"draft": "hello"},
        spec={"outputs": {"approved": "bool"}},
    )
    complete_human_task(conn, "approve-1", {"approved": True})
    ensure_human_task_token(conn, "approve-1", channel="telegram")
    record_human_task_notification(
        conn,
        "approve-1",
        channel="telegram",
        target="chat-1",
        external_id="message-1",
    )
    conn.close()

    report = inspect_store_storage(path)

    assert report.database_bytes > 0
    assert report.event_counts["seed"] == 2
    assert report.event_counts["trace"] == 5
    assert report.completed_tasks == 1
    assert report.pending_tasks == 0
    assert report.task_tokens == 1
    assert report.task_notifications == 1
    assert report.snapshot_roles == ("A", "B")
    assert report.roles_without_snapshot == ()


def test_compaction_plan_uses_both_endpoint_and_journal_floors(tmp_path):
    path = str(tmp_path / "deployment.sqlite")
    _collectable_store(path)

    plan = plan_store_compaction(path)

    assert plan.removable_messages == 1
    assert plan.removable_journal == 1
    assert plan.roles_without_snapshot == ()


def test_compaction_preserves_recovery_high_water_and_seed_rows(tmp_path):
    path = str(tmp_path / "deployment.sqlite")
    message, journal = _collectable_store(path)
    before = open_store(path)
    trace_ids = [
        int(row[0])
        for row in before.execute(
            "SELECT rowid FROM events WHERE kind='trace' ORDER BY rowid"
        ).fetchall()
    ]
    before.close()

    result = compact_store(path)
    conn = open_store(path)

    assert result.deleted_messages == 1
    assert result.deleted_journal == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM events WHERE kind='seed'"
    ).fetchone()[0] == 2
    assert conn.execute(
        "SELECT COUNT(*) FROM events WHERE kind='trace'"
    ).fetchone()[0] == 5
    assert [
        int(row[0])
        for row in conn.execute(
            "SELECT rowid FROM events WHERE kind='trace' ORDER BY rowid"
        ).fetchall()
    ] == trace_ids
    assert recovery_high_water(conn, "A") == {
        "out": message,
        "journal": journal,
    }
    snapshot = load_snapshot(conn, "A")
    assert snapshot is not None
    assert _floor_coherent(conn, "A", snapshot["floor"]) is True
    assert DurableChannel(
        conn,
        "A",
        since=snapshot["floor"],
    ).position()["out"] == message


def test_roles_without_snapshots_block_core_compaction(tmp_path):
    path = str(tmp_path / "deployment.sqlite")
    conn = open_store(path)
    conn.execute(
        "INSERT INTO events(sender,receiver,channel,kind,payload) "
        "VALUES('A',NULL,NULL,'seed','{}')"
    )
    for index in range(3):
        record_trace_event(conn, "A", {"type": "idle", "index": index})
    conn.close()

    plan = plan_store_compaction(path)
    result = compact_store(path)

    assert plan.roles_without_snapshot == ("A",)
    assert plan.removable_core == 0
    assert result.deleted_total == 0
    conn = open_store(path)
    assert conn.execute(
        "SELECT COUNT(*) FROM events WHERE kind='trace'"
    ).fetchone()[0] == 3
