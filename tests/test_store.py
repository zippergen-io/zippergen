import sqlite3
import pytest
from zippergen.store import (
    complete_human_task,
    ensure_human_task,
    ensure_human_task_token,
    human_task_id,
    load_adapter_state,
    list_trace_events,
    list_workflow_results,
    load_human_task,
    load_human_task_notification,
    load_human_task_token,
    load_workflow_result,
    mark_human_task_token_used,
    open_store,
    record_trace_event,
    record_human_task_notification,
    chan_key,
    ReplayMismatch,
    write_adapter_state,
    write_workflow_result,
)

def test_open_store_creates_tables(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {
        "events",
        "cursors",
        "snapshots",
        "human_tasks",
        "human_task_tokens",
        "human_task_notifications",
        "adapter_state",
        "workflow_results",
    } <= names

def test_open_store_is_wal(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"

def test_insert_event_autoincrements_rowid(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    conn.execute("BEGIN")
    c1 = conn.execute("INSERT INTO events(sender,receiver,channel,kind,payload,causal_stamp)"
                      " VALUES('A','B','main','msg','[1]',NULL)")
    c2 = conn.execute("INSERT INTO events(sender,receiver,channel,kind,payload,causal_stamp)"
                      " VALUES('A','B','main','msg','[2]',NULL)")
    conn.execute("COMMIT")
    assert c2.lastrowid == c1.lastrowid + 1

def test_chan_key_roundtrip():
    assert chan_key("A", "B", "main") == "A|B|main"
    assert chan_key("A", "B", "main").split("|") == ["A", "B", "main"]


def test_human_task_lifecycle(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    task_id = human_task_id("A", [0], "abc", 0)
    conn.execute("BEGIN")
    task, created = ensure_human_task(
        conn,
        task_id=task_id,
        role="A",
        locator=[0],
        action="review",
        input_hash="abc",
        inputs={"prompt": "plan"},
        spec={"kind": "confirm", "output": "approved"},
    )
    conn.execute("COMMIT")
    assert created is True
    assert task["status"] == "pending"
    assert task["inputs"] == {"prompt": "plan"}

    conn.execute("BEGIN")
    same, created_again = ensure_human_task(
        conn,
        task_id=task_id,
        role="A",
        locator=[0],
        action="review",
        input_hash="abc",
        inputs={"prompt": "changed"},
        spec={"kind": "confirm", "output": "approved"},
    )
    conn.execute("COMMIT")
    assert created_again is False
    assert same["inputs"] == {"prompt": "plan"}

    conn.execute("BEGIN")
    done = complete_human_task(conn, task_id, {"approved": True})
    conn.execute("COMMIT")
    assert done["status"] == "done"
    assert load_human_task(conn, task_id)["result"] == {"approved": True}

    conn.execute("BEGIN")
    still_done = complete_human_task(conn, task_id, {"approved": False})
    conn.execute("COMMIT")
    assert still_done["result"] == {"approved": True}


def test_human_task_token_lifecycle(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    task_id = human_task_id("A", [0], "abc", 0)
    ensure_human_task(
        conn,
        task_id=task_id,
        role="A",
        locator=[0],
        action="review",
        input_hash="abc",
        inputs={"prompt": "plan"},
        spec={"kind": "confirm", "output": "approved"},
    )

    first = ensure_human_task_token(conn, task_id, channel="email")
    second = ensure_human_task_token(conn, task_id, channel="email")
    other = ensure_human_task_token(conn, task_id, channel="telegram")

    assert first == second
    assert first["token"].startswith("zg_")
    assert first["channel"] == "email"
    assert other["token"] != first["token"]
    assert load_human_task_token(conn, first["token"])["task_id"] == task_id

    used = mark_human_task_token_used(conn, first["token"])
    assert used["used_at"] is not None


def test_human_task_notification_lifecycle(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    task_id = human_task_id("A", [0], "abc", 0)
    ensure_human_task(
        conn,
        task_id=task_id,
        role="A",
        locator=[0],
        action="review",
        input_hash="abc",
        inputs={"prompt": "plan"},
        spec={"kind": "confirm", "output": "approved"},
    )

    first = record_human_task_notification(
        conn,
        task_id,
        channel="telegram",
        target="123",
        external_id="msg-1",
    )
    second = record_human_task_notification(
        conn,
        task_id,
        channel="telegram",
        target="123",
        external_id=None,
    )

    assert first["task_id"] == task_id
    assert first["external_id"] == "msg-1"
    assert second["external_id"] == "msg-1"
    assert second["sent_at"] >= first["sent_at"]
    assert load_human_task_notification(
        conn,
        task_id,
        channel="telegram",
        target="123",
    ) == second


def test_adapter_state_lifecycle(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    assert load_adapter_state(conn, "telegram:offset", 0) == 0

    write_adapter_state(conn, "telegram:offset", 42)

    assert load_adapter_state(conn, "telegram:offset") == 42


def test_workflow_result_lifecycle(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    assert load_workflow_result(conn, "wf") is None

    write_workflow_result(conn, "wf", (1, True))
    assert load_workflow_result(conn, "wf") == [1, True]

    created_at = conn.execute(
        "SELECT created_at FROM workflow_results WHERE workflow='wf'"
    ).fetchone()[0]
    write_workflow_result(conn, "wf", {"answer": 2})
    assert load_workflow_result(conn, "wf") == {"answer": 2}
    row = conn.execute(
        "SELECT COUNT(*), created_at FROM workflow_results WHERE workflow='wf'"
    ).fetchone()
    assert row == (1, created_at)
    results = list_workflow_results(conn)
    assert len(results) == 1
    assert results[0]["workflow"] == "wf"
    assert results[0]["value"] == {"answer": 2}
    assert results[0]["created_at"] == created_at
    assert results[0]["updated_at"] >= created_at


def test_trace_event_lifecycle(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    first = record_trace_event(conn, "A", {"type": "send", "from": "A", "to": "B", "values": (1,)})
    second = record_trace_event(conn, "B", {"type": "recv", "from": "A", "to": "B", "bindings": {"n": 1}})

    assert list_trace_events(conn) == [
        {"rowid": first, "event": {"type": "send", "from": "A", "to": "B", "values": [1]}},
        {"rowid": second, "event": {"type": "recv", "from": "A", "to": "B", "bindings": {"n": 1}}},
    ]
    assert list_trace_events(conn, after_rowid=first) == [
        {"rowid": second, "event": {"type": "recv", "from": "A", "to": "B", "bindings": {"n": 1}}},
    ]

from collections import deque
from zippergen.store import DurableChannel

def test_durable_put_live_inserts_and_recv_reads(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    a = DurableChannel(conn, "A")
    b = DurableChannel(conn, "B")
    conn.execute("BEGIN"); a.put("A", "B", "main", (5,)); a.commit_txn()
    conn.execute("BEGIN")
    item = b.try_get("A", "B", "main")
    assert item is not None and item[1] == (5,)
    b.commit_txn()
    # cursor is durable: a fresh DurableChannel replays the already-consumed
    # recv once (restart semantics, see test_durable_replay_reserves_...)
    # but never serves it a second time as a new live read.
    b2 = DurableChannel(conn, "B")
    conn.execute("BEGIN")
    assert b2.try_get("A", "B", "main")[1] == (5,)
    assert b2.try_get("A", "B", "main") is None
    b2.rollback_txn()


def test_durable_put_roundtrips_causal_metadata(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    a = DurableChannel(conn, "A")
    b = DurableChannel(conn, "B")
    vc = {"A": 2, "B": 0}
    view = {"A": {123: True}, "B": {}}
    field_view = {"A": {"src": "v1"}, "B": {}}
    conn.execute("BEGIN")
    a.put("A", "B", "main", ("v1",), vc, view, field_view)
    a.commit_txn()

    conn.execute("BEGIN")
    item = b.try_get("A", "B", "main")
    b.rollback_txn()

    assert item is not None
    assert item[2] == vc
    assert item[3] == view
    assert item[4] == field_view


def test_durable_put_reads_legacy_vc_only_causal_stamp(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    conn.execute("BEGIN")
    conn.execute(
        "INSERT INTO events(sender,receiver,channel,kind,payload,causal_stamp) "
        "VALUES(?,?,?,?,?,?)",
        ("A", "B", "main", "msg", "[1]", '{"A": 1, "B": 0}'),
    )
    conn.execute("COMMIT")
    b = DurableChannel(conn, "B")

    conn.execute("BEGIN")
    item = b.try_get("A", "B", "main")
    b.rollback_txn()

    assert item is not None
    assert item[2] == {"A": 1, "B": 0}
    assert item[3] is None
    assert item[4] is None


def test_durable_rollback_does_not_advance_cursor(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    a = DurableChannel(conn, "A")
    conn.execute("BEGIN"); a.put("A", "B", "main", (9,)); a.commit_txn()
    b = DurableChannel(conn, "B")
    conn.execute("BEGIN"); assert b.try_get("A", "B", "main")[1] == (9,); b.rollback_txn()
    # not consumed -> re-delivered on a fresh receiver
    b2 = DurableChannel(conn, "B")
    conn.execute("BEGIN"); assert b2.try_get("A", "B", "main")[1] == (9,); b2.rollback_txn()

def test_durable_replay_reserves_recorded_sends_and_recvs(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    # Simulate a prior committed history: A sent (1,), and B consumed it.
    a = DurableChannel(conn, "A")
    conn.execute("BEGIN"); a.put("A", "B", "main", (1,)); a.commit_txn()
    b = DurableChannel(conn, "B")
    conn.execute("BEGIN"); b.try_get("A", "B", "main"); b.commit_txn()
    # Restart B: it must replay its already-consumed recv, not re-read live.
    b_restart = DurableChannel(conn, "B")
    assert b_restart.replaying() is True
    item = b_restart.try_get("A", "B", "main")   # served from replay queue, no txn
    assert item[1] == (1,)
    assert b_restart.replaying() is False
    # Restart A: it must NOT re-INSERT its recorded send.
    a_restart = DurableChannel(conn, "A")
    assert a_restart.replaying() is True
    a_restart.put("A", "B", "main", (1,))        # reserved from replay, no INSERT
    assert a_restart.replaying() is False
    count = conn.execute("SELECT COUNT(*) FROM events WHERE sender='A'").fetchone()[0]
    assert count == 1


# ---------------------------------------------------------------------------
# Snapshots (Task 1)
# ---------------------------------------------------------------------------
from zippergen.store import write_snapshot, load_snapshot


def test_snapshot_roundtrip(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    assert load_snapshot(conn, "A") is None
    write_snapshot(conn, "A", {"n": 3}, [1], {"out": 5, "cursors": {"B|A|main": 4}})
    snap = load_snapshot(conn, "A")
    assert snap == {"env": {"n": 3}, "locator": [1], "floor": {"out": 5, "cursors": {"B|A|main": 4}}}


def test_snapshot_is_latest_only(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    write_snapshot(conn, "A", {"n": 1}, [], {"out": 0, "cursors": {}})
    write_snapshot(conn, "A", {"n": 2}, [], {"out": 9, "cursors": {}})
    assert load_snapshot(conn, "A")["env"] == {"n": 2}
    assert conn.execute("SELECT COUNT(*) FROM snapshots WHERE role='A'").fetchone()[0] == 1


def test_write_snapshot_nonserializable_raises_no_open_txn(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    import pytest
    with pytest.raises(TypeError):
        write_snapshot(conn, "A", {"bad": object()}, [], {"out": 0, "cursors": {}})
    # connection is still usable (no dangling transaction)
    write_snapshot(conn, "A", {"n": 1}, [], {"out": 0, "cursors": {}})
    assert load_snapshot(conn, "A")["env"] == {"n": 1}


# ---------------------------------------------------------------------------
# position() + since (tail-only replay) (Task 3)
# ---------------------------------------------------------------------------
def test_position_reports_out_and_cursors(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    a = DurableChannel(conn, "A")
    conn.execute("BEGIN"); r = a.put("A", "B", "main", (1,)); a.commit_txn()
    b = DurableChannel(conn, "B")
    conn.execute("BEGIN"); b.try_get("A", "B", "main"); b.commit_txn()
    assert a.position()["out"] == r
    assert b.position()["cursors"]["A|B|main"] == r


def test_since_none_replays_all_history(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    a = DurableChannel(conn, "A")
    conn.execute("BEGIN"); a.put("A", "B", "main", (1,)); a.commit_txn()
    a2 = DurableChannel(conn, "A", since=None)
    assert a2.replaying() is True  # full history reserved


def test_since_floor_replays_only_tail(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    a = DurableChannel(conn, "A")
    conn.execute("BEGIN"); r1 = a.put("A", "B", "main", (1,)); a.commit_txn()
    conn.execute("BEGIN"); r2 = a.put("A", "B", "main", (2,)); a.commit_txn()
    # Resume with a floor at r1: only the r2 send is in the tail.
    a2 = DurableChannel(conn, "A", since={"out": r1, "cursors": {}})
    a2.put("A", "B", "main", (2,))   # reserves r2 (no new insert)
    assert a2.replaying() is False
    assert conn.execute("SELECT COUNT(*) FROM events WHERE sender='A'").fetchone()[0] == 2


def test_since_floor_filters_inbound_tail(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    a = DurableChannel(conn, "A")
    conn.execute("BEGIN"); a.put("A", "B", "main", (1,)); a.commit_txn()
    conn.execute("BEGIN"); r2 = a.put("A", "B", "main", (2,)); a.commit_txn()
    b = DurableChannel(conn, "B")
    conn.execute("BEGIN"); b.try_get("A", "B", "main"); b.try_get("A", "B", "main"); b.commit_txn()
    # B resumes with a floor at the first consumed rowid: only the 2nd is replayed.
    b2 = DurableChannel(conn, "B", since={"out": 0, "cursors": {"A|B|main": r2 - 1}})
    item = b2.try_get("A", "B", "main")     # replays r2
    assert item[0] == r2
    assert b2.replaying() is False


# ---------------------------------------------------------------------------
# ReplayMismatch (Task 1)
# ---------------------------------------------------------------------------
def test_reserved_send_payload_mismatch_raises(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    a = DurableChannel(conn, "A")
    conn.execute("BEGIN"); a.put("A", "B", "main", (1,)); a.commit_txn()
    # Restart A: the recorded send (1,) is reserved on replay.
    a2 = DurableChannel(conn, "A")
    assert a2.replaying() is True
    with pytest.raises(ReplayMismatch):
        a2.put("A", "B", "main", (2,))          # diverged payload


def test_reserved_send_match_ok(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    a = DurableChannel(conn, "A")
    conn.execute("BEGIN"); a.put("A", "B", "main", (1,)); a.commit_txn()
    a2 = DurableChannel(conn, "A")
    assert a2.put("A", "B", "main", (1,))        # matches -> reserved, no raise
    assert a2.replaying() is False


# ---------------------------------------------------------------------------
# Journal (Task 3)
# ---------------------------------------------------------------------------
def test_journal_record_and_consume_fifo(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    a = DurableChannel(conn, "A")
    conn.execute("BEGIN")
    a.record_act({"status": "done", "locator": [0], "action": "llm", "input_hash": "h", "outputs": {"y": 1}})
    a.record_decision({"status": "done", "locator": [1], "value": True})
    a.commit_txn()
    # A fresh channel replays the journal in FIFO order via the cursor.
    a2 = DurableChannel(conn, "A")
    p0 = a2.consume_journal("act", [0], "h")
    assert p0["outputs"] == {"y": 1}
    p1 = a2.consume_journal("decision", [1])
    assert p1["value"] is True
    assert a2.consume_journal("act", [2]) is None          # nothing left -> live path
    assert a2.position()["journal"] == a2._journal_consumed

def test_journal_locator_mismatch_raises(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    a = DurableChannel(conn, "A")
    conn.execute("BEGIN"); a.record_act({"status": "done", "locator": [0], "action": "llm", "outputs": {}}); a.commit_txn()
    a2 = DurableChannel(conn, "A")
    with pytest.raises(ReplayMismatch):
        a2.consume_journal("act", [9])                     # wrong locator

def test_journal_non_strict_matches_locator_out_of_order(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    a = DurableChannel(conn, "A")
    conn.execute("BEGIN")
    a.record_act({"status": "done", "locator": [1], "action": "llm", "input_hash": "h1", "outputs": {"z": 1}})
    a.record_act({"status": "done", "locator": [0], "action": "llm", "input_hash": "h0", "outputs": {"y": 2}})
    a.commit_txn()

    a2 = DurableChannel(conn, "A")
    p0 = a2.consume_journal("act", [0], "h0", strict=False)
    assert p0["outputs"] == {"y": 2}
    p1 = a2.consume_journal("act", [1], "h1", strict=False)
    assert p1["outputs"] == {"z": 1}


def test_journal_non_strict_miss_does_not_consume_rows(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    a = DurableChannel(conn, "A")
    conn.execute("BEGIN")
    a.record_act({"status": "done", "locator": [1], "action": "llm", "outputs": {"z": 1}})
    a.record_act({"status": "done", "locator": [0], "action": "llm", "outputs": {"y": 2}})
    a.commit_txn()

    a2 = DurableChannel(conn, "A")
    assert a2.consume_journal("act", [9], strict=False) is None
    assert a2.consume_journal("act", [0], strict=False)["outputs"] == {"y": 2}

def test_record_act_does_not_advance_cursor(tmp_path):
    # act rows are consumed by a separate pass; recording must leave the cursor
    # below the new row so the next consume_journal picks it up.
    conn = open_store(str(tmp_path / "s.sqlite"))
    a = DurableChannel(conn, "A")
    conn.execute("BEGIN"); rid = a.record_act({"status": "done", "locator": [0], "action": "llm", "outputs": {"y": 2}}); a.commit_txn()
    assert a._journal_consumed < rid
    got = a.consume_journal("act", [0])
    assert got["outputs"] == {"y": 2} and a._journal_consumed == rid
