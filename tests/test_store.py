import sqlite3
from zippergen.store import open_store, chan_key

def test_open_store_creates_tables(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {"events", "cursors"} <= names

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
