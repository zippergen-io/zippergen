"""The durable store's own surface: schema, state, messages, human tasks.

Recovery-by-replay is gone, so there are no journal, snapshot, cursor or floor
tests here. What remains is the state the runtime actually keeps.
"""

import json
import sqlite3
from pathlib import Path

import pytest

from zippergen.store import (
    DurableChannel,
    RoleStateConflict,
    StoreSchemaError,
    WorkflowIdentityError,
    claim_workflow_identity,
    complete_human_task,
    ensure_human_task,
    ensure_human_task_token,
    human_task_id,
    list_history,
    list_outstanding_messages,
    list_role_states,
    list_workflow_results,
    load_adapter_state,
    load_human_task,
    load_human_task_notification,
    load_human_task_token,
    load_role_state,
    load_workflow_result,
    mark_human_task_token_used,
    open_store,
    prune_history,
    record_history,
    record_human_task_notification,
    set_role_status,
    write_adapter_state,
    write_role_state,
    write_workflow_result,
)


EXPECTED_TABLES = {
    "adapter_state",
    "history",
    "human_task_notifications",
    "human_task_tokens",
    "human_tasks",
    "outstanding_messages",
    "role_state",
    "store_meta",
    "workflow_results",
}


def test_open_store_creates_exactly_the_tables_the_model_needs(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    try:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert names == EXPECTED_TABLES
    finally:
        conn.close()


def test_open_store_is_wal_and_owner_private(tmp_path):
    path = tmp_path / "s.sqlite"
    conn = open_store(str(path))
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        conn.close()
    assert oct(path.stat().st_mode)[-3:] == "600"


def test_open_store_refuses_an_unknown_current_state_schema(tmp_path):
    path = tmp_path / "s.sqlite"
    conn = open_store(str(path))
    conn.execute(
        "UPDATE store_meta SET value='999' WHERE key='schema_version'"
    )
    conn.close()

    with pytest.raises(StoreSchemaError, match="schema 999"):
        open_store(str(path))


# ---------------------------------------------------------------------------
# Role state
# ---------------------------------------------------------------------------


def test_role_state_round_trips_and_keeps_only_the_latest(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    try:
        assert load_role_state(conn, "A") is None
        conn.execute("BEGIN IMMEDIATE")
        write_role_state(
            conn,
            "A",
            env={"x": 1},
            control={"k": "at", "p": [0]},
            monitor={"vc": {"A": 1}},
            steps=1,
            status="running",
        )
        conn.execute("COMMIT")
        conn.execute("BEGIN IMMEDIATE")
        write_role_state(
            conn,
            "A",
            env={"x": 2},
            control={"k": "done"},
            monitor=None,
            steps=2,
            status="done",
            expected_steps=1,
        )
        conn.execute("COMMIT")

        state = load_role_state(conn, "A")
        assert state["env"] == {"x": 2}
        assert state["control"] == {"k": "done"}
        assert state["monitor"] is None
        assert state["steps"] == 2
        assert len(list_role_states(conn)) == 1
    finally:
        conn.close()


def test_role_state_preserves_nested_coordination_container_types(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    value = ([1, 2], (3, 4), {"items": [5, 6]})
    try:
        conn.execute("BEGIN IMMEDIATE")
        write_role_state(
            conn,
            "A",
            env={"value": value},
            control={"k": "done"},
            monitor=None,
            steps=1,
            status="done",
        )
        conn.execute("COMMIT")

        assert load_role_state(conn, "A")["env"] == {"value": value}
    finally:
        conn.close()


def test_role_state_compare_and_swap_rejects_a_stale_step(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    try:
        conn.execute("BEGIN IMMEDIATE")
        write_role_state(
            conn,
            "A",
            env={"x": 1},
            control={"k": "at", "p": [0]},
            monitor=None,
            steps=3,
            status="running",
        )
        conn.execute("COMMIT")

        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(RoleStateConflict, match="another runner"):
            write_role_state(
                conn,
                "A",
                env={"x": 2},
                control={"k": "done"},
                monitor=None,
                steps=4,
                status="done",
                expected_steps=2,
            )
        conn.execute("ROLLBACK")

        state = load_role_state(conn, "A")
        assert state["steps"] == 3
        assert state["env"] == {"x": 1}
    finally:
        conn.close()


def test_status_updates_do_not_disturb_recovery_state(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    try:
        conn.execute("BEGIN IMMEDIATE")
        write_role_state(
            conn,
            "A",
            env={"x": 1},
            control={"k": "at", "p": [0]},
            monitor=None,
            steps=7,
            status="running",
        )
        conn.execute("COMMIT")

        set_role_status(conn, "A", "waiting_human", {"action": "approve"})

        state = load_role_state(conn, "A")
        assert state["status"] == "waiting_human"
        assert state["detail"] == {"action": "approve"}
        assert state["control"] == {"k": "at", "p": [0]}
        assert state["steps"] == 7
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Workflow identity
# ---------------------------------------------------------------------------


def test_identity_is_claimed_once_and_then_enforced(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    try:
        claim_workflow_identity(conn, "demo", "abc")
        claim_workflow_identity(conn, "demo", "abc")
        with pytest.raises(WorkflowIdentityError):
            claim_workflow_identity(conn, "demo", "def")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Outstanding messages
# ---------------------------------------------------------------------------


def test_a_send_is_outstanding_until_the_receiver_deletes_it(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    try:
        sender = DurableChannel(conn, "A")
        conn.execute("BEGIN IMMEDIATE")
        sender.put("A", "B", "main", ("hello",))
        conn.execute("COMMIT")
        assert len(list_outstanding_messages(conn)) == 1

        receiver = DurableChannel(conn, "B")
        conn.execute("BEGIN IMMEDIATE")
        item = receiver.try_get("A", "B", "main")
        assert item is not None and item[1] == ("hello",)
        receiver.delete_taken()
        conn.execute("COMMIT")
        receiver.clear_taken()

        assert list_outstanding_messages(conn) == []
    finally:
        conn.close()


def test_a_message_preserves_nested_coordination_container_types(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    value = ([1, 2], (3, 4), {"items": [5, 6]})
    try:
        sender = DurableChannel(conn, "A")
        conn.execute("BEGIN IMMEDIATE")
        sender.put("A", "B", "main", (value,))
        conn.execute("COMMIT")

        receiver = DurableChannel(conn, "B")
        conn.execute("BEGIN IMMEDIATE")
        item = receiver.try_get("A", "B", "main")
        assert item is not None and item[1] == (value,)
        conn.execute("ROLLBACK")
    finally:
        conn.close()


def test_a_rolled_back_receive_leaves_the_message_outstanding(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    try:
        sender = DurableChannel(conn, "A")
        conn.execute("BEGIN IMMEDIATE")
        sender.put("A", "B", "main", ("hello",))
        conn.execute("COMMIT")

        receiver = DurableChannel(conn, "B")
        conn.execute("BEGIN IMMEDIATE")
        assert receiver.try_get("A", "B", "main") is not None
        conn.execute("ROLLBACK")
        receiver.clear_taken()

        assert len(list_outstanding_messages(conn)) == 1
        conn.execute("BEGIN IMMEDIATE")
        assert receiver.try_get("A", "B", "main") is not None
        conn.execute("ROLLBACK")
        receiver.clear_taken()
    finally:
        conn.close()


def test_one_route_is_fifo_and_a_taken_message_is_not_offered_twice(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    try:
        sender = DurableChannel(conn, "A")
        conn.execute("BEGIN IMMEDIATE")
        for value in ("first", "second"):
            sender.put("A", "B", "main", (value,))
        conn.execute("COMMIT")

        receiver = DurableChannel(conn, "B")
        conn.execute("BEGIN IMMEDIATE")
        assert receiver.try_get("A", "B", "main")[1] == ("first",)
        assert receiver.try_get("A", "B", "main")[1] == ("second",)
        assert receiver.try_get("A", "B", "main") is None
        conn.execute("ROLLBACK")
        receiver.clear_taken()
    finally:
        conn.close()


def test_causal_metadata_round_trips_on_an_outstanding_message(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    try:
        sender = DurableChannel(conn, "A")
        conn.execute("BEGIN IMMEDIATE")
        sender.put(
            "A",
            "B",
            "main",
            ("payload",),
            vc={"A": 3, "B": 1},
            view={"A": {7: True}},
            field_view={"A": {"coordinates": (1, [2, 3])}},
        )
        conn.execute("COMMIT")

        receiver = DurableChannel(conn, "B")
        conn.execute("BEGIN IMMEDIATE")
        _id, values, vc, view, field_view = receiver.try_get("A", "B", "main")
        conn.execute("ROLLBACK")
        receiver.clear_taken()

        assert values == ("payload",)
        assert vc == {"A": 3, "B": 1}
        assert view == {"A": {7: True}}
        assert field_view == {"A": {"coordinates": (1, [2, 3])}}
        assert type(field_view["A"]["coordinates"]) is tuple
    finally:
        conn.close()


def test_a_coregion_receive_prefers_the_earliest_send(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    try:
        conn.execute("BEGIN IMMEDIATE")
        DurableChannel(conn, "C").put("C", "R", "main", ("third",))
        DurableChannel(conn, "A").put("A", "R", "main", ("first",))
        conn.execute("COMMIT")

        receiver = DurableChannel(conn, "R")
        conn.execute("BEGIN IMMEDIATE")
        sender, item = receiver.try_get_any("R", {"A", "C"}, "main")
        conn.execute("ROLLBACK")
        receiver.clear_taken()

        assert sender == "C", "send order wins, not sender name"
        assert item[1] == ("third",)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# History: optional
# ---------------------------------------------------------------------------


def test_history_records_and_prunes(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    try:
        for index in range(20):
            record_history(conn, "A", {"type": "step", "index": index})
        assert len(list_history(conn)) == 20

        conn.execute("BEGIN IMMEDIATE")
        removed = prune_history(conn, keep=5)
        conn.execute("COMMIT")

        assert removed == 15
        rows = list_history(conn)
        assert len(rows) == 5
        assert rows[-1]["event"]["index"] == 19
    finally:
        conn.close()


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
        spec={"kind": "confirm", "output": "approved", "output_type": "bool"},
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
        spec={"kind": "confirm", "output": "approved", "output_type": "bool"},
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


def test_human_task_store_rejects_an_invalid_response(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    ensure_human_task(
        conn,
        task_id="choice-task",
        role="A",
        locator=[0],
        action="choose",
        input_hash=None,
        inputs={},
        spec={
            "kind": "select",
            "output": "choice",
            "output_type": "str",
            "rendered": {"prefill": "A\nB"},
        },
    )

    with pytest.raises(ValueError, match="must be one of: A, B"):
        complete_human_task(conn, "choice-task", {"choice": "C"})

    assert load_human_task(conn, "choice-task")["status"] == "pending"


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
        spec={"kind": "confirm", "output": "approved", "output_type": "bool"},
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
        spec={"kind": "confirm", "output": "approved", "output_type": "bool"},
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
    assert load_workflow_result(conn, "wf") == (1, True)

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
