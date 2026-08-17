import io
import sqlite3
import threading
import urllib.error
import urllib.request

import pytest

from zippergen.store import (
    ensure_human_task,
    load_adapter_state,
    load_human_task,
    load_human_task_notification,
    load_human_task_token,
    open_store,
)
from zippergen.telegram_notify import (
    TelegramAPIError,
    TelegramBotClient,
    TelegramDeploymentNotifier,
    TelegramNotifier,
    build_reply_markup,
    format_task_message,
    parse_callback_data,
    parse_text_response,
)


class FakeTelegramClient:
    def __init__(self, updates=None):
        self.sent = []
        self.answers = []
        self.edits = []
        self._updates = list(updates or [])

    def send_message(self, chat_id, text, reply_markup=None):
        self.sent.append({
            "chat_id": chat_id,
            "text": text,
            "reply_markup": reply_markup,
        })
        return {"result": {"message_id": len(self.sent)}}

    def get_updates(self, *, offset=None, timeout=0, allowed_updates=None):
        return list(self._updates)

    def answer_callback_query(self, callback_query_id, text=None):
        self.answers.append({"callback_query_id": callback_query_id, "text": text})

    def edit_message_reply_markup(self, *, chat_id, message_id, reply_markup=None):
        self.edits.append({
            "chat_id": chat_id,
            "message_id": message_id,
            "reply_markup": reply_markup,
        })


class ExpiredCallbackClient(FakeTelegramClient):
    """Telegram client whose short-lived callback acknowledgement expired."""

    def __init__(self, updates=None):
        super().__init__(updates)
        self.offsets = []

    def get_updates(self, *, offset=None, timeout=0, allowed_updates=None):
        self.offsets.append(offset)
        return [
            update
            for update in self._updates
            if offset is None or int(update["update_id"]) >= offset
        ]

    def answer_callback_query(self, callback_query_id, text=None):
        raise TelegramAPIError(
            "Telegram answerCallbackQuery failed: HTTP 400 query is too old"
        )


def test_long_poll_http_timeout_has_margin_beyond_telegram_timeout(monkeypatch):
    client = TelegramBotClient("private-token", timeout=20)
    captured = {}

    def request(method, *, request_timeout=None, **params):
        captured.update(
            method=method,
            request_timeout=request_timeout,
            params=params,
        )
        return {"result": []}

    monkeypatch.setattr(client, "request", request)

    assert client.get_updates(timeout=20) == []
    assert captured == {
        "method": "getUpdates",
        "request_timeout": 30.0,
        "params": {"timeout": 20},
    }


def test_notifier_identifies_store_errors_as_store_errors(
    tmp_path, monkeypatch, capsys
):
    stop = threading.Event()
    notifier = TelegramDeploymentNotifier(
        str(tmp_path / "deployment.sqlite"),
        FakeTelegramClient(),
        connection="telegram-main",
        routes={},
        assignments={},
    )

    def fail_from_store():
        stop.set()
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(notifier, "send_pending_once", fail_from_store)

    notifier.run_forever(interval=0, poll_timeout=0, stop_event=stop)

    error = capsys.readouterr().err
    assert "Durable store unavailable for Telegram delivery" in error
    assert "disk I/O error" in error
    assert "Telegram API retrying" not in error


def test_deployment_notifiers_keep_independent_offsets_per_connection(tmp_path):
    store_path = str(tmp_path / "deployment.sqlite")

    class PollClient(FakeTelegramClient):
        def __init__(self, updates):
            super().__init__(updates)
            self.offsets = []

        def get_updates(self, *, offset=None, timeout=0, allowed_updates=None):
            self.offsets.append(offset)
            return super().get_updates(
                offset=offset,
                timeout=timeout,
                allowed_updates=allowed_updates,
            )

    first_client = PollClient([{"update_id": 100}])
    second_client = PollClient([])
    first = TelegramDeploymentNotifier(
        store_path,
        first_client,
        connection="first-bot",
        routes={},
        assignments={},
    )
    second = TelegramDeploymentNotifier(
        store_path,
        second_client,
        connection="second-bot",
        routes={},
        assignments={},
    )

    first.poll_updates_once()
    second.poll_updates_once()

    assert first_client.offsets == [None]
    assert second_client.offsets == [None]


def _create_task(
    store_path,
    *,
    task_id="task-1",
    kind="confirm",
    output_type="bool",
    prefill="Draft text",
):
    conn = open_store(str(store_path))
    try:
        ensure_human_task(
            conn,
            task_id=task_id,
            role="User",
            locator=[0],
            action="review",
            input_hash=None,
            inputs={"prompt": "Approve?"},
            spec={
                "kind": kind,
                "output": "approved" if output_type == "bool" else "reply",
                "output_type": output_type,
                "rendered": {
                    "instruction": "Approve the request?",
                    "context": "Request context",
                    "prefill": prefill,
                },
                "submit_label": "Approve",
                "cancel_label": "Decline",
            },
        )
    finally:
        conn.close()


def test_telegram_notifier_sends_pending_task_once(tmp_path):
    store_path = tmp_path / "notify.sqlite"
    _create_task(store_path)
    client = FakeTelegramClient()
    notifier = TelegramNotifier(str(store_path), client, chat_id="123")

    assert notifier.send_pending_once() == 1
    assert notifier.send_pending_once() == 0

    assert len(client.sent) == 1
    sent = client.sent[0]
    assert sent["chat_id"] == "123"
    assert "ZipperGen human task" in sent["text"]
    assert "Approve the request?" in sent["text"]
    assert sent["reply_markup"]["inline_keyboard"][0][0]["text"] == "Approve"
    conn = open_store(str(store_path))
    try:
        task = load_human_task(conn, "task-1")
        token = load_human_task_token(
            conn,
            sent["reply_markup"]["inline_keyboard"][0][0]["callback_data"].split(":", 2)[2],
        )
        notification = load_human_task_notification(
            conn,
            "task-1",
            channel="telegram",
            target="123",
        )
        assert task["status"] == "pending"
        assert token["task_id"] == "task-1"
        assert notification["external_id"] == "1"
    finally:
        conn.close()


def test_telegram_callback_completes_boolean_task(tmp_path):
    store_path = tmp_path / "callback.sqlite"
    _create_task(store_path)
    client = FakeTelegramClient()
    notifier = TelegramNotifier(str(store_path), client, chat_id="123")
    notifier.send_pending_once()
    token = client.sent[0]["reply_markup"]["inline_keyboard"][0][0]["callback_data"].split(":", 2)[2]

    assert notifier.process_update({
        "update_id": 1,
        "callback_query": {
            "id": "cb-1",
            "data": f"zg:no:{token}",
            "message": {"message_id": 99, "chat": {"id": 123}},
        },
    }) is True

    conn = open_store(str(store_path))
    try:
        assert load_human_task(conn, "task-1")["result"] == {"approved": False}
        assert load_human_task_token(conn, token)["used_at"] is not None
    finally:
        conn.close()
    assert client.answers == [{"callback_query_id": "cb-1", "text": "Recorded."}]
    assert client.edits == [{"chat_id": "123", "message_id": 99, "reply_markup": None}]


def test_expired_callback_ack_does_not_undo_answer_or_block_offset(tmp_path):
    store_path = tmp_path / "expired-callback.sqlite"
    _create_task(store_path)
    client = ExpiredCallbackClient()
    notifier = TelegramDeploymentNotifier(
        str(store_path),
        client,
        connection="approval-bot",
        routes={
            "approval-chat": {
                "chat_id": "123",
                "channel": "telegram:approval-chat",
            }
        },
        assignments={"User": "approval-chat"},
    )
    assert notifier.send_pending_once() == 1
    token = client.sent[0]["reply_markup"]["inline_keyboard"][0][0][
        "callback_data"
    ].split(":", 2)[2]
    client._updates = [
        {
            "update_id": 41,
            "callback_query": {
                "id": "expired-callback",
                "data": f"zg:no:{token}",
                "message": {"message_id": 99, "chat": {"id": 123}},
            },
        }
    ]

    assert notifier.poll_updates_once() == 1
    assert notifier.poll_updates_once() == 0

    conn = open_store(str(store_path))
    try:
        assert load_human_task(conn, "task-1")["result"] == {
            "approved": False
        }
        assert load_adapter_state(
            conn, "telegram:deployment:approval-bot:offset"
        ) == 41
    finally:
        conn.close()
    assert client.offsets == [None, 42]


def test_invalid_expired_callback_is_consumed_once(tmp_path):
    store_path = tmp_path / "invalid-callback.sqlite"
    client = ExpiredCallbackClient(
        [
            {
                "update_id": 57,
                "callback_query": {
                    "id": "invalid-callback",
                    "data": "zg:yes:no-such-token",
                    "message": {"message_id": 100, "chat": {"id": 123}},
                },
            }
        ]
    )
    notifier = TelegramDeploymentNotifier(
        str(store_path),
        client,
        connection="approval-bot",
        routes={
            "approval-chat": {
                "chat_id": "123",
                "channel": "telegram:approval-chat",
            }
        },
        assignments={"User": "approval-chat"},
    )

    assert notifier.poll_updates_once() == 0
    assert notifier.poll_updates_once() == 0

    conn = open_store(str(store_path))
    try:
        assert load_adapter_state(
            conn, "telegram:deployment:approval-bot:offset"
        ) == 57
    finally:
        conn.close()
    assert client.offsets == [None, 58]


def test_telegram_text_command_completes_string_task(tmp_path):
    store_path = tmp_path / "text.sqlite"
    _create_task(store_path, kind="edit", output_type="str")
    client = FakeTelegramClient()
    notifier = TelegramNotifier(str(store_path), client, chat_id="123")
    notifier.send_pending_once()
    conn = open_store(str(store_path))
    try:
        token = conn.execute("SELECT token FROM human_task_tokens").fetchone()[0]
    finally:
        conn.close()

    assert notifier.process_update({
        "update_id": 1,
        "message": {"chat": {"id": "123"}, "text": f"/zg {token} Edited reply"},
    }) is True

    conn = open_store(str(store_path))
    try:
        assert load_human_task(conn, "task-1")["result"] == {"reply": "Edited reply"}
    finally:
        conn.close()
    assert client.sent[-1]["text"] == "Recorded response for task task-1."


def test_telegram_parsers_and_formatting():
    task = {
        "task_id": "task-1",
        "role": "User",
        "action": "review",
        "spec": {
            "kind": "ack",
            "output": "ack",
            "output_type": "bool",
            "rendered": {"instruction": "Done"},
        },
    }

    assert parse_callback_data("zg:yes:abc") == ("abc", True)
    assert parse_callback_data("zg:no:abc") == ("abc", False)
    assert parse_text_response("/zg@bot abc hello") == ("abc", "hello")
    assert "Done" in format_task_message(task, "abc")
    assert build_reply_markup(task, "abc") == {
        "inline_keyboard": [[{"text": "Acknowledge", "callback_data": "zg:yes:abc"}]]
    }


def test_telegram_select_uses_task_specific_buttons(tmp_path):
    store_path = tmp_path / "select.sqlite"
    _create_task(
        store_path,
        kind="select",
        output_type="str",
        prefill="Thursday, 11 AM\nFriday, 10 AM",
    )
    client = FakeTelegramClient()
    notifier = TelegramNotifier(str(store_path), client, chat_id="123")

    assert notifier.send_pending_once() == 1
    buttons = client.sent[0]["reply_markup"]["inline_keyboard"]
    assert [row[0]["text"] for row in buttons] == [
        "1. Thursday, 11 AM",
        "2. Friday, 10 AM",
    ]
    token = buttons[1][0]["callback_data"].split(":", 3)[3]
    assert notifier.process_update({
        "update_id": 1,
        "callback_query": {
            "id": "select-1",
            "data": f"zg:option:2:{token}",
            "message": {"message_id": 1, "chat": {"id": 123}},
        },
    })
    conn = open_store(str(store_path))
    try:
        assert load_human_task(conn, "task-1")["result"] == {
            "reply": "Friday, 10 AM"
        }
    finally:
        conn.close()


def test_telegram_numbered_direct_reply_is_correlated_by_message(tmp_path):
    store_path = tmp_path / "reply.sqlite"
    _create_task(
        store_path,
        kind="select",
        output_type="str",
        prefill="Thursday, 11 AM\nFriday, 10 AM",
    )
    client = FakeTelegramClient()
    notifier = TelegramNotifier(str(store_path), client, chat_id="123")
    notifier.send_pending_once()

    assert notifier.process_update({
        "update_id": 1,
        "message": {
            "chat": {"id": 123},
            "text": "2",
            "reply_to_message": {"message_id": 1},
        },
    })
    conn = open_store(str(store_path))
    try:
        assert load_human_task(conn, "task-1")["result"] == {
            "reply": "Friday, 10 AM"
        }
    finally:
        conn.close()


def test_deployment_notifier_shares_one_configuration_across_participants(
    tmp_path,
):
    store_path = tmp_path / "shared.sqlite"
    _create_task(store_path, task_id="writer-task")
    conn = open_store(str(store_path))
    try:
        ensure_human_task(
            conn,
            task_id="reviewer-task",
            role="Reviewer",
            locator=[1],
            action="review",
            input_hash=None,
            inputs={},
            spec={
                "kind": "confirm",
                "output": "approved",
                "output_type": "bool",
                "rendered": {"instruction": "Approve?", "context": None},
            },
        )
        conn.execute(
            "UPDATE human_tasks SET role='Writer' WHERE task_id='writer-task'"
        )
    finally:
        conn.close()
    client = FakeTelegramClient()
    notifier = TelegramDeploymentNotifier(
        str(store_path),
        client,
        connection="telegram-main",
        routes={
            "team-chat": {
                "chat_id": "123",
                "channel": "telegram:team-chat",
            }
        },
        assignments={
            "Writer": "team-chat",
            "Reviewer": "team-chat",
        },
    )

    assert notifier.send_pending_once() == 2
    assert {item["chat_id"] for item in client.sent} == {"123"}


def test_deployment_notifier_action_route_overrides_participant_route(
    tmp_path,
):
    store_path = tmp_path / "override.sqlite"
    _create_task(store_path)
    conn = open_store(str(store_path))
    try:
        conn.execute(
            "UPDATE human_tasks SET role='Human', action='approve_contract'"
        )
    finally:
        conn.close()
    client = FakeTelegramClient()
    notifier = TelegramDeploymentNotifier(
        str(store_path),
        client,
        connection="telegram-main",
        routes={
            "general": {
                "chat_id": "111",
                "channel": "telegram:general",
            },
            "legal": {
                "chat_id": "222",
                "channel": "telegram:legal",
            },
        },
        assignments={
            "Human": "general",
            "Human.approve_contract": "legal",
        },
    )

    assert notifier.send_pending_once() == 1
    assert client.sent[0]["chat_id"] == "222"


# ---------------------------------------------------------------------------
# The outside world fails; the connector does not
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.URLError("name resolution failed"),
        urllib.error.HTTPError("u", 502, "Bad Gateway", {}, io.BytesIO(b"upstream")),
        TimeoutError("timed out"),
        ConnectionResetError("reset by peer"),
        ValueError("Expecting value: line 1 column 1"),
    ],
    ids=["dns", "http-502", "timeout", "reset", "malformed-json"],
)
def test_every_transport_failure_becomes_one_named_error(failure, monkeypatch):
    """Callers must not carry a list of ways HTTP can fail.

    The client owns the HTTP, so it owns naming the failure. Anything else
    means every loop grows its own tuple, and those tuples drift.
    """

    client = TelegramBotClient("token")

    def explode(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(urllib.request, "urlopen", explode)

    with pytest.raises(TelegramAPIError):
        client.request("getUpdates")


def test_the_poller_survives_an_outage_and_backs_off(tmp_path, monkeypatch, capsys):
    """A Telegram outage is expected input, not a reason to stop delivering."""

    stop = threading.Event()
    notifier = TelegramDeploymentNotifier(
        str(tmp_path / "deployment.sqlite"),
        FakeTelegramClient(),
        connection="telegram-main",
        routes={},
        assignments={},
    )
    attempts = {"n": 0}

    def fail_then_stop():
        attempts["n"] += 1
        if attempts["n"] >= 3:
            stop.set()
        raise TelegramAPIError("Telegram getUpdates failed: name resolution failed")

    monkeypatch.setattr(notifier, "send_pending_once", fail_then_stop)
    waits: list[float] = []
    monkeypatch.setattr(stop, "wait", lambda delay: waits.append(delay))

    notifier.run_forever(interval=1.0, poll_timeout=0, stop_event=stop)

    assert attempts["n"] == 3, "the poller kept going through the outage"
    assert waits == [2.0, 4.0, 8.0], "and waited longer each time"
    assert "Telegram connector retrying in 2s" in capsys.readouterr().err


def test_backoff_is_capped_and_resets_after_a_success(tmp_path, monkeypatch):
    """A long outage must not become an unbounded wait, nor a request flood."""

    notifier = TelegramDeploymentNotifier(
        str(tmp_path / "deployment.sqlite"),
        FakeTelegramClient(),
        connection="telegram-main",
        routes={},
        assignments={},
    )

    delay = 1.0
    for _ in range(20):
        delay = notifier._backoff(delay)
    assert delay == notifier.MAX_RETRY_DELAY

    stop = threading.Event()
    outcomes = iter([TelegramAPIError("down"), TelegramAPIError("down"), None])
    waits: list[float] = []

    def maybe_fail():
        outcome = next(outcomes, None)
        if outcome is not None:
            raise outcome
        stop.set()

    monkeypatch.setattr(notifier, "send_pending_once", maybe_fail)
    monkeypatch.setattr(notifier, "poll_updates_once", lambda **_kwargs: 0)
    monkeypatch.setattr(stop, "wait", lambda delay: waits.append(delay))

    notifier.run_forever(interval=1.0, poll_timeout=0, stop_event=stop)

    assert waits == [2.0, 4.0, 1.0], "a success returns to the normal interval"


def test_a_defect_in_our_own_code_still_crashes_the_poller(tmp_path, monkeypatch):
    """Retrying a bug every two seconds would hide it and fix nothing."""

    stop = threading.Event()
    notifier = TelegramDeploymentNotifier(
        str(tmp_path / "deployment.sqlite"),
        FakeTelegramClient(),
        connection="telegram-main",
        routes={},
        assignments={},
    )

    def programming_error():
        raise TypeError("unsupported operand type(s)")

    monkeypatch.setattr(notifier, "send_pending_once", programming_error)

    with pytest.raises(TypeError):
        notifier.run_forever(interval=0, poll_timeout=0, stop_event=stop)


def test_health_is_published_so_status_can_say_so(tmp_path, monkeypatch):
    """A revoked token and a five-minute outage look the same in the log."""

    from zippergen.store import list_connector_health, open_store

    store = tmp_path / "deployment.sqlite"
    open_store(str(store)).close()
    stop = threading.Event()
    notifier = TelegramDeploymentNotifier(
        str(store),
        FakeTelegramClient(),
        connection="approval-bot",
        routes={},
        assignments={},
    )
    monkeypatch.setattr(stop, "wait", lambda _delay: None)

    def fail_once():
        stop.set()
        raise TelegramAPIError("Telegram getUpdates failed: HTTP 401 Unauthorized")

    monkeypatch.setattr(notifier, "send_pending_once", fail_once)
    notifier.run_forever(interval=0, poll_timeout=0, stop_event=stop)

    conn = open_store(str(store))
    try:
        health = list_connector_health(conn)
    finally:
        conn.close()
    assert len(health) == 1
    assert health[0]["connector"] == "telegram:approval-bot"
    assert health[0]["healthy"] is False
    assert "401" in health[0]["detail"]


def test_health_is_written_only_when_it_changes(tmp_path):
    """A poller runs every two seconds; it must not write every two seconds."""

    from zippergen.store import open_store, record_connector_health

    store = tmp_path / "deployment.sqlite"
    conn = open_store(str(store))
    try:
        def record(healthy, detail=""):
            conn.execute("BEGIN IMMEDIATE")
            record_connector_health(
                conn, "telegram:bot", healthy=healthy, detail=detail
            )
            conn.execute("COMMIT")

        def written_at():
            return conn.execute(
                "SELECT updated_at FROM adapter_state "
                "WHERE key='connector-health:telegram:bot'"
            ).fetchone()[0]

        record(False, "down")
        first = written_at()
        record(False, "still down")
        record(False, "still down")
        assert written_at() == first, "an unchanged state must not rewrite the row"

        record(True)
        assert written_at() != first, "a change must be recorded"
    finally:
        conn.close()
