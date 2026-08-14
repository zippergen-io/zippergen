import sqlite3
import threading

from zippergen.store import (
    ensure_human_task,
    load_human_task,
    load_human_task_notification,
    load_human_task_token,
    open_store,
)
from zippergen.telegram_notify import (
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
