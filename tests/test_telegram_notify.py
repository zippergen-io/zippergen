from zippergen.store import (
    ensure_human_task,
    load_human_task,
    load_human_task_notification,
    load_human_task_token,
    open_store,
)
from zippergen.telegram_notify import (
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


def _create_task(store_path, *, task_id="task-1", kind="confirm", output_type="bool"):
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
                    "prefill": "Draft text",
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
