import io
import sqlite3
import threading
import urllib.error
import urllib.request

import pytest


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """The shared bot inbox lives in ZIPPERGEN_HOME, so give every test its own."""

    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "zg-home"))

from zippergen.store import (
    ensure_human_task,
    ensure_human_task_token,
    load_adapter_state,
    load_human_task,
    load_human_task_notification,
    load_human_task_token,
    open_store,
)
from zippergen.telegram_inbox import (
    NOT_MINE,
    RETRY,
    SETTLED,
    consume_once,
    list_updates,
    open_inbox,
    record_updates,
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
        self.fetches = []

    def send_message(self, chat_id, text, reply_markup=None):
        self.sent.append({
            "chat_id": chat_id,
            "text": text,
            "reply_markup": reply_markup,
        })
        return {"result": {"message_id": len(self.sent)}}

    def get_updates(self, *, offset=None, timeout=0, allowed_updates=None):
        self.fetches.append(offset)
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
        fingerprint="fingerprint-test",
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


def test_one_bot_is_read_once_however_many_connections_name_it(tmp_path):
    """The identity that decides coordination is the token, not the name.

    Two provider connections holding the same token are the same bot, so they
    share one cursor. Reading Telegram's queue confirms and destroys, so a
    second independent reader would consume the first one's updates.
    """

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

    same_bot = "fingerprint-shared"
    first_client = PollClient([{"update_id": 100}])
    second_client = PollClient([{"update_id": 101}])
    first = TelegramDeploymentNotifier(
        store_path, first_client, connection="ops-bot",
        routes={}, assignments={}, fingerprint=same_bot,
    )
    second = TelegramDeploymentNotifier(
        store_path, second_client, connection="review-bot",
        routes={}, assignments={}, fingerprint=same_bot,
    )

    first.poll_updates_once()
    second.poll_updates_once()

    assert first_client.offsets == [None], "the first reader starts from scratch"
    assert second_client.offsets == [101], (
        "the second reader continues the shared cursor rather than restarting"
    )

    other_bot = PollClient([])
    third = TelegramDeploymentNotifier(
        store_path, other_bot, connection="other-bot",
        routes={}, assignments={}, fingerprint="fingerprint-other",
    )
    third.poll_updates_once()
    assert other_bot.offsets == [None], "a different bot keeps its own cursor"


def test_a_second_process_does_not_fetch_while_one_is_fetching(tmp_path):
    """Fetching is done on everyone's behalf, so only one process may do it."""

    from zippergen.telegram_inbox import poll_lock

    store_path = str(tmp_path / "deployment.sqlite")
    client = FakeTelegramClient([{"update_id": 100}])
    notifier = TelegramDeploymentNotifier(
        store_path, client, connection="bot",
        routes={}, assignments={}, fingerprint="fingerprint-busy",
    )

    with poll_lock("fingerprint-busy") as acquired:
        assert acquired
        notifier.poll_updates_once()

    assert client.fetches == [], "it must not fetch while another holds the lock"


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
    }) == SETTLED

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
        fingerprint="fingerprint-test",
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
    finally:
        conn.close()

    from zippergen.telegram_inbox import open_inbox, read_offset

    inbox = open_inbox("fingerprint-test")
    try:
        assert read_offset(inbox) == 41
    finally:
        inbox.close()
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
        fingerprint="fingerprint-test",
    )

    assert notifier.poll_updates_once() == 0
    assert notifier.poll_updates_once() == 0

    # The cursor belongs to the bot now, not to this deployment's store.
    from zippergen.telegram_inbox import open_inbox, read_offset

    inbox = open_inbox("fingerprint-test")
    try:
        assert read_offset(inbox) == 57
    finally:
        inbox.close()
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
    }) == SETTLED

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
        fingerprint="fingerprint-test",
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
        fingerprint="fingerprint-test",
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
        fingerprint="fingerprint-test",
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
        fingerprint="fingerprint-test",
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
        fingerprint="fingerprint-test",
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
        fingerprint="fingerprint-test",
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


# ---------------------------------------------------------------------------
# Several deployments, one bot
# ---------------------------------------------------------------------------


def test_two_deployments_sharing_a_bot_each_get_their_own_approval(tmp_path):
    """The failure this whole design exists to prevent.

    Telegram's queue is single-consumer and reading it destroys. Two
    deployments polling independently would confirm each other's updates out of
    existence, and a human's answer would vanish. Here the bot is read once and
    each deployment takes what its own token identifies.
    """

    class QueueClient(FakeTelegramClient):
        """Behaves like Telegram: an offset confirms, and confirmed is gone."""

        def __init__(self, queue):
            super().__init__([])
            self.queue = queue

        def get_updates(self, *, offset=None, timeout=0, allowed_updates=None):
            self.fetches.append(offset)
            if offset is not None:
                self.queue[:] = [
                    item for item in self.queue if item["update_id"] >= offset
                ]
            return list(self.queue)

    bot = "fingerprint-shared-bot"
    stores = {}
    notifiers = {}
    tokens = {}
    for name in ("alpha", "beta"):
        store = tmp_path / f"{name}.sqlite"
        stores[name] = store
        _create_task(store, task_id=f"task-{name}")
        conn = open_store(str(store))
        try:
            tokens[name] = ensure_human_task_token(
                conn, f"task-{name}", channel="telegram:approval-chat"
            )["token"]
        finally:
            conn.close()

    # One update per deployment, interleaved, in one shared bot queue.
    updates = [
        {
            "update_id": 500,
            "callback_query": {
                "id": "c1",
                "data": f"zg:yes:{tokens['alpha']}",
                "message": {"chat": {"id": 4242}, "message_id": 1},
            },
        },
        {
            "update_id": 501,
            "callback_query": {
                "id": "c2",
                "data": f"zg:no:{tokens['beta']}",
                "message": {"chat": {"id": 4242}, "message_id": 2},
            },
        },
    ]

    # One queue, as Telegram has one queue per bot.
    shared_queue = list(updates)
    for name in ("alpha", "beta"):
        notifiers[name] = TelegramDeploymentNotifier(
            str(stores[name]),
            QueueClient(shared_queue),
            connection="approval-bot",
            routes={
                "approval-chat": {
                    "chat_id": "4242",
                    "channel": "telegram:approval-chat",
                }
            },
            assignments={"User": "approval-chat"},
            fingerprint=bot,
        )

    # The interleaving is the point. alpha polls twice before beta polls at
    # all, so its second fetch confirms both updates to Telegram and they are
    # gone from the queue. Under a per-deployment cursor beta would find
    # nothing and lose an answer a person had already given.
    assert notifiers["alpha"].poll_updates_once() == 1
    notifiers["alpha"].poll_updates_once()
    assert shared_queue == [], "Telegram has forgotten both updates by now"

    assert notifiers["beta"].poll_updates_once() == 1

    for name, expected in (("alpha", True), ("beta", False)):
        conn = open_store(str(stores[name]))
        try:
            task = load_human_task(conn, f"task-{name}")
            assert task["status"] == "done", f"{name} lost its approval"
            assert task["result"]["approved"] is expected
        finally:
            conn.close()

    from zippergen.telegram_inbox import list_updates, open_inbox

    inbox = open_inbox(bot)
    try:
        assert list_updates(inbox) == [], "both updates were absorbed"
    finally:
        inbox.close()


def test_an_update_for_another_deployment_is_left_alone(tmp_path):
    """A poller must never discard what it cannot resolve.

    It is holding the queue on everyone's behalf, so an unrecognised update
    belongs to a deployment that has not read yet, not to nobody.
    """

    from zippergen.telegram_inbox import list_updates, open_inbox

    bot = "fingerprint-foreign"
    store = tmp_path / "mine.sqlite"
    _create_task(store, task_id="task-mine")
    client = FakeTelegramClient([
        {
            "update_id": 700,
            "callback_query": {
                "id": "c9",
                "data": "zg:yes:zg_belongs_to_someone_else",
                "message": {"chat": {"id": 4242}, "message_id": 1},
            },
        }
    ])
    notifier = TelegramDeploymentNotifier(
        str(store),
        client,
        connection="approval-bot",
        routes={
            "approval-chat": {
                "chat_id": "4242",
                "channel": "telegram:approval-chat",
            }
        },
        assignments={"User": "approval-chat"},
        fingerprint=bot,
    )

    assert notifier.poll_updates_once() == 0

    inbox = open_inbox(bot)
    try:
        assert [item[0] for item in list_updates(inbox)] == [700]
    finally:
        inbox.close()


def test_unclaimed_updates_age_out_but_recent_ones_wait(tmp_path, monkeypatch):
    """An update waits for a stopped deployment; only age says it is orphaned."""

    import time as _time

    from zippergen.telegram_inbox import (
        count_stale_updates,
        list_updates,
        open_inbox,
        prune_updates,
        record_updates,
    )

    conn = open_inbox("fingerprint-aging")
    try:
        record_updates(conn, [{"update_id": 1}, {"update_id": 2}], offset=2)
        conn.execute(
            "UPDATE inbox SET received_at=? WHERE update_id=1",
            (_time.time() - 60 * 86400,),
        )

        assert count_stale_updates(conn, older_than_days=30) == 1
        conn.execute("BEGIN IMMEDIATE")
        removed = prune_updates(conn, older_than_days=30)
        conn.execute("COMMIT")

        assert removed == 1
        assert [item[0] for item in list_updates(conn)] == [2], (
            "a recent update must keep waiting for its deployment"
        )
    finally:
        conn.close()


def test_the_lock_file_is_never_removed(tmp_path):
    """flock lives on the inode, not the path.

    Unlinking the lock file while a process holds it would let the next process
    lock a fresh inode and believe it had exclusive access, which is exactly
    the mutual exclusion this design depends on.
    """

    from zippergen.telegram_inbox import lock_path, poll_lock

    with poll_lock("fingerprint-keeps-lock") as acquired:
        assert acquired
    path = lock_path("fingerprint-keeps-lock")
    assert path.exists(), "the lock file outlives the lock, on purpose"

    # And it is reusable rather than recreated.
    inode = path.stat().st_ino
    with poll_lock("fingerprint-keeps-lock") as acquired:
        assert acquired
    assert path.stat().st_ino == inode


def test_every_bot_read_goes_through_the_shared_cursor():
    """The invariant, checked structurally rather than remembered.

    Telegram's queue is single-consumer. One exceptional code path calling
    getUpdates on its own would silently reintroduce the bug this whole design
    exists to prevent, so there must be exactly one caller.
    """

    import ast
    import pathlib

    source_root = pathlib.Path(__file__).resolve().parents[1] / "src" / "zippergen"
    callers: list[str] = []
    for path in sorted(source_root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get_updates"
            ):
                callers.append(f"{path.name}:{node.lineno}")

    assert len(callers) == 1, (
        "every getUpdates must go through telegram_inbox.fetch_once, "
        f"but found {callers}"
    )
    assert callers[0].startswith("telegram_inbox.py"), callers


def test_an_answer_that_arrived_twice_does_not_stick_in_the_shared_inbox(
    tmp_path, monkeypatch
):
    """Completing the task and dropping the update are two databases apart.

    A crash between them leaves an answer whose task is already complete. On
    the next poll the task refuses to complete again -- correctly -- and the
    old code read that refusal as "belongs to another deployment" and kept the
    update forever, re-contacting Telegram on every cycle.
    """

    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))
    store = tmp_path / "twice.sqlite"
    _create_task(store, task_id="task-1")
    conn = open_store(str(store))
    try:
        token = ensure_human_task_token(
            conn, "task-1", channel="telegram:approval-chat"
        )["token"]
    finally:
        conn.close()

    update = {
        "update_id": 900,
        "callback_query": {
            "id": "cb-900",
            "data": f"zg:yes:{token}",
            "message": {"chat": {"id": 4242}, "message_id": 7},
        },
    }
    client = FakeTelegramClient([update])
    notifier = TelegramDeploymentNotifier(
        str(store),
        client,
        connection="approval-bot",
        routes={"approval-chat": {"chat_id": "4242", "channel": "telegram:approval-chat"}},
        assignments={"Mailbox": "approval-chat"},
        fingerprint="fingerprint-twice",
    )

    # First pass completes the task. Simulate the crash by putting the update
    # back exactly as Telegram would have re-delivered it.
    assert notifier.process_update(update) == SETTLED
    inbox = open_inbox("fingerprint-twice")
    try:
        record_updates(inbox, [update], offset=900)
        assert len(list_updates(inbox)) == 1
    finally:
        inbox.close()

    # Second pass: the task is already complete, so nothing more can come of
    # this answer. It must be dropped, not retried forever.
    assert notifier.process_update(update) == SETTLED
    assert consume_once("fingerprint-twice", notifier.process_update) == 1

    inbox = open_inbox("fingerprint-twice")
    try:
        assert list_updates(inbox) == [], "a spent answer must not be retained"
    finally:
        inbox.close()


def test_an_answer_for_another_deployment_is_left_alone(tmp_path, monkeypatch):
    """The other half of the rule: unknown tokens are somebody else's."""

    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))
    store = tmp_path / "mine.sqlite"
    _create_task(store, task_id="task-mine")
    update = {
        "update_id": 901,
        "callback_query": {
            "id": "cb-901",
            "data": "zg:yes:token-belonging-to-another-store",
            "message": {"chat": {"id": 4242}, "message_id": 8},
        },
    }
    notifier = TelegramDeploymentNotifier(
        str(store),
        FakeTelegramClient([]),
        connection="approval-bot",
        routes={"approval-chat": {"chat_id": "4242", "channel": "telegram:approval-chat"}},
        assignments={"Mailbox": "approval-chat"},
        fingerprint="fingerprint-other",
    )

    assert notifier.process_update(update) == NOT_MINE

    inbox = open_inbox("fingerprint-other")
    try:
        record_updates(inbox, [update], offset=901)
        assert consume_once("fingerprint-other", notifier.process_update) == 0
        assert len(list_updates(inbox)) == 1, "another deployment still needs it"
    finally:
        inbox.close()


def test_an_unusable_answer_is_explained_once_and_the_task_stays_pending(
    tmp_path, monkeypatch
):
    """"maybe" to a yes/no question is the person's mistake, not a fault.

    The message is spent -- repeating the complaint on every poll would be
    worse than useless -- but the task must stay pending so a better answer
    can still arrive.
    """

    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))
    store = tmp_path / "unusable.sqlite"
    _create_task(store, task_id="task-1")
    conn = open_store(str(store))
    try:
        token = ensure_human_task_token(
            conn, "task-1", channel="telegram:approval-chat"
        )["token"]
    finally:
        conn.close()

    update = {
        "update_id": 910,
        "message": {"chat": {"id": 4242}, "text": f"/zg {token} maybe"},
    }
    client = FakeTelegramClient([])
    notifier = TelegramDeploymentNotifier(
        str(store),
        client,
        connection="approval-bot",
        routes={"approval-chat": {"chat_id": "4242", "channel": "telegram:approval-chat"}},
        assignments={"Mailbox": "approval-chat"},
        fingerprint="fingerprint-unusable",
    )

    assert notifier.process_update(update) == SETTLED

    inbox = open_inbox("fingerprint-unusable")
    try:
        record_updates(inbox, [update], offset=910)
        assert consume_once("fingerprint-unusable", notifier.process_update) == 1
        assert list_updates(inbox) == [], "a rejected message must not be kept"
    finally:
        inbox.close()

    conn = open_store(str(store))
    try:
        task = load_human_task(conn, "task-1")
        assert task is not None and task["status"] == "pending", (
            "the person must still be able to answer properly"
        )
    finally:
        conn.close()


def test_a_storage_fault_keeps_the_answer_for_the_next_poll(tmp_path, monkeypatch):
    """Transient means retry. Only storage and network count as transient."""

    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))
    store = tmp_path / "faulty.sqlite"
    _create_task(store, task_id="task-1")
    conn = open_store(str(store))
    try:
        token = ensure_human_task_token(
            conn, "task-1", channel="telegram:approval-chat"
        )["token"]
    finally:
        conn.close()

    notifier = TelegramDeploymentNotifier(
        str(store),
        FakeTelegramClient([]),
        connection="approval-bot",
        routes={"approval-chat": {"chat_id": "4242", "channel": "telegram:approval-chat"}},
        assignments={"Mailbox": "approval-chat"},
        fingerprint="fingerprint-faulty",
    )
    monkeypatch.setattr(
        "zippergen.telegram_notify.complete_task_with_token",
        lambda *a, **k: (_ for _ in ()).throw(sqlite3.OperationalError("locked")),
    )

    outcome = notifier.process_update({
        "update_id": 911,
        "callback_query": {
            "id": "cb-911",
            "data": f"zg:yes:{token}",
            "message": {"chat": {"id": 4242}, "message_id": 3},
        },
    })

    assert outcome == RETRY


def test_a_defect_in_processing_is_not_swallowed_as_a_retry(tmp_path, monkeypatch):
    """A bug here must crash, not spin quietly on every polling cycle."""

    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))
    store = tmp_path / "buggy.sqlite"
    _create_task(store, task_id="task-1")
    conn = open_store(str(store))
    try:
        token = ensure_human_task_token(
            conn, "task-1", channel="telegram:approval-chat"
        )["token"]
    finally:
        conn.close()

    notifier = TelegramDeploymentNotifier(
        str(store),
        FakeTelegramClient([]),
        connection="approval-bot",
        routes={"approval-chat": {"chat_id": "4242", "channel": "telegram:approval-chat"}},
        assignments={"Mailbox": "approval-chat"},
        fingerprint="fingerprint-buggy",
    )
    monkeypatch.setattr(
        "zippergen.telegram_notify.complete_task_with_token",
        lambda *a, **k: (_ for _ in ()).throw(AttributeError("typo")),
    )

    with pytest.raises(AttributeError):
        notifier.process_update({
            "update_id": 912,
            "callback_query": {
                "id": "cb-912",
                "data": f"zg:yes:{token}",
                "message": {"chat": {"id": 4242}, "message_id": 4},
            },
        })


def test_a_corrupt_task_specification_is_not_reported_as_a_bad_answer(
    tmp_path, monkeypatch
):
    """A malformed stored spec is a defect here, not a person's mistake.

    Both used to raise ValueError, so treating every ValueError as an invalid
    answer consumed the update and left the task pending -- hiding a store or
    ZipperGen fault behind a message about the reply.
    """

    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))
    store = tmp_path / "corrupt.sqlite"
    _create_task(store, task_id="task-1")
    conn = open_store(str(store))
    try:
        token = ensure_human_task_token(
            conn, "task-1", channel="telegram:approval-chat"
        )["token"]
    finally:
        conn.close()

    notifier = TelegramDeploymentNotifier(
        str(store),
        FakeTelegramClient([]),
        connection="approval-bot",
        routes={"approval-chat": {"chat_id": "4242", "channel": "telegram:approval-chat"}},
        assignments={"Mailbox": "approval-chat"},
        fingerprint="fingerprint-corrupt",
    )
    monkeypatch.setattr(
        "zippergen.telegram_notify.human_task_result_from_value",
        lambda *a, **k: (_ for _ in ()).throw(
            ValueError("Human task specification kind 'nonsense' is unsupported.")
        ),
    )

    with pytest.raises(ValueError, match="specification"):
        notifier.process_update({
            "update_id": 913,
            "callback_query": {
                "id": "cb-913",
                "data": f"zg:yes:{token}",
                "message": {"chat": {"id": 4242}, "message_id": 5},
            },
        })
