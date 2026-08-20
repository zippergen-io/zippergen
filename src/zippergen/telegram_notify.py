"""Telegram notification adapter for durable human tasks."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

from zippergen.human_tasks import (
    human_task_options,
    human_task_result_from_value,
    validate_human_task_spec,
)
from zippergen.telegram_inbox import (
    NOT_MINE,
    RETRY,
    SETTLED,
    bot_fingerprint,
    consume_once,
    fetch_once,
)
from zippergen.store import (
    complete_human_task,
    ensure_human_task_token,
    load_human_task,
    load_human_task_notification,
    load_human_task_notification_by_external,
    load_human_task_token,
    record_connector_health,
    mark_human_task_token_used,
    open_store,
    record_human_task_notification,
)


class TelegramAPIError(RuntimeError):
    """Raised when Telegram returns an unsuccessful Bot API response."""


class TelegramBotClient:
    def __init__(self, token: str, *, timeout: float = 20.0) -> None:
        if not token:
            raise ValueError(
                "Telegram bot token is required. Set ZIPPERGEN_TELEGRAM_TOKEN "
                "or pass --bot-token."
            )
        self.token = token
        self.timeout = timeout

    def request(
        self,
        method: str,
        *,
        request_timeout: float | None = None,
        **params,
    ) -> dict:
        body = json.dumps(params).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{self.token}/{method}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                req,
                timeout=self.timeout if request_timeout is None else request_timeout,
            ) as resp:
                payload = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise TelegramAPIError(
                f"Telegram {method} failed: HTTP {exc.code} {detail}"
            ) from exc
        except (OSError, ValueError) as exc:
            # A refused connection, a DNS failure, a socket timeout and a
            # malformed body are all one thing to every caller: this call did
            # not work. Naming them here, where the HTTP lives, keeps every
            # caller free of a list that would drift as the failures do.
            raise TelegramAPIError(f"Telegram {method} failed: {exc}") from exc
        if not payload.get("ok", False):
            raise TelegramAPIError(f"Telegram {method} failed: {payload}")
        return payload

    def send_message(self, chat_id: str, text: str, reply_markup: dict | None = None) -> dict:
        params: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        if reply_markup:
            params["reply_markup"] = reply_markup
        return self.request("sendMessage", **params)

    def get_updates(
        self,
        *,
        offset: int | None = None,
        timeout: float = 0,
        allowed_updates: list[str] | None = None,
    ) -> list[dict]:
        params: dict[str, Any] = {"timeout": int(timeout)}
        if offset is not None:
            params["offset"] = offset
        if allowed_updates is not None:
            params["allowed_updates"] = allowed_updates
        # Telegram may legitimately hold the request for the full long-poll
        # duration. The HTTP socket therefore needs a margin beyond the API
        # timeout; using the same value makes ordinary empty polls race the
        # client deadline and appear as failures.
        http_timeout = max(self.timeout, float(timeout) + 10.0)
        return list(
            self.request(
                "getUpdates",
                request_timeout=http_timeout,
                **params,
            ).get("result", [])
        )

    def answer_callback_query(self, callback_query_id: str, text: str | None = None) -> None:
        params: dict[str, Any] = {"callback_query_id": callback_query_id}
        if text:
            params["text"] = text
        self.request("answerCallbackQuery", **params)

    def edit_message_reply_markup(
        self,
        *,
        chat_id: str,
        message_id: int,
        reply_markup: dict | None = None,
    ) -> None:
        params: dict[str, Any] = {"chat_id": chat_id, "message_id": message_id}
        if reply_markup is not None:
            params["reply_markup"] = reply_markup
        self.request("editMessageReplyMarkup", **params)


def load_telegram_token(explicit: str | None = None) -> str:
    return explicit or os.environ.get("ZIPPERGEN_TELEGRAM_TOKEN", "")


def load_telegram_chat_id(explicit: str | None = None) -> str:
    return explicit or os.environ.get("ZIPPERGEN_TELEGRAM_CHAT_ID", "")


class TaskNotForThisRoute(ValueError):
    """This answer is not addressed to this deployment.

    Several deployments may share one bot, so an unknown token is the normal
    case rather than a fault: it belongs to somebody else's store, and this
    process must leave it alone.
    """


class TaskAlreadySettled(ValueError):
    """This answer is addressed here, but nothing more can come of it.

    The task is already complete, or its store was reset. Either way the
    answer is spent, and keeping it would mean retrying forever.
    """


def complete_task_with_token(
    conn,
    token: str,
    value: object = None,
    *,
    channel: str | None = None,
) -> dict:
    token_record = load_human_task_token(conn, token)
    if token_record is None:
        raise TaskNotForThisRoute(f"Human task token not found: {token}")
    if channel is not None and token_record["channel"] != channel:
        raise TaskNotForThisRoute(
            "This response belongs to another connector route."
        )
    task = load_human_task(conn, token_record["task_id"])
    if task is None:
        raise TaskAlreadySettled(
            f"Human task not found: {token_record['task_id']}"
        )
    if task["status"] != "pending":
        raise TaskAlreadySettled(
            f"Human task {task['task_id']} is already {task['status']}."
        )
    result = human_task_result_from_value(task["spec"], value)
    task = complete_human_task(conn, task["task_id"], result)
    mark_human_task_token_used(conn, token)
    return task


def parse_callback_data(data: str) -> tuple[str, object] | None:
    parts = data.split(":")
    if len(parts) < 3 or parts[0] != "zg":
        return None
    if parts[1] == "yes":
        return ":".join(parts[2:]), True
    if parts[1] == "no":
        return ":".join(parts[2:]), False
    if parts[1] == "option" and len(parts) >= 4:
        return ":".join(parts[3:]), parts[2]
    return None


def parse_text_response(text: str) -> tuple[str, str | None] | None:
    parts = text.strip().split(maxsplit=2)
    if len(parts) < 2:
        return None
    command = parts[0].split("@", 1)[0].lower()
    if command not in {"/zg", "zg"}:
        return None
    value = parts[2] if len(parts) == 3 else None
    return parts[1], value


def _short_text(value: object, *, limit: int = 1200) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[: limit - 1] + "..."


def format_task_message(task: dict, token: str) -> str:
    spec = validate_human_task_spec(task.get("spec") or {})
    rendered = spec.get("rendered") or {}
    lines = [
        "ZipperGen human task",
        f"Task: {task['task_id']}",
        f"Action: {task['role']}.{task['action']} ({spec.get('kind', 'human')})",
    ]
    instruction = rendered.get("instruction")
    context = rendered.get("context")
    prefill = rendered.get("prefill")
    if instruction:
        lines.extend(["", "Instruction:", _short_text(instruction)])
    if context:
        lines.extend(["", "Context:", _short_text(context)])
    if prefill:
        lines.extend(["", "Prefill:", _short_text(prefill)])
    lines.extend(["", f"Token: {token}"])
    if spec.get("output_type") == "bool":
        lines.append("Use the buttons below, or reply with:")
        lines.append(f"/zg {token} yes")
        if spec.get("kind") != "ack":
            lines.append(f"/zg {token} no")
    else:
        options = human_task_options(spec)
        if options:
            lines.append("Choose an option:")
            lines.extend(
                f"{index}. {_short_text(option, limit=300)}"
                for index, option in enumerate(options, 1)
            )
            lines.append("Use a button below, or reply with:")
            lines.append(f"/zg {token} <number>")
        else:
            lines.append("Reply with:")
            lines.append(f"/zg {token} <your text>")
    return "\n".join(lines)[:4096]


def build_reply_markup(task: dict, token: str) -> dict | None:
    spec = validate_human_task_spec(task.get("spec") or {})
    if spec.get("output_type") != "bool":
        options = human_task_options(spec)
        if not options:
            return None
        return {
            "inline_keyboard": [
                [
                    {
                        "text": f"{index}. {option}"[:64],
                        "callback_data": f"zg:option:{index}:{token}",
                    }
                ]
                for index, option in enumerate(options, 1)
            ]
        }
    yes_label = spec.get("submit_label") or ("Acknowledge" if spec.get("kind") == "ack" else "Confirm")
    row = [{"text": yes_label, "callback_data": f"zg:yes:{token}"}]
    if spec.get("kind") != "ack":
        row.append({"text": spec.get("cancel_label") or "Decline", "callback_data": f"zg:no:{token}"})
    return {"inline_keyboard": [row]}


@dataclass
class TelegramNotifier:
    store_path: str
    client: TelegramBotClient
    chat_id: str
    channel: str = "telegram"
    limit: int | None = None
    #: Which bot this reads. Every reader of a bot shares one cursor, so this
    #: command is not an exception to that rule just because it is run by hand.
    fingerprint: str = ""

    @property
    def _target(self) -> str:
        return str(self.chat_id)

    @property
    def _bot(self) -> str:
        return self.fingerprint or bot_fingerprint(self.client.token)

    def send_pending_once(self, *, resend: bool = False) -> int:
        conn = open_store(self.store_path)
        try:
            query = (
                "SELECT task_id FROM human_tasks WHERE status='pending' "
                "ORDER BY updated_at DESC, task_id"
            )
            params: tuple[object, ...] = ()
            if self.limit is not None:
                query += " LIMIT ?"
                params = (self.limit,)
            rows = conn.execute(query, params).fetchall()
            sent = 0
            for row in rows:
                task = load_human_task(conn, row[0])
                if task is None:
                    continue
                token = ensure_human_task_token(conn, task["task_id"], channel=self.channel)["token"]
                if (
                    not resend
                    and load_human_task_notification(
                        conn,
                        task["task_id"],
                        channel=self.channel,
                        target=self._target,
                    )
                    is not None
                ):
                    continue
                result = self.client.send_message(
                    self._target,
                    format_task_message(task, token),
                    reply_markup=build_reply_markup(task, token),
                )
                message_id = result.get("result", {}).get("message_id")
                record_human_task_notification(
                    conn,
                    task["task_id"],
                    channel=self.channel,
                    target=self._target,
                    external_id=None if message_id is None else str(message_id),
                )
                sent += 1
            return sent
        finally:
            conn.close()

    def process_update(self, update: dict) -> str:
        callback = update.get("callback_query")
        if callback:
            return self._process_callback(callback)
        message = update.get("message") or update.get("edited_message")
        if message:
            return self._process_message(message)
        return NOT_MINE

    def poll_updates_once(self, *, timeout: float = 0) -> int:
        fetch_once(self.client, self._bot, timeout=timeout)
        return consume_once(self._bot, self.process_update)

    def _chat_matches(self, chat_id: object) -> bool:
        return str(chat_id) == self._target

    def _answer_callback_best_effort(
        self, callback: Mapping[str, object], text: str
    ) -> None:
        """Acknowledge Telegram UI state without affecting task delivery.

        Callback queries expire quickly.  Recording the durable human answer
        must therefore never depend on Telegram still accepting this
        short-lived acknowledgement.
        """
        callback_id = callback.get("id")
        if callback_id is None:
            return
        try:
            self.client.answer_callback_query(str(callback_id), text)
        except Exception:
            pass

    def _clear_callback_buttons_best_effort(
        self, message: Mapping[str, object]
    ) -> None:
        """Remove stale buttons when possible; they are not durable state."""
        message_id = message.get("message_id")
        if message_id is None:
            return
        try:
            self.client.edit_message_reply_markup(
                chat_id=self._target,
                message_id=int(str(message_id)),
            )
        except Exception:
            pass

    def _process_callback(self, callback: dict) -> str:
        parsed = parse_callback_data(str(callback.get("data") or ""))
        if parsed is None:
            return NOT_MINE
        message = callback.get("message") or {}
        chat = message.get("chat") or {}
        if not self._chat_matches(chat.get("id")):
            self._answer_callback_best_effort(
                callback, "This task belongs to another chat."
            )
            return NOT_MINE

        token, value = parsed
        try:
            conn = open_store(self.store_path)
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    task = complete_task_with_token(
                        conn, token, value, channel=self.channel
                    )
                    conn.execute("COMMIT")
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
            finally:
                conn.close()
        except TaskNotForThisRoute:
            # Another deployment sharing this bot may own it. Say nothing and
            # leave it for them.
            return NOT_MINE
        except TaskAlreadySettled as exc:
            # Ours, and spent. Answering once is courteous; keeping it would
            # mean answering on every poll until it ages out.
            self._answer_callback_best_effort(callback, str(exc))
            return SETTLED
        except Exception:
            # A database or network fault is temporary. Keep the update so the
            # answer is not lost, and try again next time.
            return RETRY

        self._answer_callback_best_effort(callback, "Recorded.")
        self._clear_callback_buttons_best_effort(message)
        return SETTLED

    def _process_message(self, message: dict) -> str:
        chat = message.get("chat") or {}
        if not self._chat_matches(chat.get("id")):
            return NOT_MINE
        text = str(message.get("text") or "")
        parsed = parse_text_response(text)
        if parsed is None:
            replied = message.get("reply_to_message") or {}
            external_id = replied.get("message_id")
            if external_id is not None and text.strip():
                conn = open_store(self.store_path)
                try:
                    notification = (
                        load_human_task_notification_by_external(
                            conn,
                            channel=self.channel,
                            target=self._target,
                            external_id=str(external_id),
                        )
                    )
                    if notification is not None:
                        token = ensure_human_task_token(
                            conn,
                            notification["task_id"],
                            channel=self.channel,
                        )["token"]
                        parsed = (token, text.strip())
                finally:
                    conn.close()
        if parsed is None:
            return NOT_MINE
        token, value = parsed
        try:
            conn = open_store(self.store_path)
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    task = complete_task_with_token(
                        conn, token, value, channel=self.channel
                    )
                    conn.execute("COMMIT")
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
            finally:
                conn.close()
            self.client.send_message(self._target, f"Recorded response for task {task['task_id']}.")
            return SETTLED
        except TaskNotForThisRoute:
            return NOT_MINE
        except TaskAlreadySettled as exc:
            self.client.send_message(
                self._target, f"Could not record response: {exc}"
            )
            return SETTLED
        except Exception as exc:
            self.client.send_message(self._target, f"Could not record response: {exc}")
            return RETRY


@dataclass
class TelegramDeploymentNotifier:
    """Route one deployment's human tasks through reusable Telegram chats.

    One instance polls a bot exactly once, even when several named
    configurations use that bot.  Participant routes are overridden by exact
    ``Participant.action`` routes.
    """

    store_path: str
    client: TelegramBotClient
    connection: str
    routes: Mapping[str, Mapping[str, object]]
    assignments: Mapping[str, str]
    limit: int | None = None
    #: Which bot this polls, as a hash of its token. Two provider connections
    #: holding the same token are the same bot and must share one reader, so
    #: this, not the connection name, is the identity that decides coordination.
    fingerprint: str = ""

    #: An outage lasts minutes, not seconds. Backing off to a minute keeps a
    #: long one from becoming a request flood, and still recovers promptly.
    MAX_RETRY_DELAY = 60.0

    @property
    def _bot(self) -> str:
        """The shared identity this poller coordinates on."""

        if not self.fingerprint:
            raise ValueError(
                "TelegramDeploymentNotifier needs the bot fingerprint: two "
                "connections holding one token must share a reader."
            )
        return self.fingerprint

    def _backoff(self, delay: float) -> float:
        return min(delay * 2, self.MAX_RETRY_DELAY)

    def _record_health(self, *, healthy: bool, detail: str = "") -> None:
        """Publish whether the bot is reachable, so status can say so.

        A revoked token and a five-minute outage look the same in the log.
        They look different here, because one of them never clears.
        """

        try:
            conn = open_store(self.store_path)
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    record_connector_health(
                        conn,
                        f"telegram:{self.connection}",
                        healthy=healthy,
                        detail=detail,
                    )
                    conn.execute("COMMIT")
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
            finally:
                conn.close()
        except sqlite3.DatabaseError:
            # Reporting health must never be the thing that stops delivery.
            return

    def _configuration_for_task(self, task: dict) -> str | None:
        action_target = f"{task['role']}.{task['action']}"
        return self.assignments.get(action_target) or self.assignments.get(
            str(task["role"])
        )

    def send_pending_once(self, *, resend: bool = False) -> int:
        conn = open_store(self.store_path)
        try:
            query = (
                "SELECT task_id FROM human_tasks WHERE status='pending' "
                "ORDER BY updated_at DESC, task_id"
            )
            params: tuple[object, ...] = ()
            if self.limit is not None:
                query += " LIMIT ?"
                params = (self.limit,)
            sent = 0
            for row in conn.execute(query, params).fetchall():
                task = load_human_task(conn, row[0])
                if task is None:
                    continue
                configuration = self._configuration_for_task(task)
                route = self.routes.get(configuration or "")
                if route is None:
                    continue
                chat_id = str(route.get("chat_id") or "")
                channel = str(
                    route.get("channel")
                    or f"telegram:{configuration}"
                )
                token = ensure_human_task_token(
                    conn, task["task_id"], channel=channel
                )["token"]
                if (
                    not resend
                    and load_human_task_notification(
                        conn,
                        task["task_id"],
                        channel=channel,
                        target=chat_id,
                    )
                    is not None
                ):
                    continue
                result = self.client.send_message(
                    chat_id,
                    format_task_message(task, token),
                    reply_markup=build_reply_markup(task, token),
                )
                message_id = result.get("result", {}).get("message_id")
                record_human_task_notification(
                    conn,
                    task["task_id"],
                    channel=channel,
                    target=chat_id,
                    external_id=(
                        None if message_id is None else str(message_id)
                    ),
                )
                sent += 1
            return sent
        finally:
            conn.close()

    def _notifier_for_update(
        self, update: dict
    ) -> TelegramNotifier | None:
        callback = update.get("callback_query") or {}
        message = (
            callback.get("message")
            or update.get("message")
            or update.get("edited_message")
            or {}
        )
        chat_id = str((message.get("chat") or {}).get("id") or "")
        candidates = [
            (name, route)
            for name, route in self.routes.items()
            if str(route.get("chat_id") or "") == chat_id
        ]
        if not candidates:
            return None

        token: str | None = None
        if callback:
            parsed = parse_callback_data(str(callback.get("data") or ""))
            if parsed is not None:
                token = parsed[0]
        else:
            parsed_text = parse_text_response(
                str(message.get("text") or "")
            )
            if parsed_text is not None:
                token = parsed_text[0]
        if token is not None:
            conn = open_store(self.store_path)
            try:
                record = load_human_task_token(conn, token)
            finally:
                conn.close()
            if record is not None:
                candidates = [
                    (name, route)
                    for name, route in candidates
                    if str(
                        route.get("channel")
                        or f"telegram:{name}"
                    ) == record["channel"]
                ]
        elif not callback:
            replied = message.get("reply_to_message") or {}
            external_id = replied.get("message_id")
            if external_id is not None:
                conn = open_store(self.store_path)
                try:
                    candidates = [
                        (name, route)
                        for name, route in candidates
                        if load_human_task_notification_by_external(
                            conn,
                            channel=str(
                                route.get("channel")
                                or f"telegram:{name}"
                            ),
                            target=chat_id,
                            external_id=str(external_id),
                        )
                        is not None
                    ]
                finally:
                    conn.close()
        if len(candidates) != 1:
            return None
        name, route = candidates[0]
        return TelegramNotifier(
            store_path=self.store_path,
            client=self.client,
            chat_id=chat_id,
            channel=str(
                route.get("channel") or f"telegram:{name}"
            ),
        )

    def process_update(self, update: dict) -> str:
        notifier = self._notifier_for_update(update)
        if notifier is None:
            return NOT_MINE
        return notifier.process_update(update)

    def poll_updates_once(self, *, timeout: float = 0) -> int:
        """Fetch on everyone's behalf if allowed, then take what is ours.

        Several deployments may share this bot. Only one of them may read
        Telegram's queue, because reading it confirms and destroys. Consuming
        the shared inbox needs no such coordination: an update is identified by
        a token this deployment issued, or by a message it sent.
        """

        fetch_once(self.client, self._bot, timeout=timeout)
        return consume_once(self._bot, self.process_update)

    def run_forever(
        self,
        *,
        interval: float = 2.0,
        poll_timeout: float = 20.0,
        stop_event: threading.Event | None = None,
    ) -> None:
        delay = interval
        while stop_event is None or not stop_event.is_set():
            # Only the two ways the outside world fails are caught. A defect in
            # our own code must still crash the poller loudly: retrying it every
            # two seconds forever would hide the bug and fix nothing.
            try:
                self.send_pending_once()
                self.poll_updates_once(timeout=poll_timeout)
            except sqlite3.DatabaseError as exc:
                # The store is the one place health could be recorded, so a
                # store failure can only be reported to the log.
                delay = self._backoff(delay)
                print(
                    f"Durable store unavailable for Telegram delivery "
                    f"({self.store_path}), retrying in {delay:g}s: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            except TelegramAPIError as exc:
                delay = self._backoff(delay)
                self._record_health(healthy=False, detail=str(exc))
                print(
                    f"Telegram connector retrying in {delay:g}s: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                delay = interval
                self._record_health(healthy=True)
            if stop_event is None:
                time.sleep(delay)
            else:
                stop_event.wait(delay)


@dataclass
class TelegramNotifierGroup:
    """Run one poller per Telegram bot behind one connector interface."""

    notifiers: tuple[TelegramDeploymentNotifier, ...]

    def run_forever(
        self,
        *,
        interval: float = 2.0,
        poll_timeout: float = 20.0,
        stop_event: threading.Event | None = None,
    ) -> None:
        threads = [
            threading.Thread(
                target=notifier.run_forever,
                kwargs={
                    "interval": interval,
                    "poll_timeout": poll_timeout,
                    "stop_event": stop_event,
                },
                name=f"connector-telegram-poller-{index}",
                daemon=True,
            )
            for index, notifier in enumerate(self.notifiers, start=1)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
