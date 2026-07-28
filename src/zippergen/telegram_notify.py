"""Telegram notification adapter for durable human tasks."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

from zippergen.store import (
    complete_human_task,
    ensure_human_task_token,
    load_adapter_state,
    load_human_task,
    load_human_task_notification,
    load_human_task_notification_by_external,
    load_human_task_token,
    mark_human_task_token_used,
    open_store,
    record_human_task_notification,
    write_adapter_state,
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

    def request(self, method: str, **params) -> dict:
        body = json.dumps(params).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{self.token}/{method}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise TelegramAPIError(f"Telegram {method} failed: HTTP {exc.code} {detail}") from exc
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
        return list(self.request("getUpdates", **params).get("result", []))

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


def parse_bool_value(raw: object) -> bool:
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in {"true", "yes", "1", "y", "approve", "approved", "ack"}:
        return True
    if text in {"false", "no", "0", "n", "decline", "declined", "reject", "rejected"}:
        return False
    raise ValueError(f"Cannot parse boolean human response: {raw!r}")


def _task_options(task: dict) -> list[str]:
    spec = task.get("spec") or {}
    if spec.get("kind") != "select":
        return []
    rendered = spec.get("rendered") or {}
    return [
        line.strip()
        for line in str(rendered.get("prefill") or "").splitlines()
        if line.strip()
    ]


def result_from_human_value(task: dict, value: object = None) -> dict:
    spec = task.get("spec") or {}
    output = spec.get("output")
    if not output:
        raise ValueError(f"Task {task['task_id']} has no output field in its spec.")
    output_type = spec.get("output_type", "str")
    if output_type == "bool":
        return {output: True if value is None else parse_bool_value(value)}
    if value is None:
        raise ValueError(f"Task {task['task_id']} requires a text value for {output!r}.")
    options = _task_options(task)
    if options:
        raw = str(value).strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            value = options[int(raw) - 1]
        elif raw not in options:
            raise ValueError(
                f"Choose a number between 1 and {len(options)}."
            )
    return {output: str(value)}


def complete_task_with_token(
    conn,
    token: str,
    value: object = None,
    *,
    channel: str | None = None,
) -> dict:
    token_record = load_human_task_token(conn, token)
    if token_record is None:
        raise ValueError(f"Human task token not found: {token}")
    if channel is not None and token_record["channel"] != channel:
        raise ValueError("This response belongs to another connector route.")
    task = load_human_task(conn, token_record["task_id"])
    if task is None:
        raise ValueError(f"Human task not found: {token_record['task_id']}")
    if task["status"] != "pending":
        raise ValueError(f"Human task {task['task_id']} is already {task['status']}.")
    result = result_from_human_value(task, value)
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
    spec = task.get("spec") or {}
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
        options = _task_options(task)
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
    spec = task.get("spec") or {}
    if spec.get("output_type") != "bool":
        options = _task_options(task)
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

    @property
    def _target(self) -> str:
        return str(self.chat_id)

    @property
    def _offset_key(self) -> str:
        return f"telegram:{self.channel}:{self._target}:offset"

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

    def process_update(self, update: dict) -> bool:
        callback = update.get("callback_query")
        if callback:
            return self._process_callback(callback)
        message = update.get("message") or update.get("edited_message")
        if message:
            return self._process_message(message)
        return False

    def poll_updates_once(self, *, timeout: float = 0) -> int:
        conn = open_store(self.store_path)
        try:
            offset = int(load_adapter_state(conn, self._offset_key, 0) or 0)
        finally:
            conn.close()

        updates = self.client.get_updates(
            offset=offset + 1 if offset else None,
            timeout=timeout,
            allowed_updates=["message", "callback_query"],
        )
        processed = 0
        max_update_id = offset
        for update in updates:
            max_update_id = max(max_update_id, int(update.get("update_id", 0)))
            if self.process_update(update):
                processed += 1

        if max_update_id > offset:
            conn = open_store(self.store_path)
            try:
                write_adapter_state(conn, self._offset_key, max_update_id)
            finally:
                conn.close()
        return processed

    def run_forever(self, *, interval: float = 2.0, poll_timeout: float = 20.0, resend: bool = False) -> None:
        while True:
            self.send_pending_once(resend=resend)
            self.poll_updates_once(timeout=poll_timeout)
            time.sleep(interval)

    def _chat_matches(self, chat_id: object) -> bool:
        return str(chat_id) == self._target

    def _process_callback(self, callback: dict) -> bool:
        parsed = parse_callback_data(str(callback.get("data") or ""))
        if parsed is None:
            return False
        message = callback.get("message") or {}
        chat = message.get("chat") or {}
        if not self._chat_matches(chat.get("id")):
            self.client.answer_callback_query(callback["id"], "This task belongs to another chat.")
            return False

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
            self.client.answer_callback_query(callback["id"], "Recorded.")
            message_id = message.get("message_id")
            if message_id is not None:
                try:
                    self.client.edit_message_reply_markup(chat_id=self._target, message_id=int(message_id))
                except TelegramAPIError:
                    pass
            return True
        except Exception as exc:
            self.client.answer_callback_query(callback["id"], str(exc))
            return False

    def _process_message(self, message: dict) -> bool:
        chat = message.get("chat") or {}
        if not self._chat_matches(chat.get("id")):
            return False
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
            return False
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
            return True
        except Exception as exc:
            self.client.send_message(self._target, f"Could not record response: {exc}")
            return False


@dataclass
class TelegramDeploymentNotifier:
    """Route one deployment's human tasks through reusable Telegram chats.

    One instance polls a bot exactly once, even when several named
    configurations use that bot.  Participant routes are overridden by exact
    ``Participant.action`` routes.
    """

    store_path: str
    client: TelegramBotClient
    routes: Mapping[str, Mapping[str, object]]
    assignments: Mapping[str, str]
    limit: int | None = None

    @property
    def _offset_key(self) -> str:
        return "telegram:deployment:offset"

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

    def process_update(self, update: dict) -> bool:
        notifier = self._notifier_for_update(update)
        return notifier.process_update(update) if notifier is not None else False

    def poll_updates_once(self, *, timeout: float = 0) -> int:
        conn = open_store(self.store_path)
        try:
            offset = int(load_adapter_state(conn, self._offset_key, 0) or 0)
        finally:
            conn.close()
        updates = self.client.get_updates(
            offset=offset + 1 if offset else None,
            timeout=timeout,
            allowed_updates=["message", "callback_query"],
        )
        processed = 0
        max_update_id = offset
        for update in updates:
            max_update_id = max(
                max_update_id, int(update.get("update_id", 0))
            )
            if self.process_update(update):
                processed += 1
        if max_update_id > offset:
            conn = open_store(self.store_path)
            try:
                write_adapter_state(
                    conn, self._offset_key, max_update_id
                )
            finally:
                conn.close()
        return processed

    def run_forever(
        self,
        *,
        interval: float = 2.0,
        poll_timeout: float = 20.0,
    ) -> None:
        while True:
            try:
                self.send_pending_once()
                self.poll_updates_once(timeout=poll_timeout)
            except Exception as exc:
                print(
                    f"Telegram connector retrying after error: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            time.sleep(interval)
