"""Telegram bot client for command_center.py.

Polls for new messages via getUpdates and sends replies via sendMessage.
No OAuth, no credentials file — just a bot token.

Setup (one-time, ~2 minutes):
    1. Open Telegram and message @BotFather
    2. Send /newbot, follow the prompts, copy the token
    3. Send your new bot any message to open a conversation
    4. Set the token:
         export ZIPPERGEN_TELEGRAM_TOKEN=<your-token>
       or pass it at runtime:
         ZIPPERGEN_TELEGRAM_TOKEN=<token> python examples/command_center.py --live

No packages beyond stdlib are required.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import TypedDict


TOKEN = os.environ.get("ZIPPERGEN_TELEGRAM_TOKEN", "")

# In-memory queue of fetched but unprocessed messages.
_pending: list["TelegramMeta"] = []
_last_update_id: int = 0


class TelegramMeta(TypedDict):
    update_id: int
    chat_id:   int
    sender:    str
    text:      str


def _api(method: str, **params) -> dict:
    if not TOKEN:
        raise RuntimeError(
            "ZIPPERGEN_TELEGRAM_TOKEN is not set. "
            "Get a token from @BotFather and export it."
        )
    url = f"https://api.telegram.org/bot{TOKEN}/{method}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def _poll() -> None:
    """Fetch new updates from Telegram into the in-memory queue."""
    global _last_update_id
    result = _api("getUpdates", offset=_last_update_id + 1, timeout=0)
    for upd in result.get("result", []):
        _last_update_id = max(_last_update_id, upd["update_id"])
        msg = upd.get("message") or upd.get("edited_message")
        if not msg or "text" not in msg:
            continue
        sender_obj = msg.get("from", {})
        first = sender_obj.get("first_name", "")
        last  = sender_obj.get("last_name", "")
        sender = (first + " " + last).strip() or sender_obj.get("username", "unknown")
        _pending.append(TelegramMeta(
            update_id=upd["update_id"],
            chat_id=msg["chat"]["id"],
            sender=sender,
            text=msg["text"].strip(),
        ))


def count_unread_messages() -> int:
    _poll()
    return len(_pending)


def fetch_one_message() -> TelegramMeta | None:
    _poll()
    return _pending.pop(0) if _pending else None


def send_message(chat_id: int, text: str) -> str:
    result = _api("sendMessage", chat_id=chat_id, text=text)
    msg_id = result.get("result", {}).get("message_id", "?")
    print(f"[Telegram] Sent message_id={msg_id} to chat_id={chat_id}")
    return f"sent:{msg_id}"
