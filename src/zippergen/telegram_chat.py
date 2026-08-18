"""Send a message to a bound Telegram chat from an ``@effect``.

A human action asks a question and waits for the answer, which makes it a step
in the protocol. Telling somebody what happened is not that: nothing waits for
it, and no other participant depends on it. It is an ordinary side effect, so
it belongs in an ``@effect`` and it gets an ordinary connector requirement:

    ConnectorRequirement(
        name="call-alerts",
        kind="telegram",
        participant="Mailbox",
        access="write",
        description="Short notice when a call arrives. No answer expected.",
    )

    @effect
    def announce(subject: str) -> int:
        TelegramChat.from_requirement("call-alerts").send(subject)
        return 1

The chat id and the bot token arrive through the deployment's connector
routing, exactly as the Gmail and Sheets helpers receive theirs, so the
workflow source names neither.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from zippergen.connectors import requirement_binding


class TelegramChatError(RuntimeError):
    """A clear Telegram connector error suitable for command output."""


@dataclass(frozen=True)
class TelegramChat:
    """One bound Telegram destination, for announcements rather than questions."""

    requirement: str
    chat_id: str
    token: str
    access: str = "write"

    @classmethod
    def from_requirement(cls, requirement: str) -> "TelegramChat":
        binding = requirement_binding(
            requirement, kind="telegram", error=TelegramChatError
        )
        token_env = str(binding.get("token_env") or "")
        token = os.environ.get(token_env, "") if token_env else ""
        if not token:
            raise TelegramChatError(
                f"The Telegram bot token is missing for {requirement!r}."
            )
        chat_id = str(binding.get("chat_id") or "")
        if not chat_id:
            raise TelegramChatError(
                f"Connector requirement {requirement!r} has no chat id. Set one "
                f"with 'zippergen connector configure NAME CONNECTION telegram "
                f"--chat-id ID'."
            )
        return cls(
            requirement=requirement,
            chat_id=chat_id,
            token=token,
            access=str(binding.get("access") or "write"),
        )

    def send(self, text: str) -> None:
        """Post one message, refusing if the requirement only declared reading.

        The workflow states its intent in the requirement, so a workflow that
        announced read-only access cannot quietly start posting.
        """

        if self.access == "read-only":
            raise TelegramChatError(
                f"Connector requirement {self.requirement!r} is declared "
                "read-only, so it cannot send messages. Declare "
                "access='write' on the ConnectorRequirement."
            )
        message = str(text).strip()
        if not message:
            raise TelegramChatError("A Telegram message cannot be empty.")
        from zippergen.telegram_notify import TelegramBotClient

        TelegramBotClient(self.token).send_message(self.chat_id, message)


__all__ = ["TelegramChat", "TelegramChatError"]
