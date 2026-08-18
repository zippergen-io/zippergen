"""Announcing is a side effect, not a question, so it gets an effect helper."""

import json

import pytest

from zippergen.connectors import CONNECTORS_ENV
from zippergen.telegram_chat import TelegramChat, TelegramChatError


def _bind(monkeypatch, **overrides):
    record = {
        "kind": "telegram",
        "access": "write",
        "chat_id": "4242",
        "token_env": "ZIPPERGEN_PROVIDER_APPROVALBOT_BOT_TOKEN",
    }
    record.update(overrides)
    monkeypatch.setenv(
        CONNECTORS_ENV, json.dumps({"requirement:call-alerts": record})
    )
    monkeypatch.setenv(record["token_env"], "secret-token")


def test_a_bound_requirement_yields_a_usable_chat(monkeypatch):
    _bind(monkeypatch)

    chat = TelegramChat.from_requirement("call-alerts")

    assert chat.chat_id == "4242"
    assert chat.token == "secret-token"


def test_sending_posts_one_message(monkeypatch):
    _bind(monkeypatch)
    sent = []

    class FakeClient:
        def __init__(self, token):
            self.token = token

        def send_message(self, chat_id, text, reply_markup=None):
            sent.append((self.token, chat_id, text))
            return {}

    monkeypatch.setattr("zippergen.telegram_notify.TelegramBotClient", FakeClient)

    TelegramChat.from_requirement("call-alerts").send("  New call: ERC Starting  ")

    assert sent == [("secret-token", "4242", "New call: ERC Starting")]


def test_a_read_only_requirement_cannot_send(monkeypatch):
    """The workflow states its intent in the declaration; sending must honour it."""

    _bind(monkeypatch, access="read-only")

    with pytest.raises(TelegramChatError, match="read-only"):
        TelegramChat.from_requirement("call-alerts").send("anything")


def test_an_unbound_or_mismatched_requirement_says_which(monkeypatch):
    monkeypatch.setenv(CONNECTORS_ENV, json.dumps({}))
    with pytest.raises(TelegramChatError, match="not bound"):
        TelegramChat.from_requirement("call-alerts")

    monkeypatch.setenv(
        CONNECTORS_ENV,
        json.dumps({"requirement:call-alerts": {"kind": "gmail"}}),
    )
    with pytest.raises(TelegramChatError, match="not telegram"):
        TelegramChat.from_requirement("call-alerts")


def test_a_missing_token_or_chat_id_is_named(monkeypatch):
    _bind(monkeypatch)
    monkeypatch.delenv("ZIPPERGEN_PROVIDER_APPROVALBOT_BOT_TOKEN")
    with pytest.raises(TelegramChatError, match="token is missing"):
        TelegramChat.from_requirement("call-alerts")

    _bind(monkeypatch, chat_id="")
    with pytest.raises(TelegramChatError, match="no chat id"):
        TelegramChat.from_requirement("call-alerts")


def test_every_connector_module_shares_one_binding_lookup():
    """Three copies of the same lookup was the reason to factor it out."""

    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "zippergen"
    definitions = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_binding_records":
                definitions.append(path.name)

    assert definitions == [], (
        "connector modules must use connectors.requirement_binding, "
        f"but these still parse the routing themselves: {definitions}"
    )
