import json
from types import SimpleNamespace

import pytest

from zippergen import Lifeline, llm, workflow
from zippergen.backends import backend_from_spec, make_openai_backend


ConfigUser = Lifeline("ConfigUser")
ConfigConflictUser = Lifeline("ConfigConflictUser")


@llm(system="Echo.", user="{topic}", parse="text", outputs=(("draft", str),))
def config_reply(topic: str) -> None: ...


@workflow
def config_workflow(topic: str @ ConfigUser) -> str:
    ConfigUser: draft = config_reply(topic)
    return draft @ ConfigUser


@workflow
def config_conflict(topic: str @ ConfigConflictUser) -> str:
    return topic @ ConfigConflictUser


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps({
            "choices": [{"message": {"content": "hello"}}],
        }).encode("utf-8")


def test_openai_backend_accepts_custom_base_url(monkeypatch):
    seen = {}

    def fake_urlopen(req, *, timeout):
        seen["url"] = req.full_url
        seen["timeout"] = timeout
        seen["payload"] = json.loads(req.data.decode("utf-8"))
        seen["auth"] = req.get_header("Authorization")
        return _Response()

    monkeypatch.setattr("zippergen.backends.request.urlopen", fake_urlopen)

    backend = make_openai_backend(
        api_key="EMPTY",
        model="Qwen/Qwen2.5-7B-Instruct",
        base_url="http://127.0.0.1:8000/v1/",
        timeout=12,
    )
    action = SimpleNamespace(
        name="say",
        system_prompt="You are concise.",
        user_prompt="Say hello.",
        outputs=(("text", str),),
        parse_format="text",
    )

    assert backend(action, {}) == {"text": "hello"}
    assert seen["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert seen["timeout"] == 12
    assert seen["auth"] == "Bearer EMPTY"
    assert seen["payload"]["model"] == "Qwen/Qwen2.5-7B-Instruct"


def test_backend_from_spec_accepts_inline_openai_model(monkeypatch):
    seen = {}

    def fake_urlopen(req, *, timeout):
        seen["url"] = req.full_url
        seen["timeout"] = timeout
        seen["payload"] = json.loads(req.data.decode("utf-8"))
        seen["auth"] = req.get_header("Authorization")
        return _Response()

    monkeypatch.setattr("zippergen.backends.request.urlopen", fake_urlopen)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    backend, label = backend_from_spec("openai:gpt-4o")
    action = SimpleNamespace(
        name="say",
        system_prompt="You are concise.",
        user_prompt="Say hello.",
        outputs=(("text", str),),
        parse_format="text",
    )

    assert backend(action, {}) == {"text": "hello"}
    assert label == "OpenAI-compatible (gpt-4o)"
    assert seen["url"] == "https://api.openai.com/v1/chat/completions"
    assert seen["timeout"] == 90.0
    assert seen["auth"] == "Bearer test-key"
    assert seen["payload"]["model"] == "gpt-4o"


def test_backend_from_spec_accepts_ollama_model_with_colon(monkeypatch):
    seen = {}

    def fake_urlopen(req, *, timeout):
        seen["url"] = req.full_url
        seen["timeout"] = timeout
        seen["payload"] = json.loads(req.data.decode("utf-8"))
        seen["auth"] = req.get_header("Authorization")
        return _Response()

    monkeypatch.setattr("zippergen.backends.request.urlopen", fake_urlopen)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

    backend, label = backend_from_spec("ollama:qwen2.5:7b")
    action = SimpleNamespace(
        name="say",
        system_prompt="You are concise.",
        user_prompt="Say hello.",
        outputs=(("text", str),),
        parse_format="text",
    )

    assert backend(action, {}) == {"text": "hello"}
    assert label == "Ollama (qwen2.5:7b)"
    assert seen["url"] == "http://127.0.0.1:11434/v1/chat/completions"
    assert seen["timeout"] == 120.0
    assert seen["auth"] == "Bearer ollama"
    assert seen["payload"]["model"] == "qwen2.5:7b"
    assert seen["payload"]["max_tokens"] == 512


def test_workflow_configure_accepts_positional_llm_spec():
    config_workflow.configure("mock", ui=False, execution="memory", timeout=5)

    assert config_workflow(topic="hello") == "[config_reply:draft]"


def test_workflow_configure_rejects_llm_and_llms_together():
    with pytest.raises(ValueError, match="either 'llm'"):
        config_conflict.configure(llm="mock", llms="mock")
