import json
import threading
from types import SimpleNamespace

import pytest

from zippergen import Json, Lifeline, llm, workflow
from zippergen.backends import (
    _parse_response,
    ManagedBackend,
    backend_from_spec,
    make_lifeline_router,
    make_openai_backend,
    router_from_specs,
)


ConfigUser = Lifeline("ConfigUser")
ConfigConflictUser = Lifeline("ConfigConflictUser")
ConfigObserver = Lifeline("ConfigObserver")


@llm(system="Echo.", user="{topic}", parse="text", outputs=(("draft", str),))
def config_reply(topic: str) -> None: ...


@workflow
def config_workflow(topic: str @ ConfigUser) -> str:
    ConfigUser: draft = config_reply(topic)
    return draft @ ConfigUser


@workflow
def config_observed_workflow(topic: str @ ConfigUser) -> str:
    ConfigUser: draft = config_reply(topic)
    ConfigUser(draft) >> ConfigObserver(draft)
    return draft @ ConfigObserver


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


def test_backend_parses_nested_json_output_without_stringifying_it():
    action = SimpleNamespace(
        name="extract",
        outputs=(("record", Json),),
        parse_format="json",
    )
    content = json.dumps(
        {
            "record": {
                "caller": "Alice",
                "slots": ["Thursday", {"hour": 11}],
                "confirmed": False,
                "note": None,
            }
        }
    )

    assert _parse_response(action, content) == json.loads(content)


def test_backend_rejects_non_finite_json_output():
    action = SimpleNamespace(
        name="extract",
        outputs=(("record", Json),),
        parse_format="json",
    )

    with pytest.raises(RuntimeError, match="not a finite number"):
        _parse_response(action, '{"record": {"score": NaN}}')


def test_managed_backend_is_lazy_and_releases_after_call():
    calls = []

    def factory():
        calls.append("factory")

        def backend(action, inputs):
            calls.append("call")
            return {"text": "done"}

        return backend

    def release():
        calls.append("release")

    backend = ManagedBackend(factory, release=release, idle_timeout=0)
    assert backend.loaded is False

    assert backend(None, {}) == {"text": "done"}

    assert backend.loaded is False
    assert calls == ["factory", "call", "release"]


def test_router_uses_route_specific_idle_release(monkeypatch):
    selected: list[tuple[str, float | None]] = []

    def fake_backend_from_spec(spec, *, fallback=None, idle_timeout=None):
        selected.append((spec, idle_timeout))
        return (lambda action, inputs: {"text": "done"}), spec

    monkeypatch.setattr(
        "zippergen.backends.backend_from_spec",
        fake_backend_from_spec,
    )

    router_from_specs(
        {
            "Writer": "local:qwen2.5:7b",
            "Reviewer": "local:mistral",
        },
        idle_timeout=600,
        idle_timeouts={"Writer": 0},
    )

    assert selected == [
        ("local:qwen2.5:7b", 0),
        ("local:mistral", 600),
    ]


def test_router_shares_one_managed_backend_for_one_local_configuration(
    monkeypatch,
):
    built = []

    def fake_backend_from_spec(spec, *, fallback=None, idle_timeout=None):
        backend = lambda action, inputs: {"text": "done"}
        built.append((spec, idle_timeout, backend))
        return backend, spec

    monkeypatch.setattr(
        "zippergen.backends.backend_from_spec",
        fake_backend_from_spec,
    )

    router_from_specs(
        {
            "Writer": "local:qwen2.5:7b",
            "Reviewer": "ollama:qwen2.5:7b",
        },
        idle_timeouts={"Writer": 300, "Reviewer": 300},
    )

    assert len(built) == 1


def test_router_rejects_conflicting_idle_policies_across_local_aliases(
    monkeypatch,
):
    monkeypatch.setattr(
        "zippergen.backends.backend_from_spec",
        lambda spec, *, fallback=None, idle_timeout=None: (
            lambda action, inputs: {"text": "done"},
            spec,
        ),
    )

    with pytest.raises(ValueError, match="conflicting idle release policies"):
        router_from_specs(
            {
                "Writer": "local:qwen2.5:7b",
                "Reviewer": "ollama:qwen2.5:7b",
            },
            idle_timeouts={"Writer": 300, "Reviewer": 0},
        )


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


def test_ollama_backend_idle_timeout_unloads_model(monkeypatch):
    requests = []

    def fake_urlopen(req, *, timeout):
        requests.append({
            "url": req.full_url,
            "timeout": timeout,
            "payload": json.loads(req.data.decode("utf-8")),
        })
        return _Response()

    monkeypatch.setattr("zippergen.backends.request.urlopen", fake_urlopen)

    backend, label = backend_from_spec("ollama:qwen2.5:7b", idle_timeout=0)
    action = SimpleNamespace(
        name="say",
        system_prompt="You are concise.",
        user_prompt="Say hello.",
        outputs=(("text", str),),
        parse_format="text",
    )

    assert backend(action, {}) == {"text": "hello"}
    assert label == "Ollama (qwen2.5:7b)"
    assert [item["url"] for item in requests] == [
        "http://127.0.0.1:11434/v1/chat/completions",
        "http://127.0.0.1:11434/api/chat",
    ]
    assert requests[1]["payload"] == {
        "model": "qwen2.5:7b",
        "messages": [],
        "keep_alive": 0,
    }


def test_workflow_configure_accepts_positional_llm_spec():
    config_workflow.configure("mock", execution="memory", timeout=5)

    assert config_workflow(topic="hello") == "[config_reply:draft]"


def test_workflow_configure_does_not_route_non_llm_participants():
    config_observed_workflow.configure(
        "local:qwen2.5:14b",
        llm_idle_timeouts={"ConfigUser": 300},
        execution="memory",
        timeout=5,
    )


def test_workflow_configure_rejects_llm_and_llms_together():
    with pytest.raises(ValueError, match="either 'llm'"):
        config_conflict.configure(llm="mock", llms="mock")


def test_workflow_configure_rejects_negative_llm_idle_timeout():
    with pytest.raises(ValueError, match="llm_idle_timeout"):
        config_conflict.configure(llm="mock", llm_idle_timeout=-1)


def test_lifeline_router_prefers_an_exact_action_override():
    selected = []

    def participant_backend(action, _inputs):
        selected.append(("participant", action.name))
        return {}

    def action_backend(action, _inputs):
        selected.append(("action", action.name))
        return {}

    router = make_lifeline_router({
        "Writer": participant_backend,
        "Writer.revise": action_backend,
    })

    def invoke():
        router(SimpleNamespace(name="draft"), {})
        router(SimpleNamespace(name="revise"), {})

    thread = threading.Thread(target=invoke, name="Writer")
    thread.start()
    thread.join()

    assert selected == [
        ("participant", "draft"),
        ("action", "revise"),
    ]
