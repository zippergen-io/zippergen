import json
import threading
from types import SimpleNamespace

import pytest

from zippergen import Json, Lifeline, llm, workflow
from zippergen.models import ModelSettings
from zippergen.backends import (
    _parse_response,
    ManagedBackend,
    backend_from_spec,
    make_lifeline_router,
    make_anthropic_backend,
    make_openai_backend,
    router_from_specs,
)
from zippergen.llm_policy import LLMInvalidResponseError, LLMPermanentError


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


def test_action_temperature_overrides_the_model_configuration(monkeypatch):
    seen = {}

    def fake_urlopen(req, *, timeout):
        seen.update(json.loads(req.data.decode("utf-8")))
        return _Response()

    monkeypatch.setattr("zippergen.backends.request.urlopen", fake_urlopen)
    backend = make_openai_backend(
        api_key="test", model="gpt-4o-mini", temperature=0.7
    )
    action = SimpleNamespace(
        name="classify",
        system_prompt="Classify.",
        user_prompt="Input.",
        outputs=(("text", str),),
        parse_format="text",
        temperature=0.0,
    )

    backend(action, {})

    assert seen["temperature"] == 0.0


@pytest.mark.parametrize("model", ["claude-opus-5", "claude-opus-4-8", "claude-sonnet-5"])
def test_new_claude_models_omit_implicit_temperature(monkeypatch, model):
    seen = {}

    def fake_request(req, *, timeout):
        seen.update(json.loads(req.data.decode("utf-8")))
        return {"content": [{"type": "text", "text": "hello"}]}

    monkeypatch.setattr("zippergen.backends._json_request", fake_request)
    backend = make_anthropic_backend(api_key="test", model=model)
    action = SimpleNamespace(
        name="say",
        system_prompt="Be brief.",
        user_prompt="Hello.",
        outputs=(("text", str),),
        parse_format="text",
        temperature=None,
    )

    assert backend(action, {}) == {"text": "hello"}
    assert "temperature" not in seen


def test_new_claude_models_reject_explicit_temperature():
    backend = make_anthropic_backend(api_key="test", model="claude-opus-5")
    action = SimpleNamespace(
        name="say",
        system_prompt="Be brief.",
        user_prompt="Hello.",
        outputs=(("text", str),),
        parse_format="text",
        temperature=0.0,
    )

    with pytest.raises(LLMPermanentError, match="does not support"):
        backend(action, {})


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


@pytest.mark.parametrize("content", ["", "   ", "\n\t"])
def test_backend_rejects_empty_text_output(content):
    action = SimpleNamespace(
        name="draft",
        outputs=(("draft", str),),
        parse_format="text",
    )

    with pytest.raises(LLMInvalidResponseError, match="returned empty text output"):
        _parse_response(action, content)


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

    def fake_backend_from_spec(spec, *, fallback=None, settings=None):
        selected.append((spec, settings.idle_timeout if settings else None))
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
        settings={"Writer": ModelSettings(idle_timeout=0)},
    )

    assert selected == [
        ("local:qwen2.5:7b", 0),
        ("local:mistral", 600),
    ]


def test_router_shares_one_managed_backend_for_one_local_configuration(
    monkeypatch,
):
    built = []

    def fake_backend_from_spec(spec, *, fallback=None, settings=None):
        backend = lambda action, inputs: {"text": "done"}
        built.append((spec, settings, backend))
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
        settings={
            "Writer": ModelSettings(idle_timeout=300),
            "Reviewer": ModelSettings(idle_timeout=300),
        },
    )

    assert len(built) == 1


def test_router_rejects_conflicting_idle_policies_across_local_aliases(
    monkeypatch,
):
    monkeypatch.setattr(
        "zippergen.backends.backend_from_spec",
        lambda spec, *, fallback=None, settings=None: (
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
            settings={
                "Writer": ModelSettings(idle_timeout=300),
                "Reviewer": ModelSettings(idle_timeout=0),
            },
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


def test_named_connections_isolate_two_keys_for_the_same_provider(monkeypatch):
    authorizations: list[str | None] = []

    def fake_urlopen(req, *, timeout):
        authorizations.append(req.get_header("Authorization"))
        return _Response()

    monkeypatch.setattr("zippergen.backends.request.urlopen", fake_urlopen)
    monkeypatch.setenv("ZIPPERGEN_PROVIDER_OPENAI_DASH_A_API_KEY", "key-a")
    monkeypatch.setenv("ZIPPERGEN_PROVIDER_OPENAI_DASH_B_API_KEY", "key-b")
    action = SimpleNamespace(
        name="say",
        system_prompt="You are concise.",
        user_prompt="Say hello.",
        outputs=(("text", str),),
        parse_format="text",
    )

    first, _ = backend_from_spec("openai@openai-a:gpt-4o-mini")
    second, _ = backend_from_spec("openai@openai-b:gpt-4o-mini")
    first(action, {})
    second(action, {})

    assert authorizations == ["Bearer key-a", "Bearer key-b"]


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

    backend, label = backend_from_spec(
        "ollama:qwen2.5:7b", settings=ModelSettings(idle_timeout=0)
    )
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
        llm_settings={"ConfigUser": ModelSettings(idle_timeout=300)},
        execution="memory",
        timeout=5,
    )


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


# One settings value, not one dictionary per setting. Threading each knob
# separately is why `max_tokens` -- an ordinary inference setting -- was
# reachable only through an environment variable while `temperature` was
# configured beside the model.


def test_configured_model_settings_reach_the_provider(monkeypatch):
    import zippergen.backends as backends_module

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        backends_module,
        "make_openai_backend",
        lambda **kwargs: (seen.update(kwargs), (lambda a, i: {}))[1],
    )

    backend_from_spec(
        "openai:gpt-4o",
        settings=ModelSettings(temperature=0.2, max_tokens=4096, timeout=120),
    )

    assert seen["max_tokens"] == 4096
    assert seen["timeout"] == 120
    assert seen["temperature"] == 0.2


def test_a_configured_setting_beats_the_environment(monkeypatch):
    """The environment stays an operational override, not the way in."""

    import zippergen.backends as backends_module

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MAX_TOKENS", "999")
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        backends_module,
        "make_openai_backend",
        lambda **kwargs: (seen.update(kwargs), (lambda a, i: {}))[1],
    )

    backend_from_spec("openai:gpt-4o", settings=ModelSettings(max_tokens=4096))
    assert seen["max_tokens"] == 4096

    seen.clear()
    backend_from_spec("openai:gpt-4o")
    assert seen["max_tokens"] == 999, "the environment still applies when unset"


def test_two_routes_share_a_backend_only_when_every_setting_matches(monkeypatch):
    built: list[object] = []
    monkeypatch.setattr(
        "zippergen.backends.backend_from_spec",
        lambda spec, *, fallback=None, settings=None: (
            built.append(settings),
            (lambda a, i: {}),
        )[1] and ((lambda a, i: {}), spec),
    )

    router_from_specs(
        {"Writer": "openai:gpt-4o", "Reviewer": "openai:gpt-4o"},
        settings={
            "Writer": ModelSettings(max_tokens=4096),
            "Reviewer": ModelSettings(max_tokens=512),
        },
    )

    assert len(built) == 2, "different max_tokens must not share one backend"
