import json
from types import SimpleNamespace

from zippergen.backends import make_openai_backend


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
