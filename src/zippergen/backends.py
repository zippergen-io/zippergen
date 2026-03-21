"""Small ready-to-use LLM backend helpers for examples and quick starts."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from urllib import request
from urllib.error import HTTPError, URLError

__all__ = [
    "make_mistral_backend",
    "make_openai_backend",
    "make_anthropic_backend",
    "make_lifeline_router",
    "router_from_env",
]


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "1"}:
            return True
        if text in {"false", "no", "0"}:
            return False
    raise ValueError(f"Cannot coerce {value!r} to bool.")


def _coerce_output(value: object, type_: type) -> object:
    if type_ is bool:
        return _coerce_bool(value)
    if type_ is str:
        return str(value)
    if type_ is int:
        if not isinstance(value, (str, int, float)):
            raise ValueError(f"Cannot coerce {value!r} to int.")
        return int(value)
    if type_ is float:
        if not isinstance(value, (str, int, float)):
            raise ValueError(f"Cannot coerce {value!r} to float.")
        return float(value)
    return value


def _retry_json_request(req: request.Request, *, timeout: float, max_retries: int) -> dict:
    for attempt in range(max_retries + 1):
        try:
            with request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code in {429, 500, 502, 503, 504} and attempt < max_retries:
                time.sleep(1.0 + attempt)
                continue
            raise RuntimeError(f"API error: {detail}") from exc
        except URLError as exc:
            if attempt < max_retries:
                time.sleep(1.0 + attempt)
                continue
            raise RuntimeError(f"Could not reach API: {exc.reason}") from exc
    raise RuntimeError("Unreachable.")


def _json_instruction(action) -> str:
    return (
        "Return only valid JSON with exactly these keys: "
        + ", ".join(name for name, _ in action.outputs)
    )


def _short_content(content: str, limit: int = 220) -> str:
    content = content.strip()
    return content if len(content) <= limit else content[: limit - 1] + "…"


def _validate_output_keys(action, raw_outputs: object) -> dict[str, object]:
    if not isinstance(raw_outputs, dict):
        raise ValueError(
            f"expected a JSON object, got {type(raw_outputs).__name__}"
        )

    expected = [name for name, _ in action.outputs]
    expected_set = set(expected)
    actual_set = set(raw_outputs.keys())
    missing = [name for name in expected if name not in raw_outputs]
    extra = sorted(actual_set - expected_set)

    if missing or extra:
        parts: list[str] = []
        if missing:
            parts.append("missing keys: " + ", ".join(missing))
        if extra:
            parts.append("unexpected keys: " + ", ".join(extra))
        raise ValueError("; ".join(parts))

    return raw_outputs


def _coerce_outputs(action, raw_outputs: dict[str, object]) -> dict[str, object]:
    validated = _validate_output_keys(action, raw_outputs)
    coerced: dict[str, object] = {}
    for name, type_ in action.outputs:
        try:
            coerced[name] = _coerce_output(validated[name], type_)
        except ValueError as exc:
            raise ValueError(f"field '{name}': {exc}") from exc
    return coerced


def _build_messages(action, inputs: dict[str, object]) -> tuple[list[dict], bool]:
    """
    Build the messages list and return whether to request JSON response format.

    ``parse_format`` controls how the LLM is instructed to respond:
    - ``"json"``  — ask for a JSON object with exactly the declared output keys (default).
    - ``"text"``  — ask for a plain-text response; single str output only.
    - ``"bool"``  — ask for a plain true/false response; single bool output only.
    """
    user_prompt = action.user_prompt.format(**inputs)
    parse = getattr(action, "parse_format", "json") or "json"

    if parse in {"text", "bool"} and len(action.outputs) == 1:
        if parse == "bool":
            instruction = "Reply with exactly one word: true or false."
        else:
            instruction = ""  # plain text — no extra instruction needed
        content = f"{user_prompt}\n\n{instruction}".rstrip()
        messages = [
            {"role": "system", "content": action.system_prompt},
            {"role": "user", "content": content},
        ]
        return messages, False  # no JSON response format

    # Default: JSON mode
    messages = [
        {"role": "system", "content": action.system_prompt},
        {"role": "user", "content": f"{user_prompt}\n\n{_json_instruction(action)}"},
    ]
    return messages, True


def _parse_response(action, content: str) -> dict[str, object]:
    """Parse a raw LLM text response according to parse_format."""
    parse = getattr(action, "parse_format", "json") or "json"
    if parse in {"text", "bool"} and len(action.outputs) == 1:
        name, type_ = action.outputs[0]
        try:
            return {name: _coerce_output(content.strip(), type_)}
        except ValueError as exc:
            raise RuntimeError(
                f"LLM action '{action.name}' returned invalid {parse} output: {exc}. "
                f"Raw response: {_short_content(content)!r}"
            ) from exc

    try:
        raw_outputs = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"LLM action '{action.name}' returned invalid JSON: {exc.msg}. "
            f"Raw response: {_short_content(content)!r}"
        ) from exc

    try:
        return _coerce_outputs(action, raw_outputs)
    except ValueError as exc:
        expected = ", ".join(name for name, _ in action.outputs)
        raise RuntimeError(
            f"LLM action '{action.name}' returned invalid JSON output: {exc}. "
            f"Expected keys: {expected}. Raw response: {_short_content(content)!r}"
        ) from exc


def make_mistral_backend(
    *,
    api_key: str,
    model: str = "mistral-small-latest",
    temperature: float = 0.2,
    timeout: float = 60.0,
    max_retries: int = 2,
) -> Callable:
    """Return a Mistral backend callable compatible with ``Workflow.configure``."""

    def backend(action, inputs: dict[str, object]) -> dict[str, object]:
        messages, use_json = _build_messages(action, inputs)
        payload: dict = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
        }
        if use_json:
            payload["response_format"] = {"type": "json_object"}
        req = request.Request(
            "https://api.mistral.ai/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        body = _retry_json_request(req, timeout=timeout, max_retries=max_retries)
        content = body["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected Mistral response content: {content!r}")
        return _parse_response(action, content)

    backend._zippergen_lock = threading.Lock()  # type: ignore[attr-defined]
    return backend


def make_openai_backend(
    *,
    api_key: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    timeout: float = 60.0,
    max_retries: int = 2,
) -> Callable:
    """Return an OpenAI backend callable compatible with ``Workflow.configure``."""

    def backend(action, inputs: dict[str, object]) -> dict[str, object]:
        messages, use_json = _build_messages(action, inputs)
        payload: dict = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
        }
        if use_json:
            payload["response_format"] = {"type": "json_object"}
        req = request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        body = _retry_json_request(req, timeout=timeout, max_retries=max_retries)
        content = body["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected OpenAI response content: {content!r}")
        return _parse_response(action, content)

    backend._zippergen_lock = threading.Lock()  # type: ignore[attr-defined]
    return backend


def make_anthropic_backend(
    *,
    api_key: str,
    model: str = "claude-sonnet-4-5",
    max_tokens: int = 1024,
    timeout: float = 60.0,
    max_retries: int = 2,
) -> Callable:
    """Return an Anthropic Claude backend callable compatible with ``Workflow.configure``."""

    def backend(action, inputs: dict[str, object]) -> dict[str, object]:
        user_prompt = action.user_prompt.format(**inputs)
        payload: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "system": action.system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if getattr(action, "parse_format", "json") == "json":
            payload["messages"][0]["content"] = f"{user_prompt}\n\n{_json_instruction(action)}"

        req = request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        body = _retry_json_request(req, timeout=timeout, max_retries=max_retries)
        content_blocks = body.get("content")
        if not isinstance(content_blocks, list):
            raise RuntimeError(f"Unexpected Anthropic response content: {content_blocks!r}")
        text_parts: list[str] = [
            block["text"]  # type: ignore[index]  — guarded by isinstance check
            for block in content_blocks
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
        ]
        if not text_parts:
            raise RuntimeError(f"Unexpected Anthropic response content: {content_blocks!r}")
        return _parse_response(action, "\n".join(text_parts))

    backend._zippergen_lock = threading.Lock()  # type: ignore[attr-defined]
    return backend


def make_lifeline_router(backends: dict[str, Callable]) -> Callable:
    """Route LLM calls by lifeline name."""

    def backend(action, inputs: dict[str, object], lifeline_name: str) -> dict[str, object]:
        if lifeline_name not in backends:
            raise RuntimeError(f"No backend configured for lifeline {lifeline_name!r}.")
        return backends[lifeline_name](action, inputs)

    def lock_for(lifeline_name: str):
        target = backends.get(lifeline_name)
        if target is None:
            return None
        return getattr(target, "_zippergen_lock", None)

    backend._zippergen_accepts_lifeline = True  # type: ignore[attr-defined]
    backend._zippergen_lock_for = lock_for      # type: ignore[attr-defined]
    return backend


def _backend_from_env(provider: str) -> tuple[Callable, str]:
    provider = provider.lower()
    if provider == "mistral":
        api_key = os.environ.get("MISTRAL_API_KEY")
        model = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY is not set.")
        return make_mistral_backend(api_key=api_key, model=model), f"Mistral ({model})"
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        return make_openai_backend(api_key=api_key, model=model), f"OpenAI ({model})"
    if provider in {"anthropic", "claude"}:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        return make_anthropic_backend(api_key=api_key, model=model), f"Claude ({model})"
    raise RuntimeError(f"Unsupported provider {provider!r}.")


def router_from_env(
    routes: dict[str, str],
    *,
    fallback: Callable | None = None,
    fallback_label: str = "mock LLM",
) -> tuple[Callable, str]:
    """Build a per-lifeline backend router from env-configured providers."""

    if not routes:
        if fallback is None:
            raise RuntimeError("No routes configured.")
        return fallback, fallback_label

    shared_backends: dict[str, tuple[Callable, str]] = {}
    built_backends: dict[str, Callable] = {}
    labels: list[str] = []
    for lifeline_name, provider in routes.items():
        provider_key = provider.lower()
        if provider_key not in shared_backends:
            shared_backends[provider_key] = _backend_from_env(provider_key)
        built_backends[lifeline_name], label = shared_backends[provider_key]
        labels.append(f"{lifeline_name}={label}")
    return make_lifeline_router(built_backends), ", ".join(labels)
