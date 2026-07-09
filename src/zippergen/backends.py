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
    "ManagedBackend",
    "backend_from_spec",
    "make_mistral_backend",
    "make_openai_backend",
    "make_anthropic_backend",
    "make_lifeline_router",
    "router_from_specs",
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
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    delay = float(retry_after) if retry_after else (2.0 ** attempt * 2)
                except ValueError:
                    delay = 1.0 + attempt
                time.sleep(delay)
                continue
            raise RuntimeError(f"API error: {detail}") from exc
        except URLError as exc:
            if attempt < max_retries:
                time.sleep(1.0 + attempt)
                continue
            raise RuntimeError(f"Could not reach API: {exc.reason}") from exc
        except OSError as exc:
            # Catches ConnectionResetError and similar low-level socket errors
            # that occur during response reading, after urlopen() succeeds.
            if attempt < max_retries:
                time.sleep(1.0 + attempt)
                continue
            raise RuntimeError(f"Connection error: {exc}") from exc
    raise RuntimeError("Unreachable.")


_TYPE_NAMES = {bool: "boolean (true or false)", str: "string", int: "integer", float: "number"}


def _json_instruction(action) -> str:
    fields = ", ".join(
        f"{name} ({_TYPE_NAMES.get(t, t.__name__)})"
        for name, t in action.outputs
    )
    return "Return only valid JSON with exactly these keys: " + fields


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


class ManagedBackend:
    """Lazy backend wrapper with optional idle release.

    The backend factory is called only when an LLM action reaches this backend.
    If ``idle_timeout`` is set, ``release`` is called after the backend has been
    idle for that many seconds. A timeout of ``0`` releases immediately after
    each call.
    """

    def __init__(
        self,
        factory: Callable[[], Callable],
        *,
        release: Callable[[], None] | None = None,
        idle_timeout: float | None = None,
    ):
        if idle_timeout is not None and idle_timeout < 0:
            raise ValueError("idle_timeout must be non-negative.")
        self._factory = factory
        self._release = release
        self._idle_timeout = idle_timeout
        self._backend: Callable | None = None
        self._timer: threading.Timer | None = None
        self._lock = threading.RLock()
        self._active = 0
        self._last_used = 0.0

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._backend is not None

    def __call__(self, action, inputs: dict[str, object]) -> dict[str, object]:
        backend = self._acquire()
        try:
            return backend(action, inputs)
        finally:
            self._finish_call()

    def close(self) -> None:
        self.release()

    def release(self) -> None:
        release = None
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            if self._backend is not None:
                self._backend = None
                release = self._release
        if release is not None:
            release()

    def _acquire(self) -> Callable:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            if self._backend is None:
                self._backend = self._factory()
            self._active += 1
            return self._backend

    def _finish_call(self) -> None:
        release_now = None
        with self._lock:
            self._active -= 1
            self._last_used = time.monotonic()
            if self._active != 0 or self._idle_timeout is None:
                return
            if self._idle_timeout == 0:
                self._backend = None
                release_now = self._release
            else:
                self._timer = threading.Timer(self._idle_timeout, self._release_if_idle)
                self._timer.daemon = True
                self._timer.start()
        if release_now is not None:
            release_now()

    def _release_if_idle(self) -> None:
        release = None
        with self._lock:
            if self._backend is None or self._active:
                return
            assert self._idle_timeout is not None
            elapsed = time.monotonic() - self._last_used
            remaining = self._idle_timeout - elapsed
            if remaining > 0:
                self._timer = threading.Timer(remaining, self._release_if_idle)
                self._timer.daemon = True
                self._timer.start()
                return
            self._timer = None
            self._backend = None
            release = self._release
        if release is not None:
            release()


def make_mistral_backend(
    *,
    api_key: str,
    model: str = "mistral-small-latest",
    temperature: float = 0.2,
    max_tokens: int = 2048,
    timeout: float = 90.0,
    max_retries: int = 3,
) -> Callable:
    """Return a Mistral backend callable compatible with ``Workflow.configure``."""

    def backend(action, inputs: dict[str, object]) -> dict[str, object]:
        messages, use_json = _build_messages(action, inputs)
        payload: dict = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
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

    return backend


def _openai_uses_completion_tokens(model: str) -> bool:
    """Return True for models that require max_completion_tokens and reject temperature."""
    import re as _re
    return bool(_re.match(r'^o\d', model)) or model.startswith("gpt-5")


def make_openai_backend(
    *,
    api_key: str,
    model: str = "gpt-4o-mini",
    base_url: str = "https://api.openai.com/v1",
    temperature: float = 0.2,
    max_tokens: int = 2048,
    timeout: float = 90.0,
    max_retries: int = 3,
) -> Callable:
    """Return an OpenAI-compatible backend callable for ``Workflow.configure``."""

    endpoint = base_url.rstrip("/") + "/chat/completions"

    def backend(action, inputs: dict[str, object]) -> dict[str, object]:
        messages, use_json = _build_messages(action, inputs)
        payload: dict = {"model": model, "messages": messages}
        if _openai_uses_completion_tokens(model):
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["temperature"] = temperature
            payload["max_tokens"] = max_tokens
        if use_json:
            payload["response_format"] = {"type": "json_object"}
        req = request.Request(
            endpoint,
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

    return backend


def make_anthropic_backend(
    *,
    api_key: str,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 1024,
    timeout: float = 90.0,
    max_retries: int = 3,
) -> Callable:
    """Return an Anthropic Claude backend callable compatible with ``Workflow.configure``."""

    def backend(action, inputs: dict[str, object]) -> dict[str, object]:
        user_prompt = action.user_prompt.format(**inputs)
        parse = getattr(action, "parse_format", "json") or "json"
        if parse == "json":
            content = f"{user_prompt}\n\n{_json_instruction(action)}"
        elif parse == "bool" and len(action.outputs) == 1:
            content = f"{user_prompt}\n\nReply with exactly one word: true or false."
        else:
            content = user_prompt
        payload: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "system": action.system_prompt,
            "messages": [{"role": "user", "content": content}],
        }

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

    return backend


def make_lifeline_router(backends: dict[str, Callable]) -> Callable:
    """Route LLM calls to the backend registered for the calling lifeline.

    The calling lifeline is identified by the current thread name, which the
    runtime sets to the lifeline name when it creates each thread.
    """

    def backend(action, inputs: dict[str, object]) -> dict[str, object]:
        lifeline_name = threading.current_thread().name
        if lifeline_name not in backends:
            raise RuntimeError(f"No backend configured for lifeline {lifeline_name!r}.")
        return backends[lifeline_name](action, inputs)

    return backend


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}.") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number, got {raw!r}.") from exc


def _env_optional_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number, got {raw!r}.") from exc
    if value < 0:
        raise RuntimeError(f"{name} must be non-negative, got {raw!r}.")
    return value


def _split_llm_spec(spec: str) -> tuple[str, str | None]:
    provider, sep, model = spec.strip().partition(":")
    provider = provider.strip().lower()
    model = model.strip() if sep else None
    if not provider:
        raise RuntimeError("LLM spec is empty.")
    if sep and not model:
        raise RuntimeError(f"LLM spec {spec!r} is missing a model after ':'.")
    return provider, model


def _ollama_native_base_url(openai_base_url: str) -> str:
    base = openai_base_url.rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def _make_ollama_release(*, model: str, base_url: str, timeout: float) -> Callable[[], None]:
    endpoint = _ollama_native_base_url(base_url) + "/api/chat"

    def release() -> None:
        payload = {"model": model, "messages": [], "keep_alive": 0}
        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            _retry_json_request(req, timeout=timeout, max_retries=0)
        except RuntimeError:
            # Best effort: model release should not make a workflow fail after
            # the LLM action already completed successfully.
            pass

    return release


def backend_from_spec(
    spec: str,
    *,
    fallback: Callable | None = None,
    idle_timeout: float | None = None,
) -> tuple[Callable, str]:
    """Build an LLM backend from a compact spec such as ``"openai:gpt-4o"``.

    Supported specs:
    - ``"mock"`` for the supplied fallback backend
    - ``"openai"`` or ``"openai:<model>"``
    - ``"ollama"`` / ``"local"`` or ``"ollama:<model>"``
    - ``"mistral"`` or ``"mistral:<model>"``
    - ``"anthropic"`` / ``"claude"`` or ``"claude:<model>"``

    API keys and base URLs come from environment variables.  For example,
    ``OPENAI_API_KEY`` is used for OpenAI and ``OLLAMA_BASE_URL`` can override
    the local Ollama endpoint.
    """

    provider, model = _split_llm_spec(spec)
    if provider == "mock":
        if fallback is None:
            raise RuntimeError("LLM spec 'mock' requires a fallback backend.")
        return fallback, "mock LLM"
    if provider == "mistral":
        api_key = os.environ.get("MISTRAL_API_KEY")
        model = model or os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY is not set.")
        return (
            make_mistral_backend(
                api_key=api_key,
                model=model,
                max_tokens=_env_int("MISTRAL_MAX_TOKENS", 2048),
                timeout=_env_float("MISTRAL_TIMEOUT", 90.0),
            ),
            f"Mistral ({model})",
        )
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        return (
            make_openai_backend(
                api_key=api_key,
                model=model,
                base_url=base_url,
                max_tokens=_env_int("OPENAI_MAX_TOKENS", 2048),
                timeout=_env_float("OPENAI_TIMEOUT", 90.0),
            ),
            f"OpenAI-compatible ({model})",
        )
    if provider in {"ollama", "local"}:
        model = model or os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
        api_key = os.environ.get("OLLAMA_API_KEY", "ollama")
        max_tokens = _env_int("OLLAMA_MAX_TOKENS", 512)
        timeout = _env_float("OLLAMA_TIMEOUT", 120.0)
        if idle_timeout is None:
            idle_timeout = _env_optional_float("OLLAMA_IDLE_TIMEOUT")
        release_timeout = _env_float("OLLAMA_RELEASE_TIMEOUT", 5.0)
        return (
            ManagedBackend(
                lambda: make_openai_backend(
                    api_key=api_key,
                    model=model,
                    base_url=base_url,
                    max_tokens=max_tokens,
                    timeout=timeout,
                ),
                release=_make_ollama_release(
                    model=model,
                    base_url=base_url,
                    timeout=release_timeout,
                ),
                idle_timeout=idle_timeout,
            ),
            f"Ollama ({model})",
        )
    if provider in {"anthropic", "claude"}:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        return (
            make_anthropic_backend(
                api_key=api_key,
                model=model,
                max_tokens=_env_int("ANTHROPIC_MAX_TOKENS", 1024),
                timeout=_env_float("ANTHROPIC_TIMEOUT", 90.0),
            ),
            f"Claude ({model})",
        )
    raise RuntimeError(
        f"Unsupported LLM provider {provider!r}. Use 'mock', 'openai:<model>', "
        "'ollama:<model>', 'mistral:<model>', or 'claude:<model>'."
    )


def router_from_specs(
    routes: dict[str, str | Callable],
    *,
    fallback: Callable | None = None,
    fallback_label: str = "mock LLM",
    idle_timeout: float | None = None,
) -> tuple[Callable, str]:
    """Build a per-lifeline backend router from compact LLM specs.

    Values in ``routes`` can be an LLM spec string (``"openai:gpt-4o"``,
    ``"ollama:qwen2.5:7b"``, ``"mistral"``, ``"mock"``) or a pre-built backend callable
    (e.g. ``make_mistral_backend(api_key=...)``).
    """

    if not routes:
        if fallback is None:
            raise RuntimeError("No routes configured.")
        return fallback, fallback_label

    built_backends: dict[str, Callable] = {}
    labels: list[str] = []
    for lifeline_name, provider in routes.items():
        if callable(provider):
            built_backends[lifeline_name] = provider
            labels.append(f"{lifeline_name}=custom")
        else:
            backend, label = backend_from_spec(
                provider,
                fallback=fallback,
                idle_timeout=idle_timeout,
            )
            built_backends[lifeline_name] = backend
            labels.append(f"{lifeline_name}={label}")
    return make_lifeline_router(built_backends), ", ".join(labels)


def router_from_env(
    routes: dict[str, str | Callable],
    *,
    fallback: Callable | None = None,
    fallback_label: str = "mock LLM",
    idle_timeout: float | None = None,
) -> tuple[Callable, str]:
    """Backward-compatible alias for :func:`router_from_specs`."""

    return router_from_specs(
        routes,
        fallback=fallback,
        fallback_label=fallback_label,
        idle_timeout=idle_timeout,
    )
