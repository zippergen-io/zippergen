"""Small ready-to-use LLM backend helpers for examples and quick starts."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from datetime import datetime
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zippergen.models import ModelSettings
from dataclasses import dataclass
from pathlib import Path
from urllib import request
from urllib.error import HTTPError, URLError

from zippergen.llm_policy import (
    LLMError,
    LLMInvalidResponseError,
    LLMPermanentError,
    LLMTransientError,
)
from zippergen.syntax import Json, validate_zvalue

__all__ = [
    "ManagedBackend",
    "backend_from_spec",
    "make_mistral_backend",
    "make_openai_backend",
    "make_anthropic_backend",
    "make_lifeline_router",
    "PROVIDER_API_KEY_VARIABLES",
    "make_scripted_backend",
    "load_scripted_script",
    "validate_local_idle_policies",
    "router_from_specs",
]


# Which environment variable holds each provider's API key. `backend_from_spec`
# reads these directly; anything that needs to *report* what is missing without
# building a backend should use this rather than repeating the names.
PROVIDER_API_KEY_VARIABLES: Mapping[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}

_DEFAULT_TEMPERATURE = 0.2


def _selected_temperature(action, configured: float | None) -> tuple[float, bool]:
    """Return the action-over-model temperature and whether it was explicit."""

    action_value = getattr(action, "temperature", None)
    explicit = action_value is not None or configured is not None
    selected = action_value if action_value is not None else configured
    value = _DEFAULT_TEMPERATURE if selected is None else float(selected)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise LLMPermanentError("LLM temperature must be between 0 and 1.")
    return value, explicit


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
    if type_ is Json:
        try:
            return validate_zvalue(value, Json)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc
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


#: Statuses worth waiting for. Everything else the provider returns is a
#: statement about the request, and repeating it produces the same statement.
_TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _retry_after_seconds(headers) -> float | None:
    """Read ``Retry-After`` in either standard form.

    The header is defined as a delay in seconds or an HTTP date. Supporting
    only the first silently ignored providers that send the second, which then
    got the invented backoff instead of the wait they asked for.
    """

    raw = headers.get("Retry-After") if headers else None
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        pass
    try:
        from email.utils import parsedate_to_datetime

        when = parsedate_to_datetime(str(raw))
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    now = datetime.now(when.tzinfo) if when.tzinfo else datetime.now()
    return max(0.0, (when - now).total_seconds())


def _chat_completion_text(body: object, *, provider: str) -> str:
    """Pull the message text out of an OpenAI-shaped envelope.

    A body can be valid JSON and still unusable -- a missing key, an empty
    choices list, a non-string content. That is a malformed response, worth
    another sample, not a defect in ZipperGen. Only the response's shape is
    converted here; anything else this function could raise would be a bug and
    is left alone.
    """

    try:
        content = body["choices"][0]["message"]["content"]  # type: ignore[index]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMInvalidResponseError(
            f"{provider} response has no message content: {_short_content(repr(body))}"
        ) from exc
    if not isinstance(content, str):
        raise LLMInvalidResponseError(
            f"Unexpected {provider} response content: {content!r}"
        )
    return content


def _anthropic_text(body: object) -> str:
    """Join the text blocks of an Anthropic envelope, or say it is unusable."""

    content_blocks = body.get("content") if isinstance(body, dict) else None
    text_parts = (
        [
            block["text"]
            for block in content_blocks
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ]
        if isinstance(content_blocks, list)
        else []
    )
    if not text_parts:
        raise LLMInvalidResponseError(
            f"Unexpected Anthropic response content: "
            f"{_short_content(repr(content_blocks))}"
        )
    return "\n".join(text_parts)


def _json_request(req: request.Request, *, timeout: float) -> dict:
    """Perform one HTTP call and classify any failure.

    There is no retry here. This is the boundary that knows what a provider's
    status codes mean; deciding what to do about them belongs to one policy,
    in ``llm_policy``, which also sees parsing and validation failures.
    """

    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code in _TRANSIENT_STATUS:
            raise LLMTransientError(
                f"API error {exc.code}: {detail}",
                retry_after=_retry_after_seconds(exc.headers),
            ) from exc
        raise LLMPermanentError(f"API error {exc.code}: {detail}") from exc
    except URLError as exc:
        raise LLMTransientError(f"Could not reach API: {exc.reason}") from exc
    except TimeoutError as exc:
        raise LLMTransientError(f"API request timed out after {timeout}s") from exc
    except OSError as exc:
        # ConnectionResetError and similar, raised while reading the response
        # after urlopen() has already succeeded.
        raise LLMTransientError(f"Connection error: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise LLMInvalidResponseError(
            f"API returned a body that is not JSON: {exc.msg}"
        ) from exc


_TYPE_NAMES = {
    bool: "boolean (true or false)",
    str: "string",
    int: "integer",
    float: "number",
    Json: "JSON value",
}


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
        stripped = content.strip()
        if not stripped:
            raise LLMInvalidResponseError(
                f"LLM action '{action.name}' returned empty {parse} output."
            )
        try:
            return {name: _coerce_output(stripped, type_)}
        except ValueError as exc:
            raise LLMInvalidResponseError(
                f"LLM action '{action.name}' returned invalid {parse} output: {exc}. "
                f"Raw response: {_short_content(content)!r}"
            ) from exc

    try:
        raw_outputs = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMInvalidResponseError(
            f"LLM action '{action.name}' returned invalid JSON: {exc.msg}. "
            f"Raw response: {_short_content(content)!r}"
        ) from exc

    try:
        return _coerce_outputs(action, raw_outputs)
    except ValueError as exc:
        expected = ", ".join(name for name, _ in action.outputs)
        raise LLMInvalidResponseError(
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
        if idle_timeout is not None and (
            not math.isfinite(idle_timeout) or idle_timeout < 0
        ):
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
    temperature: float | None = None,
    max_tokens: int = 2048,
    timeout: float = 90.0,
) -> Callable:
    """Return a Mistral backend callable compatible with ``Workflow.configure``."""

    def backend(action, inputs: dict[str, object]) -> dict[str, object]:
        messages, use_json = _build_messages(action, inputs)
        selected_temperature, _explicit = _selected_temperature(action, temperature)
        payload: dict = {
            "model": model,
            "temperature": selected_temperature,
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
        body = _json_request(req, timeout=timeout)
        content = _chat_completion_text(body, provider="Mistral")
        return _parse_response(action, content)

    return backend


def _openai_uses_completion_tokens(model: str) -> bool:
    """Return True for models that require max_completion_tokens and reject temperature."""
    import re as _re
    return bool(_re.match(r'^o\d', model)) or model.startswith("gpt-5")


def _anthropic_rejects_sampling_parameters(model: str) -> bool:
    """Return True for Claude models that removed temperature/top-p/top-k."""

    normalized = model.casefold().replace(".", "-").replace("_", "-")
    families = (
        "fable-5",
        "mythos-5",
        "mythos-preview",
        "opus-5",
        "opus-4-8",
        "opus-4-7",
        "sonnet-5",
    )
    return any(family in normalized for family in families)


def model_accepts_temperature(spec: str) -> bool:
    """Return whether this routed model accepts a temperature parameter."""

    provider_token, model = _split_llm_spec(spec)
    provider, _connection = _provider_parts(provider_token)
    if model is None:
        return True
    if provider in {"openai", "ollama", "local"}:
        return not _openai_uses_completion_tokens(model)
    if provider in {"anthropic", "claude"}:
        return not _anthropic_rejects_sampling_parameters(model)
    return True


def make_openai_backend(
    *,
    api_key: str,
    model: str = "gpt-4o-mini",
    base_url: str = "https://api.openai.com/v1",
    temperature: float | None = None,
    max_tokens: int = 2048,
    timeout: float = 90.0,
) -> Callable:
    """Return an OpenAI-compatible backend callable for ``Workflow.configure``."""

    endpoint = base_url.rstrip("/") + "/chat/completions"

    def backend(action, inputs: dict[str, object]) -> dict[str, object]:
        messages, use_json = _build_messages(action, inputs)
        selected_temperature, explicit_temperature = _selected_temperature(
            action, temperature
        )
        payload: dict = {"model": model, "messages": messages}
        if _openai_uses_completion_tokens(model):
            if explicit_temperature:
                raise LLMPermanentError(
                    f"Model {model!r} does not support an explicit temperature."
                )
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["temperature"] = selected_temperature
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
        body = _json_request(req, timeout=timeout)
        content = _chat_completion_text(body, provider="OpenAI")
        return _parse_response(action, content)

    return backend


def make_anthropic_backend(
    *,
    api_key: str,
    model: str = "claude-sonnet-4-6",
    temperature: float | None = None,
    max_tokens: int = 1024,
    timeout: float = 90.0,
) -> Callable:
    """Return an Anthropic Claude backend callable compatible with ``Workflow.configure``."""

    def backend(action, inputs: dict[str, object]) -> dict[str, object]:
        selected_temperature, explicit_temperature = _selected_temperature(
            action, temperature
        )
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
        if _anthropic_rejects_sampling_parameters(model):
            if explicit_temperature:
                raise LLMPermanentError(
                    f"Model {model!r} does not support an explicit temperature; "
                    "use Anthropic output_config.effort where appropriate."
                )
        else:
            payload["temperature"] = selected_temperature

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
        body = _json_request(req, timeout=timeout)
        return _parse_response(action, _anthropic_text(body))

    return backend


@dataclass(frozen=True)
class _ScriptedResponses:
    """Either a constant answer, or a finite sequence that must be consumed.

    The distinction is the point. A constant says *"this participant always
    answers this way"*; a sequence says *"exactly these calls are expected"*,
    and running past its end is a control-flow change worth failing on.
    """

    responses: tuple[dict[str, object], ...]
    repeating: bool


def _read_scripted_entry(key: str, value: object) -> _ScriptedResponses:
    """A bare object repeats; a list is a finite, exhaustible sequence."""

    entries = value if isinstance(value, list) else [value]
    normalized: list[dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError(
                f"Scripted responses for {key!r} must be objects mapping "
                f"output name to value; got {type(entry).__name__}."
            )
        response: dict[str, object] = {}
        for output_name, output_value in entry.items():
            if not isinstance(output_name, str):
                raise RuntimeError(
                    f"Scripted response output names for {key!r} must be strings."
                )
            response[output_name] = output_value
        normalized.append(response)
    if not entries:
        raise RuntimeError(
            f"Scripted responses for {key!r} are empty; give at least one."
        )
    return _ScriptedResponses(
        responses=tuple(normalized),
        repeating=not isinstance(value, list),
    )


def load_scripted_script(path: str | Path) -> dict[str, _ScriptedResponses]:
    """Read a scripted-response file, checking its shape before it is used.

    ``{"LLM1.assess": {...}}`` answers every call the same way.
    ``{"LLM1.assess": [{...}, {...}]}`` expects exactly two calls.
    """

    source = Path(path).expanduser()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Scripted response file not found: {source}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Scripted response file {source} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise RuntimeError(
            f"Scripted response file {source} must be a JSON object mapping "
            "'Participant.action' or 'action' to a response or a list of them."
        )
    return {
        str(key): _read_scripted_entry(str(key), value)
        for key, value in raw.items()
    }


def make_scripted_backend(
    script: Mapping[str, object],
) -> Callable:
    """Replay recorded outputs so every protocol branch is reachable.

    ``mock`` returns one placeholder for every action, so a workflow driven by
    it takes whichever branch that placeholder selects and no other. In a
    consensus protocol that means immediate agreement, and everything behind a
    decision is unreachable. This backend answers per action instead.

    Keys are ``"Participant.action"``, falling back to ``"action"`` for every
    participant. A bare object answers every call the same way; a list is a
    finite sequence and **running past its end is an error**, so a control-flow
    change that calls an action more often than expected fails loudly rather
    than being absorbed.
    """

    entries = {
        key: value
        if isinstance(value, _ScriptedResponses)
        else _read_scripted_entry(key, value)
        for key, value in script.items()
    }
    used: dict[str, int] = {key: 0 for key in entries}
    lock = threading.Lock()

    def backend(action, inputs: dict[str, object]) -> dict[str, object]:
        del inputs  # a scripted response does not depend on the prompt
        lifeline_name = threading.current_thread().name
        target = f"{lifeline_name}.{action.name}"
        key = target if target in entries else action.name
        if key not in entries:
            known = ", ".join(sorted(entries)) or "none"
            raise RuntimeError(
                f"No scripted response for {target!r}. Add {target!r} or "
                f"{action.name!r} to the response file. Scripted: {known}."
            )

        entry = entries[key]
        with lock:
            index = used[key]
            used[key] = index + 1
        if entry.repeating:
            response = entry.responses[0]
        elif index < len(entry.responses):
            response = entry.responses[index]
        else:
            count = len(entry.responses)
            raise RuntimeError(
                f"Scripted responses for {key!r} are exhausted: "
                f"{count} given, call {index + 1} requested. Either the "
                "workflow calls it more often than expected, or the script "
                "needs more entries. Use a bare object instead of a list to "
                "answer every call the same way."
            )

        expected = [name for name, _type in action.outputs]
        absent = [name for name in expected if name not in response]
        if absent:
            raise RuntimeError(
                f"Scripted response for {key!r} is missing "
                f"{', '.join(absent)}; {action.name} declares "
                f"{', '.join(expected)}."
            )
        return {name: response[name] for name in expected}

    return backend


def make_lifeline_router(backends: dict[str, Callable]) -> Callable:
    """Route LLM calls by action override, then by calling lifeline.

    The calling lifeline is identified by the current thread name, which the
    runtime sets to the lifeline name when it creates each thread.  A key such
    as ``Writer.revise_answer`` takes precedence over ``Writer``.
    """

    def backend(action, inputs: dict[str, object]) -> dict[str, object]:
        lifeline_name = threading.current_thread().name
        action_target = f"{lifeline_name}.{action.name}"
        selected = backends.get(action_target) or backends.get(lifeline_name)
        if selected is None:
            raise RuntimeError(f"No backend configured for lifeline {lifeline_name!r}.")
        return selected(action, inputs)

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
    from zippergen.provider_connections import split_model_spec

    try:
        provider, connection, model = split_model_spec(spec)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    return provider + (f"@{connection}" if connection else ""), model


def _provider_parts(provider_token: str) -> tuple[str, str | None]:
    provider, separator, connection = provider_token.partition("@")
    return provider, connection if separator else None


def _connection_environment(
    connection: str | None,
    field: str,
    fallback: str,
    default: str | None = None,
) -> str | None:
    if connection:
        from zippergen.provider_connections import provider_environment_name

        return os.environ.get(provider_environment_name(connection, field), default)
    return os.environ.get(fallback, default)


def _missing_credential(connection: str | None, environment: str) -> RuntimeError:
    if connection:
        return RuntimeError(
            f"Provider connection {connection!r} has no credential in this "
            f"process. Run 'zg provider set-credential {connection}' in the "
            "project, or provide its standard environment variable before "
            "ZipperGen resolves the connection."
        )
    return RuntimeError(f"{environment} is not set.")


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
            _json_request(req, timeout=timeout)
        except (LLMError, OSError):
            # Best effort: model release should not make a workflow fail after
            # the LLM action already completed successfully.
            pass

    return release


def backend_from_spec(
    spec: str,
    *,
    fallback: Callable | None = None,
    settings: "ModelSettings | None" = None,
) -> tuple[Callable, str]:
    """Build an LLM backend from a compact spec such as ``"openai:gpt-4o"``.

    Supported specs:
    - ``"mock"`` for the supplied fallback backend
    - ``"scripted:<file.json>"`` for deterministic recorded responses
    - ``"openai"`` or ``"openai:<model>"``
    - ``"ollama"`` / ``"local"`` or ``"ollama:<model>"``
    - ``"mistral"`` or ``"mistral:<model>"``
    - ``"anthropic"`` / ``"claude"`` or ``"claude:<model>"``

    API keys and base URLs come from environment variables.  For example,
    ``OPENAI_API_KEY`` is used for OpenAI and ``OLLAMA_BASE_URL`` can override
    the local Ollama endpoint.
    """

    from zippergen.models import ModelSettings

    chosen = settings or ModelSettings()
    temperature = chosen.temperature
    idle_timeout = chosen.idle_timeout

    def sized(variable: str, fallback_value: int) -> int:
        """Configured max tokens, else the environment, else the built-in default.

        Note the order: configuration wins. The environment is a fallback for a
        setting nobody configured, not an override of one somebody did -- a
        value written beside the model must not be changed by a process-wide
        variable that says nothing about which model it meant.
        """

        return (
            chosen.max_tokens
            if chosen.max_tokens is not None
            else _env_int(variable, fallback_value)
        )

    def waited(variable: str, fallback_value: float) -> float:
        return (
            chosen.timeout
            if chosen.timeout is not None
            else _env_float(variable, fallback_value)
        )

    provider_token, model = _split_llm_spec(spec)
    provider, connection = _provider_parts(provider_token)
    if provider == "scripted":
        if not model:
            raise RuntimeError(
                "LLM spec 'scripted' needs a response file: "
                "scripted:responses.json"
            )
        return (
            make_scripted_backend(load_scripted_script(model)),
            f"scripted ({model})",
        )
    if provider == "mock":
        if fallback is None:
            raise RuntimeError("LLM spec 'mock' requires a fallback backend.")
        return fallback, "mock LLM"
    if provider == "mistral":
        api_key = _connection_environment(
            connection, "api_key", "MISTRAL_API_KEY"
        )
        model = model or os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
        if not api_key:
            raise _missing_credential(connection, "MISTRAL_API_KEY")
        return (
            make_mistral_backend(
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=sized("MISTRAL_MAX_TOKENS", 2048),
                timeout=waited("MISTRAL_TIMEOUT", 90.0),
            ),
            f"Mistral ({model})",
        )
    if provider == "openai":
        api_key = _connection_environment(
            connection, "api_key", "OPENAI_API_KEY"
        )
        model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        base_url = _connection_environment(
            connection,
            "base_url",
            "OPENAI_BASE_URL",
            "https://api.openai.com/v1",
        )
        if not api_key:
            raise _missing_credential(connection, "OPENAI_API_KEY")
        assert base_url is not None
        return (
            make_openai_backend(
                api_key=api_key,
                model=model,
                base_url=base_url,
                temperature=temperature,
                max_tokens=sized("OPENAI_MAX_TOKENS", 2048),
                timeout=waited("OPENAI_TIMEOUT", 90.0),
            ),
            f"OpenAI-compatible ({model})",
        )
    if provider in {"ollama", "local"}:
        model = model or os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
        base_url = _connection_environment(
            connection,
            "base_url",
            "OLLAMA_BASE_URL",
            "http://127.0.0.1:11434/v1",
        )
        api_key = _connection_environment(
            connection, "api_key", "OLLAMA_API_KEY", "ollama"
        )
        assert base_url is not None
        assert api_key is not None
        max_tokens = sized("OLLAMA_MAX_TOKENS", 512)
        timeout = waited("OLLAMA_TIMEOUT", 120.0)
        if idle_timeout is None:
            idle_timeout = _env_optional_float("OLLAMA_IDLE_TIMEOUT")
        release_timeout = _env_float("OLLAMA_RELEASE_TIMEOUT", 5.0)
        return (
            ManagedBackend(
                lambda: make_openai_backend(
                    api_key=api_key,
                    model=model,
                    base_url=base_url,
                    temperature=temperature,
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
        api_key = _connection_environment(
            connection, "api_key", "ANTHROPIC_API_KEY"
        )
        model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        if not api_key:
            raise _missing_credential(connection, "ANTHROPIC_API_KEY")
        return (
            make_anthropic_backend(
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=sized("ANTHROPIC_MAX_TOKENS", 1024),
                timeout=waited("ANTHROPIC_TIMEOUT", 90.0),
            ),
            f"Claude ({model})",
        )
    raise RuntimeError(
        f"Unsupported LLM provider {provider!r}. Use 'mock', 'openai:<model>', "
        "'ollama:<model>', 'mistral:<model>', or 'claude:<model>'."
    )


def validate_local_idle_policies(
    routes: Mapping[str, str | Callable],
    *,
    idle_timeout: float | None = None,
    settings: "Mapping[str, ModelSettings] | None" = None,
) -> None:
    """Reject contradictory release policies for one physical local model."""

    route_settings = dict(settings or {})
    unknown_idle_routes = sorted(set(route_settings) - set(routes))
    if unknown_idle_routes:
        raise ValueError(
            "Idle release refers to unknown LLM route(s): "
            + ", ".join(unknown_idle_routes)
        )
    route_idle_timeouts = {
        route: chosen.idle_timeout
        for route, chosen in route_settings.items()
        if chosen.idle_timeout is not None
    }

    policies: dict[str, list[tuple[str, float | None]]] = {}
    for route, provider in routes.items():
        if callable(provider):
            continue
        provider_token, model = _split_llm_spec(provider)
        provider_name, connection = _provider_parts(provider_token)
        if provider_name not in {"local", "ollama"}:
            continue
        identity = f"@{connection}" if connection else ""
        physical_spec = (
            f"local{identity}:{model}" if model is not None else f"local{identity}"
        )
        selected = route_idle_timeouts.get(route, idle_timeout)
        policies.setdefault(physical_spec, []).append((route, selected))

    for physical_spec, entries in policies.items():
        if len({policy for _route, policy in entries}) <= 1:
            continue
        details = ", ".join(
            f"{route}={'never' if policy is None else f'{policy:g} s'}"
            for route, policy in entries
        )
        raise ValueError(
            f"Local model {physical_spec!r} has conflicting idle release "
            f"policies: {details}."
        )


def router_from_specs(
    routes: dict[str, str | Callable],
    *,
    fallback: Callable | None = None,
    fallback_label: str = "mock LLM",
    idle_timeout: float | None = None,
    settings: "Mapping[str, ModelSettings] | None" = None,
) -> tuple[Callable, str]:
    """Build a participant and action backend router from compact LLM specs.

    Values in ``routes`` can be an LLM spec string (``"openai:gpt-4o"``,
    ``"ollama:qwen2.5:7b"``, ``"mistral"``, ``"mock"``) or a pre-built backend callable
    (e.g. ``make_mistral_backend(api_key=...)``).
    """

    if not routes:
        if fallback is None:
            raise RuntimeError("No routes configured.")
        return fallback, fallback_label

    from zippergen.models import ModelSettings

    built_backends: dict[str, Callable] = {}
    labels: list[str] = []
    # Two backends may be shared only when every setting that reaches the
    # provider matches, so the whole settings value is part of the key.
    shared_backends: dict[
        tuple[str, ModelSettings], tuple[Callable, str]
    ] = {}
    route_settings = dict(settings or {})
    unknown_routes = sorted(set(route_settings) - set(routes))
    if unknown_routes:
        raise ValueError(
            "Model settings refer to unknown LLM route(s): "
            + ", ".join(unknown_routes)
        )
    validate_local_idle_policies(
        routes,
        idle_timeout=idle_timeout,
        settings=route_settings,
    )
    for lifeline_name, provider in routes.items():
        if callable(provider):
            built_backends[lifeline_name] = provider
            labels.append(f"{lifeline_name}=custom")
        else:
            chosen = route_settings.get(lifeline_name, ModelSettings())
            selected_idle_timeout = (
                chosen.idle_timeout
                if chosen.idle_timeout is not None
                else idle_timeout
            )
            provider_token, model = _split_llm_spec(provider)
            provider_name, connection = _provider_parts(provider_token)
            managed_local = provider_name in {"local", "ollama"}
            physical_spec = (
                f"local@{connection}:{model}"
                if connection and model is not None
                else (
                    f"local:{model}" if model is not None else "local"
                )
            ) if managed_local else provider
            effective = ModelSettings(
                temperature=chosen.temperature,
                max_tokens=chosen.max_tokens,
                timeout=chosen.timeout,
                idle_timeout=selected_idle_timeout if managed_local else None,
            )
            cache_key = (physical_spec, effective)
            cached = shared_backends.get(cache_key)
            if cached is None:
                cached = backend_from_spec(
                    provider,
                    fallback=fallback,
                    settings=effective,
                )
                shared_backends[cache_key] = cached
            backend, label = cached
            built_backends[lifeline_name] = backend
            labels.append(f"{lifeline_name}={label}")
    return make_lifeline_router(built_backends), ", ".join(labels)
