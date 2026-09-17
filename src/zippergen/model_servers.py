"""Discover models on a configured local server without generating text."""

from __future__ import annotations

import ipaddress
import json
import ssl
from http.client import HTTPException
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit


LOCAL_BASE_URL = "http://127.0.0.1:11434/v1"
_MAX_LIST_BYTES = 1024 * 1024


class ModelDiscoveryError(Exception):
    """The model list could not be obtained. Manual configuration still works."""


def display_server_url(base_url: str) -> str:
    """Show the destination without URL credentials or query parameters."""

    try:
        parsed = urlsplit(base_url)
        host = parsed.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        if parsed.port is not None:
            host += f":{parsed.port}"
        text = urlunsplit((parsed.scheme, host, parsed.path, "", ""))
        return "".join(char for char in text if char.isprintable())
    except ValueError:
        return "(invalid server URL)"


def server_connection_help(base_url: str) -> str:
    """Explain where a connection must be reachable, including local tunnels."""

    try:
        host = urlsplit(base_url).hostname or ""
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host.rstrip(".").casefold() == "localhost"
    except ValueError:
        loopback = False
    if loopback:
        return (
            "This address points to the computer running ZipperGen. "
            "Start the model server here, or start your SSH tunnel here if "
            "the model runs on another machine. An ordinary SSH login does "
            "not create a tunnel."
        )
    return "Check that the model server is running and reachable from this computer."


def is_connection_failure(exc: BaseException) -> bool:
    """Recognize network failures wrapped by the model backend."""

    seen: set[int] = set()
    while id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, (HTTPError, ssl.SSLError)):
            return False  # HTTP and TLS errors need their specific diagnostics.
        if isinstance(exc, URLError) and isinstance(exc.reason, ssl.SSLError):
            return False
        if isinstance(exc, (URLError, OSError)):
            return True
        cause = exc.__cause__
        if cause is None:
            break
        exc = cause
    return False


def discover_models(base_url: str, *, timeout: float = 3.0) -> tuple[str, ...]:
    """Read the OpenAI-compatible /models list used by Ollama and LM Studio.

    Discovery is a convenience, not a requirement of a model connection. No
    workflow contents, credentials or generation request are sent.
    """

    try:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Expected an HTTP(S) base URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Expected a base URL without credentials or a query")
        # Validate the port before urllib sees it.
        _ = parsed.port
        endpoint = base_url.rstrip("/") + "/models"
        req = request.Request(endpoint, headers={"Accept": "application/json"})
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read(_MAX_LIST_BYTES + 1)
    except HTTPError as exc:
        status = exc.code
        exc.close()
        raise ModelDiscoveryError(
            f"The model-list request returned HTTP {status}. "
            "Check the base URL. You can also enter a model name manually."
        ) from None
    except (URLError, OSError, HTTPException) as exc:
        reason = exc.reason if isinstance(exc, URLError) else exc
        if isinstance(reason, ssl.SSLCertVerificationError):
            raise ModelDiscoveryError(
                "The server's HTTPS certificate could not be verified. "
                "Check its certificate and your trusted CA certificates."
            ) from None
        elif isinstance(reason, TimeoutError):
            detail = "The server did not respond in time."
        elif isinstance(reason, ConnectionRefusedError):
            detail = "The connection was refused."
        else:
            detail = "Could not connect to the model server."
        raise ModelDiscoveryError(
            detail + " " + server_connection_help(base_url)
        ) from None
    except ValueError:
        raise ModelDiscoveryError(
            "Use an HTTP(S) base URL without credentials, query parameters or a fragment."
        ) from None

    if len(body) > _MAX_LIST_BYTES:
        raise ModelDiscoveryError("The server's model list is too large to display.")
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeError, RecursionError):
        raise ModelDiscoveryError("The server did not return a JSON model list.") from None
    entries = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ModelDiscoveryError("The server did not return an OpenAI-compatible model list.")
    models: set[str] = set()
    for item in entries:
        name = item.get("id") if isinstance(item, dict) else None
        if (
            not isinstance(name, str) or not name.strip()
            or len(name) > 512 or not name.isprintable()
        ):
            raise ModelDiscoveryError("The server returned an invalid model name.")
        models.add(name)
    return tuple(sorted(models))
