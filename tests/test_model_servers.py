"""Local model discovery uses a bounded read and never generates text."""

from io import BytesIO
import json
import ssl
from urllib.error import HTTPError, URLError

import pytest

from zippergen.model_servers import (
    ModelDiscoveryError,
    discover_models,
    display_server_url,
    is_connection_failure,
    server_connection_help,
)


def test_discovery_uses_model_list_and_preserves_server_prefix(monkeypatch):
    seen = []

    def respond(req, *, timeout):
        seen.append((req.full_url, req.get_method(), req.data, timeout))
        return BytesIO(json.dumps({"data": [
            {"id": "qwen3:32b"}, {"id": "another/model"}, {"id": "qwen3:32b"},
        ]}).encode())

    monkeypatch.setattr("zippergen.model_servers.request.urlopen", respond)
    assert discover_models("http://gpu:1234/prefix/v1/") == ("another/model", "qwen3:32b")
    assert seen == [("http://gpu:1234/prefix/v1/models", "GET", None, 3.0)]


@pytest.mark.parametrize("body", [
    b"<html>wrong endpoint</html>", b"{}", b'{"data":{}}',
    b'{"data":[{}]}', b'{"data":[{"id":12}]}',
    b'{"data":[{"id":" "}]}', b'[' * 2000,
    b'{"data":[{"id":"escape\\u001b[31m"}]}',
    b"x" * (1024 * 1024 + 1),
])
def test_unusable_lists_allow_a_controlled_manual_fallback(monkeypatch, body):
    monkeypatch.setattr("zippergen.model_servers.request.urlopen", lambda *_a, **_k: BytesIO(body))
    with pytest.raises(ModelDiscoveryError):
        discover_models("http://gpu:1234/v1")


def test_empty_list_is_distinct_from_connection_failure(monkeypatch):
    monkeypatch.setattr("zippergen.model_servers.request.urlopen", lambda *_a, **_k: BytesIO(b'{"data":[]}'))
    assert discover_models("http://gpu:1234/v1") == ()


@pytest.mark.parametrize("url", [
    "file:///tmp/models", "http://user:secret@gpu/v1", "http://gpu/v1?key=secret",
    "http://gpu:bad/v1", "http://gpu/v1#fragment",
])
def test_invalid_or_credential_bearing_urls_are_not_requested(monkeypatch, url):
    monkeypatch.setattr("zippergen.model_servers.request.urlopen", lambda *_a, **_k: pytest.fail("unexpected request"))
    with pytest.raises(ModelDiscoveryError):
        discover_models(url)


@pytest.mark.parametrize("error, message", [
    (URLError(ConnectionRefusedError(61, "refused")), "connection was refused"),
    (TimeoutError(), "did not respond in time"),
    (HTTPError("http://localhost/v1/models", 404, "private error body", {}, BytesIO()), "HTTP 404"),
    (URLError(ssl.SSLCertVerificationError()), "certificate could not be verified"),
])
def test_connection_errors_are_actionable_without_echoing_response_bodies(monkeypatch, error, message):
    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr("zippergen.model_servers.request.urlopen", fail)
    with pytest.raises(ModelDiscoveryError, match=message) as caught:
        discover_models("http://localhost:1234/v1")
    assert "private error body" not in str(caught.value)
    if "refused" in message:
        assert "SSH tunnel here" in str(caught.value)
    if "certificate" in message:
        assert "SSH" not in str(caught.value)


@pytest.mark.parametrize("url", ["http://127.0.0.1:11434/v1", "http://localhost:1234/v1", "http://[::1]:8000/v1"])
def test_loopback_help_refers_to_the_computer_running_zippergen(url):
    assert "computer running ZipperGen" in server_connection_help(url)
    assert "SSH tunnel here" in server_connection_help(url)
    assert "SSH" not in server_connection_help("http://gpu:11434/v1")


def test_url_display_does_not_disclose_credentials_or_query():
    assert display_server_url("https://user:secret@[::1]:1234/v1?key=private#fragment") == "https://[::1]:1234/v1"


def test_generation_failure_recognizes_wrapped_network_errors():
    wrapped = RuntimeError("backend error")
    wrapped.__cause__ = URLError(ConnectionRefusedError())
    assert is_connection_failure(wrapped)
    wrapped.__cause__ = HTTPError("http://gpu", 404, "missing model", {}, BytesIO())
    assert not is_connection_failure(wrapped)
    wrapped.__cause__ = URLError(ssl.SSLCertVerificationError())
    assert not is_connection_failure(wrapped)
    assert not is_connection_failure(RuntimeError("bad model output"))
