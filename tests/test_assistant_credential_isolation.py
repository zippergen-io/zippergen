"""No workflow credential reaches an assistant subprocess.

An assistant action reads workflow values, which may come from untrusted
input, and runs a CLI with shell access. A workflow process holds credentials
for every model and connector it routes to, and two of those names --
``OPENAI_API_KEY`` and ``ANTHROPIC_API_KEY`` -- are also names the assistant
CLIs read. Forwarding them both crossed the boundary and silently overrode the
login the operator established, spending the wrong account.

The rule is stated once here as a cross product: every way a workflow can hold
a credential, against every assistant backend.
"""

import os

from zippergen.assistant_backends import (
    _ASSISTANT_AUTH_ENVIRONMENT,
    _assistant_environment,
)

import pytest


BACKENDS = sorted(_ASSISTANT_AUTH_ENVIRONMENT)

#: Every environment variable through which a workflow may hold a credential:
#: model provider keys, provider base URLs, and connector secrets.
WORKFLOW_CREDENTIALS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "AZURE_OPENAI_API_KEY",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CODEX_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "MISTRAL_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "TELEGRAM_BOT_TOKEN",
    "ZIPPERGEN_CONNECTORS_JSON",
)

#: Where each CLI keeps the login it established for itself. These must
#: survive, or the assistant cannot authenticate at all.
LOGIN_LOCATIONS = {
    "codex": "CODEX_HOME",
    "claude": "CLAUDE_CONFIG_DIR",
}

SENTINEL = "workflow-credential-sentinel"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("credential", WORKFLOW_CREDENTIALS)
def test_no_workflow_credential_reaches_the_child(
    monkeypatch: pytest.MonkeyPatch, backend: str, credential: str
) -> None:
    monkeypatch.setenv(credential, SENTINEL)
    assert credential not in _assistant_environment(backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_child_keeps_the_path_to_its_own_login(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    monkeypatch.setenv("HOME", "/home/operator")
    monkeypatch.setenv(LOGIN_LOCATIONS[backend], "/home/operator/.cli")
    child = _assistant_environment(backend)
    assert child["HOME"] == "/home/operator"
    assert child[LOGIN_LOCATIONS[backend]] == "/home/operator/.cli"


@pytest.mark.parametrize("backend", BACKENDS)
def test_one_backend_never_sees_another_backends_configuration(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    for location in LOGIN_LOCATIONS.values():
        monkeypatch.setenv(location, "/home/operator/.cli")
    child = _assistant_environment(backend)
    foreign = set(LOGIN_LOCATIONS.values()) - {LOGIN_LOCATIONS[backend]}
    assert not foreign & set(child)


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_allowed_set_holds_no_credential_name(backend: str) -> None:
    """The list itself is checked, not only what today's environment holds."""

    assert not set(WORKFLOW_CREDENTIALS) & _ASSISTANT_AUTH_ENVIRONMENT[backend]


@pytest.mark.parametrize("backend", BACKENDS)
def test_an_unrelated_setting_still_passes_through(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """Least privilege, not an empty environment: the CLI still needs a PATH."""

    monkeypatch.setenv("PATH", "/usr/bin")
    assert _assistant_environment(backend)["PATH"] == "/usr/bin"
    assert "ZIPPERGEN_SENTINEL_UNKNOWN" not in os.environ
