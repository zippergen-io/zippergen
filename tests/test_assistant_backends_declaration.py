"""Every supported assistant backend is executable, configurable, and offered.

The set ``{"codex", "claude"}`` used to be written out in ten places: routing,
runtime construction, workspace parsing and writing, checks, completion, and
two CLI choice lists. Each copy looked reasonable on its own, so a partial
update was the likely outcome -- a backend that runs but cannot be configured,
or is configurable but rejected on load.
"""

import pathlib
import re

from zippergen.assistant_backends import (
    ASSISTANT_BACKEND_SPECS,
    ASSISTANT_BACKENDS,
    _ASSISTANT_AUTH_ENVIRONMENT,
    _REQUIRED_CLI_OPTIONS,
    assistant_backend_spec,
)
from zippergen.serve import _parse_cli_args

import pytest


def test_no_module_writes_the_backend_set_out_by_hand() -> None:
    source_root = pathlib.Path(__file__).resolve().parents[1] / "src" / "zippergen"
    pair = re.compile(r'"codex",\s*"claude"|"claude",\s*"codex"')
    offenders = [
        path.name
        for path in source_root.rglob("*.py")
        if path.name != "assistant_backends.py" and pair.search(path.read_text())
    ]
    assert not offenders, (
        "these modules rebuild the backend list instead of reading "
        f"ASSISTANT_BACKENDS: {offenders}"
    )


@pytest.mark.parametrize("backend", ASSISTANT_BACKENDS)
def test_every_declared_backend_can_actually_be_run(backend: str) -> None:
    """Declared but not runnable is the failure this pairing prevents."""

    assert backend in _REQUIRED_CLI_OPTIONS
    assert backend in _ASSISTANT_AUTH_ENVIRONMENT


@pytest.mark.parametrize("backend", ASSISTANT_BACKENDS)
def test_every_declared_backend_can_be_configured(backend: str) -> None:
    _, namespace = _parse_cli_args(["assistant", "configure", "impl", backend])
    assert namespace.backend == backend


@pytest.mark.parametrize("backend", ASSISTANT_BACKENDS)
def test_every_declared_backend_is_offered_by_completion(backend: str) -> None:
    from zippergen.completion import completion_candidates

    assert backend in completion_candidates("assistant-backends", "")


def test_nothing_runnable_is_left_undeclared() -> None:
    """The other direction: a backend with an implementation but no entry."""

    assert set(_REQUIRED_CLI_OPTIONS) == set(ASSISTANT_BACKENDS)
    assert set(_ASSISTANT_AUTH_ENVIRONMENT) == set(ASSISTANT_BACKENDS)


# ---------------------------------------------------------------------------
# Declared means adapted, not merely accepted
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", ASSISTANT_BACKEND_SPECS, ids=lambda s: s.name)
def test_every_declared_backend_brings_its_own_adapter(spec) -> None:
    """Label, help command and argv are per-backend and must be declared.

    These were `if codex else claude` in three places, so a third backend
    could satisfy every name-based test and then be executed through the
    Claude branch -- with Claude's flags, Claude's permission model, and
    Claude's tool list.
    """

    assert spec.label and spec.label != spec.name
    assert spec.required_options
    assert spec.login_environment
    assert callable(spec.help_command)
    assert callable(spec.command)

    help_command = spec.help_command("/usr/bin/thing")
    assert help_command[0] == "/usr/bin/thing"


@pytest.mark.parametrize("spec", ASSISTANT_BACKEND_SPECS, ids=lambda s: s.name)
def test_each_backend_builds_a_command_only_it_would_build(spec, tmp_path) -> None:
    """No backend may silently produce another backend's argv."""

    from types import SimpleNamespace

    action = SimpleNamespace(
        access="read-only", shell="disabled", external_tools="none", name="a"
    )
    built = spec.command(f"/usr/bin/{spec.name}", tmp_path, action)

    assert built[0] == f"/usr/bin/{spec.name}"
    # Every option this backend is checked for must be one it actually uses.
    used = set(built)
    declared = set(spec.required_options)
    assert declared <= used | {opt for opt in declared if opt.startswith("--config")}

    others = [s for s in ASSISTANT_BACKEND_SPECS if s.name != spec.name]
    for other in others:
        theirs = set(other.required_options) - declared
        assert not (theirs & used), (
            f"{spec.name} builds a command carrying {other.name}'s options"
        )


def test_an_undeclared_backend_is_refused_rather_than_defaulted() -> None:
    """The failure mode: a name with no adapter running as some other backend."""

    assert assistant_backend_spec("gemini") is None
    assert assistant_backend_spec("") is None
    assert assistant_backend_spec(None) is None

    from zippergen.assistant_backends import check_cli_assistant

    checked = check_cli_assistant("gemini")
    assert not checked.supported
    assert "codex" in checked.detail and "claude" in checked.detail


def test_no_module_branches_on_a_backend_name() -> None:
    import pathlib
    import re

    names = [spec.name for spec in ASSISTANT_BACKEND_SPECS]
    branch = re.compile(
        r"(?:==|!=)\s*[\"'](?:" + "|".join(re.escape(n) for n in names) + r")[\"']"
    )
    source_root = pathlib.Path(__file__).resolve().parents[1] / "src" / "zippergen"
    offenders = sorted(
        path.name
        for path in source_root.rglob("*.py")
        if path.name != "assistant_backends.py" and branch.search(path.read_text())
    )
    assert not offenders, (
        "these modules branch on a backend name instead of reading its spec: "
        f"{offenders}"
    )
