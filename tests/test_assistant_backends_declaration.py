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
    ASSISTANT_BACKENDS,
    _ASSISTANT_AUTH_ENVIRONMENT,
    _REQUIRED_CLI_OPTIONS,
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
