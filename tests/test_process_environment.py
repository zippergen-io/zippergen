"""One overlay contract, used by runs, checks, and foreground commands.

Three near-identical context managers existed, and only the CLI's coerced its
values with ``str``. A readiness check could therefore observe a different
environment from the run it was checking.
"""

import os

from zippergen.process_environment import temporary_environment

import pytest


def test_a_value_is_visible_inside_and_gone_after(monkeypatch) -> None:
    monkeypatch.delenv("ZIPPERGEN_TEST_OVERLAY", raising=False)
    with temporary_environment({"ZIPPERGEN_TEST_OVERLAY": "on"}):
        assert os.environ["ZIPPERGEN_TEST_OVERLAY"] == "on"
    assert "ZIPPERGEN_TEST_OVERLAY" not in os.environ


def test_an_existing_value_is_restored_exactly(monkeypatch) -> None:
    monkeypatch.setenv("ZIPPERGEN_TEST_OVERLAY", "original")
    with temporary_environment({"ZIPPERGEN_TEST_OVERLAY": "replaced"}):
        assert os.environ["ZIPPERGEN_TEST_OVERLAY"] == "replaced"
    assert os.environ["ZIPPERGEN_TEST_OVERLAY"] == "original"


def test_restoration_survives_an_exception(monkeypatch) -> None:
    monkeypatch.setenv("ZIPPERGEN_TEST_OVERLAY", "original")
    with pytest.raises(RuntimeError):
        with temporary_environment({"ZIPPERGEN_TEST_OVERLAY": "replaced"}):
            raise RuntimeError("boom")
    assert os.environ["ZIPPERGEN_TEST_OVERLAY"] == "original"


def test_non_string_values_are_coerced_the_same_way_for_every_caller(
    monkeypatch,
) -> None:
    """The divergence that made a check disagree with the run it checked."""

    monkeypatch.delenv("ZIPPERGEN_TEST_OVERLAY", raising=False)
    with temporary_environment({"ZIPPERGEN_TEST_OVERLAY": 8080}):
        assert os.environ["ZIPPERGEN_TEST_OVERLAY"] == "8080"


def test_no_module_keeps_its_own_copy_of_the_overlay() -> None:
    import pathlib
    import re

    source_root = pathlib.Path(__file__).resolve().parents[1] / "src" / "zippergen"
    own = re.compile(r"def _temporary_environment\(")
    offenders = [
        path.name for path in source_root.rglob("*.py") if own.search(path.read_text())
    ]
    assert not offenders, (
        "these modules keep a private overlay instead of using "
        f"process_environment.temporary_environment: {offenders}"
    )
