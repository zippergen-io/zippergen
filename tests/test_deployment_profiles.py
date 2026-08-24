"""Reading a stored deployment profile through a strict version gate."""

import json

import pytest

from zippergen.deployment_profiles import (
    DEPLOYMENT_PROFILE_SCHEMA_VERSION,
    _load_deployment_profile,
)
from zippergen.value_codec import encode_value


def _write_profile(home, name: str, profile: dict) -> None:
    directory = home / "deployments"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(json.dumps(profile, indent=2))


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "zg-home"))
    return tmp_path / "zg-home"


def test_a_profile_with_an_older_schema_is_refused(home):
    _write_profile(home, "old", {
        "schema_version": DEPLOYMENT_PROFILE_SCHEMA_VERSION - 1,
        "name": "old",
        "inputs": {},
    })

    with pytest.raises(SystemExit, match="No migration is available"):
        _load_deployment_profile("old")


def test_a_current_profile_is_left_alone(home):
    _write_profile(home, "now", {
        "schema_version": DEPLOYMENT_PROFILE_SCHEMA_VERSION,
        "name": "now",
        "inputs": encode_value({"number": 3}),
    })

    profile = _load_deployment_profile("now")

    assert profile["inputs"] == {"number": 3}


def test_a_newer_profile_is_refused_and_says_why(home):
    _write_profile(home, "future", {
        "schema_version": DEPLOYMENT_PROFILE_SCHEMA_VERSION + 1,
        "name": "future",
        "inputs": {},
    })

    with pytest.raises(SystemExit, match="written by a newer ZipperGen"):
        _load_deployment_profile("future")


def test_a_profile_too_old_is_refused(home):
    _write_profile(home, "ancient", {"schema_version": 1, "name": "ancient", "inputs": {}})

    with pytest.raises(SystemExit, match="No migration is available"):
        _load_deployment_profile("ancient")


def test_a_profile_with_no_schema_is_refused(home):
    _write_profile(home, "bare", {"name": "bare", "inputs": {}})

    with pytest.raises(SystemExit, match="does not say which schema"):
        _load_deployment_profile("bare")
