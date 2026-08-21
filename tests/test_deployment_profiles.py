"""Reading a stored deployment profile, including older ones.

A profile is configuration, not durable recovery state, so it is carried
forward across schema changes rather than refused. The durable store keeps the
stricter rule for itself: control positions only mean something under the
program that wrote them.
"""

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


def test_a_schema_2_profile_is_carried_forward(home):
    """The upgrade path that had no way out.

    `zg deploy` is the command that writes a current profile, and it loads the
    existing one first. Refusing schema 2 meant the advice to redeploy could not
    be followed, and `deploy remove` keeps the profile, so nothing unblocked it.
    """

    _write_profile(home, "old", {
        "schema_version": 2,
        "name": "old",
        "inputs": {},
        "options": {"send_mode": "send", "certified": "a@b.c"},
    })

    profile = _load_deployment_profile("old")

    assert profile["schema_version"] == DEPLOYMENT_PROFILE_SCHEMA_VERSION
    assert profile["inputs"] == {}
    # The settings someone answered are the reason to migrate rather than start
    # again, so they must survive.
    assert profile["options"] == {"send_mode": "send", "certified": "a@b.c"}


def test_schema_2_inputs_keep_their_values_and_types(home):
    _write_profile(home, "old", {
        "schema_version": 2,
        "name": "old",
        "inputs": {"number": 3, "label": "x", "flag": True},
    })

    inputs = _load_deployment_profile("old")["inputs"]

    assert inputs == {"number": 3, "label": "x", "flag": True}
    assert isinstance(inputs["number"], int)
    assert isinstance(inputs["flag"], bool)


def test_reading_a_profile_does_not_rewrite_it(home):
    """Migration happens in memory; the next command that edits it writes it."""

    _write_profile(home, "old", {"schema_version": 2, "name": "old", "inputs": {}})
    path = home / "deployments" / "old.json"
    before = path.read_text()

    _load_deployment_profile("old")

    assert path.read_text() == before


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


def test_a_profile_too_old_to_carry_forward_says_what_to_do(home):
    _write_profile(home, "ancient", {"schema_version": 1, "name": "ancient", "inputs": {}})

    with pytest.raises(SystemExit, match="Remove the file"):
        _load_deployment_profile("ancient")


def test_a_profile_with_no_schema_is_refused(home):
    _write_profile(home, "bare", {"name": "bare", "inputs": {}})

    with pytest.raises(SystemExit, match="does not say which schema"):
        _load_deployment_profile("bare")
