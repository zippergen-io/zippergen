"""The manifest's shape is stated once and carried, not re-established.

`project_manifest` validated every section on the way in and then returned
`dict[str, object]`, discarding the guarantee. Its writer and each family
accessor rebuilt it with `assert isinstance` -- twenty-six of them -- so the
schema effectively lived in four places and a reshaped field had to be changed
in all of them or configuration would be dropped on the next write.
"""

import pathlib
import re

from zippergen.workspace import ProjectManifest, Workspace

import pytest


MALFORMED = [
    ("providers", "not-a-section"),
    ("providers", {"connections": "not-a-table"}),
    ("providers", {"connections": {"bot": "not-a-table"}}),
    ("models", {"configurations": "not-a-table"}),
    ("models", {"configurations": {"fast": "not-a-table"}}),
    ("models", {"assignments": "not-a-table"}),
    ("models", {"assignments": {"lifelines": "not-a-table"}}),
    ("assistants", {"configurations": {"impl": ["not", "a", "table"]}}),
    ("connectors", {"bindings": "not-a-table"}),
    ("connectors", {"assignments": {"actions": 7}}),
]


@pytest.mark.parametrize("section,value", MALFORMED)
def test_a_malformed_section_is_refused_not_cast(section, value) -> None:
    """The type must be a fact, not a claim.

    A cast once accepted `{"connections": "not-a-table"}` as `Providers` --
    telling contributors a malformed value was impossible while the writer
    could still meet one and fail later on an incidental attribute error.
    Every section now goes through the same decoder the file goes through.
    """

    from zippergen.workspace import (
        WorkspaceError,
        _decode_connectors,
        _decode_providers,
        _decode_routed,
        _section,
    )

    decoders = {
        "providers": _decode_providers,
        "models": lambda v: _decode_routed(v, field="models", default="mock"),
        "assistants": lambda v: _decode_routed(v, field="assistants", default=""),
        "connectors": _decode_connectors,
    }
    with pytest.raises(WorkspaceError):
        _section(value, {}, decode=decoders[section])


def test_a_caller_replacement_is_decoded_exactly_like_the_file(tmp_path) -> None:
    """One decoder, so 'validated' and 'typed' cannot drift apart."""

    from zippergen.workspace import _decode_routed

    written = {
        "configurations": {"fast": {"provider": "openai", "temperature": 0.2}},
        "assignments": {"default": "fast", "lifelines": {}, "actions": {}},
    }
    decoded = _decode_routed(written, field="models", default="mock")

    # Structure validated, and the typed literal preserved for the writer.
    assert decoded["assignments"]["default"] == "fast"
    assert decoded["configurations"]["fast"]["temperature"] == 0.2


def test_every_declared_section_is_actually_produced(tmp_path) -> None:
    """A key in the type with no value would fail only on a later write."""

    workspace = Workspace(str(tmp_path))
    workspace.initialize_project(name="shape")
    manifest = workspace.project_manifest()
    for key in ProjectManifest.__annotations__:
        assert key in manifest, f"the loader never produces {key!r}"


def test_defaults_and_loaded_manifests_have_the_same_keys(tmp_path) -> None:
    """The two return paths must not drift apart."""

    absent = Workspace(str(tmp_path / "nothing-here")).project_manifest()
    workspace = Workspace(str(tmp_path / "real"))
    workspace.initialize_project(name="shape")
    present = workspace.project_manifest()
    assert set(absent) == set(present)
    assert absent["exists"] is False
    assert present["exists"] is True


def test_a_round_trip_preserves_every_family(tmp_path) -> None:
    """What the writer emits is what the loader reads back."""

    workspace = Workspace(str(tmp_path))
    workspace.initialize_project(name="shape")
    workspace.write_configuration_values({"task": "fix the bug", "rounds": 4})
    workspace.save_provider_connection("bot", {"kind": "telegram"})

    reloaded = Workspace(str(tmp_path)).project_manifest()
    assert reloaded["configuration"]["task"] == "fix the bug"
    assert "bot" in reloaded["providers"]["connections"]
