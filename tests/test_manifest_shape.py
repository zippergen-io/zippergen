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


def test_no_module_re_establishes_the_shape_with_assertions() -> None:
    source_root = pathlib.Path(__file__).resolve().parents[1] / "src" / "zippergen"
    # Only assertions about the manifest. Reports and decoded JSON are
    # genuinely untyped at their boundaries and may still be checked.
    guard = re.compile(
        r"project_manifest\(\)[^\n]*\n(?:[^\n]*\n){0,3}?[^\n]*assert isinstance"
    )
    offenders = sorted(
        path.name for path in source_root.rglob("*.py") if guard.search(path.read_text())
    )
    assert not offenders, (
        "these modules re-assert a shape the manifest type already carries: "
        f"{offenders}"
    )


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
