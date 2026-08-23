"""`init` creates the few files an ordinary ZipperGen project needs.

It bootstraps files and stops. It asks nothing, configures nothing, and never
overwrites existing project guidance.
"""

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest


def _init(directory: Path, *arguments: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "zippergen.serve", "init", *arguments],
        capture_output=True,
        text=True,
        cwd=directory,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_it_creates_a_project_and_stops(tmp_path):
    output = _init(tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        ".zippergen",
        "AGENTS.md",
        "CLAUDE.md",
        "specification.md",
        "zippergen.toml",
    ]
    assert "created" in output


def test_the_manifest_is_valid_and_names_the_directory(tmp_path):
    project = tmp_path / "call-intake"
    project.mkdir()

    _init(project)
    manifest = tomllib.loads((project / "zippergen.toml").read_text())

    assert manifest["name"] == "call-intake"
    assert manifest["specification_file"] == "specification.md"
    # Everything in this file is a choice a person made. Identity and schema
    # bookkeeping are not choices, and are not written here.
    assert "project_id" not in manifest
    assert "schema_version" not in manifest


def test_an_explicit_name_overrides_the_directory(tmp_path):
    _init(tmp_path, "intake")

    assert tomllib.loads((tmp_path / "zippergen.toml").read_text())["name"] == "intake"


def test_the_generated_agents_md_points_at_the_packaged_skill(tmp_path):
    _init(tmp_path)
    content = (tmp_path / "AGENTS.md").read_text()

    assert "zippergen skill" in content
    # It is committed and shared, so it must not name one machine.
    assert "/Users/" not in content
    assert "/home/" not in content


def test_the_generated_claude_md_imports_the_shared_guidance(tmp_path):
    _init(tmp_path)

    assert (tmp_path / "CLAUDE.md").read_text() == "@AGENTS.md\n"


def test_an_existing_agents_md_is_never_overwritten(tmp_path):
    """Someone else's guidance is not ours to replace."""

    agents = tmp_path / "AGENTS.md"
    original = "# My project\n\nSome existing guidance.\n"
    agents.write_text(original)

    output = _init(tmp_path)

    assert agents.read_text() == original
    assert "left alone" in output
    # The user is told exactly what to add instead.
    assert "zippergen skill" in output


def test_an_existing_claude_md_is_never_overwritten(tmp_path):
    claude = tmp_path / "CLAUDE.md"
    original = "# Existing Claude guidance\n"
    claude.write_text(original)

    output = _init(tmp_path)

    assert claude.read_text() == original
    assert "left alone" in output
    assert "@AGENTS.md" in output


def test_an_existing_specification_is_never_overwritten(tmp_path):
    specification = tmp_path / "specification.md"
    specification.write_text("Already written by hand.\n")

    _init(tmp_path)

    assert specification.read_text() == "Already written by hand.\n"


def test_running_it_twice_changes_nothing(tmp_path):
    _init(tmp_path)
    before = {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
        if path.is_file()
    }

    output = _init(tmp_path)

    after = {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
        if path.is_file()
    }
    assert after == before
    assert "created" not in output


def test_it_can_create_a_project_elsewhere(tmp_path):
    target = tmp_path / "somewhere" / "deep"

    _init(tmp_path, "--directory", str(target))

    assert (target / "zippergen.toml").is_file()


def test_it_asks_nothing(tmp_path):
    """No questionnaire. It must finish with stdin closed."""

    result = subprocess.run(
        [sys.executable, "-m", "zippergen.serve", "init"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        stdin=subprocess.DEVNULL,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("filename", ["zippergen.lock", "workflow.py"])
def test_it_creates_no_workflow_or_lock(tmp_path, filename):
    """Configuration and the workflow are not init's business."""

    _init(tmp_path)

    assert not (tmp_path / filename).exists()


def test_it_mints_a_local_identity_that_ignores_itself(tmp_path):
    """The identity keys private state, so it must not travel with a clone.

    Creating a project is the one moment it is minted. Keeping it in an
    ignored local file, rather than in versioned configuration, is what makes
    "do not copy this value" unnecessary to say.
    """

    _init(tmp_path)

    identity = (tmp_path / ".zippergen" / "project-id").read_text().strip()
    assert len(identity) == 32
    assert (tmp_path / ".zippergen" / ".gitignore").read_text().strip() == "*"
    assert "project_id" not in (tmp_path / "zippergen.toml").read_text()
