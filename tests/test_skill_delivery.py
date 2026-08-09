"""The coding-agent skill must reach an installed user, not only a checkout."""

import subprocess
import sys
import shlex
import tomllib
from pathlib import Path

import pytest

from zippergen.skill import (
    SKILL_NAME,
    Skill,
    SkillNotFound,
    agents_md,
    load_skill,
    skill_directory,
)
from zippergen.serve import _parse_cli_args

REPO = Path(__file__).resolve().parents[1]
CHECKOUT_SKILL = REPO / ".agents" / "skills" / SKILL_NAME


def test_the_skill_ships_inside_the_package():
    """It must sit under the package, or a wheel will not carry it."""

    directory = skill_directory()

    assert directory.is_relative_to(Path(sys.modules["zippergen"].__file__).parent)
    assert (directory / "SKILL.md").is_file()


def test_packaging_declares_the_skill_files():
    """package-data must name them, or setuptools omits non-Python files."""

    config = tomllib.loads((REPO / "pyproject.toml").read_text())
    patterns = config["tool"]["setuptools"]["package-data"]["zippergen"]

    assert any("skills/" in pattern and pattern.endswith(".md") for pattern in patterns)


def test_the_checkout_copy_matches_the_packaged_one():
    """Codex discovers `.agents/skills/`; the package is what ships.

    Two copies exist for two different discovery paths, so this asserts they
    cannot drift apart.
    """

    packaged = skill_directory()
    for path in sorted(packaged.rglob("*")):
        if not path.is_file():
            continue
        mirrored = CHECKOUT_SKILL / path.relative_to(packaged)
        assert mirrored.is_file(), f"{mirrored} is missing from the checkout copy"
        assert mirrored.read_bytes() == path.read_bytes(), f"{mirrored} has drifted"


def test_loading_returns_the_body_and_its_references():
    skill = load_skill()

    assert isinstance(skill, Skill)
    assert "# ZipperGen Workflows" in skill.body
    assert [name for name, _text in skill.references] == ["dsl-and-cli"]


def test_rendering_can_omit_the_references():
    skill = load_skill()

    assert "# Reference: dsl-and-cli" in skill.render()
    assert "# Reference:" not in skill.render(include_references=False)


def test_a_missing_skill_says_the_install_is_incomplete():
    with pytest.raises(SkillNotFound, match="Reinstall zippergen"):
        skill_directory("no-such-skill")


def test_the_generated_agents_md_carries_no_absolute_path():
    """It is committed and shared, so it must not name one machine."""

    content = agents_md("diagnosis")

    assert "zippergen skill" in content
    assert "/Users/" not in content
    assert "/home/" not in content
    assert str(REPO) not in content


def test_the_command_prints_the_skill_and_the_agents_md(tmp_path):
    def run(*arguments):
        result = subprocess.run(
            [sys.executable, "-m", "zippergen.serve", *arguments],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout

    assert "# ZipperGen Workflows" in run("skill")
    assert "# Reference: dsl-and-cli" not in run("skill", "--no-references")
    assert "zippergen skill" in run("skill", "--agents-md", "--project", "demo")


def test_every_standalone_skill_command_matches_the_real_parser():
    """The skill is the user procedure, so its commands cannot drift."""

    skill = load_skill()
    documents = [skill.body, *(text for _name, text in skill.references)]
    commands: list[str] = []
    for document in documents:
        for raw_line in document.splitlines():
            line = raw_line.strip()
            if "| zg " in line:
                line = "zg " + line.split("| zg ", 1)[1]
            if line.startswith(("zg ", "zippergen ")):
                commands.append(line)

    assert commands
    for command in commands:
        tokens = shlex.split(command, comments=True)
        _parser, arguments = _parse_cli_args(tokens[1:])
        assert arguments.cmd, command
