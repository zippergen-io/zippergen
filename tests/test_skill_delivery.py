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


FAMILIES = {"provider", "model", "assistant", "connector"}


def _cli_commands() -> list[tuple[str, ...]]:
    """Every command a person can type, read from the real parser."""

    import argparse

    from zippergen.serve import HIDDEN_COMMANDS, _parse_cli_args

    parser, _arguments = _parse_cli_args([])

    def subcommands(node):
        for action in node._actions:
            if isinstance(action, argparse._SubParsersAction):
                return action.choices
        return {}

    found: list[tuple[str, ...]] = []

    def walk(node, path):
        for name, child in subcommands(node).items():
            if name in HIDDEN_COMMANDS:
                continue
            found.append((*path, name))
            walk(child, (*path, name))

    walk(parser, ())
    return found


def _documented(command: tuple[str, ...], text: str) -> bool:
    if " ".join(command) in text:
        return True
    # The four configuration families share one documented pattern rather than
    # repeating every verb four times, e.g. "TYPE rename OLD NEW".
    return (
        len(command) == 2
        and command[0] in FAMILIES
        and f"TYPE {command[1]}" in text
    )


@pytest.mark.parametrize(
    "documents",
    [
        ("docs/workflow-development-deployment-guide.tex",),
        (
            "src/zippergen/skills/zippergen-workflows/SKILL.md",
            "src/zippergen/skills/zippergen-workflows/references/dsl-and-cli.md",
        ),
    ],
    ids=["guide", "skill"],
)
def test_every_command_is_documented(documents):
    """Prose has no tests, so it drifts every time behaviour improves.

    Four times in one day a document went quietly wrong because a command
    changed. This does not check that the words are *right* -- nothing can --
    but it does catch the case where a command exists and nobody wrote it
    down at all.
    """

    text = "".join((REPO / name).read_text() for name in documents)
    missing = sorted(
        " ".join(command)
        for command in _cli_commands()
        if not _documented(command, text)
    )

    assert not missing, "undocumented commands: " + ", ".join(missing)


def test_the_skill_says_how_to_decide_what_a_user_configures(tmp_path):
    """An agent that only sees the mechanism will hard-code values.

    The reference shows `DeploymentField` with a secret token, which teaches
    how to declare one and nothing about when to. Without the judgment, a
    generated workflow gets an address written into it, and changing it later
    needs a code change and a redeploy.
    """

    text = (skill_directory() / "SKILL.md").read_text(encoding="utf-8")

    assert "Decide what the user configures" in text
    # The three questions, in order, and the case for leaving a value alone.
    assert "connector" in text and "DeploymentField" in text
    assert "would a second deployment of this same workflow plausibly answer" in text
    assert "it is not configuration" in text
    # One place for answers, so no second mechanism is invented.
    assert "[configuration]" in text
