"""Deliver the coding-agent skill from the installed package.

A ZipperGen project is an ordinary directory. Nothing in it tells a coding
agent how ZipperGen works, and until now the only delivery path was the
instruction text the deleted shell generated when it invoked an assistant — which
required a git checkout and disappears with the shell.

This module makes the skill reachable from the installed package instead, so
an ordinary install is enough:

    zippergen skill              print it
    zippergen skill --agents-md  print an AGENTS.md that points at it

The `AGENTS.md` deliberately carries no absolute paths. It is committed and
shared, so it must mean the same thing on a colleague's machine and on a
server.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SKILL_NAME = "zippergen-workflows"

AGENTS_MD_TEMPLATE = """# {project}

A ZipperGen project. ZipperGen is a Python DSL for multi-agent coordination:
the workflow declares which participants exchange what, in which order, and
its coordination properties are machine-checkable.

Before editing workflow code, run

    zippergen skill

and follow it completely.

| file | holds |
|---|---|
| `specification.md` | what the workflow is meant to do |
| `{workflow}` | the implementation |
| `zippergen.toml` | project configuration |

Do not deploy or start a service unless you are asked to.
"""

# Claude Code discovers CLAUDE.md, while Codex discovers AGENTS.md. Keep the
# actual project instructions in one file and make Claude import that file.
CLAUDE_MD_TEMPLATE = "@AGENTS.md\n"


class SkillNotFound(RuntimeError):
    """The packaged skill is missing, so the install is incomplete."""


@dataclass(frozen=True)
class Skill:
    """The skill and the references it links, read from the package."""

    name: str
    body: str
    references: tuple[tuple[str, str], ...]

    def render(self, *, include_references: bool = True) -> str:
        parts = [self.body.rstrip()]
        if include_references:
            for title, text in self.references:
                parts.append(f"\n\n---\n\n# Reference: {title}\n\n{text.rstrip()}")
        return "\n".join(parts) + "\n"


def skill_directory(name: str = SKILL_NAME) -> Path:
    """Return the packaged skill directory, or say the install is incomplete."""

    directory = Path(__file__).resolve().parent / "skills" / name
    if not (directory / "SKILL.md").is_file():
        raise SkillNotFound(
            f"The packaged skill {name!r} is missing from "
            f"{directory}. Reinstall zippergen, or run from a checkout."
        )
    return directory


def load_skill(name: str = SKILL_NAME) -> Skill:
    """Read the skill and its reference files from the package."""

    directory = skill_directory(name)
    body = (directory / "SKILL.md").read_text(encoding="utf-8")
    references: list[tuple[str, str]] = []
    reference_directory = directory / "references"
    if reference_directory.is_dir():
        for path in sorted(reference_directory.glob("*.md")):
            references.append((path.stem, path.read_text(encoding="utf-8")))
    return Skill(name=name, body=body, references=tuple(references))


def agents_md(project: str, workflow: str = "workflow.py") -> str:
    """Return an AGENTS.md that points at the skill without an absolute path."""

    return AGENTS_MD_TEMPLATE.format(project=project, workflow=workflow)


def claude_md() -> str:
    """Point Claude Code at the same committed guidance Codex reads."""

    return CLAUDE_MD_TEMPLATE
