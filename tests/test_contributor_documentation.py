"""What a contributor needs must be in the repository they clone.

The architecture guide and the statement of which constructs each theorem
covers lived only in `CLAUDE.md`, which is gitignored. Corrections made there
were true on one machine and absent from every clone.
"""

import pathlib
import subprocess

import pytest


REPO = pathlib.Path(__file__).resolve().parents[1]
GUIDE = REPO / "docs" / "architecture.md"


def _is_tracked(path: pathlib.Path) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path.relative_to(REPO))],
        cwd=REPO,
        capture_output=True,
    )
    return result.returncode == 0


@pytest.mark.skipif(
    not (REPO / ".git").exists(), reason="not a git checkout"
)
def test_the_architecture_guide_ships() -> None:
    assert GUIDE.is_file()
    assert _is_tracked(GUIDE), (
        "docs/architecture.md is not tracked, so a fresh clone would not "
        "receive the module boundaries or the theorem scope"
    )


def test_the_guide_states_what_each_theorem_covers() -> None:
    """The over-claim this document exists to prevent."""

    text = GUIDE.read_text()
    assert "CoregionStmt" in text
    assert "future work" in text
    assert "ISoLA" in text and "EXPRESS/SOS" in text


def test_the_guide_states_the_module_boundary() -> None:
    """The rule the CLI refactor was justified by."""

    text = GUIDE.read_text()
    assert "serve.py" in text
    assert "argument parsing and dispatch" in text


@pytest.mark.skipif(
    not (REPO / ".git").exists(), reason="not a git checkout"
)
def test_no_untracked_file_is_the_only_home_for_contributor_guidance() -> None:
    """`CLAUDE.md` may point at the guide; it may not be the guide."""

    local = REPO / "CLAUDE.md"
    if not local.is_file() or _is_tracked(local):
        return
    text = local.read_text()
    assert "docs/architecture.md" in text, (
        "the untracked CLAUDE.md must point at the tracked guide"
    )
    assert len(text.splitlines()) < 60, (
        "the untracked CLAUDE.md has grown into a second architecture guide; "
        "contributors who clone this repository would not receive it"
    )
