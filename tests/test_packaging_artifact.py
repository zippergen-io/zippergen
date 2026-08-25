"""The next wheel must contain exactly this package, and nothing left over.

setuptools copies sources into ``build/lib`` and never removes what is no
longer there. A module deleted from ``src/`` therefore survives in that tree
and is packaged into the next wheel -- shipping code the tests no longer cover
and the repository no longer has. The tree is gitignored, so nothing in review
or CI can see it: the only place the trap is armed is the machine that builds
the release.

This test is that machine's check. It is instant, and it fails exactly when a
build would be wrong.
"""

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_LIB = PROJECT_ROOT / "build" / "lib"
PACKAGE = PROJECT_ROOT / "src" / "zippergen"


def _modules(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    return {str(path.relative_to(root)) for path in root.rglob("*.py")}


def test_no_stale_module_is_waiting_to_be_packaged() -> None:
    staged = _modules(BUILD_LIB / "zippergen")
    if not staged:
        pytest.skip("no build tree on this machine; nothing can be stale")
    stale = sorted(staged - _modules(PACKAGE))
    assert not stale, (
        "build/lib holds module(s) that src/ no longer has: "
        + ", ".join(stale)
        + ". The next wheel would ship them. Delete build/ and rebuild."
    )
