"""The agent-facing layer must not be built on top of Studio.

Studio is dead architecture. It stays only until it can be deleted, and
nothing new may depend on it — otherwise the replacement inherits exactly the
complexity it exists to remove:

    new layer -> wrappers -> Studio -> ZipperGen      (what we must not build)
    new layer -> ZipperGen core                       (what we are building)

Reusing Studio's *code* is fine and expected, but by extracting it downward
into an ordinary module both callers can use — not by importing Studio.

Add a module to AGENT_LAYER when it becomes part of that layer.
"""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "zippergen"

# Modules that make up the agent-facing replacement. Each must be usable with
# Studio deleted.
AGENT_LAYER = ("skill.py",)

# Modules on their way out. The fence is a denylist of dead architecture, not
# an allow-list of approved core: core grows all the time and an allow-list
# would turn every reasonable new module into a test failure.
DEAD_ARCHITECTURE = (
    "zippergen.studio",
    "zippergen.natural_language",
)


def _imported_modules(path: Path) -> set[str]:
    """Every module named by an import, including inside functions."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("module", AGENT_LAYER)
def test_the_agent_layer_does_not_import_studio(module):
    path = PACKAGE / module
    assert path.is_file(), f"{module} is listed in AGENT_LAYER but does not exist"

    offending = sorted(
        name
        for name in _imported_modules(path)
        if name.startswith(DEAD_ARCHITECTURE)
    )

    assert not offending, (
        f"{module} imports {', '.join(offending)}. If the logic it needs "
        "exists only inside Studio, extract that logic into an ordinary "
        "module and have both callers use it."
    )


def test_the_scripted_backend_is_independent_of_studio():
    """It sits beside the real and mock backends, not above Studio."""

    assert not any(
        name.startswith(DEAD_ARCHITECTURE)
        for name in _imported_modules(PACKAGE / "backends.py")
    )
