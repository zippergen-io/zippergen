"""The agent-facing layer depends directly on ordinary ZipperGen modules.

The removed interactive application must not return as an intermediate layer.
Add a module to AGENT_LAYER when it becomes part of the coding-agent surface.
"""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "zippergen"

# Modules that make up the agent-facing integration.
AGENT_LAYER = ("skill.py",)

# Removed modules that must not be reintroduced as dependencies.
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
def test_the_agent_layer_does_not_import_removed_application_modules(module):
    path = PACKAGE / module
    assert path.is_file(), f"{module} is listed in AGENT_LAYER but does not exist"

    offending = sorted(
        name
        for name in _imported_modules(path)
        if name.startswith(DEAD_ARCHITECTURE)
    )

    assert not offending, (
        f"{module} imports removed application code: {', '.join(offending)}. "
        "Put shared behavior in an ordinary core module."
    )


def test_the_scripted_backend_is_an_ordinary_runtime_backend():
    """It sits beside the real and mock backends."""

    assert not any(
        name.startswith(DEAD_ARCHITECTURE)
        for name in _imported_modules(PACKAGE / "backends.py")
    )
