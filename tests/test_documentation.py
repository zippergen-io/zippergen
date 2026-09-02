import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_documented_python_examples_exist():
    documents = (
        ROOT / "README.md",
        ROOT / "docs" / "first-workflow.tex",
        ROOT / "docs" / "workflow-development-deployment-guide.tex",
    )
    missing: list[str] = []

    for document in documents:
        text = document.read_text().replace(r"\_", "_")
        references = set(re.findall(r"examples/[A-Za-z0-9_./-]+\.py", text))
        missing.extend(
            f"{document.relative_to(ROOT)}: {reference}"
            for reference in sorted(references)
            if not (ROOT / reference).is_file()
        )

    assert not missing, "documented examples do not exist:\n" + "\n".join(missing)


def test_public_install_instructions_use_pypi():
    """Public instructions must not drift back to a mutable Git branch."""

    root = Path(__file__).resolve().parents[1]
    searched = [
        *root.glob("*.md"),
        *(root / "docs").rglob("*.md"),
        *(root / "docs").rglob("*.tex"),
        *(root / "src").rglob("*.py"),
        *(root / "src").rglob("*.md"),
        *(root / ".agents").rglob("*.md"),
    ]
    offenders = [
        str(path.relative_to(root))
        for path in searched
        if "git+https://github.com/zippergen-io/zippergen.git" in path.read_text(
            encoding="utf-8"
        )
    ]

    assert not offenders, (
        "these still tell readers to install from the Git branch: "
        + ", ".join(offenders)
    )
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "uv tool install zippergen" in readme
    assert "pipx install zippergen" in readme
    assert ".venv/bin/pip install zippergen" in readme
