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


def test_no_document_tells_a_reader_to_install_from_pypi_yet():
    """`pip install zippergen` fails: the package is not published.

    It is the first command a reader runs, so it must work. Delete this test
    when the name is on PyPI and the plain command is true again.
    """

    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sources = [root / "README.md", *(root / "docs").glob("*.tex")]
    sources += list((root / "docs").glob("*.md"))
    offenders = [
        path.relative_to(root)
        for path in sources
        if "pip install zippergen" in path.read_text(encoding="utf-8")
    ]

    assert not offenders, (
        "these tell a reader to install a package that is not published: "
        + ", ".join(str(path) for path in offenders)
    )
