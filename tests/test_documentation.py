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
