"""Published PDFs identify the exact source from which they were built."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ("first-workflow", "workflow-development-deployment-guide")


@pytest.mark.parametrize("name", DOCUMENTS)
def test_committed_pdf_matches_its_source_stamp(name: str) -> None:
    source = ROOT / "docs" / f"{name}.tex"
    pdf = ROOT / "docs" / f"{name}.pdf"
    stamp = ROOT / "docs" / f"{name}.pdf.source.sha256"

    assert pdf.is_file(), f"published PDF is missing: {pdf}"
    assert stamp.is_file(), f"PDF source stamp is missing: {stamp}"
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    assert stamp.read_text().strip() == actual, (
        f"{source} changed without rebuilding {pdf}; run make docs"
    )
