"""Privacy scan rejects source metadata, decisions, PII, paths, and internal IDs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_evidence.annotation.package import scan_clean_package
from rag_evidence.errors import DataError


@pytest.mark.parametrize(
    "poison",
    [
        {"expected_answerability": "unanswerable"},
        {"transformation": "missing_hop"},
        {"answerability": "answerable"},
        {"adjudication_id": "adj-0123456789abcdef01234567"},
        {"contact": "person@example.com"},
        {"path": r"C:\\Users\\private\\repo"},
        {"passage": "ch-0123456789abcdef01234567-p00"},
    ],
)
def test_clean_export_scanner_rejects_poisoned_artifact(
    tmp_path: Path, poison: dict[str, object]
) -> None:
    (tmp_path / "poison.json").write_text(json.dumps(poison), encoding="utf-8")

    with pytest.raises(DataError, match="clean-package privacy scan failed"):
        scan_clean_package(tmp_path)


def test_clean_export_scanner_rejects_decision_jsonl(tmp_path: Path) -> None:
    (tmp_path / "submissions.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(DataError, match="decision artifact"):
        scan_clean_package(tmp_path)
