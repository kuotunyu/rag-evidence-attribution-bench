"""Synthetic rehearsal exercises the full local path without touching formal artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rag_evidence.annotation.rehearsal import run_synthetic_rehearsal
from rag_evidence.errors import DataError


def _formal_hashes(repo_root: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((repo_root / "pilot" / "v0.2" / "packages").glob("*.json"))
    }


@pytest.mark.parametrize(
    "relative",
    [Path("pilot/v0.2/synthetic-rehearsal"), Path("results/v2/synthetic-rehearsal")],
)
def test_rehearsal_refuses_formal_or_results_output_paths(
    repo_root: Path,
    relative: Path,
) -> None:
    with pytest.raises(DataError, match=r"outside the repository|formal"):
        run_synthetic_rehearsal(repo_root / relative, repo_root)


def test_rehearsal_covers_full_path_twice_without_changing_formal_packages(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    before = _formal_hashes(repo_root)

    result = run_synthetic_rehearsal(tmp_path / "synthetic-rehearsal", repo_root)

    assert result.tasks == 40
    assert result.amendments == 1
    assert result.disagreements >= 2
    assert result.adjudications == result.disagreements
    assert result.dataset_defect_exclusions == 1
    assert result.repeat_byte_identical is True
    assert result.verdict == "READY_FOR_HUMAN_FREEZE_REVIEW"
    assert _formal_hashes(repo_root) == before
    for run_name in ("run-1", "run-2"):
        run_root = tmp_path / "synthetic-rehearsal" / run_name
        assert (run_root / "SYNTHETIC-NOT-HUMAN-DATA.txt").exists()
        assert (
            json.loads((run_root / "final" / "privacy-scan.json").read_text(encoding="utf-8"))[
                "passed"
            ]
            is True
        )
        questions = [
            task["question"]
            for package in json.loads(
                (run_root / "inputs" / "manifest.json").read_text(encoding="utf-8")
            )["packages"]
            for task in package["tasks"]
        ]
        assert questions
        assert all("invented Lumen archive" in question for question in questions)


def test_rehearsal_refuses_nonempty_output_directory(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    out = tmp_path / "synthetic-rehearsal"
    out.mkdir()
    (out / "unrelated.txt").write_text("preserve", encoding="utf-8")

    with pytest.raises(DataError, match="absent or empty"):
        run_synthetic_rehearsal(out, repo_root)

    assert (out / "unrelated.txt").read_text(encoding="utf-8") == "preserve"
