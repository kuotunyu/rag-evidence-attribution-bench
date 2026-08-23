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
    assert result.schema_versions
    assert all(version.endswith("-v2") for version in result.schema_versions)
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
            for task in json.loads(
                (run_root / "inputs/assignment-manifest-v2.json").read_text(encoding="utf-8")
            )["tasks"]
        ]
        assert questions
        assert all("invented Lumen archive" in question for question in questions)
        human_files = (
            list((run_root / "inputs").glob("ann-pilot-*.json"))
            + list((run_root / "inputs").glob("submission-*.jsonl"))
            + list((run_root / "inputs").glob("amendment-*.jsonl"))
            + [run_root / "adjudications.jsonl"]
        )
        human_text = "\n".join(path.read_text(encoding="utf-8") for path in human_files)
        assert '"blinded_parent_group"' not in human_text
        assert '"internal_group_id"' not in human_text
        assert '-v1"' not in human_text


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
