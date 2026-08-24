"""The annotation bootstrap lock is minimal, pinned, hashed, and reproducible."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_evidence.annotation.dependency_lock import (
    build_annotation_lock,
    validate_annotation_lock,
)
from rag_evidence.errors import DataError

LOCK_RELATIVE = Path("pilot/v0.2/annotation-requirements-py311.lock")


@pytest.mark.parametrize(
    "bad_line",
    [
        "-e .",
        ".",
        "../local-project",
        "git+https://example.invalid/repo.git",
        "fastapi>=0.111",
        "fastapi==0.116.1",
        "rag-evidence-attribution-bench==0.2.0 \\",
        "--extra-index-url https://user:secret@example.invalid/simple",
        "unsafe @ file:///tmp/unsafe.whl",
    ],
)
def test_lock_validator_rejects_non_third_party_or_unlocked_entries(
    repo_root: Path,
    bad_line: str,
) -> None:
    valid = (repo_root / LOCK_RELATIVE).read_text(encoding="utf-8")

    with pytest.raises(DataError):
        validate_annotation_lock(valid + "\n" + bad_line + "\n")


def test_committed_lock_is_reproducible_and_structurally_valid(repo_root: Path) -> None:
    committed = (repo_root / LOCK_RELATIVE).read_bytes()

    validate_annotation_lock(committed.decode("utf-8"))
    assert build_annotation_lock(repo_root) == committed


def test_lock_and_runbook_preserve_exact_binary_only_bootstrap(repo_root: Path) -> None:
    lock_text = (repo_root / LOCK_RELATIVE).read_text(encoding="utf-8").casefold()
    runbook = (repo_root / "pilot/v0.2/HANDOFF_RUNBOOK.md").read_text(encoding="utf-8")

    for forbidden in (
        "rag-evidence-attribution-bench",
        "torch==",
        "transformers==",
        "datasets==",
        "gradio==",
        "-e ",
        "git+",
        "file:",
    ):
        assert forbidden not in lock_text
    assert "--require-hashes" in runbook
    assert "--only-binary=:all:" in runbook
    assert "pip install --no-deps" in runbook
    assert "pip install --upgrade pip" not in runbook
