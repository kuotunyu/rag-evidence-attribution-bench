"""Committed pilot artifacts remain blind and contain no scientific decisions."""

from __future__ import annotations

import subprocess
from pathlib import Path

from rag_evidence.annotation.package import scan_clean_package


def test_manifest_v2_is_explicitly_allowed_by_gitignore(repo_root: Path) -> None:
    lines = (repo_root / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "!data/manifests/split_manifest_v2.json" in lines
    assert "!data/manifests/challenge_manifest_v1.json" in lines


def test_private_annotation_outputs_are_explicitly_gitignored(repo_root: Path) -> None:
    lines = set((repo_root / ".gitignore").read_text(encoding="utf-8").splitlines())

    assert {
        "*.whl",
        "*.tar.gz",
        "**/handoff-receipt.json",
        "**/SHA256SUMS",
        "**/submissions.jsonl",
        "**/amendments.jsonl",
        "**/adjudications.jsonl",
        "**/eligibility.json",
        "**/iaa.json",
        "**/annotation-state/",
        "**/coordinator-private/",
        "**/synthetic-rehearsal/",
        "pilot/v0.2/delivery*/",
    } <= lines


def test_committed_pilot_package_is_clean(repo_root: Path) -> None:
    package_root = repo_root / "pilot" / "v0.2" / "packages"
    assert package_root.exists()
    assert scan_clean_package(package_root) == {
        "files": 3,
        "packages": 2,
        "tasks": 40,
        "decisions": 0,
    }
    forbidden = ("submission", "amendment", "adjudication", "eligibility")
    assert not [
        path for path in package_root.rglob("*") if any(term in path.name for term in forbidden)
    ]


def test_generated_handoff_and_decision_artifacts_are_not_tracked(repo_root: Path) -> None:
    tracked = set(
        subprocess.check_output(
            ["git", "ls-files"],
            cwd=repo_root,
            text=True,
        ).splitlines()
    )
    forbidden_suffixes = (".whl", ".tar.gz", ".zip")
    forbidden_names = {"handoff-receipt.json", "SHA256SUMS"}

    assert not [
        path
        for path in tracked
        if path.endswith(forbidden_suffixes) or Path(path).name in forbidden_names
    ]
