"""Committed pilot artifacts remain blind and contain no scientific decisions."""

from __future__ import annotations

from pathlib import Path

from rag_evidence.annotation.package import scan_clean_package


def test_manifest_v2_is_explicitly_allowed_by_gitignore(repo_root: Path) -> None:
    lines = (repo_root / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "!data/manifests/split_manifest_v2.json" in lines
    assert "!data/manifests/challenge_manifest_v1.json" in lines


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
