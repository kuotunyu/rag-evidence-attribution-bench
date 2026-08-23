"""Exact detached Git checkouts fail closed on every dirty-file class."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from rag_evidence.annotation.source_verify import verify_checkout, verify_tracked_blob
from rag_evidence.errors import DataError


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


@pytest.fixture
def git_checkout(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "checkout"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.name", "Test Owner")
    _git(root, "config", "user.email", "owner@example.invalid")
    _git(root, "config", "core.autocrlf", "false")
    (root / ".gitignore").write_text(
        "src/overlay.py\n.env\n.venv/\n.pytest_cache/\nbuild/\n",
        encoding="utf-8",
    )
    (root / "tracked.txt").write_text("tracked source\n", encoding="utf-8")
    _git(root, "add", ".gitignore", "tracked.txt")
    _git(root, "commit", "-m", "fixture")
    commit = _git(root, "rev-parse", "HEAD")
    _git(root, "switch", "--detach", commit)
    return root, commit


@pytest.mark.parametrize(
    "poison",
    [
        "tracked-change",
        "staged-change",
        "untracked-file",
        "ignored-source-overlay",
        "ignored-build-config",
        "ignored-venv",
        "ignored-cache",
    ],
)
def test_checkout_poison_fails_closed(
    git_checkout: tuple[Path, str],
    poison: str,
) -> None:
    root, commit = git_checkout
    if poison == "tracked-change":
        (root / "tracked.txt").write_text("changed\n", encoding="utf-8")
    elif poison == "staged-change":
        (root / "tracked.txt").write_text("staged\n", encoding="utf-8")
        _git(root, "add", "tracked.txt")
    elif poison == "untracked-file":
        (root / "untracked.txt").write_text("poison\n", encoding="utf-8")
    else:
        relative = {
            "ignored-source-overlay": "src/overlay.py",
            "ignored-build-config": ".env",
            "ignored-venv": ".venv/marker.txt",
            "ignored-cache": ".pytest_cache/marker.txt",
        }[poison]
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ignored poison\n", encoding="utf-8")

    with pytest.raises(DataError, match="clean"):
        verify_checkout(root, commit, phase="pre-build", logical_checkout="fixture")


def test_post_build_ignored_file_is_rejected(git_checkout: tuple[Path, str]) -> None:
    root, commit = git_checkout
    verify_checkout(root, commit, phase="pre-build", logical_checkout="fixture")
    poison = root / "build/ignored.txt"
    poison.parent.mkdir()
    poison.write_text("created during build\n", encoding="utf-8")

    with pytest.raises(DataError, match="clean"):
        verify_checkout(root, commit, phase="post-build", logical_checkout="fixture")


def test_checkout_requires_exact_full_commit_and_detached_head(
    git_checkout: tuple[Path, str],
) -> None:
    root, commit = git_checkout
    verified = verify_checkout(root, commit, phase="pre-build", logical_checkout="fixture")

    assert verified.commit_sha == commit
    assert len(verified.tree_sha) == 40
    assert {record.phase for record in verified.commands} == {"pre-build"}
    assert all(record.argv[0] == "git" for record in verified.commands)
    with pytest.raises(DataError, match="40-character"):
        verify_checkout(root, "HEAD", phase="pre-build", logical_checkout="fixture")
    _git(root, "switch", "-")
    with pytest.raises(DataError, match="detached"):
        verify_checkout(root, commit, phase="pre-build", logical_checkout="fixture")


def test_tracked_blob_matches_worktree_and_expected_sha256(
    git_checkout: tuple[Path, str],
) -> None:
    root, commit = git_checkout
    checkout = verify_checkout(
        root, commit, phase="pre-build", logical_checkout="fixture"
    )
    expected = hashlib.sha256((root / "tracked.txt").read_bytes()).hexdigest()

    blob_sha = verify_tracked_blob(checkout, "tracked.txt", expected)

    assert len(blob_sha) == 40
    with pytest.raises(DataError, match="hash"):
        verify_tracked_blob(checkout, "tracked.txt", "0" * 64)
    with pytest.raises(DataError, match="tracked"):
        verify_tracked_blob(checkout, ".gitignore/../missing.txt", expected)
