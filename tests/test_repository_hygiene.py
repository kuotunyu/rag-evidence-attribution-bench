"""Release-critical repository tracking rules."""

from __future__ import annotations

from pathlib import Path


def test_manifest_v2_is_explicitly_allowed_by_gitignore() -> None:
    root = Path(__file__).resolve().parents[1]
    lines = (root / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "!data/manifests/split_manifest_v2.json" in lines
    assert "!data/manifests/challenge_manifest_v1.json" in lines
