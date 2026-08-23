"""Every source that can enter a human kit passes the static privacy boundary."""

from __future__ import annotations

from pathlib import Path

from rag_evidence.annotation.privacy import scan_delivery_sources


def test_all_human_delivery_sources_pass_static_negative_scan(repo_root: Path) -> None:
    assert scan_delivery_sources(repo_root) == ()


def test_public_pilot_claim_boundary_is_explicit(repo_root: Path) -> None:
    paths = (
        repo_root / "README.md",
        repo_root / "README_en.md",
        repo_root / "PILOT_PROTOCOL.md",
        repo_root / "pilot/v0.2/README.md",
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths).casefold()

    assert "explicit coordinator metadata" in text
    assert "transformation identity" in text
    assert "expected-answerability" in text
    assert "does not establish semantic unlinkability" in text
    assert "does not establish independent sibling perception" in text
    assert "tooling and instruction feasibility" in text


def test_public_status_does_not_claim_human_pilot_started(repo_root: Path) -> None:
    text = "\n".join(
        (repo_root / name).read_text(encoding="utf-8")
        for name in ("README.md", "README_en.md", "pilot/v0.2/README.md")
    ).casefold()

    assert "human_pilot_not_started" in text
    assert "pilot_ready_for_separate_authorization" not in text
