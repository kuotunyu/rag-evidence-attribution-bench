"""Public-facing metadata must stay aligned with the repository state."""

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRERELEASE_VERSION = "0.2.0" + ".dev0"


def test_public_docs_do_not_reference_internal_handoff_files() -> None:
    public_files = (
        REPO_ROOT / "README.md",
        REPO_ROOT / "README_en.md",
        REPO_ROOT / "docs" / "RERANKING_RUNBOOK.md",
        REPO_ROOT / "scripts" / "make_colab_bundle.py",
    )
    forbidden = (
        "TRANSFER.md",
        "PROGRESS.md",
        "has no GitHub remote yet",
        "it has never executed on GitHub",
    )

    for path in public_files:
        text = path.read_text(encoding="utf-8")
        for stale_text in forbidden:
            assert stale_text not in text, f"{path.name} contains stale text: {stale_text}"

    assert not (REPO_ROOT / "TRANSFER.md").exists()


def test_readmes_publish_stable_infrastructure_identity_and_historical_baseline() -> None:
    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["version"] == "0.2.0"
    historical_url = (
        "https://github.com/kuotunyu/rag-evidence-attribution-bench/releases/tag/v0.1.0"
    )

    for readme_name in ("README.md", "README_en.md"):
        readme = (REPO_ROOT / readme_name).read_text(encoding="utf-8")
        first_screen = "\n".join(readme.splitlines()[:45]).casefold()
        assert "v0.2.0" in first_screen
        assert "annotation infrastructure" in first_screen
        assert "not_conducted" in first_screen
        assert "independent-human pilot was not conducted" in first_screen
        assert "synthetic" in first_screen
        assert "not human evidence" in first_screen
        assert historical_url in readme


def test_v020_release_notes_enforce_claim_boundaries() -> None:
    release_notes = (REPO_ROOT / "docs/RELEASE_NOTES_V0.2.0.md").read_text(encoding="utf-8")
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    combined = f"{release_notes}\n{changelog}".casefold()

    for required in (
        "annotation infrastructure",
        "human pilot: not_conducted",
        "no human iaa",
        "confirmatory protocol: draft / not_authorized",
        "synthetic rehearsal is not human evidence",
        "windows and linux",
        "no sandbox or prompt-injection certification",
    ):
        assert required in combined

    assert PRERELEASE_VERSION not in release_notes


def test_v020_operational_docs_preserve_protocol_and_mark_pilot_not_conducted() -> None:
    operational_files = (
        REPO_ROOT / "PILOT_PROTOCOL.md",
        REPO_ROOT / "pilot" / "v0.2" / "README.md",
        REPO_ROOT / "pilot" / "v0.2" / "COMPLETION_CHECKLIST.md",
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in operational_files)

    assert "pilot-v0.2.2-draft" in combined
    assert "372096b" in combined
    assert "obsolete" in combined.casefold()
    assert "rag-evidence annotation collect" in combined
    assert "rag-evidence annotation adjudicate" in combined
    assert "rag-evidence annotation finalize-pilot" in combined
    assert "both Cohen's kappa and nominal Krippendorff's alpha" in combined
    assert "HUMAN_PILOT_NOT_CONDUCTED" in combined
    assert PRERELEASE_VERSION not in combined
