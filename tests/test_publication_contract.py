"""Public-facing metadata must stay aligned with the repository state."""

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_readmes_link_to_the_versioned_baseline_release() -> None:
    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = metadata["project"]["version"]
    release_url = (
        f"https://github.com/kuotunyu/rag-evidence-attribution-bench/releases/tag/v{version}"
    )

    for readme_name in ("README.md", "README_en.md"):
        readme = (REPO_ROOT / readme_name).read_text(encoding="utf-8")
        assert release_url in readme


def test_b01_docs_mark_old_packages_obsolete_and_do_not_authorize_execution() -> None:
    operational_files = (
        REPO_ROOT / "PILOT_PROTOCOL.md",
        REPO_ROOT / "pilot" / "v0.2" / "README.md",
        REPO_ROOT / "pilot" / "v0.2" / "COMPLETION_CHECKLIST.md",
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in operational_files)

    assert "pilot-v0.2.1-draft" in combined
    assert "372096b" in combined
    assert "obsolete" in combined.casefold()
    assert "rag-evidence annotation collect" in combined
    assert "rag-evidence annotation adjudicate" in combined
    assert "rag-evidence annotation finalize-pilot" in combined
    assert "both Cohen's kappa and nominal Krippendorff's alpha" in combined
    assert "does not authorize the human pilot" in combined
