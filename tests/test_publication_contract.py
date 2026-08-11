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
