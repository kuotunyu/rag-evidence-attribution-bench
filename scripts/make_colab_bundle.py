"""Build reab_bundle.zip for Colab: all git-tracked files + the prepared data splits.

The bundle exists because the repo has no GitHub remote yet — the user uploads ONE zip
to Drive (MyDrive/reab/reab_bundle.zip). Shipping the prepared splits means Colab never
downloads the HotpotQA dataset (fingerprints still verified at every stage start).

Usage: uv run python scripts/make_colab_bundle.py [output_zip]
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else repo_root / "reab_bundle.zip"

    tracked = subprocess.run(
        ["git", "ls-files"], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout.splitlines()

    prepared = sorted((repo_root / "data" / "prepared").glob("*.jsonl"))
    if not prepared:
        raise SystemExit(
            "data/prepared/*.jsonl missing — run `python -m rag_evidence.cli data prepare "
            "--config configs/smoke.yaml` first (the bundle must ship the prepared splits)"
        )

    n = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in tracked:
            f = repo_root / rel
            if f.is_file():
                zf.write(f, rel)
                n += 1
        for f in prepared:
            zf.write(f, f"data/prepared/{f.name}")
            n += 1
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"wrote {out} ({n} files, {size_mb:.1f} MB)")
    print("next: open a Colab notebook and pick this file when the upload cell prompts")


if __name__ == "__main__":
    main()
