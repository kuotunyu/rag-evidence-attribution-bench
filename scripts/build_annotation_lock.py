"""Write or verify the canonical Python 3.11 annotation requirements lock."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_evidence.annotation.dependency_lock import build_annotation_lock

LOCK_RELATIVE = Path("pilot/v0.2/annotation-requirements-py311.lock")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    repository_root = args.repository_root.resolve()
    lock_path = repository_root / LOCK_RELATIVE
    generated = build_annotation_lock(repository_root)
    if args.check:
        return 0 if lock_path.is_file() and lock_path.read_bytes() == generated else 1
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
