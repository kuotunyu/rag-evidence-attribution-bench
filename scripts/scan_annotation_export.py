"""Exit nonzero unless a pilot package contains blind assignments and zero decisions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rag_evidence.annotation.package import scan_clean_package
from rag_evidence.errors import DataError


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: scan_annotation_export.py PACKAGE_DIR", file=sys.stderr)
        return 2
    try:
        result = scan_clean_package(Path(sys.argv[1]))
    except DataError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "PASS", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
