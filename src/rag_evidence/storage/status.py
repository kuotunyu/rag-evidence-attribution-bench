"""The `status` command: per-stage completed/expected counts for the config's split.

First thing to run after a Colab reconnect — shows exactly where every stage stands.
"""

from __future__ import annotations

from pathlib import Path

from rag_evidence.config import AppConfig
from rag_evidence.storage.artifacts import RECORDS_FILE, RUN_META_FILE, read_json, read_records


def _count_records(path: Path) -> int:
    return sum(1 for _ in read_records(path)) if path.exists() else 0


def _walk_runs(raw_split_dir: Path) -> list[tuple[str, Path]]:
    """(label, run_dir) for every run under the split, attribution nested by mode."""
    runs: list[tuple[str, Path]] = []
    for stage in ("retrieve", "generate"):
        base = raw_split_dir / stage
        if base.exists():
            runs += [(f"{stage}/{d.name}", d) for d in sorted(base.iterdir()) if d.is_dir()]
    attribute_dir = raw_split_dir / "attribute"
    if attribute_dir.exists():
        for mode_dir in sorted(attribute_dir.iterdir()):
            if mode_dir.is_dir():
                runs += [
                    (f"attribute/{mode_dir.name}/{d.name}", d)
                    for d in sorted(mode_dir.iterdir())
                    if d.is_dir()
                ]
    return runs


def print_status(cfg: AppConfig) -> None:
    raw_split = cfg.results_raw_dir / cfg.split
    print(f"split: {cfg.split}")

    samples = raw_split / "samples" / RECORDS_FILE
    prepared = cfg.prepared_file
    print(
        f"  data      prepared={_count_records(prepared) if prepared.exists() else 'MISSING'} "
        f"samples={_count_records(samples) if samples.exists() else 'MISSING'}"
    )

    runs = _walk_runs(raw_split)
    if not runs:
        print("  (no stage runs yet)")
        return
    for label, run_dir in runs:
        n = _count_records(run_dir / RECORDS_FILE)
        meta_path = run_dir / RUN_META_FILE
        if meta_path.exists():
            meta = read_json(meta_path)
            expected = meta.get("expected_count", "?")
            status = meta.get("status", "?")
            kind = meta.get("execution_kind", "?")
            print(f"  {label:40s} {n}/{expected}  status={status}  kind={kind}")
        else:
            print(f"  {label:40s} {n}/?  (no run_meta.json)")

    summary = cfg.results_derived_dir / "summary.json"
    print(f"  summary.json: {'present' if summary.exists() else 'not yet generated'}")
