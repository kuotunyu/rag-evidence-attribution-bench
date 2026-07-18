"""JSONL checkpoint files and atomic JSON artifacts.

Integrity model for JSONL checkpoints (append-only, one JSON object per line):
- every appended line is flushed + fsynced, so a crash loses at most the in-flight record;
- on read, a malformed *final* line is treated as a crash artifact: dropped with a warning;
- a malformed line anywhere else is corruption and raises (never silently skip data);
- resume = build the set of completed dedup keys and skip them.

Whole-file JSON artifacts (manifests, summaries) use write-to-temp + os.replace, which is
atomic on NTFS and the strongest available primitive on Drive-FUSE mounts.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from rag_evidence.errors import ArtifactError

logger = logging.getLogger(__name__)


def append_record(path: Path, record: Mapping[str, Any]) -> None:
    """Append one record as a single JSON line; flush + fsync before returning."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    if "\n" in line:  # cannot happen with json.dumps, but the invariant is load-bearing
        raise ArtifactError("record serialized with embedded newline")
    try:
        with path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    except OSError as exc:
        raise ArtifactError(f"failed to append record to {path}: {exc}") from exc


def read_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield records; tolerate a truncated final line (crash), refuse mid-file corruption."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError as exc:
        raise ArtifactError(f"failed to read {path}: {exc}") from exc
    last_index = len(lines) - 1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as exc:
            if i == last_index:
                logger.warning(
                    "dropping truncated final line of %s (interrupted write); "
                    "it will be recomputed on resume",
                    path,
                )
                return
            raise ArtifactError(
                f"{path}:{i + 1} is malformed JSON mid-file — the checkpoint is corrupt; "
                "refusing to resume silently"
            ) from exc
        if not isinstance(obj, dict):
            raise ArtifactError(f"{path}:{i + 1} is not a JSON object")
        yield obj


def completed_keys(
    path: Path, key_fields: Sequence[str] = ("question_id",)
) -> set[tuple[Any, ...]]:
    """Dedup keys of all completed records; empty set if the file does not exist."""
    if not path.exists():
        return set()
    keys: set[tuple[Any, ...]] = set()
    for rec in read_records(path):
        try:
            keys.add(tuple(rec[f] for f in key_fields))
        except KeyError as exc:
            raise ArtifactError(f"record in {path} missing dedup field {exc}") from exc
    return keys


def write_json_atomic(path: Path, obj: Any) -> None:
    """Write JSON via temp file + os.replace (atomic on NTFS/POSIX; best-effort on FUSE)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise ArtifactError(f"failed to write {path}: {exc}") from exc


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except OSError as exc:
        raise ArtifactError(f"failed to read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{path} is not valid JSON: {exc}") from exc


def stage_dir(results_raw: Path, split: str, stage: str, name: str) -> Path:
    """Canonical location of a run: results/raw/{split}/{stage}/{name}/"""
    return results_raw / split / stage / name


RECORDS_FILE = "records.jsonl"
RUN_META_FILE = "run_meta.json"
