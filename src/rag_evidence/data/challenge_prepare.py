"""Validate v2 source artifacts and materialize answerability challenge v1."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rag_evidence.config import AppConfig
from rag_evidence.data.challenge_manifest import (
    build_challenge_manifest,
    validate_challenge_manifest,
)
from rag_evidence.data.challenge_schema import SOURCE_SPLITS, ChallengeRecord
from rag_evidence.data.challenge_transforms import build_challenge_records
from rag_evidence.data.manifest_v2 import validate_manifest_v2_structure
from rag_evidence.data.schema import Example
from rag_evidence.errors import DataError
from rag_evidence.storage.artifacts import (
    read_json,
    read_records,
    write_json_atomic,
    write_records_atomic,
)

logger = logging.getLogger(__name__)


def _load_source_manifest(cfg: AppConfig) -> tuple[bytes, dict[str, Any]]:
    if cfg.data.manifest_schema_version != 2:
        raise DataError("answerability challenge requires source manifest schema v2")
    path = cfg.manifest_file
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise DataError(f"challenge source manifest {path} is unavailable: {exc}") from exc
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise DataError(f"challenge source manifest {path} is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise DataError("challenge source manifest must be a JSON object")
    validate_manifest_v2_structure(parsed)
    if parsed.get("seed") != cfg.data.split_seed:
        raise DataError("challenge source manifest seed does not match config")
    requested_sizes = parsed["selection"]["requested_sizes"]
    if requested_sizes != dict(cfg.data.split_sizes):
        raise DataError("challenge source manifest split sizes do not match config")
    dataset = parsed["dataset"]
    if (
        dataset.get("requested_revision") != cfg.data.hf_revision
        or dataset.get("resolved_revision") != cfg.data.hf_revision
    ):
        raise DataError("challenge source manifest revision does not match config")
    return content, parsed


def _load_parents(
    cfg: AppConfig, source_manifest: Mapping[str, Any]
) -> tuple[dict[str, tuple[Example, ...]], dict[str, str]]:
    hashes = source_manifest.get("example_hashes")
    if not isinstance(hashes, dict):
        raise DataError("challenge source manifest example hashes are missing")

    parents_by_split: dict[str, tuple[Example, ...]] = {}
    parent_fingerprints: dict[str, str] = {}
    all_question_ids: list[str] = []
    for split in SOURCE_SPLITS:
        path = Path(cfg.data.prepared_dir) / f"{split}.jsonl"
        if not path.exists():
            raise DataError(f"source prepared file {path} not found; run `data prepare` first")
        rows = list(read_records(path))
        expected = source_manifest["splits"][split]["question_ids"]
        got: list[str] = []
        for row in rows:
            qid = row.get("question_id")
            if not isinstance(qid, str):
                raise DataError(f"source prepared split {split} has an invalid question ID")
            got.append(qid)
        if len(got) != len(set(got)):
            raise DataError(f"source prepared split {split} contains duplicate question IDs")
        if sorted(got) != sorted(expected):
            raise DataError(
                f"source prepared split {split} does not match source manifest question IDs"
            )

        parents: list[Example] = []
        for row in rows:
            qid = row["question_id"]
            expected_fingerprint = hashes.get(qid)
            actual_fingerprint = row.get("raw_fingerprint")
            if (
                not isinstance(expected_fingerprint, str)
                or not isinstance(actual_fingerprint, str)
                or actual_fingerprint != expected_fingerprint
            ):
                raise DataError(
                    f"source fingerprint mismatch for question {qid}: "
                    f"expected {expected_fingerprint}, got {actual_fingerprint}"
                )
            try:
                parent = Example.from_json(row)
            except (KeyError, TypeError, ValueError) as exc:
                raise DataError(f"invalid source prepared row for question {qid}: {exc}") from exc
            if parent.dropped_supporting_facts:
                raise DataError(
                    f"source question {qid} has dropped supporting facts and is ineligible"
                )
            parents.append(parent)
            parent_fingerprints[qid] = actual_fingerprint
            all_question_ids.append(qid)
        parents_by_split[split] = tuple(sorted(parents, key=lambda row: row.question_id))

    if len(all_question_ids) != len(set(all_question_ids)):
        raise DataError("source prepared question IDs overlap across splits")
    if set(parent_fingerprints) != set(hashes):
        raise DataError("source manifest hash inventory does not match prepared split inventory")
    return parents_by_split, parent_fingerprints


def _json_rows(
    records_by_split: Mapping[str, Sequence[ChallengeRecord]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        split: [record.to_json() for record in records_by_split[split]] for split in SOURCE_SPLITS
    }


def prepare_challenge(cfg: AppConfig) -> None:
    """Build challenge rows only after validating every immutable source binding."""
    source_bytes, source_manifest = _load_source_manifest(cfg)
    parents_by_split, parent_fingerprints = _load_parents(cfg, source_manifest)
    records = build_challenge_records(
        parents_by_split,
        parent_fingerprints,
        seed=cfg.challenge.seed,
        transform_version=cfg.challenge.transform_version,
    )
    rows = _json_rows(records)

    challenge_path = cfg.challenge_manifest_file
    if challenge_path.exists():
        existing = read_json(challenge_path)
        if not isinstance(existing, dict):
            raise DataError("challenge manifest must be a JSON object")
        validate_challenge_manifest(
            existing,
            rows,
            source_manifest_path=cfg.manifest_file,
            source_manifest_bytes=source_bytes,
            source_manifest=source_manifest,
            parents_by_split=parents_by_split,
            parent_fingerprints=parent_fingerprints,
        )
    else:
        existing = build_challenge_manifest(
            records,
            source_manifest_path=cfg.manifest_file,
            source_manifest_bytes=source_bytes,
            source_manifest=source_manifest,
            seed=cfg.challenge.seed,
            transform_version=cfg.challenge.transform_version,
        )
        validate_challenge_manifest(
            existing,
            rows,
            source_manifest_path=cfg.manifest_file,
            source_manifest_bytes=source_bytes,
            source_manifest=source_manifest,
            parents_by_split=parents_by_split,
            parent_fingerprints=parent_fingerprints,
        )
        write_json_atomic(challenge_path, existing)
        logger.info("wrote immutable challenge manifest %s", challenge_path)

    for split in SOURCE_SPLITS:
        prepared_path = cfg.challenge_prepared_dir / f"{split}.jsonl"
        samples_path = cfg.results_raw_dir / split / "challenge" / "samples" / "records.jsonl"
        write_records_atomic(prepared_path, rows[split])
        write_records_atomic(samples_path, rows[split])
        logger.info("prepared challenge %s: %d records", split, len(rows[split]))
