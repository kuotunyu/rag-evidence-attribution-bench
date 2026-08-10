"""Immutable manifest construction and validation for challenge schema v1."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rag_evidence.data.challenge_schema import (
    SOURCE_SPLITS,
    TRANSFORMATIONS,
    ChallengeRecord,
)
from rag_evidence.data.challenge_transforms import build_challenge_records
from rag_evidence.data.schema import Example
from rag_evidence.errors import DataError

CHALLENGE_MANIFEST_SCHEMA_VERSION = 1
_TOP_FIELDS = {"schema_version", "source", "generation", "splits", "record_hashes"}


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _records_hash(records: Sequence[ChallengeRecord]) -> str:
    lines = "\n".join(
        f"{record.challenge_id}:{record.content_hash}"
        for record in sorted(records, key=lambda row: row.challenge_id)
    )
    return hashlib.sha256(lines.encode()).hexdigest()


def _source_block(
    *,
    source_manifest_path: Path,
    source_manifest_bytes: bytes,
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if source_manifest.get("schema_version") != 2:
        raise DataError("challenge source manifest must use schema version 2")
    try:
        dataset = source_manifest["dataset"]
        fingerprint = source_manifest["fingerprint"]
        requested_revision = dataset["requested_revision"]
        resolved_revision = dataset["resolved_revision"]
        dataset_hash = fingerprint["dataset_hash"]
        split_hashes = fingerprint["split_hashes"]
    except (KeyError, TypeError) as exc:
        raise DataError(f"challenge source manifest is incomplete: {exc}") from exc
    if requested_revision != resolved_revision:
        raise DataError("challenge source revision requested/resolved mismatch")
    return {
        "manifest_path": source_manifest_path.as_posix(),
        "manifest_sha256": _sha256_bytes(source_manifest_bytes),
        "schema_version": 2,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "dataset_hash": dataset_hash,
        "split_hashes": {split: split_hashes[split] for split in SOURCE_SPLITS},
    }


def build_challenge_manifest(
    records_by_split: Mapping[str, Sequence[ChallengeRecord]],
    *,
    source_manifest_path: Path,
    source_manifest_bytes: bytes,
    source_manifest: Mapping[str, Any],
    seed: int,
    transform_version: str,
) -> dict[str, Any]:
    """Build a byte-deterministic challenge manifest from validated records."""
    if set(records_by_split) != set(SOURCE_SPLITS):
        raise DataError(f"challenge records must cover splits {SOURCE_SPLITS}")
    normalized = {
        split: tuple(sorted(records_by_split[split], key=lambda row: row.challenge_id))
        for split in SOURCE_SPLITS
    }
    all_records = [record for split in SOURCE_SPLITS for record in normalized[split]]
    ids = [record.challenge_id for record in all_records]
    if len(ids) != len(set(ids)):
        raise DataError("duplicate challenge IDs cannot enter manifest")
    for split, records in normalized.items():
        if any(record.source_split != split for record in records):
            raise DataError(f"challenge record stored under wrong split {split}")
    split_rows = {}
    for split in SOURCE_SPLITS:
        records = normalized[split]
        split_rows[split] = {
            "parent_count": len({record.parent_question_id for record in records}),
            "record_count": len(records),
            "transformation_counts": dict(
                sorted(Counter(record.transformation for record in records).items())
            ),
            "expected_answerability_counts": dict(
                sorted(Counter(record.expected_answerability for record in records).items())
            ),
            "review_status_counts": dict(
                sorted(Counter(record.review.status for record in records).items())
            ),
            "confirmatory_eligible_counts": dict(
                sorted(
                    Counter(
                        str(record.review.confirmatory_eligible).lower() for record in records
                    ).items()
                )
            ),
            "record_ids": [record.challenge_id for record in records],
            "records_hash": _records_hash(records),
        }
    return {
        "schema_version": CHALLENGE_MANIFEST_SCHEMA_VERSION,
        "source": _source_block(
            source_manifest_path=source_manifest_path,
            source_manifest_bytes=source_manifest_bytes,
            source_manifest=source_manifest,
        ),
        "generation": {
            "seed": seed,
            "transform_version": transform_version,
            "transformations": list(TRANSFORMATIONS),
        },
        "splits": split_rows,
        "record_hashes": {
            record.challenge_id: record.content_hash
            for record in sorted(all_records, key=lambda row: row.challenge_id)
        },
    }


def _parse_rows(
    records_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, tuple[ChallengeRecord, ...]]:
    if set(records_by_split) != set(SOURCE_SPLITS):
        raise DataError(f"challenge rows must cover splits {SOURCE_SPLITS}")
    parsed: dict[str, tuple[ChallengeRecord, ...]] = {}
    all_ids: list[str] = []
    for split in SOURCE_SPLITS:
        records = tuple(ChallengeRecord.from_json(row) for row in records_by_split[split])
        if any(record.source_split != split for record in records):
            raise DataError(f"challenge row stored under wrong split {split}")
        parsed[split] = tuple(sorted(records, key=lambda row: row.challenge_id))
        all_ids.extend(record.challenge_id for record in records)
    if len(all_ids) != len(set(all_ids)):
        raise DataError("duplicate challenge IDs in challenge rows")
    return parsed


def validate_challenge_manifest(
    manifest: Mapping[str, Any],
    records_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    source_manifest_path: Path,
    source_manifest_bytes: bytes,
    source_manifest: Mapping[str, Any],
    parents_by_split: Mapping[str, Sequence[Example]],
    parent_fingerprints: Mapping[str, str],
) -> None:
    """Independently regenerate records and every source-bound manifest invariant."""
    if set(manifest) != _TOP_FIELDS:
        raise DataError("challenge manifest top-level fields are invalid")
    if manifest.get("schema_version") != CHALLENGE_MANIFEST_SCHEMA_VERSION:
        raise DataError("challenge manifest schema version must be 1")
    try:
        parsed_source = json.loads(source_manifest_bytes)
    except json.JSONDecodeError as exc:
        raise DataError("challenge source manifest bytes are not valid JSON") from exc
    if parsed_source != source_manifest:
        raise DataError("challenge source manifest bytes do not match parsed source")
    expected_source = _source_block(
        source_manifest_path=source_manifest_path,
        source_manifest_bytes=source_manifest_bytes,
        source_manifest=source_manifest,
    )
    actual_source = manifest.get("source")
    if not isinstance(actual_source, dict):
        raise DataError("challenge source block is missing")
    for field, expected in expected_source.items():
        if actual_source.get(field) != expected:
            label = "revision" if "revision" in field else "source"
            raise DataError(f"challenge {label} field {field} mismatch")
    generation = manifest.get("generation")
    if not isinstance(generation, dict):
        raise DataError("challenge generation block is missing")
    if generation.get("transform_version") != "challenge-v1":
        raise DataError("challenge generation transform version mismatch")
    if generation.get("transformations") != list(TRANSFORMATIONS):
        raise DataError("challenge generation transformations mismatch")
    seed = generation.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise DataError("challenge generation seed is invalid")
    parsed = _parse_rows(records_by_split)
    expected_records = build_challenge_records(
        parents_by_split,
        parent_fingerprints,
        seed=seed,
        transform_version="challenge-v1",
    )
    for split in SOURCE_SPLITS:
        actual_rows = [record.to_json() for record in parsed[split]]
        expected_rows = [record.to_json() for record in expected_records[split]]
        if actual_rows != expected_rows:
            raise DataError(f"challenge record content mismatch in split {split}")
    expected_manifest = build_challenge_manifest(
        expected_records,
        source_manifest_path=source_manifest_path,
        source_manifest_bytes=source_manifest_bytes,
        source_manifest=source_manifest,
        seed=seed,
        transform_version="challenge-v1",
    )
    actual_hashes = manifest.get("record_hashes")
    if actual_hashes != expected_manifest["record_hashes"]:
        raise DataError("challenge record hash inventory mismatch")
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or set(splits) != set(SOURCE_SPLITS):
        raise DataError("challenge manifest split blocks are invalid")
    for split in SOURCE_SPLITS:
        actual = splits[split]
        expected = expected_manifest["splits"][split]
        if not isinstance(actual, dict):
            raise DataError(f"challenge split {split} is not an object")
        for field, expected_value in expected.items():
            if actual.get(field) != expected_value:
                if field == "records_hash":
                    label = "split hash"
                elif field == "transformation_counts":
                    label = "transformation"
                elif field == "review_status_counts":
                    label = "review"
                else:
                    label = "split accounting"
                raise DataError(f"challenge {label} mismatch for {split}: {field}")
