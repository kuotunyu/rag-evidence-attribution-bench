"""Challenge manifests bind every deterministic record to the frozen source manifest."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from rag_evidence.data.challenge_manifest import (
    build_challenge_manifest,
    validate_challenge_manifest,
)
from rag_evidence.data.challenge_transforms import build_challenge_records
from rag_evidence.data.schema import Example, Passage
from rag_evidence.errors import DataError

REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"


def _parent(qid: str, peer: str) -> Example:
    return Example(
        question_id=qid,
        question=f"Who founded Project {qid}?",
        answer=f"Founder {qid}",
        level="hard",
        qtype="bridge",
        passages=(
            Passage(
                passage_id=f"{qid}-p00",
                index=0,
                title=f"Project {qid}",
                sentences=(f"Project {qid} was founded by Founder {qid}.",),
                is_gold=True,
            ),
            Passage(
                passage_id=f"{qid}-p01",
                index=1,
                title=f"Link {qid}",
                sentences=(f"Link {qid} identifies Project {qid}.",),
                is_gold=True,
            ),
            Passage(
                passage_id=f"{qid}-p02",
                index=2,
                title=f"Distractor {peer}",
                sentences=(f"Distractor {peer} concerns an unrelated festival.",),
                is_gold=False,
            ),
        ),
        gold_passage_ids=(f"{qid}-p00", f"{qid}-p01"),
        supporting_fact_sentence_ids=(f"{qid}-p00-s00", f"{qid}-p01-s00"),
    )


def _parents() -> dict[str, tuple[Example, ...]]:
    return {
        split: (
            _parent(f"{split}-one", f"{split}-alpha"),
            _parent(f"{split}-two", f"{split}-beta"),
        )
        for split in ("smoke", "dev", "eval")
    }


def _fingerprints() -> dict[str, str]:
    return {
        parent.question_id: f"{index:064x}"
        for index, parent in enumerate(
            (parent for rows in _parents().values() for parent in rows), start=1
        )
    }


def _source_manifest() -> dict[str, Any]:
    parents = _parents()
    fingerprints = _fingerprints()
    return {
        "schema_version": 2,
        "dataset": {
            "requested_revision": REVISION,
            "resolved_revision": REVISION,
        },
        "fingerprint": {
            "dataset_hash": "d" * 64,
            "split_hashes": {"smoke": "1" * 64, "dev": "2" * 64, "eval": "3" * 64},
        },
        "splits": {
            split: {"question_ids": [parent.question_id for parent in rows]}
            for split, rows in parents.items()
        },
        "example_hashes": fingerprints,
    }


def _source_bytes() -> bytes:
    return (json.dumps(_source_manifest(), ensure_ascii=False, indent=2) + "\n").encode()


def _records():
    return build_challenge_records(
        _parents(),
        _fingerprints(),
        seed=20260810,
        transform_version="challenge-v1",
    )


def _rows(records=None) -> dict[str, list[dict[str, Any]]]:
    source = _records() if records is None else records
    return {
        split: [record.to_json() for record in split_records]
        for split, split_records in source.items()
    }


def _manifest(records=None) -> dict[str, Any]:
    return build_challenge_manifest(
        _records() if records is None else records,
        source_manifest_path=Path("data/manifests/split_manifest_v2.json"),
        source_manifest_bytes=_source_bytes(),
        source_manifest=_source_manifest(),
        seed=20260810,
        transform_version="challenge-v1",
    )


def _validate(manifest: dict[str, Any], rows=None, source_bytes: bytes | None = None) -> None:
    validate_challenge_manifest(
        manifest,
        _rows() if rows is None else rows,
        source_manifest_path=Path("data/manifests/split_manifest_v2.json"),
        source_manifest_bytes=_source_bytes() if source_bytes is None else source_bytes,
        source_manifest=_source_manifest(),
        parents_by_split=_parents(),
        parent_fingerprints=_fingerprints(),
    )


def test_manifest_is_reorder_stable_and_validates_against_source() -> None:
    records = _records()
    reversed_records = {
        split: tuple(reversed(rows))
        for split, rows in reversed(tuple(records.items()))
    }

    manifest = _manifest(records)
    rebuilt = _manifest(reversed_records)

    assert manifest == rebuilt
    assert manifest["schema_version"] == 1
    assert manifest["source"]["resolved_revision"] == REVISION
    assert manifest["source"]["dataset_hash"] == "d" * 64
    assert set(manifest["splits"]) == {"smoke", "dev", "eval"}
    for split in ("smoke", "dev", "eval"):
        assert manifest["splits"][split]["parent_count"] == 2
        assert manifest["splits"][split]["record_count"] == 6
        assert manifest["splits"][split]["transformation_counts"] == {
            "answer_bearing_distractor": 2,
            "evidence_swap": 2,
            "missing_hop": 2,
        }
        assert manifest["splits"][split]["review_status_counts"] == {
            "not_required": 2,
            "pending_two_annotators": 4,
        }
    _validate(manifest)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("source_sha", "source"),
        ("revision", "revision"),
        ("record_hash", "record hash"),
        ("split_hash", "split hash"),
        ("transformation_count", "transformation"),
        ("review_count", "review"),
    ],
)
def test_manifest_rejects_corrupt_accounting(field: str, message: str) -> None:
    manifest = copy.deepcopy(_manifest())
    first_id = next(iter(manifest["record_hashes"]))
    if field == "source_sha":
        manifest["source"]["manifest_sha256"] = "0" * 64
    elif field == "revision":
        manifest["source"]["resolved_revision"] = "f" * 40
    elif field == "record_hash":
        manifest["record_hashes"][first_id] = "0" * 64
    elif field == "split_hash":
        manifest["splits"]["eval"]["records_hash"] = "0" * 64
    elif field == "transformation_count":
        manifest["splits"]["eval"]["transformation_counts"]["missing_hop"] -= 1
    else:
        manifest["splits"]["eval"]["review_status_counts"][
            "pending_two_annotators"
        ] -= 1

    with pytest.raises(DataError, match=message):
        _validate(manifest)


def test_manifest_rejects_source_bytes_drift_duplicate_ids_and_nested_content_drift() -> None:
    manifest = _manifest()

    with pytest.raises(DataError, match="source"):
        _validate(manifest, source_bytes=_source_bytes() + b" ")

    duplicate_rows = _rows()
    duplicate_rows["eval"].append(copy.deepcopy(duplicate_rows["eval"][0]))
    with pytest.raises(DataError, match="duplicate"):
        _validate(manifest, rows=duplicate_rows)

    changed_rows = _rows()
    changed_rows["eval"][0]["example"]["question"] = "Changed?"
    with pytest.raises(DataError, match="content hash"):
        _validate(manifest, rows=changed_rows)
