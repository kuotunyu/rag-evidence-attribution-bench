"""Manifest v2 is deterministic, group-disjoint, and independently validated."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from rag_evidence.data.grouping import build_split_groups, supporting_fact_errors
from rag_evidence.data.manifest_v2 import build_manifest_v2, validate_manifest_v2
from rag_evidence.data.splits import load_manifest
from rag_evidence.errors import DataError
from rag_evidence.storage.artifacts import write_json_atomic

REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
SIZES = {"smoke": 1, "dev": 1, "eval": 2}
DATASET_INFO = {
    "hf_path": "hotpotqa/hotpot_qa",
    "hf_config": "distractor",
    "hf_split": "validation",
    "requested_revision": REVISION,
    "resolved_revision": REVISION,
}


def _raw(
    qid: str,
    title: str,
    sentence: str,
    *,
    qtype: str = "bridge",
    level: str = "medium",
    supporting_title: str | None = None,
) -> dict[str, Any]:
    return {
        "question_id": qid,
        "question": f"Question {qid}?",
        "answer": f"Answer {qid}",
        "type": qtype,
        "level": level,
        "context": [[title, [sentence]]],
        "supporting_facts": [[supporting_title or title, 0]],
    }


def _raws() -> list[dict[str, Any]]:
    return [
        _raw("q-a", "Shared", "First shared member."),
        _raw("q-b", " shared ", "Second shared member.", qtype="comparison"),
        _raw("q-c", "Solo C", "Unique C.", level="hard"),
        _raw("q-d", "Solo D", "Unique D.", qtype="comparison"),
        _raw("q-e", "Solo E", "Unique E."),
        _raw("q-f", "Solo F", "Unique F.", qtype="comparison", level="hard"),
        _raw("q-invalid", "Present", "Malformed.", supporting_title="Absent"),
    ]


def _manifest() -> dict[str, Any]:
    return build_manifest_v2(_raws(), seed=23, requested_sizes=SIZES, dataset_info=DATASET_INFO)


def test_manifest_v2_is_deterministic_and_records_the_full_audit_contract() -> None:
    manifest = _manifest()
    reversed_manifest = build_manifest_v2(
        list(reversed(_raws())),
        seed=23,
        requested_sizes=SIZES,
        dataset_info=DATASET_INFO,
    )

    assert manifest == reversed_manifest
    assert json.dumps(manifest, sort_keys=True, ensure_ascii=False) == json.dumps(
        reversed_manifest, sort_keys=True, ensure_ascii=False
    )
    assert manifest["schema_version"] == 2
    assert "created_utc" not in manifest
    assert manifest["dataset"]["requested_revision"] == REVISION
    assert manifest["dataset"]["resolved_revision"] == REVISION
    assert manifest["dataset"]["num_examples"] == 7
    assert manifest["dataset"]["eligible_examples"] == 6
    assert manifest["dataset"]["excluded_examples"] == 1
    assert manifest["dataset"]["exclusion_reasons"] == {"title_not_in_context": 1}
    assert manifest["selection"]["requested_sizes"] == SIZES
    assert manifest["selection"]["realized_sizes"] == SIZES
    assert manifest["overlap_audit"] == {
        "question_id": 0,
        "normalized_title": 0,
        "canonical_paragraph": 0,
    }
    selected_qids = [qid for split in manifest["splits"].values() for qid in split["question_ids"]]
    assert len(selected_qids) == len(set(selected_qids)) == sum(SIZES.values())
    eligible = [raw for raw in _raws() if not supporting_fact_errors(raw)]
    groups = {group.group_id: set(group.question_ids) for group in build_split_groups(eligible)}
    for split in manifest["splits"].values():
        split_qids = set(split["question_ids"])
        assert split["size"] == len(split_qids)
        assert sum(split["type_distribution"].values()) == split["size"]
        assert sum(split["level_distribution"].values()) == split["size"]
        for group_id in split["group_ids"]:
            assert groups[group_id] <= split_qids
    validate_manifest_v2(manifest, _raws())


def test_load_manifest_accepts_valid_v2_and_rejects_bad_revision(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_json_atomic(path, _manifest())
    assert load_manifest(path)["schema_version"] == 2

    corrupt = _manifest()
    corrupt["dataset"]["resolved_revision"] = "main"
    write_json_atomic(path, corrupt)
    with pytest.raises(DataError, match="revision"):
        load_manifest(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("fingerprint", "fingerprint"),
        ("group", "group"),
        ("question", "question"),
        ("overlap", "overlap"),
        ("supporting", "supporting"),
        ("group_count", "group"),
        ("distribution", "distribution"),
        ("unselected", "selection"),
    ],
)
def test_manifest_v2_validation_fails_closed_on_corruption(mutation: str, message: str) -> None:
    manifest = _manifest()
    raws = _raws()
    if mutation == "fingerprint":
        selected = manifest["splits"]["eval"]["question_ids"][0]
        next(raw for raw in raws if raw["question_id"] == selected)["context"][0][1][0] += " edit"
    elif mutation == "group":
        manifest["splits"]["eval"]["group_ids"][0] = "0" * 64
    elif mutation == "question":
        manifest["splits"]["smoke"]["question_ids"][0] = manifest["splits"]["dev"]["question_ids"][
            0
        ]
    elif mutation == "overlap":
        manifest["overlap_audit"]["normalized_title"] = 1
    elif mutation == "supporting":
        selected = manifest["splits"]["eval"]["question_ids"][0]
        next(raw for raw in raws if raw["question_id"] == selected)["supporting_facts"][0][1] = 9
    elif mutation == "group_count":
        manifest["grouping"]["num_groups"] += 1
    elif mutation == "distribution":
        manifest["dataset"]["eligible_type_distribution"]["bridge"] += 1
    elif mutation == "unselected":
        manifest["selection"]["unselected_groups"] += 1
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(mutation)

    with pytest.raises(DataError, match=message):
        validate_manifest_v2(manifest, raws)


def test_manifest_v2_rejects_revision_mismatch_before_selection() -> None:
    info = copy.deepcopy(DATASET_INFO)
    info["resolved_revision"] = "f" * 40

    with pytest.raises(DataError, match="revision"):
        build_manifest_v2(_raws(), seed=23, requested_sizes=SIZES, dataset_info=info)
