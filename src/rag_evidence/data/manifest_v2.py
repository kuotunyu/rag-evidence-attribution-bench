"""Construction and independent validation for leakage-resistant manifest schema v2."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from rag_evidence.data.allocation import allocate_split_groups
from rag_evidence.data.grouping import (
    build_split_groups,
    normalize_title,
    paragraph_fingerprint,
    supporting_fact_errors,
)
from rag_evidence.data.splits import example_fingerprint
from rag_evidence.errors import DataError

MANIFEST_V2_SCHEMA_VERSION = 2
_SPLIT_NAMES = ("smoke", "dev", "eval")
_DRAW_ORDER = ("eval", "dev", "smoke")
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


def _lines_hash(qid_to_hash: dict[str, str]) -> str:
    import hashlib

    joined = "\n".join(f"{qid}:{qid_to_hash[qid]}" for qid in sorted(qid_to_hash))
    return hashlib.sha256(joined.encode()).hexdigest()


def _distribution(raws: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(raw[field]) for raw in raws).items()))


def _validate_revisions(dataset: dict[str, Any]) -> None:
    requested = dataset.get("requested_revision")
    resolved = dataset.get("resolved_revision")
    if not isinstance(requested, str) or not _REVISION_PATTERN.fullmatch(requested):
        raise DataError("manifest v2 requested revision must be an exact 40-character SHA")
    if not isinstance(resolved, str) or not _REVISION_PATTERN.fullmatch(resolved):
        raise DataError("manifest v2 resolved revision must be an exact 40-character SHA")
    if requested != resolved:
        raise DataError(
            f"dataset revision mismatch: requested {requested}, resolved {resolved}"
        )


def _eligible_pool(
    raw_examples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    eligible: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    excluded = 0
    for raw in raw_examples:
        errors = supporting_fact_errors(raw)
        if errors:
            excluded += 1
            reasons.update(errors)
        else:
            eligible.append(raw)
    return eligible, dict(sorted(reasons.items())), excluded


def _overlap_audit(
    split_qids: dict[str, list[str]], raw_by_qid: dict[str, dict[str, Any]]
) -> dict[str, int]:
    qid_owners: dict[str, set[str]] = {}
    title_owners: dict[str, set[str]] = {}
    paragraph_owners: dict[str, set[str]] = {}
    for split, qids in split_qids.items():
        for qid in qids:
            qid_owners.setdefault(qid, set()).add(split)
            for title, sentences in raw_by_qid[qid]["context"]:
                title_owners.setdefault(normalize_title(str(title)), set()).add(split)
                paragraph_owners.setdefault(
                    paragraph_fingerprint([str(sentence) for sentence in sentences]), set()
                ).add(split)
    return {
        "question_id": sum(len(owners) > 1 for owners in qid_owners.values()),
        "normalized_title": sum(len(owners) > 1 for owners in title_owners.values()),
        "canonical_paragraph": sum(
            len(owners) > 1 for owners in paragraph_owners.values()
        ),
    }


def validate_manifest_v2_structure(manifest: dict[str, Any]) -> None:
    """Validate the self-contained structural contract without source examples."""
    if manifest.get("schema_version") != MANIFEST_V2_SCHEMA_VERSION:
        raise DataError(
            f"manifest schema_version must be {MANIFEST_V2_SCHEMA_VERSION}, "
            f"got {manifest.get('schema_version')}"
        )
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        raise DataError("manifest v2 dataset block is missing")
    _validate_revisions(dataset)
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or set(splits) != set(_SPLIT_NAMES):
        raise DataError(f"manifest v2 splits must be exactly {_SPLIT_NAMES}")
    all_qids: list[str] = []
    for split in _SPLIT_NAMES:
        row = splits[split]
        qids = row.get("question_ids")
        group_ids = row.get("group_ids")
        if not isinstance(qids, list) or not all(isinstance(qid, str) for qid in qids):
            raise DataError(f"manifest v2 split {split} question IDs are invalid")
        if row.get("size") != len(qids):
            raise DataError(f"manifest v2 split {split} question size mismatch")
        if not isinstance(group_ids, list) or not all(
            isinstance(group_id, str) and len(group_id) == 64 for group_id in group_ids
        ):
            raise DataError(f"manifest v2 split {split} group IDs are invalid")
        all_qids.extend(qids)
    if len(all_qids) != len(set(all_qids)):
        raise DataError("manifest v2 question IDs overlap across splits")
    selection = manifest.get("selection")
    if not isinstance(selection, dict):
        raise DataError("manifest v2 selection block is missing")
    for key in ("requested_sizes", "realized_sizes"):
        values = selection.get(key)
        if not isinstance(values, dict) or set(values) != set(_SPLIT_NAMES):
            raise DataError(f"manifest v2 {key} must cover {_SPLIT_NAMES}")
    audit = manifest.get("overlap_audit")
    expected_audit_keys = {"question_id", "normalized_title", "canonical_paragraph"}
    if not isinstance(audit, dict) or set(audit) != expected_audit_keys:
        raise DataError("manifest v2 overlap audit is incomplete")
    if not all(isinstance(value, int) and value >= 0 for value in audit.values()):
        raise DataError("manifest v2 overlap audit counts are invalid")


def build_manifest_v2(
    raw_examples: list[dict[str, Any]],
    *,
    seed: int,
    requested_sizes: dict[str, int],
    dataset_info: dict[str, Any],
) -> dict[str, Any]:
    """Build a byte-deterministic manifest from complete connected components."""
    _validate_revisions(dataset_info)
    raws = sorted(raw_examples, key=lambda raw: str(raw["question_id"]))
    all_hashes = {str(raw["question_id"]): example_fingerprint(raw) for raw in raws}
    if len(all_hashes) != len(raws):
        raise DataError("duplicate question IDs cannot enter manifest v2")
    eligible, exclusion_reasons, excluded_count = _eligible_pool(raws)
    groups = build_split_groups(eligible)
    allocation = allocate_split_groups(
        groups, requested_sizes=requested_sizes, seed=seed
    )
    raw_by_qid = {str(raw["question_id"]): raw for raw in raws}
    split_qids = {
        split: sorted(qid for group in allocation.groups[split] for qid in group.question_ids)
        for split in _SPLIT_NAMES
    }
    split_rows = {
        split: {
            "size": len(split_qids[split]),
            "public_frozen": split == "eval",
            "question_ids": split_qids[split],
            "group_ids": sorted(group.group_id for group in allocation.groups[split]),
            "type_distribution": _distribution(
                [raw_by_qid[qid] for qid in split_qids[split]], "type"
            ),
            "level_distribution": _distribution(
                [raw_by_qid[qid] for qid in split_qids[split]], "level"
            ),
        }
        for split in _SPLIT_NAMES
    }
    selected_qids = sorted(qid for qids in split_qids.values() for qid in qids)
    eligible_hashes = {
        str(raw["question_id"]): all_hashes[str(raw["question_id"])] for raw in eligible
    }
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_V2_SCHEMA_VERSION,
        "seed": seed,
        "dataset": {
            "hf_path": dataset_info["hf_path"],
            "hf_config": dataset_info["hf_config"],
            "hf_split": dataset_info["hf_split"],
            "requested_revision": dataset_info["requested_revision"],
            "resolved_revision": dataset_info["resolved_revision"],
            "num_examples": len(raws),
            "eligible_examples": len(eligible),
            "excluded_examples": excluded_count,
            "exclusion_reasons": exclusion_reasons,
            "eligible_type_distribution": _distribution(eligible, "type"),
            "eligible_level_distribution": _distribution(eligible, "level"),
        },
        "fingerprint": {
            "algorithm": "sha256",
            "canonicalization": (
                "per-example compact sorted-key canonical JSON, UTF-8; dataset hashes are "
                "newline-joined sorted question_id:fingerprint pairs"
            ),
            "dataset_hash": _lines_hash(all_hashes),
            "eligible_dataset_hash": _lines_hash(eligible_hashes),
            "split_hashes": {
                split: _lines_hash({qid: all_hashes[qid] for qid in split_qids[split]})
                for split in _SPLIT_NAMES
            },
        },
        "grouping": {
            "method": "connected_components_by_normalized_title_or_canonical_paragraph",
            "title_normalization": "Unicode NFKC, whitespace collapse, trim, casefold",
            "paragraph_canonicalization": (
                "ordered sentences; Unicode NFKC and whitespace collapse; compact JSON; sha256"
            ),
            "num_groups": len(groups),
            "largest_group_size": max((group.size for group in groups), default=0),
            "selected_groups": sum(len(rows) for rows in allocation.groups.values()),
        },
        "selection": {
            "method": "seeded_whole_group_dynamic_programming",
            "draw_order": list(_DRAW_ORDER),
            "requested_sizes": dict(requested_sizes),
            "realized_sizes": allocation.realized_sizes,
            "unselected_groups": allocation.unselected_group_count,
            "unselected_questions": allocation.unselected_question_count,
        },
        "overlap_audit": _overlap_audit(split_qids, raw_by_qid),
        "splits": split_rows,
        "example_hashes": {qid: all_hashes[qid] for qid in selected_qids},
    }
    validate_manifest_v2_structure(manifest)
    if any(manifest["overlap_audit"].values()):
        raise DataError(f"manifest v2 allocation produced overlap: {manifest['overlap_audit']}")
    return manifest


def validate_manifest_v2(
    manifest: dict[str, Any], raw_examples: list[dict[str, Any]]
) -> None:
    """Recompute every source-dependent manifest-v2 invariant and fail closed."""
    validate_manifest_v2_structure(manifest)
    raws = sorted(raw_examples, key=lambda raw: str(raw["question_id"]))
    raw_by_qid = {str(raw["question_id"]): raw for raw in raws}
    if len(raw_by_qid) != len(raws):
        raise DataError("source question IDs contain duplicates")
    selected_qids = [
        qid for split in _SPLIT_NAMES for qid in manifest["splits"][split]["question_ids"]
    ]
    missing = sorted(set(selected_qids) - set(raw_by_qid))
    if missing:
        raise DataError(f"selected question IDs missing from source: {missing[:3]}")
    for qid in selected_qids:
        errors = supporting_fact_errors(raw_by_qid[qid])
        if errors:
            raise DataError(f"selected question {qid} has supporting-fact errors: {errors}")

    all_hashes = {qid: example_fingerprint(raw) for qid, raw in raw_by_qid.items()}
    if manifest["fingerprint"]["dataset_hash"] != _lines_hash(all_hashes):
        raise DataError("source dataset fingerprint does not match manifest v2")
    expected_selected_hashes = {qid: all_hashes[qid] for qid in sorted(selected_qids)}
    if manifest.get("example_hashes") != expected_selected_hashes:
        raise DataError("selected example fingerprint does not match manifest v2")

    eligible, exclusion_reasons, excluded_count = _eligible_pool(raws)
    dataset = manifest["dataset"]
    if dataset["num_examples"] != len(raws):
        raise DataError("manifest v2 source example count mismatch")
    if dataset["eligible_examples"] != len(eligible):
        raise DataError("manifest v2 eligible example count mismatch")
    if dataset["excluded_examples"] != excluded_count:
        raise DataError("manifest v2 excluded example count mismatch")
    if dataset["exclusion_reasons"] != exclusion_reasons:
        raise DataError("manifest v2 supporting-fact exclusion reasons mismatch")
    if dataset["eligible_type_distribution"] != _distribution(eligible, "type"):
        raise DataError("manifest v2 eligible type distribution mismatch")
    if dataset["eligible_level_distribution"] != _distribution(eligible, "level"):
        raise DataError("manifest v2 eligible level distribution mismatch")
    eligible_hashes = {
        str(raw["question_id"]): all_hashes[str(raw["question_id"])] for raw in eligible
    }
    if manifest["fingerprint"]["eligible_dataset_hash"] != _lines_hash(eligible_hashes):
        raise DataError("eligible dataset fingerprint does not match manifest v2")

    groups = build_split_groups(eligible)
    grouping = manifest["grouping"]
    if grouping["num_groups"] != len(groups):
        raise DataError("manifest v2 group count mismatch")
    if grouping["largest_group_size"] != max(
        (group.size for group in groups), default=0
    ):
        raise DataError("manifest v2 largest group size mismatch")
    allocation = allocate_split_groups(
        groups,
        requested_sizes=dict(manifest["selection"]["requested_sizes"]),
        seed=int(manifest["seed"]),
    )
    for split in _SPLIT_NAMES:
        expected_qids = sorted(
            qid for group in allocation.groups[split] for qid in group.question_ids
        )
        row = manifest["splits"][split]
        if row["question_ids"] != expected_qids:
            raise DataError(f"manifest v2 split {split} question selection mismatch")
        expected_group_ids = sorted(group.group_id for group in allocation.groups[split])
        if row["group_ids"] != expected_group_ids:
            raise DataError(f"manifest v2 split {split} group membership mismatch")
        split_raws = [raw_by_qid[qid] for qid in expected_qids]
        if row["type_distribution"] != _distribution(split_raws, "type"):
            raise DataError(f"manifest v2 split {split} type distribution mismatch")
        if row["level_distribution"] != _distribution(split_raws, "level"):
            raise DataError(f"manifest v2 split {split} level distribution mismatch")
        split_hash = _lines_hash({qid: all_hashes[qid] for qid in expected_qids})
        if manifest["fingerprint"]["split_hashes"][split] != split_hash:
            raise DataError(f"manifest v2 split {split} fingerprint mismatch")
    if manifest["selection"]["realized_sizes"] != allocation.realized_sizes:
        raise DataError("manifest v2 realized sizes mismatch")
    if manifest["selection"]["unselected_groups"] != allocation.unselected_group_count:
        raise DataError("manifest v2 selection unselected group count mismatch")
    if (
        manifest["selection"]["unselected_questions"]
        != allocation.unselected_question_count
    ):
        raise DataError("manifest v2 selection unselected question count mismatch")
    if grouping["selected_groups"] != sum(
        len(rows) for rows in allocation.groups.values()
    ):
        raise DataError("manifest v2 selected group count mismatch")
    split_qids = {
        split: list(manifest["splits"][split]["question_ids"]) for split in _SPLIT_NAMES
    }
    recomputed_overlap = _overlap_audit(split_qids, raw_by_qid)
    if manifest["overlap_audit"] != recomputed_overlap or any(recomputed_overlap.values()):
        raise DataError(
            f"manifest v2 overlap audit mismatch: recorded={manifest['overlap_audit']}, "
            f"recomputed={recomputed_overlap}"
        )
