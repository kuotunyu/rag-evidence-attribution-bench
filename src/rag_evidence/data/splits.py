"""Split manifest: seeded disjoint splits + content fingerprints.

The manifest is COMMITTED and is the ground truth for (a) which questions belong to which
split and (b) what their content must hash to. Positional passage IDs are safe only
because every stage verifies these fingerprints before using the data.

The locked eval split is drawn FIRST from the seeded shuffle so nothing tuned on
smoke/dev can leak into it. `data prepare` never regenerates an existing manifest.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

from rag_evidence.errors import ConfigError, DataError, FingerprintMismatchError
from rag_evidence.storage.artifacts import read_json, write_json_atomic

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA_VERSION = 1
# draw order is part of the scheme: locked eval first
_DRAW_ORDER = ("eval", "dev", "smoke")


def canonical_raw(raw: dict[str, Any]) -> dict[str, Any]:
    """The exact content that is fingerprinted, in canonical structure."""
    return {
        "question_id": raw["question_id"],
        "question": raw["question"],
        "answer": raw["answer"],
        "type": raw["type"],
        "level": raw["level"],
        "context": [[title, list(sents)] for title, sents in raw["context"]],
        "supporting_facts": [[t, int(s)] for t, s in raw["supporting_facts"]],
    }


def example_fingerprint(raw: dict[str, Any]) -> str:
    payload = json.dumps(
        canonical_raw(raw), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _lines_hash(qid_to_hash: dict[str, str]) -> str:
    joined = "\n".join(f"{qid}:{qid_to_hash[qid]}" for qid in sorted(qid_to_hash))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def build_manifest(
    raw_examples: list[dict[str, Any]],
    *,
    seed: int,
    sizes: dict[str, int],
    dataset_info: dict[str, Any],
    created_utc: str,
) -> dict[str, Any]:
    total_needed = sum(sizes.values())
    if len(raw_examples) < total_needed:
        raise DataError(
            f"need {total_needed} examples for the splits, dataset has {len(raw_examples)}"
        )
    all_hashes = {raw["question_id"]: example_fingerprint(raw) for raw in raw_examples}
    if len(all_hashes) != len(raw_examples):
        raise DataError("duplicate question_ids in the dataset — cannot build manifest")

    qids = sorted(all_hashes)  # sort first so the shuffle is order-independent
    random.Random(seed).shuffle(qids)

    splits: dict[str, dict[str, Any]] = {}
    cursor = 0
    for split in _DRAW_ORDER:
        n = sizes[split]
        chosen = sorted(qids[cursor : cursor + n])
        cursor += n
        splits[split] = {
            "size": n,
            "locked": split == "eval",
            "question_ids": chosen,
        }

    selected = [qid for s in splits.values() for qid in s["question_ids"]]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_utc": created_utc,
        "seed": seed,
        "dataset": {**dataset_info, "num_examples": len(raw_examples)},
        "fingerprint": {
            "algorithm": "sha256",
            "canonicalization": (
                "per-example sha256 over compact sorted-key JSON of {question_id, question, "
                "answer, type, level, context:[[title,[sentences]]...], "
                "supporting_facts:[[title,sent_id]...]}, UTF-8"
            ),
            "dataset_hash": _lines_hash(all_hashes),
            "split_hashes": {
                name: _lines_hash({q: all_hashes[q] for q in s["question_ids"]})
                for name, s in splits.items()
            },
        },
        "selection": {
            "method": "seeded_shuffle_disjoint",
            "draw_order": list(_DRAW_ORDER),
            "notes": "sorted qids shuffled with `seed`; eval drawn first (locked), then dev, smoke",
        },
        "splits": {name: splits[name] for name in ("smoke", "dev", "eval")},
        "example_hashes": {qid: all_hashes[qid] for qid in sorted(selected)},
    }


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise DataError(
            f"manifest {path} has schema_version {manifest.get('schema_version')}, "
            f"expected {MANIFEST_SCHEMA_VERSION}"
        )
    return manifest


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if path.exists():
        raise DataError(
            f"manifest {path} already exists — it is never regenerated automatically "
            "(the eval split is locked). Delete it manually only if you intend to "
            "invalidate every existing result."
        )
    write_json_atomic(path, manifest)


def check_manifest_matches_config(
    manifest: dict[str, Any], *, seed: int, sizes: dict[str, int]
) -> None:
    if manifest["seed"] != seed:
        raise ConfigError(
            f"config split_seed={seed} but committed manifest was built with "
            f"seed={manifest['seed']}; refusing to mix"
        )
    for name, size in sizes.items():
        actual = manifest["splits"][name]["size"]
        if actual != size:
            raise ConfigError(
                f"config split_sizes[{name}]={size} but manifest has {actual}; refusing to mix"
            )


def verify_examples(
    raw_by_qid: dict[str, dict[str, Any]], manifest: dict[str, Any], split: str
) -> None:
    """Verify content fingerprints for every question of a split. Loud failure on drift."""
    expected = manifest["example_hashes"]
    for qid in manifest["splits"][split]["question_ids"]:
        if qid not in raw_by_qid:
            raise DataError(f"question {qid} (split {split}) missing from the loaded dataset")
        actual = example_fingerprint(raw_by_qid[qid])
        if actual != expected[qid]:
            raise FingerprintMismatchError(qid, expected[qid], actual)
    logger.info(
        "fingerprints verified: %d/%d examples of split %r",
        len(manifest["splits"][split]["question_ids"]),
        manifest["splits"][split]["size"],
        split,
    )
