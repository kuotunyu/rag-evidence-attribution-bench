"""Split manifest: determinism, disjointness, fingerprint drift detection, lock rules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_evidence.data.hotpot import normalize_hf_example
from rag_evidence.data.splits import (
    build_manifest,
    check_manifest_matches_config,
    example_fingerprint,
    save_manifest,
    verify_examples,
)
from rag_evidence.errors import ConfigError, DataError, FingerprintMismatchError

SIZES = {"smoke": 1, "dev": 1, "eval": 1}
DSINFO = {"hf_path": "x", "hf_config": "y", "hf_split": "z", "hf_revision": None}


def _raws(fixtures_dir: Path) -> list[dict]:
    rows = json.loads((fixtures_dir / "tiny_hotpot.json").read_text(encoding="utf-8"))["rows"]
    return [normalize_hf_example(r) for r in rows]


def _manifest(fixtures_dir: Path, seed: int = 7) -> dict:
    return build_manifest(
        _raws(fixtures_dir), seed=seed, sizes=SIZES, dataset_info=DSINFO, created_utc="t"
    )


def test_deterministic_and_disjoint(fixtures_dir: Path) -> None:
    m1, m2 = _manifest(fixtures_dir), _manifest(fixtures_dir)
    assert m1["splits"] == m2["splits"]
    all_qids = [q for s in m1["splits"].values() for q in s["question_ids"]]
    assert len(all_qids) == len(set(all_qids)) == 3
    assert m1["splits"]["eval"]["locked"] is True
    assert m1["splits"]["smoke"]["locked"] is False
    assert _manifest(fixtures_dir, seed=8)["splits"] != m1["splits"]


def test_insufficient_examples_raises(fixtures_dir: Path) -> None:
    with pytest.raises(DataError, match="need"):
        build_manifest(
            _raws(fixtures_dir)[:2], seed=7, sizes=SIZES, dataset_info=DSINFO, created_utc="t"
        )


def test_verify_detects_content_drift(fixtures_dir: Path) -> None:
    raws = _raws(fixtures_dir)
    manifest = _manifest(fixtures_dir)
    by_qid = {r["question_id"]: r for r in raws}
    verify_examples(by_qid, manifest, "eval")  # clean pass

    drifted_qid = manifest["splits"]["eval"]["question_ids"][0]
    tampered = json.loads(json.dumps(by_qid[drifted_qid]))
    tampered["context"][0][1][0] = "Edited sentence."
    with pytest.raises(FingerprintMismatchError):
        verify_examples({**by_qid, drifted_qid: tampered}, manifest, "eval")


def test_fingerprint_sensitive_to_answer(fixtures_dir: Path) -> None:
    raw = _raws(fixtures_dir)[0]
    tampered = json.loads(json.dumps(raw))
    tampered["answer"] = "different"
    assert example_fingerprint(raw) != example_fingerprint(tampered)


def test_save_manifest_refuses_overwrite(fixtures_dir: Path, tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    save_manifest(path, _manifest(fixtures_dir))
    with pytest.raises(DataError, match="never regenerated"):
        save_manifest(path, _manifest(fixtures_dir))


def test_manifest_config_consistency(fixtures_dir: Path) -> None:
    manifest = _manifest(fixtures_dir)
    check_manifest_matches_config(manifest, seed=7, sizes=SIZES)
    with pytest.raises(ConfigError, match="seed"):
        check_manifest_matches_config(manifest, seed=99, sizes=SIZES)
    with pytest.raises(ConfigError, match="split_sizes"):
        check_manifest_matches_config(manifest, seed=7, sizes={**SIZES, "dev": 5})
