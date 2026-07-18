"""Shared fixtures. The offline guard runs for EVERY test: CI and local test runs must
never touch the Hugging Face Hub — all tests use synthetic fixtures and fake backends."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _offline_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))
    # env overrides must never leak from the developer's shell into tests
    for var in (
        "RAG_EVIDENCE_HOST",
        "RAG_EVIDENCE_PORT",
        "RAG_EVIDENCE_RESULTS_RAW",
        "RAG_EVIDENCE_RESULTS_DERIVED",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture()
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture()
def repo_root() -> Path:
    return Path(__file__).parent.parent


TINY_CONFIG_YAML = """\
run_name: tiny
split: smoke
seed: 7
data:
  split_seed: 7
  split_sizes: {{smoke: 3, dev: 1, eval: 1}}
generation:
  backend: fake
  name: fake
attribution:
  controls:
    retrieval_run: bm25
    shuffled_source: leave_one_out
{extra}
"""


@pytest.fixture()
def tiny_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A complete tiny repo layout under tmp_path (cwd is switched there):
    committed-style manifest + prepared JSONL built from the synthetic fixture,
    and a config using the FakeLM backend. Returns the loaded AppConfig.
    """
    import json as _json

    from rag_evidence.config import load_config
    from rag_evidence.data.hotpot import build_example, normalize_hf_example
    from rag_evidence.data.splits import build_manifest, save_manifest
    from rag_evidence.storage.artifacts import append_record

    monkeypatch.chdir(tmp_path)
    rows = _json.loads((FIXTURES_DIR / "tiny_hotpot.json").read_text(encoding="utf-8"))["rows"]
    raws = [normalize_hf_example(r) for r in rows]
    manifest = build_manifest(
        raws,
        seed=7,
        sizes={"smoke": 3, "dev": 1, "eval": 1},
        dataset_info={
            "hf_path": "synthetic",
            "hf_config": "tiny",
            "hf_split": "test",
            "hf_revision": None,
        },
        created_utc="2026-01-01T00:00:00Z",
    )
    (tmp_path / "data" / "manifests").mkdir(parents=True)
    save_manifest(tmp_path / "data" / "manifests" / "split_manifest.json", manifest)

    raws_by_qid = {r["question_id"]: r for r in raws}
    prepared_dir = tmp_path / "data" / "prepared"
    prepared_dir.mkdir(parents=True)
    for split_name, info in manifest["splits"].items():
        out = prepared_dir / f"{split_name}.jsonl"
        samples = tmp_path / "results" / "raw" / split_name / "samples" / "records.jsonl"
        for qid in info["question_ids"]:
            record = build_example(raws_by_qid[qid]).to_json()
            record["raw_fingerprint"] = manifest["example_hashes"][qid]
            append_record(out, record)
            append_record(samples, record)

    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(TINY_CONFIG_YAML.format(extra=""), encoding="utf-8")
    return load_config(config_path)
