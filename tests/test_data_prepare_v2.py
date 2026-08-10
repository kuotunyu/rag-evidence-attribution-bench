"""Pinned dataset-v2 preparation writes only versioned, validated artifacts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rag_evidence.config import AppConfig, load_config
from rag_evidence.data.hotpot import prepare_data, verify_resolved_dataset_revision
from rag_evidence.data.manifest_v2 import validate_manifest_v2
from rag_evidence.errors import DataError
from rag_evidence.storage.artifacts import read_json

REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"


def _raw(qid: str, *, qtype: str) -> dict[str, Any]:
    title = f"Title {qid}"
    return {
        "question_id": qid,
        "question": f"Question {qid}?",
        "answer": f"Answer {qid}",
        "type": qtype,
        "level": "medium",
        "context": [[title, [f"Evidence for {qid}."]]],
        "supporting_facts": [[title, 0]],
    }


def _raws() -> list[dict[str, Any]]:
    return [
        _raw("q-a", qtype="bridge"),
        _raw("q-b", qtype="comparison"),
        _raw("q-c", qtype="bridge"),
        _raw("q-d", qtype="comparison"),
        _raw("q-e", qtype="bridge"),
        _raw("q-f", qtype="comparison"),
    ]


def _config(tmp_path: Path) -> AppConfig:
    path = tmp_path / "v2.yaml"
    path.write_text(
        f"""\
run_name: v2-test
split: eval
seed: 7
data:
  hf_path: hotpotqa/hotpot_qa
  hf_config: distractor
  hf_split: validation
  hf_revision: {REVISION}
  manifest_schema_version: 2
  manifest_path: data/manifests/split_manifest_v2.json
  prepared_dir: data/v2/prepared
  split_seed: 19
  split_sizes: {{smoke: 1, dev: 1, eval: 2}}
paths:
  data_dir: data
  results_raw: results/v2/raw
  results_derived: results/v2/derived
  assets_dir: results/v2/assets
""",
        encoding="utf-8",
    )
    return load_config(path)


def _artifact_snapshot() -> dict[str, bytes]:
    roots = (Path("data/manifests"), Path("data/v2"), Path("results/v2"))
    return {
        path.as_posix(): path.read_bytes()
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_resolved_dataset_revision_returns_an_exact_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    class FakeApi:
        def dataset_info(self, *, repo_id: str, revision: str) -> SimpleNamespace:
            return SimpleNamespace(id=repo_id, sha=revision)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)

    assert verify_resolved_dataset_revision("hotpotqa/hotpot_qa", REVISION) == REVISION


def test_resolved_dataset_revision_rejects_a_different_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    class FakeApi:
        def dataset_info(self, *, repo_id: str, revision: str) -> SimpleNamespace:
            return SimpleNamespace(id=repo_id, sha="f" * 40, requested=revision)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)

    with pytest.raises(DataError, match="revision mismatch"):
        verify_resolved_dataset_revision("hotpotqa/hotpot_qa", REVISION)


def test_prepare_v2_verifies_revision_then_writes_idempotent_versioned_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rag_evidence.data import hotpot

    monkeypatch.chdir(tmp_path)
    cfg = _config(tmp_path)
    events: list[str] = []

    def resolve(_repo_id: str, _requested: str) -> str:
        events.append("resolve")
        return REVISION

    def load(_cfg: AppConfig) -> list[dict[str, Any]]:
        events.append("load")
        return _raws()

    monkeypatch.setattr(hotpot, "verify_resolved_dataset_revision", resolve)
    monkeypatch.setattr(hotpot, "load_raw_examples", load)

    prepare_data(cfg)

    assert events[:2] == ["resolve", "load"]
    manifest = read_json(cfg.manifest_file)
    validate_manifest_v2(manifest, _raws())
    assert manifest["dataset"]["resolved_revision"] == REVISION
    for split in ("smoke", "dev", "eval"):
        assert (Path("data/v2/prepared") / f"{split}.jsonl").exists()
        assert (Path("results/v2/raw") / split / "samples/records.jsonl").exists()
    assert not Path("data/prepared").exists()
    assert not Path("results/raw").exists()
    first = _artifact_snapshot()

    prepare_data(cfg)

    assert _artifact_snapshot() == first


def test_prepare_v2_rejects_revision_mismatch_before_loading_or_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rag_evidence.data import hotpot

    monkeypatch.chdir(tmp_path)
    cfg = _config(tmp_path)
    monkeypatch.setattr(
        hotpot,
        "verify_resolved_dataset_revision",
        lambda _repo_id, _requested: (_ for _ in ()).throw(DataError("revision mismatch")),
    )

    def unexpected_load(_cfg: AppConfig) -> list[dict[str, Any]]:
        raise AssertionError("raw data loaded before revision verification")

    monkeypatch.setattr(hotpot, "load_raw_examples", unexpected_load)

    with pytest.raises(DataError, match="revision mismatch"):
        prepare_data(cfg)

    assert not Path("data/manifests/split_manifest_v2.json").exists()
    assert not Path("data/v2").exists()
    assert not Path("results/v2").exists()
