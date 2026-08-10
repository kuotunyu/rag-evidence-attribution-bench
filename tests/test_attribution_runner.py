"""Attribution stage runner: all methods + controls over the tiny env with FakeLM."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rag_evidence.attribution.run import run_attribution_stage
from rag_evidence.config import AppConfig
from rag_evidence.generation.run import run_generation_stage
from rag_evidence.retrieval.run import run_retrieval_stage
from rag_evidence.storage.artifacts import read_json, read_records

METHODS_ALL_MODES = [
    "leave_one_out",
    "embedding",
    "embedding_question",
    "embedding_answer",
    "embedding_question_answer",
    "control_random",
    "control_retrieval",
    "control_lexical",
    "control_lexical_question",
    "control_lexical_answer",
    "control_lexical_question_answer",
    "control_length",
    "oracle_gold",
    "control_answer_string",
    "control_shuffled",  # must run after leave_one_out (its source)
]


class _StubEmbedder:
    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(32, dtype=np.float32)
        for tok in text.lower().split():
            v[hash(tok) % 32] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_queries(self, texts, batch_size: int = 1):
        return np.stack([self._vec(t) for t in texts])

    def embed_passages(self, texts, batch_size: int = 1):
        return np.stack([self._vec(t) for t in texts])


@pytest.fixture()
def attributed_env(tiny_env: AppConfig, monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    from rag_evidence import embeddings as emb_mod

    monkeypatch.setattr(emb_mod.Embedder, "load", classmethod(lambda cls, *a, **k: _StubEmbedder()))
    cfg = tiny_env
    run_retrieval_stage(cfg, method="bm25", resume=False, limit=None)
    run_generation_stage(cfg, resume=False, limit=None)
    for method in [*METHODS_ALL_MODES, "citations"]:
        run_attribution_stage(cfg, method=method, mode=None, resume=False, limit=None)
    return cfg


def _records(cfg: AppConfig, mode: str, method: str) -> list[dict]:
    path = Path(cfg.paths.results_raw) / cfg.split / "attribute" / mode / method / "records.jsonl"
    return list(read_records(path)) if path.exists() else []


def test_all_methods_produce_records_in_both_modes(attributed_env: AppConfig) -> None:
    cfg = attributed_env
    for method in METHODS_ALL_MODES:
        for mode in ("gold", "generated"):
            records = _records(cfg, mode, method)
            assert len(records) == 3, (method, mode)
            ok = [r for r in records if not r["skipped"] and r["error"] is None]
            assert ok, (method, mode)
            for rec in ok:
                assert set(rec["scores"]) == set(rec["context_passage_ids"])
                assert set(rec["raw_scores"]) == set(rec["context_passage_ids"])
                assert all(0.0 <= v <= 1.0 for v in rec["scores"].values())
                # FakeLM generator is available → faithfulness must be computed
                assert rec["faithfulness"] is not None, (method, mode)
                assert rec["faithfulness"]["k"] == cfg.attribution.faithfulness.k


def test_citations_gold_mode_skipped_generated_mode_works(attributed_env: AppConfig) -> None:
    cfg = attributed_env
    assert _records(cfg, "gold", "citations") == []  # whole mode skipped, no dir
    records = _records(cfg, "generated", "citations")
    assert len(records) == 3
    for rec in records:
        if not rec["skipped"]:
            assert set(rec["raw_scores"].values()) <= {0.0, 1.0}


def test_citations_gold_mode_explicit_request_errors(attributed_env: AppConfig) -> None:
    from rag_evidence.errors import ConfigError

    with pytest.raises(ConfigError, match="teacher-forced"):
        run_attribution_stage(
            attributed_env, method="citations", mode="gold", resume=False, limit=None
        )


def test_loo_metadata_and_model_calls(attributed_env: AppConfig) -> None:
    records = [r for r in _records(attributed_env, "gold", "leave_one_out") if not r["skipped"]]
    for rec in records:
        n = len(rec["context_passage_ids"])
        assert len(rec["metadata"]["ablations"]) == n
        assert rec["metadata"]["s_full"]["num_target_tokens"] >= 1
        assert rec["num_model_calls"] >= 1  # cache may absorb repeats, never zero fresh work


def test_shuffled_control_permutes_loo_scores(attributed_env: AppConfig) -> None:
    cfg = attributed_env
    loo = {r["question_id"]: r for r in _records(cfg, "gold", "leave_one_out")}
    shuffled = {r["question_id"]: r for r in _records(cfg, "gold", "control_shuffled")}
    for qid, rec in shuffled.items():
        if rec["skipped"]:
            continue
        assert sorted(rec["raw_scores"].values()) == pytest.approx(
            sorted(loo[qid]["raw_scores"].values())
        )


def test_execution_kind_is_mock_with_fake_backend(attributed_env: AppConfig) -> None:
    meta = read_json(
        Path(attributed_env.paths.results_raw)
        / attributed_env.split
        / "attribute"
        / "gold"
        / "leave_one_out"
        / "run_meta.json"
    )
    assert meta["execution_kind"] == "mock"


def test_run_metadata_preserves_method_control_role(attributed_env: AppConfig) -> None:
    base = Path(attributed_env.paths.results_raw) / attributed_env.split / "attribute" / "gold"
    oracle = read_json(base / "oracle_gold" / "run_meta.json")
    embedding = read_json(base / "embedding" / "run_meta.json")

    assert oracle["config_scientific"]["is_control"] is True
    assert embedding["config_scientific"]["is_control"] is False
