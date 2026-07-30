"""Cross-encoder adapter boundary, score cache, rerank runner, and arm isolation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rag_evidence.config import (
    AppConfig,
    RerankerConfig,
    RerankingExperimentConfig,
)
from rag_evidence.errors import ArtifactError, ConfigError
from rag_evidence.evaluation.evaluate import evaluate_all
from rag_evidence.generation.run import run_generation_stage
from rag_evidence.reranking.experiment import arm_config
from rag_evidence.reranking.report import build_reranking_comparison
from rag_evidence.retrieval.run import run_retrieval_stage
from rag_evidence.storage.artifacts import append_record, read_json, read_records


class _StubEmbedder:
    def _vec(self, text: str) -> np.ndarray:
        vec = np.zeros(64, dtype=np.float32)
        for token in text.lower().split():
            vec[hash(token) % len(vec)] += 1
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def embed_queries(self, texts, batch_size: int = 1):
        return np.stack([self._vec(text) for text in texts])

    def embed_passages(self, texts, batch_size: int = 1):
        return np.stack([self._vec(text) for text in texts])


class _StubReranker:
    def score_pairs(self, pairs):
        # Deterministic cross-pair interaction surrogate; descending output is easy to verify.
        return [float(len(query) + 2 * len(passage)) for query, passage in pairs]

    def info(self) -> dict[str, Any]:
        return {
            "adapter": "stub",
            "model_id": "stub",
            "model_revision": "0" * 40,
            "tokenizer_id": "stub",
            "tokenizer_revision": "0" * 40,
            "max_length": 512,
            "device": "cpu",
            "dtype_requested": "float32",
            "dtype_effective": "float32",
            "batch_size": 16,
        }


def _expand_prepared_to_ten(cfg: AppConfig) -> None:
    path = cfg.prepared_file
    records = list(read_records(path))
    path.unlink()
    for record in records:
        qid = record["question_id"]
        for index in range(len(record["passages"]), 10):
            record["passages"].append(
                {
                    "passage_id": f"{qid}-p{index:02d}",
                    "index": index,
                    "title": f"Synthetic distractor {index}",
                    "sentences": [f"Unrelated deterministic passage number {index}."],
                    "is_gold": False,
                }
            )
        append_record(path, record)


@pytest.fixture()
def reranking_env(tiny_env: AppConfig, monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    from rag_evidence import embeddings as embeddings_module
    from rag_evidence.retrieval import reranker as reranker_module

    _expand_prepared_to_ten(tiny_env)
    monkeypatch.setattr(
        embeddings_module.Embedder,
        "load",
        classmethod(lambda cls, *args, **kwargs: _StubEmbedder()),
    )
    run_retrieval_stage(tiny_env, method="bm25", resume=False, limit=None)
    run_retrieval_stage(tiny_env, method="dense", resume=False, limit=None)
    run_retrieval_stage(tiny_env, method="hybrid_rrf", resume=False, limit=None)

    prereg = Path("prereg.md")
    prereg.write_text("locked synthetic preregistration\n", encoding="utf-8")
    prereg_hash = hashlib.sha256(prereg.read_bytes()).hexdigest()
    secondary = Path("secondary.md")
    secondary.write_text("locked synthetic secondary analysis\n", encoding="utf-8")
    secondary_hash = hashlib.sha256(secondary.read_bytes()).hexdigest()
    paths = tiny_env.paths.model_copy(
        update={
            "results_raw": "results/reranking/raw",
            "results_derived": "results/reranking/derived",
        }
    )
    experiment = RerankingExperimentConfig(
        enabled=True,
        preregistration_path=str(prereg),
        preregistration_sha256=prereg_hash,
        secondary_analysis_path=str(secondary),
        secondary_analysis_sha256=secondary_hash,
        baseline_results_raw="results/raw",
        reranker=RerankerConfig(),
    )
    cfg = tiny_env.model_copy(update={"paths": paths, "reranking": experiment})
    monkeypatch.setattr(reranker_module, "build_reranker", lambda _cfg: _StubReranker())
    return cfg


def test_rerank_runner_orders_all_candidates_and_records_systems(
    reranking_env: AppConfig,
) -> None:
    cfg = reranking_env
    run_retrieval_stage(cfg, method="hybrid_rrf_rerank", resume=False, limit=None)
    run_dir = Path(cfg.paths.results_raw) / "smoke/retrieve/hybrid_rrf_rerank"
    records = list(read_records(run_dir / "records.jsonl"))
    assert len(records) == 3
    for record in records:
        assert record["error"] is None
        assert len(record["ranking"]) == 10
        assert record["candidate_count"] == 10
        assert record["cache_hits"] == 0
        assert record["cache_misses"] == 10
        scores = [item["rerank_score"] for item in record["ranking"]]
        assert scores == sorted(scores, reverse=True)
        assert record["estimated_end_to_end_latency_ms"] >= record["rerank_latency_ms"]

    meta = read_json(run_dir / "run_meta.json")
    assert meta["config_scientific"]["preregistration_sha256"]
    assert meta["reranker_systems"]["cache_misses"] == 30
    assert meta["run_peak_vram_mb"] is None

    # A no-op resume must preserve cumulative cache/system accounting.
    run_retrieval_stage(cfg, method="hybrid_rrf_rerank", resume=True, limit=None)
    resumed = read_json(run_dir / "run_meta.json")
    assert resumed["reranker_systems"]["cache_misses"] == 30
    assert len(resumed["reranker_systems"]["model_load_s_sessions"]) == 2


def test_score_cache_roundtrip_and_identity_guard(tmp_path: Path) -> None:
    from rag_evidence.retrieval.reranker import RerankScoreCache

    path = tmp_path / "scores.jsonl"
    identity = {"model": "m", "revision": "a"}
    cache = RerankScoreCache(path, identity)
    assert cache.get("key") is None
    cache.put("key", 1.25, question_id="q", passage_id="p")
    assert RerankScoreCache(path, identity).get("key") == pytest.approx(1.25)
    with pytest.raises(ArtifactError, match="identity mismatch"):
        RerankScoreCache(path, {"model": "different"})


def test_preregistration_drift_is_refused(reranking_env: AppConfig) -> None:
    Path(reranking_env.reranking.preregistration_path).write_text(
        "changed after lock\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="preregistration hash mismatch"):
        run_retrieval_stage(
            reranking_env,
            method="hybrid_rrf_rerank",
            resume=False,
            limit=1,
        )


def test_secondary_analysis_drift_is_refused(
    reranking_env: AppConfig,
) -> None:
    Path(reranking_env.reranking.secondary_analysis_path).write_text(
        "changed after lock\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="secondary analysis plan hash mismatch"):
        build_reranking_comparison(reranking_env)


def test_arm_config_changes_only_context_selection_and_namespace(
    reranking_env: AppConfig,
) -> None:
    cfg = arm_config(reranking_env, "hybrid_rrf_rerank")
    assert cfg.generation.context_source == "retrieval"
    assert cfg.generation.retrieval_run == "hybrid_rrf_rerank"
    assert cfg.generation.top_k_context == 5
    assert cfg.attribution.run_namespace == "hybrid_rrf_rerank"
    assert cfg.attribution.gold_context_source == "generation"
    assert cfg.generation.model_id == reranking_env.generation.model_id
    assert cfg.generation.max_new_tokens == reranking_env.generation.max_new_tokens


def test_four_arm_downstream_comparison_is_machine_generated(
    reranking_env: AppConfig,
) -> None:
    from rag_evidence.attribution.run import run_attribution_stage

    cfg = reranking_env
    run_retrieval_stage(cfg, method="hybrid_rrf_rerank", resume=False, limit=None)
    for arm in cfg.reranking.arms:
        arm_cfg = arm_config(cfg, arm)
        run_generation_stage(arm_cfg, resume=False, limit=None)
        for method in ("leave_one_out", "embedding", "citations", "control_random"):
            run_attribution_stage(arm_cfg, method=method, mode=None, resume=False, limit=None)
    evaluate_all(cfg)
    comparison = build_reranking_comparison(cfg)
    assert comparison["schema_version"] == 2
    assert comparison["secondary_analysis"]["sha256"] == (cfg.reranking.secondary_analysis_sha256)
    assert set(comparison["retrieval"]) == set(cfg.reranking.arms)
    assert all(comparison["generation"][arm] for arm in cfg.reranking.arms)
    assert set(comparison["attribution"]["generated"]) == set(cfg.reranking.arms)
    assert "leave_one_out" in comparison["attribution_stability_top2_jaccard"]
    assert comparison["systems"]["hybrid_rrf_rerank"]["retrieval_end_to_end_latency_ms"]
    assert comparison["incremental_cost_vs_hybrid_rrf"]["retrieval_latency_ms"]
    assert comparison["incremental_cost_vs_hybrid_rrf"]["local_api_fee"] == 0.0
    assert comparison["paired_bootstrap"]["retrieval_ndcg_at_5"]["n_pairs"] == 3
    assert sum(comparison["primary_transfer_taxonomy_counts"].values()) == 3
    assert all(
        sum(counts.values()) == 3
        for counts in comparison["transfer_taxonomy_counts_by_method"].values()
    )
    cache_observations = comparison["systems"]["hybrid_rrf_rerank"]["cache_observations"]
    assert cache_observations["cold"]["n"] == 3
    assert cache_observations["warm"]["n"] == 0
    assert "generated_sufficiency" in comparison["deltas_vs_hybrid_rrf"]
    assert "generated_comprehensiveness" in comparison["deltas_vs_hybrid_rrf"]
    split_dir = Path(cfg.paths.results_derived) / cfg.split
    assert (split_dir / "reranking_comparison.json").exists()
    assert len(list(read_records(split_dir / "reranking_error_analysis.jsonl"))) == 3
    assert comparison["decision"]["formal"] is False
    assert comparison["decision"]["complete_generation"] is True
    assert comparison["decision"]["complete_attribution"] is False
    assert comparison["decision"]["quality_gate"] is None
