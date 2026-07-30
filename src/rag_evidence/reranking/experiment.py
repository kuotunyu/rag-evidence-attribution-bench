"""Build one fair downstream arm from the shared reranking extension config."""

from __future__ import annotations

from pathlib import Path

from rag_evidence.config import AppConfig
from rag_evidence.errors import ConfigError


def arm_config(cfg: AppConfig, arm: str) -> AppConfig:
    """Return a frozen config for one arm without mutating the loaded base config."""
    if not cfg.reranking.enabled:
        raise ConfigError("reranking arm commands require reranking.enabled=true")
    if arm not in cfg.reranking.arms:
        raise ConfigError(f"arm {arm!r} is not preregistered in {cfg.reranking.arms}")

    retrieval_root = (
        cfg.results_raw_dir
        if arm == "hybrid_rrf_rerank"
        else Path(cfg.reranking.baseline_results_raw)
    )
    base_name = cfg.generation.name
    generation_name = (
        f"{base_name}-{cfg.reranking.experiment_id}-{arm}-k{cfg.reranking.final_context_k}"
    )
    generation = cfg.generation.model_copy(
        update={
            "name": generation_name,
            "context_source": "retrieval",
            "retrieval_run": arm,
            "retrieval_results_raw": str(retrieval_root).replace("\\", "/"),
            "top_k_context": cfg.reranking.final_context_k,
        }
    )
    controls = cfg.attribution.controls.model_copy(
        update={
            "retrieval_run": "bm25",
            "retrieval_results_raw": cfg.reranking.baseline_results_raw,
        }
    )
    attribution = cfg.attribution.model_copy(
        update={
            "controls": controls,
            "run_namespace": arm,
            "gold_context_source": "generation",
        }
    )
    return cfg.model_copy(update={"generation": generation, "attribution": attribution})
