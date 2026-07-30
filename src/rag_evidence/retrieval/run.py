"""The `retrieve` stage runner: per-question ranking with checkpoint/resume."""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

from rag_evidence.config import AppConfig
from rag_evidence.data.hotpot import load_prepared_verified
from rag_evidence.data.schema import Example
from rag_evidence.errors import ConfigError, UpstreamMissingError
from rag_evidence.retrieval.base import Retriever
from rag_evidence.storage.artifacts import (
    RECORDS_FILE,
    append_record,
    completed_keys,
    read_records,
    stage_dir,
)
from rag_evidence.storage.runmeta import (
    check_dtype_compatible,
    finalize_run,
    start_or_resume_run,
    update_model_info,
)
from rag_evidence.telemetry import SampleTimer, peak_vram_mb, reset_peak_vram

logger = logging.getLogger(__name__)


def _load_stored_records(results_raw: Path, split: str, method: str) -> dict[str, dict[str, Any]]:
    path = stage_dir(results_raw, split, "retrieve", method) / RECORDS_FILE
    if not path.exists():
        raise UpstreamMissingError(f"stored retrieval run {method!r} is missing ({path})")
    return {str(rec["question_id"]): rec for rec in read_records(path) if rec.get("error") is None}


def _load_stored_rankings(
    cfg: AppConfig, method: str, *, results_raw: Path | None = None
) -> dict[str, dict[str, int]]:
    """{question_id: {passage_id: rank}} from a completed retrieval run."""
    root = results_raw or cfg.results_raw_dir
    path = stage_dir(root, cfg.split, "retrieve", method) / RECORDS_FILE
    if not path.exists():
        raise UpstreamMissingError(
            f"hybrid_rrf needs a completed `retrieve --method {method}` run first ({path} missing)"
        )
    out: dict[str, dict[str, int]] = {}
    for rec in read_records(path):
        if rec.get("error") is None:
            out[rec["question_id"]] = {r["passage_id"]: r["rank"] for r in rec["ranking"]}
    return out


def _verified_preregistration(cfg: AppConfig) -> str:
    path = Path(cfg.reranking.preregistration_path)
    if not path.exists():
        raise ConfigError(f"reranking preregistration file is missing: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = cfg.reranking.preregistration_sha256
    if actual != expected:
        raise ConfigError(
            f"reranking preregistration hash mismatch for {path}: expected {expected}, "
            f"got {actual}; the locked design may have changed"
        )
    return actual


def _build_retriever(cfg: AppConfig, method: str) -> Retriever:
    if method == "bm25":
        from rag_evidence.retrieval.bm25 import BM25Retriever

        return BM25Retriever(k1=cfg.retrieval.bm25.k1, b=cfg.retrieval.bm25.b)
    if method == "dense":
        from rag_evidence.embeddings import Embedder, EmbeddingCache, model_tag
        from rag_evidence.retrieval.dense import DenseRetriever

        embedder = Embedder.load(
            cfg.retrieval.dense.model_id,
            device=cfg.runtime.device,
            max_length=cfg.retrieval.dense.max_length,
            normalize=cfg.retrieval.dense.normalize,
        )
        cache = EmbeddingCache(
            cfg.results_raw_dir
            / "cache"
            / "embeddings"
            / model_tag(cfg.retrieval.dense.model_id)
            / f"{cfg.split}_passages.npz"
        )
        return DenseRetriever(embedder, cache, batch_size=cfg.retrieval.dense.batch_size)
    raise ValueError(f"unknown retrieval method {method!r}")


def run_retrieval_stage(cfg: AppConfig, *, method: str, resume: bool, limit: int | None) -> None:
    examples: list[Example] = load_prepared_verified(cfg)
    if limit is not None:
        examples = examples[:limit]

    prereg_sha256: str | None = None
    if method == "hybrid_rrf_rerank":
        if not cfg.reranking.enabled:
            raise ConfigError(
                "hybrid_rrf_rerank requires reranking.enabled=true in an extension config"
            )
        prereg_sha256 = _verified_preregistration(cfg)

    run_dir = stage_dir(cfg.results_raw_dir, cfg.split, "retrieve", method)
    meta = start_or_resume_run(
        run_dir,
        cfg,
        stage="retrieve",
        name=method,
        resume=resume,
        execution_kind="real",
        expected_count=len(examples),
        limit=limit,
        sci_extra={
            "method": method,
            **(
                {
                    "experiment_id": cfg.reranking.experiment_id,
                    "preregistration_sha256": prereg_sha256,
                }
                if method == "hybrid_rrf_rerank"
                else {}
            ),
        },
    )
    records_path = run_dir / RECORDS_FILE
    done = completed_keys(records_path) if resume else set()

    sources: dict[str, dict[str, dict[str, int]]] = {}
    source_records: dict[str, dict[str, dict[str, Any]]] = {}
    retriever = None
    reranker = None
    rerank_cache = None
    model_load_s: float | None = None
    if method == "hybrid_rrf":
        sources = {m: _load_stored_rankings(cfg, m) for m in ("bm25", "dense")}
    elif method == "hybrid_rrf_rerank":
        from rag_evidence.embeddings import model_tag
        from rag_evidence.retrieval.reranker import (
            RerankScoreCache,
            build_reranker,
            cache_identity,
        )

        baseline_raw = Path(cfg.reranking.baseline_results_raw)
        source_records = {
            source: _load_stored_records(baseline_raw, cfg.split, source)
            for source in ("bm25", "dense", "hybrid_rrf")
        }
        reset_peak_vram()
        load_start = time.perf_counter()
        reranker = build_reranker(cfg.reranking.reranker)
        model_load_s = time.perf_counter() - load_start
        check_dtype_compatible(meta, reranker.info(), run_dir)
        update_model_info(run_dir, meta, reranker.info())
        cache_path = None
        if cfg.reranking.reranker.cache_enabled:
            rcfg = cfg.reranking.reranker
            cache_path = (
                cfg.results_raw_dir
                / "cache"
                / "reranker"
                / model_tag(rcfg.model_id)
                / rcfg.model_revision
                / f"{cfg.split}.jsonl"
            )
        rerank_cache = RerankScoreCache(cache_path, cache_identity(cfg.reranking.reranker))
    else:
        retriever = _build_retriever(cfg, method)

    wall_start = time.perf_counter()
    n_success = n_failed = n_skipped = 0
    total_cache_hits = total_cache_misses = total_pairs = 0
    total_scoring_s = 0.0
    for example in examples:
        if (example.question_id,) in done:
            n_skipped += 1
            continue
        record: dict[str, object] = {"question_id": example.question_id, "method": method}
        ranking_json: list[dict[str, Any]]
        try:
            with SampleTimer() as timer:
                if method == "hybrid_rrf":
                    from rag_evidence.retrieval.hybrid import rrf_fuse

                    order = [p.passage_id for p in example.passages]
                    per_source = []
                    for source_name, ranks in sources.items():
                        if example.question_id not in ranks:
                            raise UpstreamMissingError(
                                f"{source_name} run has no record for {example.question_id}"
                            )
                        per_source.append(ranks[example.question_id])
                    ranking = rrf_fuse(per_source, rrf_k=cfg.retrieval.rrf_k, original_order=order)
                    ranking_json = [r.to_json() for r in ranking]
                elif method == "hybrid_rrf_rerank":
                    from rag_evidence.retrieval.dense import passage_embed_text
                    from rag_evidence.retrieval.reranker import score_cache_key

                    assert reranker is not None and rerank_cache is not None
                    hybrid_rec = source_records["hybrid_rrf"].get(example.question_id)
                    if hybrid_rec is None:
                        raise UpstreamMissingError(
                            f"hybrid_rrf run has no record for {example.question_id}"
                        )
                    hybrid_order = [item["passage_id"] for item in hybrid_rec["ranking"]]
                    candidate_k = cfg.reranking.reranker.candidate_k
                    if candidate_k > len(hybrid_order):
                        raise ConfigError(
                            f"reranker candidate_k={candidate_k} exceeds the question's "
                            f"{len(hybrid_order)}-passage candidate corpus"
                        )
                    candidates = hybrid_order[:candidate_k]
                    texts = {
                        pid: passage_embed_text(
                            example.passage_by_id(pid).title, example.passage_by_id(pid).text
                        )
                        for pid in candidates
                    }
                    identity = rerank_cache.identity
                    cached_scores: dict[str, float] = {}
                    missing: list[str] = []
                    keys: dict[str, str] = {}
                    for pid in candidates:
                        key = score_cache_key(
                            identity,
                            question=example.question,
                            passage_id=pid,
                            passage_text=texts[pid],
                        )
                        keys[pid] = key
                        hit = rerank_cache.get(key)
                        if hit is None:
                            missing.append(pid)
                        else:
                            cached_scores[pid] = hit

                    with SampleTimer() as scoring_timer:
                        fresh = reranker.score_pairs(
                            [(example.question, texts[pid]) for pid in missing]
                        )
                    if len(fresh) != len(missing):
                        raise ValueError("reranker adapter returned the wrong number of scores")
                    for pid, score in zip(missing, fresh, strict=True):
                        cached_scores[pid] = score
                        rerank_cache.put(
                            keys[pid],
                            score,
                            question_id=example.question_id,
                            passage_id=pid,
                        )
                    ordered_candidates = sorted(
                        candidates,
                        key=lambda pid: (-cached_scores[pid], hybrid_order.index(pid)),
                    )
                    tail = hybrid_order[candidate_k:]
                    final_order = [*ordered_candidates, *tail]
                    floor = min(cached_scores.values()) if cached_scores else 0.0
                    ranking_json = []
                    for rank, pid in enumerate(final_order, start=1):
                        was_reranked = pid in cached_scores
                        ranking_json.append(
                            {
                                "passage_id": pid,
                                "score": (
                                    float(cached_scores[pid])
                                    if was_reranked
                                    else float(floor - 1.0 - hybrid_order.index(pid))
                                ),
                                "rank": rank,
                                "source_rank": hybrid_order.index(pid) + 1,
                                "rerank_score": (
                                    float(cached_scores[pid]) if was_reranked else None
                                ),
                                "was_reranked": was_reranked,
                            }
                        )
                    record["candidate_count"] = candidate_k
                    record["cache_hits"] = candidate_k - len(missing)
                    record["cache_misses"] = len(missing)
                    record["scored_pair_count"] = len(missing)
                    record["scoring_latency_ms"] = round(scoring_timer.elapsed_ms, 3)
                    source_latency = {
                        source: float(source_records[source][example.question_id]["latency_ms"])
                        for source in ("bm25", "dense", "hybrid_rrf")
                    }
                    first_stage_parallel = (
                        max(source_latency["bm25"], source_latency["dense"])
                        + source_latency["hybrid_rrf"]
                    )
                    record["first_stage_latency_ms"] = source_latency
                    record["first_stage_parallel_estimate_ms"] = round(first_stage_parallel, 3)
                    total_cache_hits += candidate_k - len(missing)
                    total_cache_misses += len(missing)
                    total_pairs += candidate_k
                    total_scoring_s += scoring_timer.elapsed_s
                else:
                    assert retriever is not None
                    ranking = retriever.rank(example)
                    ranking_json = [r.to_json() for r in ranking]
            record["ranking"] = ranking_json
            record["latency_ms"] = round(timer.elapsed_ms, 3)
            if method == "hybrid_rrf_rerank":
                record["rerank_latency_ms"] = record["latency_ms"]
                first_stage_estimate = record["first_stage_parallel_estimate_ms"]
                assert isinstance(first_stage_estimate, int | float)
                record["estimated_end_to_end_latency_ms"] = round(
                    float(first_stage_estimate) + timer.elapsed_ms, 3
                )
                record["peak_vram_mb"] = peak_vram_mb()
            record["error"] = None
            n_success += 1
        except Exception as exc:  # per-sample failure: record and continue
            logger.exception("retrieval failed for %s", example.question_id)
            record.update(
                ranking=None,
                latency_ms=None,
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            if method == "hybrid_rrf_rerank":
                record["rerank_latency_ms"] = None
            n_failed += 1
        append_record(records_path, record)

    if method == "hybrid_rrf_rerank":
        previous_systems = meta.get("reranker_systems") or {}
        cumulative_hits = int(previous_systems.get("cache_hits", 0)) + total_cache_hits
        cumulative_misses = int(previous_systems.get("cache_misses", 0)) + total_cache_misses
        cumulative_pairs = int(previous_systems.get("pairs_seen", 0)) + total_pairs
        cumulative_scored = int(previous_systems.get("pairs_scored", 0)) + total_cache_misses
        cumulative_scoring_s = float(previous_systems.get("scoring_s", 0.0)) + total_scoring_s
        load_sessions = list(previous_systems.get("model_load_s_sessions", []))
        load_sessions.append(round(model_load_s or 0.0, 6))
        meta["reranker_systems"] = {
            "model_load_s": round(model_load_s or 0.0, 6),
            "model_load_s_sessions": load_sessions,
            "cache_hits": cumulative_hits,
            "cache_misses": cumulative_misses,
            "cache_hit_rate": (
                cumulative_hits / (cumulative_hits + cumulative_misses)
                if cumulative_hits + cumulative_misses
                else None
            ),
            "pairs_seen": cumulative_pairs,
            "pairs_scored": cumulative_scored,
            "scoring_s": cumulative_scoring_s,
            "scored_pairs_per_s": (
                cumulative_scored / cumulative_scoring_s
                if cumulative_scored and cumulative_scoring_s > 0
                else None
            ),
        }
    finalize_run(
        run_dir,
        meta,
        status="completed" if n_failed == 0 else "completed_with_failures",
        n_attempted=n_success + n_failed,
        n_success=n_success,
        n_failed=n_failed,
        n_skipped=n_skipped,
        run_peak_vram_mb=peak_vram_mb() if method == "hybrid_rrf_rerank" else None,
        total_wall_s=round(time.perf_counter() - wall_start, 3),
    )
    logger.info(
        "retrieve[%s] %s: %d ok, %d failed, %d resumed-skip -> %s",
        method,
        cfg.split,
        n_success,
        n_failed,
        n_skipped,
        run_dir,
    )
