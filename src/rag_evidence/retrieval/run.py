"""The `retrieve` stage runner: per-question ranking with checkpoint/resume."""

from __future__ import annotations

import logging
import time

from rag_evidence.config import AppConfig
from rag_evidence.data.hotpot import load_prepared_verified
from rag_evidence.data.schema import Example
from rag_evidence.errors import UpstreamMissingError
from rag_evidence.retrieval.base import Retriever
from rag_evidence.storage.artifacts import (
    RECORDS_FILE,
    append_record,
    completed_keys,
    read_records,
    stage_dir,
)
from rag_evidence.storage.runmeta import finalize_run, start_or_resume_run
from rag_evidence.telemetry import SampleTimer

logger = logging.getLogger(__name__)


def _load_stored_rankings(cfg: AppConfig, method: str) -> dict[str, dict[str, int]]:
    """{question_id: {passage_id: rank}} from a completed retrieval run."""
    path = stage_dir(cfg.results_raw_dir, cfg.split, "retrieve", method) / RECORDS_FILE
    if not path.exists():
        raise UpstreamMissingError(
            f"hybrid_rrf needs a completed `retrieve --method {method}` run first ({path} missing)"
        )
    out: dict[str, dict[str, int]] = {}
    for rec in read_records(path):
        if rec.get("error") is None:
            out[rec["question_id"]] = {r["passage_id"]: r["rank"] for r in rec["ranking"]}
    return out


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
        sci_extra={"method": method},
    )
    records_path = run_dir / RECORDS_FILE
    done = completed_keys(records_path) if resume else set()

    sources: dict[str, dict[str, dict[str, int]]] = {}
    retriever = None
    if method == "hybrid_rrf":
        sources = {m: _load_stored_rankings(cfg, m) for m in ("bm25", "dense")}
    else:
        retriever = _build_retriever(cfg, method)

    wall_start = time.perf_counter()
    n_success = n_failed = n_skipped = 0
    for example in examples:
        if (example.question_id,) in done:
            n_skipped += 1
            continue
        record: dict[str, object] = {"question_id": example.question_id, "method": method}
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
                else:
                    assert retriever is not None
                    ranking = retriever.rank(example)
            record["ranking"] = [r.to_json() for r in ranking]
            record["latency_ms"] = round(timer.elapsed_ms, 3)
            record["error"] = None
            n_success += 1
        except Exception as exc:  # per-sample failure: record and continue
            logger.exception("retrieval failed for %s", example.question_id)
            record.update(
                ranking=None,
                latency_ms=None,
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            n_failed += 1
        append_record(records_path, record)

    finalize_run(
        run_dir,
        meta,
        status="completed" if n_failed == 0 else "completed_with_failures",
        n_attempted=n_success + n_failed,
        n_success=n_success,
        n_failed=n_failed,
        n_skipped=n_skipped,
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
