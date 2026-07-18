"""The `generate` stage runner: cited answers, checkpoint/resume, VRAM/latency capture."""

from __future__ import annotations

import logging
import time
from typing import Literal

from rag_evidence.config import AppConfig, resolve_device, resolve_dtype
from rag_evidence.data import ids
from rag_evidence.data.hotpot import load_prepared_verified
from rag_evidence.data.schema import Example, Passage
from rag_evidence.errors import RagEvidenceError, UpstreamMissingError
from rag_evidence.generation.backends import build_backend
from rag_evidence.generation.citations import is_abstention, parse_citations, strip_citations
from rag_evidence.generation.prompts import build_messages, prompt_hash
from rag_evidence.metrics.text import exact_match, f1_score
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

_MAX_CONSECUTIVE_OOM = 3


def generation_run_name(cfg: AppConfig) -> str:
    if cfg.generation.backend == "fake":
        return "fake"
    quant = f"_{cfg.generation.quantization}" if cfg.generation.quantization != "none" else ""
    return f"{cfg.generation.name}{quant}"


def select_context(cfg: AppConfig, example: Example) -> list[Passage]:
    """Prompt context: full dataset order (default) or top-k of a stored retrieval run."""
    if cfg.generation.context_source == "dataset":
        return list(example.passages)
    method = cfg.generation.retrieval_run
    if not method:
        raise UpstreamMissingError(
            "generation.context_source=retrieval requires generation.retrieval_run"
        )
    path = stage_dir(cfg.results_raw_dir, cfg.split, "retrieve", method) / RECORDS_FILE
    if not path.exists():
        raise UpstreamMissingError(f"retrieval run {method} not found at {path}")
    ranking: list[str] | None = None
    for rec in read_records(path):
        if rec["question_id"] == example.question_id and rec.get("error") is None:
            ranking = [r["passage_id"] for r in rec["ranking"]]
            break
    if ranking is None:
        raise UpstreamMissingError(
            f"retrieval run {method} has no successful record for {example.question_id}"
        )
    top = ranking[: cfg.generation.top_k_context]
    return [example.passage_by_id(pid) for pid in top]


def _is_oom(exc: BaseException) -> bool:
    return "OutOfMemoryError" in type(exc).__name__


def run_generation_stage(cfg: AppConfig, *, resume: bool, limit: int | None) -> None:
    examples = load_prepared_verified(cfg)
    if limit is not None:
        examples = examples[:limit]

    name = generation_run_name(cfg)
    execution_kind: Literal["real", "mock"] = "mock" if cfg.generation.backend == "fake" else "real"
    run_dir = stage_dir(cfg.results_raw_dir, cfg.split, "generate", name)
    meta = start_or_resume_run(
        run_dir,
        cfg,
        stage="generate",
        name=name,
        resume=resume,
        execution_kind=execution_kind,
        expected_count=len(examples),
        limit=limit,
        sci_extra={"prompt_hash": prompt_hash(cfg.generation.prompt_version)},
    )
    records_path = run_dir / RECORDS_FILE
    done = completed_keys(records_path) if resume else set()

    device = resolve_device(cfg.runtime.device)
    dtype = resolve_dtype(cfg.generation.dtype, device)
    backend = build_backend(
        cfg.generation,
        device=device,
        dtype=dtype,
        seed=cfg.seed,
        explicit_cpu=cfg.runtime.device == "cpu",
    )
    check_dtype_compatible(meta, backend.info(), run_dir)
    update_model_info(run_dir, meta, backend.info())

    reset_peak_vram()
    wall_start = time.perf_counter()
    n_success = n_failed = n_skipped = 0
    consecutive_oom = 0
    for example in examples:
        if (example.question_id,) in done:
            n_skipped += 1
            continue
        context = select_context(cfg, example)
        alias_map = ids.make_alias_map([p.passage_id for p in context])
        messages = build_messages(
            example.question, context, alias_map, version=cfg.generation.prompt_version
        )
        record: dict[str, object] = {
            "question_id": example.question_id,
            "context_passage_ids": [p.passage_id for p in context],
            "alias_map": alias_map,
            "messages": messages,
            "prompt_version": cfg.generation.prompt_version,
        }
        try:
            with SampleTimer() as timer:
                out = backend.generate(messages, max_new_tokens=cfg.generation.max_new_tokens)
            citations = parse_citations(out.text, alias_map)
            abstained = is_abstention(out.text)
            answer_text = "" if abstained else strip_citations(out.text)
            f1, _prec, _rec = f1_score(answer_text, example.answer)
            record.update(
                response_text=out.text,
                answer_text=answer_text,
                abstained=abstained,
                citations_raw=list(citations.raw_aliases),
                cited_passage_ids=list(citations.cited_passage_ids),
                invalid_citations=list(citations.invalid_aliases),
                em=exact_match(answer_text, example.answer) if not abstained else 0,
                f1=round(f1, 6) if not abstained else 0.0,
                prompt_tokens=out.prompt_tokens,
                completion_tokens=out.completion_tokens,
                latency_ms=round(timer.elapsed_ms, 3),
                peak_vram_mb=peak_vram_mb(),
                error=None,
            )
            n_success += 1
            consecutive_oom = 0
        except RagEvidenceError:
            raise  # configuration/upstream problems abort the run — they affect every sample
        except Exception as exc:
            logger.exception("generation failed for %s", example.question_id)
            record.update(
                response_text=None,
                answer_text=None,
                abstained=None,
                citations_raw=None,
                cited_passage_ids=None,
                invalid_citations=None,
                em=None,
                f1=None,
                prompt_tokens=None,
                completion_tokens=None,
                latency_ms=None,
                peak_vram_mb=peak_vram_mb(),
                error={"type": type(exc).__name__, "message": str(exc)[:500]},
            )
            n_failed += 1
            if _is_oom(exc):
                consecutive_oom += 1
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass
                if consecutive_oom >= _MAX_CONSECUTIVE_OOM:
                    append_record(records_path, record)
                    finalize_run(
                        run_dir,
                        meta,
                        status="aborted_oom",
                        n_attempted=n_success + n_failed,
                        n_success=n_success,
                        n_failed=n_failed,
                        n_skipped=n_skipped,
                        run_peak_vram_mb=peak_vram_mb(),
                        total_wall_s=round(time.perf_counter() - wall_start, 3),
                    )
                    raise RagEvidenceError(
                        f"{_MAX_CONSECUTIVE_OOM} consecutive CUDA OOMs — systemic, aborting; "
                        "reduce context or use 4-bit quantization"
                    ) from exc
            else:
                consecutive_oom = 0
        append_record(records_path, record)

    finalize_run(
        run_dir,
        meta,
        status="completed" if n_failed == 0 else "completed_with_failures",
        n_attempted=n_success + n_failed,
        n_success=n_success,
        n_failed=n_failed,
        n_skipped=n_skipped,
        run_peak_vram_mb=peak_vram_mb(),
        total_wall_s=round(time.perf_counter() - wall_start, 3),
    )
    logger.info(
        "generate[%s] %s: %d ok, %d failed, %d resumed-skip -> %s",
        name,
        cfg.split,
        n_success,
        n_failed,
        n_skipped,
        run_dir,
    )
