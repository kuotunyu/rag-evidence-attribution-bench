"""The `attribute` stage runner: modes A (gold) / B (generated), resume, faithfulness.

Mode A: target = official gold answer; all samples; context = dataset order.
Mode B: target = the model's generated answer; all non-abstained generated samples
        (faithfulness is ground-truth-free and meaningful for wrong answers too); the
        prompt reuses the generation record's exact context order and alias_map. The
        correct-subset restriction happens in `evaluate`, never here.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import rag_evidence.attribution  # noqa: F401 — registers built-in methods
from rag_evidence.attribution.base import (
    AttributionMethod,
    ModelResources,
    normalize_scores,
    validate_scores,
)
from rag_evidence.attribution.registry import get_method
from rag_evidence.attribution.scoring import LogprobScorer
from rag_evidence.config import AppConfig, resolve_device, resolve_dtype
from rag_evidence.data import ids as ids_mod
from rag_evidence.data.hotpot import load_execution_examples, load_execution_metadata
from rag_evidence.data.schema import Example, Passage
from rag_evidence.errors import ConfigError, GpuRequiredError, UpstreamMissingError
from rag_evidence.generation.prompts import prompt_hash
from rag_evidence.generation.run import generation_run_name
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


def _load_generation_records(cfg: AppConfig) -> dict[str, dict[str, Any]]:
    name = generation_run_name(cfg)
    path = stage_dir(cfg.results_raw_dir, cfg.split, "generate", name) / RECORDS_FILE
    if not path.exists():
        raise UpstreamMissingError(
            f"mode 'generated' needs a completed generate run ({path} missing)"
        )
    return {rec["question_id"]: rec for rec in read_records(path)}


def _load_retrieval_ranks(cfg: AppConfig, method: str) -> dict[str, dict[str, int]]:
    root = (
        Path(cfg.attribution.controls.retrieval_results_raw)
        if cfg.attribution.controls.retrieval_results_raw
        else cfg.results_raw_dir
    )
    path = stage_dir(root, cfg.split, "retrieve", method) / RECORDS_FILE
    if not path.exists():
        raise UpstreamMissingError(
            f"control_retrieval needs `retrieve --method {method}` first ({path} missing)"
        )
    return {
        rec["question_id"]: {r["passage_id"]: r["rank"] for r in rec["ranking"]}
        for rec in read_records(path)
        if rec.get("error") is None
    }


def _load_source_scores(cfg: AppConfig, mode: str, source: str) -> dict[str, dict[str, float]]:
    path = _mode_dir(cfg, mode, source) / RECORDS_FILE
    if not path.exists():
        raise UpstreamMissingError(
            f"control_shuffled needs a completed `attribute --method {source}` run for "
            f"mode {mode!r} first ({path} missing)"
        )
    return {
        rec["question_id"]: dict(rec["raw_scores"])
        for rec in read_records(path)
        if rec.get("error") is None and not rec.get("skipped")
    }


def _mode_dir(cfg: AppConfig, mode: str, method: str) -> Path:
    return stage_dir(cfg.results_raw_dir, cfg.split, "attribute", mode) / attribution_run_name(
        cfg, method
    )


def attribution_run_name(cfg: AppConfig, method: str) -> str:
    namespace = cfg.attribution.run_namespace
    return f"{namespace}__{method}" if namespace else method


class _SampleContext:
    """Everything mode-dependent about one sample's attribution call."""

    def __init__(
        self,
        example: Example,
        *,
        target_answer: str,
        target_source: str,
        passages: Sequence[Passage],
        alias_map: dict[str, str],
        generation_record: dict[str, Any] | None,
    ) -> None:
        self.example = example
        self.target_answer = target_answer
        self.target_source = target_source
        self.passages = list(passages)
        self.alias_map = alias_map
        self.generation_record = generation_record


def _sample_context(
    cfg: AppConfig, example: Example, mode: str, gen_records: Mapping[str, dict[str, Any]]
) -> tuple[_SampleContext | None, str | None]:
    """Returns (context, skip_reason). Exactly one is non-None."""
    if mode == "gold":
        if cfg.attribution.gold_context_source == "generation":
            rec = gen_records.get(example.question_id)
            if rec is None:
                return None, "generation_context_missing"
            passages = [example.passage_by_id(pid) for pid in rec["context_passage_ids"]]
            alias_map = dict(rec["alias_map"])
            source = f"gold:context:{generation_run_name(cfg)}"
        else:
            passages = list(example.passages)
            alias_map = ids_mod.make_alias_map([p.passage_id for p in passages])
            source = "gold"
        return (
            _SampleContext(
                example,
                target_answer=example.answer,
                target_source=source,
                passages=passages,
                alias_map=alias_map,
                generation_record=None,
            ),
            None,
        )
    rec = gen_records.get(example.question_id)
    if rec is None:
        return None, "generation_missing"
    if rec.get("error") is not None:
        return None, "generation_failed"
    if rec.get("abstained"):
        return None, "abstained"
    if not rec.get("answer_text"):
        return None, "empty_answer"
    passages = [example.passage_by_id(pid) for pid in rec["context_passage_ids"]]
    return (
        _SampleContext(
            example,
            target_answer=rec["answer_text"],  # already citation-stripped by generate
            target_source=f"generated:{generation_run_name(cfg)}",
            passages=passages,
            alias_map=dict(rec["alias_map"]),
            generation_record=rec,
        ),
        None,
    )


def _build_resources(
    cfg: AppConfig, method_obj: AttributionMethod, *, need_faithfulness: bool
) -> tuple[ModelResources, LogprobScorer | None, dict[str, Any] | None, Literal["real", "mock"]]:
    """Load only what the method + faithfulness need. Returns
    (resources, scorer, faithfulness_unavailable_error, execution_kind)."""
    resources = ModelResources()
    scorer: LogprobScorer | None = None
    faith_error: dict[str, Any] | None = None
    execution_kind: Literal["real", "mock"] = "real"

    needs_generator = method_obj.requires_generator or need_faithfulness
    if needs_generator:
        device = resolve_device(cfg.runtime.device)
        dtype = resolve_dtype(cfg.generation.dtype, device)
        if cfg.generation.backend == "fake":
            execution_kind = "mock"
        try:
            from rag_evidence.generation.backends import build_backend

            backend = build_backend(
                cfg.generation,
                device=device,
                dtype=dtype,
                seed=cfg.seed,
                explicit_cpu=cfg.runtime.device == "cpu",
            )
            resources.generator = backend
            cache_dir = cfg.results_raw_dir / "cache" / "logprobs" / cfg.split
            scorer = LogprobScorer(
                backend,
                cache_dir / f"{generation_run_name(cfg)}.jsonl",
                prompt_version=cfg.generation.prompt_version,
            )
        except GpuRequiredError as exc:
            if method_obj.requires_generator:
                raise  # the method itself cannot run — abort the whole run
            # only faithfulness wanted the generator: degrade openly, never silently
            faith_error = {"type": "GpuRequiredError", "message": str(exc)[:300]}
            logger.warning("faithfulness passes skipped for this run (no GPU here): %s", exc)

    if method_obj.requires_embedder:
        from rag_evidence.embeddings import Embedder, EmbeddingCache, model_tag

        resources.embedder = Embedder.load(
            cfg.attribution.embedding.model_id,
            device=cfg.runtime.device,
            max_length=cfg.retrieval.dense.max_length,
            normalize=True,
        )
        resources.embedding_cache = EmbeddingCache(
            cfg.results_raw_dir
            / "cache"
            / "embeddings"
            / model_tag(cfg.attribution.embedding.model_id)
            / f"{cfg.split}_passages.npz"
        )
    return resources, scorer, faith_error, execution_kind


def run_attribution_stage(
    cfg: AppConfig,
    *,
    method: str,
    mode: str | None,
    resume: bool,
    limit: int | None,
    retry_failures: bool = False,
) -> None:
    if retry_failures and not resume:
        raise ConfigError("--retry-failures requires --resume so the failed attempt is preserved")

    method_obj = get_method(method)
    modes = [mode] if mode else [m for m in cfg.attribution.modes]

    runnable_modes: list[str] = []
    for m in modes:
        if m == "gold" and not method_obj.supports_teacher_forced:
            if mode is not None:  # explicitly requested → that's a config error
                raise ConfigError(f"method {method!r} does not support teacher-forced (gold) mode")
            logger.info("method %s does not support mode 'gold' — skipping that mode", method)
            continue
        if m == "generated" and not method_obj.supports_generated:
            if mode is not None:
                raise ConfigError(f"method {method!r} does not support generated mode")
            logger.info("method %s does not support mode 'generated' — skipped", method)
            continue
        runnable_modes.append(m)

    if not runnable_modes:
        return

    # Build the model/embedder resources ONCE per CLI invocation and reuse across
    # modes: a fresh Qwen3-4B load is the single biggest fixed cost of an `attribute`
    # call, and teacher-forced scoring has no cross-mode state, so gold+generated can
    # safely share one loaded backend instead of paying the load twice.
    need_faith = cfg.attribution.faithfulness.enabled
    resources, scorer, faith_unavailable, execution_kind = _build_resources(
        cfg, method_obj, need_faithfulness=need_faith
    )

    for m in runnable_modes:
        _run_one_mode(
            cfg,
            method_obj,
            method,
            m,
            resume=resume,
            limit=limit,
            retry_failures=retry_failures,
            resources=resources,
            scorer=scorer,
            faith_unavailable=faith_unavailable,
            execution_kind=execution_kind,
        )


def _run_one_mode(
    cfg: AppConfig,
    method_obj: AttributionMethod,
    method: str,
    mode: str,
    *,
    resume: bool,
    limit: int | None,
    retry_failures: bool,
    resources: ModelResources,
    scorer: LogprobScorer | None,
    faith_unavailable: dict[str, Any] | None,
    execution_kind: Literal["real", "mock"],
) -> None:
    examples = load_execution_examples(cfg)
    execution_metadata = load_execution_metadata(cfg)
    if limit is not None:
        examples = examples[:limit]

    gen_records: dict[str, dict[str, Any]] = {}
    if mode == "generated" or cfg.attribution.gold_context_source == "generation":
        gen_records = _load_generation_records(cfg)

    retrieval_ranks: dict[str, dict[str, int]] = {}
    if method == "control_retrieval":
        retrieval_ranks = _load_retrieval_ranks(cfg, cfg.attribution.controls.retrieval_run)
    source_scores: dict[str, dict[str, float]] = {}
    if method == "control_shuffled":
        source_scores = _load_source_scores(cfg, mode, cfg.attribution.controls.shuffled_source)

    need_faith = cfg.attribution.faithfulness.enabled

    output_name = attribution_run_name(cfg, method)
    run_dir = _mode_dir(cfg, mode, method)
    meta = start_or_resume_run(
        run_dir,
        cfg,
        stage="attribute",
        name=f"{mode}/{output_name}",
        resume=resume,
        execution_kind=execution_kind,
        expected_count=len(examples),
        limit=limit,
        sci_extra={
            "method": method,
            "is_control": method_obj.is_control,
            "mode": mode,
            "run_namespace": cfg.attribution.run_namespace,
            "generation_run": generation_run_name(cfg),
            "prompt_hash": prompt_hash(cfg.generation.prompt_version),
        },
    )
    if resources.generator is not None:
        check_dtype_compatible(meta, resources.generator.info(), run_dir)
        update_model_info(run_dir, meta, resources.generator.info())

    records_path = run_dir / RECORDS_FILE
    records_before_retry = list(read_records(records_path)) if retry_failures else []
    failures_before_retry = sum(rec.get("error") is not None for rec in records_before_retry)
    done = completed_keys(records_path, include_failed=not retry_failures) if resume else set()

    reset_peak_vram()
    wall_start = time.perf_counter()
    n_success = n_failed = n_skipped_resume = n_skipped_sample = 0
    for example in examples:
        if (example.question_id,) in done:
            n_skipped_resume += 1
            continue
        sample, skip_reason = _sample_context(cfg, example, mode, gen_records)
        base_record: dict[str, Any] = {
            "question_id": example.question_id,
            "method": method,
            "mode": mode,
            "run_namespace": cfg.attribution.run_namespace,
            "generation_run": generation_run_name(cfg),
        }
        challenge = execution_metadata.get(example.question_id)
        if challenge is not None:
            base_record["challenge"] = challenge
        if (
            mode == "gold"
            and challenge is not None
            and challenge["human_answerability"] != "answerable"
        ):
            sample = None
            skip_reason = "human_unanswerable"
        if sample is None:
            base_record.update(
                skipped=True,
                skip_reason=skip_reason,
                target_answer=None,
                target_source=None,
                context_passage_ids=None,
                alias_map=None,
                scores=None,
                raw_scores=None,
                degenerate_scores=None,
                metadata=None,
                faithfulness=None,
                faithfulness_error=None,
                num_model_calls=0,
                latency_s=None,
                peak_vram_mb=None,
                error=None,
            )
            append_record(records_path, base_record)
            n_skipped_sample += 1
            continue

        method_config: dict[str, Any] = {
            "seed": cfg.seed,
            "question_id": example.question_id,
            "alias_map": sample.alias_map,
            "prompt_version": cfg.generation.prompt_version,
            "scorer": scorer,
            "generation_record": sample.generation_record,
            "retrieval_run_name": cfg.attribution.controls.retrieval_run,
            "shuffled_source_name": cfg.attribution.controls.shuffled_source,
        }
        if method == "control_retrieval":
            method_config["retrieval_ranks"] = retrieval_ranks.get(example.question_id)
        if method == "control_shuffled":
            method_config["source_scores"] = source_scores.get(example.question_id)

        try:
            with SampleTimer() as timer:
                result = method_obj.attribute(
                    sample.example.question,
                    sample.passages,
                    sample.target_answer,
                    resources,
                    method_config,
                )
                validate_scores(result.raw_scores, [p.passage_id for p in sample.passages])
                scores, degenerate = normalize_scores(result.raw_scores)

                faithfulness: dict[str, Any] | None = None
                faith_error = faith_unavailable
                if need_faith and faith_error is None and scorer is not None:
                    from rag_evidence.attribution.faithfulness import compute_faithfulness

                    try:
                        faithfulness = compute_faithfulness(
                            question_id=example.question_id,
                            question=sample.example.question,
                            passages=sample.passages,
                            alias_map=sample.alias_map,
                            target_answer=sample.target_answer,
                            raw_scores=result.raw_scores,
                            scorer=scorer,
                            k=cfg.attribution.faithfulness.k,
                        )
                    except Exception as exc:  # partial failure: scores stand, faith doesn't
                        logger.exception("faithfulness failed for %s", example.question_id)
                        faith_error = {"type": type(exc).__name__, "message": str(exc)[:300]}

            base_record.update(
                skipped=False,
                skip_reason=None,
                target_answer=sample.target_answer,
                target_source=sample.target_source,
                context_passage_ids=[p.passage_id for p in sample.passages],
                alias_map=sample.alias_map,
                scores=scores,
                raw_scores=dict(result.raw_scores),
                degenerate_scores=degenerate,
                metadata=dict(result.metadata),
                faithfulness=faithfulness,
                faithfulness_error=faith_error,
                num_model_calls=result.num_model_calls,
                latency_s=round(timer.elapsed_s, 4),
                peak_vram_mb=peak_vram_mb(),
                error=None,
            )
            n_success += 1
        except Exception as exc:
            logger.exception("attribution %s/%s failed for %s", mode, method, example.question_id)
            base_record.update(
                skipped=False,
                skip_reason=None,
                target_answer=sample.target_answer,
                target_source=sample.target_source,
                context_passage_ids=[p.passage_id for p in sample.passages],
                alias_map=sample.alias_map,
                scores=None,
                raw_scores=None,
                degenerate_scores=None,
                metadata=None,
                faithfulness=None,
                faithfulness_error=None,
                num_model_calls=0,
                latency_s=None,
                peak_vram_mb=peak_vram_mb(),
                error={"type": type(exc).__name__, "message": str(exc)[:500]},
            )
            n_failed += 1
        append_record(records_path, base_record)

    run_peak = peak_vram_mb()
    total_wall_s = round(time.perf_counter() - wall_start, 3)
    final_status = "completed" if n_failed == 0 else "completed_with_failures"
    final_attempted = n_success + n_failed
    final_success = n_success
    final_failed = n_failed
    final_skipped = n_skipped_resume + n_skipped_sample
    if retry_failures:
        all_attempts = list(read_records(records_path))
        latest = {str(rec["question_id"]): rec for rec in all_attempts}
        latest_records = list(latest.values())
        final_failed = sum(rec.get("error") is not None for rec in latest_records)
        final_skipped = sum(bool(rec.get("skipped")) for rec in latest_records)
        final_success = sum(
            not rec.get("skipped") and rec.get("error") is None for rec in latest_records
        )
        final_attempted = final_success + final_failed
        final_status = "completed" if final_failed == 0 else "completed_with_failures"
        previous_peak = meta.get("run_peak_vram_mb")
        if previous_peak is not None:
            run_peak = max(float(previous_peak), float(run_peak or 0.0))
        total_wall_s = round(float(meta.get("total_wall_s") or 0.0) + total_wall_s, 3)
        meta["failure_retry"] = {
            "enabled": True,
            "failures_before_retry": failures_before_retry,
            "attempted_this_session": n_success + n_failed,
            "resolved_this_session": n_success,
            "failed_this_session": n_failed,
            "remaining_latest_failures": final_failed,
            "record_attempts_total": len(all_attempts),
            "latest_unique_records": len(latest_records),
        }

    finalize_run(
        run_dir,
        meta,
        status=final_status,
        n_attempted=final_attempted,
        n_success=final_success,
        n_failed=final_failed,
        n_skipped=final_skipped,
        run_peak_vram_mb=run_peak,
        total_wall_s=total_wall_s,
    )
    logger.info(
        "attribute[%s/%s] %s: %d ok, %d failed, %d skipped-sample, %d resumed-skip -> %s",
        mode,
        method,
        cfg.split,
        n_success,
        n_failed,
        n_skipped_sample,
        n_skipped_resume,
        run_dir,
    )
