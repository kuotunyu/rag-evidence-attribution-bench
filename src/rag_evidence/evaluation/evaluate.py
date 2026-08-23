"""The `evaluate` stage: aggregate results/raw/{split} into results/derived.

Pure CPU arithmetic over stored records — every model-dependent number (logprobs,
VRAM, latency) was measured at execution time and is only aggregated here.

Mode B rule (spec-critical): agreement-with-supporting-facts metrics are aggregated
ONLY over samples the model answered correctly (EM==1 by default); supporting facts
are never treated as causal ground truth for a wrong answer. Subset sizes are always
reported. Correctness stored at generation time is RECOMPUTED here and any mismatch
is a hard error — imported records must be self-consistent.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import rag_evidence
from rag_evidence.config import AppConfig
from rag_evidence.data.hotpot import load_execution_examples
from rag_evidence.data.schema import Example
from rag_evidence.data.splits import load_manifest
from rag_evidence.errors import DataError, UpstreamMissingError
from rag_evidence.evaluation.comparisons import AnalysisTier, compare_attribution_methods
from rag_evidence.evaluation.construct import evaluate_construct_validation
from rag_evidence.evaluation.protocol import build_experiment_matrix
from rag_evidence.evaluation.schema import upgrade_attribution_metrics
from rag_evidence.metrics.attribution import (
    attribution_ndcg,
    average_precision,
    prf_at_k,
    rank_passages,
)
from rag_evidence.metrics.retrieval import mrr, ndcg_at_k, percentiles, recall_at_k
from rag_evidence.metrics.text import exact_match
from rag_evidence.storage.artifacts import (
    RECORDS_FILE,
    RUN_META_FILE,
    append_record,
    read_json,
    read_records,
    write_json_atomic,
)

logger = logging.getLogger(__name__)

SUMMARY_SCHEMA_VERSION = 2


def _mean(values: Iterable[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _run_dirs(base: Path) -> list[Path]:
    if not base.exists():
        return []
    return sorted(p for p in base.iterdir() if (p / RECORDS_FILE).exists())


def _is_partial(meta: dict[str, Any], records: list[dict[str, Any]]) -> bool:
    """A run is partial if it never finalized or has fewer records than expected."""
    if meta.get("status") in (None, "running", "interrupted", "aborted_oom"):
        return True
    expected = meta.get("expected_count")
    return expected is not None and len(records) < expected


def _load_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta = read_json(run_dir / RUN_META_FILE) if (run_dir / RUN_META_FILE).exists() else {}
    return meta, list(read_records(run_dir / RECORDS_FILE))


def _referenced_run_dirs(*sections: dict[str, Any]) -> set[Path]:
    """Return only run directories actually consumed into this evaluation."""
    found: set[Path] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            run = value.get("run")
            if isinstance(run, str):
                found.add(Path(run))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for section in sections:
        visit(section)
    return found


def _source_snapshot_utc(*sections: dict[str, Any]) -> str | None:
    """Use immutable run metadata, never the evaluator's wall clock, as provenance."""
    timestamps: list[str] = []
    for run_dir in sorted(_referenced_run_dirs(*sections)):
        meta_path = run_dir / RUN_META_FILE
        if not meta_path.exists():
            raise DataError(f"consumed run has no {RUN_META_FILE}: {run_dir}")
        meta = read_json(meta_path)
        timestamp = meta.get("finished_utc") or meta.get("started_utc")
        if not isinstance(timestamp, str) or not timestamp:
            raise DataError(f"consumed run has no source timestamp: {run_dir}")
        timestamps.append(timestamp)
    return max(timestamps, default=None)


# ------------------------------------------------------------------------- retrieval


def evaluate_retrieval(
    cfg: AppConfig, examples: dict[str, Example], *, allow_partial: bool
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for run_dir in _run_dirs(cfg.results_raw_dir / cfg.split / "retrieve"):
        method = run_dir.name
        meta, records = _load_run(run_dir)
        partial = _is_partial(meta, records)
        if partial and not allow_partial:
            logger.warning(
                "EXCLUDING partial run %s (%d/%s records, status=%s) — pass --allow-partial "
                "to include it, which marks the summary entry partial",
                run_dir,
                len(records),
                meta.get("expected_count"),
                meta.get("status"),
            )
            continue
        ok = [r for r in records if r.get("error") is None]
        failed = [r for r in records if r.get("error") is not None]
        recalls: dict[int, list[float]] = {k: [] for k in cfg.retrieval.ks}
        complete_coverages: dict[int, list[float]] = {k: [] for k in cfg.retrieval.ks}
        ndcgs_at: dict[int, list[float]] = {k: [] for k in cfg.retrieval.ks}
        mrrs: list[float] = []
        ndcgs: list[float] = []
        latencies: list[float] = []
        rerank_latencies: list[float] = []
        end_to_end_estimates: list[float] = []
        cache_hits = cache_misses = scored_pairs = 0
        scoring_s = 0.0
        for rec in ok:
            example = examples.get(rec["question_id"])
            if example is None:
                raise DataError(f"retrieval record for unknown question {rec['question_id']}")
            ranking = [r["passage_id"] for r in rec["ranking"]]
            gold = set(example.gold_passage_ids)
            for k in cfg.retrieval.ks:
                recalls[k].append(recall_at_k(ranking, gold, k))
                complete_coverages[k].append(float(gold.issubset(set(ranking[:k]))))
                ndcgs_at[k].append(ndcg_at_k(ranking, gold, k))
            mrrs.append(mrr(ranking, gold))
            ndcgs.append(ndcg_at_k(ranking, gold, 10))
            latencies.append(rec["latency_ms"])
            if rec.get("rerank_latency_ms") is not None:
                rerank_latencies.append(float(rec["rerank_latency_ms"]))
            if rec.get("estimated_end_to_end_latency_ms") is not None:
                end_to_end_estimates.append(float(rec["estimated_end_to_end_latency_ms"]))
            cache_hits += int(rec.get("cache_hits", 0))
            cache_misses += int(rec.get("cache_misses", 0))
            scored_pairs += int(rec.get("scored_pair_count", 0))
            scoring_s += float(rec.get("scoring_latency_ms", 0.0)) / 1000.0
        out[method] = {
            "run": str(run_dir).replace("\\", "/"),
            "execution_kind": meta.get("execution_kind", "real"),
            "partial": partial,
            "device": (meta.get("env") or {}).get("gpu_name") or "cpu",
            "n_evaluated": len(ok),
            "n_failed": len(failed),
            "failure_rate": len(failed) / max(1, len(records)),
            "recall_at": {str(k): _mean(v) for k, v in recalls.items()},
            "evidence_coverage_at": {str(k): _mean(v) for k, v in recalls.items()},
            "multi_hop_necessary_passage_coverage_at": {
                str(k): _mean(v) for k, v in complete_coverages.items()
            },
            "mrr": _mean(mrrs),
            "ndcg_at_10": _mean(ndcgs),
            "ndcg_at": {str(k): _mean(v) for k, v in ndcgs_at.items()},
            "latency_ms": percentiles(latencies),
            "rerank_latency_ms": percentiles(rerank_latencies) if rerank_latencies else None,
            "estimated_end_to_end_latency_ms": (
                percentiles(end_to_end_estimates) if end_to_end_estimates else None
            ),
            "peak_vram_mb": meta.get("run_peak_vram_mb"),
            "cache": (
                {
                    "hits": cache_hits,
                    "misses": cache_misses,
                    "hit_rate": (
                        cache_hits / (cache_hits + cache_misses)
                        if cache_hits + cache_misses
                        else None
                    ),
                }
                if cache_hits + cache_misses
                else None
            ),
            "throughput_pairs_per_s": (
                scored_pairs / scoring_s if scored_pairs and scoring_s > 0 else None
            ),
            "reranker_systems": meta.get("reranker_systems"),
        }
    return out


# ------------------------------------------------------------------------ generation


def evaluate_generation(
    cfg: AppConfig, examples: dict[str, Example], *, allow_partial: bool
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    """Returns (metrics_by_run_name, records_by_run_name[qid])."""
    out: dict[str, Any] = {}
    records_by_run: dict[str, dict[str, dict[str, Any]]] = {}
    for run_dir in _run_dirs(cfg.results_raw_dir / cfg.split / "generate"):
        name = run_dir.name
        meta, records = _load_run(run_dir)
        partial = _is_partial(meta, records)
        if partial and not allow_partial:
            logger.warning(
                "EXCLUDING partial generation run %s (%d/%s, status=%s) — --allow-partial "
                "includes it and marks the summary entry partial",
                run_dir,
                len(records),
                meta.get("expected_count"),
                meta.get("status"),
            )
            continue
        records_by_run[name] = {r["question_id"]: r for r in records}
        ok = [r for r in records if r.get("error") is None]
        failed = [r for r in records if r.get("error") is not None]
        answered = [r for r in ok if not r["abstained"]]
        abstained = [r for r in ok if r["abstained"]]

        mismatches = []
        for rec in ok:
            example = examples[rec["question_id"]]
            if (
                rec.get("em") is None
                and (rec.get("challenge") or {}).get("human_answerability") == "unanswerable"
            ):
                continue
            em_here = exact_match(rec["answer_text"], example.answer) if not rec["abstained"] else 0
            if em_here != rec["em"]:
                mismatches.append(rec["question_id"])
        if mismatches:
            raise DataError(
                f"stored EM flags disagree with local recomputation for {mismatches[:5]} "
                "(normalization drift between execution and evaluation?) — refusing to evaluate"
            )

        cit_p: list[float] = []
        cit_r: list[float] = []
        cit_f1: list[float] = []
        citation_coverage_all: list[float] = []
        complete_citation_coverage_all: list[float] = []
        n_no_citation = 0
        for rec in ok:
            example = examples[rec["question_id"]]
            gold = set(example.gold_passage_ids)
            valid = set(rec.get("cited_passage_ids") or []) if not rec["abstained"] else set()
            citation_coverage_all.append(len(valid & gold) / len(gold))
            complete_citation_coverage_all.append(float(gold.issubset(valid)))
        for rec in answered:
            example = examples[rec["question_id"]]
            gold = set(example.gold_passage_ids)
            valid = set(rec["cited_passage_ids"])
            n_invalid = len(rec["invalid_citations"])
            n_cited = len(valid) + n_invalid
            if n_cited == 0:
                n_no_citation += 1
                cit_p.append(0.0)
                cit_r.append(0.0)
                cit_f1.append(0.0)
                continue
            p = len(valid & gold) / n_cited  # hallucinated aliases penalize precision
            r = len(valid & gold) / len(gold)
            cit_p.append(p)
            cit_r.append(r)
            cit_f1.append(2 * p * r / (p + r) if (p + r) else 0.0)

        latencies = [r["latency_ms"] for r in ok]
        total_latency_s = sum(latencies) / 1000.0 if latencies else 0.0
        total_completion = sum(r["completion_tokens"] for r in ok)
        out[name] = {
            "run": str(run_dir).replace("\\", "/"),
            "execution_kind": meta.get("execution_kind", "real"),
            "partial": partial,
            "device": (meta.get("env") or {}).get("gpu_name") or "cpu",
            "model": meta.get("model", {}),
            "n_evaluated": len(ok),
            "n_failed": len(failed),
            "failure_rate": len(failed) / max(1, len(records)),
            "abstain_rate": len(abstained) / max(1, len(ok)),
            "em": _mean(r["em"] for r in ok if r.get("em") is not None),
            "f1": _mean(r["f1"] for r in ok if r.get("f1") is not None),
            "answerability_accuracy": _mean(
                float(r["answerability_correct"])
                for r in ok
                if r.get("answerability_correct") is not None
            ),
            "n_answerable": sum(
                (r.get("challenge") or {}).get("human_answerability") == "answerable" for r in ok
            ),
            "n_unanswerable": sum(
                (r.get("challenge") or {}).get("human_answerability") == "unanswerable" for r in ok
            ),
            "citation": {
                "precision": _mean(cit_p),
                "recall": _mean(cit_r),
                "f1": _mean(cit_f1),
                "no_citation_rate": n_no_citation / max(1, len(answered)),
                "n_scored": len(answered),
                "coverage_all": _mean(citation_coverage_all),
                "complete_coverage_all": _mean(complete_citation_coverage_all),
            },
            "latency_ms": percentiles(latencies),
            "throughput_tokens_per_s": (
                total_completion / total_latency_s if total_latency_s > 0 else None
            ),
            "peak_vram_mb": meta.get("run_peak_vram_mb"),
        }
    return out, records_by_run


# ----------------------------------------------------------------------- attribution


def _is_correct(cfg: AppConfig, gen_rec: dict[str, Any]) -> bool:
    if gen_rec.get("em") is None:
        return False
    if cfg.evaluation.correctness_criterion == "em":
        return bool(gen_rec["em"] == 1)
    return bool(gen_rec["f1"] is not None and gen_rec["f1"] >= 0.5)  # f1_05


def evaluate_attribution(
    cfg: AppConfig,
    examples: dict[str, Example],
    gen_records_by_run: dict[str, dict[str, dict[str, Any]]],
    *,
    allow_partial: bool,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for mode in ("gold", "generated"):
        mode_dir = cfg.results_raw_dir / cfg.split / "attribute" / mode
        methods = _run_dirs(mode_dir)
        if not methods:
            continue
        mode_out: dict[str, Any] = {}
        generated_subsets: dict[str, dict[str, Any]] = {}
        legacy_by_run: dict[str, dict[str, Any]] = {}
        per_sample_by_run: dict[str, dict[str, dict[str, Any]]] = {}

        for run_dir in methods:
            run_key = run_dir.name
            meta, records = _load_run(run_dir)
            record_attempts = len(records)
            historical_failures = sum(rec.get("error") is not None for rec in records)
            records = list({str(rec["question_id"]): rec for rec in records}.values())
            retry_records = record_attempts - len(records)
            scientific = meta.get("config_scientific") or {}
            record_method = records[0].get("method") if records else None
            actual_method = str(scientific.get("method") or record_method or run_key)
            namespace = scientific.get("run_namespace")
            generation_run = scientific.get("generation_run")
            if generation_run is None and records:
                generation_run = records[0].get("generation_run")

            subset: dict[str, Any] | None = None
            if mode == "generated":
                if not generation_run:
                    # Backward compatibility for v0.1 artifacts, which predate explicit
                    # generation-run linkage and only ever had one generation run.
                    generation_run = next(iter(gen_records_by_run), None)
                gen_records = gen_records_by_run.get(str(generation_run), {})
                if not gen_records:
                    raise UpstreamMissingError(
                        f"attribution run {run_key!r} references generation run "
                        f"{generation_run!r}, but its records are missing"
                    )
                gen_ok = [r for r in gen_records.values() if r.get("error") is None]
                n_abstained = sum(1 for r in gen_ok if r["abstained"])
                correct_qids = {
                    r["question_id"] for r in gen_ok if not r["abstained"] and _is_correct(cfg, r)
                }
                subset = {
                    "criterion": cfg.evaluation.correctness_criterion,
                    "n_total": len(examples),
                    "n_generated_ok": len(gen_ok),
                    "n_abstained": n_abstained,
                    "n_correct": len(correct_qids),
                }
                generated_subsets[str(generation_run)] = subset
            else:
                correct_qids = set(examples)

            partial = _is_partial(meta, records)
            if partial and not allow_partial:
                logger.warning(
                    "EXCLUDING partial attribution run %s (%d/%s, status=%s)",
                    run_dir,
                    len(records),
                    meta.get("expected_count"),
                    meta.get("status"),
                )
                continue
            per_sample_path = (
                cfg.results_derived_dir
                / cfg.split
                / "attribution"
                / mode
                / f"{run_key}_per_sample.jsonl"
            )
            per_sample_path.parent.mkdir(parents=True, exist_ok=True)
            per_sample_path.unlink(missing_ok=True)

            ok = [r for r in records if not r.get("skipped") and r.get("error") is None]
            failed = [r for r in records if r.get("error") is not None]
            skipped = [r for r in records if r.get("skipped")]

            agg_prf: dict[int, dict[str, list[float]]] = {
                k: {"p": [], "r": [], "f1": []} for k in cfg.evaluation.attribution_ks
            }
            aps: list[float] = []
            ndcgs: list[float] = []
            suffs: list[float] = []
            comps: list[float] = []
            secs: list[float] = []
            calls: list[float] = []
            n_agreement = 0
            sample_rows: dict[str, dict[str, Any]] = {}

            for rec in ok:
                example = examples[rec["question_id"]]
                gold = set(example.gold_passage_ids)
                ranking = rank_passages(rec["raw_scores"], rec["context_passage_ids"])
                in_subset = rec["question_id"] in correct_qids
                sample_row: dict[str, Any] = {
                    "question_id": rec["question_id"],
                    "in_agreement_subset": in_subset,
                    "exclusion_reason": None if in_subset else "not_correct",
                    "ap": average_precision(ranking, gold),
                    "ndcg_at_10": attribution_ndcg(ranking, gold, 10),
                    "p_at": {},
                    "r_at": {},
                    "f1_at": {},
                }
                for k in cfg.evaluation.attribution_ks:
                    prf = prf_at_k(ranking, gold, k)
                    sample_row["p_at"][str(k)] = prf["p"]
                    sample_row["r_at"][str(k)] = prf["r"]
                    sample_row["f1_at"][str(k)] = prf["f1"]
                faith = rec.get("faithfulness")
                sample_row["sufficiency"] = faith["sufficiency"] if faith else None
                sample_row["comprehensiveness"] = faith["comprehensiveness"] if faith else None
                append_record(per_sample_path, sample_row)
                sample_rows[str(rec["question_id"])] = sample_row

                if in_subset:  # agreement metrics only over the qualifying subset
                    n_agreement += 1
                    for k in cfg.evaluation.attribution_ks:
                        agg_prf[k]["p"].append(sample_row["p_at"][str(k)])
                        agg_prf[k]["r"].append(sample_row["r_at"][str(k)])
                        agg_prf[k]["f1"].append(sample_row["f1_at"][str(k)])
                    aps.append(sample_row["ap"])
                    ndcgs.append(sample_row["ndcg_at_10"])
                # faithfulness is ground-truth-free → aggregated over ALL attributed
                if faith:
                    suffs.append(faith["sufficiency"])
                    comps.append(faith["comprehensiveness"])
                if rec.get("latency_s") is not None:
                    secs.append(rec["latency_s"])
                calls.append(rec.get("num_model_calls", 0))

            legacy_metrics = {
                "run": str(run_dir).replace("\\", "/"),
                "execution_kind": meta.get("execution_kind", "real"),
                "partial": partial,
                "device": (meta.get("env") or {}).get("gpu_name") or "cpu",
                "method": actual_method,
                "run_namespace": namespace,
                "generation_run": generation_run,
                "subset": subset,
                "is_control": bool(
                    scientific.get("is_control", actual_method.startswith("control_"))
                ),
                "n_attempted": len(ok) + len(failed),
                "n_success": len(ok),
                "n_failed": len(failed),
                "n_skipped": len(skipped),
                "n_record_attempts": record_attempts,
                "n_retry_records": retry_records,
                "n_historical_failure_records": historical_failures,
                "failure_rate": len(failed) / max(1, len(ok) + len(failed)),
                "n_agreement": n_agreement,
                "prf_at": {
                    str(k): {"p": _mean(v["p"]), "r": _mean(v["r"]), "f1": _mean(v["f1"])}
                    for k, v in agg_prf.items()
                },
                "auprc": _mean(aps),
                "ndcg_at_10": _mean(ndcgs),
                "sufficiency": _mean(suffs),
                "comprehensiveness": _mean(comps),
                "n_faithfulness": len(suffs),
                "seconds_per_sample": _mean(secs),
                "num_model_calls_mean": _mean(calls),
                "peak_vram_mb": meta.get("run_peak_vram_mb"),
            }
            legacy_by_run[run_key] = legacy_metrics
            per_sample_by_run[run_key] = sample_rows

        runs_by_method: dict[str, list[str]] = {}
        for run_key, legacy in legacy_by_run.items():
            runs_by_method.setdefault(str(legacy["method"]), []).append(run_key)
        unique_run_by_method = {
            method: run_keys[0] for method, run_keys in runs_by_method.items() if len(run_keys) == 1
        }
        per_sample_by_method = {
            method: per_sample_by_run[run_key] for method, run_key in unique_run_by_method.items()
        }
        method_status = {
            method: {
                "partial": bool(legacy_by_run[run_key]["partial"]),
                "eligible": bool(
                    legacy_by_run[run_key]["execution_kind"] != "mock"
                    and legacy_by_run[run_key]["n_success"] > 0
                ),
            }
            for method, run_key in unique_run_by_method.items()
        }
        construct_validation = evaluate_construct_validation(
            per_sample_by_method,
            method_status=method_status,
            split=cfg.split,
            mode=mode,
            global_seed=cfg.seed,
            resamples=cfg.evaluation.bootstrap_resamples,
            confidence=cfg.evaluation.bootstrap_confidence,
            tolerance=cfg.evaluation.bootstrap_tolerance,
        )
        agreement_subset = (
            "all_successful_attributions"
            if mode == "gold"
            else (
                "generated_exact_match"
                if cfg.evaluation.correctness_criterion == "em"
                else "generated_f1_at_least_0_5"
            )
        )
        for run_key, legacy in legacy_by_run.items():
            mode_out[run_key] = upgrade_attribution_metrics(
                legacy,
                agreement_subset=agreement_subset,
                causal_validation_status=construct_validation["status"],
            )

        paired_comparisons: dict[str, Any] = {}
        candidate_method = cfg.evaluation.confirmatory_method
        comparator_method = cfg.evaluation.primary_comparator
        control_methods = {
            method
            for method, run_key in unique_run_by_method.items()
            if bool(legacy_by_run[run_key]["is_control"])
        }
        real_methods = sorted(set(unique_run_by_method) - control_methods)
        comparator_methods = sorted(
            control_methods | ({"leave_one_out", "embedding"} & set(unique_run_by_method))
        )
        for candidate in real_methods:
            for comparator in comparator_methods:
                if candidate == comparator:
                    continue
                analysis_tier: AnalysisTier = (
                    "confirmatory"
                    if candidate == candidate_method and comparator == comparator_method
                    else ("exploratory" if candidate in {"arc_jsd", "contextcite"} else "secondary")
                )
                comparison_key = f"{candidate}__vs__{comparator}"
                paired_comparisons[comparison_key] = compare_attribution_methods(
                    per_sample_by_method[candidate],
                    per_sample_by_method[comparator],
                    split=cfg.split,
                    mode=mode,
                    candidate_method=candidate,
                    comparator_method=comparator,
                    analysis_tier=analysis_tier,
                    primary_k=cfg.evaluation.primary_k,
                    global_seed=cfg.seed,
                    resamples=cfg.evaluation.bootstrap_resamples,
                    confidence=cfg.evaluation.bootstrap_confidence,
                    tolerance=cfg.evaluation.bootstrap_tolerance,
                )
        mode_out["paired_comparisons"] = paired_comparisons
        mode_out["construct_validation"] = construct_validation
        # Preserve the original summary shape bit-for-bit in meaning (and keep the
        # existing report renderer working) when every run shares one generation.
        if mode == "generated" and len(generated_subsets) == 1:
            generation_run, subset = next(iter(generated_subsets.items()))
            mode_out["generation_run"] = generation_run
            mode_out["subset"] = subset
        elif mode == "generated" and generated_subsets:
            mode_out["subsets"] = generated_subsets
        out[mode] = mode_out
    return out


# ---------------------------------------------------------------------------- driver


def evaluate_all(cfg: AppConfig, *, allow_partial: bool = False) -> None:
    split = cfg.split
    examples = {e.question_id: e for e in load_execution_examples(cfg)}
    derived_split = cfg.results_derived_dir / split
    derived_split.mkdir(parents=True, exist_ok=True)

    retrieval = evaluate_retrieval(cfg, examples, allow_partial=allow_partial)
    generation, gen_records = evaluate_generation(cfg, examples, allow_partial=allow_partial)
    attribution = evaluate_attribution(cfg, examples, gen_records, allow_partial=allow_partial)

    if retrieval:
        write_json_atomic(derived_split / "retrieval_metrics.json", retrieval)
    if generation:
        write_json_atomic(derived_split / "generation_metrics.json", generation)
    for mode, methods in attribution.items():
        for method, metrics in methods.items():
            if isinstance(metrics, dict) and "run" in metrics:
                write_json_atomic(
                    derived_split / "attribution" / mode / f"{method}_metrics.json", metrics
                )

    discovered_methods = {
        mode: {
            str(entry["method"])
            for entry in methods.values()
            if isinstance(entry, dict) and entry.get("schema_version") == 2
        }
        for mode, methods in attribution.items()
    }
    split_payload = {
        "n_questions": len(examples),
        "source_snapshot_utc": _source_snapshot_utc(retrieval, generation, attribution),
        "retrieval": retrieval,
        "generation": generation,
        "attribution": attribution,
        "experiment_matrix": build_experiment_matrix(discovered_methods),
    }
    summary_path = cfg.results_derived_dir / "summary.json"
    summary: dict[str, Any] = (
        read_json(summary_path)
        if summary_path.exists()
        else {"schema_version": SUMMARY_SCHEMA_VERSION, "splits": {}}
    )
    manifest = load_manifest(cfg.manifest_file)
    summary.pop("generated_utc", None)
    summary["splits"][split] = split_payload
    summary.update(
        schema_version=SUMMARY_SCHEMA_VERSION,
        source_snapshot_utc=max(
            (
                payload["source_snapshot_utc"]
                for payload in summary["splits"].values()
                if payload.get("source_snapshot_utc") is not None
            ),
            default=None,
        ),
        package_version=rag_evidence.__version__,
        dataset_hash=manifest["fingerprint"]["dataset_hash"],
        primary_k=cfg.evaluation.primary_k,
        execution=cfg.execution.model_dump(mode="json"),
    )
    write_json_atomic(summary_path, summary)
    logger.info(
        "evaluate %s: retrieval=%d runs, generation=%d runs, attribution modes=%s -> %s",
        split,
        len(retrieval),
        len(generation),
        sorted(attribution),
        summary_path,
    )
