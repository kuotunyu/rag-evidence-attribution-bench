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

import datetime as _dt
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import rag_evidence
from rag_evidence.config import AppConfig
from rag_evidence.data.hotpot import load_prepared_verified
from rag_evidence.data.schema import Example
from rag_evidence.data.splits import load_manifest
from rag_evidence.errors import DataError, UpstreamMissingError
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

SUMMARY_SCHEMA_VERSION = 1


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
        mrrs: list[float] = []
        ndcgs: list[float] = []
        latencies: list[float] = []
        for rec in ok:
            example = examples.get(rec["question_id"])
            if example is None:
                raise DataError(f"retrieval record for unknown question {rec['question_id']}")
            ranking = [r["passage_id"] for r in rec["ranking"]]
            gold = set(example.gold_passage_ids)
            for k in cfg.retrieval.ks:
                recalls[k].append(recall_at_k(ranking, gold, k))
            mrrs.append(mrr(ranking, gold))
            ndcgs.append(ndcg_at_k(ranking, gold, 10))
            latencies.append(rec["latency_ms"])
        out[method] = {
            "run": str(run_dir).replace("\\", "/"),
            "execution_kind": meta.get("execution_kind", "real"),
            "partial": partial,
            "device": (meta.get("env") or {}).get("gpu_name") or "cpu",
            "n_evaluated": len(ok),
            "n_failed": len(failed),
            "failure_rate": len(failed) / max(1, len(records)),
            "recall_at": {str(k): _mean(v) for k, v in recalls.items()},
            "mrr": _mean(mrrs),
            "ndcg_at_10": _mean(ndcgs),
            "latency_ms": percentiles(latencies),
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
        n_no_citation = 0
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
            "em": _mean(r["em"] for r in ok),
            "f1": _mean(r["f1"] for r in ok),
            "citation": {
                "precision": _mean(cit_p),
                "recall": _mean(cit_r),
                "f1": _mean(cit_f1),
                "no_citation_rate": n_no_citation / max(1, len(answered)),
                "n_scored": len(answered),
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
    from rag_evidence.generation.run import generation_run_name

    out: dict[str, Any] = {}
    gen_name = generation_run_name(cfg)
    for mode in ("gold", "generated"):
        mode_dir = cfg.results_raw_dir / cfg.split / "attribute" / mode
        methods = _run_dirs(mode_dir)
        if not methods:
            continue
        mode_out: dict[str, Any] = {}

        gen_records = gen_records_by_run.get(gen_name, {})
        if mode == "generated":
            if not gen_records:
                raise UpstreamMissingError(
                    f"attribution mode 'generated' exists but generation run {gen_name!r} "
                    "records are missing — evaluate needs them for subset filtering"
                )
            n_total = len(examples)
            gen_ok = [r for r in gen_records.values() if r.get("error") is None]
            n_abstained = sum(1 for r in gen_ok if r["abstained"])
            correct_qids = {
                r["question_id"] for r in gen_ok if not r["abstained"] and _is_correct(cfg, r)
            }
            mode_out["generation_run"] = gen_name
            mode_out["subset"] = {
                "criterion": cfg.evaluation.correctness_criterion,
                "n_total": n_total,
                "n_generated_ok": len(gen_ok),
                "n_abstained": n_abstained,
                "n_correct": len(correct_qids),
            }
        else:
            correct_qids = set(examples)  # mode A: every sample qualifies

        for run_dir in methods:
            method = run_dir.name
            meta, records = _load_run(run_dir)
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
                / f"{method}_per_sample.jsonl"
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

            mode_out[method] = {
                "run": str(run_dir).replace("\\", "/"),
                "execution_kind": meta.get("execution_kind", "real"),
                "partial": partial,
                "device": (meta.get("env") or {}).get("gpu_name") or "cpu",
                "is_control": method.startswith("control_"),
                "n_attempted": len(ok) + len(failed),
                "n_success": len(ok),
                "n_failed": len(failed),
                "n_skipped": len(skipped),
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
        out[mode] = mode_out
    return out


# ---------------------------------------------------------------------------- driver


def evaluate_all(cfg: AppConfig, *, allow_partial: bool = False) -> None:
    split = cfg.split
    examples = {e.question_id: e for e in load_prepared_verified(cfg)}
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

    split_payload = {
        "n_questions": len(examples),
        "retrieval": retrieval,
        "generation": generation,
        "attribution": attribution,
    }
    summary_path = cfg.results_derived_dir / "summary.json"
    summary: dict[str, Any] = (
        read_json(summary_path)
        if summary_path.exists()
        else {"schema_version": SUMMARY_SCHEMA_VERSION, "splits": {}}
    )
    manifest = load_manifest(cfg.manifest_file)
    summary.update(
        schema_version=SUMMARY_SCHEMA_VERSION,
        generated_utc=_dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        package_version=rag_evidence.__version__,
        dataset_hash=manifest["fingerprint"]["dataset_hash"],
        primary_k=cfg.evaluation.primary_k,
    )
    summary["splits"][split] = split_payload
    write_json_atomic(summary_path, summary)
    logger.info(
        "evaluate %s: retrieval=%d runs, generation=%d runs, attribution modes=%s -> %s",
        split,
        len(retrieval),
        len(generation),
        sorted(attribution),
        summary_path,
    )
