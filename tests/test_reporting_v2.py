"""Truthful rendering of schema-v2 attribution results."""

from __future__ import annotations

from rag_evidence.reporting.report import render_results_block


def _method() -> dict[str, object]:
    return {
        "schema_version": 2,
        "run": "results/raw/eval/attribute/generated/leave_one_out",
        "execution_kind": "real",
        "partial": False,
        "device": "NVIDIA A100",
        "method": "leave_one_out",
        "run_namespace": None,
        "generation_run": "qwen3-4b",
        "is_control": False,
        "agreement": {
            "n": 87,
            "subset": "generated_exact_match",
            "prf_at": {"2": {"p": 0.79, "r": 0.79, "f1": 0.79}},
            "mean_average_precision": 0.86,
            "ndcg_at_10": 0.92,
        },
        "causal_dependence": {
            "n": 192,
            "subset": "all_successful_non_abstained",
            "sufficiency_mean": 0.86,
            "comprehensiveness_mean": 7.16,
            "validation_status": "not_run",
        },
        "execution": {
            "n_attempted": 192,
            "n_success": 192,
            "n_failed": 0,
            "n_skipped": 48,
            "n_record_attempts": 240,
            "n_retry_records": 0,
            "n_historical_failure_records": 0,
            "failure_rate": 0.0,
            "seconds_per_sample": 0.65,
            "num_model_calls_mean": 13.0,
            "peak_vram_mb": 9109.7,
        },
        "legacy_v1": {},
    }


def _summary() -> dict[str, object]:
    comparison_metric = {
        "n_pairs": 87,
        "exclusions": {
            "candidate_missing": 0,
            "comparator_missing": 0,
            "candidate_null": 0,
            "comparator_null": 0,
        },
        "mean_delta": 0.14,
        "ci_low": 0.08,
        "ci_high": 0.20,
        "favorable_direction": "higher",
    }
    return {
        "schema_version": 2,
        "source_snapshot_utc": "2026-07-25T03:39:22Z",
        "package_version": "0.1.0",
        "dataset_hash": "a" * 64,
        "primary_k": 2,
        "splits": {
            "eval": {
                "n_questions": 240,
                "retrieval": {},
                "generation": {},
                "attribution": {
                    "generated": {
                        "leave_one_out": _method(),
                        "paired_comparisons": {
                            "leave_one_out__vs__control_lexical": {
                                "candidate_method": "leave_one_out",
                                "comparator_method": "control_lexical",
                                "analysis_tier": "confirmatory",
                                "metrics": {"f1_at_2": comparison_metric},
                            }
                        },
                        "causal_validation": {
                            "status": "not_run",
                            "missing_methods": ["oracle_gold"],
                            "comparisons": {},
                        },
                        "subset": {
                            "criterion": "em",
                            "n_total": 240,
                            "n_generated_ok": 240,
                            "n_abstained": 48,
                            "n_correct": 87,
                        },
                    }
                },
            }
        },
    }


def test_v2_report_names_estimands_denominators_and_paired_interval() -> None:
    block = render_results_block(_summary())

    assert "MAP" in block
    assert "n agreement" in block
    assert "n causal" in block
    assert "95% CI" in block
    assert "n_pairs" in block
    assert "0.140 [0.080, 0.200]" in block
    assert "Causal-dependence validation: **NOT RUN**" in block
    assert "AUPRC" not in block
    assert "faithfulness improvement" not in block.lower()


def test_v2_report_exposes_pair_exclusions() -> None:
    block = render_results_block(_summary())

    assert "candidate_missing=0" in block
    assert "comparator_missing=0" in block
    assert "candidate_null=0" in block
    assert "comparator_null=0" in block
