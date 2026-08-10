"""Paired attribution-method comparisons built from per-question rows."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

from rag_evidence.evaluation.statistics import Direction, paired_bootstrap_difference

AnalysisTier = Literal["confirmatory", "secondary", "exploratory"]
SampleRow = Mapping[str, Any]


def _agreement_value(row: SampleRow, field: str, primary_k: int) -> float | None:
    if not row.get("in_agreement_subset"):
        return None
    if field == "f1":
        return float(row["f1_at"][str(primary_k)])
    return float(row[field])


def _diagnostic_value(row: SampleRow, field: str) -> float | None:
    value = row.get(field)
    return float(value) if value is not None else None


def _values(
    rows: Mapping[str, SampleRow], extractor: Callable[[SampleRow], float | None]
) -> dict[str, float | None]:
    return {qid: extractor(row) for qid, row in rows.items()}


def compare_attribution_methods(
    candidate_rows: Mapping[str, SampleRow],
    comparator_rows: Mapping[str, SampleRow],
    *,
    split: str,
    mode: str,
    candidate_method: str,
    comparator_method: str,
    analysis_tier: AnalysisTier,
    primary_k: int,
    global_seed: int,
    resamples: int,
    confidence: float,
    tolerance: float,
) -> dict[str, Any]:
    """Compare method rows with per-estimand complete-pair alignment."""
    specs: tuple[tuple[str, Callable[[SampleRow], float | None], Direction], ...] = (
        (
            f"f1_at_{primary_k}",
            lambda row: _agreement_value(row, "f1", primary_k),
            "higher",
        ),
        (
            "mean_average_precision",
            lambda row: _agreement_value(row, "ap", primary_k),
            "higher",
        ),
        (
            "ndcg_at_10",
            lambda row: _agreement_value(row, "ndcg_at_10", primary_k),
            "higher",
        ),
        ("sufficiency", lambda row: _diagnostic_value(row, "sufficiency"), "lower"),
        (
            "comprehensiveness",
            lambda row: _diagnostic_value(row, "comprehensiveness"),
            "higher",
        ),
    )
    metrics: dict[str, Any] = {}
    for metric_name, extractor, direction in specs:
        metrics[metric_name] = paired_bootstrap_difference(
            _values(candidate_rows, extractor),
            _values(comparator_rows, extractor),
            global_seed=global_seed,
            seed_parts=(split, mode, candidate_method, comparator_method, metric_name),
            resamples=resamples,
            confidence=confidence,
            favorable_direction=direction,
            tolerance=tolerance,
        )
    return {
        "candidate_method": candidate_method,
        "comparator_method": comparator_method,
        "analysis_tier": analysis_tier,
        "metrics": metrics,
    }
