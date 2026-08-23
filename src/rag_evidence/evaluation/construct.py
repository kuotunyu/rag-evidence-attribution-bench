"""Fail-closed construct checks for target-dependence diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rag_evidence.evaluation.schema import ConstructValidationSummary
from rag_evidence.evaluation.statistics import Direction, paired_bootstrap_difference

_ORACLE = "oracle_gold"
_NEGATIVE_CONTROLS = ("control_random", "control_answer_string")


def _metric_values(rows: Mapping[str, Mapping[str, Any]], metric: str) -> dict[str, float | None]:
    return {
        qid: float(row[metric]) if row.get(metric) is not None else None
        for qid, row in rows.items()
    }


def evaluate_construct_validation(
    per_sample_by_method: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    method_status: Mapping[str, Mapping[str, Any]] | None = None,
    split: str,
    mode: str,
    global_seed: int,
    resamples: int,
    confidence: float,
    tolerance: float,
) -> dict[str, Any]:
    """Validate oracle interventions against random and lexical negatives."""
    required = {_ORACLE, *_NEGATIVE_CONTROLS}
    missing = sorted(required - set(per_sample_by_method))
    if missing:
        return ConstructValidationSummary(
            status="not_run",
            missing_methods=tuple(missing),
            invalid_methods={},
            comparisons={},
        ).model_dump(mode="json")

    invalid: dict[str, str] = {}
    for method in sorted(required):
        artifact_status = (method_status or {}).get(method, {})
        if bool(artifact_status.get("partial")):
            invalid[method] = "partial"
        elif artifact_status.get("eligible") is False:
            invalid[method] = "ineligible"
        elif not per_sample_by_method[method]:
            invalid[method] = "empty"
    if invalid:
        gate_status = "failed" if "partial" in invalid.values() else "not_run"
        return ConstructValidationSummary(
            status=gate_status,
            missing_methods=(),
            invalid_methods=invalid,
            comparisons={},
        ).model_dump(mode="json")

    metric_specs: tuple[tuple[str, Direction], ...] = (
        ("sufficiency", "lower"),
        ("comprehensiveness", "higher"),
    )
    comparisons: dict[str, Any] = {}
    all_favorable = True
    oracle_rows = per_sample_by_method[_ORACLE]
    for comparator in _NEGATIVE_CONTROLS:
        metrics: dict[str, Any] = {}
        for metric, direction in metric_specs:
            result = paired_bootstrap_difference(
                _metric_values(oracle_rows, metric),
                _metric_values(per_sample_by_method[comparator], metric),
                global_seed=global_seed,
                seed_parts=(split, mode, _ORACLE, comparator, metric),
                resamples=resamples,
                confidence=confidence,
                favorable_direction=direction,
                tolerance=tolerance,
            )
            metrics[metric] = result
            all_favorable = all_favorable and result["favorable"] is True
        comparisons[f"{_ORACLE}__vs__{comparator}"] = metrics
    return ConstructValidationSummary(
        status="passed" if all_favorable else "failed",
        missing_methods=(),
        invalid_methods={},
        comparisons=comparisons,
    ).model_dump(mode="json")
