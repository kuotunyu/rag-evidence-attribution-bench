"""Schema-v2 attribution estimands and denominator invariants."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from rag_evidence.evaluation.schema import AttributionSummaryV2, upgrade_attribution_metrics

LEGACY: dict[str, object] = {
    "run": "results/raw/eval/attribute/generated/leave_one_out",
    "execution_kind": "real",
    "partial": False,
    "device": "NVIDIA A100",
    "method": "leave_one_out",
    "run_namespace": None,
    "generation_run": "qwen3-4b",
    "subset": {"criterion": "em", "n_correct": 87},
    "is_control": False,
    "n_attempted": 192,
    "n_success": 192,
    "n_failed": 0,
    "n_skipped": 48,
    "n_record_attempts": 240,
    "n_retry_records": 0,
    "n_historical_failure_records": 0,
    "failure_rate": 0.0,
    "n_agreement": 87,
    "prf_at": {"2": {"p": 0.793, "r": 0.793, "f1": 0.793}},
    "auprc": 0.857,
    "ndcg_at_10": 0.921,
    "sufficiency": 0.860,
    "comprehensiveness": 7.156,
    "n_faithfulness": 192,
    "seconds_per_sample": 0.650,
    "num_model_calls_mean": 13.0,
    "peak_vram_mb": 9109.7,
}


def test_upgrade_separates_estimands_and_preserves_legacy() -> None:
    upgraded = upgrade_attribution_metrics(
        LEGACY,
        agreement_subset="generated_exact_match",
        causal_validation_status="not_run",
    )

    assert upgraded["schema_version"] == 2
    assert upgraded["agreement"]["n"] == 87
    assert upgraded["agreement"]["mean_average_precision"] == 0.857
    assert upgraded["causal_dependence"]["n"] == 192
    assert upgraded["causal_dependence"]["validation_status"] == "not_run"
    assert upgraded["execution"]["n_success"] == 192
    assert upgraded["legacy_v1"] == LEGACY
    assert "auprc" not in upgraded


def test_upgrade_does_not_alias_mutable_legacy_data() -> None:
    source = copy.deepcopy(LEGACY)
    upgraded = upgrade_attribution_metrics(
        source,
        agreement_subset="generated_exact_match",
        causal_validation_status="not_run",
    )

    source["prf_at"] = {}

    assert upgraded["legacy_v1"]["prf_at"] == LEGACY["prf_at"]


@pytest.mark.parametrize("missing", ["n_agreement", "n_faithfulness"])
def test_upgrade_rejects_missing_estimand_denominator(missing: str) -> None:
    legacy = copy.deepcopy(LEGACY)
    legacy.pop(missing)

    with pytest.raises(ValueError, match=missing):
        upgrade_attribution_metrics(
            legacy,
            agreement_subset="generated_exact_match",
            causal_validation_status="not_run",
        )


@pytest.mark.parametrize("count_field", ["n_agreement", "n_faithfulness"])
def test_schema_rejects_estimand_count_above_success_count(count_field: str) -> None:
    legacy = copy.deepcopy(LEGACY)
    legacy[count_field] = 193

    with pytest.raises(ValidationError, match="n_success"):
        upgrade_attribution_metrics(
            legacy,
            agreement_subset="generated_exact_match",
            causal_validation_status="not_run",
        )


def test_schema_round_trip_is_stable_and_forbids_extra_keys() -> None:
    payload = upgrade_attribution_metrics(
        LEGACY,
        agreement_subset="generated_exact_match",
        causal_validation_status="not_run",
    )
    round_trip = AttributionSummaryV2.model_validate(payload).model_dump(mode="json")

    assert round_trip == payload

    payload["invented_metric"] = 1.0
    with pytest.raises(ValidationError, match="extra_forbidden"):
        AttributionSummaryV2.model_validate(payload)
