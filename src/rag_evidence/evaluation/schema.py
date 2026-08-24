"""Strict schema-v2 models for attribution estimands and execution metadata."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ValidationStatus = Literal["passed", "failed", "not_run"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PRFSummary(_StrictModel):
    p: float | None
    r: float | None
    f1: float | None


class AgreementSummary(_StrictModel):
    n: int = Field(ge=0)
    subset: str
    prf_at: dict[str, PRFSummary]
    mean_average_precision: float | None
    ndcg_at_10: float | None


class CausalDependenceSummary(_StrictModel):
    n: int = Field(ge=0)
    subset: str
    sufficiency_mean: float | None
    comprehensiveness_mean: float | None
    validation_status: ValidationStatus
    interpretation: Literal["diagnostic_not_causal_effect"] = "diagnostic_not_causal_effect"


class ConstructValidationSummary(_StrictModel):
    status: ValidationStatus
    missing_methods: tuple[str, ...]
    invalid_methods: dict[str, str]
    comparisons: dict[str, Any]

    @model_validator(mode="after")
    def _status_is_fail_closed(self) -> ConstructValidationSummary:
        if self.status == "passed" and (self.missing_methods or self.invalid_methods):
            raise ValueError("construct validation cannot pass with missing/invalid methods")
        if self.status == "not_run" and self.comparisons:
            raise ValueError("not-run construct validation cannot contain comparisons")
        return self


class ExecutionSummary(_StrictModel):
    n_attempted: int = Field(ge=0)
    n_success: int = Field(ge=0)
    n_failed: int = Field(ge=0)
    n_skipped: int = Field(ge=0)
    n_record_attempts: int = Field(ge=0)
    n_retry_records: int = Field(ge=0)
    n_historical_failure_records: int = Field(ge=0)
    failure_rate: float = Field(ge=0.0, le=1.0)
    seconds_per_sample: float | None
    num_model_calls_mean: float | None
    peak_vram_mb: float | None

    @model_validator(mode="after")
    def _attempts_equal_outcomes(self) -> ExecutionSummary:
        if self.n_attempted != self.n_success + self.n_failed:
            raise ValueError("execution.n_attempted must equal n_success + n_failed")
        return self


class AttributionSummaryV2(_StrictModel):
    schema_version: Literal[2] = 2
    run: str
    execution_kind: Literal["real", "mock"]
    partial: bool
    device: str
    method: str
    run_namespace: str | None
    generation_run: str | None
    is_control: bool
    agreement: AgreementSummary
    causal_dependence: CausalDependenceSummary
    execution: ExecutionSummary
    legacy_v1: dict[str, Any]

    @model_validator(mode="after")
    def _estimand_counts_do_not_exceed_successes(self) -> AttributionSummaryV2:
        if self.agreement.n > self.execution.n_success:
            raise ValueError("agreement.n must not exceed execution.n_success")
        if self.causal_dependence.n > self.execution.n_success:
            raise ValueError("causal_dependence.n must not exceed execution.n_success")
        return self


def upgrade_attribution_metrics(
    legacy: Mapping[str, Any],
    *,
    agreement_subset: str,
    causal_validation_status: ValidationStatus,
) -> dict[str, Any]:
    """Convert one flattened schema-v1 method aggregate into validated schema v2."""
    required = (
        "run",
        "execution_kind",
        "partial",
        "device",
        "method",
        "is_control",
        "n_attempted",
        "n_success",
        "n_failed",
        "n_skipped",
        "n_record_attempts",
        "n_retry_records",
        "n_historical_failure_records",
        "failure_rate",
        "n_agreement",
        "prf_at",
        "auprc",
        "ndcg_at_10",
        "sufficiency",
        "comprehensiveness",
        "n_faithfulness",
        "seconds_per_sample",
        "num_model_calls_mean",
        "peak_vram_mb",
    )
    missing = [field for field in required if field not in legacy]
    if missing:
        raise ValueError(f"legacy attribution metrics missing required fields: {missing}")

    model = AttributionSummaryV2(
        run=str(legacy["run"]),
        execution_kind=legacy["execution_kind"],
        partial=bool(legacy["partial"]),
        device=str(legacy["device"]),
        method=str(legacy["method"]),
        run_namespace=legacy.get("run_namespace"),
        generation_run=legacy.get("generation_run"),
        is_control=bool(legacy["is_control"]),
        agreement=AgreementSummary(
            n=legacy["n_agreement"],
            subset=agreement_subset,
            prf_at=legacy["prf_at"],
            mean_average_precision=legacy["auprc"],
            ndcg_at_10=legacy["ndcg_at_10"],
        ),
        causal_dependence=CausalDependenceSummary(
            n=legacy["n_faithfulness"],
            subset="all_successful_non_abstained",
            sufficiency_mean=legacy["sufficiency"],
            comprehensiveness_mean=legacy["comprehensiveness"],
            validation_status=causal_validation_status,
        ),
        execution=ExecutionSummary(
            n_attempted=legacy["n_attempted"],
            n_success=legacy["n_success"],
            n_failed=legacy["n_failed"],
            n_skipped=legacy["n_skipped"],
            n_record_attempts=legacy["n_record_attempts"],
            n_retry_records=legacy["n_retry_records"],
            n_historical_failure_records=legacy["n_historical_failure_records"],
            failure_rate=legacy["failure_rate"],
            seconds_per_sample=legacy["seconds_per_sample"],
            num_model_calls_mean=legacy["num_model_calls_mean"],
            peak_vram_mb=legacy["peak_vram_mb"],
        ),
        legacy_v1=copy.deepcopy(dict(legacy)),
    )
    return model.model_dump(mode="json")
