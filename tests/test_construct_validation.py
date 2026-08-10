"""Causal-dependence construct validation over prespecified controls."""

from __future__ import annotations

from rag_evidence.evaluation.construct import evaluate_construct_validation


def _rows(sufficiency: float, comprehensiveness: float) -> dict[str, dict[str, float]]:
    return {
        "q1": {
            "sufficiency": sufficiency,
            "comprehensiveness": comprehensiveness,
        },
        "q2": {
            "sufficiency": sufficiency,
            "comprehensiveness": comprehensiveness,
        },
    }


def _evaluate(methods: dict[str, dict[str, dict[str, float]]]) -> dict[str, object]:
    return evaluate_construct_validation(
        methods,
        split="eval",
        mode="gold",
        global_seed=42,
        resamples=1_000,
        confidence=0.95,
        tolerance=0.0,
    )


def test_construct_validation_is_not_run_when_required_artifacts_are_missing() -> None:
    result = _evaluate({"control_random": _rows(2.0, 2.0)})

    assert result["status"] == "not_run"
    assert result["missing_methods"] == ["control_answer_string", "oracle_gold"]
    assert result["comparisons"] == {}


def test_construct_validation_passes_only_when_oracle_beats_both_negatives() -> None:
    result = _evaluate(
        {
            "oracle_gold": _rows(1.0, 4.0),
            "control_random": _rows(2.0, 2.0),
            "control_answer_string": _rows(3.0, 1.0),
        }
    )

    assert result["status"] == "passed"
    assert result["missing_methods"] == []
    comparisons = result["comparisons"]
    for comparator in ("control_random", "control_answer_string"):
        metrics = comparisons[f"oracle_gold__vs__{comparator}"]
        assert metrics["sufficiency"]["favorable"] is True
        assert metrics["comprehensiveness"]["favorable"] is True
        assert metrics["sufficiency"]["n_pairs"] == 2


def test_construct_validation_fails_on_wrong_direction() -> None:
    result = _evaluate(
        {
            "oracle_gold": _rows(1.0, 0.0),
            "control_random": _rows(2.0, 2.0),
            "control_answer_string": _rows(3.0, 1.0),
        }
    )

    assert result["status"] == "failed"
    failed = result["comparisons"]["oracle_gold__vs__control_random"]
    assert failed["comprehensiveness"]["favorable"] is False
