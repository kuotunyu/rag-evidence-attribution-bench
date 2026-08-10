"""Paired uncertainty over question-aligned attribution metrics."""

from __future__ import annotations

import pytest

from rag_evidence.evaluation.statistics import paired_bootstrap_difference


def test_paired_bootstrap_aligns_question_ids_and_counts_exclusions() -> None:
    result = paired_bootstrap_difference(
        {"q1": 0.8, "q2": None, "q3": 0.4},
        {"q1": 0.5, "q2": 0.2, "q4": 0.1},
        global_seed=42,
        seed_parts=("eval", "gold", "leave_one_out", "control_lexical", "f1_at_2"),
        resamples=1_000,
        confidence=0.95,
        favorable_direction="higher",
        tolerance=0.0,
    )

    assert result["n_pairs"] == 1
    assert result["n_candidate"] == 2
    assert result["n_comparator"] == 3
    assert result["mean_delta"] == pytest.approx(0.3)
    assert result["ci_low"] == pytest.approx(0.3)
    assert result["ci_high"] == pytest.approx(0.3)
    assert result["exclusions"] == {
        "candidate_missing": 1,
        "comparator_missing": 1,
        "candidate_null": 1,
        "comparator_null": 0,
    }


def test_paired_bootstrap_is_deterministic_and_mapping_order_independent() -> None:
    kwargs = {
        "global_seed": 7,
        "seed_parts": ("dev", "generated", "embedding", "control_random", "map"),
        "resamples": 1_000,
        "confidence": 0.95,
        "favorable_direction": "higher",
        "tolerance": 0.0,
    }
    first = paired_bootstrap_difference(
        {"q1": 0.8, "q2": 0.1, "q3": 0.7},
        {"q1": 0.5, "q2": 0.4, "q3": 0.2},
        **kwargs,
    )
    reordered = paired_bootstrap_difference(
        {"q3": 0.7, "q1": 0.8, "q2": 0.1},
        {"q2": 0.4, "q3": 0.2, "q1": 0.5},
        **kwargs,
    )

    assert first == reordered


def test_lower_direction_treats_negative_interval_as_favorable() -> None:
    result = paired_bootstrap_difference(
        {"q1": 1.0, "q2": 2.0},
        {"q1": 2.0, "q2": 3.0},
        global_seed=11,
        seed_parts=("eval", "gold", "leave_one_out", "control_lexical", "sufficiency"),
        resamples=1_000,
        confidence=0.95,
        favorable_direction="lower",
        tolerance=0.0,
    )

    assert result["mean_delta"] == -1.0
    assert result["ci_low"] == -1.0
    assert result["ci_high"] == -1.0
    assert result["ci_excludes_zero"] is True
    assert result["favorable"] is True


def test_no_complete_pairs_returns_null_effect_instead_of_zero() -> None:
    result = paired_bootstrap_difference(
        {"q1": None},
        {"q2": 0.5},
        global_seed=1,
        seed_parts=("smoke", "gold", "a", "b", "metric"),
        resamples=1_000,
        confidence=0.95,
        favorable_direction="higher",
        tolerance=0.0,
    )

    assert result["n_pairs"] == 0
    assert result["mean_delta"] is None
    assert result["ci_low"] is None
    assert result["ci_high"] is None
    assert result["ci_excludes_zero"] is None
    assert result["favorable"] is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"resamples": 0}, "resamples"),
        ({"confidence": 0.0}, "confidence"),
        ({"confidence": 1.0}, "confidence"),
        ({"tolerance": -0.1}, "tolerance"),
        ({"favorable_direction": "sideways"}, "favorable_direction"),
    ],
)
def test_invalid_bootstrap_options_are_rejected(overrides: dict[str, object], message: str) -> None:
    options: dict[str, object] = {
        "global_seed": 1,
        "seed_parts": ("smoke", "gold", "a", "b", "metric"),
        "resamples": 1_000,
        "confidence": 0.95,
        "favorable_direction": "higher",
        "tolerance": 0.0,
    }
    options.update(overrides)

    with pytest.raises(ValueError, match=message):
        paired_bootstrap_difference({"q": 1.0}, {"q": 0.0}, **options)  # type: ignore[arg-type]
