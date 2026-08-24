"""Hand-checked Wilson, Holm, and nested cluster-bootstrap behavior."""

from __future__ import annotations

import pytest

from rag_evidence.annotation.statistics import (
    NestedObservation,
    cluster_bootstrap_difference,
    holm_adjust,
    wilson_interval,
)


def test_wilson_interval_matches_known_binomial_values() -> None:
    low, high = wilson_interval(5, 10, confidence=0.95)

    assert low == pytest.approx(0.236593, abs=1e-6)
    assert high == pytest.approx(0.763407, abs=1e-6)
    assert wilson_interval(0, 0, confidence=0.95) == (None, None)


def test_holm_adjustment_is_monotone_in_sorted_p_value_order() -> None:
    adjusted = holm_adjust({"f1": 0.01, "map": 0.04, "citation": 0.03})

    assert adjusted == {
        "f1": pytest.approx(0.03),
        "map": pytest.approx(0.06),
        "citation": pytest.approx(0.06),
    }


def test_parent_nested_variants_are_averaged_before_cluster_resampling() -> None:
    observations = [
        NestedObservation(
            cluster_id="g1",
            parent_id="p-heavy",
            variant=f"v{i}",
            candidate=1.0,
            comparator=0.0,
        )
        for i in range(10)
    ] + [
        NestedObservation(
            cluster_id="g1",
            parent_id="p-light",
            variant="v0",
            candidate=-1.0,
            comparator=0.0,
        )
    ]

    result = cluster_bootstrap_difference(
        observations,
        seed=7,
        seed_parts=("hand-check",),
        resamples=1_000,
        confidence=0.95,
    )

    # Parent means are +1 and -1, so the parent-weighted estimand is exactly zero.
    # Treating eleven variants as independent rows would incorrectly produce 9/11.
    assert result.mean_delta == pytest.approx(0.0)
    assert result.ci_low == pytest.approx(0.0)
    assert result.ci_high == pytest.approx(0.0)
    assert result.n_clusters == 1
    assert result.n_parents == 2
    assert result.n_variants == 11


def test_bootstrap_samples_whole_leakage_groups_and_is_deterministic() -> None:
    observations = [
        NestedObservation("g-positive", "p1", "missing", 1.0, 0.0),
        NestedObservation("g-positive", "p1", "swap", 1.0, 0.0),
        NestedObservation("g-negative", "p2", "missing", -1.0, 0.0),
        NestedObservation("g-negative", "p2", "swap", -1.0, 0.0),
    ]
    kwargs = {
        "seed": 23,
        "seed_parts": ("eval", "loo", "lexical"),
        "resamples": 2_000,
        "confidence": 0.95,
    }

    first = cluster_bootstrap_difference(observations, **kwargs)
    repeated = cluster_bootstrap_difference(list(reversed(observations)), **kwargs)

    assert first == repeated
    assert first.mean_delta == pytest.approx(0.0)
    assert first.ci_low == pytest.approx(-1.0)
    assert first.ci_high == pytest.approx(1.0)
    assert first.n_clusters == 2


def test_incomplete_method_pairs_are_excluded_not_imputed() -> None:
    result = cluster_bootstrap_difference(
        [
            NestedObservation("g1", "p1", "missing", 1.0, 0.0),
            NestedObservation("g1", "p1", "swap", None, 0.0),
            NestedObservation("g2", "p2", "missing", 2.0, None),
        ],
        seed=1,
        seed_parts=("missing",),
        resamples=1_000,
        confidence=0.95,
    )

    assert result.n_variants == 1
    assert result.exclusions == {"candidate_missing": 1, "comparator_missing": 1}
    assert result.mean_delta == 1.0


@pytest.mark.parametrize(
    ("successes", "total"),
    [(-1, 10), (11, 10), (1, -1)],
)
def test_wilson_rejects_invalid_counts(successes: int, total: int) -> None:
    with pytest.raises(ValueError):
        wilson_interval(successes, total, confidence=0.95)
