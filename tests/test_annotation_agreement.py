"""Hand-calculated inter-annotator and evidence-set agreement fixtures."""

from __future__ import annotations

import pytest

from rag_evidence.annotation.agreement import evidence_agreement, nominal_agreement


def test_nominal_agreement_reports_missingness_prevalence_kappa_and_alpha() -> None:
    result = nominal_agreement(
        [
            ("answerable", "answerable"),
            ("answerable", "unanswerable"),
            ("unanswerable", "unanswerable"),
            (None, "answerable"),
        ]
    )

    assert result.n_total == 4
    assert result.n_complete == 3
    assert result.n_missing == 1
    assert result.raw_agreement == pytest.approx(2 / 3)
    assert result.cohen_kappa == pytest.approx(0.4)
    # Pooled available labels: answerable=4, unanswerable=3; Do=1/3, De=4/7.
    assert result.krippendorff_alpha == pytest.approx(5 / 12)
    assert result.left_prevalence == {
        "answerable": pytest.approx(2 / 3),
        "unanswerable": pytest.approx(1 / 3),
    }
    assert result.right_prevalence == {
        "answerable": pytest.approx(1 / 3),
        "unanswerable": pytest.approx(2 / 3),
    }
    assert result.pooled_prevalence == {
        "answerable": pytest.approx(4 / 7),
        "unanswerable": pytest.approx(3 / 7),
    }


def test_missing_pairs_are_not_negative_labels() -> None:
    result = nominal_agreement([(None, "answerable"), ("unanswerable", None)])

    assert result.n_complete == 0
    assert result.n_missing == 2
    assert result.raw_agreement is None
    assert result.cohen_kappa is None
    assert result.krippendorff_alpha is None


def test_perfect_single_class_agreement_is_raw_perfect_but_kappa_undefined() -> None:
    result = nominal_agreement([("unclear", "unclear"), ("unclear", "unclear")])

    assert result.raw_agreement == 1.0
    assert result.cohen_kappa is None
    assert result.krippendorff_alpha == 1.0


def test_evidence_agreement_uses_exact_family_and_symmetric_best_matching() -> None:
    result = evidence_agreement(
        left_sets=[{"P1.S1", "P2.S1"}, {"P3.S1"}],
        right_sets=[{"P2.S1", "P1.S1"}, {"P3.S1", "P3.S2"}],
    )

    assert result.exact_set_family_agreement is False
    assert result.jaccard == pytest.approx(0.75)
    assert result.set_f1 == pytest.approx(5 / 6)


def test_evidence_agreement_handles_empty_families_explicitly() -> None:
    both_empty = evidence_agreement([], [])
    one_empty = evidence_agreement([{"P1.S1"}], [])

    assert both_empty.exact_set_family_agreement is True
    assert both_empty.jaccard == 1.0
    assert both_empty.set_f1 == 1.0
    assert one_empty.exact_set_family_agreement is False
    assert one_empty.jaccard == 0.0
    assert one_empty.set_f1 == 0.0
