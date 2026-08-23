"""Auditable human-agreement metrics with explicit missingness and prevalence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence, Set
from dataclasses import dataclass

from rag_evidence.annotation.models import AnswerabilityAnnotation


@dataclass(frozen=True)
class NominalAgreement:
    n_total: int
    n_complete: int
    n_missing: int
    raw_agreement: float | None
    cohen_kappa: float | None
    krippendorff_alpha: float | None
    left_prevalence: dict[str, float]
    right_prevalence: dict[str, float]
    pooled_prevalence: dict[str, float]


@dataclass(frozen=True)
class EvidenceAgreement:
    exact_set_family_agreement: bool
    jaccard: float
    set_f1: float


@dataclass(frozen=True)
class EvidenceAgreementSummary:
    n_total_tasks: int
    n_comparable: int
    n_excluded_not_both_answerable: int
    n_invalid_empty_family: int
    exact_set_family_agreement_rate: float | None
    mean_jaccard: float | None
    mean_set_f1: float | None


def _prevalence(values: Iterable[str]) -> dict[str, float]:
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {label: counts[label] / total for label in sorted(counts)}


def nominal_agreement(
    pairs: Sequence[tuple[str | None, str | None]],
) -> NominalAgreement:
    """Two-coder nominal agreement; incomplete pairs never become a negative class."""
    complete = [(left, right) for left, right in pairs if left is not None and right is not None]
    left_complete = [left for left, _right in complete]
    right_complete = [right for _left, right in complete]
    pooled_available = [value for pair in pairs for value in pair if value is not None]
    n_complete = len(complete)
    if n_complete == 0:
        return NominalAgreement(
            n_total=len(pairs),
            n_complete=0,
            n_missing=len(pairs),
            raw_agreement=None,
            cohen_kappa=None,
            krippendorff_alpha=None,
            left_prevalence={},
            right_prevalence={},
            pooled_prevalence=_prevalence(pooled_available),
        )

    agreements = sum(left == right for left, right in complete)
    observed = agreements / n_complete
    left_prevalence = _prevalence(left_complete)
    right_prevalence = _prevalence(right_complete)
    labels = set(left_prevalence) | set(right_prevalence)
    expected = sum(
        left_prevalence.get(label, 0.0) * right_prevalence.get(label, 0.0) for label in labels
    )
    kappa = None if expected == 1.0 else (observed - expected) / (1.0 - expected)

    pooled_counts = Counter(pooled_available)
    pooled_total = sum(pooled_counts.values())
    expected_disagreement = (
        1.0
        - sum(count * (count - 1) for count in pooled_counts.values())
        / (pooled_total * (pooled_total - 1))
        if pooled_total > 1
        else 0.0
    )
    observed_disagreement = 1.0 - observed
    if expected_disagreement == 0.0:
        alpha = 1.0 if observed_disagreement == 0.0 else None
    else:
        alpha = 1.0 - observed_disagreement / expected_disagreement
    return NominalAgreement(
        n_total=len(pairs),
        n_complete=n_complete,
        n_missing=len(pairs) - n_complete,
        raw_agreement=observed,
        cohen_kappa=kappa,
        krippendorff_alpha=alpha,
        left_prevalence=left_prevalence,
        right_prevalence=right_prevalence,
        pooled_prevalence=_prevalence(pooled_available),
    )


def _canonical_family(sets: Sequence[Set[str]]) -> tuple[tuple[str, ...], ...]:
    return tuple(sorted(tuple(sorted(evidence_set)) for evidence_set in sets))


def _jaccard(left: Set[str], right: Set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _set_f1(left: Set[str], right: Set[str]) -> float:
    denominator = len(left) + len(right)
    return 2 * len(left & right) / denominator if denominator else 1.0


def _symmetric_best_match(
    left: Sequence[Set[str]],
    right: Sequence[Set[str]],
    metric: Callable[[Set[str], Set[str]], float],
) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    left_best = [max(metric(item, other) for other in right) for item in left]
    right_best = [max(metric(item, other) for other in left) for item in right]
    return (sum(left_best) / len(left_best) + sum(right_best) / len(right_best)) / 2.0


def evidence_agreement(
    left_sets: Sequence[Set[str]],
    right_sets: Sequence[Set[str]],
) -> EvidenceAgreement:
    """Compare alternative minimal-evidence set families without merging alternatives."""
    return EvidenceAgreement(
        exact_set_family_agreement=_canonical_family(left_sets) == _canonical_family(right_sets),
        jaccard=_symmetric_best_match(left_sets, right_sets, _jaccard),
        set_f1=_symmetric_best_match(left_sets, right_sets, _set_f1),
    )


def aggregate_evidence_agreement(
    pairs: Sequence[
        tuple[AnswerabilityAnnotation | None, AnswerabilityAnnotation | None]
    ],
) -> EvidenceAgreementSummary:
    """Aggregate only valid pairs where both humans independently chose answerable."""
    comparable: list[EvidenceAgreement] = []
    excluded = 0
    invalid_empty = 0
    for left, right in pairs:
        if (
            left is None
            or right is None
            or left.answerability != "answerable"
            or right.answerability != "answerable"
        ):
            excluded += 1
            continue
        if (
            not left.minimal_sufficient_evidence_sets
            or not right.minimal_sufficient_evidence_sets
        ):
            invalid_empty += 1
            continue
        comparable.append(
            evidence_agreement(
                [set(evidence_set) for evidence_set in left.minimal_sufficient_evidence_sets],
                [set(evidence_set) for evidence_set in right.minimal_sufficient_evidence_sets],
            )
        )
    count = len(comparable)
    return EvidenceAgreementSummary(
        n_total_tasks=len(pairs),
        n_comparable=count,
        n_excluded_not_both_answerable=excluded,
        n_invalid_empty_family=invalid_empty,
        exact_set_family_agreement_rate=(
            sum(item.exact_set_family_agreement for item in comparable) / count
            if count
            else None
        ),
        mean_jaccard=(sum(item.jaccard for item in comparable) / count if count else None),
        mean_set_f1=(sum(item.set_f1 for item in comparable) / count if count else None),
    )
