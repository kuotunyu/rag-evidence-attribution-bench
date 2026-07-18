"""Attribution agreement metrics vs supporting-fact gold passages.

All rankings use the single deterministic tie-break of this package: descending raw
score, then original context position — this is what makes binary-scored methods
(citations, controls) reproducible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rag_evidence.metrics.retrieval import ndcg_at_k  # same binary-gain definition


def rank_passages(raw_scores: Mapping[str, float], original_order: Sequence[str]) -> list[str]:
    pos = {pid: i for i, pid in enumerate(original_order)}
    return sorted(raw_scores, key=lambda pid: (-raw_scores[pid], pos[pid]))


def prf_at_k(ranking: Sequence[str], gold: set[str], k: int) -> dict[str, float]:
    if k <= 0:
        raise ValueError("k must be positive")
    top = set(ranking[:k])
    hits = len(top & gold)
    precision = hits / k
    recall = hits / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"p": precision, "r": recall, "f1": f1}


def average_precision(ranking: Sequence[str], gold: set[str]) -> float:
    """AP over the full ranking; positives = gold supporting passages. Macro-averaging
    per-sample APs across a split is what summary.json reports as AUPRC."""
    if not gold:
        raise ValueError("gold set must not be empty")
    hits = 0
    precision_sum = 0.0
    for i, pid in enumerate(ranking, start=1):
        if pid in gold:
            hits += 1
            precision_sum += hits / i
    return precision_sum / len(gold)


def attribution_ndcg(ranking: Sequence[str], gold: set[str], k: int = 10) -> float:
    return ndcg_at_k(ranking, gold, k)
