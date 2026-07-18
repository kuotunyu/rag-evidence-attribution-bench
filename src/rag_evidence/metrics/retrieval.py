"""Ranking metrics for retrieval (binary relevance: the gold supporting passages)."""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(ranking: Sequence[str], gold: set[str], k: int) -> float:
    if not gold:
        raise ValueError("gold set must not be empty")
    return len(set(ranking[:k]) & gold) / len(gold)


def mrr(ranking: Sequence[str], gold: set[str]) -> float:
    for i, pid in enumerate(ranking, start=1):
        if pid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranking: Sequence[str], gold: set[str], k: int) -> float:
    """Binary gains: DCG = Σ 1/log2(rank+1) over gold hits in the top k."""
    if not gold:
        raise ValueError("gold set must not be empty")
    dcg = sum(1.0 / math.log2(i + 1) for i, pid in enumerate(ranking[:k], start=1) if pid in gold)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg


def percentiles(values: Sequence[float]) -> dict[str, float]:
    """mean/p50/p95 summary used for latency fields throughout."""
    if not values:
        return {"mean": float("nan"), "p50": float("nan"), "p95": float("nan")}
    ordered = sorted(values)

    def pct(p: float) -> float:
        # nearest-rank on the sorted list; simple and deterministic
        idx = min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))
        return ordered[idx]

    return {
        "mean": sum(ordered) / len(ordered),
        "p50": pct(0.50),
        "p95": pct(0.95),
    }
