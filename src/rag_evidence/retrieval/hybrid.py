"""Hybrid retrieval: Reciprocal Rank Fusion over two STORED runs (bm25 + dense).

No recomputation — RRF consumes the ranks already on disk, which keeps the fusion
honest (it can only see what its sources produced) and makes it runnable on CPU
anywhere, including after a Colab import."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rag_evidence.retrieval.base import RankedPassage


def rrf_fuse(
    rankings: Sequence[Mapping[str, int]],
    *,
    rrf_k: int = 60,
    original_order: Sequence[str],
) -> list[RankedPassage]:
    """rankings: per source, {passage_id: rank (1-based)}. Score = Σ 1/(rrf_k + rank).

    Ties break by position in `original_order` (the example's dataset passage order),
    consistent with every other ranking in this package.
    """
    if not rankings:
        raise ValueError("rrf_fuse needs at least one source ranking")
    ids = set(original_order)
    for i, src in enumerate(rankings):
        if set(src) != ids:
            raise ValueError(
                f"source {i} ranks a different passage set than the example "
                f"({sorted(src)} vs {sorted(ids)})"
            )
    scores = {
        pid: sum(1.0 / (rrf_k + src[pid]) for src in rankings) for pid in original_order
    }
    ordered = sorted(
        enumerate(original_order), key=lambda t: (-scores[t[1]], t[0])
    )
    return [
        RankedPassage(passage_id=pid, score=scores[pid], rank=i + 1)
        for i, (_pos, pid) in enumerate(ordered)
    ]
