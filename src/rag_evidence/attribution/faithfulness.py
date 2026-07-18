"""Sufficiency / comprehensiveness (ERASER conventions, teacher-forced logprobs).

Let m(ctx) = MEAN per-token logprob of the (citation-stripped) target under ctx.
    sufficiency        = m(full) − m(top-k only)        — LOWER is better
    comprehensiveness  = m(full) − m(full − top-k)      — HIGHER is better

Computed inside the `attribute` stage (the generator is already resident) and stored
as raw numbers per sample, so `evaluate` stays pure CPU arithmetic. Mean per-token
(not sum) so cross-sample averages aren't dominated by long answers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rag_evidence.attribution.scoring import LogprobScorer
from rag_evidence.data.schema import Passage


def top_k_ids(
    raw_scores: Mapping[str, float], original_order: Sequence[str], k: int
) -> list[str]:
    """Top-k passage ids by raw score; ties break by original context position."""
    pos = {pid: i for i, pid in enumerate(original_order)}
    ordered = sorted(raw_scores, key=lambda pid: (-raw_scores[pid], pos[pid]))
    return ordered[:k]


def compute_faithfulness(
    *,
    question_id: str,
    question: str,
    passages: Sequence[Passage],
    alias_map: dict[str, str],
    target_answer: str,
    raw_scores: Mapping[str, float],
    scorer: LogprobScorer,
    k: int,
) -> dict[str, Any]:
    order = [p.passage_id for p in passages]
    top = set(top_k_ids(raw_scores, order, k))
    keep_top = [p for p in passages if p.passage_id in top]
    keep_rest = [p for p in passages if p.passage_id not in top]

    m_full = scorer.score(question_id, question, passages, alias_map, target_answer).mean_logprob
    m_top = scorer.score(question_id, question, keep_top, alias_map, target_answer).mean_logprob
    m_rest = scorer.score(question_id, question, keep_rest, alias_map, target_answer).mean_logprob

    return {
        "k": k,
        "top_k_passage_ids": [p.passage_id for p in keep_top],
        "m_full": m_full,
        "m_top_k": m_top,
        "m_without_top_k": m_rest,
        "sufficiency": m_full - m_top,
        "comprehensiveness": m_full - m_rest,
    }
