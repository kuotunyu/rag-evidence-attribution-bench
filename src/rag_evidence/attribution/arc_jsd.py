"""ARC-JSD (EXPERIMENTAL native port) — arXiv:2505.16415, ruizheliUOA/ARC_JSD.

Score(passage) = mean over target-token positions of the Jensen-Shannon divergence
between the model's next-token distributions with the full context vs. with the passage
removed (teacher-forced, so positions align across the two conditions). Higher = the
passage shifts the answer distribution more = more important.

STATUS: experimental until validated against the official implementation (the official
code pins transformers==4.43.3 and supports Qwen2.5/Gemma-2, not Qwen3 — see
legacy/arc_jsd/ for the reproduction environment and protocol). Differences from the
official setup are deliberate and documented: passage-level sources (their RAG setting
uses retrieved documents), our prompt template, Qwen3 instead of Qwen2.5.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from rag_evidence.attribution.base import AttributionMethod, AttributionResult, ModelResources
from rag_evidence.attribution.registry import register
from rag_evidence.data.schema import Passage
from rag_evidence.errors import UpstreamMissingError
from rag_evidence.generation.prompts import build_messages

_EPS = 1e-12


def jsd_from_logprobs(lp_p: np.ndarray, lp_q: np.ndarray) -> float:
    """Jensen-Shannon divergence (natural log) between two log-prob vectors."""
    p = np.exp(lp_p.astype(np.float64))
    q = np.exp(lp_q.astype(np.float64))
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * (np.log(p + _EPS) - np.log(m + _EPS))))
    kl_qm = float(np.sum(q * (np.log(q + _EPS) - np.log(m + _EPS))))
    return max(0.0, 0.5 * kl_pm + 0.5 * kl_qm)


def mean_positional_jsd(dist_full: np.ndarray, dist_ablated: np.ndarray) -> float:
    if dist_full.shape != dist_ablated.shape:
        raise ValueError(f"distribution shape mismatch: {dist_full.shape} vs {dist_ablated.shape}")
    return float(
        np.mean([jsd_from_logprobs(dist_full[i], dist_ablated[i]) for i in range(len(dist_full))])
    )


@register
class ArcJsdAttribution(AttributionMethod):
    name = "arc_jsd"
    requires_generator = True

    def attribute(
        self,
        question: str,
        passages: Sequence[Passage],
        target_answer: str,
        model: ModelResources | None,
        method_config: Mapping[str, Any],
    ) -> AttributionResult:
        if model is None or model.generator is None:
            raise UpstreamMissingError("arc_jsd needs the generator backend")
        alias_map: dict[str, str] = method_config["alias_map"]
        prompt_version: str = method_config.get("prompt_version", "v1")

        messages_full = build_messages(question, passages, alias_map, version=prompt_version)
        dist_full = np.asarray(model.generator.target_distributions(messages_full, target_answer))

        raw: dict[str, float] = {}
        for held_out in passages:
            subset = [p for p in passages if p.passage_id != held_out.passage_id]
            messages = build_messages(question, subset, alias_map, version=prompt_version)
            dist_ablated = np.asarray(model.generator.target_distributions(messages, target_answer))
            raw[held_out.passage_id] = mean_positional_jsd(dist_full, dist_ablated)

        return AttributionResult(
            raw_scores=raw,
            metadata={
                "experimental": True,
                "reference": "arXiv:2505.16415 (ARC-JSD); native Qwen3 port, unvalidated "
                "against the official Qwen2.5 implementation — see legacy/arc_jsd/",
                "score_definition": "mean positional JSD(full || minus passage), teacher-forced",
                "n_target_positions": int(dist_full.shape[0]),
            },
            num_model_calls=len(passages) + 1,
        )
