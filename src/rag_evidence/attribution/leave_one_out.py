"""Leave-one-passage-out: the primary CAUSAL baseline.

raw_score(passage) = logprob(target | full context) − logprob(target | context − passage)
under teacher forcing. Higher = removing the passage hurts more = more causally
important; negative values (helpful-to-remove distractors) are legitimate and kept.

Aliases stay bound to passage identity across ablations (no renumbering), and the
target is citation-stripped upstream — ablations are never scored on predicting
citation tokens whose validity the ablation itself changes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rag_evidence.attribution.base import AttributionMethod, AttributionResult, ModelResources
from rag_evidence.attribution.registry import register
from rag_evidence.attribution.scoring import LogprobScorer
from rag_evidence.data.schema import Passage
from rag_evidence.errors import UpstreamMissingError


@register
class LeaveOneOutAttribution(AttributionMethod):
    name = "leave_one_out"
    requires_generator = True

    def attribute(
        self,
        question: str,
        passages: Sequence[Passage],
        target_answer: str,
        model: ModelResources | None,
        method_config: Mapping[str, Any],
    ) -> AttributionResult:
        scorer: LogprobScorer | None = method_config.get("scorer")
        if scorer is None:
            raise UpstreamMissingError("leave_one_out needs a LogprobScorer in method_config")
        question_id: str = method_config["question_id"]
        alias_map: dict[str, str] = method_config["alias_map"]

        calls_before = scorer.calls
        s_full = scorer.score(question_id, question, passages, alias_map, target_answer)
        raw: dict[str, float] = {}
        ablation_scores: dict[str, dict[str, float | int]] = {}
        for held_out in passages:
            subset = [p for p in passages if p.passage_id != held_out.passage_id]
            s_without = scorer.score(question_id, question, subset, alias_map, target_answer)
            raw[held_out.passage_id] = s_full.sum_logprob - s_without.sum_logprob
            ablation_scores[held_out.passage_id] = s_without.to_json()

        return AttributionResult(
            raw_scores=raw,
            metadata={
                "s_full": s_full.to_json(),
                "ablations": ablation_scores,
                "score_definition": "sum_logprob(full) - sum_logprob(minus passage)",
            },
            num_model_calls=scorer.calls - calls_before,
        )
