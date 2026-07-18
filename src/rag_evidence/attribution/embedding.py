"""Embedding-relevance attribution: cosine(query, passage) with Qwen3-Embedding.

Query text is `"{question} {answer}"`: answer-only degenerates on HotpotQA's many
yes/no and 1–3 token answers (embedding "yes" carries no evidence signal). The
question-only ablation is visible in the same tables via the retrieval-rank control,
so the answer's marginal contribution stays measurable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rag_evidence.attribution.base import AttributionMethod, AttributionResult, ModelResources
from rag_evidence.attribution.registry import register
from rag_evidence.data.schema import Passage
from rag_evidence.errors import UpstreamMissingError


@register
class EmbeddingAttribution(AttributionMethod):
    name = "embedding"
    requires_embedder = True

    def attribute(
        self,
        question: str,
        passages: Sequence[Passage],
        target_answer: str,
        model: ModelResources | None,
        method_config: Mapping[str, Any],
    ) -> AttributionResult:
        if model is None or model.embedder is None:
            raise UpstreamMissingError("embedding method needs an embedder")
        from rag_evidence.retrieval.dense import passage_embed_text

        query = f"{question.strip()} {target_answer.strip()}".strip()
        texts = [passage_embed_text(p.title, p.text) for p in passages]
        keys = [p.passage_id for p in passages]
        if model.embedding_cache is not None:
            vectors = model.embedding_cache.get_or_compute(
                keys, texts, lambda t: model.embedder.embed_passages(t)
            )
        else:
            vectors = model.embedder.embed_passages(texts)
        query_vec = model.embedder.embed_queries([query])[0]
        sims = vectors @ query_vec
        raw = {pid: float(s) for pid, s in zip(keys, sims, strict=True)}
        return AttributionResult(
            raw_scores=raw,
            metadata={"query_text_convention": "question + ' ' + answer"},
            num_model_calls=0,
        )
