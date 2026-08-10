"""Embedding-relevance attribution under explicit query-source conventions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from rag_evidence.attribution.base import AttributionMethod, AttributionResult, ModelResources
from rag_evidence.attribution.query import QueryConvention, compose_query
from rag_evidence.attribution.registry import register
from rag_evidence.data.schema import Passage
from rag_evidence.errors import UpstreamMissingError


class _EmbeddingByQuery(AttributionMethod):
    convention: ClassVar[QueryConvention]
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

        query = compose_query(question, target_answer, self.convention)
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
            metadata={"query_text_convention": self.convention},
            num_model_calls=0,
        )


@register
class EmbeddingQuestion(_EmbeddingByQuery):
    name = "embedding_question"
    convention = "question"


@register
class EmbeddingAnswer(_EmbeddingByQuery):
    name = "embedding_answer"
    convention = "answer"


@register
class EmbeddingQuestionAnswer(_EmbeddingByQuery):
    name = "embedding_question_answer"
    convention = "question_answer"


@register
class EmbeddingAttribution(_EmbeddingByQuery):
    """Schema-v1 compatibility alias for `embedding_question_answer`."""

    name = "embedding"
    convention = "question_answer"
