"""Dense retrieval with Qwen3-Embedding: cosine(query, passage) over the candidate set.

Vectors are L2-normalized by the embedder, so cosine reduces to a dot product.
Passage vectors go through the shared EmbeddingCache (also reused by the embedding
attribution method)."""

from __future__ import annotations

from rag_evidence.data.schema import Example
from rag_evidence.embeddings import Embedder, EmbeddingCache
from rag_evidence.retrieval.base import RankedPassage, to_ranked


def passage_embed_text(title: str, text: str) -> str:
    """Single canonical passage rendering for ALL embedding consumers."""
    return f"{title}\n{text}"


class DenseRetriever:
    name = "dense"

    def __init__(self, embedder: Embedder, cache: EmbeddingCache, batch_size: int = 8) -> None:
        self.embedder = embedder
        self.cache = cache
        self.batch_size = batch_size

    def rank(self, example: Example) -> list[RankedPassage]:
        passage_vecs = self.cache.get_or_compute(
            [p.passage_id for p in example.passages],
            [passage_embed_text(p.title, p.text) for p in example.passages],
            lambda texts: self.embedder.embed_passages(texts, batch_size=self.batch_size),
        )
        query_vec = self.embedder.embed_queries([example.question], batch_size=1)[0]
        scores = passage_vecs @ query_vec
        return to_ranked(
            [
                (p.passage_id, float(s))
                for p, s in zip(example.passages, scores, strict=True)
            ]
        )
