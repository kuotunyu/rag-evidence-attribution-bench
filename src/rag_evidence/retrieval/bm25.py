"""BM25 over a question's candidate passages (rank_bm25; index rebuilt per question —
trivially cheap at 10 passages and keeps k1/b honest per-query parameters)."""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from rag_evidence.data.schema import Example
from rag_evidence.retrieval.base import RankedPassage, to_ranked, tokenize


class BM25Retriever:
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def rank(self, example: Example) -> list[RankedPassage]:
        corpus = [tokenize(f"{p.title} {p.text}") for p in example.passages]
        bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)
        scores = bm25.get_scores(tokenize(example.question))
        return to_ranked(
            [(p.passage_id, float(s)) for p, s in zip(example.passages, scores, strict=True)]
        )
