"""Retrieval metrics vs hand-computed values; BM25/dense/RRF behavior on the fixture."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from rag_evidence.data.hotpot import build_example, normalize_hf_example
from rag_evidence.metrics.retrieval import mrr, ndcg_at_k, percentiles, recall_at_k
from rag_evidence.retrieval.base import to_ranked
from rag_evidence.retrieval.bm25 import BM25Retriever
from rag_evidence.retrieval.hybrid import rrf_fuse

RANKING = ["a", "b", "c", "d"]


def test_recall_at_k_hand_values() -> None:
    assert recall_at_k(RANKING, {"a", "c"}, 2) == 0.5
    assert recall_at_k(RANKING, {"a", "c"}, 3) == 1.0
    assert recall_at_k(RANKING, {"x", "y"}, 4) == 0.0


def test_mrr_hand_values() -> None:
    assert mrr(RANKING, {"a", "c"}) == 1.0
    assert mrr(RANKING, {"b", "d"}) == 0.5
    assert mrr(RANKING, {"zzz"}) == 0.0


def test_ndcg_hand_value() -> None:
    # gold at ranks 1 and 3: DCG = 1/log2(2) + 1/log2(4) = 1.5
    # IDCG (2 golds) = 1/log2(2) + 1/log2(3)
    expected = 1.5 / (1.0 + 1.0 / math.log2(3))
    assert ndcg_at_k(RANKING, {"a", "c"}, 4) == pytest.approx(expected)
    assert ndcg_at_k(RANKING, {"a", "b"}, 4) == pytest.approx(1.0)


def test_percentiles_simple() -> None:
    s = percentiles([10.0, 20.0, 30.0, 40.0])
    assert s["mean"] == 25.0
    assert s["p50"] == 20.0
    assert s["p95"] == 40.0


def test_to_ranked_deterministic_tiebreak() -> None:
    ranked = to_ranked([("p0", 1.0), ("p1", 2.0), ("p2", 2.0)])
    assert [r.passage_id for r in ranked] == ["p1", "p2", "p0"]  # tie: original order
    assert [r.rank for r in ranked] == [1, 2, 3]


def _example(fixtures_dir: Path, i: int):
    rows = json.loads((fixtures_dir / "tiny_hotpot.json").read_text(encoding="utf-8"))["rows"]
    return build_example(normalize_hf_example(rows[i]))


def test_bm25_ranks_gold_top2_on_fixture(fixtures_dir: Path) -> None:
    ex = _example(fixtures_dir, 0)  # question shares distinctive terms with gold passages
    ranking = BM25Retriever().rank(ex)
    assert {r.passage_id for r in ranking[:2]} == set(ex.gold_passage_ids)
    assert [r.rank for r in ranking] == list(range(1, len(ex.passages) + 1))


def test_rrf_fuse_hand_value() -> None:
    order = ["p0", "p1", "p2"]
    src_a = {"p0": 1, "p1": 2, "p2": 3}
    src_b = {"p0": 3, "p1": 1, "p2": 2}
    fused = rrf_fuse([src_a, src_b], rrf_k=60, original_order=order)
    score = {r.passage_id: r.score for r in fused}
    assert score["p0"] == pytest.approx(1 / 61 + 1 / 63)
    assert score["p1"] == pytest.approx(1 / 62 + 1 / 61)
    # p1 has the highest sum → rank 1
    assert fused[0].passage_id == "p1"
    with pytest.raises(ValueError, match="different passage set"):
        rrf_fuse([{"p0": 1}], original_order=order)


class _StubEmbedder:
    """Bag-of-words hash embedding — deterministic, overlap → similarity."""

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(64, dtype=np.float32)
        for tok in text.lower().split():
            v[hash(tok) % 64] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_queries(self, texts, batch_size: int = 1):
        return np.stack([self._vec(t) for t in texts])

    def embed_passages(self, texts, batch_size: int = 1):
        return np.stack([self._vec(t) for t in texts])


def test_dense_retriever_with_stub_and_cache(fixtures_dir: Path, tmp_path: Path) -> None:
    from rag_evidence.embeddings import EmbeddingCache
    from rag_evidence.retrieval.dense import DenseRetriever

    ex = _example(fixtures_dir, 2)
    cache_path = tmp_path / "cache.npz"
    retriever = DenseRetriever(_StubEmbedder(), EmbeddingCache(cache_path), batch_size=4)  # type: ignore[arg-type]
    ranking1 = retriever.rank(ex)
    assert cache_path.exists()
    # second call must hit the cache and produce identical output
    retriever2 = DenseRetriever(_StubEmbedder(), EmbeddingCache(cache_path), batch_size=4)  # type: ignore[arg-type]
    ranking2 = retriever2.rank(ex)
    assert [r.passage_id for r in ranking1] == [r.passage_id for r in ranking2]
    assert len(ranking1) == len(ex.passages)
