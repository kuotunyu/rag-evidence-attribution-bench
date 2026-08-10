"""Explicit attribution query-source ablations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from rag_evidence.attribution.base import ModelResources
from rag_evidence.attribution.query import QueryConvention, compose_query
from rag_evidence.attribution.registry import get_method
from rag_evidence.data.schema import Passage


@pytest.mark.parametrize(
    ("convention", "expected"),
    [
        ("question", "Who built it?"),
        ("answer", "Nora Finch"),
        ("question_answer", "Who built it? Nora Finch"),
    ],
)
def test_compose_query_is_explicit(convention: QueryConvention, expected: str) -> None:
    assert compose_query(" Who built it? ", " Nora Finch ", convention) == expected


def test_compose_query_joins_nonempty_components_without_extra_whitespace() -> None:
    assert compose_query("  ", " Nora ", "question_answer") == "Nora"
    assert compose_query(" Who? ", "  ", "question_answer") == "Who?"


def test_compose_query_rejects_unknown_convention() -> None:
    with pytest.raises(ValueError, match="query convention"):
        compose_query("q", "a", "unknown")  # type: ignore[arg-type]


class _RecordingEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_queries(self, texts: list[str], batch_size: int = 1) -> np.ndarray:
        self.queries.extend(texts)
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    def embed_passages(self, texts: list[str], batch_size: int = 1) -> np.ndarray:
        return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)


def _passages() -> list[Passage]:
    return [
        Passage(
            passage_id="p0",
            index=0,
            title="Question clue",
            sentences=("Who built the bridge?",),
            is_gold=True,
        ),
        Passage(
            passage_id="p1",
            index=1,
            title="Answer clue",
            sentences=("Nora Finch",),
            is_gold=False,
        ),
    ]


@pytest.mark.parametrize(
    ("method_name", "expected_convention", "expected_query"),
    [
        ("embedding_question", "question", "Who built it?"),
        ("embedding_answer", "answer", "Nora Finch"),
        ("embedding_question_answer", "question_answer", "Who built it? Nora Finch"),
        ("embedding", "question_answer", "Who built it? Nora Finch"),
    ],
)
def test_embedding_variants_send_the_declared_query(
    method_name: str, expected_convention: str, expected_query: str
) -> None:
    embedder = _RecordingEmbedder()
    result = get_method(method_name).attribute(
        "Who built it?",
        _passages(),
        "Nora Finch",
        ModelResources(embedder=embedder),
        {},
    )

    assert embedder.queries == [expected_query]
    assert result.metadata["query_text_convention"] == expected_convention
    assert result.raw_scores == {"p0": 1.0, "p1": 0.0}


@pytest.mark.parametrize(
    ("method_name", "expected_convention", "favored_passage"),
    [
        ("control_lexical_question", "question", "p0"),
        ("control_lexical_answer", "answer", "p1"),
        ("control_lexical_question_answer", "question_answer", "p0"),
        ("control_lexical", "question_answer", "p0"),
    ],
)
def test_lexical_variants_use_the_declared_query(
    method_name: str, expected_convention: str, favored_passage: str
) -> None:
    result = get_method(method_name).attribute(
        "Who built the bridge?",
        _passages(),
        "Nora Finch",
        None,
        {},
    )

    assert result.metadata["query_text_convention"] == expected_convention
    assert max(result.raw_scores, key=result.raw_scores.__getitem__) == favored_passage


def test_compatibility_aliases_match_question_answer_variants() -> None:
    args: tuple[Any, ...] = (
        "Who built the bridge?",
        _passages(),
        "Nora Finch",
        None,
        {},
    )
    alias = get_method("control_lexical").attribute(*args)
    explicit = get_method("control_lexical_question_answer").attribute(*args)

    assert alias == explicit
