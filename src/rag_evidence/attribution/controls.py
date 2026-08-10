"""Controls: null / trivial baselines that flow through the identical pipeline.

Every control is an AttributionMethod (is_control=True), so faithfulness passes and all
metrics apply to controls exactly as to real methods — that is the point: they calibrate
what "no attribution signal" and "trivial signal" look like in every table.

Seeding: sha256-derived per (global seed, method, question_id) — deterministic across
machines, independent per sample and per method.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from rag_evidence.attribution.base import AttributionMethod, AttributionResult, ModelResources
from rag_evidence.attribution.query import QueryConvention, compose_query
from rag_evidence.attribution.registry import register
from rag_evidence.data.schema import Passage
from rag_evidence.errors import UpstreamMissingError
from rag_evidence.metrics.text import normalize_answer
from rag_evidence.retrieval.base import tokenize
from rag_evidence.telemetry import derive_seed

# small built-in stopword list — no NLTK download dependency
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "of",
        "in",
        "on",
        "at",
        "to",
        "from",
        "by",
        "with",
        "for",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "he",
        "she",
        "they",
        "them",
        "his",
        "her",
        "their",
        "we",
        "you",
        "your",
        "i",
        "my",
        "me",
        "our",
        "us",
        "who",
        "whom",
        "which",
        "what",
        "where",
        "when",
        "why",
        "how",
        "not",
        "no",
        "nor",
        "do",
        "does",
        "did",
        "done",
        "have",
        "has",
        "had",
        "having",
        "will",
        "would",
        "can",
        "could",
        "should",
        "may",
        "might",
        "must",
    ]
)


def _rng(method_config: Mapping[str, Any], method_name: str) -> random.Random:
    return random.Random(
        derive_seed(int(method_config["seed"]), method_name, str(method_config["question_id"]))
    )


class _Control(AttributionMethod):
    is_control: ClassVar[bool] = True


@register
class OracleGoldAttribution(_Control):
    """Positive control: rank annotated supporting passages first."""

    name = "oracle_gold"

    def attribute(
        self,
        question: str,
        passages: Sequence[Passage],
        target_answer: str,
        model: ModelResources | None,
        method_config: Mapping[str, Any],
    ) -> AttributionResult:
        return AttributionResult(
            raw_scores={p.passage_id: float(p.is_gold) for p in passages},
            metadata={"control_role": "positive_control"},
        )


def _contains_contiguous_tokens(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    width = len(needle)
    return any(
        list(haystack[start : start + width]) == list(needle)
        for start in range(len(haystack) - width + 1)
    )


@register
class AnswerStringControl(_Control):
    """Lexical negative control: rank passages containing the normalized answer."""

    name = "control_answer_string"

    def attribute(
        self,
        question: str,
        passages: Sequence[Passage],
        target_answer: str,
        model: ModelResources | None,
        method_config: Mapping[str, Any],
    ) -> AttributionResult:
        answer_tokens = normalize_answer(target_answer).split()
        raw = {
            p.passage_id: float(
                _contains_contiguous_tokens(
                    normalize_answer(f"{p.title} {p.text}").split(), answer_tokens
                )
            )
            for p in passages
        }
        return AttributionResult(
            raw_scores=raw,
            metadata={"control_role": "lexical_negative_control"},
        )


@register
class RandomControl(_Control):
    name = "control_random"

    def attribute(
        self,
        question: str,
        passages: Sequence[Passage],
        target_answer: str,
        model: ModelResources | None,
        method_config: Mapping[str, Any],
    ) -> AttributionResult:
        rng = _rng(method_config, self.name)
        return AttributionResult(raw_scores={p.passage_id: rng.random() for p in passages})


@register
class RetrievalRankControl(_Control):
    """Question-only retrieval ranking as attribution: raw = -rank (rank-based, so BM25
    and dense sources are interchangeable and scale-free)."""

    name = "control_retrieval"

    def attribute(
        self,
        question: str,
        passages: Sequence[Passage],
        target_answer: str,
        model: ModelResources | None,
        method_config: Mapping[str, Any],
    ) -> AttributionResult:
        ranks: dict[str, int] | None = method_config.get("retrieval_ranks")
        if ranks is None:
            raise UpstreamMissingError("control_retrieval needs retrieval_ranks in method_config")
        missing = [p.passage_id for p in passages if p.passage_id not in ranks]
        if missing:
            raise UpstreamMissingError(f"retrieval run lacks ranks for {missing[:3]}")
        return AttributionResult(
            raw_scores={p.passage_id: -float(ranks[p.passage_id]) for p in passages},
            metadata={"source_run": method_config.get("retrieval_run_name")},
        )


class _LexicalOverlapByQuery(_Control):
    """Token overlap between (question + answer) and the passage — same query-text
    convention as the embedding method, making the pair a lexical-vs-dense ablation."""

    convention: ClassVar[QueryConvention]

    def attribute(
        self,
        question: str,
        passages: Sequence[Passage],
        target_answer: str,
        model: ModelResources | None,
        method_config: Mapping[str, Any],
    ) -> AttributionResult:
        query = compose_query(question, target_answer, self.convention)
        query_tokens = {t for t in tokenize(query) if t not in _STOPWORDS}
        raw: dict[str, float] = {}
        for p in passages:
            p_tokens = {t for t in tokenize(f"{p.title} {p.text}") if t not in _STOPWORDS}
            raw[p.passage_id] = len(query_tokens & p_tokens) / max(1, len(query_tokens))
        return AttributionResult(
            raw_scores=raw,
            metadata={"query_text_convention": self.convention},
        )


@register
class LexicalQuestionControl(_LexicalOverlapByQuery):
    name = "control_lexical_question"
    convention = "question"


@register
class LexicalAnswerControl(_LexicalOverlapByQuery):
    name = "control_lexical_answer"
    convention = "answer"


@register
class LexicalQuestionAnswerControl(_LexicalOverlapByQuery):
    name = "control_lexical_question_answer"
    convention = "question_answer"


@register
class LexicalOverlapControl(_LexicalOverlapByQuery):
    """Schema-v1 compatibility alias for `control_lexical_question_answer`."""

    name = "control_lexical"
    convention = "question_answer"


@register
class PassageLengthControl(_Control):
    name = "control_length"

    def attribute(
        self,
        question: str,
        passages: Sequence[Passage],
        target_answer: str,
        model: ModelResources | None,
        method_config: Mapping[str, Any],
    ) -> AttributionResult:
        return AttributionResult(
            raw_scores={p.passage_id: float(len(p.text.split())) for p in passages}
        )


@register
class ShuffledScoresControl(_Control):
    """Permutes a REAL method's per-sample score vector across that sample's passages:
    preserves the score distribution, destroys the alignment — the sharpest null for
    AUPRC/nDCG."""

    name = "control_shuffled"

    def attribute(
        self,
        question: str,
        passages: Sequence[Passage],
        target_answer: str,
        model: ModelResources | None,
        method_config: Mapping[str, Any],
    ) -> AttributionResult:
        source: dict[str, float] | None = method_config.get("source_scores")
        if source is None:
            raise UpstreamMissingError(
                "control_shuffled needs source_scores (a completed attribution run of "
                "attribution.controls.shuffled_source) for this sample"
            )
        pids = [p.passage_id for p in passages]
        if set(source) != set(pids):
            raise UpstreamMissingError("source_scores passage set mismatch")
        values = [source[pid] for pid in pids]
        _rng(method_config, self.name).shuffle(values)
        return AttributionResult(
            raw_scores=dict(zip(pids, values, strict=True)),
            metadata={"source_method": method_config.get("shuffled_source_name")},
        )
