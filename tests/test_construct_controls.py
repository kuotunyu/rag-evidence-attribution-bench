"""Positive and lexical-negative controls for causal construct checks."""

from __future__ import annotations

from rag_evidence.attribution.controls import AnswerStringControl, OracleGoldAttribution
from rag_evidence.attribution.registry import get_method
from rag_evidence.data.schema import Passage


def _passages() -> list[Passage]:
    return [
        Passage(
            passage_id="p0",
            index=0,
            title="Bridge",
            sentences=("The bridge was built by its architect.",),
            is_gold=True,
        ),
        Passage(
            passage_id="p1",
            index=1,
            title="Architect",
            sentences=("The architect was born in Dunmore.",),
            is_gold=True,
        ),
        Passage(
            passage_id="p2",
            index=2,
            title="Awards",
            sentences=("Nora Finch won an unrelated award.",),
            is_gold=False,
        ),
    ]


def test_oracle_gold_scores_only_annotated_support() -> None:
    method = OracleGoldAttribution()
    result = method.attribute("q", _passages(), "Nora Finch", None, {})

    assert method.is_control is True
    assert result.raw_scores == {"p0": 1.0, "p1": 1.0, "p2": 0.0}
    assert result.metadata["control_role"] == "positive_control"


def test_answer_string_control_scores_only_contiguous_normalized_answer() -> None:
    method = AnswerStringControl()
    result = method.attribute("q", _passages(), "The Nora Finch!", None, {})

    assert method.is_control is True
    assert result.raw_scores == {"p0": 0.0, "p1": 0.0, "p2": 1.0}
    assert result.metadata["control_role"] == "lexical_negative_control"


def test_answer_string_control_does_not_match_discontiguous_tokens() -> None:
    passages = [
        Passage(
            passage_id="p0",
            index=0,
            title="Nora",
            sentences=("Many unrelated words separate the name Finch.",),
            is_gold=False,
        )
    ]

    result = AnswerStringControl().attribute("q", passages, "Nora Finch", None, {})

    assert result.raw_scores == {"p0": 0.0}


def test_answer_string_control_empty_normalized_answer_scores_zero() -> None:
    result = AnswerStringControl().attribute("q", _passages(), "the", None, {})

    assert set(result.raw_scores.values()) == {0.0}


def test_construct_controls_are_registered_under_protocol_names() -> None:
    assert isinstance(get_method("oracle_gold"), OracleGoldAttribution)
    assert isinstance(get_method("control_answer_string"), AnswerStringControl)
