"""Faithfulness (sufficiency/comprehensiveness) and LOO with FakeLM logprobs.

FakeLM's contract: logprob = base(target) + Σ passage_weight(title, text, target),
weight ≈ 2.0 for passages whose title appears in the target (or target in text), tiny
otherwise. So LOO must rank 'important' passages first, comprehensiveness must be
large positive when they are removed, and sufficiency small when they are kept.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_evidence.attribution.faithfulness import compute_faithfulness, top_k_ids
from rag_evidence.attribution.leave_one_out import LeaveOneOutAttribution
from rag_evidence.attribution.scoring import LogprobScorer
from rag_evidence.data import ids
from rag_evidence.data.schema import Passage
from rag_evidence.generation.backends import FakeLM


def _passages() -> list[Passage]:
    mk = lambda i, title, text, gold: Passage(  # noqa: E731
        passage_id=f"q-p{i:02d}", index=i, title=title, sentences=(text,), is_gold=gold
    )
    return [
        mk(0, "Alpha Bridge", "The Alpha Bridge was built by Nora Finch.", True),
        mk(1, "Nora Finch", "Nora Finch is an engineer from Dunmore.", True),
        mk(2, "Cloud Farm", "Cloud Farm grows barley.", False),
        mk(3, "Green Lane", "Green Lane is a cycling path.", False),
    ]


TARGET = "Nora Finch built the Alpha Bridge"  # contains both gold titles


@pytest.fixture()
def scorer(tmp_path: Path) -> LogprobScorer:
    return LogprobScorer(FakeLM(), tmp_path / "cache.jsonl", prompt_version="v1")


def test_loo_ranks_important_passages_first(scorer: LogprobScorer) -> None:
    passages = _passages()
    alias_map = ids.make_alias_map([p.passage_id for p in passages])
    result = LeaveOneOutAttribution().attribute(
        "Who built the Alpha Bridge?",
        passages,
        TARGET,
        None,
        {"scorer": scorer, "question_id": "q", "alias_map": alias_map},
    )
    ranked = sorted(result.raw_scores, key=result.raw_scores.__getitem__, reverse=True)
    assert set(ranked[:2]) == {"q-p00", "q-p01"}
    # expected raw score = FakeLM weight of the removed passage
    for p in passages:
        expected = FakeLM.passage_weight(p.title, p.text, TARGET)
        assert result.raw_scores[p.passage_id] == pytest.approx(expected, abs=1e-9)
    assert result.num_model_calls == 5  # 1 full + 4 ablations, no cache hits yet


def test_faithfulness_signs_and_cache_reuse(scorer: LogprobScorer) -> None:
    passages = _passages()
    alias_map = ids.make_alias_map([p.passage_id for p in passages])
    raw_scores = {"q-p00": 2.0, "q-p01": 1.9, "q-p02": 0.01, "q-p03": 0.02}
    calls_before = scorer.calls
    faith = compute_faithfulness(
        question_id="q",
        question="Who built the Alpha Bridge?",
        passages=passages,
        alias_map=alias_map,
        target_answer=TARGET,
        raw_scores=raw_scores,
        scorer=scorer,
        k=2,
    )
    assert faith["top_k_passage_ids"] == ["q-p00", "q-p01"]
    # keeping only the two important passages: sufficiency ≈ 0 (tiny weights lost)
    assert abs(faith["sufficiency"]) < 0.1
    # removing them: comprehensiveness strongly positive (≈ 4.0 / n_target_tokens)
    assert faith["comprehensiveness"] > 0.5
    assert scorer.calls == calls_before + 3  # full, top-k-only, without-top-k

    # identical inputs → pure cache hits
    calls_mid = scorer.calls
    compute_faithfulness(
        question_id="q",
        question="Who built the Alpha Bridge?",
        passages=passages,
        alias_map=alias_map,
        target_answer=TARGET,
        raw_scores=raw_scores,
        scorer=scorer,
        k=2,
    )
    assert scorer.calls == calls_mid


def test_top_k_tiebreak() -> None:
    order = ["a", "b", "c"]
    assert top_k_ids({"a": 1.0, "b": 1.0, "c": 0.0}, order, 1) == ["a"]
    assert top_k_ids({"a": 0.0, "b": 1.0, "c": 1.0}, order, 1) == ["b"]


def test_scorer_cache_persists_across_instances(tmp_path: Path) -> None:
    passages = _passages()
    alias_map = ids.make_alias_map([p.passage_id for p in passages])
    path = tmp_path / "cache.jsonl"
    s1 = LogprobScorer(FakeLM(), path, prompt_version="v1")
    first = s1.score("q", "Q?", passages, alias_map, TARGET)
    assert s1.calls == 1
    s2 = LogprobScorer(FakeLM(), path, prompt_version="v1")
    second = s2.score("q", "Q?", passages, alias_map, TARGET)
    assert s2.calls == 0  # served from disk cache
    assert second.sum_logprob == pytest.approx(first.sum_logprob)
