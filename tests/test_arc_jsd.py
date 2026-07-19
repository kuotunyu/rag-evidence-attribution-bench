"""Native ARC-JSD (experimental): JSD math + ranking property with FakeLM."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rag_evidence.attribution.arc_jsd import ArcJsdAttribution, jsd_from_logprobs
from rag_evidence.attribution.base import ModelResources
from rag_evidence.data import ids
from rag_evidence.data.schema import Passage
from rag_evidence.generation.backends import FakeLM


def _logprobs(p: list[float]) -> np.ndarray:
    arr = np.asarray(p, dtype=np.float64)
    arr = arr / arr.sum()
    return np.log(arr)


def test_jsd_identical_is_zero() -> None:
    lp = _logprobs([0.5, 0.3, 0.2])
    assert jsd_from_logprobs(lp, lp) == pytest.approx(0.0, abs=1e-9)


def test_jsd_disjoint_is_ln2() -> None:
    lp_p = _logprobs([1.0, 1e-300])
    lp_q = _logprobs([1e-300, 1.0])
    assert jsd_from_logprobs(lp_p, lp_q) == pytest.approx(math.log(2), rel=1e-3)


def test_jsd_symmetric_and_nonnegative() -> None:
    lp_p = _logprobs([0.7, 0.2, 0.1])
    lp_q = _logprobs([0.1, 0.3, 0.6])
    a = jsd_from_logprobs(lp_p, lp_q)
    b = jsd_from_logprobs(lp_q, lp_p)
    assert a == pytest.approx(b)
    assert 0.0 < a < math.log(2)


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


def test_arc_jsd_ranks_important_passages_first() -> None:
    passages = _passages()
    target = "Nora Finch built the Alpha Bridge"  # contains both gold titles
    result = ArcJsdAttribution().attribute(
        "Who built the Alpha Bridge?",
        passages,
        target,
        ModelResources(generator=FakeLM()),
        {"alias_map": ids.make_alias_map([p.passage_id for p in passages])},
    )
    assert result.num_model_calls == 5
    assert all(v >= 0.0 for v in result.raw_scores.values())
    ranked = sorted(result.raw_scores, key=result.raw_scores.__getitem__, reverse=True)
    assert set(ranked[:2]) == {"q-p00", "q-p01"}
    assert result.metadata["experimental"] is True
