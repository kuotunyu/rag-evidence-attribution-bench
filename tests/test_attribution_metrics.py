"""Attribution metrics vs hand-computed values, cross-checked against sklearn."""

from __future__ import annotations

import pytest
from sklearn.metrics import average_precision_score

from rag_evidence.metrics.attribution import (
    average_precision,
    attribution_ndcg,
    prf_at_k,
    rank_passages,
)

ORDER = [f"p{i}" for i in range(5)]


def test_rank_passages_tiebreak_by_position() -> None:
    raw = {"p0": 0.5, "p1": 0.9, "p2": 0.5, "p3": 0.1, "p4": 0.9}
    assert rank_passages(raw, ORDER) == ["p1", "p4", "p0", "p2", "p3"]


def test_prf_at_k_hand_values() -> None:
    ranking = ["p1", "p4", "p0", "p2", "p3"]
    gold = {"p1", "p0"}
    assert prf_at_k(ranking, gold, 1) == {"p": 1.0, "r": 0.5, "f1": pytest.approx(2 / 3)}
    at2 = prf_at_k(ranking, gold, 2)
    assert at2["p"] == 0.5 and at2["r"] == 0.5 and at2["f1"] == 0.5
    at3 = prf_at_k(ranking, gold, 3)
    assert at3["p"] == pytest.approx(2 / 3) and at3["r"] == 1.0


def test_average_precision_hand_value_and_sklearn_crosscheck() -> None:
    ranking = ["p1", "p4", "p0", "p2", "p3"]
    gold = {"p1", "p0"}
    # hits at ranks 1 and 3: AP = (1/1 + 2/3) / 2
    assert average_precision(ranking, gold) == pytest.approx((1.0 + 2 / 3) / 2)

    # sklearn cross-check: scores consistent with the ranking, binary labels
    scores = {"p1": 5.0, "p4": 4.0, "p0": 3.0, "p2": 2.0, "p3": 1.0}
    y_true = [1 if pid in gold else 0 for pid in ORDER]
    y_score = [scores[pid] for pid in ORDER]
    assert average_precision(ranking, gold) == pytest.approx(
        float(average_precision_score(y_true, y_score))
    )


def test_ndcg_perfect_and_worst() -> None:
    gold = {"p0", "p1"}
    assert attribution_ndcg(["p0", "p1", "p2", "p3", "p4"], gold, 5) == pytest.approx(1.0)
    worst = attribution_ndcg(["p2", "p3", "p4", "p0", "p1"], gold, 5)
    assert 0.0 < worst < 1.0
