"""HotpotQA official answer normalization, EM and F1 (hotpot_evaluate_v1.py conventions)."""

from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = _ARTICLES_RE.sub(" ", s)
    return " ".join(s.split())


def exact_match(prediction: str, gold: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> tuple[float, float, float]:
    """(f1, precision, recall) with the official yes/no/noanswer guard."""
    norm_pred = normalize_answer(prediction)
    norm_gold = normalize_answer(gold)
    special = {"yes", "no", "noanswer"}
    if (norm_pred in special or norm_gold in special) and norm_pred != norm_gold:
        return 0.0, 0.0, 0.0
    pred_tokens = norm_pred.split()
    gold_tokens = norm_gold.split()
    if not pred_tokens or not gold_tokens:
        same = float(pred_tokens == gold_tokens)
        return same, same, same
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall
