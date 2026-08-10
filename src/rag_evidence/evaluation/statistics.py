"""Question-aligned paired uncertainty for benchmark comparisons."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from typing import Any, Literal

Direction = Literal["higher", "lower"]


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("quantile needs at least one value")
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])


def _derived_seed(global_seed: int, seed_parts: Sequence[str]) -> int:
    material = json.dumps([global_seed, *seed_parts], ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def paired_bootstrap_difference(
    candidate: Mapping[str, float | None],
    comparator: Mapping[str, float | None],
    *,
    global_seed: int,
    seed_parts: Sequence[str],
    resamples: int = 10_000,
    confidence: float = 0.95,
    favorable_direction: Direction,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """Bootstrap mean(candidate - comparator) over complete question-ID pairs."""
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    if favorable_direction not in ("higher", "lower"):
        raise ValueError("favorable_direction must be 'higher' or 'lower'")

    all_ids = sorted(set(candidate) | set(comparator))
    exclusions = {
        "candidate_missing": sum(qid not in candidate for qid in all_ids),
        "comparator_missing": sum(qid not in comparator for qid in all_ids),
        "candidate_null": sum(qid in candidate and candidate[qid] is None for qid in all_ids),
        "comparator_null": sum(qid in comparator and comparator[qid] is None for qid in all_ids),
    }
    paired_ids = [
        qid
        for qid in all_ids
        if qid in candidate
        and qid in comparator
        and candidate[qid] is not None
        and comparator[qid] is not None
    ]
    deltas: list[float] = []
    for qid in paired_ids:
        candidate_value = candidate[qid]
        comparator_value = comparator[qid]
        if candidate_value is None or comparator_value is None:  # narrowed by paired_ids
            raise AssertionError("complete pair unexpectedly contains a null value")
        deltas.append(float(candidate_value) - float(comparator_value))
    seed = _derived_seed(global_seed, seed_parts)
    result: dict[str, Any] = {
        "n_pairs": len(deltas),
        "n_candidate": sum(value is not None for value in candidate.values()),
        "n_comparator": sum(value is not None for value in comparator.values()),
        "exclusions": exclusions,
        "mean_delta": sum(deltas) / len(deltas) if deltas else None,
        "ci_low": None,
        "ci_high": None,
        "ci_excludes_zero": None,
        "favorable_direction": favorable_direction,
        "favorable": None,
        "confidence": confidence,
        "resamples": resamples,
        "seed_uint64": str(seed),
    }
    if not deltas:
        return result

    rng = random.Random(seed)
    n = len(deltas)
    bootstrap_means = [
        sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)
    ]
    bootstrap_means.sort()
    alpha = (1.0 - confidence) / 2.0
    ci_low = _quantile(bootstrap_means, alpha)
    ci_high = _quantile(bootstrap_means, 1.0 - alpha)
    result.update(
        {
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci_excludes_zero": bool(ci_low > tolerance or ci_high < -tolerance),
            "favorable": (
                ci_low > tolerance if favorable_direction == "higher" else ci_high < -tolerance
            ),
        }
    )
    return result
