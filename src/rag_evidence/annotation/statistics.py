"""Wilson, Holm, and leakage-group bootstrap statistics for annotation analysis."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics as stdlib_statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class NestedObservation:
    cluster_id: str
    parent_id: str
    variant: str
    candidate: float | None
    comparator: float | None


@dataclass(frozen=True)
class ClusterBootstrapResult:
    mean_delta: float
    ci_low: float
    ci_high: float
    n_clusters: int
    n_parents: int
    n_variants: int
    exclusions: dict[str, int]
    resamples: int
    confidence: float
    seed_uint64: int


def wilson_interval(
    successes: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> tuple[float | None, float | None]:
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("Wilson counts require 0 <= successes <= total")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if total == 0:
        return None, None
    z = stdlib_statistics.NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z2 / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    for name, value in p_values.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"p-value for {name} must be between zero and one")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[name] = running
    return {name: adjusted[name] for name in p_values}


def _derived_seed(seed: int, parts: Sequence[str]) -> int:
    material = json.dumps([seed, *parts], ensure_ascii=False, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")


def _quantile(values: Sequence[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + fraction * (values[upper] - values[lower])


def cluster_bootstrap_difference(
    observations: Sequence[NestedObservation],
    *,
    seed: int,
    seed_parts: Sequence[str],
    resamples: int = 10_000,
    confidence: float = 0.95,
) -> ClusterBootstrapResult:
    """Bootstrap leakage groups while retaining parents and variants nested inside."""
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    exclusions = {
        "candidate_missing": sum(item.candidate is None for item in observations),
        "comparator_missing": sum(item.comparator is None for item in observations),
    }
    complete = sorted(
        (
            item
            for item in observations
            if item.candidate is not None and item.comparator is not None
        ),
        key=lambda item: (item.cluster_id, item.parent_id, item.variant),
    )
    identities = [(item.cluster_id, item.parent_id, item.variant) for item in complete]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate cluster/parent/variant observation")
    if not complete:
        raise ValueError("cluster bootstrap requires at least one complete method pair")

    parent_clusters: dict[str, str] = {}
    deltas_by_parent: dict[str, list[float]] = defaultdict(list)
    for item in complete:
        prior_cluster = parent_clusters.setdefault(item.parent_id, item.cluster_id)
        if prior_cluster != item.cluster_id:
            raise ValueError("one parent cannot belong to multiple leakage groups")
        assert item.candidate is not None and item.comparator is not None
        deltas_by_parent[item.parent_id].append(item.candidate - item.comparator)
    parent_means = {
        parent: sum(values) / len(values) for parent, values in deltas_by_parent.items()
    }
    parents_by_cluster: dict[str, list[float]] = defaultdict(list)
    for parent, mean in parent_means.items():
        parents_by_cluster[parent_clusters[parent]].append(mean)
    cluster_ids = sorted(parents_by_cluster)
    point = sum(parent_means.values()) / len(parent_means)

    derived_seed = _derived_seed(seed, seed_parts)
    rng = random.Random(derived_seed)
    bootstrap_means: list[float] = []
    for _ in range(resamples):
        sampled_values: list[float] = []
        for _cluster_draw in cluster_ids:
            sampled_cluster = cluster_ids[rng.randrange(len(cluster_ids))]
            sampled_values.extend(parents_by_cluster[sampled_cluster])
        bootstrap_means.append(sum(sampled_values) / len(sampled_values))
    bootstrap_means.sort()
    alpha = (1.0 - confidence) / 2.0
    return ClusterBootstrapResult(
        mean_delta=point,
        ci_low=_quantile(bootstrap_means, alpha),
        ci_high=_quantile(bootstrap_means, 1.0 - alpha),
        n_clusters=len(cluster_ids),
        n_parents=len(parent_means),
        n_variants=len(complete),
        exclusions=exclusions,
        resamples=resamples,
        confidence=confidence,
        seed_uint64=derived_seed,
    )
