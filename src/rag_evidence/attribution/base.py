"""Attribution protocol. The required interface is honored verbatim:

    attribute(question, passages, target_answer, model, method_config)
        -> passage scores + metadata

Controls implement the same protocol (is_control=True) so they flow through the
identical runner, artifact schema, and metric pipeline as real methods.
"""

from __future__ import annotations

import abc
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from rag_evidence.data.schema import Passage
from rag_evidence.errors import InvalidOutputError
from rag_evidence.generation.backends import GeneratorBackend


@dataclass(frozen=True)
class AttributionResult:
    raw_scores: Mapping[str, float]  # passage_id -> method-native score (higher = more important)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    num_model_calls: int = 0


@dataclass
class ModelResources:
    """What the `model` argument carries; runner populates only what the method declares."""

    generator: GeneratorBackend | None = None
    embedder: Any | None = None  # rag_evidence.embeddings.Embedder (kept Any: torch-free import)
    embedding_cache: Any | None = None


class AttributionMethod(abc.ABC):
    name: ClassVar[str]
    requires_generator: ClassVar[bool] = False
    requires_embedder: ClassVar[bool] = False
    supports_teacher_forced: ClassVar[bool] = True  # mode A (gold answer)
    supports_generated: ClassVar[bool] = True  # mode B (generated answer)
    is_control: ClassVar[bool] = False

    @abc.abstractmethod
    def attribute(
        self,
        question: str,
        passages: Sequence[Passage],
        target_answer: str,
        model: ModelResources | None,
        method_config: Mapping[str, Any],
    ) -> AttributionResult: ...


def normalize_scores(raw: Mapping[str, float]) -> tuple[dict[str, float], bool]:
    """Per-sample min-max to [0,1]; all-equal scores map to 0.5 with degenerate=True.

    Ranking metrics use raw_scores (min-max is monotone, so results are identical);
    normalized scores exist for cross-method displays and the shuffled control.
    """
    values = list(raw.values())
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return dict.fromkeys(raw, 0.5), True
    span = hi - lo
    return {pid: (v - lo) / span for pid, v in raw.items()}, False


def validate_scores(raw: Mapping[str, float], expected_ids: Sequence[str]) -> None:
    """Central output validation — no method can silently emit garbage."""
    if set(raw) != set(expected_ids):
        raise InvalidOutputError(
            f"score keys != context passage set (got {len(raw)}, expected {len(expected_ids)})"
        )
    bad = [pid for pid, v in raw.items() if not math.isfinite(v)]
    if bad:
        raise InvalidOutputError(f"non-finite scores for {bad[:3]}")
