"""Shared teacher-forced logprob scoring with an on-disk cache.

The full-context logprob of a target is method-independent; LOO, faithfulness, and any
future causal method share it through this cache (results/raw/cache/logprobs/…,
gitignored). Keys are content-derived: (question_id, kept passage ids in order, target).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path

from rag_evidence.data.schema import Passage
from rag_evidence.generation.backends import GeneratorBackend, TargetScore
from rag_evidence.generation.prompts import build_messages
from rag_evidence.storage.artifacts import append_record, read_records

logger = logging.getLogger(__name__)


def _key(question_id: str, passage_ids: Sequence[str], target: str) -> str:
    material = question_id + "|" + ",".join(passage_ids) + "|" + target
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class LogprobScorer:
    """Scores target logprobs under passage subsets, caching by content key."""

    def __init__(
        self,
        backend: GeneratorBackend,
        cache_path: Path | None,
        *,
        prompt_version: str,
    ) -> None:
        self.backend = backend
        self.cache_path = cache_path
        self.prompt_version = prompt_version
        self.calls = 0  # model forward passes actually made (cache misses)
        self._cache: dict[str, TargetScore] = {}
        if cache_path is not None and cache_path.exists():
            for rec in read_records(cache_path):
                self._cache[rec["key"]] = TargetScore(
                    sum_logprob=rec["sum_logprob"],
                    num_target_tokens=rec["num_target_tokens"],
                )
            logger.info("logprob cache: %d entries loaded", len(self._cache))

    def score(
        self,
        question_id: str,
        question: str,
        passages: Sequence[Passage],
        alias_map: dict[str, str],
        target: str,
    ) -> TargetScore:
        key = _key(question_id, [p.passage_id for p in passages], target)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        messages = build_messages(question, passages, alias_map, version=self.prompt_version)
        score = self.backend.target_logprob(messages, target)
        self.calls += 1
        self._cache[key] = score
        if self.cache_path is not None:
            append_record(
                self.cache_path,
                {
                    "key": key,
                    "question_id": question_id,
                    "passage_ids": [p.passage_id for p in passages],
                    "target_sha8": hashlib.sha256(target.encode("utf-8")).hexdigest()[:8],
                    "sum_logprob": score.sum_logprob,
                    "num_target_tokens": score.num_target_tokens,
                },
            )
        return score
