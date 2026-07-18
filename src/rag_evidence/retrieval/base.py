"""Retriever protocol. Candidate-set shaped on purpose: in the distractor setting each
question ranks its own 10 passages. A future corpus-level mode would add an upstream
candidate provider without touching this interface or any downstream schema."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from rag_evidence.data.schema import Example

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class RankedPassage:
    passage_id: str
    score: float
    rank: int  # 1-based

    def to_json(self) -> dict[str, float | int | str]:
        return {"passage_id": self.passage_id, "score": self.score, "rank": self.rank}


class Retriever(Protocol):
    name: str

    def rank(self, example: Example) -> list[RankedPassage]: ...


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer shared by BM25 and the lexical control."""
    return _TOKEN_RE.findall(text.lower())


def to_ranked(scored: list[tuple[str, float]]) -> list[RankedPassage]:
    """Sort by (-score, original position) — deterministic tie-break — and assign ranks."""
    ordered = sorted(enumerate(scored), key=lambda t: (-t[1][1], t[0]))
    return [
        RankedPassage(passage_id=pid, score=float(score), rank=i + 1)
        for i, (_orig, (pid, score)) in enumerate(ordered)
    ]
