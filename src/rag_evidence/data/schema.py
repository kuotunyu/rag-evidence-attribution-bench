"""Normalized dataset records (prepared JSONL rows)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Passage:
    passage_id: str
    index: int
    title: str
    sentences: tuple[str, ...]
    is_gold: bool

    @property
    def text(self) -> str:
        return " ".join(s.strip() for s in self.sentences).strip()

    def to_json(self) -> dict[str, Any]:
        return {
            "passage_id": self.passage_id,
            "index": self.index,
            "title": self.title,
            "sentences": list(self.sentences),
            "is_gold": self.is_gold,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Passage:
        return cls(
            passage_id=d["passage_id"],
            index=d["index"],
            title=d["title"],
            sentences=tuple(d["sentences"]),
            is_gold=d["is_gold"],
        )


@dataclass(frozen=True)
class Example:
    question_id: str
    question: str
    answer: str
    level: str
    qtype: str
    passages: tuple[Passage, ...]
    gold_passage_ids: tuple[str, ...]
    supporting_fact_sentence_ids: tuple[str, ...]
    dropped_supporting_facts: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def passage_by_id(self, pid: str) -> Passage:
        for p in self.passages:
            if p.passage_id == pid:
                return p
        raise KeyError(pid)

    def to_json(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "answer": self.answer,
            "level": self.level,
            "qtype": self.qtype,
            "passages": [p.to_json() for p in self.passages],
            "gold_passage_ids": list(self.gold_passage_ids),
            "supporting_fact_sentence_ids": list(self.supporting_fact_sentence_ids),
            "dropped_supporting_facts": list(self.dropped_supporting_facts),
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Example:
        return cls(
            question_id=d["question_id"],
            question=d["question"],
            answer=d["answer"],
            level=d["level"],
            qtype=d["qtype"],
            passages=tuple(Passage.from_json(p) for p in d["passages"]),
            gold_passage_ids=tuple(d["gold_passage_ids"]),
            supporting_fact_sentence_ids=tuple(d["supporting_fact_sentence_ids"]),
            dropped_supporting_facts=tuple(d["dropped_supporting_facts"]),
        )
