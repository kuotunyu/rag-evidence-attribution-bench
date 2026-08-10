"""Canonical identities and connected components for leakage-resistant splits."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rag_evidence.errors import DataError


def _canonical_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def normalize_title(title: str) -> str:
    """Normalize a title for cross-question identity comparisons."""
    return _canonical_text(title).casefold()


def paragraph_fingerprint(sentences: Sequence[str]) -> str:
    """Hash ordered paragraph sentences after Unicode/whitespace canonicalization."""
    payload = json.dumps(
        [_canonical_text(sentence) for sentence in sentences],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def supporting_fact_errors(raw: Mapping[str, Any]) -> tuple[str, ...]:
    """Return unique reasons that make supporting annotations unresolvable."""
    context = raw.get("context", [])
    titles = [str(row[0]) for row in context]
    duplicate_titles = {title for title, count in Counter(titles).items() if count > 1}
    if duplicate_titles:
        return ("duplicate_context_title",)

    title_to_sentences = {str(title): sentences for title, sentences in context}
    errors: list[str] = []
    for title, sent_id in raw.get("supporting_facts", []):
        sentences = title_to_sentences.get(str(title))
        if sentences is None:
            if "title_not_in_context" not in errors:
                errors.append("title_not_in_context")
            continue
        if (
            not isinstance(sent_id, int) or not 0 <= sent_id < len(sentences)
        ) and "sent_id_out_of_range" not in errors:
            errors.append("sent_id_out_of_range")
    return tuple(errors)


@dataclass(frozen=True)
class SplitGroup:
    """An indivisible connected component for split allocation."""

    group_id: str
    question_ids: tuple[str, ...]
    normalized_titles: tuple[str, ...]
    paragraph_hashes: tuple[str, ...]
    bridge_count: int
    comparison_count: int

    @property
    def size(self) -> int:
        return len(self.question_ids)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def build_split_groups(raw_examples: Sequence[dict[str, Any]]) -> tuple[SplitGroup, ...]:
    """Join questions connected by any normalized title or paragraph fingerprint."""
    raws = sorted(raw_examples, key=lambda raw: str(raw["question_id"]))
    qids = [str(raw["question_id"]) for raw in raws]
    if len(qids) != len(set(qids)):
        raise DataError("duplicate question_ids cannot be grouped")

    union_find = _UnionFind(len(raws))
    owner_by_identity: dict[tuple[str, str], int] = {}
    identities_by_index: list[tuple[set[str], set[str]]] = []
    for index, raw in enumerate(raws):
        titles: set[str] = set()
        paragraphs: set[str] = set()
        for title, sentences in raw["context"]:
            title_key = normalize_title(str(title))
            paragraph_key = paragraph_fingerprint([str(sentence) for sentence in sentences])
            titles.add(title_key)
            paragraphs.add(paragraph_key)
            for identity in (("title", title_key), ("paragraph", paragraph_key)):
                owner = owner_by_identity.setdefault(identity, index)
                union_find.union(index, owner)
        identities_by_index.append((titles, paragraphs))

    members_by_root: dict[int, list[int]] = {}
    for index in range(len(raws)):
        members_by_root.setdefault(union_find.find(index), []).append(index)

    groups: list[SplitGroup] = []
    for members in members_by_root.values():
        member_qids = tuple(sorted(qids[index] for index in members))
        group_id = hashlib.sha256("\n".join(member_qids).encode()).hexdigest()
        group_titles = tuple(
            sorted({title for index in members for title in identities_by_index[index][0]})
        )
        group_paragraphs = tuple(
            sorted({value for index in members for value in identities_by_index[index][1]})
        )
        groups.append(
            SplitGroup(
                group_id=group_id,
                question_ids=member_qids,
                normalized_titles=group_titles,
                paragraph_hashes=group_paragraphs,
                bridge_count=sum(raws[index]["type"] == "bridge" for index in members),
                comparison_count=sum(raws[index]["type"] == "comparison" for index in members),
            )
        )
    return tuple(sorted(groups, key=lambda group: group.question_ids))
