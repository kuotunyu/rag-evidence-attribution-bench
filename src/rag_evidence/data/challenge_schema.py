"""Strict records for deterministic answerability challenge transformations."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from rag_evidence.data import ids
from rag_evidence.data.schema import Example, Passage
from rag_evidence.errors import DataError

CHALLENGE_SCHEMA_VERSION = 1
TRANSFORMATIONS = (
    "missing_hop",
    "answer_bearing_distractor",
    "evidence_swap",
)
SOURCE_SPLITS = ("smoke", "dev", "eval")

Transformation = Literal["missing_hop", "answer_bearing_distractor", "evidence_swap"]
Answerability = Literal["answerable", "unanswerable"]
ReviewStatus = Literal["not_required", "pending_two_annotators"]

_HEX_64 = re.compile(r"[0-9a-f]{64}")
_CHALLENGE_ID = re.compile(r"ch-[0-9a-f]{24}")
_SENTENCE_ID = re.compile(r"^(?P<pid>.+-p(?P<pidx>\d{2,}))-s(?P<sidx>\d{2,})$")
_TOP_FIELDS = {
    "schema_version",
    "challenge_id",
    "parent_question_id",
    "source_split",
    "transformation",
    "transform_version",
    "seed",
    "expected_answerability",
    "review",
    "changed_fields",
    "provenance",
    "parent_fingerprint",
    "content_hash",
    "example",
}
_EXAMPLE_FIELDS = {
    "question_id",
    "question",
    "answer",
    "level",
    "qtype",
    "passages",
    "gold_passage_ids",
    "supporting_fact_sentence_ids",
    "dropped_supporting_facts",
}
_PASSAGE_FIELDS = {"passage_id", "index", "title", "sentences", "is_gold"}
_REVIEW_FIELDS = {"required", "status", "confirmatory_eligible"}


def _exact_fields(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(payload)
    if actual != expected:
        raise DataError(
            f"challenge {label} fields mismatch: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def record_content_hash(payload_without_content_hash: Mapping[str, Any]) -> str:
    material = json.dumps(
        payload_without_content_hash,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode()).hexdigest()


def make_challenge_id(
    parent_question_id: str,
    source_split: str,
    transformation: str,
    *,
    transform_version: str,
) -> str:
    material = json.dumps(
        [transform_version, source_split, parent_question_id, transformation],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "ch-" + hashlib.sha256(material.encode()).hexdigest()[:24]


@dataclass(frozen=True)
class ChallengeReview:
    required: bool
    status: ReviewStatus
    confirmatory_eligible: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "required": self.required,
            "status": self.status,
            "confirmatory_eligible": self.confirmatory_eligible,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> ChallengeReview:
        _exact_fields(payload, _REVIEW_FIELDS, "review")
        values = (
            payload["required"],
            payload["status"],
            payload["confirmatory_eligible"],
        )
        if values == (False, "not_required", True):
            return cls(False, "not_required", True)
        if values == (True, "pending_two_annotators", False):
            return cls(True, "pending_two_annotators", False)
        raise DataError(f"invalid challenge review gate: {values}")


PENDING_REVIEW = ChallengeReview(True, "pending_two_annotators", False)
REVIEW_NOT_REQUIRED = ChallengeReview(False, "not_required", True)


def remap_example(
    example: Example,
    challenge_id: str,
    *,
    passages: Sequence[Passage],
    gold_slots: set[int],
    supporting_slots: set[tuple[int, int]],
) -> Example:
    if _CHALLENGE_ID.fullmatch(challenge_id) is None:
        raise DataError(f"invalid challenge ID for remapping: {challenge_id}")
    valid_slots = set(range(len(passages)))
    if not gold_slots <= valid_slots:
        raise DataError("challenge gold passage slot is out of range")
    for passage_slot, sentence_slot in supporting_slots:
        if passage_slot not in gold_slots:
            raise DataError("challenge supporting sentence belongs to a non-gold passage")
        if not 0 <= sentence_slot < len(passages[passage_slot].sentences):
            raise DataError("challenge supporting sentence slot is out of range")
    remapped_passages = tuple(
        Passage(
            passage_id=ids.passage_id(challenge_id, index),
            index=index,
            title=passage.title,
            sentences=tuple(passage.sentences),
            is_gold=index in gold_slots,
        )
        for index, passage in enumerate(passages)
    )
    return Example(
        question_id=challenge_id,
        question=example.question,
        answer=example.answer,
        level=example.level,
        qtype=example.qtype,
        passages=remapped_passages,
        gold_passage_ids=tuple(ids.passage_id(challenge_id, index) for index in sorted(gold_slots)),
        supporting_fact_sentence_ids=tuple(
            ids.sentence_id(ids.passage_id(challenge_id, pidx), sidx)
            for pidx, sidx in sorted(supporting_slots)
        ),
        dropped_supporting_facts=(),
    )


def _validate_example(payload: Mapping[str, Any], challenge_id: str) -> Example:
    _exact_fields(payload, _EXAMPLE_FIELDS, "example")
    passages_payload = payload.get("passages")
    if not isinstance(passages_payload, list):
        raise DataError("challenge example passages must be a list")
    for passage_payload in passages_payload:
        if not isinstance(passage_payload, dict):
            raise DataError("challenge passage must be an object")
        _exact_fields(passage_payload, _PASSAGE_FIELDS, "passage")
    try:
        example = Example.from_json(dict(payload))
    except (KeyError, TypeError, ValueError) as exc:
        raise DataError(f"invalid challenge example: {exc}") from exc
    if example.question_id != challenge_id:
        raise DataError("challenge example question ID does not match challenge ID")
    expected_passage_ids = [
        ids.passage_id(challenge_id, index) for index in range(len(example.passages))
    ]
    if [passage.passage_id for passage in example.passages] != expected_passage_ids:
        raise DataError("challenge passage IDs are not stable slot IDs")
    if [passage.index for passage in example.passages] != list(range(len(example.passages))):
        raise DataError("challenge passage indices are not canonical")
    expected_gold = tuple(passage.passage_id for passage in example.passages if passage.is_gold)
    if example.gold_passage_ids != expected_gold:
        raise DataError("challenge gold passage labels do not match passage flags")
    for sentence_id in example.supporting_fact_sentence_ids:
        match = _SENTENCE_ID.fullmatch(sentence_id)
        if match is None or match.group("pid") not in expected_gold:
            raise DataError("challenge supporting sentence ID is invalid")
        passage_index = int(match.group("pidx"))
        sentence_index = int(match.group("sidx"))
        if passage_index >= len(example.passages) or sentence_index >= len(
            example.passages[passage_index].sentences
        ):
            raise DataError("challenge supporting sentence ID is out of range")
    if example.dropped_supporting_facts:
        raise DataError("challenge example cannot contain dropped supporting facts")
    return example


@dataclass(frozen=True)
class ChallengeRecord:
    challenge_id: str
    parent_question_id: str
    source_split: Literal["smoke", "dev", "eval"]
    transformation: Transformation
    transform_version: str
    seed: int
    expected_answerability: Answerability
    review: ChallengeReview
    changed_fields: tuple[str, ...]
    provenance: dict[str, Any]
    parent_fingerprint: str
    content_hash: str
    example: Example

    def _payload_without_hash(self) -> dict[str, Any]:
        return {
            "schema_version": CHALLENGE_SCHEMA_VERSION,
            "challenge_id": self.challenge_id,
            "parent_question_id": self.parent_question_id,
            "source_split": self.source_split,
            "transformation": self.transformation,
            "transform_version": self.transform_version,
            "seed": self.seed,
            "expected_answerability": self.expected_answerability,
            "review": self.review.to_json(),
            "changed_fields": list(self.changed_fields),
            "provenance": self.provenance,
            "parent_fingerprint": self.parent_fingerprint,
            "example": self.example.to_json(),
        }

    def to_json(self) -> dict[str, Any]:
        payload = self._payload_without_hash()
        return {**payload, "content_hash": self.content_hash}

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> ChallengeRecord:
        _exact_fields(payload, _TOP_FIELDS, "record")
        if payload.get("schema_version") != CHALLENGE_SCHEMA_VERSION:
            raise DataError("challenge record schema version must be 1")
        challenge_id = payload.get("challenge_id")
        if not isinstance(challenge_id, str) or _CHALLENGE_ID.fullmatch(challenge_id) is None:
            raise DataError("challenge ID must match ch-[0-9a-f]{24}")
        parent_question_id = payload.get("parent_question_id")
        source_split = payload.get("source_split")
        transformation = payload.get("transformation")
        transform_version = payload.get("transform_version")
        seed = payload.get("seed")
        answerability = payload.get("expected_answerability")
        if not isinstance(parent_question_id, str) or not parent_question_id:
            raise DataError("challenge parent question ID is invalid")
        if source_split not in SOURCE_SPLITS:
            raise DataError("challenge source split is invalid")
        if transformation not in TRANSFORMATIONS:
            raise DataError("challenge transformation is invalid")
        if transform_version != "challenge-v1":
            raise DataError("challenge transform version must be challenge-v1")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise DataError("challenge seed must be an integer")
        if answerability not in ("answerable", "unanswerable"):
            raise DataError("challenge expected answerability is invalid")
        expected_id = make_challenge_id(
            parent_question_id,
            source_split,
            transformation,
            transform_version=transform_version,
        )
        if challenge_id != expected_id:
            raise DataError("challenge ID does not match record identity")
        review_payload = payload.get("review")
        if not isinstance(review_payload, dict):
            raise DataError("challenge review must be an object")
        review = ChallengeReview.from_json(review_payload)
        expected_pair = (
            ("answerable", REVIEW_NOT_REQUIRED)
            if transformation == "answer_bearing_distractor"
            else ("unanswerable", PENDING_REVIEW)
        )
        if (answerability, review) != expected_pair:
            raise DataError("challenge answerability and review gate are inconsistent")
        changed_fields_payload = payload.get("changed_fields")
        if not isinstance(changed_fields_payload, list) or not all(
            isinstance(path, str) and path.startswith("/example/")
            for path in changed_fields_payload
        ):
            raise DataError("challenge changed_fields must be example JSON Pointer paths")
        changed_fields = tuple(changed_fields_payload)
        if not changed_fields or changed_fields != tuple(sorted(set(changed_fields))):
            raise DataError("challenge changed_fields must be nonempty, sorted, and unique")
        provenance = payload.get("provenance")
        if not isinstance(provenance, dict):
            raise DataError("challenge provenance must be an object")
        parent_fingerprint = payload.get("parent_fingerprint")
        if not isinstance(parent_fingerprint, str) or _HEX_64.fullmatch(parent_fingerprint) is None:
            raise DataError("challenge parent fingerprint must be 64 lowercase hex characters")
        example_payload = payload.get("example")
        if not isinstance(example_payload, dict):
            raise DataError("challenge example must be an object")
        example = _validate_example(example_payload, challenge_id)
        content_hash = payload.get("content_hash")
        if not isinstance(content_hash, str) or _HEX_64.fullmatch(content_hash) is None:
            raise DataError("challenge content hash must be 64 lowercase hex characters")
        unhashed = {key: value for key, value in payload.items() if key != "content_hash"}
        if record_content_hash(unhashed) != content_hash:
            raise DataError("challenge content hash mismatch")
        return cls(
            challenge_id=challenge_id,
            parent_question_id=parent_question_id,
            source_split=source_split,
            transformation=transformation,
            transform_version=transform_version,
            seed=seed,
            expected_answerability=answerability,
            review=review,
            changed_fields=changed_fields,
            provenance=dict(provenance),
            parent_fingerprint=parent_fingerprint,
            content_hash=content_hash,
            example=example,
        )
