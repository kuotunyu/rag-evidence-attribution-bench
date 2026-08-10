"""Pure deterministic transformations for answerability challenge records."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

from rag_evidence.data import ids
from rag_evidence.data.challenge_schema import (
    PENDING_REVIEW,
    REVIEW_NOT_REQUIRED,
    ChallengeRecord,
    Transformation,
    make_challenge_id,
    record_content_hash,
    remap_example,
)
from rag_evidence.data.grouping import normalize_title, paragraph_fingerprint
from rag_evidence.data.schema import Example, Passage
from rag_evidence.errors import DataError
from rag_evidence.metrics.text import normalize_answer


def _rank(seed: int, parent_id: str, transformation: str, candidate_id: str) -> str:
    material = f"{seed}:{parent_id}:{transformation}:{candidate_id}"
    return hashlib.sha256(material.encode()).hexdigest()


def _contains_normalized(text: str, target: str) -> bool:
    normalized_text = normalize_answer(text)
    normalized_target = normalize_answer(target)
    if not normalized_target:
        raise DataError("challenge parent answer is empty after normalization")
    return f" {normalized_target} " in f" {normalized_text} "


def _source_slots(example: Example) -> tuple[set[int], set[tuple[int, int]]]:
    gold_slots: set[int] = set()
    for passage_id in example.gold_passage_ids:
        question_id, passage_index = ids.parse_passage_id(passage_id)
        if question_id != example.question_id or passage_index >= len(example.passages):
            raise DataError("challenge parent gold passage ID is invalid")
        gold_slots.add(passage_index)
    supporting_slots: set[tuple[int, int]] = set()
    for sentence_id in example.supporting_fact_sentence_ids:
        try:
            passage_id, raw_sentence_index = sentence_id.rsplit("-s", 1)
            sentence_index = int(raw_sentence_index)
            question_id, passage_index = ids.parse_passage_id(passage_id)
        except (ValueError, TypeError) as exc:
            raise DataError("challenge parent supporting sentence ID is invalid") from exc
        if (
            question_id != example.question_id
            or passage_index not in gold_slots
            or sentence_index >= len(example.passages[passage_index].sentences)
        ):
            raise DataError("challenge parent supporting sentence ID is invalid")
        supporting_slots.add((passage_index, sentence_index))
    if not gold_slots or not supporting_slots:
        raise DataError("challenge parent requires gold passages and supporting sentences")
    return gold_slots, supporting_slots


def _record(
    *,
    parent: Example,
    source_split: str,
    transformation: Transformation,
    transform_version: str,
    seed: int,
    expected_answerability: str,
    review: dict[str, Any],
    changed_fields: list[str],
    provenance: dict[str, Any],
    parent_fingerprint: str,
    transformed: Example,
) -> ChallengeRecord:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "challenge_id": transformed.question_id,
        "parent_question_id": parent.question_id,
        "source_split": source_split,
        "transformation": transformation,
        "transform_version": transform_version,
        "seed": seed,
        "expected_answerability": expected_answerability,
        "review": review,
        "changed_fields": sorted(changed_fields),
        "provenance": provenance,
        "parent_fingerprint": parent_fingerprint,
        "example": transformed.to_json(),
    }
    return ChallengeRecord.from_json(
        {**payload, "content_hash": record_content_hash(payload)}
    )


def build_missing_hop(
    parent: Example,
    split_parents: Sequence[Example],
    *,
    source_split: str,
    seed: int,
    transform_version: str,
    parent_fingerprint: str,
) -> ChallengeRecord:
    """Replace one complete supporting passage with a safe same-split distractor."""
    gold_slots, supporting_slots = _source_slots(parent)
    parent_titles = {normalize_title(passage.title) for passage in parent.passages}
    parent_paragraphs = {
        paragraph_fingerprint(list(passage.sentences)) for passage in parent.passages
    }
    donor_candidates: list[tuple[Example, Passage]] = []
    for peer in split_parents:
        if peer.question_id == parent.question_id:
            continue
        for passage in peer.passages:
            if passage.is_gold:
                continue
            if normalize_title(passage.title) in parent_titles:
                continue
            if paragraph_fingerprint(list(passage.sentences)) in parent_paragraphs:
                continue
            if _contains_normalized(passage.text, parent.answer):
                continue
            donor_candidates.append((peer, passage))
    if not donor_candidates:
        raise DataError(f"no safe missing-hop donor for parent {parent.question_id}")
    donor_parent, donor = min(
        donor_candidates,
        key=lambda item: _rank(
            seed,
            parent.question_id,
            "missing_hop",
            f"{item[0].question_id}:{item[1].passage_id}",
        ),
    )
    replaced_slot = min(
        gold_slots,
        key=lambda slot: _rank(
            seed, parent.question_id, "missing_hop", parent.passages[slot].passage_id
        ),
    )
    removed_passage = parent.passages[replaced_slot]
    removed_sentence_ids = [
        ids.sentence_id(removed_passage.passage_id, sentence_slot)
        for passage_slot, sentence_slot in sorted(supporting_slots)
        if passage_slot == replaced_slot
    ]
    transformed_passages = list(parent.passages)
    transformed_passages[replaced_slot] = donor
    remaining_gold = gold_slots - {replaced_slot}
    remaining_support = {
        slot for slot in supporting_slots if slot[0] != replaced_slot
    }
    challenge_id = make_challenge_id(
        parent.question_id,
        source_split,
        "missing_hop",
        transform_version=transform_version,
    )
    transformed = remap_example(
        parent,
        challenge_id,
        passages=transformed_passages,
        gold_slots=remaining_gold,
        supporting_slots=remaining_support,
    )
    return _record(
        parent=parent,
        source_split=source_split,
        transformation="missing_hop",
        transform_version=transform_version,
        seed=seed,
        expected_answerability="unanswerable",
        review=PENDING_REVIEW.to_json(),
        changed_fields=[
            "/example/gold_passage_ids",
            f"/example/passages/{replaced_slot}",
            "/example/supporting_fact_sentence_ids",
        ],
        provenance={
            "replaced_slot": replaced_slot,
            "removed_source_passage_id": removed_passage.passage_id,
            "removed_source_sentence_ids": removed_sentence_ids,
            "donor_parent_question_id": donor_parent.question_id,
            "donor_source_passage_id": donor.passage_id,
            "donor_title": donor.title,
        },
        parent_fingerprint=parent_fingerprint,
        transformed=transformed,
    )


def build_answer_bearing_distractor(
    parent: Example,
    *,
    source_split: str,
    seed: int,
    transform_version: str,
    parent_fingerprint: str,
) -> ChallengeRecord:
    """Append a controlled answer mention to one existing non-gold passage."""
    gold_slots, supporting_slots = _source_slots(parent)
    candidates = [
        (index, passage)
        for index, passage in enumerate(parent.passages)
        if not passage.is_gold and not _contains_normalized(passage.text, parent.answer)
    ]
    if not candidates:
        raise DataError(
            f"no answer-free non-gold passage for parent {parent.question_id}"
        )
    source_slot, source_passage = min(
        candidates,
        key=lambda item: _rank(
            seed,
            parent.question_id,
            "answer_bearing_distractor",
            item[1].passage_id,
        ),
    )
    inserted_sentence = (
        f"{source_passage.title} has also been associated with {parent.answer}."
    )
    changed_passage = Passage(
        passage_id=source_passage.passage_id,
        index=source_passage.index,
        title=source_passage.title,
        sentences=(*source_passage.sentences, inserted_sentence),
        is_gold=False,
    )
    transformed_passages = list(parent.passages)
    transformed_passages[source_slot] = changed_passage
    challenge_id = make_challenge_id(
        parent.question_id,
        source_split,
        "answer_bearing_distractor",
        transform_version=transform_version,
    )
    transformed = remap_example(
        parent,
        challenge_id,
        passages=transformed_passages,
        gold_slots=gold_slots,
        supporting_slots=supporting_slots,
    )
    return _record(
        parent=parent,
        source_split=source_split,
        transformation="answer_bearing_distractor",
        transform_version=transform_version,
        seed=seed,
        expected_answerability="answerable",
        review=REVIEW_NOT_REQUIRED.to_json(),
        changed_fields=[f"/example/passages/{source_slot}/sentences"],
        provenance={
            "source_slot": source_slot,
            "source_passage_id": source_passage.passage_id,
            "inserted_sentence": inserted_sentence,
        },
        parent_fingerprint=parent_fingerprint,
        transformed=transformed,
    )


_AUXILIARY_RE = re.compile(
    r"\b(is|are|was|were|has|have|had|can|could|will|would|do|does|did)\b"
    r"(?P<negation>\s+not\b)?",
    re.IGNORECASE,
)


def _edit_evidence_sentence(
    sentence: str, *, answer: str, replacement_entity: str
) -> tuple[str, str]:
    raw_answer = answer.strip()
    if not normalize_answer(raw_answer):
        raise DataError("challenge parent answer is empty after normalization")
    answer_match = re.search(re.escape(raw_answer), sentence, flags=re.IGNORECASE)
    if answer_match is not None:
        edited = (
            sentence[: answer_match.start()]
            + replacement_entity
            + sentence[answer_match.end() :]
        )
        return edited, "answer_substitution"

    auxiliary_match = _AUXILIARY_RE.search(sentence)
    if auxiliary_match is not None:
        auxiliary = auxiliary_match.group(1)
        replacement = auxiliary if auxiliary_match.group("negation") else auxiliary + " not"
        edited = (
            sentence[: auxiliary_match.start()]
            + replacement
            + sentence[auxiliary_match.end() :]
        )
        return edited, "negation_toggle"
    return "It is not true that " + sentence, "negation_prefix"


def build_evidence_swap(
    parent: Example,
    *,
    source_split: str,
    seed: int,
    transform_version: str,
    parent_fingerprint: str,
) -> ChallengeRecord:
    """Edit one supporting sentence and remove its now-invalid evidence label."""
    gold_slots, supporting_slots = _source_slots(parent)
    replacement_candidates = [
        passage
        for passage in parent.passages
        if not passage.is_gold
        and normalize_answer(passage.title)
        and normalize_answer(passage.title) != normalize_answer(parent.answer)
    ]
    if not replacement_candidates:
        raise DataError(
            f"no non-gold replacement entity for parent {parent.question_id}"
        )
    replacement_source = min(
        replacement_candidates,
        key=lambda passage: _rank(
            seed,
            parent.question_id,
            "evidence_swap",
            passage.passage_id,
        ),
    )
    passage_slot, sentence_slot = min(
        supporting_slots,
        key=lambda slot: _rank(
            seed,
            parent.question_id,
            "evidence_swap",
            ids.sentence_id(parent.passages[slot[0]].passage_id, slot[1]),
        ),
    )
    source_passage = parent.passages[passage_slot]
    original_sentence = source_passage.sentences[sentence_slot]
    edited_sentence, operator = _edit_evidence_sentence(
        original_sentence,
        answer=parent.answer,
        replacement_entity=replacement_source.title,
    )
    changed_sentences = list(source_passage.sentences)
    changed_sentences[sentence_slot] = edited_sentence
    changed_passage = Passage(
        passage_id=source_passage.passage_id,
        index=source_passage.index,
        title=source_passage.title,
        sentences=tuple(changed_sentences),
        is_gold=source_passage.is_gold,
    )
    transformed_passages = list(parent.passages)
    transformed_passages[passage_slot] = changed_passage
    remaining_support = supporting_slots - {(passage_slot, sentence_slot)}
    remaining_gold = set(gold_slots)
    removed_gold_passage = not any(
        candidate_passage == passage_slot for candidate_passage, _ in remaining_support
    )
    if removed_gold_passage:
        remaining_gold.remove(passage_slot)
    challenge_id = make_challenge_id(
        parent.question_id,
        source_split,
        "evidence_swap",
        transform_version=transform_version,
    )
    transformed = remap_example(
        parent,
        challenge_id,
        passages=transformed_passages,
        gold_slots=remaining_gold,
        supporting_slots=remaining_support,
    )
    changed_fields = [
        f"/example/passages/{passage_slot}/sentences/{sentence_slot}",
        "/example/supporting_fact_sentence_ids",
    ]
    if removed_gold_passage:
        changed_fields.append("/example/gold_passage_ids")
    source_sentence_id = ids.sentence_id(source_passage.passage_id, sentence_slot)
    return _record(
        parent=parent,
        source_split=source_split,
        transformation="evidence_swap",
        transform_version=transform_version,
        seed=seed,
        expected_answerability="unanswerable",
        review=PENDING_REVIEW.to_json(),
        changed_fields=changed_fields,
        provenance={
            "source_passage_slot": passage_slot,
            "source_sentence_slot": sentence_slot,
            "source_sentence_id": source_sentence_id,
            "replacement_source_passage_id": replacement_source.passage_id,
            "replacement_entity": replacement_source.title,
            "edit_operator": operator,
            "original_sentence": original_sentence,
            "edited_sentence": edited_sentence,
        },
        parent_fingerprint=parent_fingerprint,
        transformed=transformed,
    )


def build_challenge_records(
    parents_by_split: Mapping[str, Sequence[Example]],
    parent_fingerprints: Mapping[str, str],
    *,
    seed: int,
    transform_version: str,
) -> dict[str, tuple[ChallengeRecord, ...]]:
    """Build three deterministic challenge records for every supplied parent."""
    output: dict[str, tuple[ChallengeRecord, ...]] = {}
    for source_split in sorted(parents_by_split):
        parents = tuple(
            sorted(parents_by_split[source_split], key=lambda row: row.question_id)
        )
        if len({parent.question_id for parent in parents}) != len(parents):
            raise DataError(f"duplicate challenge parent in split {source_split}")
        records: list[ChallengeRecord] = []
        for parent in parents:
            try:
                parent_fingerprint = parent_fingerprints[parent.question_id]
            except KeyError as exc:
                raise DataError(
                    f"missing fingerprint for challenge parent {parent.question_id}"
                ) from exc
            records.extend(
                (
                    build_missing_hop(
                        parent,
                        parents,
                        source_split=source_split,
                        seed=seed,
                        transform_version=transform_version,
                        parent_fingerprint=parent_fingerprint,
                    ),
                    build_answer_bearing_distractor(
                        parent,
                        source_split=source_split,
                        seed=seed,
                        transform_version=transform_version,
                        parent_fingerprint=parent_fingerprint,
                    ),
                    build_evidence_swap(
                        parent,
                        source_split=source_split,
                        seed=seed,
                        transform_version=transform_version,
                        parent_fingerprint=parent_fingerprint,
                    ),
                )
            )
        output[source_split] = tuple(
            sorted(records, key=lambda record: record.challenge_id)
        )
    return output
