"""Strict, hash-bound annotation artifacts reject ambiguous or leaky records."""

from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from rag_evidence.annotation.models import (
    AdjudicationRecord,
    AnnotationAmendment,
    AnswerabilityAnnotation,
    BlindTask,
    CitationAnnotation,
    EligibilityArtifact,
    EligibilityRecord,
    artifact_hash,
)


def _sha(payload: object) -> str:
    material = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(material.encode()).hexdigest()


def _task_payload() -> dict[str, object]:
    content = {
        "challenge_id": "ch-0123456789abcdef01234567",
        "blinded_parent_group": "bg-0123456789abcdef01234567",
        "instruction_version": "pilot-v0.2-draft",
        "instruction_hash": "1" * 64,
        "question": "Which city is the architect from?",
        "passages": [
            {
                "alias": "P1",
                "title": "Bridge",
                "sentences": [
                    {"alias": "P1.S1", "text": "The bridge was designed by Noor."},
                    {"alias": "P1.S2", "text": "It opened in 1999."},
                ],
            },
            {
                "alias": "P2",
                "title": "Noor",
                "sentences": [{"alias": "P2.S1", "text": "Noor is from Riverton."}],
            },
        ],
    }
    return {
        "schema_version": "blind-task-v1",
        "annotation_task_id": "task-0123456789abcdef01234567",
        **content,
        "task_content_hash": _sha(content),
        "assignment_batch": "pilot-batch-01",
    }


def _answer_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "answerability-annotation-v1",
        "annotation_task_id": "task-0123456789abcdef01234567",
        "challenge_id": "ch-0123456789abcdef01234567",
        "blinded_parent_group": "bg-0123456789abcdef01234567",
        "annotator_pseudonym": "ann-r7",
        "instruction_version": "pilot-v0.2-draft",
        "instruction_hash": "1" * 64,
        "task_content_hash": _task_payload()["task_content_hash"],
        "assignment_batch": "pilot-batch-01",
        "answerability": "answerable",
        "answer_text": "Riverton",
        "minimal_sufficient_evidence_sets": [["P1.S1", "P2.S1"]],
        "evidence_exhaustive": True,
        "ambiguity": False,
        "dataset_defect": False,
        "confidence": 5,
        "rationale": "P1 identifies the architect and P2 states the city.",
        "started_at": "2026-08-23T01:00:00Z",
        "submitted_at": "2026-08-23T01:04:00Z",
    }
    payload.update(updates)
    return payload


def test_blind_task_accepts_only_hash_bound_allowlisted_fields() -> None:
    task = BlindTask.model_validate(_task_payload())

    assert task.passages[1].sentences[0].alias == "P2.S1"
    assert task.task_content_hash == _task_payload()["task_content_hash"]

    poisoned = {**_task_payload(), "expected_answerability": "unanswerable"}
    with pytest.raises(ValidationError, match="extra"):
        BlindTask.model_validate(poisoned)

    tampered = {**_task_payload(), "question": "Changed question"}
    with pytest.raises(ValidationError, match="content hash"):
        BlindTask.model_validate(tampered)


def test_answerability_annotation_enforces_conditional_evidence_contract() -> None:
    answer = AnswerabilityAnnotation.model_validate(_answer_payload())
    assert answer.answer_text == "Riverton"
    assert answer.minimal_sufficient_evidence_sets == (("P1.S1", "P2.S1"),)

    with pytest.raises(ValidationError, match="answer text"):
        AnswerabilityAnnotation.model_validate(_answer_payload(answer_text=" "))
    with pytest.raises(ValidationError, match="evidence set"):
        AnswerabilityAnnotation.model_validate(_answer_payload(minimal_sufficient_evidence_sets=[]))
    with pytest.raises(ValidationError, match="must be empty"):
        AnswerabilityAnnotation.model_validate(_answer_payload(answerability="unanswerable"))

    unanswerable = AnswerabilityAnnotation.model_validate(
        _answer_payload(
            answerability="unanswerable",
            answer_text=None,
            minimal_sufficient_evidence_sets=[],
            evidence_exhaustive=None,
        )
    )
    assert unanswerable.minimal_sufficient_evidence_sets == ()


def test_annotation_rejects_duplicate_aliases_pii_and_bad_time_order() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        AnswerabilityAnnotation.model_validate(
            _answer_payload(minimal_sufficient_evidence_sets=[["P1.S1", "P1.S1"]])
        )
    with pytest.raises(ValidationError, match="pseudonym"):
        AnswerabilityAnnotation.model_validate(
            _answer_payload(annotator_pseudonym="person@example.com")
        )
    with pytest.raises(ValidationError, match="after"):
        AnswerabilityAnnotation.model_validate(_answer_payload(submitted_at="2026-08-23T00:59:59Z"))
    with pytest.raises(ValidationError, match="UTC"):
        AnswerabilityAnnotation.model_validate(
            _answer_payload(submitted_at="2026-08-23T01:04:00+08:00")
        )


def test_citation_annotation_keeps_answer_and_citation_judgments_separate() -> None:
    citation = CitationAnnotation.model_validate(
        {
            "schema_version": "citation-annotation-v1",
            "annotation_task_id": "task-0123456789abcdef01234567",
            "challenge_id": "ch-0123456789abcdef01234567",
            "blinded_parent_group": "bg-0123456789abcdef01234567",
            "annotator_pseudonym": "ann-r7",
            "instruction_version": "pilot-v0.2-draft",
            "instruction_hash": "1" * 64,
            "task_content_hash": _task_payload()["task_content_hash"],
            "assignment_batch": "pilot-batch-01",
            "generated_answer": "Riverton [P2.S1]",
            "sentence_citation_ids": ["P2.S1"],
            "citation_support": "supported",
            "missing_evidence": False,
            "answer_correctness": "correct",
            "abstention_correctness": "not_applicable",
            "confidence": 4,
            "rationale": "The cited sentence directly states the answer.",
            "started_at": "2026-08-23T01:00:00Z",
            "submitted_at": "2026-08-23T01:03:00Z",
        }
    )

    assert citation.answer_correctness == "correct"
    assert citation.citation_support == "supported"


def test_amendment_binds_original_and_preserves_replacement() -> None:
    original = AnswerabilityAnnotation.model_validate(_answer_payload())
    replacement = AnswerabilityAnnotation.model_validate(
        _answer_payload(
            answer_text="Riverton City",
            rationale="Correction: the visible passage uses the full city name.",
            submitted_at="2026-08-23T01:06:00Z",
        )
    )
    amendment = AnnotationAmendment.model_validate(
        {
            "schema_version": "annotation-amendment-v1",
            "amendment_id": "amend-0123456789abcdef01234567",
            "original_annotation_hash": artifact_hash(original),
            "previous_amendment_hash": None,
            "annotator_pseudonym": "ann-r7",
            "reason": "Answer text needed the full visible name.",
            "replacement": replacement.model_dump(mode="json"),
            "created_at": "2026-08-23T01:07:00Z",
        }
    )

    assert amendment.original_annotation_hash == artifact_hash(original)
    assert amendment.replacement.answer_text == "Riverton City"


def test_adjudication_requires_two_independent_originals_and_a_third_human() -> None:
    left = AnswerabilityAnnotation.model_validate(_answer_payload())
    right = AnswerabilityAnnotation.model_validate(
        _answer_payload(
            annotator_pseudonym="ann-k2",
            answerability="unclear",
            answer_text=None,
            minimal_sufficient_evidence_sets=[],
            evidence_exhaustive=None,
            rationale="The second hop may be ambiguous.",
        )
    )
    adjudication = AdjudicationRecord.model_validate(
        {
            "schema_version": "adjudication-v1",
            "adjudication_id": "adj-0123456789abcdef01234567",
            "annotation_task_id": left.annotation_task_id,
            "challenge_id": left.challenge_id,
            "blinded_parent_group": left.blinded_parent_group,
            "adjudicator_pseudonym": "ann-j9",
            "left": left.model_dump(mode="json"),
            "right": right.model_dump(mode="json"),
            "final_answerability": "answerable",
            "answer_text": "Riverton",
            "minimal_sufficient_evidence_sets": [["P1.S1", "P2.S1"]],
            "evidence_exhaustive": True,
            "excluded": False,
            "exclusion_reason": None,
            "rationale": "Both visible sentences form a sufficient chain.",
            "created_at": "2026-08-23T02:00:00Z",
        }
    )

    assert adjudication.left.annotator_pseudonym == "ann-r7"
    assert adjudication.right.annotator_pseudonym == "ann-k2"

    with pytest.raises(ValidationError, match="third independent"):
        AdjudicationRecord.model_validate(
            {**adjudication.model_dump(mode="json"), "adjudicator_pseudonym": "ann-r7"}
        )


def test_eligibility_never_resolves_disagreement_without_adjudication() -> None:
    left = AnswerabilityAnnotation.model_validate(_answer_payload())
    right = AnswerabilityAnnotation.model_validate(
        _answer_payload(
            annotator_pseudonym="ann-k2",
            answerability="unanswerable",
            answer_text=None,
            minimal_sufficient_evidence_sets=[],
            evidence_exhaustive=None,
            rationale="The visible evidence does not complete the second hop.",
        )
    )
    base = {
        "schema_version": "eligibility-record-v1",
        "challenge_id": left.challenge_id,
        "task_content_hash": left.task_content_hash,
        "source_annotation_hashes": [artifact_hash(left), artifact_hash(right)],
        "source_answerabilities": [left.answerability, right.answerability],
        "adjudication_hash": None,
        "final_answerability": "answerable",
        "eligible": True,
        "exclusion_reason": None,
    }
    with pytest.raises(ValidationError, match="adjudication"):
        EligibilityRecord.model_validate(base)

    record = EligibilityRecord.model_validate({**base, "adjudication_hash": "9" * 64})
    artifact = EligibilityArtifact.model_validate(
        {
            "schema_version": "eligibility-artifact-v1",
            "phase": "confirmatory",
            "protocol_version": "v2-confirmatory-frozen",
            "protocol_hash": "8" * 64,
            "generated_at": "2026-08-23T03:00:00Z",
            "records": [record.model_dump(mode="json")],
        }
    )
    assert artifact.records[0].eligible is True
