"""Strict, versioned records for independent-human annotation artifacts."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_HEX64_RE = re.compile(r"[0-9a-f]{64}")
_PSEUDONYM_RE = re.compile(r"ann-[a-z0-9](?:[a-z0-9-]{0,30}[a-z0-9])?")
_SENTENCE_ALIAS_RE = re.compile(r"P[1-9]\d*\.S[1-9]\d*")
_SLUG_RE = re.compile(r"[a-z0-9][a-z0-9._-]*")

Hash64 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
TaskId = Annotated[str, Field(pattern=r"^task-[0-9a-f]{24}$")]
ChallengeId = Annotated[str, Field(pattern=r"^ch-[0-9a-f]{24}$")]
BlindGroupId = Annotated[str, Field(pattern=r"^bg-[0-9a-f]{24}$")]
Answerability = Literal["answerable", "unanswerable", "unclear"]
EvidenceSets = tuple[tuple[str, ...], ...]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _canonical_hash(payload: object) -> str:
    material = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def artifact_hash(record: BaseModel) -> str:
    """SHA-256 over a validated artifact's canonical JSON representation."""
    return _canonical_hash(record.model_dump(mode="json"))


def _utc(value: dt.datetime, field_name: str) -> dt.datetime:
    if value.tzinfo is None or value.utcoffset() != dt.timedelta(0):
        raise ValueError(f"{field_name} must be an explicit UTC timestamp")
    return value


def _pseudonym(value: str) -> str:
    if _PSEUDONYM_RE.fullmatch(value) is None:
        raise ValueError("annotator pseudonym must be an opaque ann-* identifier, not PII")
    return value


def _validate_evidence_sets(value: EvidenceSets) -> EvidenceSets:
    seen_families: set[tuple[str, ...]] = set()
    for evidence_set in value:
        if not evidence_set:
            raise ValueError("minimal sufficient evidence set must be nonempty")
        if len(set(evidence_set)) != len(evidence_set):
            raise ValueError("evidence set contains a duplicate sentence alias")
        invalid = [alias for alias in evidence_set if _SENTENCE_ALIAS_RE.fullmatch(alias) is None]
        if invalid:
            raise ValueError(f"invalid sentence aliases in evidence set: {invalid}")
        canonical = tuple(sorted(evidence_set))
        if canonical in seen_families:
            raise ValueError("duplicate minimal sufficient evidence set")
        seen_families.add(canonical)
    return value


def _validate_answerability_fields(
    answerability: Answerability,
    answer_text: str | None,
    evidence_sets: EvidenceSets,
    evidence_exhaustive: bool | None,
) -> None:
    if answerability == "answerable":
        if answer_text is None or not answer_text.strip():
            raise ValueError("answerable decision requires nonempty answer text")
        if not evidence_sets:
            raise ValueError("answerable decision requires at least one evidence set")
        if evidence_exhaustive is None:
            raise ValueError("answerable decision requires evidence_exhaustive")
        return
    if answer_text is not None and answer_text.strip():
        raise ValueError("answer text and evidence must be empty unless answerable")
    if evidence_sets:
        raise ValueError("answer text and evidence must be empty unless answerable")
    if evidence_exhaustive is not None:
        raise ValueError("evidence_exhaustive must be null unless answerable")


class TaskSentence(_StrictModel):
    alias: Annotated[str, Field(pattern=r"^P[1-9]\d*\.S[1-9]\d*$")]
    text: Annotated[str, Field(min_length=1)]

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("sentence text must not be blank")
        return value


class TaskPassage(_StrictModel):
    alias: Annotated[str, Field(pattern=r"^P[1-9]\d*$")]
    title: Annotated[str, Field(min_length=1)]
    sentences: tuple[TaskSentence, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _sentence_aliases_match_passage(self) -> TaskPassage:
        expected = tuple(f"{self.alias}.S{i}" for i in range(1, len(self.sentences) + 1))
        actual = tuple(sentence.alias for sentence in self.sentences)
        if actual != expected:
            raise ValueError("sentence aliases must be sequential and match the passage alias")
        return self


class BlindTask(_StrictModel):
    schema_version: Literal["blind-task-v1"] = "blind-task-v1"
    annotation_task_id: TaskId
    challenge_id: ChallengeId
    blinded_parent_group: BlindGroupId
    instruction_version: Annotated[str, Field(min_length=1)]
    instruction_hash: Hash64
    question: Annotated[str, Field(min_length=1)]
    passages: tuple[TaskPassage, ...] = Field(min_length=1)
    task_content_hash: Hash64
    assignment_batch: Annotated[str, Field(min_length=1)]

    @field_validator("instruction_version", "assignment_batch")
    @classmethod
    def _safe_slug(cls, value: str) -> str:
        if _SLUG_RE.fullmatch(value) is None:
            raise ValueError("version and batch values must be filesystem-safe slugs")
        return value

    @model_validator(mode="after")
    def _aliases_and_hash_match(self) -> BlindTask:
        expected_passages = tuple(f"P{i}" for i in range(1, len(self.passages) + 1))
        actual_passages = tuple(passage.alias for passage in self.passages)
        if actual_passages != expected_passages:
            raise ValueError("passage aliases must be sequential")
        content = {
            "challenge_id": self.challenge_id,
            "blinded_parent_group": self.blinded_parent_group,
            "instruction_version": self.instruction_version,
            "instruction_hash": self.instruction_hash,
            "question": self.question,
            "passages": [p.model_dump(mode="json") for p in self.passages],
        }
        if _canonical_hash(content) != self.task_content_hash:
            raise ValueError("blind task content hash mismatch")
        return self


class _AnnotationBase(_StrictModel):
    annotation_task_id: TaskId
    challenge_id: ChallengeId
    blinded_parent_group: BlindGroupId
    annotator_pseudonym: str
    instruction_version: Annotated[str, Field(min_length=1)]
    instruction_hash: Hash64
    task_content_hash: Hash64
    assignment_batch: Annotated[str, Field(min_length=1)]
    confidence: int = Field(ge=1, le=5)
    rationale: Annotated[str, Field(min_length=1, max_length=1000)]
    started_at: dt.datetime
    submitted_at: dt.datetime

    @field_validator("annotator_pseudonym")
    @classmethod
    def _opaque_pseudonym(cls, value: str) -> str:
        return _pseudonym(value)

    @field_validator("instruction_version", "assignment_batch")
    @classmethod
    def _annotation_slug(cls, value: str) -> str:
        if _SLUG_RE.fullmatch(value) is None:
            raise ValueError("version and batch values must be filesystem-safe slugs")
        return value

    @field_validator("rationale")
    @classmethod
    def _rationale_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("rationale must not be blank")
        return value.strip()

    @field_validator("started_at", "submitted_at")
    @classmethod
    def _timestamps_are_utc(cls, value: dt.datetime, info: object) -> dt.datetime:
        field_name = getattr(info, "field_name", "timestamp")
        return _utc(value, str(field_name))

    @model_validator(mode="after")
    def _submission_after_start(self) -> _AnnotationBase:
        if self.submitted_at < self.started_at:
            raise ValueError("submitted_at must be at or after started_at")
        return self


class AnswerabilityAnnotation(_AnnotationBase):
    schema_version: Literal["answerability-annotation-v1"] = "answerability-annotation-v1"
    answerability: Answerability
    answer_text: str | None
    minimal_sufficient_evidence_sets: EvidenceSets
    evidence_exhaustive: bool | None
    ambiguity: bool
    dataset_defect: bool

    @field_validator("minimal_sufficient_evidence_sets")
    @classmethod
    def _evidence_sets_valid(cls, value: EvidenceSets) -> EvidenceSets:
        return _validate_evidence_sets(value)

    @model_validator(mode="after")
    def _conditional_fields(self) -> AnswerabilityAnnotation:
        _validate_answerability_fields(
            self.answerability,
            self.answer_text,
            self.minimal_sufficient_evidence_sets,
            self.evidence_exhaustive,
        )
        return self


class CitationAnnotation(_AnnotationBase):
    schema_version: Literal["citation-annotation-v1"] = "citation-annotation-v1"
    generated_answer: Annotated[str, Field(min_length=1)]
    sentence_citation_ids: tuple[str, ...]
    citation_support: Literal["supported", "partial", "unsupported", "invalid"]
    missing_evidence: bool
    answer_correctness: Literal["correct", "incorrect", "unclear"]
    abstention_correctness: Literal["correct", "incorrect", "not_applicable", "unclear"]

    @field_validator("sentence_citation_ids")
    @classmethod
    def _citation_aliases(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("duplicate sentence citation ID")
        invalid = [alias for alias in value if _SENTENCE_ALIAS_RE.fullmatch(alias) is None]
        if invalid:
            raise ValueError(f"invalid sentence citation IDs: {invalid}")
        return value


AnnotationRecord = AnswerabilityAnnotation | CitationAnnotation


class AnnotationAmendment(_StrictModel):
    schema_version: Literal["annotation-amendment-v1"] = "annotation-amendment-v1"
    amendment_id: Annotated[str, Field(pattern=r"^amend-[0-9a-f]{24}$")]
    original_annotation_hash: Hash64
    previous_amendment_hash: Hash64 | None
    annotator_pseudonym: str
    reason: Annotated[str, Field(min_length=1, max_length=1000)]
    replacement: AnnotationRecord
    created_at: dt.datetime

    @field_validator("annotator_pseudonym")
    @classmethod
    def _amendment_pseudonym(cls, value: str) -> str:
        return _pseudonym(value)

    @field_validator("created_at")
    @classmethod
    def _created_utc(cls, value: dt.datetime) -> dt.datetime:
        return _utc(value, "created_at")

    @model_validator(mode="after")
    def _same_annotator(self) -> AnnotationAmendment:
        if self.annotator_pseudonym != self.replacement.annotator_pseudonym:
            raise ValueError("amendment and replacement annotator pseudonyms must match")
        if self.original_annotation_hash == artifact_hash(self.replacement):
            raise ValueError("amendment replacement must differ from the original")
        return self


class AdjudicationRecord(_StrictModel):
    schema_version: Literal["adjudication-v1"] = "adjudication-v1"
    adjudication_id: Annotated[str, Field(pattern=r"^adj-[0-9a-f]{24}$")]
    annotation_task_id: TaskId
    challenge_id: ChallengeId
    blinded_parent_group: BlindGroupId
    adjudicator_pseudonym: str
    left: AnswerabilityAnnotation
    right: AnswerabilityAnnotation
    final_answerability: Answerability
    answer_text: str | None
    minimal_sufficient_evidence_sets: EvidenceSets
    evidence_exhaustive: bool | None
    excluded: bool
    exclusion_reason: str | None
    rationale: Annotated[str, Field(min_length=1, max_length=1000)]
    created_at: dt.datetime

    @field_validator("adjudicator_pseudonym")
    @classmethod
    def _adjudicator_id(cls, value: str) -> str:
        return _pseudonym(value)

    @field_validator("minimal_sufficient_evidence_sets")
    @classmethod
    def _adjudicated_evidence(cls, value: EvidenceSets) -> EvidenceSets:
        return _validate_evidence_sets(value)

    @field_validator("created_at")
    @classmethod
    def _adjudicated_utc(cls, value: dt.datetime) -> dt.datetime:
        return _utc(value, "created_at")

    @model_validator(mode="after")
    def _bind_originals_and_decision(self) -> AdjudicationRecord:
        binding = (self.annotation_task_id, self.challenge_id, self.blinded_parent_group)
        for original in (self.left, self.right):
            if (
                original.annotation_task_id,
                original.challenge_id,
                original.blinded_parent_group,
            ) != binding:
                raise ValueError("adjudication originals do not match the task binding")
        original_annotators = {
            self.left.annotator_pseudonym,
            self.right.annotator_pseudonym,
        }
        if len(original_annotators) != 2:
            raise ValueError("adjudication requires two independent original annotators")
        if self.adjudicator_pseudonym in original_annotators:
            raise ValueError("adjudicator must be a third independent human")
        _validate_answerability_fields(
            self.final_answerability,
            self.answer_text,
            self.minimal_sufficient_evidence_sets,
            self.evidence_exhaustive,
        )
        if self.excluded and not (self.exclusion_reason and self.exclusion_reason.strip()):
            raise ValueError("excluded adjudication requires an exclusion reason")
        if not self.excluded and self.exclusion_reason is not None:
            raise ValueError("non-excluded adjudication cannot have an exclusion reason")
        if self.final_answerability == "unclear" and not self.excluded:
            raise ValueError("an unclear final decision must remain excluded")
        return self


class EligibilityRecord(_StrictModel):
    schema_version: Literal["eligibility-record-v1"] = "eligibility-record-v1"
    challenge_id: ChallengeId
    task_content_hash: Hash64
    source_annotation_hashes: tuple[Hash64, Hash64]
    source_answerabilities: tuple[Answerability, Answerability]
    adjudication_hash: Hash64 | None
    final_answerability: Answerability
    eligible: bool
    exclusion_reason: str | None

    @model_validator(mode="after")
    def _fail_closed(self) -> EligibilityRecord:
        if len(set(self.source_annotation_hashes)) != 2:
            raise ValueError("eligibility requires two distinct annotation hashes")
        disagreement = self.source_answerabilities[0] != self.source_answerabilities[1]
        final_changed = any(
            value != self.final_answerability for value in self.source_answerabilities
        )
        if (disagreement or final_changed) and self.adjudication_hash is None:
            raise ValueError("disagreement or unclear decision requires adjudication")
        if self.eligible:
            if self.final_answerability == "unclear":
                raise ValueError("unclear records cannot be eligible")
            if self.exclusion_reason is not None:
                raise ValueError("eligible record cannot have an exclusion reason")
        elif not (self.exclusion_reason and self.exclusion_reason.strip()):
            raise ValueError("ineligible record requires an exclusion reason")
        return self


class EligibilityArtifact(_StrictModel):
    schema_version: Literal["eligibility-artifact-v1"] = "eligibility-artifact-v1"
    phase: Literal["pilot", "confirmatory"]
    protocol_version: Annotated[str, Field(min_length=1)]
    protocol_hash: Hash64
    generated_at: dt.datetime
    records: tuple[EligibilityRecord, ...]

    @field_validator("generated_at")
    @classmethod
    def _artifact_utc(cls, value: dt.datetime) -> dt.datetime:
        return _utc(value, "generated_at")

    @model_validator(mode="after")
    def _unique_challenges(self) -> EligibilityArtifact:
        challenge_ids = [record.challenge_id for record in self.records]
        if len(set(challenge_ids)) != len(challenge_ids):
            raise ValueError("eligibility artifact contains duplicate challenge IDs")
        return self


# Breaking-v2 human-visible models deliberately do not inherit from the historical v1
# models because a v1 group-bearing field must never be accepted through inheritance.
class BlindTaskV2(_StrictModel):
    schema_version: Literal["blind-task-v2"] = "blind-task-v2"
    annotation_task_id: TaskId
    challenge_id: ChallengeId
    instruction_version: Annotated[str, Field(min_length=1)]
    instruction_hash: Hash64
    question: Annotated[str, Field(min_length=1)]
    passages: tuple[TaskPassage, ...] = Field(min_length=1)
    task_content_hash: Hash64
    assignment_batch: Annotated[str, Field(min_length=1)]

    @field_validator("instruction_version", "assignment_batch")
    @classmethod
    def _safe_v2_slug(cls, value: str) -> str:
        if _SLUG_RE.fullmatch(value) is None:
            raise ValueError("version and batch values must be filesystem-safe slugs")
        return value

    @model_validator(mode="after")
    def _v2_aliases_and_hash_match(self) -> BlindTaskV2:
        expected_passages = tuple(f"P{i}" for i in range(1, len(self.passages) + 1))
        actual_passages = tuple(passage.alias for passage in self.passages)
        if actual_passages != expected_passages:
            raise ValueError("passage aliases must be sequential")
        content = {
            "challenge_id": self.challenge_id,
            "instruction_version": self.instruction_version,
            "instruction_hash": self.instruction_hash,
            "question": self.question,
            "passages": [passage.model_dump(mode="json") for passage in self.passages],
        }
        if _canonical_hash(content) != self.task_content_hash:
            raise ValueError("blind task content hash mismatch")
        return self


class _AnnotationBaseV2(_StrictModel):
    annotation_task_id: TaskId
    challenge_id: ChallengeId
    annotator_pseudonym: str
    instruction_version: Annotated[str, Field(min_length=1)]
    instruction_hash: Hash64
    task_content_hash: Hash64
    assignment_batch: Annotated[str, Field(min_length=1)]
    confidence: int = Field(ge=1, le=5)
    rationale: Annotated[str, Field(min_length=1, max_length=1000)]
    started_at: dt.datetime
    submitted_at: dt.datetime

    @field_validator("annotator_pseudonym")
    @classmethod
    def _v2_opaque_pseudonym(cls, value: str) -> str:
        return _pseudonym(value)

    @field_validator("instruction_version", "assignment_batch")
    @classmethod
    def _v2_annotation_slug(cls, value: str) -> str:
        if _SLUG_RE.fullmatch(value) is None:
            raise ValueError("version and batch values must be filesystem-safe slugs")
        return value

    @field_validator("rationale")
    @classmethod
    def _v2_rationale_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("rationale must not be blank")
        return value.strip()

    @field_validator("started_at", "submitted_at")
    @classmethod
    def _v2_timestamps_are_utc(cls, value: dt.datetime, info: object) -> dt.datetime:
        field_name = getattr(info, "field_name", "timestamp")
        return _utc(value, str(field_name))

    @model_validator(mode="after")
    def _v2_submission_after_start(self) -> _AnnotationBaseV2:
        if self.submitted_at < self.started_at:
            raise ValueError("submitted_at must be at or after started_at")
        return self


class AnswerabilityAnnotationV2(_AnnotationBaseV2):
    schema_version: Literal["answerability-annotation-v2"] = "answerability-annotation-v2"
    answerability: Answerability
    answer_text: str | None
    minimal_sufficient_evidence_sets: EvidenceSets
    evidence_exhaustive: bool | None
    ambiguity: bool
    dataset_defect: bool

    @field_validator("minimal_sufficient_evidence_sets")
    @classmethod
    def _v2_evidence_sets_valid(cls, value: EvidenceSets) -> EvidenceSets:
        return _validate_evidence_sets(value)

    @model_validator(mode="after")
    def _v2_conditional_fields(self) -> AnswerabilityAnnotationV2:
        _validate_answerability_fields(
            self.answerability,
            self.answer_text,
            self.minimal_sufficient_evidence_sets,
            self.evidence_exhaustive,
        )
        return self


class CitationAnnotationV2(_AnnotationBaseV2):
    schema_version: Literal["citation-annotation-v2"] = "citation-annotation-v2"
    generated_answer: Annotated[str, Field(min_length=1)]
    sentence_citation_ids: tuple[str, ...]
    citation_support: Literal["supported", "partial", "unsupported", "invalid"]
    missing_evidence: bool
    answer_correctness: Literal["correct", "incorrect", "unclear"]
    abstention_correctness: Literal["correct", "incorrect", "not_applicable", "unclear"]

    @field_validator("sentence_citation_ids")
    @classmethod
    def _v2_citation_aliases(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("duplicate sentence citation ID")
        invalid = [alias for alias in value if _SENTENCE_ALIAS_RE.fullmatch(alias) is None]
        if invalid:
            raise ValueError(f"invalid sentence citation IDs: {invalid}")
        return value


AnnotationRecordV2 = AnswerabilityAnnotationV2 | CitationAnnotationV2
BindingTuple = tuple[TaskId, ChallengeId, Hash64, str, Hash64, str, str]


def annotation_binding(record: _AnnotationBaseV2) -> BindingTuple:
    """Return the complete owner-approved group-free assignment binding."""
    return (
        record.annotation_task_id,
        record.challenge_id,
        record.task_content_hash,
        record.instruction_version,
        record.instruction_hash,
        record.assignment_batch,
        record.annotator_pseudonym,
    )


def _task_binding_v2(record: _AnnotationBaseV2) -> tuple[str, str, str, str, str, str]:
    return (
        record.annotation_task_id,
        record.challenge_id,
        record.task_content_hash,
        record.instruction_version,
        record.instruction_hash,
        record.assignment_batch,
    )


class AnnotationAmendmentV2(_StrictModel):
    schema_version: Literal["annotation-amendment-v2"] = "annotation-amendment-v2"
    amendment_id: Annotated[str, Field(pattern=r"^amend-[0-9a-f]{24}$")]
    original_annotation_hash: Hash64
    previous_amendment_hash: Hash64 | None
    annotator_pseudonym: str
    reason: Annotated[str, Field(min_length=1, max_length=1000)]
    replacement: AnnotationRecordV2
    created_at: dt.datetime

    @field_validator("annotator_pseudonym")
    @classmethod
    def _v2_amendment_pseudonym(cls, value: str) -> str:
        return _pseudonym(value)

    @field_validator("created_at")
    @classmethod
    def _v2_created_utc(cls, value: dt.datetime) -> dt.datetime:
        return _utc(value, "created_at")

    @model_validator(mode="after")
    def _v2_same_annotator(self) -> AnnotationAmendmentV2:
        if self.annotator_pseudonym != self.replacement.annotator_pseudonym:
            raise ValueError("amendment and replacement annotator pseudonyms must match")
        if self.original_annotation_hash == artifact_hash(self.replacement):
            raise ValueError("amendment replacement must differ from the original")
        return self


class AdjudicationV2(_StrictModel):
    schema_version: Literal["adjudication-v2"] = "adjudication-v2"
    adjudication_id: Annotated[str, Field(pattern=r"^adj-[0-9a-f]{24}$")]
    annotation_task_id: TaskId
    challenge_id: ChallengeId
    adjudicator_pseudonym: str
    left: AnswerabilityAnnotationV2
    right: AnswerabilityAnnotationV2
    final_answerability: Answerability
    answer_text: str | None
    minimal_sufficient_evidence_sets: EvidenceSets
    evidence_exhaustive: bool | None
    excluded: bool
    exclusion_reason: str | None
    rationale: Annotated[str, Field(min_length=1, max_length=1000)]
    created_at: dt.datetime

    @field_validator("adjudicator_pseudonym")
    @classmethod
    def _v2_adjudicator_id(cls, value: str) -> str:
        return _pseudonym(value)

    @field_validator("minimal_sufficient_evidence_sets")
    @classmethod
    def _v2_adjudicated_evidence(cls, value: EvidenceSets) -> EvidenceSets:
        return _validate_evidence_sets(value)

    @field_validator("created_at")
    @classmethod
    def _v2_adjudicated_utc(cls, value: dt.datetime) -> dt.datetime:
        return _utc(value, "created_at")

    @model_validator(mode="after")
    def _v2_bind_originals_and_decision(self) -> AdjudicationV2:
        if self.left.annotation_task_id != self.annotation_task_id:
            raise ValueError("adjudication originals do not match the task binding")
        if self.left.challenge_id != self.challenge_id:
            raise ValueError("adjudication originals do not match the task binding")
        if _task_binding_v2(self.left) != _task_binding_v2(self.right):
            raise ValueError("adjudication originals do not match the task binding")
        original_annotators = {
            self.left.annotator_pseudonym,
            self.right.annotator_pseudonym,
        }
        if len(original_annotators) != 2:
            raise ValueError("adjudication requires two independent original annotators")
        if self.adjudicator_pseudonym in original_annotators:
            raise ValueError("adjudicator must be a third independent human")
        _validate_answerability_fields(
            self.final_answerability,
            self.answer_text,
            self.minimal_sufficient_evidence_sets,
            self.evidence_exhaustive,
        )
        if self.excluded and not (self.exclusion_reason and self.exclusion_reason.strip()):
            raise ValueError("excluded adjudication requires an exclusion reason")
        if not self.excluded and self.exclusion_reason is not None:
            raise ValueError("non-excluded adjudication cannot have an exclusion reason")
        if self.final_answerability == "unclear" and not self.excluded:
            raise ValueError("an unclear final decision must remain excluded")
        return self


class EligibilityRecordV2(_StrictModel):
    schema_version: Literal["eligibility-record-v2"] = "eligibility-record-v2"
    challenge_id: ChallengeId
    task_content_hash: Hash64
    source_annotation_hashes: tuple[Hash64, Hash64]
    source_answerabilities: tuple[Answerability, Answerability]
    adjudication_hash: Hash64 | None
    final_answerability: Answerability
    eligible: bool
    exclusion_reason: str | None

    @model_validator(mode="after")
    def _v2_fail_closed(self) -> EligibilityRecordV2:
        if len(set(self.source_annotation_hashes)) != 2:
            raise ValueError("eligibility requires two distinct annotation hashes")
        disagreement = self.source_answerabilities[0] != self.source_answerabilities[1]
        final_changed = any(
            value != self.final_answerability for value in self.source_answerabilities
        )
        if (disagreement or final_changed) and self.adjudication_hash is None:
            raise ValueError("disagreement or unclear decision requires adjudication")
        if self.eligible:
            if self.final_answerability == "unclear":
                raise ValueError("unclear records cannot be eligible")
            if self.exclusion_reason is not None:
                raise ValueError("eligible record cannot have an exclusion reason")
        elif not (self.exclusion_reason and self.exclusion_reason.strip()):
            raise ValueError("ineligible record requires an exclusion reason")
        return self


class EligibilityArtifactV2(_StrictModel):
    schema_version: Literal["eligibility-artifact-v2"] = "eligibility-artifact-v2"
    phase: Literal["pilot", "confirmatory"]
    protocol_version: Annotated[str, Field(min_length=1)]
    protocol_hash: Hash64
    generated_at: dt.datetime
    records: tuple[EligibilityRecordV2, ...]

    @field_validator("generated_at")
    @classmethod
    def _v2_artifact_utc(cls, value: dt.datetime) -> dt.datetime:
        return _utc(value, "generated_at")

    @model_validator(mode="after")
    def _v2_unique_challenges(self) -> EligibilityArtifactV2:
        challenge_ids = [record.challenge_id for record in self.records]
        if len(set(challenge_ids)) != len(challenge_ids):
            raise ValueError("eligibility artifact contains duplicate challenge IDs")
        return self
