"""Two-human completeness, disagreement, adjudication, and eligibility gates."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_evidence.annotation.assignment import AssignmentManifest, TaskAssignment
from rag_evidence.annotation.models import (
    AdjudicationRecord,
    AnswerabilityAnnotation,
    EligibilityArtifact,
    EligibilityRecord,
    artifact_hash,
)
from rag_evidence.errors import ArtifactError, DataError
from rag_evidence.storage.artifacts import append_record, read_records

DisagreementReason = Literal[
    "answerability",
    "answer_text",
    "evidence_sets",
    "ambiguity",
    "dataset_defect",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DisagreementCase(_StrictModel):
    schema_version: Literal["disagreement-case-v1"] = "disagreement-case-v1"
    annotation_task_id: str
    challenge_id: str
    blinded_parent_group: str
    reasons: tuple[DisagreementReason, ...]
    left: AnswerabilityAnnotation
    right: AnswerabilityAnnotation


class FlowAccounting(_StrictModel):
    assigned: int = Field(ge=0)
    completed: int = Field(ge=0)
    disagreed: int = Field(ge=0)
    adjudicated: int = Field(ge=0)
    excluded: int = Field(ge=0)
    eligible: int = Field(ge=0)

    @model_validator(mode="after")
    def _flow_is_coherent(self) -> FlowAccounting:
        if self.completed > self.assigned:
            raise ValueError("completed cannot exceed assigned")
        if self.disagreed > self.completed:
            raise ValueError("disagreed cannot exceed completed")
        if self.adjudicated > self.disagreed:
            raise ValueError("adjudicated cannot exceed disagreed")
        if self.excluded + self.eligible > self.completed:
            raise ValueError("excluded + eligible cannot exceed completed")
        return self


class WorkflowResult(_StrictModel):
    eligibility: EligibilityArtifact
    flow: FlowAccounting
    disagreements: tuple[DisagreementCase, ...]


def _task_map(manifest: AssignmentManifest) -> dict[str, object]:
    tasks: dict[str, object] = {}
    for package in manifest.packages:
        for task in package.tasks:
            tasks.setdefault(task.annotation_task_id, task)
    return tasks


def _index_submissions(
    manifest: AssignmentManifest,
    submissions: Sequence[AnswerabilityAnnotation],
) -> dict[str, dict[str, AnswerabilityAnnotation]]:
    assignments = {
        assignment.annotation_task_id: assignment for assignment in manifest.task_assignments
    }
    tasks = _task_map(manifest)
    indexed: dict[str, dict[str, AnswerabilityAnnotation]] = {}
    for record in submissions:
        assignment = assignments.get(record.annotation_task_id)
        if assignment is None:
            raise DataError(f"submission task {record.annotation_task_id} is not assigned")
        if record.annotator_pseudonym not in assignment.annotators:
            raise DataError("submission annotator is not assigned to the task")
        by_annotator = indexed.setdefault(record.annotation_task_id, {})
        if record.annotator_pseudonym in by_annotator:
            raise DataError(
                f"duplicate submission for {record.annotation_task_id}/{record.annotator_pseudonym}"
            )
        task = tasks[record.annotation_task_id]
        for field in (
            "challenge_id",
            "blinded_parent_group",
            "instruction_version",
            "instruction_hash",
            "task_content_hash",
            "assignment_batch",
        ):
            if getattr(record, field) != getattr(task, field):
                raise DataError(f"submission {field} does not match blind task")
        by_annotator[record.annotator_pseudonym] = record
    return indexed


def _normalized_answer(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip().casefold()


def _normalized_sets(record: AnswerabilityAnnotation) -> tuple[tuple[str, ...], ...]:
    return tuple(
        sorted(
            tuple(sorted(evidence_set)) for evidence_set in record.minimal_sufficient_evidence_sets
        )
    )


def _reasons(
    left: AnswerabilityAnnotation, right: AnswerabilityAnnotation
) -> tuple[DisagreementReason, ...]:
    reasons: list[DisagreementReason] = []
    if left.answerability != right.answerability:
        reasons.append("answerability")
    if _normalized_answer(left.answer_text) != _normalized_answer(right.answer_text):
        reasons.append("answer_text")
    if _normalized_sets(left) != _normalized_sets(right):
        reasons.append("evidence_sets")
    if left.ambiguity != right.ambiguity:
        reasons.append("ambiguity")
    if left.dataset_defect != right.dataset_defect:
        reasons.append("dataset_defect")
    return tuple(reasons)


def _case(
    assignment: TaskAssignment,
    by_annotator: Mapping[str, AnswerabilityAnnotation],
) -> DisagreementCase | None:
    left, right = sorted(by_annotator.values(), key=lambda record: record.annotator_pseudonym)
    reasons = _reasons(left, right)
    if not reasons:
        return None
    return DisagreementCase(
        annotation_task_id=assignment.annotation_task_id,
        challenge_id=left.challenge_id,
        blinded_parent_group=left.blinded_parent_group,
        reasons=reasons,
        left=left,
        right=right,
    )


def build_disagreement_queue(
    manifest: AssignmentManifest,
    submissions: Sequence[AnswerabilityAnnotation],
) -> tuple[DisagreementCase, ...]:
    indexed = _index_submissions(manifest, submissions)
    cases: list[DisagreementCase] = []
    for assignment in manifest.task_assignments:
        by_annotator = indexed.get(assignment.annotation_task_id, {})
        if len(by_annotator) != 2:
            raise DataError(
                f"task {assignment.annotation_task_id} requires two completed independent "
                "annotations before disagreement review"
            )
        case = _case(assignment, by_annotator)
        if case is not None:
            cases.append(case)
    return tuple(cases)


def _agreement_exclusion(records: Sequence[AnswerabilityAnnotation]) -> str | None:
    if records[0].answerability == "unclear":
        return "unclear human decision"
    if any(record.dataset_defect for record in records):
        return "dataset defect"
    if any(record.ambiguity for record in records):
        return "annotator-marked ambiguity"
    if records[0].answerability == "answerable" and any(
        record.evidence_exhaustive is False for record in records
    ):
        return "non-exhaustive evidence"
    return None


def _validate_adjudication(case: DisagreementCase, record: AdjudicationRecord) -> None:
    if record.annotation_task_id != case.annotation_task_id:
        raise DataError("adjudication task does not match disagreement case")
    expected = {artifact_hash(case.left), artifact_hash(case.right)}
    actual = {artifact_hash(record.left), artifact_hash(record.right)}
    if actual != expected:
        raise DataError("adjudication original annotation hashes do not match submissions")


def build_workflow_result(
    manifest: AssignmentManifest,
    submissions: Sequence[AnswerabilityAnnotation],
    *,
    adjudications: Sequence[AdjudicationRecord],
    phase: Literal["pilot", "confirmatory"],
    protocol_version: str,
    protocol_hash: str,
    generated_at: str,
) -> WorkflowResult:
    indexed = _index_submissions(manifest, submissions)
    adjudication_by_task: dict[str, AdjudicationRecord] = {}
    for record in adjudications:
        if record.annotation_task_id in adjudication_by_task:
            raise DataError(f"duplicate adjudication for {record.annotation_task_id}")
        adjudication_by_task[record.annotation_task_id] = record

    completed = disagreed = adjudicated = excluded = eligible = 0
    disagreements: list[DisagreementCase] = []
    eligibility_records: list[EligibilityRecord] = []
    consumed_adjudications: set[str] = set()

    for assignment in manifest.task_assignments:
        by_annotator = indexed.get(assignment.annotation_task_id, {})
        if len(by_annotator) != 2:
            continue
        completed += 1
        originals = tuple(
            sorted(by_annotator.values(), key=lambda record: record.annotator_pseudonym)
        )
        case = _case(assignment, by_annotator)
        adjudication_hash: str | None = None
        final_answerability = originals[0].answerability
        exclusion_reason: str | None
        if case is not None:
            disagreed += 1
            disagreements.append(case)
            adjudication = adjudication_by_task.get(assignment.annotation_task_id)
            if adjudication is None:
                continue
            _validate_adjudication(case, adjudication)
            consumed_adjudications.add(assignment.annotation_task_id)
            adjudicated += 1
            adjudication_hash = artifact_hash(adjudication)
            final_answerability = adjudication.final_answerability
            exclusion_reason = adjudication.exclusion_reason if adjudication.excluded else None
        else:
            if assignment.annotation_task_id in adjudication_by_task:
                raise DataError("adjudication supplied for a task without disagreement")
            exclusion_reason = _agreement_exclusion(originals)

        is_eligible = exclusion_reason is None and final_answerability != "unclear"
        if is_eligible:
            eligible += 1
        else:
            excluded += 1
            exclusion_reason = exclusion_reason or "unclear adjudicated decision"
        eligibility_records.append(
            EligibilityRecord(
                challenge_id=originals[0].challenge_id,
                task_content_hash=originals[0].task_content_hash,
                source_annotation_hashes=(
                    artifact_hash(originals[0]),
                    artifact_hash(originals[1]),
                ),
                source_answerabilities=(
                    originals[0].answerability,
                    originals[1].answerability,
                ),
                adjudication_hash=adjudication_hash,
                final_answerability=final_answerability,
                eligible=is_eligible,
                exclusion_reason=exclusion_reason,
            )
        )

    extra = set(adjudication_by_task) - consumed_adjudications
    if extra:
        raise DataError(f"adjudications do not match disagreement cases: {sorted(extra)}")
    artifact = EligibilityArtifact(
        phase=phase,
        protocol_version=protocol_version,
        protocol_hash=protocol_hash,
        generated_at=generated_at,
        records=tuple(eligibility_records),
    )
    flow = FlowAccounting(
        assigned=len(manifest.task_assignments),
        completed=completed,
        disagreed=disagreed,
        adjudicated=adjudicated,
        excluded=excluded,
        eligible=eligible,
    )
    return WorkflowResult(eligibility=artifact, flow=flow, disagreements=tuple(disagreements))


class AdjudicationStore:
    """Append-only third-human decisions bound to a complete disagreement queue."""

    def __init__(
        self,
        root: Path,
        manifest: AssignmentManifest,
        submissions: Sequence[AnswerabilityAnnotation],
    ) -> None:
        self.path = root / "adjudications" / "records.jsonl"
        self.cases = {
            case.annotation_task_id: case
            for case in build_disagreement_queue(manifest, submissions)
        }

    def records(self) -> tuple[AdjudicationRecord, ...]:
        if not self.path.exists():
            return ()
        return tuple(
            AdjudicationRecord.model_validate(payload) for payload in read_records(self.path)
        )

    def submit(self, payload: Mapping[str, Any]) -> AdjudicationRecord:
        record = AdjudicationRecord.model_validate(payload)
        case = self.cases.get(record.annotation_task_id)
        if case is None:
            raise ArtifactError("adjudication task is not in the disagreement queue")
        if any(
            existing.annotation_task_id == record.annotation_task_id for existing in self.records()
        ):
            raise ArtifactError(f"task {record.annotation_task_id} is already adjudicated")
        try:
            _validate_adjudication(case, record)
        except DataError as exc:
            raise ArtifactError(str(exc)) from exc
        append_record(self.path, record.model_dump(mode="json"))
        return record
