"""Two-human completeness, disagreement, adjudication, and eligibility gates."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_evidence.annotation.assignment import AssignmentManifestV2, CoordinatorTaskV2
from rag_evidence.annotation.models import (
    AdjudicationV2,
    AnnotationAmendmentV2,
    AnswerabilityAnnotationV2,
    EligibilityArtifactV2,
    EligibilityRecordV2,
    artifact_hash,
)
from rag_evidence.annotation.privacy import scan_delivery_payload
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


class DisagreementCaseV2(_StrictModel):
    schema_version: Literal["disagreement-case-v2"] = "disagreement-case-v2"
    annotation_task_id: str
    challenge_id: str
    reasons: tuple[DisagreementReason, ...]
    left: AnswerabilityAnnotationV2
    right: AnswerabilityAnnotationV2


class FlowAccounting(_StrictModel):
    schema_version: Literal["pilot-flow-accounting-v2"] = "pilot-flow-accounting-v2"
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
    eligibility: EligibilityArtifactV2
    flow: FlowAccounting
    disagreements: tuple[DisagreementCaseV2, ...]


def _task_map(manifest: AssignmentManifestV2) -> dict[str, object]:
    return {task.annotation_task_id: task for task in manifest.tasks}


def index_submissions(
    manifest: AssignmentManifestV2,
    submissions: Sequence[AnswerabilityAnnotationV2],
) -> dict[str, dict[str, AnswerabilityAnnotationV2]]:
    assignments = {
        assignment.annotation_task_id: assignment for assignment in manifest.coordinator_tasks
    }
    tasks = _task_map(manifest)
    indexed: dict[str, dict[str, AnswerabilityAnnotationV2]] = {}
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


def _normalized_sets(record: AnswerabilityAnnotationV2) -> tuple[tuple[str, ...], ...]:
    return tuple(
        sorted(
            tuple(sorted(evidence_set)) for evidence_set in record.minimal_sufficient_evidence_sets
        )
    )


def _reasons(
    left: AnswerabilityAnnotationV2, right: AnswerabilityAnnotationV2
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
    assignment: CoordinatorTaskV2,
    by_annotator: Mapping[str, AnswerabilityAnnotationV2],
) -> DisagreementCaseV2 | None:
    left, right = sorted(by_annotator.values(), key=lambda record: record.annotator_pseudonym)
    reasons = _reasons(left, right)
    if not reasons:
        return None
    return DisagreementCaseV2(
        annotation_task_id=assignment.annotation_task_id,
        challenge_id=left.challenge_id,
        reasons=reasons,
        left=left,
        right=right,
    )


def build_disagreement_queue(
    manifest: AssignmentManifestV2,
    submissions: Sequence[AnswerabilityAnnotationV2],
) -> tuple[DisagreementCaseV2, ...]:
    indexed = index_submissions(manifest, submissions)
    cases: list[DisagreementCaseV2] = []
    for assignment in manifest.coordinator_tasks:
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


def _agreement_exclusion(records: Sequence[AnswerabilityAnnotationV2]) -> str | None:
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


def _validate_adjudication(case: DisagreementCaseV2, record: AdjudicationV2) -> None:
    if record.annotation_task_id != case.annotation_task_id:
        raise DataError("adjudication task does not match disagreement case")
    expected = {artifact_hash(case.left), artifact_hash(case.right)}
    actual = {artifact_hash(record.left), artifact_hash(record.right)}
    if actual != expected:
        raise DataError("adjudication original annotation hashes do not match submissions")


def build_workflow_result(
    manifest: AssignmentManifestV2,
    submissions: Sequence[AnswerabilityAnnotationV2],
    *,
    adjudications: Sequence[AdjudicationV2],
    amendments: Sequence[AnnotationAmendmentV2] = (),
    phase: Literal["pilot", "confirmatory"],
    protocol_version: str,
    protocol_hash: str,
    generated_at: str,
) -> WorkflowResult:
    effective_submissions = tuple(submissions)
    if amendments:
        from rag_evidence.annotation.coordinator import resolve_amendments

        effective_submissions = resolve_amendments(manifest, submissions, amendments)
    indexed = index_submissions(manifest, effective_submissions)
    adjudication_by_task: dict[str, AdjudicationV2] = {}
    for record in adjudications:
        if record.annotation_task_id in adjudication_by_task:
            raise DataError(f"duplicate adjudication for {record.annotation_task_id}")
        adjudication_by_task[record.annotation_task_id] = record

    completed = disagreed = adjudicated = excluded = eligible = 0
    disagreements: list[DisagreementCaseV2] = []
    eligibility_records: list[EligibilityRecordV2] = []
    consumed_adjudications: set[str] = set()

    for assignment in manifest.coordinator_tasks:
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
            EligibilityRecordV2(
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
    artifact = EligibilityArtifactV2(
        phase=phase,
        protocol_version=protocol_version,
        protocol_hash=protocol_hash,
        generated_at=generated_at,
        records=tuple(eligibility_records),
    )
    flow = FlowAccounting(
        assigned=len(manifest.coordinator_tasks),
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
        manifest: AssignmentManifestV2,
        submissions: Sequence[AnswerabilityAnnotationV2],
    ) -> None:
        self.path = root / "adjudications" / "records.jsonl"
        self.cases = {
            case.annotation_task_id: case
            for case in build_disagreement_queue(manifest, submissions)
        }

    def records(self) -> tuple[AdjudicationV2, ...]:
        if not self.path.exists():
            return ()
        return tuple(AdjudicationV2.model_validate(payload) for payload in read_records(self.path))

    def status(self) -> dict[str, int | bool]:
        adjudicated = len(self.records())
        total = len(self.cases)
        return {
            "total": total,
            "adjudicated": adjudicated,
            "remaining": total - adjudicated,
            "complete": adjudicated == total,
        }

    def export_jsonl(self) -> str:
        payload = [record.model_dump(mode="json") for record in self.records()]
        violations = scan_delivery_payload(payload, artifact_kind="adjudication_stream")
        if violations:
            raise ArtifactError("adjudication privacy violation: " + "; ".join(violations))
        if not payload:
            return ""
        return (
            "\n".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                for record in payload
            )
            + "\n"
        )

    def submit(self, payload: Mapping[str, Any]) -> AdjudicationV2:
        violations = scan_delivery_payload(payload, artifact_kind="adjudication_record")
        if violations:
            raise ArtifactError("adjudication privacy violation: " + "; ".join(violations))
        record = AdjudicationV2.model_validate(payload)
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
