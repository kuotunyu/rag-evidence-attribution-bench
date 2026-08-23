"""Local autosave plus append-only annotation artifacts."""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from rag_evidence.annotation.assignment import AssignmentPackage
from rag_evidence.annotation.models import (
    AnnotationAmendment,
    AnswerabilityAnnotation,
    BlindTask,
    artifact_hash,
)
from rag_evidence.errors import ArtifactError
from rag_evidence.storage.artifacts import (
    append_record,
    read_json,
    read_records,
    write_json_atomic,
)

_PRIVATE_KEYS = frozenset(
    {
        "transformation",
        "transform_version",
        "expected_answerability",
        "changed_fields",
        "provenance",
        "parent_question_id",
        "parent_fingerprint",
        "gold_adjudicated_label",
        "gold_passage_ids",
        "supporting_fact_sentence_ids",
        "is_gold",
        "model_name",
        "method_name",
        "score",
        "scores",
        "abstention_expectation",
    }
)
_EMAIL_RE = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_WINDOWS_PATH_RE = re.compile(r"(?:^|\s)[A-Za-z]:[\\/]")


def _privacy_violations(payload: object) -> tuple[str, ...]:
    violations: list[str] = []

    def visit(value: object, path: str) -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key).casefold()
                if key in _PRIVATE_KEYS or key.startswith(("expected_", "gold_")):
                    violations.append(f"{path}.{raw_key}: forbidden source metadata")
                visit(child, f"{path}.{raw_key}")
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
            return
        if isinstance(value, str):
            if _EMAIL_RE.search(value):
                violations.append(f"{path}: PII-like email")
            if _WINDOWS_PATH_RE.search(value):
                violations.append(f"{path}: private filesystem path")

    visit(payload, "$")
    return tuple(sorted(set(violations)))


class AnnotationStore:
    """Package-bound state. Drafts are replaceable; scientific records are append-only."""

    def __init__(self, root: Path, package: AssignmentPackage) -> None:
        self.root = root
        self.package = package
        self._tasks = {task.annotation_task_id: task for task in package.tasks}
        self._submissions_path = root / "submissions" / "records.jsonl"
        self._amendments_path = root / "amendments" / "records.jsonl"

    def _task(self, task_id: str) -> BlindTask:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise ArtifactError(f"task {task_id} is not assigned in this package") from exc

    def _scan(self, payload: object) -> None:
        violations = _privacy_violations(payload)
        if violations:
            raise ArtifactError("annotation privacy violation: " + "; ".join(violations))

    def _validate_binding(self, record: AnswerabilityAnnotation) -> None:
        task = self._task(record.annotation_task_id)
        expected = {
            "challenge_id": task.challenge_id,
            "blinded_parent_group": task.blinded_parent_group,
            "instruction_version": task.instruction_version,
            "instruction_hash": task.instruction_hash,
            "task_content_hash": task.task_content_hash,
            "assignment_batch": task.assignment_batch,
        }
        for field, value in expected.items():
            if getattr(record, field) != value:
                label = field.replace("_", " ")
                raise ArtifactError(f"annotation {label} does not match assigned task")
        if record.annotator_pseudonym != self.package.annotator_pseudonym:
            raise ArtifactError("annotation annotator does not match the package pseudonym")

    def _draft_path(self, task_id: str) -> Path:
        task = self._task(task_id)
        return self.root / "drafts" / f"{task.annotation_task_id}.json"

    def save_draft(self, task_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        task = self._task(task_id)
        self._scan(payload)
        record = {
            "schema_version": "annotation-draft-v1",
            "annotation_task_id": task.annotation_task_id,
            "annotator_pseudonym": self.package.annotator_pseudonym,
            "instruction_hash": task.instruction_hash,
            "task_content_hash": task.task_content_hash,
            "payload": dict(payload),
            "updated_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        write_json_atomic(self._draft_path(task_id), record)
        return record

    def load_draft(self, task_id: str) -> dict[str, Any] | None:
        path = self._draft_path(task_id)
        if not path.exists():
            return None
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise ArtifactError(f"draft {path} is not a JSON object")
        return dict(payload)

    def submissions(self) -> tuple[AnswerabilityAnnotation, ...]:
        if not self._submissions_path.exists():
            return ()
        return tuple(
            AnswerabilityAnnotation.model_validate(record)
            for record in read_records(self._submissions_path)
        )

    def submit(self, payload: Mapping[str, Any]) -> AnswerabilityAnnotation:
        self._scan(payload)
        task_id = payload.get("annotation_task_id")
        if not isinstance(task_id, str):
            record = AnswerabilityAnnotation.model_validate(payload)
            raise AssertionError(f"validated annotation unexpectedly lacks task ID: {record}")
        self._task(task_id)
        record = AnswerabilityAnnotation.model_validate(payload)
        self._validate_binding(record)
        if any(
            existing.annotation_task_id == record.annotation_task_id
            and existing.annotator_pseudonym == record.annotator_pseudonym
            for existing in self.submissions()
        ):
            raise ArtifactError(
                f"task {record.annotation_task_id} is already submitted and immutable"
            )
        append_record(self._submissions_path, record.model_dump(mode="json"))
        return record

    def amendments(self) -> tuple[AnnotationAmendment, ...]:
        if not self._amendments_path.exists():
            return ()
        return tuple(
            AnnotationAmendment.model_validate(record)
            for record in read_records(self._amendments_path)
        )

    def amend(self, payload: Mapping[str, Any]) -> AnnotationAmendment:
        self._scan(payload)
        amendment = AnnotationAmendment.model_validate(payload)
        if not isinstance(amendment.replacement, AnswerabilityAnnotation):
            raise ArtifactError("this answerability package cannot store citation amendments")
        self._validate_binding(amendment.replacement)
        originals = {artifact_hash(submission): submission for submission in self.submissions()}
        original = originals.get(amendment.original_annotation_hash)
        if original is None:
            raise ArtifactError("amendment original annotation hash is not stored")
        if original.annotation_task_id != amendment.replacement.annotation_task_id:
            raise ArtifactError("amendment replacement changes the original task")
        chain = [
            existing
            for existing in self.amendments()
            if existing.original_annotation_hash == amendment.original_annotation_hash
        ]
        expected_previous = artifact_hash(chain[-1]) if chain else None
        if amendment.previous_amendment_hash != expected_previous:
            raise ArtifactError(
                "amendment previous amendment hash does not match the append-only chain"
            )
        append_record(self._amendments_path, amendment.model_dump(mode="json"))
        return amendment

    def progress(self) -> dict[str, int]:
        submitted_ids = {record.annotation_task_id for record in self.submissions()}
        total = len(self._tasks)
        submitted = len(submitted_ids)
        return {"total": total, "submitted": submitted, "remaining": total - submitted}

    def export_records(self, kind: Literal["submissions", "amendments"]) -> list[dict[str, Any]]:
        records: Sequence[AnswerabilityAnnotation | AnnotationAmendment]
        records = self.submissions() if kind == "submissions" else self.amendments()
        payload = [record.model_dump(mode="json") for record in records]
        self._scan(payload)
        return payload
