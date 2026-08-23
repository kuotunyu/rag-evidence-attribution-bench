"""Local annotation state is resumable while submitted artifacts stay append-only."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_evidence.annotation.assignment import AssignmentPackage, build_dual_assignments
from rag_evidence.annotation.blinding import project_challenge
from rag_evidence.annotation.models import (
    AnnotationAmendment,
    AnswerabilityAnnotation,
    artifact_hash,
)
from rag_evidence.annotation.privacy import scan_private_payload
from rag_evidence.annotation.store import AnnotationStore
from rag_evidence.errors import ArtifactError
from test_annotation_blinding import challenge_record


def _package() -> AssignmentPackage:
    tasks = [
        project_challenge(
            challenge_record(challenge_id=f"ch-{index:024x}", parent_id=f"parent-{index}"),
            instruction_version="pilot-v0.2-draft",
            instruction_hash="1" * 64,
            batch="pilot-batch-01",
            namespace="pilot-v0.2",
        )
        for index in range(1, 4)
    ]
    manifest = build_dual_assignments(tasks, ("ann-r7", "ann-k2"), seed=5)
    return next(p for p in manifest.packages if p.annotator_pseudonym == "ann-r7")


def _submission(package: AssignmentPackage, task_index: int = 0, **updates: object):
    task = package.tasks[task_index]
    payload: dict[str, object] = {
        "schema_version": "answerability-annotation-v1",
        "annotation_task_id": task.annotation_task_id,
        "challenge_id": task.challenge_id,
        "blinded_parent_group": task.blinded_parent_group,
        "annotator_pseudonym": package.annotator_pseudonym,
        "instruction_version": task.instruction_version,
        "instruction_hash": task.instruction_hash,
        "task_content_hash": task.task_content_hash,
        "assignment_batch": task.assignment_batch,
        "answerability": "answerable",
        "answer_text": "Riverton",
        "minimal_sufficient_evidence_sets": [["P1.S1", "P2.S1"]],
        "evidence_exhaustive": True,
        "ambiguity": False,
        "dataset_defect": False,
        "confidence": 4,
        "rationale": "The two visible sentences form the answer chain.",
        "started_at": "2026-08-23T01:00:00Z",
        "submitted_at": "2026-08-23T01:05:00Z",
    }
    payload.update(updates)
    return payload


def test_draft_autosave_replaces_only_draft_and_resumes_after_restart(tmp_path: Path) -> None:
    package = _package()
    task = package.tasks[0]
    store = AnnotationStore(tmp_path, package)

    first = {"answerability": "unclear", "confidence": 2}
    second = {"answerability": "answerable", "confidence": 4, "answer_text": "Riverton"}
    store.save_draft(task.annotation_task_id, first)
    store.save_draft(task.annotation_task_id, second)

    reopened = AnnotationStore(tmp_path, package)
    assert reopened.load_draft(task.annotation_task_id)["payload"] == second
    assert reopened.submissions() == ()


def test_submission_is_immutable_and_progress_counts_unique_tasks(tmp_path: Path) -> None:
    package = _package()
    store = AnnotationStore(tmp_path, package)
    submitted = store.submit(_submission(package))

    assert submitted.answer_text == "Riverton"
    assert store.progress() == {"total": 3, "submitted": 1, "remaining": 2}
    with pytest.raises(ArtifactError, match="already submitted"):
        store.submit(_submission(package, answer_text="Changed"))

    reopened = AnnotationStore(tmp_path, package)
    assert reopened.submissions() == (submitted,)
    assert reopened.progress()["submitted"] == 1


def test_submission_must_match_package_task_hashes_and_annotator(tmp_path: Path) -> None:
    package = _package()
    store = AnnotationStore(tmp_path, package)

    with pytest.raises(ArtifactError, match="task content hash"):
        store.submit(_submission(package, task_content_hash="9" * 64))
    with pytest.raises(ArtifactError, match="annotator"):
        store.submit(_submission(package, annotator_pseudonym="ann-k2"))
    with pytest.raises(ArtifactError, match="not assigned"):
        store.submit(_submission(package, annotation_task_id="task-0123456789abcdef01234567"))


def test_amendment_appends_a_hash_chain_without_overwriting_original(tmp_path: Path) -> None:
    package = _package()
    store = AnnotationStore(tmp_path, package)
    original = store.submit(_submission(package))
    replacement = AnswerabilityAnnotation.model_validate(
        _submission(
            package,
            answer_text="Riverton City",
            rationale="Correction: use the full visible city name.",
            submitted_at="2026-08-23T01:07:00Z",
        )
    )
    amendment = AnnotationAmendment.model_validate(
        {
            "schema_version": "annotation-amendment-v1",
            "amendment_id": "amend-0123456789abcdef01234567",
            "original_annotation_hash": artifact_hash(original),
            "previous_amendment_hash": None,
            "annotator_pseudonym": package.annotator_pseudonym,
            "reason": "The answer needed the complete visible name.",
            "replacement": replacement.model_dump(mode="json"),
            "created_at": "2026-08-23T01:08:00Z",
        }
    )
    stored = store.amend(amendment.model_dump(mode="json"))

    assert store.submissions() == (original,)
    assert store.amendments() == (stored,)
    with pytest.raises(ArtifactError, match="previous amendment hash"):
        store.amend(
            {
                **amendment.model_dump(mode="json"),
                "amendment_id": "amend-1123456789abcdef01234567",
                "created_at": "2026-08-23T01:09:00Z",
            }
        )


def test_state_rejects_hidden_source_metadata_and_email_pii(tmp_path: Path) -> None:
    package = _package()
    store = AnnotationStore(tmp_path, package)
    task_id = package.tasks[0].annotation_task_id

    with pytest.raises(ArtifactError, match="forbidden"):
        store.save_draft(task_id, {"transformation": "missing_hop"})
    with pytest.raises(ArtifactError, match="PII"):
        store.save_draft(task_id, {"feedback": "contact person@example.com"})


def test_public_privacy_scanner_rejects_private_posix_and_windows_paths() -> None:
    assert scan_private_payload({"feedback": "/Users/alice/private/state.json"})
    assert scan_private_payload({"feedback": "/home/alice/private/state.json"})
    assert scan_private_payload({"feedback": r"C:\Users\alice\private\state.json"})
    assert scan_private_payload({"feedback": "ordinary annotation rationale"}) == ()
