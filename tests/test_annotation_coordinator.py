"""Aggregate coordinator ingestion is deterministic and fails closed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_evidence.annotation.assignment import (
    AssignmentManifestV2,
    SchedulableTaskV2,
    build_dual_assignments_v2,
)
from rag_evidence.annotation.blinding import project_challenge_v2
from rag_evidence.annotation.coordinator import (
    collect_annotation_streams,
    resolve_amendments,
)
from rag_evidence.annotation.models import (
    AnnotationAmendmentV2,
    AnswerabilityAnnotationV2,
    artifact_hash,
)
from rag_evidence.annotation.privacy import scan_delivery_payload
from rag_evidence.errors import DataError
from rag_evidence.storage.artifacts import write_json_atomic, write_records_atomic
from test_annotation_blinding import challenge_record


def _manifest(task_count: int = 1) -> AssignmentManifestV2:
    tasks = [
        project_challenge_v2(
            challenge_record(
                challenge_id=f"ch-{index:024x}",
                parent_id=f"synthetic-parent-{index}",
            ),
            instruction_version="pilot-v0.2.2-draft",
            instruction_hash="1" * 64,
            batch="pilot-batch-01",
            namespace="pilot-v0.2-tests",
        )
        for index in range(1, task_count + 1)
    ]
    schedulable = tuple(
        SchedulableTaskV2(
            task=task,
            internal_group_id=f"coord-{index:024x}",
            transformation="missing_hop",
        )
        for index, task in enumerate(tasks, start=1)
    )
    _, manifest = build_dual_assignments_v2(
        schedulable, ("ann-r7", "ann-k2"), seed=11
    )
    return manifest


def _annotation(
    manifest: AssignmentManifestV2,
    task_id: str,
    annotator: str,
    *,
    answerability: str = "answerable",
    answer_text: str = "Riverton",
    rationale: str = "The visible sentences provide the answer.",
    submitted_at: str = "2026-08-23T01:05:00Z",
) -> AnswerabilityAnnotationV2:
    tasks = {task.annotation_task_id: task for task in manifest.tasks}
    task = tasks[task_id]
    is_answerable = answerability == "answerable"
    return AnswerabilityAnnotationV2.model_validate(
        {
            "schema_version": "answerability-annotation-v2",
            "annotation_task_id": task.annotation_task_id,
            "challenge_id": task.challenge_id,
            "annotator_pseudonym": annotator,
            "instruction_version": task.instruction_version,
            "instruction_hash": task.instruction_hash,
            "task_content_hash": task.task_content_hash,
            "assignment_batch": task.assignment_batch,
            "answerability": answerability,
            "answer_text": answer_text if is_answerable else None,
            "minimal_sufficient_evidence_sets": [["P1.S1"]] if is_answerable else [],
            "evidence_exhaustive": True if is_answerable else None,
            "ambiguity": False,
            "dataset_defect": False,
            "confidence": 4,
            "rationale": rationale,
            "started_at": "2026-08-23T01:00:00Z",
            "submitted_at": submitted_at,
        }
    )


def _amendment(
    original: AnswerabilityAnnotationV2,
    replacement: AnswerabilityAnnotationV2,
    *,
    suffix: str,
    previous: str | None,
    created_at: str,
) -> AnnotationAmendmentV2:
    return AnnotationAmendmentV2.model_validate(
        {
            "schema_version": "annotation-amendment-v2",
            "amendment_id": f"amend-{suffix:0>24}",
            "original_annotation_hash": artifact_hash(original),
            "previous_amendment_hash": previous,
            "annotator_pseudonym": original.annotator_pseudonym,
            "reason": "Correct the visible answer transcription.",
            "replacement": replacement.model_dump(mode="json"),
            "created_at": created_at,
        }
    )


def _chain() -> tuple[
    AssignmentManifestV2,
    tuple[AnswerabilityAnnotationV2, AnswerabilityAnnotationV2],
    AnnotationAmendmentV2,
    AnnotationAmendmentV2,
]:
    manifest = _manifest()
    task_id = manifest.coordinator_tasks[0].annotation_task_id
    left = _annotation(manifest, task_id, "ann-k2")
    right = _annotation(manifest, task_id, "ann-r7")
    first_replacement = _annotation(
        manifest,
        task_id,
        "ann-k2",
        answer_text="Riverton City",
        submitted_at="2026-08-23T01:07:00Z",
    )
    first = _amendment(
        left,
        first_replacement,
        suffix="1",
        previous=None,
        created_at="2026-08-23T01:08:00Z",
    )
    second_replacement = _annotation(
        manifest,
        task_id,
        "ann-k2",
        answer_text="Riverton County",
        submitted_at="2026-08-23T01:09:00Z",
    )
    second = _amendment(
        left,
        second_replacement,
        suffix="2",
        previous=artifact_hash(first),
        created_at="2026-08-23T01:10:00Z",
    )
    return manifest, (left, right), first, second


def test_resolve_amendments_is_order_independent_and_uses_chain_tip() -> None:
    manifest, originals, first, second = _chain()

    effective = resolve_amendments(manifest, originals, [second, first])

    by_annotator = {record.annotator_pseudonym: record for record in effective}
    assert by_annotator["ann-k2"] == second.replacement
    assert by_annotator["ann-r7"] == originals[1]


def test_resolve_amendments_rejects_broken_predecessor() -> None:
    manifest, originals, first, _second = _chain()
    broken = first.model_copy(update={"previous_amendment_hash": "f" * 64})

    with pytest.raises(DataError, match="broken predecessor"):
        resolve_amendments(manifest, originals, [broken])


def test_resolve_amendments_rejects_forked_chain() -> None:
    manifest, originals, first, second = _chain()
    fork_replacement = second.replacement.model_copy(
        update={"answer_text": "Riverton Township", "rationale": "Alternative correction."}
    )
    fork = _amendment(
        originals[0],
        fork_replacement,
        suffix="3",
        previous=artifact_hash(first),
        created_at="2026-08-23T01:11:00Z",
    )

    with pytest.raises(DataError, match="fork"):
        resolve_amendments(manifest, originals, [first, second, fork])


def test_resolve_amendments_rejects_cycle_defensively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rag_evidence.annotation.coordinator as coordinator

    manifest, originals, first, second = _chain()
    cyclic_first = first.model_copy(update={"previous_amendment_hash": "b" * 64})
    cyclic_second = second.model_copy(update={"previous_amendment_hash": "a" * 64})

    def fake_hash(record: object) -> str:
        amendment_id = getattr(record, "amendment_id", None)
        if amendment_id == cyclic_first.amendment_id:
            return "a" * 64
        if amendment_id == cyclic_second.amendment_id:
            return "b" * 64
        return artifact_hash(record)  # type: ignore[arg-type]

    monkeypatch.setattr(coordinator, "artifact_hash", fake_hash)
    with pytest.raises(DataError, match=r"cycle|head"):
        resolve_amendments(manifest, originals, [cyclic_first, cyclic_second])


def test_resolve_amendments_rejects_duplicate_ids() -> None:
    manifest, originals, first, _second = _chain()

    with pytest.raises(DataError, match="duplicate amendment ID"):
        resolve_amendments(manifest, originals, [first, first])


def test_resolve_amendments_rejects_replacement_binding_change() -> None:
    manifest = _manifest(task_count=2)
    first_task, second_task = (
        assignment.annotation_task_id for assignment in manifest.coordinator_tasks
    )
    original = _annotation(manifest, first_task, "ann-k2")
    other_task_replacement = _annotation(
        manifest,
        second_task,
        "ann-k2",
        answer_text="Different task",
    )
    amendment = _amendment(
        original,
        other_task_replacement,
        suffix="4",
        previous=None,
        created_at="2026-08-23T01:12:00Z",
    )

    with pytest.raises(DataError, match="changes the original annotation binding"):
        resolve_amendments(manifest, [original], [amendment])


def test_resolve_amendments_rejects_cross_original_predecessor() -> None:
    manifest = _manifest(task_count=2)
    first_task, second_task = (
        assignment.annotation_task_id for assignment in manifest.coordinator_tasks
    )
    first_original = _annotation(manifest, first_task, "ann-k2")
    second_original = _annotation(manifest, second_task, "ann-k2")
    first_replacement = _annotation(
        manifest,
        first_task,
        "ann-k2",
        answer_text="First corrected answer",
    )
    second_replacement = _annotation(
        manifest,
        second_task,
        "ann-k2",
        answer_text="Second corrected answer",
    )
    first = _amendment(
        first_original,
        first_replacement,
        suffix="5",
        previous=None,
        created_at="2026-08-23T01:13:00Z",
    )
    cross = _amendment(
        second_original,
        second_replacement,
        suffix="6",
        previous=artifact_hash(first),
        created_at="2026-08-23T01:14:00Z",
    )

    with pytest.raises(DataError, match="different original chain"):
        resolve_amendments(
            manifest,
            [first_original, second_original],
            [first, cross],
        )


def _write_streams(
    root: Path,
    manifest: AssignmentManifestV2,
    *,
    complete: bool,
) -> tuple[Path, list[Path], list[Path]]:
    manifest_path = root / "manifest.json"
    write_json_atomic(manifest_path, manifest.model_dump(mode="json"))
    task_ids = [assignment.annotation_task_id for assignment in manifest.coordinator_tasks]
    submissions: list[Path] = []
    amendments: list[Path] = []
    for position, annotator in enumerate(("ann-k2", "ann-r7"), start=1):
        submission_path = root / f"submission-{position}.jsonl"
        amendment_path = root / f"amendment-{position}.jsonl"
        selected = task_ids if complete else task_ids[:1]
        write_records_atomic(
            submission_path,
            (
                _annotation(manifest, task_id, annotator).model_dump(mode="json")
                for task_id in selected
            ),
        )
        write_records_atomic(amendment_path, [])
        submissions.append(submission_path)
        amendments.append(amendment_path)
    return manifest_path, submissions, amendments


def test_collect_requires_exactly_two_submission_and_amendment_files(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest_path, submissions, amendments = _write_streams(tmp_path, manifest, complete=True)

    with pytest.raises(DataError, match="exactly two submission"):
        collect_annotation_streams(
            manifest_path,
            submissions[:1],
            amendments,
            tmp_path / "out-submissions",
        )
    with pytest.raises(DataError, match="exactly two amendment"):
        collect_annotation_streams(
            manifest_path,
            submissions,
            amendments[:1],
            tmp_path / "out-amendments",
        )


def test_incomplete_collection_writes_receipt_but_no_disagreement_queue(
    tmp_path: Path,
) -> None:
    manifest = _manifest(task_count=2)
    manifest_path, submissions, amendments = _write_streams(tmp_path, manifest, complete=False)
    out = tmp_path / "out"

    result = collect_annotation_streams(manifest_path, submissions, amendments, out)

    assert result.complete is False
    assert result.completed_tasks == 1
    assert (out / "collection-receipt.json").exists()
    assert (out / "effective-submissions.jsonl").exists()
    assert not (out / "disagreements.jsonl").exists()
    payload = json.loads((out / "input-manifest.json").read_text(encoding="utf-8"))
    assert str(tmp_path) not in json.dumps(payload)


def test_complete_collection_accepts_present_empty_amendment_streams(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest_path, submissions, amendments = _write_streams(tmp_path, manifest, complete=True)
    out = tmp_path / "out"

    result = collect_annotation_streams(manifest_path, submissions, amendments, out)

    assert result.complete is True
    assert (out / "disagreements.jsonl").read_text(encoding="utf-8") == ""
    assert (out / "amendments.jsonl").read_text(encoding="utf-8") == ""
    receipt = json.loads((out / "collection-receipt.json").read_text(encoding="utf-8"))
    input_manifest = json.loads((out / "input-manifest.json").read_text(encoding="utf-8"))
    assert receipt["schema_version"] == "annotation-collection-receipt-v2"
    assert input_manifest["schema_version"] == "annotation-input-manifest-v2"
    assert scan_delivery_payload(receipt, artifact_kind="collection_receipt") == ()


def test_collection_rejects_amendment_stream_paired_to_other_pseudonym(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    manifest_path, submissions, amendments = _write_streams(tmp_path, manifest, complete=True)
    task_id = manifest.coordinator_tasks[0].annotation_task_id
    original = _annotation(manifest, task_id, "ann-r7")
    replacement = _annotation(
        manifest,
        task_id,
        "ann-r7",
        answer_text="Riverton City",
        submitted_at="2026-08-23T01:07:00Z",
    )
    wrong_stream = _amendment(
        original,
        replacement,
        suffix="7",
        previous=None,
        created_at="2026-08-23T01:08:00Z",
    )
    write_records_atomic(amendments[0], [wrong_stream.model_dump(mode="json")])

    with pytest.raises(DataError, match="not paired"):
        collect_annotation_streams(
            manifest_path,
            submissions,
            amendments,
            tmp_path / "out",
        )
