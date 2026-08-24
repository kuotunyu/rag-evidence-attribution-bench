"""Dual-review completeness and adjudication fail closed without expected labels."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_evidence.annotation.assignment import (
    AssignmentManifestV2,
    SchedulableTaskV2,
    build_dual_assignments_v2,
)
from rag_evidence.annotation.blinding import project_challenge_v2
from rag_evidence.annotation.models import (
    AdjudicationV2,
    AnnotationAmendmentV2,
    AnswerabilityAnnotationV2,
    artifact_hash,
)
from rag_evidence.annotation.workflow import (
    AdjudicationStore,
    build_disagreement_queue,
    build_workflow_result,
)
from rag_evidence.errors import ArtifactError, DataError
from test_annotation_blinding import challenge_record


def _manifest() -> AssignmentManifestV2:
    task = project_challenge_v2(
        challenge_record(),
        instruction_version="pilot-v0.2.2-draft",
        instruction_hash="1" * 64,
        batch="pilot-batch-01",
        namespace="pilot-v0.2",
    )
    _, manifest = build_dual_assignments_v2(
        (
            SchedulableTaskV2(
                task=task,
                internal_group_id="coord-000000000000000000000001",
                transformation="missing_hop",
            ),
        ),
        ("ann-r7", "ann-k2"),
        seed=11,
    )
    return manifest


def _annotation(
    manifest: AssignmentManifestV2,
    annotator: str,
    *,
    answerability: str = "answerable",
    evidence_sets: list[list[str]] | None = None,
    ambiguity: bool = False,
    dataset_defect: bool = False,
    exhaustive: bool | None = True,
    rationale: str = "The two visible sentences form the answer chain.",
    submitted_at: str = "2026-08-23T01:05:00Z",
) -> AnswerabilityAnnotationV2:
    task = manifest.tasks[0]
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
            "answer_text": "Riverton" if is_answerable else None,
            "minimal_sufficient_evidence_sets": (evidence_sets if is_answerable else [])
            or ([[]] if is_answerable else []),
            "evidence_exhaustive": exhaustive if is_answerable else None,
            "ambiguity": ambiguity,
            "dataset_defect": dataset_defect,
            "confidence": 4,
            "rationale": rationale,
            "started_at": "2026-08-23T01:00:00Z",
            "submitted_at": submitted_at,
        }
    )


def _answerable(
    manifest: AssignmentManifestV2,
    annotator: str,
    **kwargs: object,
) -> AnswerabilityAnnotationV2:
    defaults: dict[str, object] = {"evidence_sets": [["P1.S1", "P2.S1"]]}
    defaults.update(kwargs)
    return _annotation(manifest, annotator, **defaults)  # type: ignore[arg-type]


def _adjudication(
    left: AnswerabilityAnnotationV2,
    right: AnswerabilityAnnotationV2,
    *,
    excluded: bool = False,
) -> AdjudicationV2:
    return AdjudicationV2.model_validate(
        {
            "schema_version": "adjudication-v2",
            "adjudication_id": "adj-0123456789abcdef01234567",
            "annotation_task_id": left.annotation_task_id,
            "challenge_id": left.challenge_id,
            "adjudicator_pseudonym": "ann-j9",
            "left": left.model_dump(mode="json"),
            "right": right.model_dump(mode="json"),
            "final_answerability": "answerable" if not excluded else "unclear",
            "answer_text": "Riverton" if not excluded else None,
            "minimal_sufficient_evidence_sets": ([["P1.S1", "P2.S1"]] if not excluded else []),
            "evidence_exhaustive": True if not excluded else None,
            "excluded": excluded,
            "exclusion_reason": "unresolved ambiguity" if excluded else None,
            "rationale": "Independent review of both visible rationales.",
            "created_at": "2026-08-23T02:00:00Z",
        }
    )


def test_completeness_gate_refuses_missing_or_duplicate_human_judgments() -> None:
    manifest = _manifest()
    left = _answerable(manifest, "ann-r7")

    with pytest.raises(DataError, match="two completed"):
        build_disagreement_queue(manifest, [left])
    with pytest.raises(DataError, match="duplicate"):
        build_disagreement_queue(manifest, [left, left])


def test_disagreement_queue_preserves_both_original_decisions_and_reasons() -> None:
    manifest = _manifest()
    left = _answerable(manifest, "ann-r7", rationale="Visible chain is sufficient.")
    right = _annotation(
        manifest,
        "ann-k2",
        answerability="unclear",
        exhaustive=None,
        rationale="The second hop may refer to another person.",
    )

    queue = build_disagreement_queue(manifest, [left, right])

    assert len(queue) == 1
    assert queue[0].reasons == ("answerability", "answer_text", "evidence_sets")
    assert queue[0].left.rationale == "The second hop may refer to another person."
    assert queue[0].right.rationale == "Visible chain is sufficient."
    assert {queue[0].left.annotator_pseudonym, queue[0].right.annotator_pseudonym} == {
        "ann-r7",
        "ann-k2",
    }


def test_exact_agreement_becomes_eligible_without_adjudication() -> None:
    manifest = _manifest()
    submissions = [_answerable(manifest, "ann-r7"), _answerable(manifest, "ann-k2")]

    result = build_workflow_result(
        manifest,
        submissions,
        adjudications=[],
        phase="confirmatory",
        protocol_version="v2-confirmatory-frozen",
        protocol_hash="8" * 64,
        generated_at="2026-08-23T03:00:00Z",
    )

    assert result.flow.model_dump() == {
        "schema_version": "pilot-flow-accounting-v2",
        "assigned": 1,
        "completed": 1,
        "disagreed": 0,
        "adjudicated": 0,
        "excluded": 0,
        "eligible": 1,
    }
    record = result.eligibility.records[0]
    assert record.adjudication_hash is None
    assert record.eligible is True


def test_disagreement_stays_ineligible_until_independent_adjudication() -> None:
    manifest = _manifest()
    left = _answerable(manifest, "ann-r7")
    right = _annotation(manifest, "ann-k2", answerability="unanswerable", exhaustive=None)

    pending = build_workflow_result(
        manifest,
        [left, right],
        adjudications=[],
        phase="confirmatory",
        protocol_version="v2-confirmatory-frozen",
        protocol_hash="8" * 64,
        generated_at="2026-08-23T03:00:00Z",
    )
    assert pending.flow.disagreed == 1
    assert pending.flow.adjudicated == 0
    assert pending.flow.eligible == 0
    assert pending.eligibility.records == ()

    adjudication = _adjudication(left, right)
    resolved = build_workflow_result(
        manifest,
        [left, right],
        adjudications=[adjudication],
        phase="confirmatory",
        protocol_version="v2-confirmatory-frozen",
        protocol_hash="8" * 64,
        generated_at="2026-08-23T03:00:00Z",
    )
    assert resolved.flow.adjudicated == 1
    assert resolved.flow.eligible == 1
    assert resolved.eligibility.records[0].adjudication_hash == artifact_hash(adjudication)


def test_workflow_rejects_adjudication_that_does_not_preserve_originals() -> None:
    manifest = _manifest()
    left = _answerable(manifest, "ann-r7")
    right = _annotation(manifest, "ann-k2", answerability="unanswerable", exhaustive=None)
    adjudication = _adjudication(left, right)
    different_left = _answerable(
        manifest,
        "ann-r7",
        rationale="This is a different immutable original.",
    )

    with pytest.raises(DataError, match="original annotation hashes"):
        build_workflow_result(
            manifest,
            [different_left, right],
            adjudications=[adjudication],
            phase="confirmatory",
            protocol_version="v2-confirmatory-frozen",
            protocol_hash="8" * 64,
            generated_at="2026-08-23T03:00:00Z",
        )


def test_unclear_and_dataset_defects_are_excluded_not_coerced() -> None:
    manifest = _manifest()
    left = _annotation(
        manifest,
        "ann-r7",
        answerability="unclear",
        exhaustive=None,
        dataset_defect=True,
    )
    right = _annotation(
        manifest,
        "ann-k2",
        answerability="unclear",
        exhaustive=None,
        dataset_defect=True,
    )
    result = build_workflow_result(
        manifest,
        [left, right],
        adjudications=[],
        phase="confirmatory",
        protocol_version="v2-confirmatory-frozen",
        protocol_hash="8" * 64,
        generated_at="2026-08-23T03:00:00Z",
    )

    assert result.flow.excluded == 1
    assert result.flow.eligible == 0
    assert result.eligibility.records[0].final_answerability == "unclear"
    assert "unclear" in (result.eligibility.records[0].exclusion_reason or "")


def test_workflow_uses_effective_amendment_tip() -> None:
    manifest = _manifest()
    left = _answerable(manifest, "ann-r7")
    right = _annotation(manifest, "ann-k2", answerability="unanswerable", exhaustive=None)
    replacement = _answerable(
        manifest,
        "ann-k2",
        submitted_at="2026-08-23T01:07:00Z",
        rationale="Correction after rereading the visible evidence.",
    )
    amendment = AnnotationAmendmentV2.model_validate(
        {
            "schema_version": "annotation-amendment-v2",
            "amendment_id": "amend-0123456789abcdef01234567",
            "original_annotation_hash": artifact_hash(right),
            "previous_amendment_hash": None,
            "annotator_pseudonym": "ann-k2",
            "reason": "The visible chain does answer the question.",
            "replacement": replacement.model_dump(mode="json"),
            "created_at": "2026-08-23T01:08:00Z",
        }
    )

    result = build_workflow_result(
        manifest,
        [left, right],
        amendments=[amendment],
        adjudications=[],
        phase="pilot",
        protocol_version="pilot-v0.2.2-draft",
        protocol_hash="8" * 64,
        generated_at="2026-08-23T03:00:00Z",
    )

    assert result.flow.disagreed == 0
    assert result.flow.eligible == 1
    expected = tuple(
        artifact_hash(record)
        for record in sorted([replacement, left], key=lambda item: item.annotator_pseudonym)
    )
    assert result.eligibility.records[0].source_annotation_hashes == expected


def test_adjudication_store_is_append_only_and_queue_bound(tmp_path: Path) -> None:
    manifest = _manifest()
    left = _answerable(manifest, "ann-r7")
    right = _annotation(manifest, "ann-k2", answerability="unanswerable", exhaustive=None)
    store = AdjudicationStore(tmp_path, manifest, [left, right])
    adjudication = _adjudication(left, right)

    stored = store.submit(adjudication.model_dump(mode="json"))

    assert store.records() == (stored,)
    with pytest.raises(ArtifactError, match="already adjudicated"):
        store.submit(adjudication.model_dump(mode="json"))
