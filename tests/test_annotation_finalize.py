"""Pilot finalization applies the frozen pre-adjudication gate and fail-closed verdicts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from rag_evidence.annotation.agreement import nominal_agreement
from rag_evidence.annotation.assignment import AssignmentManifest, build_dual_assignments
from rag_evidence.annotation.blinding import project_challenge
from rag_evidence.annotation.finalize import (
    EXPECTED_PILOT_TASKS,
    FINAL_ARTIFACT_NAMES,
    choose_pilot_verdict,
    evaluate_iaa_gate,
    finalize_pilot,
)
from rag_evidence.annotation.models import (
    AdjudicationRecord,
    AnnotationAmendment,
    AnswerabilityAnnotation,
    artifact_hash,
)
from rag_evidence.storage.artifacts import (
    read_json,
    read_records,
    write_json_atomic,
    write_records_atomic,
)
from test_annotation_blinding import challenge_record
from test_annotation_coordinator import _annotation
from test_annotation_workflow import _adjudication

_PROTOCOL_TEXT = """# Synthetic pilot protocol\n\nProtocol ID: pilot-v0.2.1-draft\n"""


def _pilot_manifest(protocol_hash: str) -> AssignmentManifest:
    tasks = [
        project_challenge(
            challenge_record(
                challenge_id=f"ch-{index:024x}",
                parent_id=f"invented-parent-{index}",
            ),
            instruction_version="pilot-v0.2.1-draft",
            instruction_hash=protocol_hash,
            batch="pilot-batch-01",
            namespace="pilot-v0.2-finalize-tests",
        )
        for index in range(1, EXPECTED_PILOT_TASKS + 1)
    ]
    return build_dual_assignments(tasks, ("ann-r7", "ann-k2"), seed=19)


def _streams(
    manifest: AssignmentManifest,
    *,
    flips: int = 4,
    incomplete: bool = False,
    private: bool = False,
) -> tuple[list[AnswerabilityAnnotation], list[AdjudicationRecord]]:
    originals: list[AnswerabilityAnnotation] = []
    adjudications: list[AdjudicationRecord] = []
    for index, assignment in enumerate(manifest.task_assignments):
        left_label = "answerable" if index < 20 else "unanswerable"
        right_label = left_label
        if index < flips // 2 or 20 <= index < 20 + flips - flips // 2:
            right_label = "unanswerable" if left_label == "answerable" else "answerable"
        left = _annotation(
            manifest,
            assignment.annotation_task_id,
            assignment.annotators[0],
            answerability=left_label,
        )
        right = _annotation(
            manifest,
            assignment.annotation_task_id,
            assignment.annotators[1],
            answerability=right_label,
        )
        if private and index == 0:
            left = left.model_copy(
                update={"rationale": "Contact private-reviewer@example.com before review."}
            )
        originals.extend((left, right))
        if left_label != right_label:
            adjudication = _adjudication(left, right).model_copy(
                update={"adjudication_id": f"adj-{index + 1:024x}"}
            )
            adjudications.append(adjudication)
    if incomplete:
        originals.pop()
    return originals, adjudications


def _write_inputs(
    root: Path,
    *,
    flips: int = 4,
    incomplete: bool = False,
    unresolved: bool = False,
    private: bool = False,
    protocol_mismatch: bool = False,
) -> tuple[Path, Path, Path, Path, Path]:
    root.mkdir(parents=True)
    protocol = root / "PILOT_PROTOCOL.md"
    protocol.write_text(_PROTOCOL_TEXT, encoding="utf-8", newline="\n")
    protocol_hash = hashlib.sha256(protocol.read_bytes()).hexdigest()
    manifest = _pilot_manifest(protocol_hash)
    originals, adjudications = _streams(
        manifest,
        flips=flips,
        incomplete=incomplete,
        private=private,
    )
    if unresolved:
        adjudications = []
    manifest_path = root / "manifest.json"
    originals_path = root / "originals.jsonl"
    amendments_path = root / "amendments.jsonl"
    adjudications_path = root / "adjudications.jsonl"
    write_json_atomic(manifest_path, manifest.model_dump(mode="json"))
    write_records_atomic(
        originals_path,
        (record.model_dump(mode="json") for record in originals),
    )
    write_records_atomic(amendments_path, [])
    write_records_atomic(
        adjudications_path,
        (record.model_dump(mode="json") for record in adjudications),
    )
    if protocol_mismatch:
        protocol.write_text(_PROTOCOL_TEXT + "Changed after assignment.\n", encoding="utf-8")
    return manifest_path, originals_path, amendments_path, adjudications_path, protocol


def _finalize(root: Path, out: Path, **kwargs: object):
    paths = _write_inputs(root, **kwargs)
    return finalize_pilot(*paths, out)


def test_iaa_gate_requires_exact_40_defined_finite_coefficients_at_point_70() -> None:
    base = nominal_agreement(
        [("answerable", "answerable")] * 18
        + [("unanswerable", "unanswerable")] * 18
        + [("answerable", "unanswerable")] * 2
        + [("unanswerable", "answerable")] * 2
    )
    at_threshold = replace(base, cohen_kappa=0.70, krippendorff_alpha=0.70)

    assert evaluate_iaa_gate(at_threshold).passed is True
    assert evaluate_iaa_gate(replace(at_threshold, n_total=39, n_complete=39)).passed is False
    assert evaluate_iaa_gate(replace(at_threshold, cohen_kappa=None)).passed is False
    assert evaluate_iaa_gate(replace(at_threshold, krippendorff_alpha=None)).passed is False
    assert evaluate_iaa_gate(replace(at_threshold, cohen_kappa=float("nan"))).passed is False
    assert evaluate_iaa_gate(replace(at_threshold, cohen_kappa=0.69)).passed is False
    assert evaluate_iaa_gate(replace(at_threshold, krippendorff_alpha=0.69)).passed is False


def test_verdict_precedence_is_frozen() -> None:
    assert choose_pilot_verdict(privacy=True, integrity=True).value == "BLOCKED_PRIVACY"
    assert choose_pilot_verdict(integrity=True, incomplete=True).value == "BLOCKED_INTEGRITY"
    assert choose_pilot_verdict(incomplete=True, unresolved=True).value == "BLOCKED_INCOMPLETE"
    assert (
        choose_pilot_verdict(unresolved=True, low_iaa=True).value
        == "BLOCKED_UNRESOLVED_ADJUDICATION"
    )
    assert choose_pilot_verdict(low_iaa=True).value == "BLOCKED_LOW_IAA"
    assert choose_pilot_verdict().value == "READY_FOR_HUMAN_FREEZE_REVIEW"


def test_ready_finalization_writes_fixed_artifacts_and_is_byte_deterministic(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path / "inputs")

    first = finalize_pilot(*inputs, tmp_path / "out-1")
    second = finalize_pilot(*inputs, tmp_path / "out-2")

    assert first.verdict.value == "READY_FOR_HUMAN_FREEZE_REVIEW"
    assert second.verdict == first.verdict
    assert {path.name for path in (tmp_path / "out-1").iterdir()} == set(FINAL_ARTIFACT_NAMES)
    for name in FINAL_ARTIFACT_NAMES:
        assert (tmp_path / "out-1" / name).read_bytes() == (tmp_path / "out-2" / name).read_bytes()
    iaa = json.loads((tmp_path / "out-1" / "iaa.json").read_text(encoding="utf-8"))
    assert iaa["n_total"] == 40
    assert iaa["n_complete"] == 40
    assert iaa["cohen_kappa"] == pytest.approx(0.8)
    assert iaa["gate_passed"] is True
    verdict = json.loads((tmp_path / "out-1" / "pilot-verdict.json").read_text(encoding="utf-8"))
    assert verdict["authorizes_human_pilot"] is False
    assert verdict["authorizes_release"] is False
    input_manifest = json.loads(
        (tmp_path / "out-1" / "input-manifest.json").read_text(encoding="utf-8")
    )
    assert str(tmp_path) not in json.dumps(input_manifest)


def test_finalization_iaa_uses_amendment_tip_before_adjudication(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path / "inputs")
    manifest_path, originals_path, amendments_path, adjudications_path, _protocol = inputs
    manifest = AssignmentManifest.model_validate(read_json(manifest_path))
    originals = [
        AnswerabilityAnnotation.model_validate(payload) for payload in read_records(originals_path)
    ]
    first_assignment = manifest.task_assignments[0]
    first_pair = [
        record
        for record in originals
        if record.annotation_task_id == first_assignment.annotation_task_id
    ]
    left, right = sorted(first_pair, key=lambda record: record.annotator_pseudonym)
    assert left.answerability != right.answerability
    replacement = _annotation(
        manifest,
        right.annotation_task_id,
        right.annotator_pseudonym,
        answerability=left.answerability,
        submitted_at="2026-08-23T01:07:00Z",
    )
    amendment = AnnotationAmendment.model_validate(
        {
            "schema_version": "annotation-amendment-v1",
            "amendment_id": "amend-0123456789abcdef01234567",
            "original_annotation_hash": artifact_hash(right),
            "previous_amendment_hash": None,
            "annotator_pseudonym": right.annotator_pseudonym,
            "reason": "Corrected after rereading the visible passages.",
            "replacement": replacement.model_dump(mode="json"),
            "created_at": "2026-08-23T01:08:00Z",
        }
    )
    write_records_atomic(amendments_path, [amendment.model_dump(mode="json")])
    adjudications = [
        AdjudicationRecord.model_validate(payload) for payload in read_records(adjudications_path)
    ]
    write_records_atomic(
        adjudications_path,
        (
            record.model_dump(mode="json")
            for record in adjudications
            if record.annotation_task_id != first_assignment.annotation_task_id
        ),
    )

    result = finalize_pilot(*inputs, tmp_path / "out")

    assert result.verdict.value == "READY_FOR_HUMAN_FREEZE_REVIEW"
    iaa = json.loads((tmp_path / "out" / "iaa.json").read_text(encoding="utf-8"))
    assert iaa["raw_agreement"] == pytest.approx(37 / 40)
    flow = json.loads((tmp_path / "out" / "flow-accounting.json").read_text(encoding="utf-8"))
    assert flow["disagreed"] == 3


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"incomplete": True}, "BLOCKED_INCOMPLETE"),
        ({"unresolved": True}, "BLOCKED_UNRESOLVED_ADJUDICATION"),
        ({"flips": 12}, "BLOCKED_LOW_IAA"),
        ({"protocol_mismatch": True}, "BLOCKED_INTEGRITY"),
    ],
)
def test_finalization_blocks_each_operational_gate(
    tmp_path: Path,
    kwargs: dict[str, object],
    expected: str,
) -> None:
    result = _finalize(tmp_path / "inputs", tmp_path / "out", **kwargs)

    assert result.verdict.value == expected
    verdict = json.loads((tmp_path / "out" / "pilot-verdict.json").read_text(encoding="utf-8"))
    assert verdict["verdict"] == expected
    assert verdict["opens_confirmatory_sample"] is False
    assert verdict["starts_model_run"] is False


def test_privacy_block_writes_safe_artifacts_without_echoing_pii(tmp_path: Path) -> None:
    result = _finalize(
        tmp_path / "inputs",
        tmp_path / "out",
        private=True,
    )

    assert result.verdict.value == "BLOCKED_PRIVACY"
    combined = b"".join((tmp_path / "out" / name).read_bytes() for name in FINAL_ARTIFACT_NAMES)
    assert b"private-reviewer@example.com" not in combined
    privacy = json.loads((tmp_path / "out" / "privacy-scan.json").read_text(encoding="utf-8"))
    assert privacy["passed"] is False
