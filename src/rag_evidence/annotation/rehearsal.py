"""Explicitly synthetic, Git-external rehearsal of the complete pilot operations path."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from rag_evidence.annotation.app import create_adjudication_app
from rag_evidence.annotation.assignment import AssignmentManifest, build_dual_assignments
from rag_evidence.annotation.coordinator import collect_annotation_streams
from rag_evidence.annotation.finalize import PilotVerdictName, finalize_pilot
from rag_evidence.annotation.models import (
    AdjudicationRecord,
    AnnotationAmendment,
    AnswerabilityAnnotation,
    BlindTask,
    TaskPassage,
    TaskSentence,
    artifact_hash,
)
from rag_evidence.errors import DataError
from rag_evidence.storage.artifacts import (
    read_json,
    read_records,
    write_json_atomic,
    write_records_atomic,
)

_MARKER = (
    "SYNTHETIC REHEARSAL ONLY. These invented records are not human annotations, "
    "pilot results, or confirmatory evidence.\n"
)
_INSTRUCTION_VERSION = "pilot-v0.2.1-draft"
_ANNOTATORS = ("ann-synth-a", "ann-synth-b")
_ADJUDICATOR = "ann-synth-c"


@dataclass(frozen=True)
class RehearsalResult:
    tasks: int
    amendments: int
    disagreements: int
    adjudications: int
    dataset_defect_exclusions: int
    repeat_byte_identical: bool
    verdict: str
    output_dir: Path


@dataclass(frozen=True)
class _RunResult:
    tasks: int
    amendments: int
    disagreements: int
    adjudications: int
    dataset_defect_exclusions: int
    verdict: PilotVerdictName


def _canonical_hash(payload: object) -> str:
    material = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _opaque_id(prefix: str, *parts: object) -> str:
    return f"{prefix}-{_canonical_hash(parts)[:24]}"


def _build_tasks(instruction_hash: str) -> tuple[tuple[BlindTask, ...], dict[str, str]]:
    tasks: list[BlindTask] = []
    base_labels: dict[str, str] = {}
    for task_index in range(40):
        parent_index = task_index // 2
        card_index = task_index % 2
        answerable = task_index < 20
        challenge_id = _opaque_id("ch", "synthetic-rehearsal", task_index)
        group_id = _opaque_id("bg", "synthetic-rehearsal", parent_index)
        question = (
            f"In the invented Lumen archive parent {parent_index}, card {card_index}, "
            "what codeword is recorded?"
        )
        if answerable:
            first_text = "The invented archive card records the codeword Lumen."
            second_text = "A duplicate synthetic ledger also records the codeword Lumen."
        else:
            first_text = "The invented archive card records a blank codeword field."
            second_text = "The duplicate synthetic ledger says the codeword was omitted."
        passages = (
            TaskPassage(
                alias="P1",
                title=f"Invented archive card {task_index}",
                sentences=(TaskSentence(alias="P1.S1", text=first_text),),
            ),
            TaskPassage(
                alias="P2",
                title=f"Synthetic duplicate ledger {task_index}",
                sentences=(TaskSentence(alias="P2.S1", text=second_text),),
            ),
        )
        content = {
            "challenge_id": challenge_id,
            "blinded_parent_group": group_id,
            "instruction_version": _INSTRUCTION_VERSION,
            "instruction_hash": instruction_hash,
            "question": question,
            "passages": [passage.model_dump(mode="json") for passage in passages],
        }
        task = BlindTask(
            annotation_task_id=_opaque_id("task", "synthetic-rehearsal", task_index),
            challenge_id=challenge_id,
            blinded_parent_group=group_id,
            instruction_version=_INSTRUCTION_VERSION,
            instruction_hash=instruction_hash,
            question=question,
            passages=passages,
            task_content_hash=_canonical_hash(content),
            assignment_batch="synthetic-rehearsal-v1",
        )
        tasks.append(task)
        base_labels[task.annotation_task_id] = "answerable" if answerable else "unanswerable"
    return tuple(tasks), base_labels


def _timestamp(minute: int) -> str:
    value = dt.datetime(2026, 8, 23, 1, 0, tzinfo=dt.UTC) + dt.timedelta(minutes=minute)
    return value.isoformat().replace("+00:00", "Z")


def _annotation(
    task: BlindTask,
    annotator: str,
    *,
    answerability: str,
    evidence_alias: str = "P1.S1",
    dataset_defect: bool = False,
    minute: int,
    rationale: str = "Synthetic fixture decision grounded in the invented visible text.",
) -> AnswerabilityAnnotation:
    is_answerable = answerability == "answerable"
    return AnswerabilityAnnotation.model_validate(
        {
            "schema_version": "answerability-annotation-v1",
            "annotation_task_id": task.annotation_task_id,
            "challenge_id": task.challenge_id,
            "blinded_parent_group": task.blinded_parent_group,
            "annotator_pseudonym": annotator,
            "instruction_version": task.instruction_version,
            "instruction_hash": task.instruction_hash,
            "task_content_hash": task.task_content_hash,
            "assignment_batch": task.assignment_batch,
            "answerability": answerability,
            "answer_text": "Lumen" if is_answerable else None,
            "minimal_sufficient_evidence_sets": [[evidence_alias]] if is_answerable else [],
            "evidence_exhaustive": True if is_answerable else None,
            "ambiguity": False,
            "dataset_defect": dataset_defect,
            "confidence": 4,
            "rationale": rationale,
            "started_at": _timestamp(minute),
            "submitted_at": _timestamp(minute + 5),
        }
    )


def _formal_package_hashes(repository_root: Path) -> dict[str, str]:
    package_root = repository_root / "pilot" / "v0.2" / "packages"
    files = sorted(package_root.glob("*.json"))
    if {path.name for path in files} != {
        "ann-pilot-a.json",
        "ann-pilot-b.json",
        "manifest.json",
    }:
        raise DataError("canonical formal package sources are missing")
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _write_fixture_inputs(
    run_root: Path,
    repository_root: Path,
) -> tuple[
    AssignmentManifest,
    Path,
    tuple[Path, Path],
    tuple[Path, Path],
    Path,
    dict[str, str],
]:
    inputs = run_root / "inputs"
    inputs.mkdir(parents=True)
    protocol_path = inputs / "PILOT_PROTOCOL.md"
    shutil.copyfile(repository_root / "PILOT_PROTOCOL.md", protocol_path)
    instruction_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    tasks, base_labels = _build_tasks(instruction_hash)
    manifest = build_dual_assignments(tasks, _ANNOTATORS, seed=20260823)
    task_map = {
        task.annotation_task_id: task for package in manifest.packages for task in package.tasks
    }
    answerable_ids = [
        assignment.annotation_task_id
        for assignment in manifest.task_assignments
        if base_labels[assignment.annotation_task_id] == "answerable"
    ]
    unanswerable_ids = [
        assignment.annotation_task_id
        for assignment in manifest.task_assignments
        if base_labels[assignment.annotation_task_id] == "unanswerable"
    ]
    flipped = {answerable_ids[0], answerable_ids[1], unanswerable_ids[0], unanswerable_ids[1]}
    amendment_task_id = answerable_ids[0]
    evidence_task_id = answerable_ids[2]
    defect_task_id = unanswerable_ids[-1]
    streams: dict[str, list[AnswerabilityAnnotation]] = {annotator: [] for annotator in _ANNOTATORS}
    for position, assignment in enumerate(manifest.task_assignments):
        task = task_map[assignment.annotation_task_id]
        base = base_labels[assignment.annotation_task_id]
        right_label = base
        if assignment.annotation_task_id in flipped:
            right_label = "unanswerable" if base == "answerable" else "answerable"
        left = _annotation(
            task,
            assignment.annotators[0],
            answerability=base,
            dataset_defect=assignment.annotation_task_id == defect_task_id,
            minute=position * 10,
        )
        right = _annotation(
            task,
            assignment.annotators[1],
            answerability=right_label,
            evidence_alias=(
                "P2.S1" if assignment.annotation_task_id == evidence_task_id else "P1.S1"
            ),
            dataset_defect=assignment.annotation_task_id == defect_task_id,
            minute=position * 10 + 1,
        )
        streams[left.annotator_pseudonym].append(left)
        streams[right.annotator_pseudonym].append(right)

    amendment_original = next(
        record
        for record in streams[_ANNOTATORS[1]]
        if record.annotation_task_id == amendment_task_id
    )
    amendment_task = task_map[amendment_task_id]
    replacement = _annotation(
        amendment_task,
        amendment_original.annotator_pseudonym,
        answerability="answerable",
        minute=405,
        rationale="Synthetic amendment corrects the deliberately flipped fixture decision.",
    )
    amendment = AnnotationAmendment.model_validate(
        {
            "schema_version": "annotation-amendment-v1",
            "amendment_id": _opaque_id("amend", "synthetic-rehearsal", 1),
            "original_annotation_hash": artifact_hash(amendment_original),
            "previous_amendment_hash": None,
            "annotator_pseudonym": amendment_original.annotator_pseudonym,
            "reason": "Exercise one valid append-only synthetic amendment chain.",
            "replacement": replacement.model_dump(mode="json"),
            "created_at": _timestamp(411),
        }
    )
    manifest_path = inputs / "manifest.json"
    write_json_atomic(manifest_path, manifest.model_dump(mode="json"))
    for package in manifest.packages:
        write_json_atomic(
            inputs / f"{package.annotator_pseudonym}.json",
            package.blind_export(),
        )
    submission_paths = (
        inputs / "submission-a.jsonl",
        inputs / "submission-b.jsonl",
    )
    amendment_paths = (
        inputs / "amendment-a.jsonl",
        inputs / "amendment-b.jsonl",
    )
    for index, annotator in enumerate(_ANNOTATORS):
        write_records_atomic(
            submission_paths[index],
            (record.model_dump(mode="json") for record in streams[annotator]),
        )
        records = [amendment] if annotator == amendment.annotator_pseudonym else []
        write_records_atomic(
            amendment_paths[index],
            (record.model_dump(mode="json") for record in records),
        )
    return (
        manifest,
        manifest_path,
        submission_paths,
        amendment_paths,
        protocol_path,
        base_labels,
    )


def _adjudication_payload(
    case: dict[str, object],
    *,
    base_answerability: str,
    position: int,
) -> dict[str, object]:
    left = AnswerabilityAnnotation.model_validate(case["left"])
    right = AnswerabilityAnnotation.model_validate(case["right"])
    answerable = base_answerability == "answerable"
    record = AdjudicationRecord.model_validate(
        {
            "schema_version": "adjudication-v1",
            "adjudication_id": _opaque_id("adj", "synthetic-rehearsal", position),
            "annotation_task_id": left.annotation_task_id,
            "challenge_id": left.challenge_id,
            "blinded_parent_group": left.blinded_parent_group,
            "adjudicator_pseudonym": _ADJUDICATOR,
            "left": left.model_dump(mode="json"),
            "right": right.model_dump(mode="json"),
            "final_answerability": base_answerability,
            "answer_text": "Lumen" if answerable else None,
            "minimal_sufficient_evidence_sets": [["P1.S1"]] if answerable else [],
            "evidence_exhaustive": True if answerable else None,
            "excluded": False,
            "exclusion_reason": None,
            "rationale": "Synthetic third-review fixture based only on invented visible text.",
            "created_at": _timestamp(420 + position),
        }
    )
    return record.model_dump(mode="json")


def _run_once(run_root: Path, repository_root: Path) -> _RunResult:
    run_root.mkdir(parents=True)
    (run_root / "SYNTHETIC-NOT-HUMAN-DATA.txt").write_text(
        _MARKER,
        encoding="utf-8",
        newline="\n",
    )
    (
        manifest,
        manifest_path,
        submission_paths,
        amendment_paths,
        protocol_path,
        base_labels,
    ) = _write_fixture_inputs(run_root, repository_root)
    collection = collect_annotation_streams(
        manifest_path,
        submission_paths,
        amendment_paths,
        run_root / "collection",
    )
    if not collection.complete or collection.disagreements is None:
        raise DataError("synthetic collection unexpectedly failed completeness")

    try:
        from fastapi.testclient import TestClient
    except ImportError as exc:
        raise DataError("synthetic rehearsal requires the development test dependencies") from exc
    app = create_adjudication_app(
        manifest_path,
        run_root / "collection" / "effective-submissions.jsonl",
        run_root / "adjudication-state",
    )
    with TestClient(app) as client:
        initial_status = client.get("/api/status")
        if initial_status.status_code != 200:
            raise DataError("synthetic adjudication status endpoint failed")
        cases_response = client.get("/api/disagreements")
        if cases_response.status_code != 200:
            raise DataError("synthetic disagreement endpoint failed")
        cases = cases_response.json()
        for position, case in enumerate(cases, start=1):
            task_id = case.get("annotation_task_id")
            if not isinstance(task_id, str):
                raise DataError("synthetic disagreement lacks a task ID")
            response = client.post(
                "/api/adjudications",
                json=_adjudication_payload(
                    case,
                    base_answerability=base_labels[task_id],
                    position=position,
                ),
            )
            if response.status_code != 201:
                raise DataError("synthetic adjudication submission failed")
        final_status = client.get("/api/status")
        status_payload = final_status.json()
        if final_status.status_code != 200 or not status_payload.get("complete"):
            raise DataError("synthetic adjudication status did not become complete")
        export = client.get("/api/export/adjudications.jsonl")
        if export.status_code != 200:
            raise DataError("synthetic adjudication JSONL export failed")
    adjudications_path = run_root / "adjudications.jsonl"
    adjudications_path.write_text(export.text, encoding="utf-8", newline="\n")
    final = finalize_pilot(
        manifest_path,
        run_root / "collection" / "original-submissions.jsonl",
        run_root / "collection" / "amendments.jsonl",
        adjudications_path,
        protocol_path,
        run_root / "final",
    )
    eligibility = read_json(run_root / "final" / "eligibility.json")
    if not isinstance(eligibility, dict):
        raise DataError("synthetic eligibility artifact is not an object")
    records = eligibility.get("records")
    if not isinstance(records, list):
        raise DataError("synthetic eligibility records are missing")
    defect_exclusions = sum(
        record.get("exclusion_reason") == "dataset defect"
        for record in records
        if isinstance(record, dict)
    )
    adjudication_count = sum(1 for _record in read_records(adjudications_path))
    return _RunResult(
        tasks=len(manifest.task_assignments),
        amendments=collection.amendments,
        disagreements=collection.disagreements,
        adjudications=adjudication_count,
        dataset_defect_exclusions=defect_exclusions,
        verdict=final.verdict,
    )


def run_synthetic_rehearsal(
    out: Path,
    repository_root: Path,
) -> RehearsalResult:
    """Run the invented 40-task path twice and prove formal-package noninterference."""
    resolved_repository = repository_root.resolve()
    if out.resolve().is_relative_to(resolved_repository):
        raise DataError(
            "synthetic rehearsal output must remain outside the repository/formal paths"
        )
    if out.exists() and not out.is_dir():
        raise DataError("synthetic rehearsal output must be a directory")
    if out.exists() and any(out.iterdir()):
        raise DataError("synthetic rehearsal output directory must be absent or empty")
    before = _formal_package_hashes(resolved_repository)
    out.mkdir(parents=True, exist_ok=True)
    first = _run_once(out / "run-1", resolved_repository)
    if _formal_package_hashes(resolved_repository) != before:
        raise DataError("synthetic rehearsal changed canonical formal package hashes")
    second = _run_once(out / "run-2", resolved_repository)
    if _formal_package_hashes(resolved_repository) != before:
        raise DataError("synthetic rehearsal changed canonical formal package hashes")
    if first != second:
        raise DataError("synthetic repeated runs produced different accounting")
    byte_identical = _tree_bytes(out / "run-1") == _tree_bytes(out / "run-2")
    if not byte_identical:
        raise DataError("synthetic repeated runs are not byte-identical")
    return RehearsalResult(
        tasks=first.tasks,
        amendments=first.amendments,
        disagreements=first.disagreements,
        adjudications=first.adjudications,
        dataset_defect_exclusions=first.dataset_defect_exclusions,
        repeat_byte_identical=True,
        verdict=first.verdict.value,
        output_dir=out,
    )
