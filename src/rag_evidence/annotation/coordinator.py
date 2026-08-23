"""Coordinator-only collection and aggregate amendment resolution."""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from rag_evidence.annotation.assignment import AssignmentManifest
from rag_evidence.annotation.models import (
    AnnotationAmendment,
    AnswerabilityAnnotation,
    CitationAnnotation,
    artifact_hash,
)
from rag_evidence.annotation.privacy import scan_private_payload
from rag_evidence.annotation.workflow import build_disagreement_queue, index_submissions
from rag_evidence.errors import ArtifactError, DataError
from rag_evidence.storage.artifacts import (
    read_json,
    read_records,
    write_json_atomic,
    write_records_atomic,
)

_BINDING_FIELDS = (
    "annotation_task_id",
    "challenge_id",
    "blinded_parent_group",
    "annotator_pseudonym",
    "instruction_version",
    "instruction_hash",
    "task_content_hash",
    "assignment_batch",
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactDigest(_StrictModel):
    logical_name: str = Field(min_length=1)
    role: Literal["input", "output"]
    basename: str = Field(min_length=1)
    byte_size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("basename")
    @classmethod
    def _basename_only(cls, value: str) -> str:
        if Path(value).name != value or value in {".", ".."}:
            raise ValueError("artifact digest basename must not contain a path")
        return value


class CollectionReceipt(_StrictModel):
    schema_version: Literal["annotation-collection-receipt-v1"] = (
        "annotation-collection-receipt-v1"
    )
    generated_at: dt.datetime
    assigned_tasks: int = Field(ge=1)
    completed_tasks: int = Field(ge=0)
    original_submissions: int = Field(ge=0)
    amendments: int = Field(ge=0)
    effective_submissions: int = Field(ge=0)
    disagreements: int | None = Field(default=None, ge=0)
    complete: bool


class CollectionInputManifest(_StrictModel):
    schema_version: Literal["annotation-input-manifest-v1"] = "annotation-input-manifest-v1"
    artifacts: tuple[ArtifactDigest, ...]


@dataclass(frozen=True)
class CollectionResult:
    complete: bool
    assigned_tasks: int
    completed_tasks: int
    original_submissions: int
    amendments: int
    effective_submissions: int
    disagreements: int | None
    output_dir: Path


@dataclass(frozen=True)
class _AmendmentResolution:
    effective: tuple[AnswerabilityAnnotation, ...]
    ordered_amendments: tuple[AnnotationAmendment, ...]


def _submission_order(
    manifest: AssignmentManifest,
) -> dict[tuple[str, str], int]:
    order: dict[tuple[str, str], int] = {}
    for task_index, assignment in enumerate(manifest.task_assignments):
        for annotator_index, annotator in enumerate(assignment.annotators):
            order[(assignment.annotation_task_id, annotator)] = task_index * 2 + annotator_index
    return order


def _same_binding(
    original: AnswerabilityAnnotation,
    replacement: AnswerabilityAnnotation | CitationAnnotation,
) -> bool:
    return isinstance(replacement, AnswerabilityAnnotation) and all(
        getattr(original, field) == getattr(replacement, field) for field in _BINDING_FIELDS
    )


def _resolve_amendments(
    manifest: AssignmentManifest,
    originals: Sequence[AnswerabilityAnnotation],
    amendments: Sequence[AnnotationAmendment],
) -> _AmendmentResolution:
    index_submissions(manifest, originals)
    order = _submission_order(manifest)
    sorted_originals = tuple(
        sorted(
            originals,
            key=lambda record: order[(record.annotation_task_id, record.annotator_pseudonym)],
        )
    )
    originals_by_hash: dict[str, AnswerabilityAnnotation] = {}
    for original in sorted_originals:
        digest = artifact_hash(original)
        if digest in originals_by_hash:
            raise DataError("duplicate canonical original annotation hash")
        originals_by_hash[digest] = original

    amendments_by_original: dict[str, list[AnnotationAmendment]] = {
        digest: [] for digest in originals_by_hash
    }
    hashes_by_original: dict[str, dict[str, AnnotationAmendment]] = {
        digest: {} for digest in originals_by_hash
    }
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for amendment in amendments:
        if amendment.amendment_id in seen_ids:
            raise DataError(f"duplicate amendment ID: {amendment.amendment_id}")
        seen_ids.add(amendment.amendment_id)
        digest = artifact_hash(amendment)
        if digest in seen_hashes:
            raise DataError(f"duplicate amendment hash: {digest}")
        seen_hashes.add(digest)
        bound_original = originals_by_hash.get(amendment.original_annotation_hash)
        if bound_original is None:
            raise DataError("amendment original annotation hash is not in collected submissions")
        if amendment.annotator_pseudonym != bound_original.annotator_pseudonym:
            raise DataError("amendment annotator does not match its original annotation")
        if not _same_binding(bound_original, amendment.replacement):
            raise DataError("amendment replacement changes the original annotation binding")
        amendments_by_original[amendment.original_annotation_hash].append(amendment)
        hashes_by_original[amendment.original_annotation_hash][digest] = amendment

    effective_by_original = dict(originals_by_hash)
    ordered_amendments: list[AnnotationAmendment] = []
    all_amendment_hashes = {
        digest for group in hashes_by_original.values() for digest in group
    }
    for original in sorted_originals:
        original_hash = artifact_hash(original)
        group = amendments_by_original[original_hash]
        if not group:
            continue
        by_hash = hashes_by_original[original_hash]
        heads: list[AnnotationAmendment] = []
        successor: dict[str, AnnotationAmendment] = {}
        for amendment in group:
            previous = amendment.previous_amendment_hash
            if previous is None:
                heads.append(amendment)
                continue
            if previous not in by_hash:
                if previous in all_amendment_hashes:
                    raise DataError("amendment predecessor points to a different original chain")
                raise DataError("amendment chain has a broken predecessor")
            if previous in successor:
                raise DataError("amendment chain contains a fork")
            successor[previous] = amendment
        if len(heads) != 1:
            if not heads:
                raise DataError("amendment chain contains a cycle or has no head")
            raise DataError("amendment chain is disconnected and has multiple heads")

        current = heads[0]
        visited: set[str] = set()
        chain: list[AnnotationAmendment] = []
        while True:
            current_hash = artifact_hash(current)
            if current_hash in visited:
                raise DataError("amendment chain contains a cycle")
            visited.add(current_hash)
            chain.append(current)
            next_amendment = successor.get(current_hash)
            if next_amendment is None:
                break
            current = next_amendment
        if len(visited) != len(group):
            raise DataError("amendment chain is disconnected or cyclic")
        replacement = chain[-1].replacement
        if not isinstance(replacement, AnswerabilityAnnotation):
            raise DataError("answerability amendment contains a citation replacement")
        effective_by_original[original_hash] = replacement
        ordered_amendments.extend(chain)

    effective = tuple(
        effective_by_original[artifact_hash(original)] for original in sorted_originals
    )
    return _AmendmentResolution(
        effective=effective,
        ordered_amendments=tuple(ordered_amendments),
    )


def resolve_amendments(
    manifest: AssignmentManifest,
    originals: Sequence[AnswerabilityAnnotation],
    amendments: Sequence[AnnotationAmendment],
) -> tuple[AnswerabilityAnnotation, ...]:
    """Resolve one append-only chain per canonical original, independent of file order."""
    return _resolve_amendments(manifest, originals, amendments).effective


def _load_manifest(path: Path) -> AssignmentManifest:
    if not path.is_file():
        raise DataError(f"assignment manifest file is missing: {path.name}")
    try:
        payload = read_json(path)
        violations = scan_private_payload(payload)
        if violations:
            raise DataError("assignment manifest privacy violation: " + "; ".join(violations))
        return AssignmentManifest.model_validate(payload)
    except (ArtifactError, ValidationError) as exc:
        raise DataError(f"invalid assignment manifest {path.name}: {exc}") from exc


def _load_submissions(path: Path) -> tuple[AnswerabilityAnnotation, ...]:
    if not path.is_file():
        raise DataError(f"submission file is missing: {path.name}")
    try:
        records = tuple(
            AnswerabilityAnnotation.model_validate(payload) for payload in read_records(path)
        )
    except (ArtifactError, ValidationError) as exc:
        raise DataError(f"invalid submission file {path.name}: {exc}") from exc
    violations = scan_private_payload([record.model_dump(mode="json") for record in records])
    if violations:
        raise DataError(f"submission privacy violation in {path.name}: " + "; ".join(violations))
    return records


def _load_amendments(path: Path) -> tuple[AnnotationAmendment, ...]:
    if not path.is_file():
        raise DataError(f"amendment file is missing: {path.name}")
    try:
        records = tuple(
            AnnotationAmendment.model_validate(payload) for payload in read_records(path)
        )
    except (ArtifactError, ValidationError) as exc:
        raise DataError(f"invalid amendment file {path.name}: {exc}") from exc
    violations = scan_private_payload([record.model_dump(mode="json") for record in records])
    if violations:
        raise DataError(f"amendment privacy violation in {path.name}: " + "; ".join(violations))
    return records


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(
    logical_name: str,
    role: Literal["input", "output"],
    path: Path,
) -> ArtifactDigest:
    return ArtifactDigest(
        logical_name=logical_name,
        role=role,
        basename=path.name,
        byte_size=path.stat().st_size,
        sha256=_sha256_file(path),
    )


def _latest_source_time(
    originals: Sequence[AnswerabilityAnnotation],
    amendments: Sequence[AnnotationAmendment],
) -> dt.datetime:
    timestamps = [record.submitted_at for record in originals]
    timestamps.extend(record.created_at for record in amendments)
    if not timestamps:
        raise DataError("collection requires at least one submitted annotation")
    return max(timestamps)


def collect_annotation_streams(
    manifest_path: Path,
    submission_paths: Sequence[Path],
    amendment_paths: Sequence[Path],
    out: Path,
) -> CollectionResult:
    """Validate exactly two paired streams and write deterministic coordinator artifacts."""
    if len(submission_paths) != 2:
        raise DataError("collection requires exactly two submission files")
    if len(amendment_paths) != 2:
        raise DataError("collection requires exactly two amendment files")
    if out.exists() and any(out.iterdir()):
        raise DataError("collection output directory must be absent or empty")

    manifest = _load_manifest(manifest_path)
    assigned_pseudonyms = {package.annotator_pseudonym for package in manifest.packages}
    if len(assigned_pseudonyms) != 2:
        raise DataError("pilot collection requires exactly two assigned package pseudonyms")

    originals: list[AnswerabilityAnnotation] = []
    amendments: list[AnnotationAmendment] = []
    stream_pseudonyms: list[str] = []
    input_digests: list[ArtifactDigest] = [_digest("assignment-manifest", "input", manifest_path)]
    for position, (submission_path, amendment_path) in enumerate(
        zip(submission_paths, amendment_paths, strict=True), start=1
    ):
        submission_stream = _load_submissions(submission_path)
        if not submission_stream:
            raise DataError(f"submission file {position} contains no annotator pseudonym")
        pseudonyms = {record.annotator_pseudonym for record in submission_stream}
        if len(pseudonyms) != 1:
            raise DataError(f"submission file {position} must contain exactly one pseudonym")
        pseudonym = next(iter(pseudonyms))
        if pseudonym not in assigned_pseudonyms:
            raise DataError(f"submission file {position} pseudonym is not assigned")
        amendment_stream = _load_amendments(amendment_path)
        if any(record.annotator_pseudonym != pseudonym for record in amendment_stream):
            raise DataError(
                f"amendment file {position} is not paired with submission pseudonym {pseudonym}"
            )
        stream_pseudonyms.append(pseudonym)
        originals.extend(submission_stream)
        amendments.extend(amendment_stream)
        input_digests.extend(
            (
                _digest(f"submission-{position}", "input", submission_path),
                _digest(f"amendment-{position}", "input", amendment_path),
            )
        )
    if len(set(stream_pseudonyms)) != 2:
        raise DataError("submission files must contain two distinct annotator pseudonyms")

    resolution = _resolve_amendments(manifest, originals, amendments)
    indexed = index_submissions(manifest, resolution.effective)
    completed_tasks = sum(
        len(indexed.get(assignment.annotation_task_id, {})) == 2
        for assignment in manifest.task_assignments
    )
    complete = completed_tasks == len(manifest.task_assignments)
    disagreements = (
        build_disagreement_queue(manifest, resolution.effective) if complete else None
    )

    order = _submission_order(manifest)
    sorted_originals = tuple(
        sorted(
            originals,
            key=lambda record: order[(record.annotation_task_id, record.annotator_pseudonym)],
        )
    )
    out.mkdir(parents=True, exist_ok=True)
    originals_path = out / "original-submissions.jsonl"
    amendments_path = out / "amendments.jsonl"
    effective_path = out / "effective-submissions.jsonl"
    receipt_path = out / "collection-receipt.json"
    disagreement_path = out / "disagreements.jsonl"
    write_records_atomic(
        originals_path,
        (record.model_dump(mode="json") for record in sorted_originals),
    )
    write_records_atomic(
        amendments_path,
        (record.model_dump(mode="json") for record in resolution.ordered_amendments),
    )
    write_records_atomic(
        effective_path,
        (record.model_dump(mode="json") for record in resolution.effective),
    )
    if disagreements is not None:
        write_records_atomic(
            disagreement_path,
            (record.model_dump(mode="json") for record in disagreements),
        )

    receipt = CollectionReceipt(
        generated_at=_latest_source_time(sorted_originals, resolution.ordered_amendments),
        assigned_tasks=len(manifest.task_assignments),
        completed_tasks=completed_tasks,
        original_submissions=len(sorted_originals),
        amendments=len(resolution.ordered_amendments),
        effective_submissions=len(resolution.effective),
        disagreements=len(disagreements) if disagreements is not None else None,
        complete=complete,
    )
    write_json_atomic(receipt_path, receipt.model_dump(mode="json"))

    output_paths = [originals_path, amendments_path, effective_path, receipt_path]
    if disagreements is not None:
        output_paths.append(disagreement_path)
    output_digests = [
        _digest(path.stem, "output", path)
        for path in output_paths
    ]
    input_manifest = CollectionInputManifest(
        artifacts=tuple(input_digests + output_digests)
    )
    write_json_atomic(out / "input-manifest.json", input_manifest.model_dump(mode="json"))
    return CollectionResult(
        complete=complete,
        assigned_tasks=len(manifest.task_assignments),
        completed_tasks=completed_tasks,
        original_submissions=len(sorted_originals),
        amendments=len(resolution.ordered_amendments),
        effective_submissions=len(resolution.effective),
        disagreements=len(disagreements) if disagreements is not None else None,
        output_dir=out,
    )
