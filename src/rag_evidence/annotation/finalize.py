"""Deterministic pilot accounting, agreement gates, and fail-closed verdicts."""

from __future__ import annotations

import datetime as dt
import hashlib
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rag_evidence.annotation.agreement import (
    EvidenceAgreementSummary,
    NominalAgreement,
    aggregate_evidence_agreement,
    nominal_agreement,
)
from rag_evidence.annotation.assignment import AssignmentManifest
from rag_evidence.annotation.coordinator import resolve_amendments
from rag_evidence.annotation.models import (
    AdjudicationRecord,
    AnnotationAmendment,
    AnswerabilityAnnotation,
    EligibilityArtifact,
    artifact_hash,
)
from rag_evidence.annotation.privacy import scan_private_payload
from rag_evidence.annotation.workflow import (
    FlowAccounting,
    build_workflow_result,
    index_submissions,
)
from rag_evidence.errors import ArtifactError, DataError
from rag_evidence.storage.artifacts import (
    read_json,
    read_records,
    write_json_atomic,
    write_records_atomic,
)

EXPECTED_PILOT_TASKS = 40
IAA_THRESHOLD = 0.70
EXPECTED_INSTRUCTION_VERSION = "pilot-v0.2.1-draft"
FINAL_ARTIFACT_NAMES = (
    "flow-accounting.json",
    "disagreements.jsonl",
    "eligibility.json",
    "iaa.json",
    "evidence-agreement.json",
    "timing-summary.json",
    "privacy-scan.json",
    "input-manifest.json",
    "pilot-verdict.json",
    "pilot-report.md",
)
_EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PilotVerdictName(StrEnum):
    BLOCKED_PRIVACY = "BLOCKED_PRIVACY"
    BLOCKED_INTEGRITY = "BLOCKED_INTEGRITY"
    BLOCKED_INCOMPLETE = "BLOCKED_INCOMPLETE"
    BLOCKED_UNRESOLVED_ADJUDICATION = "BLOCKED_UNRESOLVED_ADJUDICATION"
    BLOCKED_LOW_IAA = "BLOCKED_LOW_IAA"
    READY_FOR_HUMAN_FREEZE_REVIEW = "READY_FOR_HUMAN_FREEZE_REVIEW"


@dataclass(frozen=True)
class IaaGateResult:
    passed: bool
    threshold: float
    expected_tasks: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PilotFinalizationResult:
    verdict: PilotVerdictName
    output_dir: Path
    blockers: tuple[str, ...]


class IaaArtifact(_StrictModel):
    schema_version: Literal["pilot-iaa-v1"] = "pilot-iaa-v1"
    generated_at: dt.datetime
    population: Literal["post-amendment-pre-adjudication"] = "post-amendment-pre-adjudication"
    required_tasks: Literal[40] = 40
    threshold: float = Field(default=IAA_THRESHOLD, ge=IAA_THRESHOLD, le=IAA_THRESHOLD)
    n_total: int = Field(ge=0)
    n_complete: int = Field(ge=0)
    n_missing: int = Field(ge=0)
    raw_agreement: float | None
    cohen_kappa: float | None
    krippendorff_alpha: float | None
    left_prevalence: dict[str, float]
    right_prevalence: dict[str, float]
    pooled_prevalence: dict[str, float]
    gate_passed: bool
    gate_reasons: tuple[str, ...]


class EvidenceAgreementArtifact(_StrictModel):
    schema_version: Literal["pilot-evidence-agreement-v1"] = "pilot-evidence-agreement-v1"
    generated_at: dt.datetime
    n_total_tasks: int = Field(ge=0)
    n_comparable: int = Field(ge=0)
    n_excluded_not_both_answerable: int = Field(ge=0)
    n_invalid_empty_family: int = Field(ge=0)
    exact_set_family_agreement_rate: float | None
    mean_jaccard: float | None
    mean_set_f1: float | None


class PrivacyScanArtifact(_StrictModel):
    schema_version: Literal["pilot-privacy-scan-v1"] = "pilot-privacy-scan-v1"
    generated_at: dt.datetime
    passed: bool
    scanned_artifacts: tuple[str, ...]
    violations: tuple[str, ...]


class PilotVerdictArtifact(_StrictModel):
    schema_version: Literal["pilot-verdict-v1"] = "pilot-verdict-v1"
    generated_at: dt.datetime
    verdict: PilotVerdictName
    ready_for_human_freeze_review: bool
    blockers: tuple[str, ...]
    opens_confirmatory_sample: Literal[False] = False
    starts_model_run: Literal[False] = False
    authorizes_human_pilot: Literal[False] = False
    authorizes_release: Literal[False] = False


def evaluate_iaa_gate(agreement: NominalAgreement) -> IaaGateResult:
    """Apply the fixed two-coefficient promotion rule to all 40 pilot tasks."""
    reasons: list[str] = []
    if agreement.n_total != EXPECTED_PILOT_TASKS:
        reasons.append("IAA population must contain exactly 40 assigned tasks")
    if agreement.n_complete != EXPECTED_PILOT_TASKS or agreement.n_missing != 0:
        reasons.append("IAA requires 40 complete post-amendment pairs and zero missing pairs")
    coefficients = (
        ("Cohen's kappa", agreement.cohen_kappa),
        ("Krippendorff's alpha", agreement.krippendorff_alpha),
    )
    for name, value in coefficients:
        if value is None or not math.isfinite(value):
            reasons.append(f"{name} must be defined and finite")
        elif value < IAA_THRESHOLD:
            reasons.append(f"{name} is below {IAA_THRESHOLD:.2f}")
    return IaaGateResult(
        passed=not reasons,
        threshold=IAA_THRESHOLD,
        expected_tasks=EXPECTED_PILOT_TASKS,
        reasons=tuple(reasons),
    )


def choose_pilot_verdict(
    *,
    privacy: bool = False,
    integrity: bool = False,
    incomplete: bool = False,
    unresolved: bool = False,
    low_iaa: bool = False,
) -> PilotVerdictName:
    """Return the first matching status in the owner-approved precedence order."""
    if privacy:
        return PilotVerdictName.BLOCKED_PRIVACY
    if integrity:
        return PilotVerdictName.BLOCKED_INTEGRITY
    if incomplete:
        return PilotVerdictName.BLOCKED_INCOMPLETE
    if unresolved:
        return PilotVerdictName.BLOCKED_UNRESOLVED_ADJUDICATION
    if low_iaa:
        return PilotVerdictName.BLOCKED_LOW_IAA
    return PilotVerdictName.READY_FOR_HUMAN_FREEZE_REVIEW


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(logical_name: str, role: Literal["input", "output"], path: Path) -> dict[str, Any]:
    return {
        "logical_name": logical_name,
        "role": role,
        "basename": path.name,
        "byte_size": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _parse_utc(value: object) -> dt.datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.utcoffset() != dt.timedelta(0):
        return None
    return parsed


def _source_generated_at(payloads: Sequence[object]) -> dt.datetime:
    timestamps: list[dt.datetime] = []

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key in {"started_at", "submitted_at", "created_at"}:
                    parsed = _parse_utc(child)
                    if parsed is not None:
                        timestamps.append(parsed)
                visit(child)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for child in value:
                visit(child)

    for payload in payloads:
        visit(payload)
    return max(timestamps, default=_EPOCH)


def _agreement_artifact(
    agreement: NominalAgreement,
    gate: IaaGateResult,
    generated_at: dt.datetime,
) -> IaaArtifact:
    return IaaArtifact(
        generated_at=generated_at,
        n_total=agreement.n_total,
        n_complete=agreement.n_complete,
        n_missing=agreement.n_missing,
        raw_agreement=agreement.raw_agreement,
        cohen_kappa=agreement.cohen_kappa,
        krippendorff_alpha=agreement.krippendorff_alpha,
        left_prevalence=agreement.left_prevalence,
        right_prevalence=agreement.right_prevalence,
        pooled_prevalence=agreement.pooled_prevalence,
        gate_passed=gate.passed,
        gate_reasons=gate.reasons,
    )


def _evidence_artifact(
    summary: EvidenceAgreementSummary,
    generated_at: dt.datetime,
) -> EvidenceAgreementArtifact:
    return EvidenceAgreementArtifact(generated_at=generated_at, **asdict(summary))


def _timing_summary(
    originals: Sequence[AnswerabilityAnnotation],
    generated_at: dt.datetime,
) -> dict[str, Any]:
    by_annotator: dict[str, list[float]] = {}
    for record in originals:
        seconds = (record.submitted_at - record.started_at).total_seconds()
        by_annotator.setdefault(record.annotator_pseudonym, []).append(seconds)
    annotators: list[dict[str, Any]] = []
    for pseudonym in sorted(by_annotator):
        values = by_annotator[pseudonym]
        annotators.append(
            {
                "annotator_pseudonym": pseudonym,
                "n_tasks": len(values),
                "total_seconds": sum(values),
                "mean_seconds": statistics.fmean(values),
                "median_seconds": statistics.median(values),
                "minimum_seconds": min(values),
                "maximum_seconds": max(values),
            }
        )
    return {
        "schema_version": "pilot-timing-summary-v1",
        "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
        "basis": "original-submission-start-to-submit",
        "annotators": annotators,
    }


def _render_report(
    verdict: PilotVerdictArtifact,
    flow: Mapping[str, Any],
    iaa: IaaArtifact,
    evidence: EvidenceAgreementArtifact,
) -> str:
    blocker_lines = "\n".join(f"- {item}" for item in verdict.blockers) or "- None"
    return (
        "# Human Annotation Pilot Operational Report\n\n"
        f"Verdict: `{verdict.verdict.value}`\n\n"
        "This report never authorizes a human pilot, confirmatory sample, model run, "
        "merge, or release. `READY_FOR_HUMAN_FREEZE_REVIEW` requests owner review only.\n\n"
        "## Flow accounting\n\n"
        f"- Assigned tasks: {flow['assigned']}\n"
        f"- Complete dual judgments: {flow['completed']}\n"
        f"- Disagreements: {flow['disagreed']}\n"
        f"- Adjudicated: {flow['adjudicated']}\n"
        f"- Excluded: {flow['excluded']}\n"
        f"- Eligible for pilot accounting: {flow['eligible']}\n\n"
        "## Frozen pre-adjudication IAA gate\n\n"
        f"- Complete pairs: {iaa.n_complete}/{iaa.required_tasks}\n"
        f"- Raw agreement (descriptive): {iaa.raw_agreement}\n"
        f"- Cohen's kappa: {iaa.cohen_kappa}\n"
        f"- Nominal Krippendorff's alpha: {iaa.krippendorff_alpha}\n"
        f"- Required threshold for each coefficient: {iaa.threshold:.2f}\n"
        f"- Gate passed: {str(iaa.gate_passed).lower()}\n\n"
        "Raw agreement and prevalence are descriptive; neither replaces either required "
        "coefficient. Adjudicated decisions are excluded from IAA.\n\n"
        "## Evidence agreement\n\n"
        f"- Comparable both-answerable pairs: {evidence.n_comparable}"
        f"/{evidence.n_total_tasks}\n"
        f"- Excluded because not both answerable: "
        f"{evidence.n_excluded_not_both_answerable}\n"
        f"- Invalid empty evidence families: {evidence.n_invalid_empty_family}\n\n"
        "## Blockers\n\n"
        f"{blocker_lines}\n"
    )


def _write_bundle(
    *,
    out: Path,
    input_paths: Sequence[tuple[str, Path]],
    generated_at: dt.datetime,
    flow: FlowAccounting,
    disagreements: Sequence[Mapping[str, Any]],
    eligibility: EligibilityArtifact,
    iaa: IaaArtifact,
    evidence: EvidenceAgreementArtifact,
    timing: Mapping[str, Any],
    privacy: PrivacyScanArtifact,
    verdict: PilotVerdictArtifact,
) -> PilotFinalizationResult:
    out.mkdir(parents=True, exist_ok=True)
    flow_payload = {
        "schema_version": "pilot-flow-accounting-v1",
        "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
        **flow.model_dump(mode="json"),
    }
    report = _render_report(verdict, flow_payload, iaa, evidence)
    payloads: dict[str, object] = {
        "flow-accounting.json": flow_payload,
        "eligibility.json": eligibility.model_dump(mode="json"),
        "iaa.json": iaa.model_dump(mode="json"),
        "evidence-agreement.json": evidence.model_dump(mode="json"),
        "timing-summary.json": dict(timing),
        "privacy-scan.json": privacy.model_dump(mode="json"),
        "pilot-verdict.json": verdict.model_dump(mode="json"),
    }
    for name, payload in payloads.items():
        write_json_atomic(out / name, payload)
    write_records_atomic(out / "disagreements.jsonl", disagreements)
    (out / "pilot-report.md").write_text(report, encoding="utf-8", newline="\n")

    output_names = [name for name in FINAL_ARTIFACT_NAMES if name != "input-manifest.json"]
    manifest_payload = {
        "schema_version": "pilot-finalization-input-manifest-v1",
        "artifacts": [
            *(_digest(logical_name, "input", path) for logical_name, path in input_paths),
            *(_digest(Path(name).stem, "output", out / name) for name in output_names),
        ],
    }
    write_json_atomic(out / "input-manifest.json", manifest_payload)
    return PilotFinalizationResult(
        verdict=verdict.verdict,
        output_dir=out,
        blockers=verdict.blockers,
    )


def _blocked_bundle(
    *,
    out: Path,
    input_paths: Sequence[tuple[str, Path]],
    generated_at: dt.datetime,
    assigned_tasks: int,
    protocol_version: str,
    protocol_hash: str,
    verdict_name: PilotVerdictName,
    blockers: Sequence[str],
    privacy_violations: Sequence[str] = (),
) -> PilotFinalizationResult:
    flow = FlowAccounting(
        assigned=assigned_tasks,
        completed=0,
        disagreed=0,
        adjudicated=0,
        excluded=0,
        eligible=0,
    )
    agreement = nominal_agreement([(None, None)] * assigned_tasks)
    gate = evaluate_iaa_gate(agreement)
    evidence = aggregate_evidence_agreement([(None, None)] * assigned_tasks)
    eligibility = EligibilityArtifact(
        phase="pilot",
        protocol_version=protocol_version,
        protocol_hash=protocol_hash,
        generated_at=generated_at,
        records=(),
    )
    privacy = PrivacyScanArtifact(
        generated_at=generated_at,
        passed=not privacy_violations,
        scanned_artifacts=tuple(logical_name for logical_name, _path in input_paths),
        violations=tuple(privacy_violations),
    )
    verdict = PilotVerdictArtifact(
        generated_at=generated_at,
        verdict=verdict_name,
        ready_for_human_freeze_review=False,
        blockers=tuple(blockers),
    )
    return _write_bundle(
        out=out,
        input_paths=input_paths,
        generated_at=generated_at,
        flow=flow,
        disagreements=(),
        eligibility=eligibility,
        iaa=_agreement_artifact(agreement, gate, generated_at),
        evidence=_evidence_artifact(evidence, generated_at),
        timing=_timing_summary((), generated_at),
        privacy=privacy,
        verdict=verdict,
    )


def _raw_assigned_count(manifest_payload: object) -> int:
    if not isinstance(manifest_payload, Mapping):
        return 0
    assignments = manifest_payload.get("task_assignments")
    if not isinstance(assignments, Sequence) or isinstance(assignments, (str, bytes)):
        return 0
    return len(assignments)


def _raw_protocol_version(manifest_payload: object) -> str:
    if not isinstance(manifest_payload, Mapping):
        return "unknown-protocol"
    packages = manifest_payload.get("packages")
    if not isinstance(packages, Sequence) or not packages:
        return "unknown-protocol"
    first = packages[0]
    if not isinstance(first, Mapping):
        return "unknown-protocol"
    value = first.get("instruction_version")
    return value if isinstance(value, str) and value else "unknown-protocol"


def finalize_pilot(
    manifest_path: Path,
    originals_path: Path,
    amendments_path: Path,
    adjudications_path: Path,
    protocol_path: Path,
    out: Path,
) -> PilotFinalizationResult:
    """Finalize one pilot without changing any source decision or opening later stages."""
    if out.exists() and any(out.iterdir()):
        raise DataError("pilot finalization output directory must be absent or empty")
    input_paths = (
        ("assignment-manifest", manifest_path),
        ("original-submissions", originals_path),
        ("amendments", amendments_path),
        ("adjudications", adjudications_path),
        ("pilot-protocol", protocol_path),
    )
    for logical_name, path in input_paths:
        if not path.is_file():
            raise DataError(f"required finalization input is missing: {logical_name}")
    protocol_bytes = protocol_path.read_bytes()
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    try:
        raw_manifest = read_json(manifest_path)
        raw_originals = tuple(read_records(originals_path))
        raw_amendments = tuple(read_records(amendments_path))
        raw_adjudications = tuple(read_records(adjudications_path))
    except ArtifactError:
        return _blocked_bundle(
            out=out,
            input_paths=input_paths,
            generated_at=_EPOCH,
            assigned_tasks=0,
            protocol_version="unknown-protocol",
            protocol_hash=protocol_hash,
            verdict_name=PilotVerdictName.BLOCKED_INTEGRITY,
            blockers=("one or more input artifacts could not be parsed",),
        )

    raw_payloads: tuple[object, ...] = (
        raw_manifest,
        raw_originals,
        raw_amendments,
        raw_adjudications,
        protocol_bytes.decode("utf-8", errors="replace"),
    )
    generated_at = _source_generated_at(raw_payloads)
    privacy_violations = scan_private_payload(raw_payloads)
    if privacy_violations:
        return _blocked_bundle(
            out=out,
            input_paths=input_paths,
            generated_at=generated_at,
            assigned_tasks=_raw_assigned_count(raw_manifest),
            protocol_version=_raw_protocol_version(raw_manifest),
            protocol_hash=protocol_hash,
            verdict_name=PilotVerdictName.BLOCKED_PRIVACY,
            blockers=("input privacy scan failed",),
            privacy_violations=privacy_violations,
        )

    try:
        manifest = AssignmentManifest.model_validate(raw_manifest)
        originals = tuple(
            AnswerabilityAnnotation.model_validate(payload) for payload in raw_originals
        )
        amendments = tuple(
            AnnotationAmendment.model_validate(payload) for payload in raw_amendments
        )
        adjudications = tuple(
            AdjudicationRecord.model_validate(payload) for payload in raw_adjudications
        )
    except ValidationError:
        return _blocked_bundle(
            out=out,
            input_paths=input_paths,
            generated_at=generated_at,
            assigned_tasks=_raw_assigned_count(raw_manifest),
            protocol_version=_raw_protocol_version(raw_manifest),
            protocol_hash=protocol_hash,
            verdict_name=PilotVerdictName.BLOCKED_INTEGRITY,
            blockers=("one or more input artifacts failed strict schema validation",),
        )

    integrity_errors: list[str] = []
    if len(manifest.task_assignments) != EXPECTED_PILOT_TASKS:
        integrity_errors.append("pilot manifest must assign exactly 40 tasks")
    instruction_bindings = {
        (package.instruction_version, package.instruction_hash) for package in manifest.packages
    }
    if len(instruction_bindings) != 1:
        integrity_errors.append("pilot packages do not share one instruction binding")
        protocol_version = _raw_protocol_version(raw_manifest)
    else:
        protocol_version, bound_hash = next(iter(instruction_bindings))
        if protocol_version != EXPECTED_INSTRUCTION_VERSION:
            integrity_errors.append(f"instruction version must be {EXPECTED_INSTRUCTION_VERSION}")
        if bound_hash != protocol_hash:
            integrity_errors.append("instruction hash does not match the exact protocol bytes")
    adjudication_ids = [record.adjudication_id for record in adjudications]
    adjudication_hashes = [artifact_hash(record) for record in adjudications]
    if len(set(adjudication_ids)) != len(adjudication_ids):
        integrity_errors.append("duplicate adjudication ID")
    if len(set(adjudication_hashes)) != len(adjudication_hashes):
        integrity_errors.append("duplicate adjudication hash")

    try:
        effective = resolve_amendments(manifest, originals, amendments)
        indexed = index_submissions(manifest, effective)
        pairs = tuple(
            (
                indexed.get(assignment.annotation_task_id, {}).get(assignment.annotators[0]),
                indexed.get(assignment.annotation_task_id, {}).get(assignment.annotators[1]),
            )
            for assignment in manifest.task_assignments
        )
        agreement = nominal_agreement(
            [
                (
                    left.answerability if left is not None else None,
                    right.answerability if right is not None else None,
                )
                for left, right in pairs
            ]
        )
        gate = evaluate_iaa_gate(agreement)
        evidence_summary = aggregate_evidence_agreement(pairs)
        workflow = build_workflow_result(
            manifest,
            effective,
            adjudications=adjudications,
            phase="pilot",
            protocol_version=protocol_version,
            protocol_hash=protocol_hash,
            generated_at=generated_at.isoformat().replace("+00:00", "Z"),
        )
    except (ArtifactError, DataError, ValidationError):
        return _blocked_bundle(
            out=out,
            input_paths=input_paths,
            generated_at=generated_at,
            assigned_tasks=len(manifest.task_assignments),
            protocol_version=protocol_version,
            protocol_hash=protocol_hash,
            verdict_name=PilotVerdictName.BLOCKED_INTEGRITY,
            blockers=(
                *integrity_errors,
                "annotation, amendment, or adjudication binding validation failed",
            ),
        )

    incomplete = workflow.flow.completed != EXPECTED_PILOT_TASKS
    unresolved = workflow.flow.adjudicated != workflow.flow.disagreed
    low_iaa = not gate.passed
    verdict_name = choose_pilot_verdict(
        integrity=bool(integrity_errors),
        incomplete=incomplete,
        unresolved=unresolved,
        low_iaa=low_iaa,
    )
    blockers = list(integrity_errors)
    if incomplete:
        blockers.append(
            f"only {workflow.flow.completed}/{EXPECTED_PILOT_TASKS} tasks have two judgments"
        )
    if unresolved:
        blockers.append(
            f"{workflow.flow.disagreed - workflow.flow.adjudicated} disagreements remain unresolved"
        )
    if low_iaa:
        blockers.extend(gate.reasons)
    verdict = PilotVerdictArtifact(
        generated_at=generated_at,
        verdict=verdict_name,
        ready_for_human_freeze_review=(
            verdict_name is PilotVerdictName.READY_FOR_HUMAN_FREEZE_REVIEW
        ),
        blockers=tuple(blockers),
    )
    iaa_artifact = _agreement_artifact(agreement, gate, generated_at)
    evidence_artifact = _evidence_artifact(evidence_summary, generated_at)
    flow_payload = workflow.flow.model_dump(mode="json")
    derived_payloads: tuple[object, ...] = (
        flow_payload,
        [case.model_dump(mode="json") for case in workflow.disagreements],
        workflow.eligibility.model_dump(mode="json"),
        iaa_artifact.model_dump(mode="json"),
        evidence_artifact.model_dump(mode="json"),
        _timing_summary(originals, generated_at),
        verdict.model_dump(mode="json"),
    )
    derived_violations = scan_private_payload(derived_payloads)
    if derived_violations:
        return _blocked_bundle(
            out=out,
            input_paths=input_paths,
            generated_at=generated_at,
            assigned_tasks=len(manifest.task_assignments),
            protocol_version=protocol_version,
            protocol_hash=protocol_hash,
            verdict_name=PilotVerdictName.BLOCKED_PRIVACY,
            blockers=("derived artifact privacy scan failed",),
            privacy_violations=derived_violations,
        )
    privacy = PrivacyScanArtifact(
        generated_at=generated_at,
        passed=True,
        scanned_artifacts=tuple(
            [logical_name for logical_name, _path in input_paths]
            + [name for name in FINAL_ARTIFACT_NAMES if name != "privacy-scan.json"]
        ),
        violations=(),
    )
    return _write_bundle(
        out=out,
        input_paths=input_paths,
        generated_at=generated_at,
        flow=workflow.flow,
        disagreements=tuple(case.model_dump(mode="json") for case in workflow.disagreements),
        eligibility=workflow.eligibility,
        iaa=iaa_artifact,
        evidence=evidence_artifact,
        timing=_timing_summary(originals, generated_at),
        privacy=privacy,
        verdict=verdict,
    )
