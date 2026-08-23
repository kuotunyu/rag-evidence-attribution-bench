"""Deterministic, decision-free pilot packaging and clean-export validation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

from pydantic import ValidationError

from rag_evidence.annotation.assignment import (
    AssignmentManifestV2,
    AssignmentPackageV2,
    SchedulableTaskV2,
    build_dual_assignments_v2,
    canonical_package_bytes,
    validate_pilot_manifest_v2,
    write_coordinator_manifest_v2,
)
from rag_evidence.annotation.blinding import project_challenge_v2
from rag_evidence.annotation.privacy import scan_delivery_payload
from rag_evidence.config import AppConfig
from rag_evidence.data.challenge_schema import ChallengeRecord
from rag_evidence.errors import DataError
from rag_evidence.storage.artifacts import read_json, read_records, write_json_atomic

PILOT_INSTRUCTION_VERSION = "pilot-v0.2.2-draft"
PILOT_BATCH = "pilot-v0.2-smoke"
PILOT_NAMESPACE = "pilot-v0.2-smoke"
PILOT_ASSIGNMENT_SEED = 20260823
PILOT_ANNOTATORS = ("ann-pilot-a", "ann-pilot-b")
PILOT_VARIANTS = ("missing_hop", "evidence_swap")

_DECISION_KEYS = frozenset(
    {
        "answerability",
        "final_answerability",
        "answer_text",
        "minimal_sufficient_evidence_sets",
        "evidence_exhaustive",
        "ambiguity",
        "dataset_defect",
        "confidence",
        "rationale",
        "source_annotation_hashes",
        "source_answerabilities",
        "adjudication_hash",
        "adjudication_id",
        "eligible",
        "exclusion_reason",
        "generated_at",
        "submitted_at",
    }
)
_DECISION_FILE_TERMS = ("submission", "amendment", "adjudication", "eligibility")


def _instruction_hash(path: Path) -> str:
    if not path.exists():
        raise DataError(f"pilot instruction file not found: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decision_violations(payload: object) -> tuple[str, ...]:
    violations: list[str] = []

    def visit(value: object, path: str) -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key).casefold()
                if key in _DECISION_KEYS or key.endswith(("_annotation", "_adjudication")):
                    violations.append(f"{path}.{raw_key}: scientific decision field")
                visit(child, f"{path}.{raw_key}")
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(payload, "$")
    return tuple(sorted(set(violations)))


def scan_clean_package(root: Path) -> dict[str, int]:
    """Validate that a pre-annotation directory contains only blind assignment JSON."""
    if not root.is_dir():
        raise DataError(f"clean package directory not found: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise DataError("clean package directory is empty")
    violations: list[str] = []
    package_models: list[AssignmentPackageV2] = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        lowered = relative.casefold()
        if any(term in lowered for term in _DECISION_FILE_TERMS) or path.suffix == ".jsonl":
            violations.append(f"{relative}: decision artifact is forbidden in a clean package")
            continue
        if path.suffix != ".json":
            violations.append(f"{relative}: only JSON assignment artifacts are allowed")
            continue
        try:
            payload = read_json(path)
        except Exception as exc:
            violations.append(f"{relative}: unreadable JSON ({exc})")
            continue
        if not isinstance(payload, dict):
            violations.append(f"{relative}: top level must be an object")
            continue
        for message in (
            *scan_delivery_payload(payload, artifact_kind="assignment_package"),
            *_decision_violations(payload),
        ):
            violations.append(f"{relative}: {message}")
        try:
            if payload.get("schema_version") == "assignment-package-v2":
                package_models.append(AssignmentPackageV2.model_validate(payload))
            else:
                violations.append(f"{relative}: unexpected clean-package schema")
        except ValidationError as exc:
            violations.append(f"{relative}: schema validation failed ({exc})")
    if violations:
        raise DataError("clean-package privacy scan failed: " + "; ".join(violations))
    actual_packages = {package.annotator_pseudonym: package for package in package_models}
    if set(actual_packages) != set(PILOT_ANNOTATORS) or len(package_models) != 2:
        raise DataError(
            "clean-package privacy scan failed: exactly one A package and one B package required"
        )
    expected_files = {f"{pseudonym}.json" for pseudonym in PILOT_ANNOTATORS}
    if {path.name for path in files} != expected_files:
        raise DataError("clean-package privacy scan failed: unexpected delivery filename")
    task_ids = {
        task.annotation_task_id for package in actual_packages.values() for task in package.tasks
    }
    if any(len(package.tasks) != 40 for package in actual_packages.values()):
        raise DataError("clean-package privacy scan failed: each pilot package requires 40 tasks")
    if len(task_ids) != 40:
        raise DataError("clean-package privacy scan failed: A/B task sets must match")
    return {
        "files": len(files),
        "packages": len(actual_packages),
        "tasks": len(task_ids),
        "decisions": 0,
    }


def _coordinator_group_id(parent_question_id: str) -> str:
    material = f"{PILOT_NAMESPACE}\0{parent_question_id}".encode()
    return f"coord-{hashlib.sha256(material).hexdigest()[:24]}"


def build_pilot_assignments(
    records: Sequence[ChallengeRecord],
    instruction_hash: str,
) -> tuple[AssignmentPackageV2, AssignmentPackageV2, AssignmentManifestV2]:
    """Build canonical group-free A/B packages and their private coordinator mapping."""
    ordered = sorted(records, key=lambda record: (record.parent_question_id, record.transformation))
    parent_variants: dict[str, set[str]] = {}
    for record in ordered:
        parent_variants.setdefault(record.parent_question_id, set()).add(record.transformation)
    expected_variants = set(PILOT_VARIANTS)
    if (
        len(parent_variants) != 20
        or len(ordered) != 40
        or any(variants != expected_variants for variants in parent_variants.values())
    ):
        raise DataError(
            "pilot source must contain exactly 20 parents with missing-hop and evidence-swap"
        )
    schedulable = tuple(
        SchedulableTaskV2(
            task=project_challenge_v2(
                record,
                instruction_version=PILOT_INSTRUCTION_VERSION,
                instruction_hash=instruction_hash,
                batch=PILOT_BATCH,
                namespace=PILOT_NAMESPACE,
            ),
            internal_group_id=_coordinator_group_id(record.parent_question_id),
            transformation=cast(
                Literal["missing_hop", "evidence_swap"], record.transformation
            ),
        )
        for record in ordered
    )
    packages, manifest = build_dual_assignments_v2(
        schedulable,
        PILOT_ANNOTATORS,
        seed=PILOT_ASSIGNMENT_SEED,
    )
    validate_pilot_manifest_v2(manifest)
    if len(packages) != 2:
        raise DataError("pilot assignment builder must produce exactly two packages")
    return packages[0], packages[1], manifest


def build_pilot_package(
    cfg: AppConfig,
    out: Path,
    *,
    instruction_path: Path = Path("PILOT_PROTOCOL.md"),
) -> AssignmentManifestV2:
    """Project all 20 smoke parents' two reviewed variants; never create decisions."""
    if cfg.split != "smoke":
        raise DataError("pilot package must be built from the Dataset-v2 smoke split")
    source = cfg.results_raw_dir / cfg.split / "challenge" / "samples" / "records.jsonl"
    if not source.exists():
        raise DataError(f"challenge source not found: {source}; run data challenge first")
    records: list[ChallengeRecord] = []
    for payload in read_records(source):
        if payload.get("source_split") == "smoke" and payload.get("transformation") in (
            "missing_hop",
            "evidence_swap",
        ):
            records.append(ChallengeRecord.from_json(payload))
    instruction_hash = _instruction_hash(instruction_path)
    package_a, package_b, manifest = build_pilot_assignments(records, instruction_hash)
    out.mkdir(parents=True, exist_ok=True)
    existing_forbidden = [
        path
        for path in out.rglob("*")
        if path.is_file() and any(term in path.name.casefold() for term in _DECISION_FILE_TERMS)
    ]
    if existing_forbidden:
        raise DataError("refusing to overwrite a package directory containing decision artifacts")
    unexpected = {
        path.name for path in out.iterdir() if path.is_file()
    } - {"ann-pilot-a.json", "ann-pilot-b.json"}
    if unexpected:
        raise DataError("refusing package output containing unexpected files")
    for package in (package_a, package_b):
        output_path = out / f"{package.annotator_pseudonym}.json"
        write_json_atomic(output_path, package.blind_export())
        if output_path.read_bytes() != canonical_package_bytes(package):
            raise DataError("package serializer did not emit canonical bytes")
    scan_clean_package(out)
    return manifest


def rebuild_coordinator_manifest(
    challenge_records: Path,
    package_a_path: Path,
    package_b_path: Path,
    output: Path,
    *,
    repository_root: Path,
) -> AssignmentManifestV2:
    """Rebuild the private mapping and bind it to exact canonical A/B source bytes."""
    try:
        package_a = AssignmentPackageV2.model_validate(read_json(package_a_path))
        package_b = AssignmentPackageV2.model_validate(read_json(package_b_path))
    except (ValidationError, TypeError, ValueError) as exc:
        raise DataError(f"invalid v2 assignment package: {exc}") from exc
    packages = {package.annotator_pseudonym: package for package in (package_a, package_b)}
    if set(packages) != set(PILOT_ANNOTATORS):
        raise DataError("coordinator rebuild requires ann-pilot-a and ann-pilot-b packages")
    bindings = {
        (package.instruction_version, package.instruction_hash, package.assignment_batch)
        for package in packages.values()
    }
    if bindings != {(PILOT_INSTRUCTION_VERSION, package_a.instruction_hash, PILOT_BATCH)}:
        raise DataError("A/B packages do not share the approved pilot binding")
    records = tuple(
        ChallengeRecord.from_json(payload) for payload in read_records(challenge_records)
    )
    expected_a, expected_b, manifest = build_pilot_assignments(records, package_a.instruction_hash)
    expected = {
        expected_a.annotator_pseudonym: expected_a,
        expected_b.annotator_pseudonym: expected_b,
    }
    paths = {
        "ann-pilot-a": package_a_path,
        "ann-pilot-b": package_b_path,
    }
    for pseudonym, package in packages.items():
        canonical = canonical_package_bytes(expected[pseudonym])
        if package != expected[pseudonym] or paths[pseudonym].read_bytes() != canonical:
            raise DataError(f"{pseudonym} package bytes do not match canonical source")
    write_coordinator_manifest_v2(output, manifest, repository_root=repository_root)
    return manifest
