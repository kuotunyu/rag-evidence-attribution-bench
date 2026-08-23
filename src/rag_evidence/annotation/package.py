"""Deterministic, decision-free pilot packaging and clean-export validation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from rag_evidence.annotation.assignment import (
    AssignmentManifest,
    AssignmentPackage,
    build_dual_assignments,
)
from rag_evidence.annotation.blinding import project_challenge, scan_blind_payload
from rag_evidence.config import AppConfig
from rag_evidence.data.challenge_schema import ChallengeRecord
from rag_evidence.errors import DataError
from rag_evidence.storage.artifacts import read_json, read_records, write_json_atomic

PILOT_INSTRUCTION_VERSION = "pilot-v0.2.1-draft"
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


def _blind_violations(payload: dict[str, Any]) -> tuple[str, ...]:
    # Assignment randomization seed is coordinator metadata, not a transformation seed.
    # It is removed only for the generic hidden-source-field scan.
    scanned = dict(payload)
    if scanned.get("schema_version") == "assignment-manifest-v1":
        scanned.pop("seed", None)
    return scan_blind_payload(scanned)


def scan_clean_package(root: Path) -> dict[str, int]:
    """Validate that a pre-annotation directory contains only blind assignment JSON."""
    if not root.is_dir():
        raise DataError(f"clean package directory not found: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise DataError("clean package directory is empty")
    violations: list[str] = []
    package_models: list[AssignmentPackage] = []
    manifest: AssignmentManifest | None = None
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
        for message in (*_blind_violations(payload), *_decision_violations(payload)):
            violations.append(f"{relative}: {message}")
        try:
            if payload.get("schema_version") == "assignment-manifest-v1":
                if manifest is not None:
                    violations.append(f"{relative}: duplicate assignment manifest")
                manifest = AssignmentManifest.model_validate(payload)
            elif payload.get("schema_version") == "assignment-package-v1":
                package_models.append(AssignmentPackage.model_validate(payload))
            else:
                violations.append(f"{relative}: unexpected clean-package schema")
        except ValidationError as exc:
            violations.append(f"{relative}: schema validation failed ({exc})")
    if violations:
        raise DataError("clean-package privacy scan failed: " + "; ".join(violations))
    if manifest is None:
        raise DataError("clean-package privacy scan failed: assignment manifest is missing")
    expected_packages = {package.annotator_pseudonym: package for package in manifest.packages}
    actual_packages = {package.annotator_pseudonym: package for package in package_models}
    if expected_packages != actual_packages or len(actual_packages) != 2:
        raise DataError(
            "clean-package privacy scan failed: exactly two package files must match manifest"
        )
    task_count = len(manifest.task_assignments)
    return {
        "files": len(files),
        "packages": len(actual_packages),
        "tasks": task_count,
        "decisions": 0,
    }


def build_pilot_package(
    cfg: AppConfig,
    out: Path,
    *,
    instruction_path: Path = Path("PILOT_PROTOCOL.md"),
) -> AssignmentManifest:
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
    records.sort(key=lambda record: (record.parent_question_id, record.transformation))
    parent_variants: dict[str, set[str]] = {}
    for record in records:
        parent_variants.setdefault(record.parent_question_id, set()).add(record.transformation)
    expected_variants = set(PILOT_VARIANTS)
    if (
        len(parent_variants) != 20
        or len(records) != 40
        or any(variants != expected_variants for variants in parent_variants.values())
    ):
        raise DataError(
            "pilot source must contain exactly 20 parents with missing-hop and evidence-swap"
        )

    instruction_hash = _instruction_hash(instruction_path)
    tasks = tuple(
        project_challenge(
            record,
            instruction_version=PILOT_INSTRUCTION_VERSION,
            instruction_hash=instruction_hash,
            batch=PILOT_BATCH,
            namespace=PILOT_NAMESPACE,
        )
        for record in records
    )
    manifest = build_dual_assignments(tasks, PILOT_ANNOTATORS, seed=PILOT_ASSIGNMENT_SEED)
    out.mkdir(parents=True, exist_ok=True)
    existing_forbidden = [
        path
        for path in out.rglob("*")
        if path.is_file() and any(term in path.name.casefold() for term in _DECISION_FILE_TERMS)
    ]
    if existing_forbidden:
        raise DataError("refusing to overwrite a package directory containing decision artifacts")
    write_json_atomic(out / "manifest.json", manifest.model_dump(mode="json"))
    for package in manifest.packages:
        write_json_atomic(out / f"{package.annotator_pseudonym}.json", package.blind_export())
    scan_clean_package(out)
    return manifest
