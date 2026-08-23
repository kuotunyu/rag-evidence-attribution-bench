"""Fail-closed execution boundary for independently human-gated challenge rows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from rag_evidence.annotation.assignment import AssignmentManifest
from rag_evidence.annotation.blinding import validate_blind_task_source
from rag_evidence.annotation.models import BlindTask, EligibilityArtifact, artifact_hash
from rag_evidence.config import AppConfig, ExecutionConfig
from rag_evidence.data.challenge_schema import ChallengeRecord
from rag_evidence.data.grouping import build_split_groups
from rag_evidence.data.schema import Example
from rag_evidence.errors import DataError
from rag_evidence.storage.artifacts import read_json, read_records

Phase = Literal["pilot", "confirmatory"]
Variant = Literal["missing_hop", "evidence_swap"]


@dataclass(frozen=True)
class EligibleChallengeSet:
    examples: tuple[Example, ...]
    metadata: dict[str, dict[str, str]]


def build_challenge_execution_config(
    cfg: AppConfig,
    *,
    phase: Phase,
    variant: Variant,
    challenge_records_path: Path,
    assignment_manifest_path: Path,
    eligibility_path: Path,
    natural_samples_path: Path,
) -> AppConfig:
    """Namespace one challenge arm without mutating the source config."""
    raw_base = Path(cfg.paths.results_raw).parent
    derived_base = Path(cfg.paths.results_derived).parent
    assets_base = Path(cfg.paths.assets_dir).parent
    scope = Path(phase) / "challenge" / variant
    paths = cfg.paths.model_copy(
        update={
            "results_raw": str(raw_base / scope / "raw"),
            "results_derived": str(derived_base / scope / "derived"),
            "assets_dir": str(assets_base / scope / "assets"),
        }
    )
    execution = ExecutionConfig(
        track="challenge",
        phase=phase,
        variant=variant,
        challenge_records_path=str(challenge_records_path),
        assignment_manifest_path=str(assignment_manifest_path),
        eligibility_path=str(eligibility_path),
        natural_samples_path=str(natural_samples_path),
    )
    attribution = cfg.attribution.model_copy(
        update={"run_namespace": f"challenge-{phase}-{variant}"}
    )
    return cfg.model_copy(
        update={"paths": paths, "execution": execution, "attribution": attribution}
    )


def _task_index(manifest: AssignmentManifest) -> dict[str, BlindTask]:
    tasks: dict[str, BlindTask] = {}
    for package in manifest.packages:
        for task in package.tasks:
            existing = tasks.setdefault(task.annotation_task_id, task)
            if existing != task:
                raise DataError("assignment manifest contains conflicting blind task copies")
    assigned = {row.annotation_task_id for row in manifest.task_assignments}
    if assigned != set(tasks):
        raise DataError("assignment manifest task/package coverage mismatch")
    return tasks


def _natural_leakage_groups(path: Path) -> dict[str, str]:
    if not path.exists():
        raise DataError(f"natural sample linkage file not found: {path}")
    raws = []
    for row in read_records(path):
        passages = row.get("passages")
        if not isinstance(passages, list):
            raise DataError("natural sample linkage row has no passages")
        raws.append(
            {
                "question_id": row["question_id"],
                "type": row["qtype"],
                "context": [[p["title"], p["sentences"]] for p in passages],
            }
        )
    return {qid: group.group_id for group in build_split_groups(raws) for qid in group.question_ids}


def load_eligible_challenge_examples(cfg: AppConfig) -> EligibleChallengeSet:
    """Validate human eligibility and source/task bindings before any stage writes."""
    execution = cfg.execution
    if execution.track != "challenge":
        raise DataError("eligible challenge loader requires execution.track=challenge")
    assert execution.phase is not None
    assert execution.variant is not None
    assert execution.challenge_records_path is not None
    assert execution.assignment_manifest_path is not None
    assert execution.eligibility_path is not None
    assert execution.natural_samples_path is not None
    try:
        manifest = AssignmentManifest.model_validate(
            read_json(Path(execution.assignment_manifest_path))
        )
        eligibility = EligibilityArtifact.model_validate(
            read_json(Path(execution.eligibility_path))
        )
    except ValidationError as exc:
        raise DataError(f"invalid human-review artifact: {exc}") from exc
    if eligibility.phase != execution.phase:
        raise DataError("eligibility phase does not match challenge execution phase")
    if execution.phase == "confirmatory" and (
        "frozen" not in eligibility.protocol_version.casefold()
        or "draft" in eligibility.protocol_version.casefold()
    ):
        raise DataError("confirmatory execution requires a frozen, non-draft protocol")

    tasks = _task_index(manifest)
    task_by_challenge = {}
    for task in tasks.values():
        challenge_id = task.challenge_id
        if challenge_id in task_by_challenge:
            raise DataError("challenge has more than one assignment task")
        task_by_challenge[challenge_id] = task
    eligibility_by_id = {row.challenge_id: row for row in eligibility.records if row.eligible}
    if not eligibility_by_id:
        raise DataError("no human-eligible challenge records; refusing to create outputs")

    source_path = Path(execution.challenge_records_path)
    if not source_path.exists():
        raise DataError(f"challenge source records not found: {source_path}")
    records = [ChallengeRecord.from_json(row) for row in read_records(source_path)]
    selected = [
        record
        for record in records
        if record.source_split == cfg.split
        and record.transformation == execution.variant
        and record.challenge_id in eligibility_by_id
    ]
    if not selected:
        raise DataError("no human-eligible rows match the configured split and variant")
    if len({record.challenge_id for record in selected}) != len(selected):
        raise DataError("duplicate challenge source records")

    leakage_groups = _natural_leakage_groups(Path(execution.natural_samples_path))
    metadata: dict[str, dict[str, str]] = {}
    examples: list[Example] = []
    eligibility_digest = artifact_hash(eligibility)
    for record in selected:
        row = eligibility_by_id[record.challenge_id]
        bound_task = task_by_challenge.get(record.challenge_id)
        if bound_task is None:
            raise DataError("eligible challenge has no blind assignment task")
        if bound_task.task_content_hash != row.task_content_hash:
            raise DataError("eligibility task content hash does not match assignment")
        validate_blind_task_source(bound_task, record)
        leakage_group = leakage_groups.get(record.parent_question_id)
        if leakage_group is None:
            raise DataError("challenge parent is missing from natural leakage linkage")
        examples.append(record.example)
        metadata[record.challenge_id] = {
            "track": "challenge",
            "phase": execution.phase,
            "variant": execution.variant,
            "parent_question_id": record.parent_question_id,
            "blinded_parent_group": bound_task.blinded_parent_group,
            "leakage_group": leakage_group,
            "human_answerability": row.final_answerability,
            "eligibility_artifact_sha256": eligibility_digest,
        }
    return EligibleChallengeSet(tuple(examples), metadata)
