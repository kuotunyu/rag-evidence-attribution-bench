"""Deterministic, dual-independent annotation assignment."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rag_evidence.annotation.blinding import scan_blind_payload
from rag_evidence.annotation.models import BlindTask, BlindTaskV2, Hash64
from rag_evidence.annotation.privacy import scan_delivery_payload
from rag_evidence.errors import DataError
from rag_evidence.storage.artifacts import write_json_atomic

_PSEUDONYM_RE = re.compile(r"ann-[a-z0-9](?:[a-z0-9-]{0,30}[a-z0-9])?")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AssignmentPackage(_StrictModel):
    schema_version: Literal["assignment-package-v1"] = "assignment-package-v1"
    annotator_pseudonym: str
    instruction_version: str
    instruction_hash: str
    assignment_batch: str
    tasks: tuple[BlindTask, ...]

    @field_validator("annotator_pseudonym")
    @classmethod
    def _pseudonym(cls, value: str) -> str:
        if _PSEUDONYM_RE.fullmatch(value) is None:
            raise ValueError("annotator pseudonym must be an opaque ann-* identifier")
        return value

    @model_validator(mode="after")
    def _package_bindings(self) -> AssignmentPackage:
        for task in self.tasks:
            if task.instruction_version != self.instruction_version:
                raise ValueError("package/task instruction version mismatch")
            if task.instruction_hash != self.instruction_hash:
                raise ValueError("package/task instruction hash mismatch")
            if task.assignment_batch != self.assignment_batch:
                raise ValueError("package/task batch mismatch")
        return self

    def blind_export(self) -> dict[str, object]:
        payload: dict[str, object] = self.model_dump(mode="json")
        violations = scan_blind_payload(payload)
        if violations:
            raise DataError("assignment package leak: " + "; ".join(violations))
        return payload


class TaskAssignment(_StrictModel):
    annotation_task_id: str
    annotators: tuple[str, str]

    @model_validator(mode="after")
    def _independent(self) -> TaskAssignment:
        if self.annotators[0] == self.annotators[1]:
            raise ValueError("task requires two independent annotators")
        return self


class AssignmentManifest(_StrictModel):
    schema_version: Literal["assignment-manifest-v1"] = "assignment-manifest-v1"
    seed: int
    task_assignments: tuple[TaskAssignment, ...]
    packages: tuple[AssignmentPackage, ...]


def _derived_seed(seed: int, *parts: str) -> int:
    material = json.dumps([seed, *parts], separators=(",", ":"), ensure_ascii=False)
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")


def _schedule(tasks: Sequence[BlindTask], *, seed: int, annotator: str) -> tuple[BlindTask, ...]:
    rng = random.Random(_derived_seed(seed, annotator))
    by_group: dict[str, list[BlindTask]] = defaultdict(list)
    for task in sorted(tasks, key=lambda item: item.annotation_task_id):
        by_group[task.blinded_parent_group].append(task)
    for group_tasks in by_group.values():
        rng.shuffle(group_tasks)
    ordered: list[BlindTask] = []
    while any(by_group.values()):
        last_group = ordered[-1].blinded_parent_group if ordered else None
        candidates = [
            group for group, group_tasks in by_group.items() if group_tasks and group != last_group
        ]
        if not candidates:
            raise DataError(f"cannot build non-adjacent sibling schedule for annotator {annotator}")
        max_remaining = max(len(by_group[group]) for group in candidates)
        largest = [group for group in candidates if len(by_group[group]) == max_remaining]
        chosen_group = largest[rng.randrange(len(largest))]
        ordered.append(by_group[chosen_group].pop())
    return tuple(ordered)


def build_dual_assignments(
    tasks: Sequence[BlindTask],
    annotators: Sequence[str],
    *,
    seed: int,
) -> AssignmentManifest:
    """Assign every task to exactly two humans and randomize each safe task order."""
    canonical_tasks = sorted(tasks, key=lambda task: task.annotation_task_id)
    task_ids = [task.annotation_task_id for task in canonical_tasks]
    if len(set(task_ids)) != len(task_ids):
        raise DataError("duplicate annotation task ID")
    canonical_annotators = sorted(set(annotators))
    if len(canonical_annotators) != len(annotators):
        raise DataError("duplicate annotator pseudonym")
    if len(canonical_annotators) < 2:
        raise DataError("dual annotation requires at least two annotators")
    for pseudonym in canonical_annotators:
        if _PSEUDONYM_RE.fullmatch(pseudonym) is None:
            raise DataError(f"invalid annotator pseudonym: {pseudonym}")
    if not canonical_tasks:
        raise DataError("assignment requires at least one task")

    bindings = {
        (
            task.instruction_version,
            task.instruction_hash,
            task.assignment_batch,
        )
        for task in canonical_tasks
    }
    if len(bindings) != 1:
        raise DataError("all assigned tasks must share instruction and batch bindings")
    instruction_version, instruction_hash, assignment_batch = next(iter(bindings))

    annotator_pairs = list(itertools.combinations(canonical_annotators, 2))
    assigned_by_annotator: dict[str, list[BlindTask]] = defaultdict(list)
    assignments: list[TaskAssignment] = []
    for task in canonical_tasks:
        pair_index = _derived_seed(seed, task.annotation_task_id) % len(annotator_pairs)
        pair = annotator_pairs[pair_index]
        assignments.append(
            TaskAssignment(annotation_task_id=task.annotation_task_id, annotators=pair)
        )
        for annotator in pair:
            assigned_by_annotator[annotator].append(task)

    packages = tuple(
        AssignmentPackage(
            annotator_pseudonym=annotator,
            instruction_version=instruction_version,
            instruction_hash=instruction_hash,
            assignment_batch=assignment_batch,
            tasks=_schedule(assigned_by_annotator[annotator], seed=seed, annotator=annotator),
        )
        for annotator in canonical_annotators
        if assigned_by_annotator[annotator]
    )
    for package in packages:
        package.blind_export()
    return AssignmentManifest(
        seed=seed,
        task_assignments=tuple(assignments),
        packages=packages,
    )


class AssignmentPackageV2(_StrictModel):
    schema_version: Literal["assignment-package-v2"] = "assignment-package-v2"
    annotator_pseudonym: str
    instruction_version: str
    instruction_hash: Hash64
    assignment_batch: str
    tasks: tuple[BlindTaskV2, ...]

    @field_validator("annotator_pseudonym")
    @classmethod
    def _v2_pseudonym(cls, value: str) -> str:
        if _PSEUDONYM_RE.fullmatch(value) is None:
            raise ValueError("annotator pseudonym must be an opaque ann-* identifier")
        return value

    @model_validator(mode="after")
    def _v2_package_bindings(self) -> AssignmentPackageV2:
        if not self.tasks:
            raise ValueError("assignment package must contain at least one task")
        for task in self.tasks:
            if task.instruction_version != self.instruction_version:
                raise ValueError("package/task instruction version mismatch")
            if task.instruction_hash != self.instruction_hash:
                raise ValueError("package/task instruction hash mismatch")
            if task.assignment_batch != self.assignment_batch:
                raise ValueError("package/task batch mismatch")
        return self

    def blind_export(self) -> dict[str, object]:
        payload: dict[str, object] = self.model_dump(mode="json")
        violations = scan_delivery_payload(payload, artifact_kind="assignment_package")
        if violations:
            raise DataError("assignment package leak: " + "; ".join(violations))
        return payload


@dataclass(frozen=True)
class SchedulableTaskV2:
    task: BlindTaskV2
    internal_group_id: str
    transformation: Literal["missing_hop", "evidence_swap"]

    def __post_init__(self) -> None:
        if re.fullmatch(r"coord-[a-z0-9._-]+", self.internal_group_id) is None:
            raise ValueError("internal group ID must be a coordinator-only opaque identifier")


class CoordinatorTaskV2(_StrictModel):
    annotation_task_id: str = Field(pattern=r"^task-[0-9a-f]{24}$")
    internal_group_id: str = Field(pattern=r"^coord-[a-z0-9._-]+$")
    transformation: Literal["missing_hop", "evidence_swap"]
    annotators: tuple[str, str]
    delivery_positions: dict[str, int]

    @model_validator(mode="after")
    def _v2_coordinator_binding(self) -> CoordinatorTaskV2:
        if len(set(self.annotators)) != 2:
            raise ValueError("task requires two independent annotators")
        if set(self.delivery_positions) != set(self.annotators):
            raise ValueError("delivery positions must bind both assigned annotators")
        if any(position < 0 for position in self.delivery_positions.values()):
            raise ValueError("delivery positions must be nonnegative")
        return self


class AssignmentManifestV2(_StrictModel):
    schema_version: Literal["assignment-manifest-v2"] = "assignment-manifest-v2"
    seed: int
    instruction_version: str
    instruction_hash: Hash64
    assignment_batch: str
    tasks: tuple[BlindTaskV2, ...]
    coordinator_tasks: tuple[CoordinatorTaskV2, ...]
    package_sha256: dict[str, Hash64]

    @model_validator(mode="after")
    def _v2_manifest_bindings(self) -> AssignmentManifestV2:
        task_ids = [task.annotation_task_id for task in self.tasks]
        coordinator_ids = [row.annotation_task_id for row in self.coordinator_tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("manifest contains duplicate task IDs")
        if len(coordinator_ids) != len(set(coordinator_ids)):
            raise ValueError("manifest contains duplicate coordinator rows")
        if set(task_ids) != set(coordinator_ids):
            raise ValueError("manifest tasks and coordinator rows differ")
        return self


def canonical_package_bytes(package: AssignmentPackageV2) -> bytes:
    """Serialize canonical package source bytes exactly as committed."""
    text = json.dumps(
        package.blind_export(),
        ensure_ascii=False,
        indent=2,
        sort_keys=False,
    )
    return (text + "\n").encode("utf-8")


def _schedule_v2(
    tasks: Sequence[SchedulableTaskV2],
    *,
    seed: int,
    annotator: str,
) -> tuple[SchedulableTaskV2, ...]:
    rng = random.Random(_derived_seed(seed, annotator))
    by_group: dict[str, list[SchedulableTaskV2]] = defaultdict(list)
    for item in sorted(tasks, key=lambda value: value.task.annotation_task_id):
        by_group[item.internal_group_id].append(item)
    for group_tasks in by_group.values():
        rng.shuffle(group_tasks)
    ordered: list[SchedulableTaskV2] = []
    while any(by_group.values()):
        last_group = ordered[-1].internal_group_id if ordered else None
        candidates = [
            group for group, group_tasks in by_group.items() if group_tasks and group != last_group
        ]
        if not candidates:
            raise DataError(f"cannot build non-adjacent sibling schedule for annotator {annotator}")
        max_remaining = max(len(by_group[group]) for group in candidates)
        largest = [group for group in candidates if len(by_group[group]) == max_remaining]
        chosen_group = largest[rng.randrange(len(largest))]
        ordered.append(by_group[chosen_group].pop())
    return tuple(ordered)


def build_dual_assignments_v2(
    tasks: Sequence[SchedulableTaskV2],
    annotators: Sequence[str],
    *,
    seed: int,
) -> tuple[tuple[AssignmentPackageV2, ...], AssignmentManifestV2]:
    """Build group-free packages plus one coordinator-only mapping."""
    canonical_tasks = sorted(tasks, key=lambda item: item.task.annotation_task_id)
    if not canonical_tasks:
        raise DataError("assignment requires at least one task")
    task_ids = [item.task.annotation_task_id for item in canonical_tasks]
    if len(task_ids) != len(set(task_ids)):
        raise DataError("duplicate annotation task ID")
    canonical_annotators = sorted(set(annotators))
    if len(canonical_annotators) != len(annotators):
        raise DataError("duplicate annotator pseudonym")
    if len(canonical_annotators) < 2:
        raise DataError("dual annotation requires at least two annotators")
    if any(_PSEUDONYM_RE.fullmatch(value) is None for value in canonical_annotators):
        raise DataError("invalid annotator pseudonym")
    bindings = {
        (
            item.task.instruction_version,
            item.task.instruction_hash,
            item.task.assignment_batch,
        )
        for item in canonical_tasks
    }
    if len(bindings) != 1:
        raise DataError("all assigned tasks must share instruction and batch bindings")
    instruction_version, instruction_hash, assignment_batch = next(iter(bindings))

    annotator_pairs = list(itertools.combinations(canonical_annotators, 2))
    assigned: dict[str, list[SchedulableTaskV2]] = defaultdict(list)
    pairs: dict[str, tuple[str, str]] = {}
    for item in canonical_tasks:
        pair = annotator_pairs[
            _derived_seed(seed, item.task.annotation_task_id) % len(annotator_pairs)
        ]
        pairs[item.task.annotation_task_id] = pair
        for annotator in pair:
            assigned[annotator].append(item)

    scheduled = {
        annotator: _schedule_v2(items, seed=seed, annotator=annotator)
        for annotator, items in sorted(assigned.items())
    }
    packages = tuple(
        AssignmentPackageV2(
            annotator_pseudonym=annotator,
            instruction_version=instruction_version,
            instruction_hash=instruction_hash,
            assignment_batch=assignment_batch,
            tasks=tuple(item.task for item in scheduled[annotator]),
        )
        for annotator in sorted(scheduled)
    )
    positions = {
        (annotator, item.task.annotation_task_id): position
        for annotator, ordered in scheduled.items()
        for position, item in enumerate(ordered)
    }
    rows = tuple(
        CoordinatorTaskV2(
            annotation_task_id=item.task.annotation_task_id,
            internal_group_id=item.internal_group_id,
            transformation=item.transformation,
            annotators=pairs[item.task.annotation_task_id],
            delivery_positions={
                annotator: positions[(annotator, item.task.annotation_task_id)]
                for annotator in pairs[item.task.annotation_task_id]
            },
        )
        for item in canonical_tasks
    )
    package_hashes = {
        f"{package.annotator_pseudonym}.json": hashlib.sha256(
            canonical_package_bytes(package)
        ).hexdigest()
        for package in packages
    }
    manifest = AssignmentManifestV2(
        seed=seed,
        instruction_version=instruction_version,
        instruction_hash=instruction_hash,
        assignment_batch=assignment_batch,
        tasks=tuple(item.task for item in canonical_tasks),
        coordinator_tasks=rows,
        package_sha256=package_hashes,
    )
    return packages, manifest


def validate_pilot_manifest_v2(manifest: AssignmentManifestV2) -> None:
    """Apply the fixed 40-task feasibility-pilot constraints."""
    if len(manifest.tasks) != 40 or len(manifest.coordinator_tasks) != 40:
        raise DataError("pilot coordinator manifest must contain exactly 40 tasks")
    by_group: dict[str, set[str]] = defaultdict(set)
    for row in manifest.coordinator_tasks:
        by_group[row.internal_group_id].add(row.transformation)
        if len(set(row.annotators)) != 2:
            raise DataError("pilot tasks require two independent annotators")
    if len(by_group) != 20:
        raise DataError("pilot coordinator manifest must contain exactly 20 groups")
    expected_variants = {"missing_hop", "evidence_swap"}
    if any(variants != expected_variants for variants in by_group.values()):
        raise DataError("each pilot group must contain two different transformation variants")
    for annotator in sorted(
        {annotator for row in manifest.coordinator_tasks for annotator in row.annotators}
    ):
        rows = sorted(
            (row for row in manifest.coordinator_tasks if annotator in row.annotators),
            key=lambda row: row.delivery_positions[annotator],
        )
        if [row.delivery_positions[annotator] for row in rows] != list(range(len(rows))):
            raise DataError("pilot delivery positions must be contiguous")
        if any(
            left.internal_group_id == right.internal_group_id
            for left, right in itertools.pairwise(rows)
        ):
            raise DataError("pilot sibling variants must be non-adjacent")
    if set(manifest.package_sha256) != {"ann-pilot-a.json", "ann-pilot-b.json"}:
        raise DataError("pilot manifest must bind the canonical A/B package hashes")


def write_coordinator_manifest_v2(
    path: Path,
    manifest: AssignmentManifestV2,
    *,
    repository_root: Path,
) -> None:
    """Write a private manifest only outside the repository."""
    resolved_repository = repository_root.resolve()
    resolved_path = path.resolve()
    if resolved_path.is_relative_to(resolved_repository):
        raise DataError("coordinator manifest output must remain outside the repository")
    validate_pilot_manifest_v2(manifest)
    write_json_atomic(path, manifest.model_dump(mode="json"))
