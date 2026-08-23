"""Deterministic, dual-independent annotation assignment."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from rag_evidence.annotation.blinding import scan_blind_payload
from rag_evidence.annotation.models import BlindTask
from rag_evidence.errors import DataError

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
