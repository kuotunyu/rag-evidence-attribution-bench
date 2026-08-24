"""Dual assignment is complete, deterministic, and sibling-safe."""

from __future__ import annotations

import json
from itertools import pairwise

import pytest

from rag_evidence.annotation.assignment import build_dual_assignments
from rag_evidence.annotation.blinding import project_challenge, scan_blind_payload
from rag_evidence.errors import DataError
from test_annotation_blinding import challenge_record


def _tasks(n_parents: int = 5):
    tasks = []
    counter = 1
    for parent in range(n_parents):
        for transform in ("missing_hop", "evidence_swap"):
            challenge_id = f"ch-{counter:024x}"
            tasks.append(
                project_challenge(
                    challenge_record(
                        challenge_id=challenge_id,
                        parent_id=f"parent-secret-{parent}",
                        transformation=transform,
                    ),
                    instruction_version="pilot-v0.2-draft",
                    instruction_hash="1" * 64,
                    batch="pilot-batch-01",
                    namespace="pilot-v0.2",
                )
            )
            counter += 1
    return tasks


def test_every_task_has_two_independent_annotators_and_no_adjacent_siblings() -> None:
    manifest = build_dual_assignments(
        _tasks(),
        annotators=("ann-r7", "ann-k2", "ann-m4"),
        seed=20260823,
    )

    coverage: dict[str, list[str]] = {}
    for package in manifest.packages:
        groups = [task.blinded_parent_group for task in package.tasks]
        assert all(left != right for left, right in pairwise(groups))
        assert scan_blind_payload(package.blind_export()) == ()
        for task in package.tasks:
            coverage.setdefault(task.annotation_task_id, []).append(package.annotator_pseudonym)

    assert set(coverage) == {task.annotation_task_id for task in _tasks()}
    assert all(len(annotators) == 2 for annotators in coverage.values())
    assert all(len(set(annotators)) == 2 for annotators in coverage.values())


def test_assignment_is_byte_deterministic_and_seed_changes_order() -> None:
    tasks = _tasks()
    first = build_dual_assignments(tasks, ("ann-r7", "ann-k2"), seed=19)
    repeated = build_dual_assignments(list(reversed(tasks)), ("ann-k2", "ann-r7"), seed=19)
    other_seed = build_dual_assignments(tasks, ("ann-r7", "ann-k2"), seed=20)

    def serialized(manifest) -> str:
        return json.dumps(manifest.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    assert serialized(first) == serialized(repeated)
    assert [task.annotation_task_id for task in first.packages[0].tasks] != [
        task.annotation_task_id for task in other_seed.packages[0].tasks
    ]


def test_assignment_rejects_duplicate_tasks_and_too_few_annotators() -> None:
    tasks = _tasks()
    with pytest.raises(DataError, match="duplicate"):
        build_dual_assignments([tasks[0], tasks[0]], ("ann-r7", "ann-k2"), seed=1)
    with pytest.raises(DataError, match="at least two"):
        build_dual_assignments(tasks, ("ann-r7",), seed=1)
    with pytest.raises(DataError, match="duplicate annotator"):
        build_dual_assignments(tasks, ("ann-r7", "ann-r7"), seed=1)


def test_assignment_fails_when_sibling_non_adjacency_is_impossible() -> None:
    with pytest.raises(DataError, match="non-adjacent"):
        build_dual_assignments(_tasks(n_parents=1), ("ann-r7", "ann-k2"), seed=1)
