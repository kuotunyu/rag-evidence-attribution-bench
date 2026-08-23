"""Coordinator-only v2 mapping binds group-free package bytes."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import pytest

import rag_evidence.annotation.assignment as assignment
from rag_evidence.annotation.models import BlindTaskV2
from rag_evidence.errors import DataError


def _sha(payload: object) -> str:
    material = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _task(index: int) -> BlindTaskV2:
    content = {
        "challenge_id": f"ch-{index:024x}",
        "instruction_version": "pilot-v0.2.2-draft",
        "instruction_hash": "1" * 64,
        "question": f"Invented question {index}?",
        "passages": [
            {
                "alias": "P1",
                "title": f"Invented passage {index}",
                "sentences": [{"alias": "P1.S1", "text": f"Invented sentence {index}."}],
            }
        ],
    }
    return BlindTaskV2.model_validate(
        {
            "schema_version": "blind-task-v2",
            "annotation_task_id": f"task-{index:024x}",
            **content,
            "task_content_hash": _sha(content),
            "assignment_batch": "pilot-v0.2-smoke",
        }
    )


def _schedulable_tasks() -> tuple[object, ...]:
    model = getattr(assignment, "SchedulableTaskV2", None)
    assert model is not None, "SchedulableTaskV2 must be implemented"
    return tuple(
        model(
            task=_task(index),
            internal_group_id=f"coord-{index // 2:04d}",
            transformation="missing_hop" if index % 2 == 0 else "evidence_swap",
        )
        for index in range(40)
    )


def _build() -> tuple[tuple[object, ...], object]:
    function = getattr(assignment, "build_dual_assignments_v2", None)
    assert function is not None, "build_dual_assignments_v2 must be implemented"
    return function(
        _schedulable_tasks(),
        ("ann-pilot-a", "ann-pilot-b"),
        seed=20260823,
    )


def test_v2_packages_are_group_free_and_hash_bound_to_private_manifest() -> None:
    packages, manifest = _build()

    assert len(packages) == 2
    assert manifest.schema_version == "assignment-manifest-v2"
    assert len(manifest.tasks) == 40
    assert len(manifest.coordinator_tasks) == 40
    assert len({row.internal_group_id for row in manifest.coordinator_tasks}) == 20
    assert {row.transformation for row in manifest.coordinator_tasks} == {
        "missing_hop",
        "evidence_swap",
    }
    for package in packages:
        serialized = json.dumps(package.model_dump(mode="json"), sort_keys=True)
        assert "group" not in serialized.casefold()
        assert "transformation" not in serialized.casefold()
        filename = f"{package.annotator_pseudonym}.json"
        canonical = assignment.canonical_package_bytes(package)
        assert manifest.package_sha256[filename] == hashlib.sha256(canonical).hexdigest()


def test_v2_schedule_is_non_adjacent_for_both_annotators() -> None:
    packages, manifest = _build()
    groups = {
        row.annotation_task_id: row.internal_group_id for row in manifest.coordinator_tasks
    }

    for package in packages:
        ordered = [groups[task.annotation_task_id] for task in package.tasks]
        assert all(left != right for left, right in itertools.pairwise(ordered))


def test_pilot_validator_requires_exact_40_tasks_and_20_groups() -> None:
    _, manifest = _build()
    validator = getattr(assignment, "validate_pilot_manifest_v2", None)
    assert validator is not None, "validate_pilot_manifest_v2 must be implemented"
    validator(manifest)

    with pytest.raises(DataError, match="40 tasks"):
        validator(manifest.model_copy(update={"tasks": manifest.tasks[:-1]}))


def test_coordinator_manifest_writer_refuses_repository_output(tmp_path: Path) -> None:
    _, manifest = _build()
    writer = getattr(assignment, "write_coordinator_manifest_v2", None)
    assert writer is not None, "write_coordinator_manifest_v2 must be implemented"
    repository_root = tmp_path / "repository"
    repository_root.mkdir()

    with pytest.raises(DataError, match="outside"):
        writer(repository_root / "private.json", manifest, repository_root=repository_root)

    external = tmp_path / "external" / "assignment-manifest-v2.json"
    writer(external, manifest, repository_root=repository_root)
    assert json.loads(external.read_text(encoding="utf-8"))["schema_version"] == (
        "assignment-manifest-v2"
    )


def test_committed_manifest_schema_matches_model(repo_root: Path) -> None:
    committed = json.loads(
        (repo_root / "pilot/v0.2/assignment-manifest-v2.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert committed == assignment.AssignmentManifestV2.model_json_schema()
