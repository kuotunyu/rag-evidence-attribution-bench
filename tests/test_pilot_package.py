"""The pilot export is deterministic, blind, dual-assigned, and decision-free."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rag_evidence.annotation.assignment import AssignmentManifest, AssignmentPackage
from rag_evidence.annotation.package import build_pilot_package, scan_clean_package
from rag_evidence.config import load_config


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_real_shaped_pilot_package_has_20_parents_40_tasks_and_no_labels(
    repo_root: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(repo_root)
    out = tmp_path / "packages"
    cfg = load_config("configs/v2/smoke.yaml")

    manifest = build_pilot_package(cfg, out)
    scan = scan_clean_package(out)

    assert len(manifest.task_assignments) == 40
    assert len(manifest.packages) == 2
    tasks = {task.annotation_task_id: task for task in manifest.packages[0].tasks}
    assert len(tasks) == 40
    assert len({task.blinded_parent_group for task in tasks.values()}) == 20
    assert all(len(row.annotators) == 2 for row in manifest.task_assignments)
    assert all(len(package.tasks) == 40 for package in manifest.packages)
    assert scan == {"files": 3, "packages": 2, "tasks": 40, "decisions": 0}

    serialized = json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False)
    for hidden in (
        "expected_answerability",
        "transformation",
        "changed_fields",
        "provenance",
        "parent_question_id",
        "gold_passage_ids",
        "supporting_fact_sentence_ids",
        '"answer":',
    ):
        assert hidden not in serialized


def test_pilot_package_is_byte_deterministic_and_bound_to_protocol_hash(
    repo_root: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(repo_root)
    cfg = load_config("configs/v2/smoke.yaml")
    first = tmp_path / "first"
    second = tmp_path / "second"

    build_pilot_package(cfg, first)
    build_pilot_package(cfg, second)

    assert _tree_bytes(first) == _tree_bytes(second)
    package = AssignmentPackage.model_validate(
        json.loads((first / "ann-pilot-a.json").read_text(encoding="utf-8"))
    )
    assert package.instruction_version == "pilot-v0.2.1-draft"
    assert (
        package.instruction_hash
        == hashlib.sha256(Path("PILOT_PROTOCOL.md").read_bytes()).hexdigest()
    )
    saved = AssignmentManifest.model_validate(
        json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    )
    assert saved == build_pilot_package(cfg, tmp_path / "third")


def test_committed_packages_share_current_full_protocol_hash(repo_root: Path) -> None:
    protocol_hash = hashlib.sha256((repo_root / "PILOT_PROTOCOL.md").read_bytes()).hexdigest()
    package_root = repo_root / "pilot" / "v0.2" / "packages"

    for name in ("ann-pilot-a.json", "ann-pilot-b.json"):
        package = AssignmentPackage.model_validate(
            json.loads((package_root / name).read_text(encoding="utf-8"))
        )
        assert package.instruction_version == "pilot-v0.2.1-draft"
        assert package.instruction_hash == protocol_hash
