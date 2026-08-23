"""The v2 handoff contract is source-only, source-bound, and delivery-disjoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_evidence.annotation.handoff import (
    HandoffSpecV2,
    resolve_handoff_sources,
    stage_disjoint_kits,
)
from rag_evidence.errors import DataError


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}


def _verify_sums(root: Path) -> None:
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", maxsplit=1)
        assert _sha256(root / relative) == expected


def _committed_spec(repo_root: Path) -> HandoffSpecV2:
    path = repo_root / "pilot/v0.2/handoff-manifest.json"
    return HandoffSpecV2.model_validate_json(path.read_text(encoding="utf-8"))


def test_committed_handoff_manifest_is_strict_source_only_v2(repo_root: Path) -> None:
    schema_path = repo_root / "pilot/v0.2/handoff-manifest.schema.json"
    spec_path = repo_root / "pilot/v0.2/handoff-manifest.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    payload = json.loads(spec_path.read_text(encoding="utf-8"))

    assert schema == HandoffSpecV2.model_json_schema()
    spec = HandoffSpecV2.model_validate(payload)
    assert spec.schema_version == "handoff-manifest-v2"
    assert spec.spec_version == "handoff-spec-v2"
    assert spec.builder_version == "handoff-builder-v2"
    assert spec.protocol_version == "pilot-v0.2.2-draft"
    assert spec.python_distribution == "0.2.0.dev0"
    assert spec.required_verification_platforms == ("Windows", "Linux")
    serialized = json.dumps(payload).casefold()
    assert "wheel_sha256" not in serialized
    assert "assignment_manifest" not in serialized
    assert "coordinator_manifest_instance" not in serialized
    assert "verified_platforms" not in serialized


def test_committed_handoff_hashes_match_all_declared_sources(repo_root: Path) -> None:
    spec = _committed_spec(repo_root)
    sources = resolve_handoff_sources(repo_root, spec)

    assert set(sources) == set(spec.source_paths) == set(spec.canonical_sha256)
    for logical_name, source in sources.items():
        assert _sha256(source) == spec.canonical_sha256[logical_name]
    assert spec.instruction_sha256 == spec.canonical_sha256["instruction"]
    assert spec.dependency_lock_sha256 == spec.canonical_sha256["dependency_lock"]
    assert spec.canonical_package_sha256 == {
        "ann-pilot-a.json": spec.canonical_sha256["package_a"],
        "ann-pilot-b.json": spec.canonical_sha256["package_b"],
    }


def test_v1_handoff_manifest_is_rejected_without_migration(repo_root: Path) -> None:
    payload = json.loads(
        (repo_root / "pilot/v0.2/handoff-manifest.json").read_text(encoding="utf-8")
    )
    payload["schema_version"] = "handoff-manifest-v1"

    with pytest.raises(ValidationError):
        HandoffSpecV2.model_validate(payload)


def test_stage_disjoint_kits_uses_closed_layout_and_own_package_only(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    spec = _committed_spec(repo_root)
    sources = resolve_handoff_sources(repo_root, spec)
    wheel = tmp_path / "wheel/rag_evidence_attribution_bench-0.2.0.dev0-py3-none-any.whl"
    wheel.parent.mkdir()
    wheel.write_bytes(b"verified wheel fixture")
    external = tmp_path / "delivery"

    kit_hashes = stage_disjoint_kits(external, wheel, sources, spec)

    common = {
        wheel.name,
        "annotation-requirements-py311.lock",
        "ONBOARDING.md",
        "HANDOFF_RUNBOOK.md",
        "start.ps1",
        "start.sh",
        "SHA256SUMS",
    }
    assert _files(external / "kit-a") == common | {"ann-pilot-a.json"}
    assert _files(external / "kit-b") == common | {"ann-pilot-b.json"}
    assert "ann-pilot-b.json" not in _files(external / "kit-a")
    assert "ann-pilot-a.json" not in _files(external / "kit-b")
    assert not any("manifest" in name.casefold() for name in _files(external))
    assert set(kit_hashes) == {"kit-a", "kit-b"}
    _verify_sums(external / "kit-a")
    _verify_sums(external / "kit-b")


def test_stage_disjoint_kits_refuses_repository_output(repo_root: Path, tmp_path: Path) -> None:
    spec = _committed_spec(repo_root)
    sources = resolve_handoff_sources(repo_root, spec)
    wheel = tmp_path / "rag_evidence_attribution_bench-0.2.0.dev0-py3-none-any.whl"
    wheel.write_bytes(b"verified wheel fixture")

    with pytest.raises(DataError, match="outside"):
        stage_disjoint_kits(repo_root / "pilot/v0.2/delivery", wheel, sources, spec)
