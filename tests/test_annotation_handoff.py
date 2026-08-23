"""External handoff kits are source-bound, disjoint, and never committed artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from rag_evidence.annotation.handoff import (
    CleanInstallVerification,
    HandoffSpec,
    build_handoff,
)
from rag_evidence.errors import DataError


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _kit_files(root: Path, kit: str) -> set[str]:
    return {
        path.relative_to(root / kit).as_posix()
        for path in (root / kit).rglob("*")
        if path.is_file()
    }


def _verify_sums(root: Path, sums_path: Path) -> None:
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", maxsplit=1)
        assert _sha256(root / relative) == expected


def test_committed_handoff_manifest_matches_schema_and_has_no_wheel_hash(
    repo_root: Path,
) -> None:
    schema_path = repo_root / "pilot" / "v0.2" / "handoff-manifest.schema.json"
    spec_path = repo_root / "pilot" / "v0.2" / "handoff-manifest.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    payload = json.loads(spec_path.read_text(encoding="utf-8"))

    assert schema == HandoffSpec.model_json_schema()
    spec = HandoffSpec.model_validate(payload)
    assert spec.schema_version == "handoff-manifest-v1"
    assert spec.spec_version == "handoff-spec-v1"
    assert spec.builder_version == "handoff-builder-v1"
    assert "wheel_sha256" not in json.dumps(payload).casefold()
    assert "wheel" not in payload["canonical_sha256"]


def test_committed_handoff_hashes_match_canonical_sources(repo_root: Path) -> None:
    spec = HandoffSpec.model_validate_json(
        (repo_root / "pilot" / "v0.2" / "handoff-manifest.json").read_text(encoding="utf-8")
    )

    for logical_name, expected_hash in spec.canonical_sha256.items():
        source = repo_root / spec.source_paths[logical_name]
        assert source.is_file(), logical_name
        assert _sha256(source) == expected_hash, logical_name


def test_handoff_kits_are_disjoint_and_receipt_is_complete(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    spec_path = repo_root / "pilot" / "v0.2" / "handoff-manifest.json"
    wheel = tmp_path / "dist" / "rag_evidence_attribution_bench-0.1.0-py3-none-any.whl"
    wheel.parent.mkdir()
    wheel.write_bytes(b"synthetic wheel bytes for a builder contract test")
    external = tmp_path / "external"
    verification = CleanInstallVerification(
        python_version="3.11.9",
        kit_a_passed=True,
        kit_b_passed=True,
        cli_help_passed=True,
        app_creation_passed=True,
    )

    result = build_handoff(
        spec_path=spec_path,
        wheel_path=wheel,
        external_root=external,
        source_commit="1" * 40,
        build_time="2026-08-23T04:00:00Z",
        clean_install=verification,
        wheel_reproducibility="byte-identical",
    )

    common = {
        wheel.name,
        "ONBOARDING.md",
        "HANDOFF_RUNBOOK.md",
        "start.ps1",
        "start.sh",
        "SHA256SUMS",
    }
    assert _kit_files(external, "kit-a") == common | {"ann-pilot-a.json"}
    assert _kit_files(external, "kit-b") == common | {"ann-pilot-b.json"}
    assert "ann-pilot-b.json" not in _kit_files(external, "kit-a")
    assert "ann-pilot-a.json" not in _kit_files(external, "kit-b")
    assert not any(
        "manifest" in name.casefold()
        for name in _kit_files(external, "kit-a") | _kit_files(external, "kit-b")
    )
    assert (external / "kit-a" / wheel.name).read_bytes() == wheel.read_bytes()
    assert (external / "kit-b" / wheel.name).read_bytes() == wheel.read_bytes()
    assert (external / "handoff-receipt.json").exists()
    assert (external / "SHA256SUMS").exists()
    assert result.source_commit_sha == "1" * 40
    assert result.wheel.sha256 == _sha256(wheel)
    assert result.wheel.reproducibility == "byte-identical"
    assert result.clean_install == verification
    assert result.instruction_version == "pilot-v0.2.1-draft"
    assert result.dependency_lock_sha256 == _sha256(repo_root / "uv.lock")
    _verify_sums(external / "kit-a", external / "kit-a" / "SHA256SUMS")
    _verify_sums(external / "kit-b", external / "kit-b" / "SHA256SUMS")
    _verify_sums(external, external / "SHA256SUMS")
    assert set(path.name for path in external.iterdir()) == {
        "kit-a",
        "kit-b",
        "handoff-receipt.json",
        "SHA256SUMS",
    }


def test_handoff_builder_refuses_repository_output_and_source_hash_mismatch(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    spec_path = repo_root / "pilot" / "v0.2" / "handoff-manifest.json"
    wheel = tmp_path / "artifact.whl"
    wheel.write_bytes(b"wheel")
    verification = CleanInstallVerification(
        python_version="3.11.9",
        kit_a_passed=True,
        kit_b_passed=True,
        cli_help_passed=True,
        app_creation_passed=True,
    )
    kwargs = {
        "spec_path": spec_path,
        "wheel_path": wheel,
        "source_commit": "2" * 40,
        "build_time": "2026-08-23T04:00:00Z",
        "clean_install": verification,
        "wheel_reproducibility": "per-build-hash-verified",
    }

    with pytest.raises(DataError, match="outside the repository"):
        build_handoff(external_root=repo_root / "pilot" / "v0.2" / "generated", **kwargs)

    spec = HandoffSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))
    copied_root = tmp_path / "copied-source"
    for relative_path in spec.source_paths.values():
        destination = copied_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo_root / relative_path, destination)
    copied_spec = copied_root / "pilot" / "v0.2" / "handoff-manifest.json"
    copied_spec.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(spec_path, copied_spec)
    copied_package = copied_root / spec.source_paths["package_a"]
    copied_package.write_bytes(copied_package.read_bytes() + b"\n")

    with pytest.raises(DataError, match="source hash mismatch"):
        build_handoff(
            external_root=tmp_path / "external",
            **{**kwargs, "spec_path": copied_spec},
        )
