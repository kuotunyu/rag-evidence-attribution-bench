"""Source-bound construction of Git-external, disjoint annotator delivery kits."""

from __future__ import annotations

import datetime as dt
import hashlib
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rag_evidence.annotation.package import scan_clean_package
from rag_evidence.errors import DataError
from rag_evidence.storage.artifacts import read_json, write_json_atomic

Hash64 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
CommitSha = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
WheelReproducibility = Literal["byte-identical", "per-build-hash-verified"]

_SOURCE_KEYS = frozenset(
    {
        "package_a",
        "package_b",
        "assignment_manifest",
        "instruction",
        "onboarding",
        "runbook",
        "launcher_a_powershell",
        "launcher_a_posix",
        "launcher_b_powershell",
        "launcher_b_posix",
        "dependency_lock",
        "builder",
        "manifest_schema",
    }
)
_KIT_A_LAYOUT = (
    "<verified-wheel>",
    "ann-pilot-a.json",
    "ONBOARDING.md",
    "HANDOFF_RUNBOOK.md",
    "start.ps1",
    "start.sh",
    "SHA256SUMS",
)
_KIT_B_LAYOUT = (
    "<verified-wheel>",
    "ann-pilot-b.json",
    "ONBOARDING.md",
    "HANDOFF_RUNBOOK.md",
    "start.ps1",
    "start.sh",
    "SHA256SUMS",
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SupportedEnvironment(_StrictModel):
    python: Literal["3.11"] = "3.11"
    operating_systems: tuple[Literal["Windows", "Linux"], ...] = ("Windows", "Linux")
    install_extra: Literal["app"] = "app"
    dependency_lock: Literal["uv.lock"] = "uv.lock"


class ExpectedLayout(_StrictModel):
    external_root_files: tuple[str, ...] = ("handoff-receipt.json", "SHA256SUMS")
    kit_a_directory: Literal["kit-a"] = "kit-a"
    kit_b_directory: Literal["kit-b"] = "kit-b"
    kit_a_files: tuple[str, ...] = _KIT_A_LAYOUT
    kit_b_files: tuple[str, ...] = _KIT_B_LAYOUT

    @model_validator(mode="after")
    def _fixed_layout(self) -> ExpectedLayout:
        if self.external_root_files != ("handoff-receipt.json", "SHA256SUMS"):
            raise ValueError("external root layout is fixed")
        if self.kit_a_files != _KIT_A_LAYOUT or self.kit_b_files != _KIT_B_LAYOUT:
            raise ValueError("annotator kit layouts are fixed")
        return self


class ReproducibleBuildRecipe(_StrictModel):
    source_date_epoch: int = Field(ge=315532800)
    build_command: tuple[str, ...]
    verification_steps: tuple[str, ...]
    wheel_policy: Literal["byte-compare-else-per-build-hash"] = (
        "byte-compare-else-per-build-hash"
    )


class HandoffSpec(_StrictModel):
    schema_version: Literal["handoff-manifest-v1"] = "handoff-manifest-v1"
    spec_version: Literal["handoff-spec-v1"] = "handoff-spec-v1"
    builder_version: Literal["handoff-builder-v1"] = "handoff-builder-v1"
    supported_environment: SupportedEnvironment
    source_paths: dict[str, str]
    canonical_sha256: dict[str, Hash64]
    instruction_version: Literal["pilot-v0.2.1-draft"]
    instruction_sha256: Hash64
    expected_layout: ExpectedLayout
    reproducible_build: ReproducibleBuildRecipe

    @field_validator("source_paths")
    @classmethod
    def _relative_source_paths(cls, value: dict[str, str]) -> dict[str, str]:
        for logical_name, raw_path in value.items():
            path = Path(raw_path)
            if path.is_absolute() or ".." in path.parts or path.as_posix() != raw_path:
                raise ValueError(f"source path {logical_name} must be normalized and relative")
        return value

    @model_validator(mode="after")
    def _complete_source_contract(self) -> HandoffSpec:
        if set(self.source_paths) != _SOURCE_KEYS:
            raise ValueError("source_paths does not contain the fixed handoff source set")
        if set(self.canonical_sha256) != _SOURCE_KEYS:
            raise ValueError("canonical_sha256 does not contain the fixed handoff source set")
        if self.canonical_sha256["instruction"] != self.instruction_sha256:
            raise ValueError("instruction hash must match the canonical instruction source")
        return self


class CleanInstallVerification(_StrictModel):
    schema_version: Literal["clean-install-verification-v1"] = (
        "clean-install-verification-v1"
    )
    python_version: str = Field(pattern=r"^3\.11(?:\.\d+)?$")
    kit_a_passed: bool
    kit_b_passed: bool
    cli_help_passed: bool
    app_creation_passed: bool

    @property
    def passed(self) -> bool:
        return all(
            (
                self.kit_a_passed,
                self.kit_b_passed,
                self.cli_help_passed,
                self.app_creation_passed,
            )
        )


class WheelReceipt(_StrictModel):
    filename: str = Field(pattern=r"^[A-Za-z0-9_.-]+\.whl$")
    byte_size: int = Field(gt=0)
    sha256: Hash64
    reproducibility: WheelReproducibility


class _BuildIdentity(_StrictModel):
    source_commit_sha: CommitSha
    build_time_utc: dt.datetime
    wheel_reproducibility: WheelReproducibility

    @field_validator("build_time_utc")
    @classmethod
    def _explicit_utc(cls, value: dt.datetime) -> dt.datetime:
        if value.tzinfo is None or value.utcoffset() != dt.timedelta(0):
            raise ValueError("build_time_utc must be an explicit UTC timestamp")
        return value


class HandoffReceipt(_StrictModel):
    schema_version: Literal["handoff-receipt-v1"] = "handoff-receipt-v1"
    builder_version: Literal["handoff-builder-v1"] = "handoff-builder-v1"
    source_commit_sha: CommitSha
    spec_sha256: Hash64
    builder_sha256: Hash64
    wheel: WheelReceipt
    package_sha256: dict[str, Hash64]
    assignment_manifest_sha256: Hash64
    runbook_sha256: Hash64
    onboarding_sha256: Hash64
    launcher_sha256: dict[str, Hash64]
    dependency_lock_sha256: Hash64
    instruction_version: Literal["pilot-v0.2.1-draft"]
    instruction_sha256: Hash64
    build_time_utc: dt.datetime
    clean_install: CleanInstallVerification
    kit_sha256sums: dict[str, Hash64]

    @field_validator("build_time_utc")
    @classmethod
    def _explicit_utc(cls, value: dt.datetime) -> dt.datetime:
        if value.tzinfo is None or value.utcoffset() != dt.timedelta(0):
            raise ValueError("build_time_utc must be an explicit UTC timestamp")
        return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_root(spec_path: Path) -> Path:
    try:
        return spec_path.resolve().parents[2]
    except IndexError as exc:
        raise DataError("handoff manifest must live at pilot/v0.2/handoff-manifest.json") from exc


def _resolve_source(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise DataError("handoff source path escapes the repository")
    return path


def _checksum_lines(root: Path, files: Mapping[str, Path]) -> str:
    return "".join(
        f"{_sha256_file(path)}  {relative_name}\n"
        for relative_name, path in sorted(files.items())
    )


def _copy_kit(
    *,
    kit_root: Path,
    wheel_path: Path,
    package_path: Path,
    onboarding_path: Path,
    runbook_path: Path,
    powershell_launcher: Path,
    posix_launcher: Path,
) -> str:
    kit_root.mkdir(parents=True)
    destinations = {
        wheel_path.name: wheel_path,
        package_path.name: package_path,
        "ONBOARDING.md": onboarding_path,
        "HANDOFF_RUNBOOK.md": runbook_path,
        "start.ps1": powershell_launcher,
        "start.sh": posix_launcher,
    }
    copied: dict[str, Path] = {}
    for name, source in destinations.items():
        destination = kit_root / name
        shutil.copyfile(source, destination)
        copied[name] = destination
    checksum_text = _checksum_lines(kit_root, copied)
    (kit_root / "SHA256SUMS").write_text(checksum_text, encoding="utf-8", newline="\n")
    return _sha256_file(kit_root / "SHA256SUMS")


def build_handoff_spec(repository_root: Path) -> HandoffSpec:
    """Build the decision-free committed contract from canonical repository sources."""
    paths = {
        "package_a": "pilot/v0.2/packages/ann-pilot-a.json",
        "package_b": "pilot/v0.2/packages/ann-pilot-b.json",
        "assignment_manifest": "pilot/v0.2/packages/manifest.json",
        "instruction": "PILOT_PROTOCOL.md",
        "onboarding": "pilot/v0.2/ONBOARDING.md",
        "runbook": "pilot/v0.2/HANDOFF_RUNBOOK.md",
        "launcher_a_powershell": "pilot/v0.2/launchers/start-a.ps1",
        "launcher_a_posix": "pilot/v0.2/launchers/start-a.sh",
        "launcher_b_powershell": "pilot/v0.2/launchers/start-b.ps1",
        "launcher_b_posix": "pilot/v0.2/launchers/start-b.sh",
        "dependency_lock": "uv.lock",
        "builder": "src/rag_evidence/annotation/handoff.py",
        "manifest_schema": "pilot/v0.2/handoff-manifest.schema.json",
    }
    hashes: dict[str, str] = {}
    for logical_name, relative_path in paths.items():
        source = _resolve_source(repository_root, relative_path)
        if not source.is_file():
            raise DataError(f"handoff source is missing: {logical_name}")
        hashes[logical_name] = _sha256_file(source)
    return HandoffSpec(
        supported_environment=SupportedEnvironment(),
        source_paths=paths,
        canonical_sha256=hashes,
        instruction_version="pilot-v0.2.1-draft",
        instruction_sha256=hashes["instruction"],
        expected_layout=ExpectedLayout(),
        reproducible_build=ReproducibleBuildRecipe(
            source_date_epoch=1787443200,
            build_command=(
                "uv",
                "build",
                "--wheel",
                "--out-dir",
                "<external-dist>",
            ),
            verification_steps=(
                "build the exact clean commit in two different temporary directories",
                "compare canonical package bytes and SHA-256 values",
                "compare wheel bytes and SHA-256 values",
                "diagnose ZIP timestamps, build metadata, and dependencies if wheels differ",
                "record per-build-hash-verified if reasonable fixes do not make wheels identical",
                "clean-install and create the annotation app with Python 3.11 in both builds",
            ),
        ),
    )


def build_handoff(
    *,
    spec_path: Path,
    wheel_path: Path,
    external_root: Path,
    source_commit: str,
    build_time: str,
    clean_install: CleanInstallVerification,
    wheel_reproducibility: WheelReproducibility,
) -> HandoffReceipt:
    """Verify committed sources and create two Git-external, mutually isolated kits."""
    identity = _BuildIdentity(
        source_commit_sha=source_commit,
        build_time_utc=build_time,
        wheel_reproducibility=wheel_reproducibility,
    )
    spec = HandoffSpec.model_validate(read_json(spec_path))
    source_root = _source_root(spec_path)
    resolved_external = external_root.resolve()
    if resolved_external.is_relative_to(source_root.resolve()):
        raise DataError("handoff output must remain outside the repository")
    if external_root.exists() and any(external_root.iterdir()):
        raise DataError("handoff output directory must be absent or empty")
    if not wheel_path.is_file() or wheel_path.suffix.casefold() != ".whl":
        raise DataError("handoff wheel must be one existing .whl file")
    if wheel_path.resolve().is_relative_to(source_root.resolve()):
        raise DataError("handoff wheel must remain outside the repository")
    if not clean_install.passed:
        raise DataError("clean-install verification must pass before handoff construction")

    sources: dict[str, Path] = {}
    for logical_name, relative_path in spec.source_paths.items():
        source = _resolve_source(source_root, relative_path)
        if not source.is_file():
            raise DataError(f"handoff source is missing: {logical_name}")
        actual = _sha256_file(source)
        if actual != spec.canonical_sha256[logical_name]:
            raise DataError(f"handoff source hash mismatch: {logical_name}")
        sources[logical_name] = source
    if _sha256_file(sources["instruction"]) != spec.instruction_sha256:
        raise DataError("handoff instruction hash mismatch")
    scan_clean_package(sources["package_a"].parent)

    external_root.mkdir(parents=True, exist_ok=True)
    kit_a_sum = _copy_kit(
        kit_root=external_root / spec.expected_layout.kit_a_directory,
        wheel_path=wheel_path,
        package_path=sources["package_a"],
        onboarding_path=sources["onboarding"],
        runbook_path=sources["runbook"],
        powershell_launcher=sources["launcher_a_powershell"],
        posix_launcher=sources["launcher_a_posix"],
    )
    kit_b_sum = _copy_kit(
        kit_root=external_root / spec.expected_layout.kit_b_directory,
        wheel_path=wheel_path,
        package_path=sources["package_b"],
        onboarding_path=sources["onboarding"],
        runbook_path=sources["runbook"],
        powershell_launcher=sources["launcher_b_powershell"],
        posix_launcher=sources["launcher_b_posix"],
    )
    receipt = HandoffReceipt(
        source_commit_sha=identity.source_commit_sha,
        spec_sha256=_sha256_file(spec_path),
        builder_sha256=spec.canonical_sha256["builder"],
        wheel=WheelReceipt(
            filename=wheel_path.name,
            byte_size=wheel_path.stat().st_size,
            sha256=_sha256_file(wheel_path),
            reproducibility=identity.wheel_reproducibility,
        ),
        package_sha256={
            "ann-pilot-a.json": spec.canonical_sha256["package_a"],
            "ann-pilot-b.json": spec.canonical_sha256["package_b"],
        },
        assignment_manifest_sha256=spec.canonical_sha256["assignment_manifest"],
        runbook_sha256=spec.canonical_sha256["runbook"],
        onboarding_sha256=spec.canonical_sha256["onboarding"],
        launcher_sha256={
            "kit-a/start.ps1": spec.canonical_sha256["launcher_a_powershell"],
            "kit-a/start.sh": spec.canonical_sha256["launcher_a_posix"],
            "kit-b/start.ps1": spec.canonical_sha256["launcher_b_powershell"],
            "kit-b/start.sh": spec.canonical_sha256["launcher_b_posix"],
        },
        dependency_lock_sha256=spec.canonical_sha256["dependency_lock"],
        instruction_version=spec.instruction_version,
        instruction_sha256=spec.instruction_sha256,
        build_time_utc=identity.build_time_utc,
        clean_install=clean_install,
        kit_sha256sums={"kit-a": kit_a_sum, "kit-b": kit_b_sum},
    )
    receipt_path = external_root / "handoff-receipt.json"
    write_json_atomic(receipt_path, receipt.model_dump(mode="json"))
    root_files = {
        "handoff-receipt.json": receipt_path,
        **{
            path.relative_to(external_root).as_posix(): path
            for kit_name in ("kit-a", "kit-b")
            for path in sorted((external_root / kit_name).rglob("*"))
            if path.is_file()
        },
    }
    (external_root / "SHA256SUMS").write_text(
        _checksum_lines(external_root, root_files),
        encoding="utf-8",
        newline="\n",
    )
    return receipt
