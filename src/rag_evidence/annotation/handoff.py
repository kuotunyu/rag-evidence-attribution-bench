"""Source-bound v2 construction of Git-external, disjoint annotation kits."""

from __future__ import annotations

import datetime as dt
import hashlib
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rag_evidence.annotation.assignment import AssignmentManifestV2, validate_pilot_manifest_v2
from rag_evidence.annotation.platform import (
    PlatformIdentityV2,
    PlatformVerificationReceiptV2,
    verify_platform_receipt,
)
from rag_evidence.annotation.privacy import scan_delivery_payload
from rag_evidence.annotation.source_verify import (
    GitCommandRecordV2,
    VerifiedCheckoutV2,
    verify_checkout,
    verify_tracked_blob,
)
from rag_evidence.annotation.wheel_verify import (
    BUILD_ARGV,
    VerifiedWheelSetV2,
    WheelIdentityV2,
    WheelSourceSpecV2,
    require_byte_identity,
    verify_wheel_payload,
)
from rag_evidence.errors import DataError
from rag_evidence.storage.artifacts import read_json, write_json_atomic

Hash64 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
CommitSha = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
_SOURCE_PATHS = {
    "package_a": "pilot/v0.2/packages/ann-pilot-a.json",
    "package_b": "pilot/v0.2/packages/ann-pilot-b.json",
    "instruction": "PILOT_PROTOCOL.md",
    "onboarding": "pilot/v0.2/ONBOARDING.md",
    "runbook": "pilot/v0.2/HANDOFF_RUNBOOK.md",
    "launcher_a_powershell": "pilot/v0.2/launchers/start-a.ps1",
    "launcher_a_posix": "pilot/v0.2/launchers/start-a.sh",
    "launcher_b_powershell": "pilot/v0.2/launchers/start-b.ps1",
    "launcher_b_posix": "pilot/v0.2/launchers/start-b.sh",
    "dependency_lock": "pilot/v0.2/annotation-requirements-py311.lock",
    "builder": "src/rag_evidence/annotation/handoff.py",
    "source_verifier": "src/rag_evidence/annotation/source_verify.py",
    "wheel_verifier": "src/rag_evidence/annotation/wheel_verify.py",
    "platform_verifier": "src/rag_evidence/annotation/platform.py",
    "platform_script": "scripts/verify_annotation_platform.py",
    "manifest_schema": "pilot/v0.2/handoff-manifest.schema.json",
}
_COMMON_KIT_LAYOUT = (
    "<verified-wheel>",
    "annotation-requirements-py311.lock",
    "ONBOARDING.md",
    "HANDOFF_RUNBOOK.md",
    "start.ps1",
    "start.sh",
    "SHA256SUMS",
)
_KIT_A_LAYOUT = (*_COMMON_KIT_LAYOUT, "ann-pilot-a.json")
_KIT_B_LAYOUT = (*_COMMON_KIT_LAYOUT, "ann-pilot-b.json")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SupportedEnvironmentV2(_StrictModel):
    python: Literal["3.11"] = "3.11"
    required_verification_platforms: tuple[Literal["Windows", "Linux"], ...] = (
        "Windows",
        "Linux",
    )
    dependency_lock: Literal["annotation-requirements-py311.lock"] = (
        "annotation-requirements-py311.lock"
    )
    dependency_bootstrap_argv: tuple[str, ...] = (
        "python",
        "-m",
        "pip",
        "install",
        "--require-hashes",
        "--only-binary=:all:",
        "-r",
        "annotation-requirements-py311.lock",
    )
    wheel_install_argv: tuple[str, ...] = (
        "python",
        "-m",
        "pip",
        "install",
        "--no-deps",
        "<verified-wheel>",
    )

    @model_validator(mode="after")
    def _fixed_bootstrap(self) -> SupportedEnvironmentV2:
        if self.required_verification_platforms != ("Windows", "Linux"):
            raise ValueError("Windows and Linux are both required verification targets")
        if self.dependency_bootstrap_argv[4:6] != (
            "--require-hashes",
            "--only-binary=:all:",
        ):
            raise ValueError("dependency bootstrap must remain hash-locked and binary-only")
        if "--no-deps" not in self.wheel_install_argv:
            raise ValueError("verified project wheel must be installed with --no-deps")
        return self


class ExpectedLayoutV2(_StrictModel):
    external_root_files: tuple[str, ...] = ("handoff-receipt.json", "SHA256SUMS")
    kit_a_directory: Literal["kit-a"] = "kit-a"
    kit_b_directory: Literal["kit-b"] = "kit-b"
    kit_a_files: tuple[str, ...] = _KIT_A_LAYOUT
    kit_b_files: tuple[str, ...] = _KIT_B_LAYOUT

    @model_validator(mode="after")
    def _closed_layout(self) -> ExpectedLayoutV2:
        if self.external_root_files != ("handoff-receipt.json", "SHA256SUMS"):
            raise ValueError("external handoff root has one fixed file layout")
        if self.kit_a_files != _KIT_A_LAYOUT or self.kit_b_files != _KIT_B_LAYOUT:
            raise ValueError("annotator delivery kits have one fixed closed layout")
        return self


class ReproducibleBuildRecipeV2(_StrictModel):
    source_date_epoch: Literal[1787443200] = 1787443200
    build_command: tuple[str, ...] = (
        "uv",
        "build",
        "--wheel",
        "--out-dir",
        "<external-dist>",
    )
    wheel_policy: Literal["byte-identical-required"] = "byte-identical-required"
    required_instances: Literal[4] = 4
    environment_policy: tuple[str, ...] = (
        "external-cache",
        "external-build-root",
        "external-output",
        "external-pycache",
        "PYTHONDONTWRITEBYTECODE=1",
    )

    @model_validator(mode="after")
    def _fixed_recipe(self) -> ReproducibleBuildRecipeV2:
        if self.build_command != (*BUILD_ARGV, "<external-dist>"):
            raise ValueError("wheel build command must match the source-controlled builder")
        return self


class HandoffSpecV2(_StrictModel):
    schema_version: Literal["handoff-manifest-v2"] = "handoff-manifest-v2"
    spec_version: Literal["handoff-spec-v2"] = "handoff-spec-v2"
    builder_version: Literal["handoff-builder-v2"] = "handoff-builder-v2"
    protocol_version: Literal["pilot-v0.2.2-draft"] = "pilot-v0.2.2-draft"
    python_distribution: Literal["0.2.0"] = "0.2.0"
    required_verification_platforms: tuple[Literal["Windows", "Linux"], ...] = (
        "Windows",
        "Linux",
    )
    supported_environment: SupportedEnvironmentV2
    source_paths: dict[str, str]
    canonical_sha256: dict[str, Hash64]
    canonical_package_sha256: dict[str, Hash64]
    instruction_version: Literal["pilot-v0.2.2-draft"] = "pilot-v0.2.2-draft"
    instruction_sha256: Hash64
    dependency_lock_sha256: Hash64
    expected_layout: ExpectedLayoutV2
    reproducible_build: ReproducibleBuildRecipeV2

    @field_validator("source_paths")
    @classmethod
    def _normalized_relative_paths(cls, value: dict[str, str]) -> dict[str, str]:
        for logical_name, raw_path in value.items():
            path = Path(raw_path)
            if path.is_absolute() or ".." in path.parts or path.as_posix() != raw_path:
                raise ValueError(f"source path {logical_name} must be normalized and relative")
        return value

    @model_validator(mode="after")
    def _fixed_source_contract(self) -> HandoffSpecV2:
        expected_keys = set(_SOURCE_PATHS)
        if self.source_paths != _SOURCE_PATHS:
            raise ValueError("source_paths does not match the fixed v2 handoff source set")
        if set(self.canonical_sha256) != expected_keys:
            raise ValueError("canonical_sha256 does not match the fixed v2 handoff source set")
        if self.required_verification_platforms != ("Windows", "Linux"):
            raise ValueError("required verification targets must be Windows and Linux")
        if self.canonical_package_sha256 != {
            "ann-pilot-a.json": self.canonical_sha256["package_a"],
            "ann-pilot-b.json": self.canonical_sha256["package_b"],
        }:
            raise ValueError("canonical package hashes must bind the A/B source files")
        if self.instruction_sha256 != self.canonical_sha256["instruction"]:
            raise ValueError("instruction hash must bind the complete protocol source")
        if self.dependency_lock_sha256 != self.canonical_sha256["dependency_lock"]:
            raise ValueError("dependency lock hash must bind the annotation runtime lock")
        return self


class PlatformReceiptSummaryV2(_StrictModel):
    platform: Literal["Windows", "Linux"]
    receipt_sha256: Hash64
    verifier_sha256: Hash64
    python_version: str
    pip_version: str
    os_version: str
    architecture: str
    smoke_artifact_sha256: dict[str, Hash64]


class HandoffReceiptV2(_StrictModel):
    schema_version: Literal["handoff-receipt-v2"] = "handoff-receipt-v2"
    builder_version: Literal["handoff-builder-v2"] = "handoff-builder-v2"
    source_commit_sha: CommitSha
    git_tree_sha: CommitSha
    python_distribution: Literal["0.2.0"] = "0.2.0"
    protocol_version: Literal["pilot-v0.2.2-draft"] = "pilot-v0.2.2-draft"
    spec_sha256: Hash64
    manifest_schema_sha256: Hash64
    builder_sha256: Hash64
    canonical_source_sha256: dict[str, Hash64]
    canonical_package_sha256: dict[str, Hash64]
    instruction_sha256: Hash64
    dependency_lock_sha256: Hash64
    source_date_epoch: Literal[1787443200] = 1787443200
    build_command: tuple[str, ...]
    wheel: WheelIdentityV2
    clean_checkout_commands: tuple[GitCommandRecordV2, ...]
    required_verification_platforms: tuple[Literal["Windows", "Linux"], ...]
    verified_platforms: tuple[Literal["Windows", "Linux"], ...]
    platform_receipts: tuple[PlatformReceiptSummaryV2, PlatformReceiptSummaryV2]
    utc_build_time: dt.datetime
    kit_sha256sums: dict[str, Hash64]

    @field_validator("utc_build_time")
    @classmethod
    def _explicit_utc(cls, value: dt.datetime) -> dt.datetime:
        if value.tzinfo is None or value.utcoffset() != dt.timedelta(0):
            raise ValueError("utc_build_time must be an explicit UTC timestamp")
        return value

    @model_validator(mode="after")
    def _complete_platform_pair(self) -> HandoffReceiptV2:
        if self.required_verification_platforms != ("Windows", "Linux"):
            raise ValueError("handoff receipt requires Windows and Linux")
        if self.verified_platforms != ("Windows", "Linux"):
            raise ValueError("handoff receipt may pass only after Windows and Linux verification")
        if tuple(item.platform for item in self.platform_receipts) != ("Windows", "Linux"):
            raise ValueError("platform receipt summaries must be ordered Windows then Linux")
        return self


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_source(root: Path, relative_path: str) -> Path:
    resolved_root = root.resolve()
    path = (resolved_root / relative_path).resolve()
    if not path.is_relative_to(resolved_root) or not path.is_file() or path.is_symlink():
        raise DataError("handoff source is missing, unsafe, or outside the repository")
    return path


def _manifest_repository_root(spec_path: Path) -> Path:
    resolved = spec_path.resolve()
    try:
        root = resolved.parents[2]
    except IndexError as exc:
        raise DataError("handoff manifest must live at pilot/v0.2/handoff-manifest.json") from exc
    if resolved != root / "pilot/v0.2/handoff-manifest.json":
        raise DataError("handoff manifest must live at pilot/v0.2/handoff-manifest.json")
    return root


def resolve_handoff_sources(
    repository_root: Path,
    spec: HandoffSpecV2,
) -> dict[str, Path]:
    """Resolve and hash every source declared by the committed handoff contract."""
    sources: dict[str, Path] = {}
    for logical_name, relative_path in spec.source_paths.items():
        source = _resolve_source(repository_root, relative_path)
        if _sha256_file(source) != spec.canonical_sha256[logical_name]:
            raise DataError(f"handoff source hash mismatch: {logical_name}")
        sources[logical_name] = source
    return sources


def build_handoff_spec(repository_root: Path) -> HandoffSpecV2:
    """Construct the source-only v2 manifest from exact canonical repository bytes."""
    hashes = {
        logical_name: _sha256_file(_resolve_source(repository_root, relative_path))
        for logical_name, relative_path in _SOURCE_PATHS.items()
    }
    return HandoffSpecV2(
        supported_environment=SupportedEnvironmentV2(),
        source_paths=dict(_SOURCE_PATHS),
        canonical_sha256=hashes,
        canonical_package_sha256={
            "ann-pilot-a.json": hashes["package_a"],
            "ann-pilot-b.json": hashes["package_b"],
        },
        instruction_sha256=hashes["instruction"],
        dependency_lock_sha256=hashes["dependency_lock"],
        expected_layout=ExpectedLayoutV2(),
        reproducible_build=ReproducibleBuildRecipeV2(),
    )


def write_handoff_spec(repository_root: Path, output: Path) -> HandoffSpecV2:
    """Refresh the generated JSON schema, then atomically write the source-only manifest."""
    root = repository_root.resolve()
    expected_output = (root / "pilot/v0.2/handoff-manifest.json").resolve()
    if output.resolve() != expected_output:
        raise DataError("handoff manifest output must be pilot/v0.2/handoff-manifest.json")
    schema_path = root / _SOURCE_PATHS["manifest_schema"]
    write_json_atomic(schema_path, HandoffSpecV2.model_json_schema())
    spec = build_handoff_spec(root)
    write_json_atomic(expected_output, spec.model_dump(mode="json"))
    return spec


def _checksum_lines(files: Mapping[str, Path]) -> str:
    return "".join(
        f"{_sha256_file(path)}  {relative}\n" for relative, path in sorted(files.items())
    )


def _copy_kit(
    root: Path,
    *,
    wheel: Path,
    package: Path,
    lock: Path,
    onboarding: Path,
    runbook: Path,
    powershell_launcher: Path,
    posix_launcher: Path,
) -> str:
    root.mkdir(parents=True)
    source_files = {
        wheel.name: wheel,
        package.name: package,
        "annotation-requirements-py311.lock": lock,
        "ONBOARDING.md": onboarding,
        "HANDOFF_RUNBOOK.md": runbook,
        "start.ps1": powershell_launcher,
        "start.sh": posix_launcher,
    }
    copied: dict[str, Path] = {}
    for name, source in source_files.items():
        destination = root / name
        shutil.copyfile(source, destination)
        copied[name] = destination
    (root / "SHA256SUMS").write_text(
        _checksum_lines(copied),
        encoding="utf-8",
        newline="\n",
    )
    return _sha256_file(root / "SHA256SUMS")


def stage_disjoint_kits(
    external_root: Path,
    wheel: Path,
    sources: Mapping[str, Path],
    spec: HandoffSpecV2,
) -> dict[str, str]:
    """Copy the closed A/B layouts; no receipt or coordinator artifact enters either kit."""
    repository_root = sources["package_a"].resolve().parents[3]
    output = external_root.resolve()
    if output.is_relative_to(repository_root):
        raise DataError("handoff output must remain outside the repository")
    if external_root.exists() and any(external_root.iterdir()):
        raise DataError("handoff output directory must be absent or empty")
    expected_wheel = "rag_evidence_attribution_bench-0.2.0-py3-none-any.whl"
    if not wheel.is_file() or wheel.name != expected_wheel:
        raise DataError("handoff requires the exact verified 0.2.0 wheel filename")
    external_root.mkdir(parents=True, exist_ok=True)
    hashes = {
        "kit-a": _copy_kit(
            external_root / spec.expected_layout.kit_a_directory,
            wheel=wheel,
            package=sources["package_a"],
            lock=sources["dependency_lock"],
            onboarding=sources["onboarding"],
            runbook=sources["runbook"],
            powershell_launcher=sources["launcher_a_powershell"],
            posix_launcher=sources["launcher_a_posix"],
        ),
        "kit-b": _copy_kit(
            external_root / spec.expected_layout.kit_b_directory,
            wheel=wheel,
            package=sources["package_b"],
            lock=sources["dependency_lock"],
            onboarding=sources["onboarding"],
            runbook=sources["runbook"],
            powershell_launcher=sources["launcher_b_powershell"],
            posix_launcher=sources["launcher_b_posix"],
        ),
    }
    expected = {
        "kit-a": set(spec.expected_layout.kit_a_files) - {"<verified-wheel>"} | {wheel.name},
        "kit-b": set(spec.expected_layout.kit_b_files) - {"<verified-wheel>"} | {wheel.name},
    }
    for kit, expected_files in expected.items():
        actual = {
            path.relative_to(external_root / kit).as_posix()
            for path in (external_root / kit).rglob("*")
            if path.is_file()
        }
        if actual != expected_files:
            raise DataError(f"{kit} delivery layout is not closed")
    return hashes


def _verify_sources_in_checkout(
    checkout: VerifiedCheckoutV2,
    spec_path: Path,
    spec: HandoffSpecV2,
) -> dict[str, Path]:
    spec_relative = "pilot/v0.2/handoff-manifest.json"
    spec_hash = _sha256_file(spec_path)
    verify_tracked_blob(checkout, spec_relative, spec_hash)
    sources = resolve_handoff_sources(checkout.root, spec)
    for logical_name, relative_path in spec.source_paths.items():
        verify_tracked_blob(checkout, relative_path, spec.canonical_sha256[logical_name])
    return sources


def _verify_wheel_evidence(
    evidence: VerifiedWheelSetV2,
    checkout_a: VerifiedCheckoutV2,
    checkout_b: VerifiedCheckoutV2,
) -> WheelIdentityV2:
    roots = {"A": checkout_a.root.resolve(), "B": checkout_b.root.resolve()}
    if len(evidence.builds) != 4:
        raise DataError("wheel evidence must contain exactly four build records")
    wheel_paths: list[Path] = []
    observed_labels: set[tuple[str, str]] = set()
    source_spec = WheelSourceSpecV2(distribution_name="rag-evidence-attribution-bench")
    for record in evidence.builds:
        label = (record.logical_checkout, record.build_label)
        if label in observed_labels or label not in {
            ("A", "supplied"),
            ("A", "replay"),
            ("B", "supplied"),
            ("B", "replay"),
        }:
            raise DataError("wheel evidence contains duplicate or unexpected build labels")
        observed_labels.add(label)
        checkout = checkout_a if record.logical_checkout == "A" else checkout_b
        if record.pre_build.root.resolve() != roots[record.logical_checkout]:
            raise DataError("wheel evidence checkout root does not match the supplied checkout")
        if record.post_build.root.resolve() != roots[record.logical_checkout]:
            raise DataError("wheel evidence post-build root does not match the supplied checkout")
        identity = verify_wheel_payload(record.wheel_path, checkout, source_spec)
        if identity != record.wheel:
            raise DataError("wheel evidence identity does not match current wheel bytes")
        wheel_paths.append(record.wheel_path)
    basic = require_byte_identity(wheel_paths)
    if basic.sha256 != evidence.canonical.sha256 or evidence.canonical != evidence.builds[0].wheel:
        raise DataError("canonical wheel evidence is inconsistent")
    return evidence.canonical


def _validate_private_manifest(path: Path, spec: HandoffSpecV2) -> None:
    try:
        manifest = AssignmentManifestV2.model_validate(read_json(path))
    except (TypeError, ValueError) as exc:
        raise DataError("invalid external coordinator manifest v2") from exc
    validate_pilot_manifest_v2(manifest)
    if manifest.package_sha256 != spec.canonical_package_sha256:
        raise DataError("coordinator manifest package hashes do not match canonical A/B sources")


def _platform_summary(
    path: Path,
    receipt: PlatformVerificationReceiptV2,
) -> PlatformReceiptSummaryV2:
    return PlatformReceiptSummaryV2(
        platform=receipt.platform,
        receipt_sha256=_sha256_file(path),
        verifier_sha256=receipt.verifier_sha256,
        python_version=receipt.python_version,
        pip_version=receipt.pip_version,
        os_version=receipt.os_version,
        architecture=receipt.architecture,
        smoke_artifact_sha256=receipt.smoke_artifact_sha256,
    )


def build_handoff_v2(
    *,
    spec_path: Path,
    checkout_a_root: Path,
    checkout_b_root: Path,
    wheel_evidence_path: Path,
    coordinator_manifest_path: Path,
    windows_receipt_path: Path,
    linux_receipt_path: Path,
    external_root: Path,
) -> HandoffReceiptV2:
    """Revalidate exact-source evidence, platform receipts, and atomically create final kits."""
    spec = HandoffSpecV2.model_validate(read_json(spec_path))
    wheel_evidence = VerifiedWheelSetV2.model_validate(read_json(wheel_evidence_path))
    commit = wheel_evidence.source_commit_sha
    roots = (checkout_a_root.resolve(), checkout_b_root.resolve())
    if roots[0] == roots[1]:
        raise DataError("handoff requires two distinct source checkouts")
    output = external_root.resolve()
    if any(output.is_relative_to(root) or root.is_relative_to(output) for root in roots):
        raise DataError("handoff output must remain outside both source checkouts")
    if output.is_relative_to(_manifest_repository_root(spec_path)):
        raise DataError("handoff output must remain outside the source repository")
    if external_root.exists():
        raise DataError("final handoff output must not already exist")

    checkout_a = verify_checkout(roots[0], commit, phase="post-build", logical_checkout="A")
    checkout_b = verify_checkout(roots[1], commit, phase="post-build", logical_checkout="B")
    if checkout_a.tree_sha != checkout_b.tree_sha:
        raise DataError("the two exact-commit checkouts have different Git trees")
    sources_a = _verify_sources_in_checkout(checkout_a, spec_path, spec)
    _verify_sources_in_checkout(checkout_b, spec_path, spec)
    if _sha256_file(Path(__file__)) != spec.canonical_sha256["builder"]:
        raise DataError("executing handoff builder bytes do not match the candidate Git blob")
    wheel = _verify_wheel_evidence(wheel_evidence, checkout_a, checkout_b)
    _validate_private_manifest(coordinator_manifest_path, spec)

    expected_platform = PlatformIdentityV2(
        source_commit_sha=commit,
        git_tree_sha=checkout_a.tree_sha,
        wheel_filename=wheel.filename,
        wheel_byte_size=wheel.byte_size,
        wheel_sha256=wheel.sha256,
        dependency_lock_sha256=spec.dependency_lock_sha256,
        protocol_sha256=spec.instruction_sha256,
        handoff_spec_sha256=_sha256_file(spec_path),
        handoff_schema_sha256=spec.canonical_sha256["manifest_schema"],
    )
    windows = verify_platform_receipt(windows_receipt_path, expected=expected_platform)
    linux = verify_platform_receipt(linux_receipt_path, expected=expected_platform)
    if (windows.platform, linux.platform) != ("Windows", "Linux"):
        raise DataError("handoff requires exactly one Windows and one Linux platform receipt")
    expected_verifier_hash = spec.canonical_sha256["platform_script"]
    if any(receipt.verifier_sha256 != expected_verifier_hash for receipt in (windows, linux)):
        raise DataError("platform receipt verifier bytes do not match the candidate source")

    clean_commands = tuple(
        command
        for record in wheel_evidence.builds
        for checkout in (record.pre_build, record.post_build)
        for command in checkout.commands
    )
    platform_summaries = (
        _platform_summary(windows_receipt_path, windows),
        _platform_summary(linux_receipt_path, linux),
    )
    external_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{external_root.name}-staging-",
        dir=external_root.parent,
    ) as temporary:
        staging = Path(temporary) / "handoff"
        kit_hashes = stage_disjoint_kits(
            staging,
            wheel_evidence.builds[0].wheel_path,
            sources_a,
            spec,
        )
        receipt = HandoffReceiptV2(
            source_commit_sha=commit,
            git_tree_sha=checkout_a.tree_sha,
            spec_sha256=_sha256_file(spec_path),
            manifest_schema_sha256=spec.canonical_sha256["manifest_schema"],
            builder_sha256=spec.canonical_sha256["builder"],
            canonical_source_sha256=spec.canonical_sha256,
            canonical_package_sha256=spec.canonical_package_sha256,
            instruction_sha256=spec.instruction_sha256,
            dependency_lock_sha256=spec.dependency_lock_sha256,
            build_command=spec.reproducible_build.build_command,
            wheel=wheel,
            clean_checkout_commands=clean_commands,
            required_verification_platforms=spec.required_verification_platforms,
            verified_platforms=("Windows", "Linux"),
            platform_receipts=platform_summaries,
            utc_build_time=dt.datetime.now(dt.UTC),
            kit_sha256sums=kit_hashes,
        )
        privacy_violations = scan_delivery_payload(
            receipt.model_dump(mode="json"),
            artifact_kind="handoff_receipt",
        )
        if privacy_violations:
            raise DataError("handoff receipt privacy scan failed: " + "; ".join(privacy_violations))
        receipt_path = staging / "handoff-receipt.json"
        write_json_atomic(receipt_path, receipt.model_dump(mode="json"))
        inventory = {
            "handoff-receipt.json": receipt_path,
            **{
                path.relative_to(staging).as_posix(): path
                for kit in ("kit-a", "kit-b")
                for path in sorted((staging / kit).rglob("*"))
                if path.is_file()
            },
        }
        (staging / "SHA256SUMS").write_text(
            _checksum_lines(inventory),
            encoding="utf-8",
            newline="\n",
        )
        os.replace(staging, external_root)
    return receipt
