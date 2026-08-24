"""Cross-platform annotation verification receipt schema and identity checks."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rag_evidence.annotation.privacy import scan_delivery_payload
from rag_evidence.errors import DataError
from rag_evidence.storage.artifacts import read_json, write_json_atomic

PlatformName = Literal["Windows", "Linux"]
CommandKind = Literal[
    "pip-version",
    "index-configuration",
    "dependency-bootstrap",
    "wheel-install",
    "installed-distributions",
    "cli-version",
    "cli-help",
    "app-creation",
    "launcher-runtime",
    "tasks-probe",
    "progress-probe",
    "empty-export",
    "loopback-ipv4",
    "loopback-ipv6",
    "loopback-localhost",
    "reject-unspecified-ipv4",
    "reject-unspecified-ipv6",
    "reject-lan",
    "reject-public",
    "reject-hostname",
]
REQUIRED_COMMAND_KINDS: tuple[CommandKind, ...] = (
    "pip-version",
    "index-configuration",
    "dependency-bootstrap",
    "wheel-install",
    "installed-distributions",
    "cli-version",
    "cli-help",
    "app-creation",
    "launcher-runtime",
    "tasks-probe",
    "progress-probe",
    "empty-export",
    "loopback-ipv4",
    "loopback-ipv6",
    "loopback-localhost",
    "reject-unspecified-ipv4",
    "reject-unspecified-ipv6",
    "reject-lan",
    "reject-public",
    "reject-hostname",
)
_HASH_PATTERN = r"^[0-9a-f]{64}$"
_COMMIT_PATTERN = r"^[0-9a-f]{40}$"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _utc(value: dt.datetime, field: str) -> dt.datetime:
    if value.tzinfo is None or value.utcoffset() != dt.timedelta(0):
        raise ValueError(f"{field} must be an explicit UTC timestamp")
    return value


def _safe_relative(value: str, field: str) -> str:
    if "\\" in value or ":" in value:
        raise ValueError(f"{field} must not contain a private or absolute path")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} must be one normalized logical path")
    return path.as_posix()


class PlatformCommandRecordV2(_StrictModel):
    schema_version: Literal["platform-command-record-v2"] = "platform-command-record-v2"
    kind: CommandKind
    argv: tuple[str, ...] = Field(min_length=1)
    logical_cwd: str = Field(min_length=1)
    exit_code: int
    stdout_sha256: str = Field(pattern=_HASH_PATTERN)
    stderr_sha256: str = Field(pattern=_HASH_PATTERN)
    started_at: dt.datetime
    ended_at: dt.datetime
    semantic_result: Literal["passed"]

    @field_validator("started_at", "ended_at")
    @classmethod
    def _times_are_utc(cls, value: dt.datetime, info: object) -> dt.datetime:
        return _utc(value, getattr(info, "field_name", "timestamp"))

    @model_validator(mode="after")
    def _safe_successful_command(self) -> PlatformCommandRecordV2:
        if self.exit_code != 0:
            raise ValueError("passed platform command must have exit code zero")
        _safe_relative(self.logical_cwd, "logical_cwd")
        if self.ended_at < self.started_at:
            raise ValueError("platform command end time precedes start time")
        for argument in self.argv:
            lowered = argument.casefold()
            if re.search(r"(?:token|password|secret|credential)=", lowered):
                raise ValueError("platform command argv contains secret material")
            if re.match(r"^[a-z]:[\\/]", argument, re.IGNORECASE) or argument.startswith(
                ("/home/", "/Users/")
            ):
                raise ValueError("platform command argv contains a private absolute path")
        return self


class InstalledDistributionV2(_StrictModel):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    version: str = Field(min_length=1)


class PlatformIdentityV2(_StrictModel):
    source_commit_sha: str = Field(pattern=_COMMIT_PATTERN)
    git_tree_sha: str = Field(pattern=_COMMIT_PATTERN)
    python_distribution: Literal["0.2.0"] = "0.2.0"
    wheel_filename: str = Field(pattern=r"^[A-Za-z0-9_.-]+\.whl$")
    wheel_byte_size: int = Field(gt=0)
    wheel_sha256: str = Field(pattern=_HASH_PATTERN)
    dependency_lock_sha256: str = Field(pattern=_HASH_PATTERN)
    protocol_version: Literal["pilot-v0.2.2-draft"] = "pilot-v0.2.2-draft"
    protocol_sha256: str = Field(pattern=_HASH_PATTERN)
    handoff_spec_sha256: str = Field(pattern=_HASH_PATTERN)
    handoff_schema_sha256: str = Field(pattern=_HASH_PATTERN)


class PlatformVerificationReceiptV2(PlatformIdentityV2):
    schema_version: Literal["platform-verification-receipt-v2"] = "platform-verification-receipt-v2"
    verifier_version: Literal["annotation-platform-verifier-v2"] = "annotation-platform-verifier-v2"
    verifier_sha256: str = Field(pattern=_HASH_PATTERN)
    platform: PlatformName
    python_version: str = Field(pattern=r"^3\.11(?:\.\d+)?$")
    os_version: str = Field(min_length=1)
    architecture: str = Field(min_length=1)
    pip_version: str = Field(min_length=1)
    index_configuration: tuple[str, ...]
    bootstrap_argv: tuple[str, ...]
    wheel_install_argv: tuple[str, ...]
    installed_distributions: tuple[InstalledDistributionV2, ...]
    commands: tuple[PlatformCommandRecordV2, ...]
    smoke_artifact_sha256: dict[str, str]
    started_at: dt.datetime
    ended_at: dt.datetime
    overall_result: Literal["passed"]

    @field_validator("started_at", "ended_at")
    @classmethod
    def _receipt_times_are_utc(cls, value: dt.datetime, info: object) -> dt.datetime:
        return _utc(value, getattr(info, "field_name", "timestamp"))

    @field_validator("smoke_artifact_sha256")
    @classmethod
    def _safe_smoke_artifacts(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("platform receipt requires generated smoke artifacts")
        for relative, digest in value.items():
            _safe_relative(relative, "smoke artifact")
            if re.fullmatch(_HASH_PATTERN, digest) is None:
                raise ValueError("smoke artifact hash must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def _complete_derived_success(self) -> PlatformVerificationReceiptV2:
        kinds = tuple(command.kind for command in self.commands)
        if len(set(kinds)) != len(kinds) or set(kinds) != set(REQUIRED_COMMAND_KINDS):
            raise ValueError("platform receipt must contain every required command exactly once")
        if self.ended_at < self.started_at:
            raise ValueError("platform receipt end time precedes start time")
        if tuple(sorted(self.installed_distributions, key=lambda item: item.name.casefold())) != (
            self.installed_distributions
        ):
            raise ValueError("installed distributions must be sorted by normalized name")
        required_bootstrap = ("--require-hashes", "--only-binary=:all:")
        if any(flag not in self.bootstrap_argv for flag in required_bootstrap):
            raise ValueError("bootstrap argv must enforce hashes and binary-only dependencies")
        if "--no-deps" not in self.wheel_install_argv:
            raise ValueError("wheel install argv must include --no-deps")
        for value in self.index_configuration:
            lowered = value.casefold()
            if "@" in value or any(term in lowered for term in ("token", "password", "secret")):
                raise ValueError("index configuration contains credentials or secret material")
        return self


def verify_platform_receipt(
    receipt_path: Path,
    *,
    expected: PlatformIdentityV2,
) -> PlatformVerificationReceiptV2:
    """Load one generated receipt, bind its identity, and verify retained smoke bytes."""
    receipt = PlatformVerificationReceiptV2.model_validate(read_json(receipt_path))
    for field, expected_value in expected.model_dump(mode="python").items():
        if getattr(receipt, field) != expected_value:
            raise DataError(f"platform receipt identity mismatch: {field}")
    root = receipt_path.resolve().parent
    for relative, expected_hash in receipt.smoke_artifact_sha256.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
            raise DataError("platform smoke artifact is missing or unsafe")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise DataError("platform smoke artifact hash mismatch")
    return receipt


def emit_platform_receipt(
    payload_path: Path,
    output_path: Path,
    evidence_root: Path,
) -> PlatformVerificationReceiptV2:
    """Validate generated payload and retained artifacts before atomically emitting a receipt."""
    receipt = PlatformVerificationReceiptV2.model_validate(read_json(payload_path))
    root = evidence_root.resolve()
    for relative, expected_hash in receipt.smoke_artifact_sha256.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
            raise DataError("platform smoke artifact is missing or unsafe")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise DataError("platform smoke artifact hash mismatch")
    violations = scan_delivery_payload(
        receipt.model_dump(mode="json"),
        artifact_kind="platform_receipt",
    )
    if violations:
        raise DataError("platform receipt privacy scan failed: " + "; ".join(violations))
    write_json_atomic(output_path, receipt.model_dump(mode="json"))
    return receipt


def _main() -> int:
    parser = argparse.ArgumentParser(description="Validate and emit one v2 platform receipt")
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        emit_platform_receipt(args.payload, args.output, args.evidence_root)
    except (DataError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
