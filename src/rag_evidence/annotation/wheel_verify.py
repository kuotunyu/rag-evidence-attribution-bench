"""Reproducible wheel builds bound to one exact, clean Git checkout."""

from __future__ import annotations

import base64
import csv
import hashlib
import os
import re
import subprocess
import zipfile
from collections.abc import Sequence
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_evidence.annotation.source_verify import (
    VerifiedCheckoutV2,
    verify_checkout,
    verify_tracked_blob,
)
from rag_evidence.errors import DataError
from rag_evidence.storage.artifacts import write_json_atomic

SOURCE_DATE_EPOCH = 1787443200
BUILD_ARGV = ("uv", "build", "--wheel", "--out-dir")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WheelSourceSpecV2(_StrictModel):
    schema_version: Literal["wheel-source-spec-v2"] = "wheel-source-spec-v2"
    distribution_name: str = Field(min_length=1)
    distribution_version: Literal["0.2.0"] = "0.2.0"
    package_source_root: Literal["src/rag_evidence"] = "src/rag_evidence"


class FileIdentityV2(_StrictModel):
    schema_version: Literal["file-identity-v2"] = "file-identity-v2"
    filename: str = Field(min_length=1)
    byte_size: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class WheelIdentityV2(_StrictModel):
    schema_version: Literal["wheel-identity-v2"] = "wheel-identity-v2"
    filename: str = Field(min_length=1)
    byte_size: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    distribution_name: str
    distribution_version: Literal["0.2.0"] = "0.2.0"
    package_files: tuple[str, ...]


class BuildEnvironmentV2(_StrictModel):
    schema_version: Literal["wheel-build-environment-v2"] = "wheel-build-environment-v2"
    source_date_epoch: Literal[1787443200] = 1787443200
    cache_root: Path
    build_root: Path
    output_root: Path
    pycache_root: Path

    @model_validator(mode="after")
    def _distinct_roots(self) -> BuildEnvironmentV2:
        roots = tuple(path.resolve() for path in self.roots)
        if len(set(roots)) != len(roots):
            raise ValueError("wheel build roots must be distinct")
        for index, root in enumerate(roots):
            if any(root.is_relative_to(other) for other in roots[:index] + roots[index + 1 :]):
                raise ValueError("wheel build roots must not contain one another")
        return self

    @property
    def roots(self) -> tuple[Path, ...]:
        return (self.cache_root, self.build_root, self.output_root, self.pycache_root)


class WheelBuildRecordV2(_StrictModel):
    schema_version: Literal["wheel-build-record-v2"] = "wheel-build-record-v2"
    logical_checkout: str
    build_label: str
    argv: tuple[str, ...]
    source_date_epoch: Literal[1787443200] = 1787443200
    environment: BuildEnvironmentV2
    exit_code: int
    stdout_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stderr_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pre_build: VerifiedCheckoutV2
    post_build: VerifiedCheckoutV2
    wheel_path: Path
    wheel: WheelIdentityV2


class VerifiedWheelSetV2(_StrictModel):
    schema_version: Literal["verified-wheel-set-v2"] = "verified-wheel-set-v2"
    source_commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_date_epoch: Literal[1787443200] = 1787443200
    canonical: WheelIdentityV2
    builds: tuple[WheelBuildRecordV2, WheelBuildRecordV2, WheelBuildRecordV2, WheelBuildRecordV2]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_identity(path: Path) -> FileIdentityV2:
    if not path.is_file() or path.suffix.casefold() != ".whl":
        raise DataError("wheel identity requires one existing .whl file")
    payload = path.read_bytes()
    if not payload:
        raise DataError("wheel must not be empty")
    return FileIdentityV2(filename=path.name, byte_size=len(payload), sha256=_sha256_bytes(payload))


def require_byte_identity(wheels: Sequence[Path]) -> FileIdentityV2:
    """Require exactly four supplied/replay wheel files to have identical bytes."""
    if len(wheels) != 4:
        raise DataError("exactly four wheels are required for byte-identical verification")
    identities = tuple(file_identity(path) for path in wheels)
    signatures = {(item.filename, item.byte_size, item.sha256) for item in identities}
    if len(signatures) != 1:
        raise DataError("all supplied and replay wheels must be byte-identical")
    first_bytes = wheels[0].read_bytes()
    if any(path.read_bytes() != first_bytes for path in wheels[1:]):
        raise DataError("all supplied and replay wheels must be byte-identical")
    return identities[0]


def _normalized_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def _safe_zip_name(value: str) -> str:
    if "\\" in value or ":" in value:
        raise DataError("wheel contains an unsafe ZIP path")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise DataError("wheel contains an unsafe ZIP path")
    normalized = path.as_posix()
    if normalized != value or value.endswith("/"):
        raise DataError("wheel members must be normalized regular-file paths")
    return normalized


def _record_digest(payload: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
    return f"sha256={encoded}"


def _tracked_package_paths(checkout: VerifiedCheckoutV2, source_root: str) -> tuple[str, ...]:
    completed = subprocess.run(
        ("git", "ls-files", "-z", "--", source_root),
        cwd=checkout.root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise DataError("tracked package source enumeration failed")
    try:
        paths = tuple(item.decode("utf-8") for item in completed.stdout.split(b"\0") if item)
    except UnicodeDecodeError as exc:
        raise DataError("tracked package source paths are not UTF-8") from exc
    if not paths:
        raise DataError("no tracked package source files were found")
    prefix = f"{source_root}/"
    if any(not path.startswith(prefix) for path in paths):
        raise DataError("tracked package enumeration escaped the source root")
    return tuple(sorted(paths))


def verify_wheel_payload(
    wheel_path: Path,
    checkout: VerifiedCheckoutV2,
    spec: WheelSourceSpecV2,
) -> WheelIdentityV2:
    """Validate wheel structure and bind all import-package bytes to committed Git blobs."""
    basic = file_identity(wheel_path)
    try:
        with zipfile.ZipFile(wheel_path) as archive:
            infos = archive.infolist()
            names = tuple(_safe_zip_name(info.filename) for info in infos)
            if len(set(names)) != len(names):
                raise DataError("wheel contains duplicate ZIP members")
            entries = {name: archive.read(info) for name, info in zip(names, infos, strict=True)}
    except (OSError, zipfile.BadZipFile) as exc:
        raise DataError("wheel is not a readable ZIP archive") from exc

    metadata_names = tuple(name for name in names if name.endswith(".dist-info/METADATA"))
    record_names = tuple(name for name in names if name.endswith(".dist-info/RECORD"))
    if len(metadata_names) != 1 or len(record_names) != 1:
        raise DataError("wheel must contain exactly one METADATA and one RECORD")
    metadata_name = metadata_names[0]
    record_name = record_names[0]
    if metadata_name.rsplit("/", 1)[0] != record_name.rsplit("/", 1)[0]:
        raise DataError("wheel METADATA and RECORD must share one dist-info directory")

    try:
        record_rows = tuple(csv.reader(entries[record_name].decode("utf-8").splitlines()))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise DataError("wheel RECORD is not valid UTF-8 CSV") from exc
    if any(len(row) != 3 for row in record_rows):
        raise DataError("wheel RECORD rows must have exactly three columns")
    record_map = {row[0]: (row[1], row[2]) for row in record_rows}
    if len(record_map) != len(record_rows) or set(record_map) != set(entries):
        raise DataError("wheel RECORD members must exactly match ZIP members")
    for name, payload in entries.items():
        recorded_hash, recorded_size = record_map[name]
        if name == record_name:
            if recorded_hash or recorded_size:
                raise DataError("wheel RECORD self-entry must have blank hash and size")
            continue
        if recorded_hash != _record_digest(payload) or recorded_size != str(len(payload)):
            raise DataError("wheel RECORD hash or size does not match member bytes")

    metadata = BytesParser().parsebytes(entries[metadata_name])
    name = metadata.get("Name", "")
    version = metadata.get("Version", "")
    if _normalized_name(name) != _normalized_name(spec.distribution_name):
        raise DataError("wheel METADATA distribution name does not match the source contract")
    if version != spec.distribution_version:
        raise DataError("wheel METADATA distribution version does not match 0.2.0")
    filename_prefix = spec.distribution_name.replace("-", "_").replace(".", "_")
    expected_filename = f"{filename_prefix}-{spec.distribution_version}-py3-none-any.whl"
    if wheel_path.name != expected_filename:
        raise DataError("wheel filename does not match the fixed distribution identity")

    tracked = _tracked_package_paths(checkout, spec.package_source_root)
    source_prefix = f"{spec.package_source_root}/"
    expected_package = {
        f"rag_evidence/{relative[len(source_prefix) :]}": relative for relative in tracked
    }
    actual_package = {name for name in names if name.startswith("rag_evidence/")}
    if actual_package != set(expected_package):
        raise DataError("wheel package files do not exactly match tracked package sources")
    for member, relative_path in expected_package.items():
        payload_hash = _sha256_bytes(entries[member])
        verify_tracked_blob(checkout, relative_path, payload_hash)

    return WheelIdentityV2(
        filename=basic.filename,
        byte_size=basic.byte_size,
        sha256=basic.sha256,
        distribution_name=_normalized_name(name),
        distribution_version=version,
        package_files=tuple(sorted(actual_package)),
    )


def _require_external_roots(checkout_root: Path, environment: BuildEnvironmentV2) -> None:
    checkout = checkout_root.resolve()
    for root in environment.roots:
        resolved = root.resolve()
        if resolved.is_relative_to(checkout) or checkout.is_relative_to(resolved):
            raise DataError("build environment roots must remain outside the source checkout")


def replay_wheel_build(
    checkout_root: Path,
    requested_commit: str,
    environment: BuildEnvironmentV2,
    *,
    logical_checkout: str,
    build_label: str,
    spec: WheelSourceSpecV2,
) -> WheelBuildRecordV2:
    """Build one wheel while bracketing the command with exact clean-checkout gates."""
    _require_external_roots(checkout_root, environment)
    for root in environment.roots:
        if root.exists() and any(root.iterdir()):
            raise DataError("wheel build environment roots must be absent or empty")
        root.mkdir(parents=True, exist_ok=True)
    pre_build = verify_checkout(
        checkout_root,
        requested_commit,
        phase="pre-build",
        logical_checkout=logical_checkout,
    )
    argv = (*BUILD_ARGV, str(environment.output_root.resolve()))
    child_environment = os.environ.copy()
    child_environment.update(
        {
            "SOURCE_DATE_EPOCH": str(environment.source_date_epoch),
            "UV_CACHE_DIR": str(environment.cache_root.resolve()),
            "TMP": str(environment.build_root.resolve()),
            "TEMP": str(environment.build_root.resolve()),
            "TMPDIR": str(environment.build_root.resolve()),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": str(environment.pycache_root.resolve()),
            "PYTHONHASHSEED": "0",
            "UV_LINK_MODE": "copy",
            "UV_NO_PROGRESS": "1",
        }
    )
    completed = subprocess.run(
        argv,
        cwd=checkout_root,
        env=child_environment,
        check=False,
        capture_output=True,
    )
    post_build = verify_checkout(
        checkout_root,
        requested_commit,
        phase="post-build",
        logical_checkout=logical_checkout,
    )
    if completed.returncode != 0:
        raise DataError("reproducible wheel build command failed")
    wheels = tuple(environment.output_root.glob("*.whl"))
    if len(wheels) != 1:
        raise DataError("wheel build must produce exactly one wheel")
    identity = verify_wheel_payload(wheels[0], post_build, spec)
    return WheelBuildRecordV2(
        logical_checkout=logical_checkout,
        build_label=build_label,
        argv=tuple(str(part) for part in argv),
        environment=environment,
        exit_code=completed.returncode,
        stdout_sha256=_sha256_bytes(completed.stdout),
        stderr_sha256=_sha256_bytes(completed.stderr),
        pre_build=pre_build,
        post_build=post_build,
        wheel_path=wheels[0].resolve(),
        wheel=identity,
    )


def build_four_wheels(
    checkout_a: Path,
    checkout_b: Path,
    source_commit: str,
    output_root: Path,
) -> VerifiedWheelSetV2:
    """Build supplied/replay instances from both exact checkouts and prove byte identity."""
    output = output_root.resolve()
    checkouts = (checkout_a.resolve(), checkout_b.resolve())
    if any(output.is_relative_to(root) or root.is_relative_to(output) for root in checkouts):
        raise DataError("wheel evidence output must remain outside both source checkouts")
    if output.exists() and any(output.iterdir()):
        raise DataError("wheel evidence output must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    spec = WheelSourceSpecV2(distribution_name="rag-evidence-attribution-bench")
    records: list[WheelBuildRecordV2] = []
    for logical_checkout, checkout in (("A", checkouts[0]), ("B", checkouts[1])):
        for build_label in ("supplied", "replay"):
            prefix = output / f"checkout-{logical_checkout.casefold()}-{build_label}"
            environment = BuildEnvironmentV2(
                cache_root=prefix.with_name(prefix.name + "-cache"),
                build_root=prefix.with_name(prefix.name + "-build"),
                output_root=prefix.with_name(prefix.name + "-dist"),
                pycache_root=prefix.with_name(prefix.name + "-pycache"),
            )
            records.append(
                replay_wheel_build(
                    checkout,
                    source_commit,
                    environment,
                    logical_checkout=logical_checkout,
                    build_label=build_label,
                    spec=spec,
                )
            )
    wheel_paths = tuple(record.wheel_path for record in records)
    basic = require_byte_identity(wheel_paths)
    if any(record.wheel.sha256 != basic.sha256 for record in records):
        raise DataError("wheel payload verification disagrees with byte-identical verification")
    canonical = records[0].wheel
    result = VerifiedWheelSetV2(
        source_commit_sha=source_commit,
        canonical=canonical,
        builds=(records[0], records[1], records[2], records[3]),
    )
    write_json_atomic(output / "wheel-verification.json", result.model_dump(mode="json"))
    return result
