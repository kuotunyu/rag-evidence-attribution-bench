"""Exact Git identity, clean-checkout, and tracked-blob verification."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from rag_evidence.errors import DataError

Phase = Literal["pre-build", "post-build"]
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_STATUS_ARGV = ("git", "status", "--porcelain=v1", "--untracked-files=all", "-z")
_IGNORED_ARGV = (
    "git",
    "ls-files",
    "--others",
    "--ignored",
    "--exclude-standard",
    "-z",
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GitCommandRecordV2(_StrictModel):
    schema_version: Literal["git-command-record-v2"] = "git-command-record-v2"
    logical_checkout: str = Field(min_length=1)
    phase: Phase
    argv: tuple[str, ...]
    exit_code: int
    stdout_byte_count: int = Field(ge=0)
    stdout_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stderr_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class VerifiedCheckoutV2(_StrictModel):
    schema_version: Literal["verified-checkout-v2"] = "verified-checkout-v2"
    logical_checkout: str
    phase: Phase
    root: Path
    commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    tree_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    commands: tuple[GitCommandRecordV2, ...]


def _run_git(
    root: Path,
    args: tuple[str, ...],
    *,
    phase: Phase,
    logical_checkout: str,
) -> tuple[GitCommandRecordV2, bytes]:
    completed = subprocess.run(
        ("git", *args),
        cwd=root,
        check=False,
        capture_output=True,
    )
    stdout = completed.stdout
    stderr = completed.stderr
    return (
        GitCommandRecordV2(
            logical_checkout=logical_checkout,
            phase=phase,
            argv=("git", *args),
            exit_code=completed.returncode,
            stdout_byte_count=len(stdout),
            stdout_sha256=hashlib.sha256(stdout).hexdigest(),
            stderr_sha256=hashlib.sha256(stderr).hexdigest(),
        ),
        stdout,
    )


def _text(record: GitCommandRecordV2, stdout: bytes, label: str) -> str:
    if record.exit_code != 0:
        raise DataError(f"Git {label} verification failed")
    try:
        return stdout.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise DataError(f"Git {label} output is not ASCII") from exc


def verify_checkout(
    root: Path,
    requested_commit: str,
    *,
    phase: Phase,
    logical_checkout: str,
) -> VerifiedCheckoutV2:
    """Require one exact detached commit and zero tracked, untracked, or ignored paths."""
    if _SHA1_RE.fullmatch(requested_commit) is None:
        raise DataError("requested commit must be one full 40-character lowercase SHA")
    resolved_root = root.resolve()
    if not resolved_root.is_dir():
        raise DataError("checkout root is not a directory")
    records: list[GitCommandRecordV2] = []

    def run(*args: str) -> tuple[GitCommandRecordV2, bytes]:
        record, stdout = _run_git(
            resolved_root,
            tuple(args),
            phase=phase,
            logical_checkout=logical_checkout,
        )
        records.append(record)
        return record, stdout

    record, stdout = run("rev-parse", "--verify", f"{requested_commit}^{{commit}}")
    resolved_commit = _text(record, stdout, "requested commit")
    if resolved_commit != requested_commit:
        raise DataError("requested commit did not resolve to the exact supplied SHA")
    record, stdout = run("rev-parse", "HEAD")
    head = _text(record, stdout, "HEAD")
    if head != requested_commit:
        raise DataError("checkout HEAD does not equal the requested commit")
    record, stdout = run("rev-parse", "--abbrev-ref", "HEAD")
    if _text(record, stdout, "detached HEAD") != "HEAD":
        raise DataError("checkout HEAD must be detached")
    record, stdout = run("rev-parse", f"{requested_commit}^{{tree}}")
    tree = _text(record, stdout, "tree")
    if _SHA1_RE.fullmatch(tree) is None:
        raise DataError("checkout tree did not resolve to one Git tree SHA")

    status_record, status_stdout = run(*_STATUS_ARGV[1:])
    ignored_record, ignored_stdout = run(*_IGNORED_ARGV[1:])
    if status_record.exit_code != 0 or ignored_record.exit_code != 0:
        raise DataError("checkout clean gate command failed")
    if status_stdout or ignored_stdout:
        raise DataError("checkout clean gate failed: tracked, untracked, or ignored paths exist")
    return VerifiedCheckoutV2(
        logical_checkout=logical_checkout,
        phase=phase,
        root=resolved_root,
        commit_sha=requested_commit,
        tree_sha=tree,
        commands=tuple(records),
    )


def _relative_git_path(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise DataError("tracked source path must be one normalized repository-relative path")
    if ":" in value:
        raise DataError("tracked source path contains a forbidden colon")
    return path.as_posix()


def verify_tracked_blob(
    checkout: VerifiedCheckoutV2,
    relative_path: str,
    expected_sha256: str,
) -> str:
    """Bind regular working bytes to both the requested Git blob and expected SHA-256."""
    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise DataError("expected tracked source hash must be lowercase SHA-256")
    relative = _relative_git_path(relative_path)
    path = (checkout.root / relative).resolve()
    if not path.is_relative_to(checkout.root) or not path.is_file() or path.is_symlink():
        raise DataError("tracked source must be one regular file inside the checkout")
    tracked_record, _ = _run_git(
        checkout.root,
        ("ls-files", "--error-unmatch", "--", relative),
        phase=checkout.phase,
        logical_checkout=checkout.logical_checkout,
    )
    if tracked_record.exit_code != 0:
        raise DataError("source path is not tracked at the requested commit")
    blob_record, blob_bytes = _run_git(
        checkout.root,
        ("cat-file", "blob", f"{checkout.commit_sha}:{relative}"),
        phase=checkout.phase,
        logical_checkout=checkout.logical_checkout,
    )
    if blob_record.exit_code != 0:
        raise DataError("tracked Git blob could not be read")
    working_bytes = path.read_bytes()
    if working_bytes != blob_bytes:
        raise DataError("working source bytes do not match the tracked Git blob")
    if hashlib.sha256(working_bytes).hexdigest() != expected_sha256:
        raise DataError("tracked source hash does not match the committed handoff spec")
    oid_record, oid_stdout = _run_git(
        checkout.root,
        ("rev-parse", f"{checkout.commit_sha}:{relative}"),
        phase=checkout.phase,
        logical_checkout=checkout.logical_checkout,
    )
    oid = _text(oid_record, oid_stdout, "blob identity")
    if _SHA1_RE.fullmatch(oid) is None:
        raise DataError("tracked source blob did not resolve to one Git object SHA")
    return oid
