"""Generate and validate the binary-only annotation runtime requirements lock."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from rag_evidence.errors import DataError

EXPORT_ARGV = (
    "uv",
    "export",
    "--frozen",
    "--only-group",
    "annotation",
    "--no-emit-project",
    "--no-header",
    "--no-annotate",
    "--format",
    "requirements-txt",
)
_ROOT_DISTRIBUTION = "rag-evidence-attribution-bench"
_REQUIREMENT_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^\s;@]+)"
    r"(?:\s*;\s*(?P<marker>.+))?$"
)
_HASH_RE = re.compile(r"(?:^|\s)--hash=sha256:([0-9a-f]{64})(?=\s|$)")


def _logical_requirements(text: str) -> tuple[str, ...]:
    if "\r" in text:
        raise DataError("annotation lock must use LF line endings")
    logical: list[str] = []
    pending = ""
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            raise DataError("annotation lock must not contain generated comments")
        continued = stripped.endswith("\\")
        part = stripped[:-1].rstrip() if continued else stripped
        pending = f"{pending} {part}".strip()
        if not continued:
            logical.append(pending)
            pending = ""
    if pending:
        raise DataError("annotation lock ends with an incomplete continuation")
    return tuple(logical)


def _normalized_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def validate_annotation_lock(text: str) -> None:
    """Reject anything except exact third-party index pins with SHA-256 hashes."""
    if not text or not text.endswith("\n"):
        raise DataError("annotation lock must be nonempty and newline terminated")
    lowered = text.casefold()
    forbidden_fragments = (
        "--index-url",
        "--extra-index-url",
        "--find-links",
        "--trusted-host",
        "git+",
        "file:",
        "hg+",
        "svn+",
        "bzr+",
        "-e ",
        "../",
        "..\\",
    )
    if any(fragment in lowered for fragment in forbidden_fragments):
        raise DataError("annotation lock contains a forbidden source or installer option")
    requirements = _logical_requirements(text)
    if not requirements:
        raise DataError("annotation lock contains no requirements")
    for block in requirements:
        hashes = _HASH_RE.findall(block)
        if not hashes:
            raise DataError("every annotation dependency must include a SHA-256 hash")
        requirement = _HASH_RE.sub("", block).strip()
        if "--" in requirement or requirement.startswith((".", "/", "\\")):
            raise DataError("annotation lock contains an installer option or local path")
        match = _REQUIREMENT_RE.fullmatch(requirement)
        if match is None:
            raise DataError("annotation dependency must be one exact index pin")
        if _normalized_name(match.group("name")) == _ROOT_DISTRIBUTION:
            raise DataError("annotation lock must not contain the root project")
        marker = match.group("marker")
        if marker is not None and not marker.strip():
            raise DataError("annotation dependency marker must not be empty")


def build_annotation_lock(repository_root: Path) -> bytes:
    """Export the frozen annotation dependency group and return validated canonical bytes."""
    completed = subprocess.run(
        EXPORT_ARGV,
        cwd=repository_root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise DataError(f"frozen annotation lock export failed: {detail}")
    text = completed.stdout.decode("utf-8").replace("\r\n", "\n")
    validate_annotation_lock(text)
    return text.encode("utf-8")
