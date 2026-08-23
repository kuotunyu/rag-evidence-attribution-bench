"""Reusable privacy and hidden-source-field scanning for annotation artifacts."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rag_evidence.storage.artifacts import read_json, read_records

_PRIVATE_KEYS = frozenset(
    {
        "transformation",
        "transform_version",
        "expected_answerability",
        "changed_fields",
        "provenance",
        "parent_question_id",
        "parent_fingerprint",
        "gold_adjudicated_label",
        "gold_passage_ids",
        "supporting_fact_sentence_ids",
        "is_gold",
        "model_name",
        "method_name",
        "score",
        "scores",
        "abstention_expectation",
    }
)
_EMAIL_RE = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_WINDOWS_PATH_RE = re.compile(r"(?:^|[\s\"'(])[A-Za-z]:[\\/]")
_POSIX_PRIVATE_PATH_RE = re.compile(r"(?:^|[\s\"'(])/(?:Users|home|root)/")
_INTERNAL_GROUP_RE = re.compile(r"^(?:bg-[0-9a-f]{24}|coord(?:inator)?-group-[a-z0-9._-]+)$")
_INTERNAL_PASSAGE_RE = re.compile(r"\bch-[0-9a-f]{24}-p\d{2,}(?:-s\d{2,})?\b")
_V1_SCHEMA_RE = re.compile(
    r"^(?:blind-task|assignment-package|assignment-manifest|annotation-draft|"
    r"answerability-annotation|citation-annotation|annotation-amendment|disagreement-case|"
    r"adjudication|eligibility-record|eligibility-artifact|annotation-collection-receipt|"
    r"annotation-input-manifest|pilot-[a-z-]+|handoff-[a-z-]+|"
    r"platform-verification-receipt)-v1$"
)
_CONCRETE_TRANSFORMATIONS = frozenset({"missing_hop", "evidence_swap"})
_FORBIDDEN_KEY_PARTS = frozenset(
    {
        "parent",
        "group",
        "sibling",
        "transformation",
        "expected",
        "gold",
        "provenance",
        "coordinator",
        "seed",
        "score",
    }
)
_DELIVERY_TEXT_SOURCES = (
    "src/rag_evidence/annotation/ui.html",
    "pilot/v0.2/ONBOARDING.md",
    "pilot/v0.2/HANDOFF_RUNBOOK.md",
    "pilot/v0.2/launchers/start-a.ps1",
    "pilot/v0.2/launchers/start-a.sh",
    "pilot/v0.2/launchers/start-b.ps1",
    "pilot/v0.2/launchers/start-b.sh",
)
_STRICT_RUNTIME_SOURCES = frozenset(
    {
        "src/rag_evidence/annotation/ui.html",
        "pilot/v0.2/launchers/start-a.ps1",
        "pilot/v0.2/launchers/start-a.sh",
        "pilot/v0.2/launchers/start-b.ps1",
        "pilot/v0.2/launchers/start-b.sh",
    }
)
_HIDDEN_FIELD_LITERALS = (
    "blinded_parent_group",
    "parent_question_id",
    "parent_fingerprint",
    "internal_group_id",
    "expected_answerability",
    "gold_adjudicated_label",
    "supporting_fact_sentence_ids",
)
_POSITIVE_OVERCLAIMS = (
    "fully sibling-blind",
    "semantic unlinkability is guaranteed",
    "guarantees semantic unlinkability",
    "independent sibling perception is established",
)


def scan_private_payload(payload: object) -> tuple[str, ...]:
    """Return stable, field-addressed privacy violations without mutating the payload."""
    violations: list[str] = []

    def visit(value: object, path: str) -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key).casefold()
                if key in _PRIVATE_KEYS or key.startswith(("expected_", "gold_")):
                    violations.append(f"{path}.{raw_key}: forbidden source metadata")
                visit(child, f"{path}.{raw_key}")
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
            return
        if isinstance(value, str):
            if _EMAIL_RE.search(value):
                violations.append(f"{path}: PII-like email")
            if _WINDOWS_PATH_RE.search(value) or _POSIX_PRIVATE_PATH_RE.search(value):
                violations.append(f"{path}: private filesystem path")

    visit(payload, "$")
    return tuple(sorted(set(violations)))


def _key_parts(key: str) -> frozenset[str]:
    return frozenset(part for part in re.split(r"[^a-z0-9]+", key.casefold()) if part)


def scan_delivery_payload(payload: object, *, artifact_kind: str) -> tuple[str, ...]:
    """Return hidden-metadata violations for a human-delivery v2 payload."""
    violations: list[str] = []

    def visit(value: object, path: str) -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key)
                child_path = f"{path}.{key}"
                lowered = key.casefold()
                if (
                    lowered in _PRIVATE_KEYS
                    or lowered.startswith(("expected_", "gold_"))
                    or _key_parts(key) & _FORBIDDEN_KEY_PARTS
                ):
                    violations.append(f"{child_path}: forbidden delivery metadata key")
                visit(child, child_path)
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
            return
        if isinstance(value, str):
            if _V1_SCHEMA_RE.fullmatch(value):
                violations.append(f"{path}: v1 schema literal")
            if _INTERNAL_GROUP_RE.fullmatch(value):
                violations.append(f"{path}: coordinator group identifier")
            if value in _CONCRETE_TRANSFORMATIONS:
                violations.append(f"{path}: transformation identity")
            if _EMAIL_RE.search(value):
                violations.append(f"{path}: PII-like email")
            if _WINDOWS_PATH_RE.search(value) or _POSIX_PRIVATE_PATH_RE.search(value):
                violations.append(f"{path}: private filesystem path")
            if _INTERNAL_PASSAGE_RE.search(value):
                violations.append(f"{path}: internal passage or sentence identifier")

    visit(payload, f"${artifact_kind}")
    return tuple(sorted(set(violations)))


def scan_delivery_tree(
    root: Path,
    *,
    allowed_files: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Scan JSON/JSONL delivery files and report stable relative-path violations."""
    if not root.is_dir():
        return (f"{root.name}: delivery root is not a directory",)
    allowed = set(allowed_files) if allowed_files is not None else None
    violations: list[str] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        if allowed is not None and relative not in allowed:
            violations.append(f"{relative}: unexpected delivery file")
            continue
        payloads: tuple[Any, ...]
        try:
            if path.suffix.casefold() == ".json":
                payloads = (read_json(path),)
            elif path.suffix.casefold() == ".jsonl":
                payloads = tuple(read_records(path))
            else:
                continue
        except Exception as exc:
            violations.append(f"{relative}: unreadable delivery JSON ({exc})")
            continue
        for index, payload in enumerate(payloads):
            location = relative if len(payloads) == 1 else f"{relative}:{index + 1}"
            for violation in scan_delivery_payload(payload, artifact_kind="tree"):
                violations.append(f"{location}: {violation}")
    return tuple(sorted(set(violations)))


def _scan_delivery_text(relative: str, text: str) -> tuple[str, ...]:
    lowered = text.casefold()
    violations: list[str] = []
    if relative in _STRICT_RUNTIME_SOURCES:
        for literal in _HIDDEN_FIELD_LITERALS:
            if literal in lowered:
                violations.append(f"{relative}: hidden field literal {literal}")
        if any(value in lowered for value in _CONCRETE_TRANSFORMATIONS):
            violations.append(f"{relative}: concrete transformation label")
    if re.search(r"(?:bg-[0-9a-f]{24}|coord-[0-9a-f]{24})", lowered):
        violations.append(f"{relative}: reserved internal group identifier")
    if re.search(r"(?:blind-task|assignment-package|answerability-annotation)-v1", lowered):
        violations.append(f"{relative}: v1 human-delivery schema literal")
    if _EMAIL_RE.search(text):
        violations.append(f"{relative}: PII-like email")
    if _WINDOWS_PATH_RE.search(text) or _POSIX_PRIVATE_PATH_RE.search(text):
        violations.append(f"{relative}: private absolute filesystem path")
    for claim in _POSITIVE_OVERCLAIMS:
        if claim in lowered:
            violations.append(f"{relative}: unsupported blinding claim")
    return tuple(violations)


def scan_delivery_sources(repository_root: Path) -> tuple[str, ...]:
    """Scan the exact committed sources that can enter an annotator delivery kit."""
    root = repository_root.resolve()
    violations = list(
        scan_delivery_tree(
            root / "pilot/v0.2/packages",
            allowed_files=("ann-pilot-a.json", "ann-pilot-b.json"),
        )
    )
    for relative in _DELIVERY_TEXT_SOURCES:
        path = root / relative
        if not path.is_file():
            violations.append(f"{relative}: delivery source is missing")
            continue
        violations.extend(_scan_delivery_text(relative, path.read_text(encoding="utf-8")))
    return tuple(sorted(set(violations)))
