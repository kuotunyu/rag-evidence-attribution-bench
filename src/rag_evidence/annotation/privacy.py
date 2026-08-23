"""Reusable privacy and hidden-source-field scanning for annotation artifacts."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

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

