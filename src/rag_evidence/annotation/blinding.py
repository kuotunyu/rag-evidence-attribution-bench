"""Server-side allowlist projection for independent annotators."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence

from rag_evidence.annotation.models import BlindTask
from rag_evidence.data.challenge_schema import ChallengeRecord
from rag_evidence.errors import DataError

_FORBIDDEN_KEYS = frozenset(
    {
        "transformation",
        "transform_version",
        "expected_answerability",
        "changed_fields",
        "provenance",
        "original_sibling_record",
        "sibling",
        "parent_question_id",
        "parent_fingerprint",
        "source_split",
        "model",
        "model_name",
        "method",
        "method_name",
        "score",
        "scores",
        "abstention_expectation",
        "gold_adjudicated_label",
        "gold_passage_ids",
        "supporting_fact_sentence_ids",
        "is_gold",
        "answer",
        "answer_text",
        "seed",
    }
)
_EMAIL_RE = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_WINDOWS_PATH_RE = re.compile(r"(?:^|\s)[A-Za-z]:[\\/]")
_POSIX_PRIVATE_PATH_RE = re.compile(r"/(?:home|users|private|root)/", flags=re.IGNORECASE)
_INTERNAL_PASSAGE_RE = re.compile(r"\bch-[0-9a-f]{24}-p\d{2,}(?:-s\d{2,})?\b")


def _hash(payload: object) -> str:
    material = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _opaque_id(prefix: str, *parts: str) -> str:
    return f"{prefix}-{_hash(list(parts))[:24]}"


def scan_blind_payload(payload: object) -> tuple[str, ...]:
    """Return every hidden-field, PII, path, or internal-ID violation."""
    violations: list[str] = []

    def visit(value: object, path: str) -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key).casefold()
                child_path = f"{path}.{raw_key}"
                if key in _FORBIDDEN_KEYS:
                    violations.append(f"{child_path}: forbidden key")
                elif key.startswith("expected_") or key.startswith("gold_"):
                    violations.append(f"{child_path}: forbidden label key")
                elif key.endswith("_score") or key.endswith("_scores"):
                    violations.append(f"{child_path}: forbidden score key")
                visit(child, child_path)
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
            return
        if isinstance(value, str):
            if _EMAIL_RE.search(value):
                violations.append(f"{path}: email/PII-like value")
            if _WINDOWS_PATH_RE.search(value) or _POSIX_PRIVATE_PATH_RE.search(value):
                violations.append(f"{path}: private filesystem path")
            if _INTERNAL_PASSAGE_RE.search(value):
                violations.append(f"{path}: internal passage/sentence ID")

    visit(payload, "$.")
    return tuple(sorted(set(violations)))


def project_challenge(
    record: ChallengeRecord,
    *,
    instruction_version: str,
    instruction_hash: str,
    batch: str,
    namespace: str,
) -> BlindTask:
    """Construct a blind task field-by-field; source metadata is never serialized."""
    blinded_parent_group = _opaque_id("bg", namespace, record.parent_question_id)
    passages = [
        {
            "alias": f"P{passage_index}",
            "title": passage.title,
            "sentences": [
                {
                    "alias": f"P{passage_index}.S{sentence_index}",
                    "text": sentence,
                }
                for sentence_index, sentence in enumerate(passage.sentences, start=1)
            ],
        }
        for passage_index, passage in enumerate(record.example.passages, start=1)
    ]
    content = {
        "challenge_id": record.challenge_id,
        "blinded_parent_group": blinded_parent_group,
        "instruction_version": instruction_version,
        "instruction_hash": instruction_hash,
        "question": record.example.question,
        "passages": passages,
    }
    payload = {
        "schema_version": "blind-task-v1",
        "annotation_task_id": _opaque_id("task", namespace, batch, record.challenge_id),
        **content,
        "task_content_hash": _hash(content),
        "assignment_batch": batch,
    }
    violations = scan_blind_payload(payload)
    if violations:
        raise DataError("blind projection leak: " + "; ".join(violations))
    return BlindTask.model_validate(payload)


def validate_blind_task_source(task: BlindTask, record: ChallengeRecord) -> None:
    """Re-bind an immutable blind task to the current visible challenge content."""
    if task.challenge_id != record.challenge_id:
        raise DataError("blind task challenge ID does not match source record")
    expected_passages = [
        {
            "alias": f"P{passage_index}",
            "title": passage.title,
            "sentences": [
                {"alias": f"P{passage_index}.S{sentence_index}", "text": sentence}
                for sentence_index, sentence in enumerate(passage.sentences, start=1)
            ],
        }
        for passage_index, passage in enumerate(record.example.passages, start=1)
    ]
    if (
        task.question != record.example.question
        or [passage.model_dump(mode="json") for passage in task.passages] != expected_passages
    ):
        raise DataError("blind task visible content does not match challenge source")
