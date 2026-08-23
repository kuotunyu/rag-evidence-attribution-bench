"""Breaking-v2 annotation models reject every legacy group-bearing input."""

from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

import rag_evidence.annotation.models as models


def _sha(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _task_payload() -> dict[str, object]:
    content = {
        "challenge_id": "ch-0123456789abcdef01234567",
        "instruction_version": "pilot-v0.2.2-draft",
        "instruction_hash": "1" * 64,
        "question": "Which city is the architect from?",
        "passages": [
            {
                "alias": "P1",
                "title": "Architect",
                "sentences": [{"alias": "P1.S1", "text": "The architect is from Riverton."}],
            }
        ],
    }
    return {
        "schema_version": "blind-task-v2",
        "annotation_task_id": "task-0123456789abcdef01234567",
        **content,
        "task_content_hash": _sha(content),
        "assignment_batch": "pilot-v0.2-smoke",
    }


def _answer_payload() -> dict[str, object]:
    return {
        "schema_version": "answerability-annotation-v2",
        "annotation_task_id": "task-0123456789abcdef01234567",
        "challenge_id": "ch-0123456789abcdef01234567",
        "annotator_pseudonym": "ann-pilot-a",
        "instruction_version": "pilot-v0.2.2-draft",
        "instruction_hash": "1" * 64,
        "task_content_hash": _task_payload()["task_content_hash"],
        "assignment_batch": "pilot-v0.2-smoke",
        "answerability": "answerable",
        "answer_text": "Riverton",
        "minimal_sufficient_evidence_sets": [["P1.S1"]],
        "evidence_exhaustive": True,
        "ambiguity": False,
        "dataset_defect": False,
        "confidence": 5,
        "rationale": "The visible sentence states the city.",
        "started_at": "2026-08-24T00:00:00Z",
        "submitted_at": "2026-08-24T00:01:00Z",
    }


def test_blind_task_v2_has_exact_group_free_fields() -> None:
    model = getattr(models, "BlindTaskV2", None)
    assert model is not None, "BlindTaskV2 must be implemented"

    task = model.model_validate(_task_payload())

    assert tuple(task.__class__.model_fields) == (
        "schema_version",
        "annotation_task_id",
        "challenge_id",
        "instruction_version",
        "instruction_hash",
        "question",
        "passages",
        "task_content_hash",
        "assignment_batch",
    )
    assert "group" not in json.dumps(task.model_dump(mode="json")).casefold()


def test_answerability_v2_binding_excludes_group_metadata() -> None:
    model = getattr(models, "AnswerabilityAnnotationV2", None)
    assert model is not None, "AnswerabilityAnnotationV2 must be implemented"

    record = model.model_validate(_answer_payload())

    assert record.task_content_hash == _task_payload()["task_content_hash"]
    assert not any("group" in name.casefold() for name in record.__class__.model_fields)


@pytest.mark.parametrize(
    ("model_name", "payload"),
    [
        (
            "BlindTaskV2",
            {
                **_task_payload(),
                "schema_version": "blind-task-v1",
                "blinded_parent_group": "bg-0123456789abcdef01234567",
            },
        ),
        (
            "AnswerabilityAnnotationV2",
            {
                **_answer_payload(),
                "schema_version": "answerability-annotation-v1",
                "blinded_parent_group": "bg-0123456789abcdef01234567",
            },
        ),
    ],
)
def test_v1_group_bearing_records_fail_closed(
    model_name: str,
    payload: dict[str, object],
) -> None:
    model = getattr(models, model_name, None)
    assert model is not None, f"{model_name} must be implemented"

    with pytest.raises(ValidationError):
        model.model_validate(payload)
