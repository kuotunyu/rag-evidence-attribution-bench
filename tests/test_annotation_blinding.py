"""Blind projection is an allowlist, not a filtered source record."""

from __future__ import annotations

import json

import pytest

from rag_evidence.annotation.blinding import project_challenge, scan_blind_payload
from rag_evidence.data.challenge_schema import PENDING_REVIEW, ChallengeRecord
from rag_evidence.data.schema import Example, Passage
from rag_evidence.errors import DataError


def challenge_record(
    *,
    challenge_id: str = "ch-0123456789abcdef01234567",
    parent_id: str = "parent-secret-1",
    transformation: str = "missing_hop",
) -> ChallengeRecord:
    passages = (
        Passage(
            passage_id=f"{challenge_id}-p00",
            index=0,
            title="Bridge",
            sentences=("The bridge was designed by Noor.", "It opened in 1999."),
            is_gold=True,
        ),
        Passage(
            passage_id=f"{challenge_id}-p01",
            index=1,
            title="Noor",
            sentences=("Noor is from Riverton.",),
            is_gold=False,
        ),
    )
    return ChallengeRecord(
        challenge_id=challenge_id,
        parent_question_id=parent_id,
        source_split="smoke",
        transformation=transformation,  # type: ignore[arg-type]
        transform_version="challenge-v1",
        seed=20260810,
        expected_answerability="unanswerable",
        review=PENDING_REVIEW,
        changed_fields=("/example/passages/0",),
        provenance={"secret_donor": "donor-parent"},
        parent_fingerprint="f" * 64,
        content_hash="e" * 64,
        example=Example(
            question_id=challenge_id,
            question="Which city is the architect from?",
            answer="Riverton",
            level="hard",
            qtype="bridge",
            passages=passages,
            gold_passage_ids=(passages[0].passage_id,),
            supporting_fact_sentence_ids=(f"{passages[0].passage_id}-s00",),
        ),
    )


def test_projection_contains_only_public_aliases_and_visible_content() -> None:
    record = challenge_record()
    task = project_challenge(
        record,
        instruction_version="pilot-v0.2-draft",
        instruction_hash="1" * 64,
        batch="pilot-batch-01",
        namespace="pilot-v0.2",
    )
    payload = task.model_dump(mode="json")

    assert set(payload) == {
        "schema_version",
        "annotation_task_id",
        "challenge_id",
        "blinded_parent_group",
        "instruction_version",
        "instruction_hash",
        "question",
        "passages",
        "task_content_hash",
        "assignment_batch",
    }
    assert payload["passages"][0] == {
        "alias": "P1",
        "title": "Bridge",
        "sentences": [
            {"alias": "P1.S1", "text": "The bridge was designed by Noor."},
            {"alias": "P1.S2", "text": "It opened in 1999."},
        ],
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    for hidden in (
        "missing_hop",
        "unanswerable",
        "changed_fields",
        "secret_donor",
        "donor-parent",
        "parent-secret-1",
        'Riverton"',
        f"{record.challenge_id}-p00",
        "is_gold",
    ):
        assert hidden not in serialized
    assert scan_blind_payload(payload) == ()


def test_projection_is_stable_but_namespace_separates_parent_pseudonyms() -> None:
    record = challenge_record()
    kwargs = {
        "instruction_version": "pilot-v0.2-draft",
        "instruction_hash": "1" * 64,
        "batch": "pilot-batch-01",
    }
    first = project_challenge(record, namespace="pilot-v0.2", **kwargs)
    repeated = project_challenge(record, namespace="pilot-v0.2", **kwargs)
    other = project_challenge(record, namespace="confirmatory-v2", **kwargs)

    assert first == repeated
    assert first.blinded_parent_group != other.blinded_parent_group
    assert first.task_content_hash != other.task_content_hash


@pytest.mark.parametrize(
    "poison",
    [
        {"expected_answerability": "unanswerable"},
        {"nested": {"transformation": "evidence_swap"}},
        {"model_name": "Qwen"},
        {"score": 0.99},
        {"contact": "person@example.com"},
        {"path": r"C:\\private\\repo"},
        {"passage_id": "ch-0123456789abcdef01234567-p00"},
    ],
)
def test_leak_scanner_reports_hidden_keys_and_private_values(poison: dict[str, object]) -> None:
    violations = scan_blind_payload(poison)
    assert violations


def test_projection_refuses_a_payload_that_fails_its_own_leak_scan(monkeypatch) -> None:
    monkeypatch.setattr(
        "rag_evidence.annotation.blinding.scan_blind_payload",
        lambda _payload: ("synthetic leak",),
    )
    with pytest.raises(DataError, match="synthetic leak"):
        project_challenge(
            challenge_record(),
            instruction_version="pilot-v0.2-draft",
            instruction_hash="1" * 64,
            batch="pilot-batch-01",
            namespace="pilot-v0.2",
        )
