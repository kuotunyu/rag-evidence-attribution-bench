"""Strict challenge IDs, stable evidence IDs, review gates, and content hashes."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest

from rag_evidence.data.challenge_schema import (
    ChallengeRecord,
    make_challenge_id,
    record_content_hash,
    remap_example,
)
from rag_evidence.data.schema import Example, Passage
from rag_evidence.errors import DataError


def _source_example() -> Example:
    return Example(
        question_id="q-parent",
        question="Who founded Blue Harbor?",
        answer="Ada Vale",
        level="hard",
        qtype="bridge",
        passages=(
            Passage(
                passage_id="q-parent-p00",
                index=0,
                title="Blue Harbor",
                sentences=("Blue Harbor is a port.", "Ada Vale founded it."),
                is_gold=True,
            ),
            Passage(
                passage_id="q-parent-p01",
                index=1,
                title="Copper Hill",
                sentences=("Copper Hill is inland.",),
                is_gold=False,
            ),
        ),
        gold_passage_ids=("q-parent-p00",),
        supporting_fact_sentence_ids=("q-parent-p00-s01",),
    )


def _valid_record_json() -> dict[str, Any]:
    challenge_id = "ch-85b7c8651ecaec71e8b772cc"
    example = remap_example(
        _source_example(),
        challenge_id,
        passages=_source_example().passages,
        gold_slots={0},
        supporting_slots={(0, 1)},
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "challenge_id": challenge_id,
        "parent_question_id": "q-parent",
        "source_split": "eval",
        "transformation": "missing_hop",
        "transform_version": "challenge-v1",
        "seed": 20260810,
        "expected_answerability": "unanswerable",
        "review": {
            "required": True,
            "status": "pending_two_annotators",
            "confirmatory_eligible": False,
        },
        "changed_fields": [
            "/example/gold_passage_ids",
            "/example/passages/0",
            "/example/supporting_fact_sentence_ids",
        ],
        "provenance": {"removed_source_passage_id": "q-parent-p00"},
        "parent_fingerprint": "a" * 64,
        "example": example.to_json(),
    }
    return {**payload, "content_hash": record_content_hash(payload)}


def test_challenge_id_and_content_hash_have_hand_checked_values() -> None:
    assert (
        make_challenge_id(
            "q-parent",
            "eval",
            "missing_hop",
            transform_version="challenge-v1",
        )
        == "ch-85b7c8651ecaec71e8b772cc"
    )
    assert record_content_hash({"b": 2, "a": "evidence"}) == (
        "ec54faa689b855a5e40ee5e6786ec558ced9b66652fc2f8d3f2065c2d76bb94a"
    )
    assert record_content_hash({"a": "evidence", "b": 2}) == record_content_hash(
        {"b": 2, "a": "evidence"}
    )


def test_remap_example_rebuilds_all_stable_ids_from_slots() -> None:
    remapped = remap_example(
        _source_example(),
        "ch-85b7c8651ecaec71e8b772cc",
        passages=_source_example().passages,
        gold_slots={0},
        supporting_slots={(0, 1)},
    )

    assert remapped.question_id == "ch-85b7c8651ecaec71e8b772cc"
    assert [passage.passage_id for passage in remapped.passages] == [
        "ch-85b7c8651ecaec71e8b772cc-p00",
        "ch-85b7c8651ecaec71e8b772cc-p01",
    ]
    assert [passage.index for passage in remapped.passages] == [0, 1]
    assert [passage.is_gold for passage in remapped.passages] == [True, False]
    assert remapped.gold_passage_ids == ("ch-85b7c8651ecaec71e8b772cc-p00",)
    assert remapped.supporting_fact_sentence_ids == ("ch-85b7c8651ecaec71e8b772cc-p00-s01",)
    assert _source_example().question_id == "q-parent"


def test_challenge_record_round_trips_strictly() -> None:
    payload = _valid_record_json()

    record = ChallengeRecord.from_json(payload)

    assert record.to_json() == payload


def _add_unknown(payload: dict[str, Any]) -> None:
    payload["surprise"] = True


def _remove_parent(payload: dict[str, Any]) -> None:
    del payload["parent_question_id"]


def _duplicate_changed_field(payload: dict[str, Any]) -> None:
    payload["changed_fields"].append(payload["changed_fields"][0])


def _break_review_gate(payload: dict[str, Any]) -> None:
    payload["review"]["status"] = "not_required"


def _break_passage_id(payload: dict[str, Any]) -> None:
    payload["example"]["passages"][0]["passage_id"] = "q-parent-p00"


def _change_content_without_hash(payload: dict[str, Any]) -> None:
    payload["example"]["question"] = "Changed?"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_add_unknown, "fields"),
        (_remove_parent, "fields"),
        (_duplicate_changed_field, "changed_fields"),
        (_break_review_gate, "review"),
        (_break_passage_id, "passage"),
        (_change_content_without_hash, "content hash"),
    ],
)
def test_challenge_record_rejects_corrupt_or_noncanonical_payloads(
    mutation: Callable[[dict[str, Any]], None], message: str
) -> None:
    payload = copy.deepcopy(_valid_record_json())
    mutation(payload)

    with pytest.raises(DataError, match=message):
        ChallengeRecord.from_json(payload)
