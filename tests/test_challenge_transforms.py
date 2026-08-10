"""Deterministic controlled challenge transforms preserve explicit scientific gates."""

from __future__ import annotations

from dataclasses import replace

import pytest

from rag_evidence.data.challenge_transforms import (
    build_answer_bearing_distractor,
    build_challenge_records,
    build_evidence_swap,
    build_missing_hop,
)
from rag_evidence.data.schema import Example, Passage
from rag_evidence.errors import DataError
from rag_evidence.metrics.text import normalize_answer


def _passage(
    qid: str,
    index: int,
    title: str,
    sentences: tuple[str, ...],
    *,
    gold: bool,
) -> Passage:
    return Passage(
        passage_id=f"{qid}-p{index:02d}",
        index=index,
        title=title,
        sentences=sentences,
        is_gold=gold,
    )


def _parent_one() -> Example:
    qid = "q-one"
    return Example(
        question_id=qid,
        question="Who founded the Blue Harbor project?",
        answer="Ada Vale",
        level="hard",
        qtype="bridge",
        passages=(
            _passage(
                qid,
                0,
                "Blue Harbor",
                ("Blue Harbor is a project.", "It was founded by Ada Vale."),
                gold=True,
            ),
            _passage(
                qid,
                1,
                "River Link",
                ("River Link funded the project.",),
                gold=True,
            ),
            _passage(
                qid,
                2,
                "Copper Hill",
                ("Copper Hill is an inland settlement.",),
                gold=False,
            ),
        ),
        gold_passage_ids=(f"{qid}-p00", f"{qid}-p01"),
        supporting_fact_sentence_ids=(f"{qid}-p00-s01", f"{qid}-p01-s00"),
    )


def _parent_two() -> Example:
    qid = "q-two"
    return Example(
        question_id=qid,
        question="Where is the Silver Meadow observatory?",
        answer="North Vale",
        level="hard",
        qtype="bridge",
        passages=(
            _passage(
                qid,
                0,
                "Observatory",
                ("The observatory is in North Vale.",),
                gold=True,
            ),
            _passage(
                qid,
                1,
                "Blue Harbor",
                ("This title overlaps the first parent.",),
                gold=False,
            ),
            _passage(
                qid,
                2,
                "Answer Gazette",
                ("Ada Vale appeared in a local notice.",),
                gold=False,
            ),
            _passage(
                qid,
                3,
                "Silver Meadow",
                ("Silver Meadow hosts an annual astronomy fair.",),
                gold=False,
            ),
        ),
        gold_passage_ids=(f"{qid}-p00",),
        supporting_fact_sentence_ids=(f"{qid}-p00-s00",),
    )


def test_missing_hop_is_order_independent_and_uses_only_a_safe_same_split_donor() -> None:
    parent = _parent_one()
    peers = (parent, _parent_two())

    record = build_missing_hop(
        parent,
        peers,
        source_split="eval",
        seed=20260810,
        transform_version="challenge-v1",
        parent_fingerprint="a" * 64,
    )
    reversed_record = build_missing_hop(
        parent,
        tuple(reversed(peers)),
        source_split="eval",
        seed=20260810,
        transform_version="challenge-v1",
        parent_fingerprint="a" * 64,
    )

    assert record == reversed_record
    assert record.transformation == "missing_hop"
    assert record.expected_answerability == "unanswerable"
    assert record.review.status == "pending_two_annotators"
    assert not record.review.confirmatory_eligible
    assert record.provenance["donor_parent_question_id"] == "q-two"
    assert record.provenance["donor_source_passage_id"] == "q-two-p03"
    donor_slot = record.provenance["replaced_slot"]
    donor = record.example.passages[donor_slot]
    assert donor.title == "Silver Meadow"
    assert normalize_answer(parent.answer) not in normalize_answer(donor.text)
    assert len(record.example.gold_passage_ids) == len(parent.gold_passage_ids) - 1
    assert len(record.example.supporting_fact_sentence_ids) == (
        len(parent.supporting_fact_sentence_ids) - 1
    )
    assert parent.to_json() == _parent_one().to_json()


def test_missing_hop_rejects_donor_exhaustion_instead_of_weakening_filters() -> None:
    parent = _parent_one()
    unsafe = _parent_two()
    unsafe = replace(unsafe, passages=unsafe.passages[:3])

    with pytest.raises(DataError, match="donor"):
        build_missing_hop(
            parent,
            (parent, unsafe),
            source_split="eval",
            seed=20260810,
            transform_version="challenge-v1",
            parent_fingerprint="a" * 64,
        )


def test_answer_bearing_distractor_adds_one_sentence_and_retains_all_evidence() -> None:
    parent = _parent_one()

    record = build_answer_bearing_distractor(
        parent,
        source_split="eval",
        seed=20260810,
        transform_version="challenge-v1",
        parent_fingerprint="a" * 64,
    )

    assert record.transformation == "answer_bearing_distractor"
    assert record.expected_answerability == "answerable"
    assert record.review.status == "not_required"
    assert record.review.confirmatory_eligible
    source_slot = record.provenance["source_slot"]
    source_passage = parent.passages[source_slot]
    changed_passage = record.example.passages[source_slot]
    assert not source_passage.is_gold
    assert changed_passage.sentences[:-1] == source_passage.sentences
    assert changed_passage.sentences[-1] == (
        "Copper Hill has also been associated with Ada Vale."
    )
    assert record.provenance["inserted_sentence"] == changed_passage.sentences[-1]
    assert normalize_answer(parent.answer) in normalize_answer(changed_passage.sentences[-1])
    assert [passage.index for passage in record.example.passages if passage.is_gold] == [0, 1]
    assert record.example.supporting_fact_sentence_ids == (
        f"{record.challenge_id}-p00-s01",
        f"{record.challenge_id}-p01-s00",
    )
    assert parent.to_json() == _parent_one().to_json()


def test_answer_bearing_distractor_requires_an_answer_free_non_gold_passage() -> None:
    parent = _parent_one()
    contaminated = replace(
        parent.passages[2], sentences=("Ada Vale is already present here.",)
    )
    parent = replace(parent, passages=(*parent.passages[:2], contaminated))

    with pytest.raises(DataError, match="answer-free"):
        build_answer_bearing_distractor(
            parent,
            source_split="eval",
            seed=20260810,
            transform_version="challenge-v1",
            parent_fingerprint="a" * 64,
        )


def _single_support_parent(*, answer: str, sentence: str) -> Example:
    qid = "q-evidence"
    return Example(
        question_id=qid,
        question="What does the evidence say?",
        answer=answer,
        level="hard",
        qtype="bridge",
        passages=(
            _passage(qid, 0, "Evidence", (sentence,), gold=True),
            _passage(
                qid,
                1,
                "Copper Hill",
                ("Copper Hill is an inland settlement.",),
                gold=False,
            ),
        ),
        gold_passage_ids=(f"{qid}-p00",),
        supporting_fact_sentence_ids=(f"{qid}-p00-s00",),
    )


@pytest.mark.parametrize(
    ("answer", "sentence", "operator", "expected_fragment"),
    [
        (
            "Ada Vale",
            "The project was founded by Ada Vale.",
            "answer_substitution",
            "founded by Copper Hill",
        ),
        (
            "North Vale",
            "The project is located by the harbor.",
            "negation_toggle",
            "is not located",
        ),
        (
            "North Vale",
            "Ships arrived yesterday.",
            "negation_prefix",
            "It is not true that Ships arrived yesterday.",
        ),
    ],
)
def test_evidence_swap_uses_a_recorded_surface_preserving_operator(
    answer: str, sentence: str, operator: str, expected_fragment: str
) -> None:
    parent = _single_support_parent(answer=answer, sentence=sentence)

    record = build_evidence_swap(
        parent,
        source_split="eval",
        seed=20260810,
        transform_version="challenge-v1",
        parent_fingerprint="a" * 64,
    )

    assert record.transformation == "evidence_swap"
    assert record.expected_answerability == "unanswerable"
    assert record.review.status == "pending_two_annotators"
    assert not record.review.confirmatory_eligible
    assert record.provenance["edit_operator"] == operator
    assert record.provenance["original_sentence"] == sentence
    assert expected_fragment in record.provenance["edited_sentence"]
    assert record.example.passages[0].sentences[0] == record.provenance["edited_sentence"]
    assert record.example.supporting_fact_sentence_ids == ()
    assert record.example.gold_passage_ids == ()
    assert not record.example.passages[0].is_gold
    assert parent.to_json() == _single_support_parent(
        answer=answer, sentence=sentence
    ).to_json()


def test_evidence_swap_retains_passage_gold_label_when_another_support_remains() -> None:
    parent = _single_support_parent(
        answer="North Vale", sentence="The project is located by the harbor."
    )
    gold = replace(
        parent.passages[0],
        sentences=("The project is located by the harbor.", "It opened in 1999."),
    )
    parent = replace(
        parent,
        passages=(gold, parent.passages[1]),
        supporting_fact_sentence_ids=("q-evidence-p00-s00", "q-evidence-p00-s01"),
    )

    record = build_evidence_swap(
        parent,
        source_split="eval",
        seed=20260810,
        transform_version="challenge-v1",
        parent_fingerprint="a" * 64,
    )

    assert record.example.gold_passage_ids == (f"{record.challenge_id}-p00",)
    assert len(record.example.supporting_fact_sentence_ids) == 1
    assert record.example.passages[0].is_gold


@pytest.mark.parametrize("missing", ["supporting", "non_gold"])
def test_evidence_swap_requires_supporting_and_non_gold_candidates(missing: str) -> None:
    parent = _single_support_parent(
        answer="North Vale", sentence="The project is located by the harbor."
    )
    if missing == "supporting":
        parent = replace(parent, supporting_fact_sentence_ids=())
    else:
        parent = replace(parent, passages=parent.passages[:1])

    with pytest.raises(DataError, match=r"supporting|non-gold"):
        build_evidence_swap(
            parent,
            source_split="eval",
            seed=20260810,
            transform_version="challenge-v1",
            parent_fingerprint="a" * 64,
        )


def test_collection_is_reorder_stable_and_emits_three_unique_records_per_parent() -> None:
    parents = (_parent_one(), _parent_two())
    fingerprints = {"q-one": "a" * 64, "q-two": "b" * 64}

    records = build_challenge_records(
        {"eval": parents},
        fingerprints,
        seed=20260810,
        transform_version="challenge-v1",
    )
    reversed_records = build_challenge_records(
        {"eval": tuple(reversed(parents))},
        dict(reversed(tuple(fingerprints.items()))),
        seed=20260810,
        transform_version="challenge-v1",
    )

    assert records == reversed_records
    assert list(records) == ["eval"]
    assert len(records["eval"]) == 6
    assert [record.challenge_id for record in records["eval"]] == sorted(
        record.challenge_id for record in records["eval"]
    )
    assert len({record.challenge_id for record in records["eval"]}) == 6
    for parent_id in ("q-one", "q-two"):
        assert {
            record.transformation
            for record in records["eval"]
            if record.parent_question_id == parent_id
        } == {"missing_hop", "answer_bearing_distractor", "evidence_swap"}
    assert sum(record.expected_answerability == "answerable" for record in records["eval"]) == 2
    assert sum(
        record.expected_answerability == "unanswerable" for record in records["eval"]
    ) == 4
    assert parents[0].to_json() == _parent_one().to_json()
    assert parents[1].to_json() == _parent_two().to_json()
