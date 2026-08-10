"""Leakage groups join questions by normalized titles or canonical paragraphs."""

from __future__ import annotations

from typing import Any

from rag_evidence.data.grouping import (
    build_split_groups,
    normalize_title,
    paragraph_fingerprint,
    supporting_fact_errors,
)


def _raw(
    qid: str,
    context: list[list[Any]],
    *,
    qtype: str = "bridge",
    supporting_facts: list[list[Any]] | None = None,
) -> dict[str, Any]:
    title = str(context[0][0])
    return {
        "question_id": qid,
        "question": f"Question {qid}?",
        "answer": f"Answer {qid}",
        "type": qtype,
        "level": "medium",
        "context": context,
        "supporting_facts": supporting_facts or [[title, 0]],
    }


def test_canonical_identifiers_normalize_only_declared_variation() -> None:
    assert normalize_title("  \uFF23afé\tLAND  ") == "café land"
    assert paragraph_fingerprint(["\uFF21 sentence. ", "Two\tspaces."]) == paragraph_fingerprint(
        ["A sentence.", "Two spaces."]
    )
    assert paragraph_fingerprint(["First.", "Second."]) != paragraph_fingerprint(
        ["Second.", "First."]
    )
    assert paragraph_fingerprint(["Case matters."]) != paragraph_fingerprint(["case matters."])


def test_shared_title_and_paragraph_form_transitive_order_independent_groups() -> None:
    shared_title = _raw("q-a", [["Shared Title", ["Alpha paragraph."]]])
    bridge = _raw(
        "q-b",
        [
            [" shared   title ", ["Different paragraph."]],
            ["Link", ["Common\tparagraph."]],
        ],
    )
    shared_paragraph = _raw("q-c", [["Other", ["Common paragraph."]]])
    isolated = _raw("q-d", [["Isolated", ["No shared content."]]], qtype="comparison")
    raws = [shared_title, bridge, shared_paragraph, isolated]

    groups = build_split_groups(raws)

    assert [group.question_ids for group in groups] == [("q-a", "q-b", "q-c"), ("q-d",)]
    assert [group.size for group in groups] == [3, 1]
    assert groups[0].bridge_count == 3
    assert groups[0].comparison_count == 0
    assert groups[1].bridge_count == 0
    assert groups[1].comparison_count == 1
    assert all(len(group.group_id) == 64 for group in groups)
    assert build_split_groups(list(reversed(raws))) == groups


def test_supporting_fact_errors_reject_every_unresolvable_annotation() -> None:
    valid = _raw(
        "valid",
        [["One", ["Sentence zero.", "Sentence one."]]],
        supporting_facts=[["One", 1]],
    )
    missing = _raw(
        "missing",
        [["One", ["Sentence zero."]]],
        supporting_facts=[["Absent", 0]],
    )
    out_of_range = _raw(
        "range",
        [["One", ["Sentence zero."]]],
        supporting_facts=[["One", 2]],
    )
    duplicate = _raw(
        "duplicate",
        [["One", ["First."]], ["One", ["Second."]]],
        supporting_facts=[["One", 0]],
    )

    assert supporting_fact_errors(valid) == ()
    assert supporting_fact_errors(missing) == ("title_not_in_context",)
    assert supporting_fact_errors(out_of_range) == ("sent_id_out_of_range",)
    assert supporting_fact_errors(duplicate) == ("duplicate_context_title",)
