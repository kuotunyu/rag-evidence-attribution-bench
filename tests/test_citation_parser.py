"""Citation parsing/stripping and abstention detection."""

from __future__ import annotations

from rag_evidence.generation.citations import is_abstention, parse_citations, strip_citations

ALIAS_MAP = {f"P{i}": f"q-p{i - 1:02d}" for i in range(1, 6)}  # P1..P5


def test_basic_forms() -> None:
    for text in ("answer [P1][P3]", "answer [P1, P3]", "answer [P1 P3]", "answer [p1] [p3]"):
        parsed = parse_citations(text, ALIAS_MAP)
        assert parsed.raw_aliases == ("P1", "P3"), text
        assert parsed.cited_passage_ids == ("q-p00", "q-p02")
        assert parsed.invalid_aliases == ()


def test_duplicates_deduplicated_in_order() -> None:
    parsed = parse_citations("x [P2][P1][P2, P1]", ALIAS_MAP)
    assert parsed.raw_aliases == ("P2", "P1")


def test_out_of_range_alias_is_invalid() -> None:
    parsed = parse_citations("x [P1][P11]", ALIAS_MAP)
    assert parsed.cited_passage_ids == ("q-p00",)
    assert parsed.invalid_aliases == ("P11",)


def test_no_citations() -> None:
    parsed = parse_citations("plain answer with no brackets", ALIAS_MAP)
    assert parsed.raw_aliases == ()
    assert parsed.cited_passage_ids == ()


def test_non_citation_brackets_untouched() -> None:
    parsed = parse_citations("see [note 3] and [P2]", ALIAS_MAP)
    assert parsed.raw_aliases == ("P2",)
    assert strip_citations("see [note 3] and [P2]") == "see [note 3] and"


def test_strip_citations() -> None:
    assert strip_citations("Blueport [P1][P2]") == "Blueport"
    assert strip_citations("[P1, P2] yes") == "yes"
    assert strip_citations("Blueport.") == "Blueport"


def test_abstention_detection() -> None:
    assert is_abstention("INSUFFICIENT EVIDENCE")
    assert is_abstention("insufficient evidence [P1]")
    assert not is_abstention("the evidence is sufficient")
