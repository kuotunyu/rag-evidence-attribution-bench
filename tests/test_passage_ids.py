"""Stable passage/sentence IDs and prompt-alias mapping."""

from __future__ import annotations

import json
from pathlib import Path

from rag_evidence.data import ids
from rag_evidence.data.hotpot import build_example, normalize_hf_example


def _rows(fixtures_dir: Path) -> list[dict]:
    return json.loads((fixtures_dir / "tiny_hotpot.json").read_text(encoding="utf-8"))["rows"]


def test_id_formats() -> None:
    pid = ids.passage_id("abc123", 3)
    assert pid == "abc123-p03"
    assert ids.sentence_id(pid, 0) == "abc123-p03-s00"
    assert ids.parse_passage_id(pid) == ("abc123", 3)


def test_ids_stable_across_calls(fixtures_dir: Path) -> None:
    row = _rows(fixtures_dir)[0]
    e1 = build_example(normalize_hf_example(row))
    e2 = build_example(normalize_hf_example(json.loads(json.dumps(row))))
    assert [p.passage_id for p in e1.passages] == [p.passage_id for p in e2.passages]
    assert e1.supporting_fact_sentence_ids == e2.supporting_fact_sentence_ids


def test_alias_map_order_and_inversion() -> None:
    pids = ["q-p00", "q-p01", "q-p02"]
    amap = ids.make_alias_map(pids)
    assert amap == {"P1": "q-p00", "P2": "q-p01", "P3": "q-p02"}
    assert ids.invert_alias_map(amap)["q-p01"] == "P2"
    assert ids.is_alias("P10") and not ids.is_alias("P0") and not ids.is_alias("Q1")


def test_example_roundtrip_json(fixtures_dir: Path) -> None:
    from rag_evidence.data.schema import Example

    example = build_example(normalize_hf_example(_rows(fixtures_dir)[0]))
    assert Example.from_json(example.to_json()) == example
