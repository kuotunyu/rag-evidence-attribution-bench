"""Supporting-fact (title, sent_id) → stable ID mapping, including dataset noise."""

from __future__ import annotations

import json
from pathlib import Path

from rag_evidence.data.hotpot import build_example, normalize_hf_example


def _example(fixtures_dir: Path, i: int):  # noqa: ANN202
    rows = json.loads((fixtures_dir / "tiny_hotpot.json").read_text(encoding="utf-8"))["rows"]
    return build_example(normalize_hf_example(rows[i]))


def test_clean_mapping(fixtures_dir: Path) -> None:
    ex = _example(fixtures_dir, 0)
    qid = ex.question_id
    assert ex.gold_passage_ids == (f"{qid}-p00", f"{qid}-p01")
    assert ex.supporting_fact_sentence_ids == (f"{qid}-p00-s01", f"{qid}-p01-s01")
    assert [p.is_gold for p in ex.passages] == [True, True, False, False, False]
    assert ex.dropped_supporting_facts == ()


def test_out_of_range_sent_id_drops_sentence_keeps_passage(fixtures_dir: Path) -> None:
    ex = _example(fixtures_dir, 1)  # Iron Bloom has sent_id 7 with only 3 sentences
    qid = ex.question_id
    assert f"{qid}-p01" in ex.gold_passage_ids  # passage still gold
    assert all(not sid.startswith(f"{qid}-p01-s07") for sid in ex.supporting_fact_sentence_ids)
    assert len(ex.dropped_supporting_facts) == 1
    assert ex.dropped_supporting_facts[0]["reason"] == "sent_id_out_of_range"


def test_title_not_in_context_dropped(fixtures_dir: Path) -> None:
    ex = _example(fixtures_dir, 2)  # "Ghost Article" is not a context title
    assert len(ex.gold_passage_ids) == 2
    drops = [d for d in ex.dropped_supporting_facts if d["reason"] == "title_not_in_context"]
    assert len(drops) == 1 and drops[0]["title"] == "Ghost Article"
