"""JSONL checkpoint integrity: append/read roundtrip, crash tolerance, dedup keys."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_evidence.errors import ArtifactError
from rag_evidence.storage.artifacts import (
    append_record,
    completed_keys,
    read_json,
    read_records,
    write_json_atomic,
    write_records_atomic,
)


def test_append_read_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    records = [
        {"question_id": "q1", "value": 1.5, "text": "中文 & unicode ✓"},
        {"question_id": "q2", "value": None, "nested": {"a": [1, 2]}},
    ]
    for rec in records:
        append_record(path, rec)
    assert list(read_records(path)) == records


def test_truncated_final_line_is_dropped(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    append_record(path, {"question_id": "q1"})
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"question_id": "q2", "val')  # simulated crash mid-write
    recs = list(read_records(path))
    assert [r["question_id"] for r in recs] == ["q1"]


def test_malformed_midfile_raises(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        fh.write('{"question_id": "q1"}\n')
        fh.write("NOT JSON\n")
        fh.write('{"question_id": "q3"}\n')
    with pytest.raises(ArtifactError, match="malformed JSON mid-file"):
        list(read_records(path))


def test_completed_keys_simple_and_composite(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    append_record(path, {"question_id": "q1", "mode": "gold"})
    append_record(path, {"question_id": "q1", "mode": "generated"})
    assert completed_keys(path) == {("q1",)}
    assert completed_keys(path, ("question_id", "mode")) == {
        ("q1", "gold"),
        ("q1", "generated"),
    }
    assert completed_keys(tmp_path / "missing.jsonl") == set()


def test_completed_keys_can_leave_failed_records_for_retry(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    append_record(path, {"question_id": "ok", "error": None})
    append_record(path, {"question_id": "failed", "error": {"type": "RuntimeError"}})
    append_record(path, {"question_id": "skipped", "skipped": True, "error": None})

    assert completed_keys(path) == {("ok",), ("failed",), ("skipped",)}
    assert completed_keys(path, include_failed=False) == {("ok",), ("skipped",)}


def test_write_json_atomic_roundtrip_and_no_temp_left(tmp_path: Path) -> None:
    path = tmp_path / "meta.json"
    obj = {"b": 1, "a": {"nested": True}}
    write_json_atomic(path, obj)
    assert read_json(path) == obj
    leftovers = [p for p in tmp_path.iterdir() if ".tmp." in p.name]
    assert leftovers == []


def test_write_records_atomic_replaces_with_compact_unicode_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text('{"stale":true}\n', encoding="utf-8")
    records = (
        {"question_id": "q1", "text": "證據"},
        {"question_id": "q2", "nested": {"ok": True}},
    )

    write_records_atomic(path, records)

    assert list(read_records(path)) == list(records)
    assert (
        path.read_bytes()
        == (
            '{"question_id":"q1","text":"證據"}\n{"question_id":"q2","nested":{"ok":true}}\n'
        ).encode()
    )
    assert not list(tmp_path.glob("records.jsonl.tmp.*"))


def test_write_records_atomic_preserves_target_after_serialization_failure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "records.jsonl"
    original = b'{"question_id":"original"}\n'
    path.write_bytes(original)

    with pytest.raises(ArtifactError, match="failed to write"):
        write_records_atomic(path, ({"not_json": object()},))

    assert path.read_bytes() == original
    assert not list(tmp_path.glob("records.jsonl.tmp.*"))
