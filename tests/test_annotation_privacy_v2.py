"""Delivery scanning rejects hidden metadata without censoring natural prose."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import rag_evidence.annotation.privacy as privacy


def scan_delivery_payload(payload: object, *, artifact_kind: str) -> tuple[str, ...]:
    function = getattr(privacy, "scan_delivery_payload", None)
    assert function is not None, "scan_delivery_payload must be implemented"
    return function(payload, artifact_kind=artifact_kind)


def scan_delivery_tree(root: Path) -> tuple[str, ...]:
    function = getattr(privacy, "scan_delivery_tree", None)
    assert function is not None, "scan_delivery_tree must be implemented"
    return function(root)


@pytest.mark.parametrize(
    "poison",
    [
        {"parent_question_id": "secret"},
        {"nested": {"sibling_id": "secret"}},
        {"group": "coord-group-01"},
        {"value": "bg-0123456789abcdef01234567"},
        {"transformation": "missing_hop"},
        {"expected_answerability": "unanswerable"},
        {"gold_label": "answerable"},
        {"schema_version": "answerability-annotation-v1"},
        {"seed": 20260823},
        {"score": 0.9},
    ],
)
def test_delivery_payload_rejects_private_keys_values_and_v1(poison: object) -> None:
    assert scan_delivery_payload(poison, artifact_kind="submission")


def test_delivery_payload_allows_parent_word_in_visible_prose() -> None:
    payload = {
        "schema_version": "blind-task-v2",
        "question": "Which parent agency manages the program?",
        "passages": [{"title": "Agency", "sentences": [{"text": "It is a parent agency."}]}],
    }

    assert scan_delivery_payload(payload, artifact_kind="blind_task") == ()


def test_delivery_tree_reports_relative_file_and_json_path(tmp_path: Path) -> None:
    root = tmp_path / "kit"
    root.mkdir()
    (root / "ann-pilot-a.json").write_text(
        json.dumps({"schema_version": "assignment-package-v2", "group_id": "secret"}),
        encoding="utf-8",
    )

    violations = scan_delivery_tree(root)

    assert violations
    assert "ann-pilot-a.json" in violations[0]
