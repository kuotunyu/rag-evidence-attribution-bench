"""The local API exposes blind tasks and validates immutable human submissions."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from rag_evidence.annotation.app import create_annotation_app
from test_annotation_store import _package, _submission


def _client(tmp_path: Path) -> tuple[TestClient, object]:
    package = _package()
    package_path = tmp_path / "package.json"
    package_path.write_text(
        json.dumps(package.blind_export(), ensure_ascii=False),
        encoding="utf-8",
    )
    return TestClient(create_annotation_app(package_path, tmp_path / "state")), package


def test_browser_and_api_receive_no_source_metadata(tmp_path: Path) -> None:
    client, package = _client(tmp_path)
    response = client.get("/api/tasks")

    assert response.status_code == 200
    payload = response.json()
    serialized = json.dumps(payload, sort_keys=True)
    assert payload["annotator_pseudonym"] == package.annotator_pseudonym
    assert payload["instruction_hash"] == "1" * 64
    for forbidden in (
        "transformation",
        "expected_answerability",
        "changed_fields",
        "provenance",
        "is_gold",
        "parent_question_id",
        "model_name",
        "method_name",
        "score",
    ):
        assert forbidden not in serialized

    html = client.get("/")
    assert html.status_code == 200
    assert "Evidence ledger" in html.text
    assert "Add evidence set" in html.text
    assert "Submit immutable decision" in html.text
    assert "expected_answerability" not in html.text
    assert "https://" not in html.text


def test_autosave_resume_submit_progress_and_export(tmp_path: Path) -> None:
    client, package = _client(tmp_path)
    task = package.tasks[0]
    draft = {"answerability": "unclear", "confidence": 2, "rationale": "Needs review."}

    assert client.put(f"/api/drafts/{task.annotation_task_id}", json=draft).status_code == 200
    assert client.get(f"/api/drafts/{task.annotation_task_id}").json()["payload"] == draft

    submission = _submission(package)
    submitted = client.post("/api/submissions", json=submission)
    assert submitted.status_code == 201
    assert client.post("/api/submissions", json=submission).status_code == 409
    assert client.get("/api/progress").json() == {
        "total": 3,
        "submitted": 1,
        "remaining": 2,
    }

    exported = client.get("/api/export/submissions")
    assert exported.status_code == 200
    assert exported.json()[0]["answer_text"] == "Riverton"
    jsonl = client.get("/api/export/submissions.jsonl")
    assert jsonl.status_code == 200
    assert len(jsonl.text.strip().splitlines()) == 1


def test_api_returns_validation_errors_without_mutating_state(tmp_path: Path) -> None:
    client, package = _client(tmp_path)
    invalid = _submission(package)
    invalid.pop("rationale")

    response = client.post("/api/submissions", json=invalid)

    assert response.status_code == 422
    assert client.get("/api/progress").json()["submitted"] == 0
    assert client.get("/api/export/submissions").json() == []


def test_unknown_task_is_not_disclosed(tmp_path: Path) -> None:
    client, _package_value = _client(tmp_path)
    task_id = "task-0123456789abcdef01234567"

    assert client.get(f"/api/tasks/{task_id}").status_code == 404
    assert (
        client.put(f"/api/drafts/{task_id}", json={"answerability": "unclear"}).status_code == 404
    )
