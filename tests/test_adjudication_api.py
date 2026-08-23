"""The third-human console exposes disagreements without expected-label leakage."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from rag_evidence.annotation.app import create_adjudication_app
from test_annotation_workflow import _adjudication, _annotation, _answerable, _manifest


def _client(tmp_path: Path, *, disagree: bool = True):
    manifest = _manifest()
    left = _answerable(manifest, "ann-r7")
    right = (
        _annotation(manifest, "ann-k2", answerability="unanswerable", exhaustive=None)
        if disagree
        else _answerable(manifest, "ann-k2")
    )
    manifest_path = tmp_path / "assignment-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False), encoding="utf-8"
    )
    submissions_path = tmp_path / "submissions.jsonl"
    submissions_path.write_text(
        "\n".join(
            json.dumps(record.model_dump(mode="json"), ensure_ascii=False)
            for record in (left, right)
        )
        + "\n",
        encoding="utf-8",
    )
    app = create_adjudication_app(manifest_path, submissions_path, tmp_path / "state")
    return TestClient(app), left, right


def test_adjudicator_sees_both_originals_and_visible_task_but_no_expected_label(
    tmp_path: Path,
) -> None:
    client, _left, _right = _client(tmp_path)

    response = client.get("/api/disagreements")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["task"]["question"] == "Which city is the architect from?"
    assert {payload[0]["left"]["answerability"], payload[0]["right"]["answerability"]} == {
        "answerable",
        "unanswerable",
    }
    serialized = json.dumps(payload, sort_keys=True)
    for forbidden in ("expected_answerability", "transformation", "changed_fields", "provenance"):
        assert forbidden not in serialized

    html = client.get("/")
    assert html.status_code == 200
    assert "Adjudication desk" in html.text
    assert "Preserve both originals" in html.text


def test_adjudication_submission_is_immutable(tmp_path: Path) -> None:
    client, left, right = _client(tmp_path)
    payload = _adjudication(left, right).model_dump(mode="json")

    response = client.post("/api/adjudications", json=payload)

    assert response.status_code == 201
    assert client.post("/api/adjudications", json=payload).status_code == 409
    exported = client.get("/api/adjudications").json()
    assert len(exported) == 1
    assert exported[0]["left"]["rationale"] == left.rationale


def test_adjudication_status_tracks_append_only_completion(tmp_path: Path) -> None:
    client, left, right = _client(tmp_path)

    assert client.get("/api/status").json() == {
        "total": 1,
        "adjudicated": 0,
        "remaining": 1,
        "complete": False,
    }
    client.post("/api/adjudications", json=_adjudication(left, right).model_dump(mode="json"))
    assert client.get("/api/status").json() == {
        "total": 1,
        "adjudicated": 1,
        "remaining": 0,
        "complete": True,
    }


def test_zero_disagreement_status_and_export_are_complete(tmp_path: Path) -> None:
    client, _left, _right = _client(tmp_path, disagree=False)

    assert client.get("/api/status").json() == {
        "total": 0,
        "adjudicated": 0,
        "remaining": 0,
        "complete": True,
    }
    response = client.get("/api/export/adjudications.jsonl")
    assert response.status_code == 200
    assert response.text == ""
    assert response.headers["cache-control"] == "no-store"


def test_adjudication_export_is_canonical_jsonl(tmp_path: Path) -> None:
    client, left, right = _client(tmp_path)
    payload = _adjudication(left, right).model_dump(mode="json")
    client.post("/api/adjudications", json=payload)

    response = client.get("/api/export/adjudications.jsonl")

    assert response.status_code == 200
    assert (
        response.text
        == json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def test_adjudication_rejects_private_data_and_has_no_correction_endpoint(
    tmp_path: Path,
) -> None:
    client, left, right = _client(tmp_path)
    payload = _adjudication(left, right).model_dump(mode="json")
    payload["rationale"] = "Contact reviewer@example.com before deciding."

    response = client.post("/api/adjudications", json=payload)

    assert response.status_code == 400
    assert "privacy" in response.json()["detail"]
    assert client.put("/api/adjudications", json=payload).status_code == 405
    assert client.delete("/api/adjudications").status_code == 405
    assert client.get("/api/adjudications").json() == []
