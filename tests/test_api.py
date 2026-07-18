"""FastAPI endpoints over precomputed tiny results (TestClient, no server)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag_evidence.api import create_app
from rag_evidence.config import AppConfig
from rag_evidence.generation.run import run_generation_stage
from rag_evidence.retrieval.run import run_retrieval_stage


@pytest.fixture()
def client(tiny_env: AppConfig) -> TestClient:
    cfg = tiny_env
    run_retrieval_stage(cfg, method="bm25", resume=False, limit=None)
    run_generation_stage(cfg, resume=False, limit=None)
    from rag_evidence.attribution.run import run_attribution_stage

    run_attribution_stage(cfg, method="control_random", mode=None, resume=False, limit=None)
    return TestClient(create_app(cfg))


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["mode"] == "precomputed"
    assert body["n_samples"] == 3


def test_methods(client: TestClient) -> None:
    body = client.get("/methods").json()
    assert body["retrieval"] == ["bm25"]
    assert body["generation"] == ["fake"]
    assert body["attribution"] == {
        "generated": ["control_random"],
        "gold": ["control_random"],
    }


def test_benchmark_record_and_404(client: TestClient) -> None:
    qid = sorted(client.app.state.store.samples)[0]
    body = client.get(f"/benchmark/{qid}").json()
    assert body["sample"]["question_id"] == qid
    assert "bm25" in body["retrieval"]
    assert "fake" in body["generation"]
    assert client.get("/benchmark/does-not-exist").status_code == 404


def test_retrieve_endpoint(client: TestClient) -> None:
    qid = sorted(client.app.state.store.samples)[0]
    body = client.post("/retrieve", json={"sample_id": qid, "method": "bm25", "k": 2}).json()
    assert len(body["ranking"]) == 2
    assert body["ranking"][0]["rank"] == 1
    # unknown method → 400 with the available list in detail
    r = client.post("/retrieve", json={"sample_id": qid, "method": "dense"})
    assert r.status_code == 400
    assert "bm25" in r.json()["detail"]


def test_answer_endpoint(client: TestClient) -> None:
    qid = sorted(client.app.state.store.samples)[0]
    body = client.post("/answer", json={"sample_id": qid}).json()
    assert body["question_id"] == qid
    assert body["response_text"]


def test_attribute_endpoint_and_validation(client: TestClient) -> None:
    qid = sorted(client.app.state.store.samples)[0]
    body = client.post(
        "/attribute", json={"sample_id": qid, "method": "control_random", "mode": "gold"}
    ).json()
    assert set(body["raw_scores"]) == set(body["context_passage_ids"])
    assert client.post("/attribute", json={"sample_id": qid}).status_code == 422  # method missing
    r = client.post("/attribute", json={"sample_id": qid, "method": "nope", "mode": "gold"})
    assert r.status_code == 400


def test_root_redirects_to_app(client: TestClient) -> None:
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/app"


def test_live_mode_refused(tiny_env: AppConfig, tmp_path) -> None:
    import json

    import yaml

    from rag_evidence.config import load_config
    from rag_evidence.errors import ConfigError

    raw = json.loads(tiny_env.model_dump_json())
    raw["serve"]["mode"] = "live"
    p = tmp_path / "live.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="live"):
        create_app(load_config(p))
