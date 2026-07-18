"""Mocked end-to-end flow: retrieve → generate(FakeLM) → attribute x3 → evaluate →
report — including the README mock-gating rule (mock numbers never rendered)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rag_evidence.attribution.run import run_attribution_stage
from rag_evidence.config import AppConfig
from rag_evidence.evaluation.evaluate import evaluate_all
from rag_evidence.generation.run import run_generation_stage
from rag_evidence.reporting.report import BEGIN_MARK, END_MARK, build_report
from rag_evidence.retrieval.run import run_retrieval_stage
from rag_evidence.storage.artifacts import read_json

README_TEMPLATE = f"""# tiny readme

{BEGIN_MARK}
placeholder
{END_MARK}

tail text stays intact
"""


class _StubEmbedder:
    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(32, dtype=np.float32)
        for tok in text.lower().split():
            v[hash(tok) % 32] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_queries(self, texts, batch_size: int = 1):
        return np.stack([self._vec(t) for t in texts])

    def embed_passages(self, texts, batch_size: int = 1):
        return np.stack([self._vec(t) for t in texts])


@pytest.fixture()
def e2e_env(tiny_env: AppConfig, monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    from rag_evidence import embeddings as emb_mod

    monkeypatch.setattr(emb_mod.Embedder, "load", classmethod(lambda cls, *a, **k: _StubEmbedder()))
    cfg = tiny_env
    Path("README.md").write_text(README_TEMPLATE, encoding="utf-8")

    run_retrieval_stage(cfg, method="bm25", resume=False, limit=None)
    run_generation_stage(cfg, resume=False, limit=None)
    for method in ("leave_one_out", "embedding", "citations", "control_random"):
        run_attribution_stage(cfg, method=method, mode=None, resume=False, limit=None)
    evaluate_all(cfg)
    build_report(cfg)
    return cfg


def test_summary_structure(e2e_env: AppConfig) -> None:
    summary = read_json(Path(e2e_env.paths.results_derived) / "summary.json")
    smoke = summary["splits"]["smoke"]
    assert smoke["n_questions"] == 3
    assert "bm25" in smoke["retrieval"]
    assert smoke["retrieval"]["bm25"]["execution_kind"] == "real"
    assert smoke["generation"]["fake"]["execution_kind"] == "mock"
    assert set(smoke["attribution"]["gold"]) >= {
        "leave_one_out",
        "embedding",
        "control_random",
    }
    assert "citations" in smoke["attribution"]["generated"]
    assert summary["dataset_hash"]


def test_readme_gating_mock_never_rendered(e2e_env: AppConfig) -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    block = text.split(BEGIN_MARK)[1].split(END_MARK)[0]
    # real bm25 retrieval IS rendered (with its execution device)
    assert "| smoke | bm25 (cpu) |" in block
    # mock generation/attribution are NOT rendered — pending lines instead
    assert "fake" not in block
    assert "PENDING" in block
    assert "leave_one_out" not in block
    # non-block parts of the README are untouched
    assert "tail text stays intact" in text


def test_report_md_lists_excluded_mock_runs(e2e_env: AppConfig) -> None:
    report = (Path(e2e_env.paths.results_derived) / "report.md").read_text(encoding="utf-8")
    assert "Excluded from this report" in report
    assert "execution_kind: mock" in report
    assert "generate/fake" in report.replace("\\", "/")


def test_figures_written_for_real_data_only(e2e_env: AppConfig) -> None:
    assets = Path(e2e_env.paths.assets_dir)
    assert (assets / "retrieval_smoke.png").exists()  # real bm25
    assert not list(assets.glob("attribution_*"))  # all attribution runs were mock


def test_evaluate_is_idempotent(e2e_env: AppConfig) -> None:
    first = read_json(Path(e2e_env.paths.results_derived) / "summary.json")
    evaluate_all(e2e_env)
    second = read_json(Path(e2e_env.paths.results_derived) / "summary.json")
    first.pop("generated_utc")
    second.pop("generated_utc")
    assert first == second
