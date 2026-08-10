"""Release regeneration must be a pure raw-to-derived projection."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_evidence.attribution.run import run_attribution_stage
from rag_evidence.config import AppConfig
from rag_evidence.evaluation.evaluate import evaluate_all
from rag_evidence.generation.run import run_generation_stage
from rag_evidence.reporting.report import BEGIN_MARK, END_MARK, build_report
from rag_evidence.retrieval.run import run_retrieval_stage
from rag_evidence.storage.artifacts import RECORDS_FILE, RUN_META_FILE, read_json


def _derived_snapshot(cfg: AppConfig) -> dict[str, bytes]:
    root = cfg.results_derived_dir
    files = {
        str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }
    files["../README.md"] = Path("README.md").read_bytes()
    return files


@pytest.fixture()
def regeneration_env(tiny_env: AppConfig) -> AppConfig:
    paths = tiny_env.paths.model_copy(
        update={
            "results_derived": "results/v1/derived",
            "assets_dir": "results/v1/assets",
        }
    )
    cfg = tiny_env.model_copy(update={"paths": paths})
    Path("README.md").write_text(
        f"# tiny\n\n{BEGIN_MARK}\nplaceholder\n{END_MARK}\n\ntail\n",
        encoding="utf-8",
    )
    run_retrieval_stage(cfg, method="bm25", resume=False, limit=None)
    run_generation_stage(cfg, resume=False, limit=None)
    run_attribution_stage(cfg, method="leave_one_out", mode=None, resume=False, limit=None)
    run_attribution_stage(cfg, method="control_lexical", mode=None, resume=False, limit=None)
    evaluate_all(cfg)
    build_report(cfg)
    return cfg


def test_raw_to_derived_regeneration_is_byte_stable(regeneration_env: AppConfig) -> None:
    summary_path = regeneration_env.results_derived_dir / "summary.json"
    first = _derived_snapshot(regeneration_env)

    evaluate_all(regeneration_env)
    build_report(regeneration_env)

    assert _derived_snapshot(regeneration_env) == first
    summary = read_json(summary_path)
    assert "generated_utc" not in summary
    consumed_timestamps = [
        read_json(meta_path)["finished_utc"]
        for meta_path in Path("results/raw/smoke").rglob(RUN_META_FILE)
        if (meta_path.parent / RECORDS_FILE).exists()
    ]
    assert summary["source_snapshot_utc"] == max(consumed_timestamps)
    assert summary["source_snapshot_utc"] == max(
        split["source_snapshot_utc"] for split in summary["splits"].values()
    )
    assert not Path("results/derived").exists()
