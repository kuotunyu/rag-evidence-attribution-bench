"""Challenge stages reuse the pipeline without crossing natural/formal output roots."""

from __future__ import annotations

from pathlib import Path

from rag_evidence.attribution.run import run_attribution_stage
from rag_evidence.config import AppConfig
from rag_evidence.evaluation.evaluate import evaluate_all
from rag_evidence.generation.run import run_generation_stage
from rag_evidence.reporting.report import build_report
from rag_evidence.storage.artifacts import read_json, read_records
from test_challenge_execution import eligible_challenge_config


def test_fake_challenge_generate_attribute_evaluate_report(
    tiny_env: AppConfig, tmp_path: Path
) -> None:
    cfg = eligible_challenge_config(tiny_env, tmp_path)

    run_generation_stage(cfg, resume=False, limit=None)
    run_attribution_stage(
        cfg,
        method="citations",
        mode="generated",
        resume=False,
        retry_failures=False,
        limit=None,
    )
    evaluate_all(cfg)
    build_report(cfg)

    generated = list(
        read_records(cfg.results_raw_dir / "smoke" / "generate" / "fake" / "records.jsonl")
    )
    assert generated[0]["challenge"]["variant"] == "missing_hop"
    assert generated[0]["challenge"]["parent_question_id"]
    assert generated[0]["challenge"]["leakage_group"]
    assert generated[0]["em"] is None
    assert generated[0]["answerability_correct"] is False

    summary = read_json(cfg.results_derived_dir / "summary.json")
    assert summary["execution"]["track"] == "challenge"
    assert summary["execution"]["phase"] == "pilot"
    assert summary["execution"]["variant"] == "missing_hop"
    assert (cfg.results_derived_dir / "report.md").exists()
    assert not Path("results/derived/summary.json").exists()
