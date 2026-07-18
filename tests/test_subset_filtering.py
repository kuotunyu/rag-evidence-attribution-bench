"""Mode-B generated-correct subset filtering in `evaluate` (spec-critical rule)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_evidence.attribution.run import run_attribution_stage
from rag_evidence.config import AppConfig
from rag_evidence.data.hotpot import load_prepared_verified
from rag_evidence.evaluation.evaluate import evaluate_all
from rag_evidence.generation.backends import FakeLM
from rag_evidence.generation.run import run_generation_stage
from rag_evidence.retrieval.run import run_retrieval_stage
from rag_evidence.storage.artifacts import read_json


@pytest.fixture()
def scripted_env(tiny_env: AppConfig, monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    """1 correct, 1 wrong, 1 abstained generated answer over the 3 smoke questions."""
    cfg = tiny_env
    examples = load_prepared_verified(cfg)
    script = {
        examples[0].question: f"{examples[0].answer} [P1][P2]",
        examples[1].question: "utterly wrong answer [P3]",
        examples[2].question: "INSUFFICIENT EVIDENCE",
    }
    from rag_evidence.generation import run as gen_run_mod

    monkeypatch.setattr(gen_run_mod, "build_backend", lambda *a, **k: FakeLM(script=script))
    run_retrieval_stage(cfg, method="bm25", resume=False, limit=None)
    run_generation_stage(cfg, resume=False, limit=None)
    run_attribution_stage(cfg, method="leave_one_out", mode=None, resume=False, limit=None)
    evaluate_all(cfg)
    return cfg


def test_subset_block_counts(scripted_env: AppConfig) -> None:
    summary = read_json(Path(scripted_env.paths.results_derived) / "summary.json")
    generated = summary["splits"]["smoke"]["attribution"]["generated"]
    assert generated["subset"] == {
        "criterion": "em",
        "n_total": 3,
        "n_generated_ok": 3,
        "n_abstained": 1,
        "n_correct": 1,
    }


def test_agreement_restricted_to_correct_subset(scripted_env: AppConfig) -> None:
    summary = read_json(Path(scripted_env.paths.results_derived) / "summary.json")
    attribution = summary["splits"]["smoke"]["attribution"]
    loo_generated = attribution["generated"]["leave_one_out"]
    # 2 non-abstained answers were attributed, but only the 1 correct one counts
    # toward agreement-with-supporting-facts metrics
    assert loo_generated["n_success"] == 2
    assert loo_generated["n_skipped"] == 1  # the abstained sample
    assert loo_generated["n_agreement"] == 1
    # faithfulness is ground-truth-free: aggregated over all attributed samples
    assert loo_generated["n_faithfulness"] == 2
    # mode A qualifies every successfully attributed sample
    loo_gold = attribution["gold"]["leave_one_out"]
    assert loo_gold["n_agreement"] == loo_gold["n_success"] == 3


def test_per_sample_file_flags_exclusions(scripted_env: AppConfig) -> None:
    from rag_evidence.storage.artifacts import read_records

    path = (
        Path(scripted_env.paths.results_derived)
        / "smoke"
        / "attribution"
        / "generated"
        / "leave_one_out_per_sample.jsonl"
    )
    rows = list(read_records(path))
    assert len(rows) == 2  # abstained sample was never attributed
    by_flag = {r["in_agreement_subset"] for r in rows}
    assert by_flag == {True, False}
    excluded = next(r for r in rows if not r["in_agreement_subset"])
    assert excluded["exclusion_reason"] == "not_correct"
