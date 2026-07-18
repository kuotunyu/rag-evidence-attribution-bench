"""Generation stage with FakeLM: end-to-end records, resume equivalence, scripting."""

from __future__ import annotations

import json
from pathlib import Path

from rag_evidence.config import AppConfig, load_config
from rag_evidence.generation.run import run_generation_stage
from rag_evidence.storage.artifacts import read_json, read_records

STABLE_FIELDS = (
    "question_id",
    "context_passage_ids",
    "alias_map",
    "response_text",
    "answer_text",
    "abstained",
    "citations_raw",
    "cited_passage_ids",
    "invalid_citations",
    "em",
    "f1",
    "error",
)


def _records(cfg: AppConfig) -> list[dict]:
    path = Path(cfg.paths.results_raw) / cfg.split / "generate" / "fake" / "records.jsonl"
    return sorted(read_records(path), key=lambda r: r["question_id"])


def _stable(records: list[dict]) -> list[dict]:
    return [{k: r[k] for k in STABLE_FIELDS} for r in records]


def test_generate_stage_end_to_end(tiny_env: AppConfig) -> None:
    cfg = tiny_env
    run_generation_stage(cfg, resume=False, limit=None)
    records = _records(cfg)
    assert len(records) == 3
    for rec in records:
        assert rec["error"] is None
        assert rec["citations_raw"], "FakeLM always cites"
        assert rec["peak_vram_mb"] is None  # CPU: null, never 0
        assert rec["prompt_tokens"] > 0 and rec["completion_tokens"] > 0
    meta = read_json(
        Path(cfg.paths.results_raw) / cfg.split / "generate" / "fake" / "run_meta.json"
    )
    assert meta["execution_kind"] == "mock"
    assert meta["status"] == "completed"
    assert meta["n_success"] == 3


def test_interrupt_then_resume_equals_uninterrupted(tiny_env: AppConfig, tmp_path: Path) -> None:
    cfg = tiny_env
    # interrupted run: only 1 sample, then resume to completion
    run_generation_stage(cfg, resume=False, limit=1)
    assert len(_records(cfg)) == 1
    run_generation_stage(cfg, resume=True, limit=None)
    resumed = _records(cfg)
    assert len(resumed) == 3

    # uninterrupted reference run in a separate results dir
    cfg2_path = tmp_path / "tiny2.yaml"
    raw = json.loads(cfg.model_dump_json())
    raw["paths"]["results_raw"] = "results2/raw"
    import yaml

    cfg2_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg2 = load_config(cfg2_path)
    run_generation_stage(cfg2, resume=False, limit=None)
    reference = _records(cfg2)

    assert _stable(resumed) == _stable(reference)


def test_scripted_answers_control_em(tiny_env: AppConfig, monkeypatch) -> None:  # noqa: ANN001
    """Script FakeLM per question: correct, wrong, abstain — em/abstain flags must follow."""
    from rag_evidence.data.hotpot import load_prepared_verified
    from rag_evidence.generation import backends as backends_mod
    from rag_evidence.generation.backends import FakeLM

    cfg = tiny_env
    examples = load_prepared_verified(cfg)
    q0, q1, q2 = (e.question for e in examples)
    a0 = examples[0].answer
    script = {
        q0: f"{a0} [P1][P2]",  # correct with citations
        q1: "utterly wrong [P1]",  # wrong
        q2: "INSUFFICIENT EVIDENCE",  # abstains
    }
    monkeypatch.setattr(
        backends_mod, "build_backend", lambda *a, **k: FakeLM(script=script)
    )
    # run.py imported build_backend by name — patch it there too
    from rag_evidence.generation import run as run_mod

    monkeypatch.setattr(run_mod, "build_backend", lambda *a, **k: FakeLM(script=script))

    run_generation_stage(cfg, resume=False, limit=None)
    by_q = {r["question_id"]: r for r in _records(cfg)}
    r0 = by_q[examples[0].question_id]
    r1 = by_q[examples[1].question_id]
    r2 = by_q[examples[2].question_id]
    assert r0["em"] == 1 and r0["abstained"] is False
    assert r1["em"] == 0 and r1["f1"] == 0.0
    assert r2["abstained"] is True and r2["em"] == 0 and r2["answer_text"] == ""


def test_synthetic_failure_recorded_not_fatal(tiny_env: AppConfig, monkeypatch) -> None:  # noqa: ANN001
    from rag_evidence.data.hotpot import load_prepared_verified
    from rag_evidence.generation import run as run_mod
    from rag_evidence.generation.backends import FakeLM

    cfg = tiny_env
    examples = load_prepared_verified(cfg)
    fail_q = examples[1].question
    monkeypatch.setattr(
        run_mod, "build_backend", lambda *a, **k: FakeLM(fail_marker=fail_q[:20])
    )
    run_generation_stage(cfg, resume=False, limit=None)
    records = _records(cfg)
    failed = [r for r in records if r["error"] is not None]
    assert len(failed) == 1
    assert failed[0]["question_id"] == examples[1].question_id
    assert failed[0]["error"]["type"] == "RuntimeError"
    meta = read_json(
        Path(cfg.paths.results_raw) / cfg.split / "generate" / "fake" / "run_meta.json"
    )
    assert meta["n_failed"] == 1 and meta["status"] == "completed_with_failures"
