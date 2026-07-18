"""Run manifest: hash stability, resume refusal on config change, dtype guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_evidence.config import load_config
from rag_evidence.errors import ArtifactError, ResumeConflictError
from rag_evidence.storage.artifacts import append_record
from rag_evidence.storage.runmeta import (
    check_dtype_compatible,
    config_hash,
    finalize_run,
    scientific_config,
    start_or_resume_run,
)

BASE = """
run_name: t
split: smoke
generation:
  max_new_tokens: {mnt}
"""


def _cfg(tmp_path: Path, max_new_tokens: int = 256):
    p = tmp_path / f"cfg{max_new_tokens}.yaml"
    p.write_text(BASE.format(mnt=max_new_tokens), encoding="utf-8")
    return load_config(p)


def test_config_hash_stable_and_sensitive(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    h1 = config_hash(scientific_config(cfg, "generate"))
    h2 = config_hash(scientific_config(cfg, "generate"))
    assert h1 == h2
    h3 = config_hash(scientific_config(_cfg(tmp_path, max_new_tokens=128), "generate"))
    assert h1 != h3


def test_fresh_run_then_resume_ok(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    run_dir = tmp_path / "run"
    meta = start_or_resume_run(
        run_dir,
        cfg,
        stage="generate",
        name="fake",
        resume=False,
        execution_kind="mock",
        expected_count=3,
    )
    finalize_run(run_dir, meta, status="interrupted", n_attempted=1, n_success=1, n_failed=0)
    meta2 = start_or_resume_run(
        run_dir,
        cfg,
        stage="generate",
        name="fake",
        resume=True,
        execution_kind="mock",
        expected_count=3,
    )
    assert meta2["run_id"] == meta["run_id"]
    assert len(meta2["resume_events"]) == 1


def test_resume_refused_on_config_change(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    start_or_resume_run(
        run_dir,
        _cfg(tmp_path),
        stage="generate",
        name="fake",
        resume=False,
        execution_kind="mock",
        expected_count=3,
    )
    with pytest.raises(ResumeConflictError, match="max_new_tokens"):
        start_or_resume_run(
            run_dir,
            _cfg(tmp_path, max_new_tokens=128),
            stage="generate",
            name="fake",
            resume=True,
            execution_kind="mock",
            expected_count=3,
        )


def test_nonresume_refused_when_records_exist(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    run_dir = tmp_path / "run"
    start_or_resume_run(
        run_dir,
        cfg,
        stage="generate",
        name="fake",
        resume=False,
        execution_kind="mock",
        expected_count=3,
    )
    append_record(run_dir / "records.jsonl", {"question_id": "q1"})
    with pytest.raises(ArtifactError, match="--resume"):
        start_or_resume_run(
            run_dir,
            cfg,
            stage="generate",
            name="fake",
            resume=False,
            execution_kind="mock",
            expected_count=3,
        )


def test_dtype_guard(tmp_path: Path) -> None:
    meta = {"model": {"dtype_effective": "float16"}}
    check_dtype_compatible(meta, {"dtype_effective": "float16"}, tmp_path)  # ok
    with pytest.raises(ResumeConflictError, match="dtype_effective"):
        check_dtype_compatible(meta, {"dtype_effective": "bfloat16"}, tmp_path)
