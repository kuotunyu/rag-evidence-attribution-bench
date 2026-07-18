"""ResultsStore behavior, including graceful handling of missing stages."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_evidence.config import AppConfig
from rag_evidence.errors import SampleNotFoundError
from rag_evidence.retrieval.run import run_retrieval_stage
from rag_evidence.store import ResultsStore


def _store(cfg: AppConfig) -> ResultsStore:
    return ResultsStore(Path(cfg.paths.results_raw), Path(cfg.paths.results_derived), cfg.split)


def test_store_with_samples_only(tiny_env: AppConfig) -> None:
    store = _store(tiny_env)
    assert len(store.list_ids()) == 3
    assert store.methods()["retrieval"] == []
    assert store.summary is None
    qid = store.list_ids()[0]
    record = store.benchmark_record(qid)
    assert record["generation"] == {} and record["attribution"] == {}
    with pytest.raises(KeyError):
        store.answer(qid)  # no generation runs → explicit error, not silence


def test_store_after_retrieval(tiny_env: AppConfig) -> None:
    run_retrieval_stage(tiny_env, method="bm25", resume=False, limit=None)
    store = _store(tiny_env)
    qid = store.list_ids()[0]
    rec = store.retrieve(qid, "bm25", k=3)
    assert len(rec["ranking"]) == 3
    with pytest.raises(SampleNotFoundError):
        store.retrieve("unknown-qid", "bm25")
