"""Shared fixtures. The offline guard runs for EVERY test: CI and local test runs must
never touch the Hugging Face Hub — all tests use synthetic fixtures and fake backends."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _offline_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))
    # env overrides must never leak from the developer's shell into tests
    for var in (
        "RAG_EVIDENCE_HOST",
        "RAG_EVIDENCE_PORT",
        "RAG_EVIDENCE_RESULTS_RAW",
        "RAG_EVIDENCE_RESULTS_DERIVED",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture()
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture()
def repo_root() -> Path:
    return Path(__file__).parent.parent
