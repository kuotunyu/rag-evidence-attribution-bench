"""Config loading: repo configs are valid; strictness and path rules are enforced."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_evidence.config import AppConfig, load_config, resolve_device, resolve_dtype
from rag_evidence.errors import ConfigError

MINIMAL = """
run_name: t
split: smoke
"""


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(text, encoding="utf-8")
    return p


@pytest.mark.parametrize("name", ["smoke.yaml", "dev.yaml", "full.yaml"])
def test_repo_configs_are_valid(repo_root: Path, name: str) -> None:
    cfg = load_config(repo_root / "configs" / name)
    assert isinstance(cfg, AppConfig)
    assert cfg.data.split_sizes == {"smoke": 20, "dev": 60, "eval": 240}


def test_minimal_config_defaults(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, MINIMAL))
    assert cfg.generation.model_id == "Qwen/Qwen3-4B-Instruct-2507"
    assert cfg.attribution.modes == ("gold", "generated")


def test_unknown_key_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"extra_forbidden|not permitted|Extra inputs"):
        load_config(_write(tmp_path, MINIMAL + "\ntypo_key: 1\n"))


def test_absolute_path_rejected(tmp_path: Path) -> None:
    bad = MINIMAL + "\npaths:\n  data_dir: C:\\Users\\someone\\data\n"
    with pytest.raises(ConfigError, match="repo-relative"):
        load_config(_write(tmp_path, bad))


def test_env_overrides_applied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_EVIDENCE_HOST", "0.0.0.0")
    monkeypatch.setenv("RAG_EVIDENCE_PORT", "9001")
    monkeypatch.setenv("RAG_EVIDENCE_RESULTS_RAW", "colab_results/raw")
    cfg = load_config(_write(tmp_path, MINIMAL))
    assert cfg.serve.host == "0.0.0.0"
    assert cfg.serve.port == 9001
    assert cfg.paths.results_raw == "colab_results/raw"


def test_resolve_device_and_dtype_cpu() -> None:
    assert resolve_device("cpu") == "cpu"
    # torch is installed as CPU build locally/CI, so auto must resolve to cpu
    assert resolve_device("auto") in {"cpu", "cuda"}
    assert resolve_dtype("auto", "cpu") == "float32"
    assert resolve_dtype("float16", "cpu") == "float16"
