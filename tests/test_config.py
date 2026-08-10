"""Config loading: repo configs are valid; strictness and path rules are enforced."""

from __future__ import annotations

import hashlib
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


@pytest.mark.parametrize(
    "name",
    [
        "smoke.yaml",
        "dev.yaml",
        "full.yaml",
        "reranking/smoke.yaml",
        "reranking/dev.yaml",
        "reranking/eval.yaml",
    ],
)
def test_repo_configs_are_valid(repo_root: Path, name: str) -> None:
    cfg = load_config(repo_root / "configs" / name)
    assert isinstance(cfg, AppConfig)
    assert cfg.data.split_sizes == {"smoke": 20, "dev": 60, "eval": 240}


def test_reranking_configs_are_separate_and_locked(repo_root: Path) -> None:
    smoke = load_config(repo_root / "configs/reranking/smoke.yaml")
    dev = load_config(repo_root / "configs/reranking/dev.yaml")
    locked = load_config(repo_root / "configs/reranking/eval.yaml")
    assert smoke.paths.results_raw == dev.paths.results_raw == locked.paths.results_raw
    assert smoke.paths.results_raw == "results/reranking/raw"
    assert smoke.reranking.reranker.device == "cpu"
    assert dev.reranking.reranker.device == locked.reranking.reranker.device == "cuda"
    assert smoke.reranking.reranker.candidate_k == 10
    assert smoke.reranking.final_context_k == 5
    assert smoke.reranking.bootstrap_resamples == 10_000
    assert smoke.reranking.bootstrap_confidence == 0.95
    secondary = repo_root / smoke.reranking.secondary_analysis_path
    assert hashlib.sha256(secondary.read_bytes()).hexdigest() == (
        smoke.reranking.secondary_analysis_sha256
    )
    assert locked.split == "eval"


@pytest.mark.parametrize(
    ("original_name", "extension_name"),
    [
        ("smoke.yaml", "reranking/smoke.yaml"),
        ("dev.yaml", "reranking/dev.yaml"),
        ("full.yaml", "reranking/eval.yaml"),
    ],
)
def test_reranking_configs_preserve_original_scientific_settings(
    repo_root: Path, original_name: str, extension_name: str
) -> None:
    original = load_config(repo_root / "configs" / original_name)
    extension = load_config(repo_root / "configs" / extension_name)
    assert extension.seed == original.seed
    assert extension.data == original.data
    assert extension.retrieval == original.retrieval
    assert extension.generation == original.generation
    assert extension.evaluation == original.evaluation
    assert extension.runtime == original.runtime
    assert extension.attribution.modes == original.attribution.modes
    assert extension.attribution.faithfulness == original.attribution.faithfulness
    assert extension.attribution.embedding == original.attribution.embedding
    assert (
        extension.attribution.controls.retrieval_run == original.attribution.controls.retrieval_run
    )
    assert (
        extension.attribution.controls.shuffled_source
        == original.attribution.controls.shuffled_source
    )


def test_minimal_config_defaults(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, MINIMAL))
    assert cfg.generation.model_id == "Qwen/Qwen3-4B-Instruct-2507"
    assert cfg.attribution.modes == ("gold", "generated")
    assert cfg.evaluation.bootstrap_resamples == 10_000
    assert cfg.evaluation.bootstrap_confidence == 0.95
    assert cfg.evaluation.bootstrap_tolerance == 0.0
    assert cfg.evaluation.confirmatory_method == "leave_one_out"
    assert cfg.evaluation.primary_comparator == "control_lexical"


@pytest.mark.parametrize("name", ["smoke.yaml", "dev.yaml", "eval.yaml"])
def test_v1_configs_read_historical_raw_and_write_versioned_derived(
    repo_root: Path, name: str
) -> None:
    cfg = load_config(repo_root / "configs" / "v1" / name)

    assert cfg.paths.results_raw == "results/raw"
    assert cfg.paths.results_derived == "results/v1/derived"
    assert cfg.paths.assets_dir == "results/v1/assets"
    assert cfg.evaluation.bootstrap_resamples == 10_000
    assert cfg.evaluation.bootstrap_confidence == 0.95
    assert cfg.evaluation.bootstrap_tolerance == 0.0
    assert cfg.evaluation.confirmatory_method == "leave_one_out"
    assert cfg.evaluation.primary_comparator == "control_lexical"


@pytest.mark.parametrize(
    ("historical_name", "v1_name"),
    [("smoke.yaml", "smoke.yaml"), ("dev.yaml", "dev.yaml"), ("full.yaml", "eval.yaml")],
)
def test_v1_configs_preserve_historical_scientific_inputs(
    repo_root: Path, historical_name: str, v1_name: str
) -> None:
    historical = load_config(repo_root / "configs" / historical_name)
    versioned = load_config(repo_root / "configs" / "v1" / v1_name)

    assert versioned.run_name == historical.run_name
    assert versioned.split == historical.split
    assert versioned.seed == historical.seed
    assert versioned.data == historical.data
    assert versioned.retrieval == historical.retrieval
    assert versioned.generation == historical.generation
    assert versioned.attribution == historical.attribution
    assert versioned.runtime == historical.runtime
    assert versioned.serve == historical.serve
    assert versioned.evaluation.attribution_ks == historical.evaluation.attribution_ks
    assert versioned.evaluation.primary_k == historical.evaluation.primary_k
    assert versioned.evaluation.correctness_criterion == historical.evaluation.correctness_criterion


@pytest.mark.parametrize(
    "evaluation",
    [
        "bootstrap_resamples: 999",
        "bootstrap_confidence: 0.0",
        "bootstrap_confidence: 1.0",
        "bootstrap_tolerance: -0.1",
        "confirmatory_method: embedding",
        "primary_comparator: control_random",
    ],
)
def test_gate_a_evaluation_protocol_is_locked(tmp_path: Path, evaluation: str) -> None:
    text = MINIMAL + f"\nevaluation:\n  {evaluation}\n"

    with pytest.raises(ConfigError, match=r"evaluation|bootstrap|confirmatory|comparator"):
        load_config(_write(tmp_path, text))


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


def test_env_override_may_be_absolute_but_yaml_may_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Colab points checkpoints at the Drive mount via env — absolute is allowed there
    monkeypatch.setenv("RAG_EVIDENCE_RESULTS_RAW", "/content/drive/MyDrive/reab/results/raw")
    cfg = load_config(_write(tmp_path, MINIMAL))
    assert cfg.paths.results_raw.startswith("/content/")


def test_resolve_device_and_dtype_cpu() -> None:
    assert resolve_device("cpu") == "cpu"
    # torch is installed as CPU build locally/CI, so auto must resolve to cpu
    assert resolve_device("auto") in {"cpu", "cuda"}
    assert resolve_dtype("auto", "cpu") == "float32"
    assert resolve_dtype("float16", "cpu") == "float16"
