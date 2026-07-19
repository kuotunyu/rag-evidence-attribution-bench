"""Configuration: YAML → validated pydantic models.

Rules enforced here:
- unknown YAML keys fail loudly (`extra="forbid"`),
- all configured paths must be repo-relative (no absolute local paths in configs),
- device/dtype are resolved through `resolve_device` / `resolve_dtype` only — no stage
  may hard-code a device,
- a small set of env vars (RAG_EVIDENCE_*) may override serve host/port and results dirs
  so Docker/Colab never edit config files.
"""

from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag_evidence.errors import ConfigError

SPLIT_NAMES = ("smoke", "dev", "eval")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _require_relative(value: str, field_name: str) -> str:
    # PureWindowsPath.is_absolute catches both C:\… and \\server\…; POSIX "/" too.
    if PureWindowsPath(value).is_absolute() or value.startswith(("/", "\\")):
        raise ValueError(
            f"{field_name} must be repo-relative, got absolute path {value!r} "
            "(configs must not contain machine-specific absolute paths)"
        )
    return value


class DataConfig(_StrictModel):
    hf_path: str = "hotpotqa/hotpot_qa"
    hf_config: str = "distractor"
    hf_split: str = "validation"
    hf_revision: str | None = None
    manifest_path: str = "data/manifests/split_manifest.json"
    prepared_dir: str = "data/prepared"
    split_seed: int = 20260718
    split_sizes: dict[str, int] = Field(
        default_factory=lambda: {"smoke": 20, "dev": 60, "eval": 240}
    )

    @field_validator("split_sizes")
    @classmethod
    def _splits(cls, v: dict[str, int]) -> dict[str, int]:
        if set(v) != set(SPLIT_NAMES):
            raise ValueError(f"split_sizes must have exactly the keys {SPLIT_NAMES}, got {set(v)}")
        if any(n <= 0 for n in v.values()):
            raise ValueError("split sizes must be positive")
        return v


class PathsConfig(_StrictModel):
    data_dir: str = "data"
    results_raw: str = "results/raw"
    results_derived: str = "results/derived"
    assets_dir: str = "assets"


class BM25Config(_StrictModel):
    k1: float = 1.5
    b: float = 0.75


class DenseConfig(_StrictModel):
    model_id: str = "Qwen/Qwen3-Embedding-0.6B"
    batch_size: int = 8
    max_length: int = 1024
    normalize: bool = True


class RetrievalConfig(_StrictModel):
    ks: tuple[int, ...] = (2, 5, 10)
    rrf_k: int = 60
    bm25: BM25Config = BM25Config()
    dense: DenseConfig = DenseConfig()


class GenerationConfig(_StrictModel):
    model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    name: str = "qwen3-4b"
    backend: Literal["qwen", "fake"] = "qwen"
    dtype: Literal["auto", "bfloat16", "float16", "float32"] = "auto"
    quantization: Literal["none", "4bit"] = "none"
    fallback_4bit: bool = True
    max_new_tokens: int = 256
    context_source: Literal["dataset", "retrieval"] = "dataset"
    retrieval_run: str | None = None
    top_k_context: int = 10
    prompt_version: str = "v1"


class FaithfulnessConfig(_StrictModel):
    enabled: bool = True
    k: int = 2


class AttributionEmbeddingConfig(_StrictModel):
    model_id: str = "Qwen/Qwen3-Embedding-0.6B"


class ControlsConfig(_StrictModel):
    retrieval_run: str = "bm25"
    shuffled_source: str = "leave_one_out"


class AttributionConfig(_StrictModel):
    modes: tuple[Literal["gold", "generated"], ...] = ("gold", "generated")
    faithfulness: FaithfulnessConfig = FaithfulnessConfig()
    embedding: AttributionEmbeddingConfig = AttributionEmbeddingConfig()
    controls: ControlsConfig = ControlsConfig()

    @field_validator("modes")
    @classmethod
    def _nonempty(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v:
            raise ValueError("attribution.modes must not be empty")
        if len(set(v)) != len(v):
            raise ValueError("attribution.modes contains duplicates")
        return v


class EvaluationConfig(_StrictModel):
    attribution_ks: tuple[int, ...] = (1, 2, 3)
    primary_k: int = 2
    correctness_criterion: Literal["em", "f1_05"] = "em"


class RuntimeConfig(_StrictModel):
    device: Literal["auto", "cpu", "cuda"] = "auto"
    log_level: str = "INFO"


class ServeConfig(_StrictModel):
    mode: Literal["precomputed", "live"] = "precomputed"
    host: str = "127.0.0.1"
    port: int = 8000


class AppConfig(_StrictModel):
    run_name: str
    split: Literal["smoke", "dev", "eval"]
    seed: int = 42
    data: DataConfig = DataConfig()
    paths: PathsConfig = PathsConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig = GenerationConfig()
    attribution: AttributionConfig = AttributionConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    serve: ServeConfig = ServeConfig()

    # --- resolved path helpers (all relative to CWD = repo root) -------------
    @property
    def results_raw_dir(self) -> Path:
        return Path(self.paths.results_raw)

    @property
    def results_derived_dir(self) -> Path:
        return Path(self.paths.results_derived)

    @property
    def manifest_file(self) -> Path:
        return Path(self.data.manifest_path)

    @property
    def prepared_file(self) -> Path:
        return Path(self.data.prepared_dir) / f"{self.split}.jsonl"


_ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    "RAG_EVIDENCE_HOST": ("serve", "host"),
    "RAG_EVIDENCE_PORT": ("serve", "port"),
    "RAG_EVIDENCE_RESULTS_RAW": ("paths", "results_raw"),
    "RAG_EVIDENCE_RESULTS_DERIVED": ("paths", "results_derived"),
}


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    for env_name, (section, key) in _ENV_OVERRIDES.items():
        value = os.environ.get(env_name)
        if value:
            raw.setdefault(section, {})[key] = value
    return raw


_YAML_PATH_FIELDS: tuple[tuple[str, str], ...] = (
    ("data", "manifest_path"),
    ("data", "prepared_dir"),
    ("paths", "data_dir"),
    ("paths", "results_raw"),
    ("paths", "results_derived"),
    ("paths", "assets_dir"),
)


def _check_yaml_paths_relative(raw: dict[str, Any], source: Path) -> None:
    """Paths written in a CONFIG FILE must be repo-relative (rule: no machine-specific
    absolute paths in committed files). Env overrides (RAG_EVIDENCE_*) may be absolute —
    that is exactly how Colab points checkpoints at a Drive mount."""
    for section, key in _YAML_PATH_FIELDS:
        value = (raw.get(section) or {}).get(key)
        if isinstance(value, str):
            try:
                _require_relative(value, f"{section}.{key}")
            except ValueError as exc:
                raise ConfigError(f"invalid config {source}: {exc}") from exc


def load_config(path: Path | str) -> AppConfig:
    """Load and validate a YAML config; raises ConfigError with a precise message."""
    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read config {p}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {p}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config {p} must be a YAML mapping, got {type(raw).__name__}")
    _check_yaml_paths_relative(raw, p)
    raw = _apply_env_overrides(raw)
    try:
        return AppConfig.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError → readable ConfigError
        raise ConfigError(f"invalid config {p}:\n{exc}") from exc


def resolve_device(pref: str) -> str:
    """The only device-selection mechanism in this package."""
    if pref != "auto":
        return pref
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(pref: str, device: str) -> str:
    """Resolve 'auto' dtype: bf16 if the GPU supports it, fp16 on other GPUs, fp32 on CPU."""
    if pref != "auto":
        return pref
    if device != "cuda":
        return "float32"
    import torch

    return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
