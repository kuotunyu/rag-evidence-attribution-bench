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
import re
from pathlib import Path, PureWindowsPath
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    manifest_schema_version: Literal[1, 2] = 1
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

    @model_validator(mode="after")
    def _v2_requires_pinned_revision(self) -> DataConfig:
        if self.manifest_schema_version == 2 and (
            self.hf_revision is None or re.fullmatch(r"[0-9a-f]{40}", self.hf_revision) is None
        ):
            raise ValueError(
                "manifest schema v2 requires hf_revision to be an exact 40-character "
                "git commit hash"
            )
        return self


class ChallengeConfig(_StrictModel):
    schema_version: Literal[1] = 1
    seed: int = 20260810
    transform_version: Literal["challenge-v1"] = "challenge-v1"
    manifest_path: str = "data/manifests/challenge_manifest_v1.json"
    prepared_dir: str = "data/v2/challenge"

    @field_validator("seed")
    @classmethod
    def _locked_seed(cls, value: int) -> int:
        if value != 20260810:
            raise ValueError("challenge seed is locked at 20260810")
        return value


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


class RerankerConfig(_StrictModel):
    adapter: Literal["hf_sequence_classification"] = "hf_sequence_classification"
    model_id: str = "BAAI/bge-reranker-v2-m3"
    model_revision: str = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    tokenizer_id: str = "BAAI/bge-reranker-v2-m3"
    tokenizer_revision: str = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    max_length: int = Field(default=512, gt=0)
    device: Literal["cpu", "cuda"] = "cpu"
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    batch_size: int = Field(default=16, gt=0)
    candidate_k: int = Field(default=10, gt=0)
    cache_enabled: bool = True

    @field_validator("model_revision", "tokenizer_revision")
    @classmethod
    def _pinned_revision(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError("reranker revisions must be exact 40-character git commit hashes")
        return value


class RerankingExperimentConfig(_StrictModel):
    enabled: bool = False
    experiment_id: str = "reranking-v1-2026-07-29"
    preregistration_path: str = "PREREGISTRATION_RERANKING.md"
    preregistration_sha256: str = "fdef3b33d5a78ac939f0cab1a7523a1a29ed3ef1ee4d8b9705447045fbc2e42f"
    secondary_analysis_path: str = "SECONDARY_ANALYSIS_RERANKING.md"
    secondary_analysis_sha256: str = (
        "4b60cc6f0029b0930701e97581e30007d49260e5948c39f67500b6b6f0846f48"
    )
    bootstrap_resamples: int = Field(default=10_000, ge=1_000)
    bootstrap_confidence: float = Field(default=0.95, gt=0.0, lt=1.0)
    transfer_tolerance: float = Field(default=1e-12, ge=0.0)
    baseline_results_raw: str = "results/raw"
    final_context_k: int = Field(default=5, gt=0)
    arms: tuple[Literal["bm25", "dense", "hybrid_rrf", "hybrid_rrf_rerank"], ...] = (
        "bm25",
        "dense",
        "hybrid_rrf",
        "hybrid_rrf_rerank",
    )
    reranker: RerankerConfig = RerankerConfig()

    @field_validator("experiment_id")
    @classmethod
    def _experiment_slug(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", value):
            raise ValueError("reranking.experiment_id must be a filesystem-safe lowercase slug")
        return value

    @field_validator("preregistration_sha256", "secondary_analysis_sha256")
    @classmethod
    def _prereg_hash(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("reranking analysis hashes must be 64 lowercase hex chars")
        return value

    @field_validator("arms")
    @classmethod
    def _arms_unique_and_complete(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        required = {"bm25", "dense", "hybrid_rrf", "hybrid_rrf_rerank"}
        if set(value) != required or len(value) != len(required):
            raise ValueError(f"reranking.arms must contain exactly {sorted(required)}")
        return value


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
    retrieval_results_raw: str | None = None
    top_k_context: int = 10
    prompt_version: str = "v1"


class FaithfulnessConfig(_StrictModel):
    enabled: bool = True
    k: int = 2


class AttributionEmbeddingConfig(_StrictModel):
    model_id: str = "Qwen/Qwen3-Embedding-0.6B"


class ControlsConfig(_StrictModel):
    retrieval_run: str = "bm25"
    retrieval_results_raw: str | None = None
    shuffled_source: str = "leave_one_out"


class AttributionConfig(_StrictModel):
    modes: tuple[Literal["gold", "generated"], ...] = ("gold", "generated")
    faithfulness: FaithfulnessConfig = FaithfulnessConfig()
    embedding: AttributionEmbeddingConfig = AttributionEmbeddingConfig()
    controls: ControlsConfig = ControlsConfig()
    run_namespace: str | None = None
    gold_context_source: Literal["dataset", "generation"] = "dataset"

    @field_validator("modes")
    @classmethod
    def _nonempty(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v:
            raise ValueError("attribution.modes must not be empty")
        if len(set(v)) != len(v):
            raise ValueError("attribution.modes contains duplicates")
        return v

    @field_validator("run_namespace")
    @classmethod
    def _namespace_slug(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", value):
            raise ValueError("attribution.run_namespace must be a filesystem-safe lowercase slug")
        return value


class EvaluationConfig(_StrictModel):
    attribution_ks: tuple[int, ...] = (1, 2, 3)
    primary_k: int = 2
    correctness_criterion: Literal["em", "f1_05"] = "em"
    bootstrap_resamples: int = Field(default=10_000, ge=1_000)
    bootstrap_confidence: float = Field(default=0.95, gt=0.0, lt=1.0)
    bootstrap_tolerance: float = Field(default=0.0, ge=0.0)
    confirmatory_method: Literal["leave_one_out"] = "leave_one_out"
    primary_comparator: Literal["control_lexical"] = "control_lexical"


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
    challenge: ChallengeConfig = ChallengeConfig()
    paths: PathsConfig = PathsConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig = GenerationConfig()
    attribution: AttributionConfig = AttributionConfig()
    reranking: RerankingExperimentConfig = RerankingExperimentConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    serve: ServeConfig = ServeConfig()

    @model_validator(mode="after")
    def _locked_reranking_design(self) -> AppConfig:
        if not self.reranking.enabled:
            return self
        rr = self.reranking.reranker
        expected = {
            "model_id": "BAAI/bge-reranker-v2-m3",
            "model_revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
            "tokenizer_id": "BAAI/bge-reranker-v2-m3",
            "tokenizer_revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
            "max_length": 512,
            "batch_size": 16,
            "candidate_k": 10,
        }
        actual = {key: getattr(rr, key) for key in expected}
        if actual != expected:
            raise ValueError(
                f"reranking config violates locked preregistration: {actual} != {expected}"
            )
        if self.reranking.final_context_k != 5:
            raise ValueError("reranking.final_context_k is preregistered at 5")
        if (
            self.reranking.bootstrap_resamples != 10_000
            or self.reranking.bootstrap_confidence != 0.95
            or self.reranking.transfer_tolerance != 1e-12
        ):
            raise ValueError(
                "secondary analysis is locked at 10000 resamples, 95% confidence, "
                "and transfer tolerance 1e-12"
            )
        expected_device_dtype = ("cpu", "float32") if self.split == "smoke" else ("cuda", "float16")
        if (rr.device, rr.dtype) != expected_device_dtype:
            raise ValueError(
                f"reranker {self.split} device/dtype must be {expected_device_dtype}, "
                f"got {(rr.device, rr.dtype)}"
            )
        return self

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

    @property
    def challenge_manifest_file(self) -> Path:
        return Path(self.challenge.manifest_path)

    @property
    def challenge_prepared_dir(self) -> Path:
        return Path(self.challenge.prepared_dir)


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
    ("challenge", "manifest_path"),
    ("challenge", "prepared_dir"),
    ("paths", "data_dir"),
    ("paths", "results_raw"),
    ("paths", "results_derived"),
    ("paths", "assets_dir"),
    ("generation", "retrieval_results_raw"),
    ("attribution", "controls.retrieval_results_raw"),
    ("reranking", "preregistration_path"),
    ("reranking", "secondary_analysis_path"),
    ("reranking", "baseline_results_raw"),
)


def _check_yaml_paths_relative(raw: dict[str, Any], source: Path) -> None:
    """Paths written in a CONFIG FILE must be repo-relative (rule: no machine-specific
    absolute paths in committed files). Env overrides (RAG_EVIDENCE_*) may be absolute —
    that is exactly how Colab points checkpoints at a Drive mount."""
    for section, key in _YAML_PATH_FIELDS:
        section_value = raw.get(section) or {}
        if "." in key:
            first, second = key.split(".", 1)
            value = (section_value.get(first) or {}).get(second)
        else:
            value = section_value.get(key)
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
