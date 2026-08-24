"""Run manifests (run_meta.json): provenance, scientific config hashing, resume policy.

Scientific config = exactly the fields that can change the numbers (dataset identity,
split, seed, model, decoding, method parameters). Paths, batch sizes, logging and other
tuning knobs are excluded on purpose. Resume REFUSES on a scientific-hash mismatch — there
is no --force: a run whose halves used different configs is worthless and invisible later.

execution_kind: "real" for genuine model/dataset executions, "mock" when the fake backend
is involved. Mock runs are quarantined by the report stage and never rendered into READMEs.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Literal

import rag_evidence
from rag_evidence.config import AppConfig
from rag_evidence.errors import ArtifactError, ResumeConflictError
from rag_evidence.storage.artifacts import (
    RECORDS_FILE,
    RUN_META_FILE,
    read_json,
    write_json_atomic,
)
from rag_evidence.telemetry import device_info, library_versions

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

Stage = Literal["retrieve", "generate", "attribute"]


def _utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def scientific_config(cfg: AppConfig, stage: str, **extra: Any) -> dict[str, Any]:
    """The stage-relevant subset of config that determines the numbers."""
    sci: dict[str, Any] = {
        "stage": stage,
        "split": cfg.split,
        "seed": cfg.seed,
        "data": {
            "hf_path": cfg.data.hf_path,
            "hf_config": cfg.data.hf_config,
            "hf_split": cfg.data.hf_split,
            "hf_revision": cfg.data.hf_revision,
            "split_seed": cfg.data.split_seed,
            "split_sizes": dict(cfg.data.split_sizes),
        },
        "execution": cfg.execution.model_dump(mode="json"),
    }
    if stage == "retrieve":
        method = str(extra.get("method"))
        sci["retrieval"] = {"method": method}
        if method == "bm25":
            sci["retrieval"]["bm25"] = {"k1": cfg.retrieval.bm25.k1, "b": cfg.retrieval.bm25.b}
        elif method == "dense":
            sci["retrieval"]["dense"] = {
                "model_id": cfg.retrieval.dense.model_id,
                "max_length": cfg.retrieval.dense.max_length,
                "normalize": cfg.retrieval.dense.normalize,
            }
        elif method == "hybrid_rrf":
            sci["retrieval"]["rrf_k"] = cfg.retrieval.rrf_k
        elif method == "hybrid_rrf_rerank":
            r = cfg.reranking
            rr = r.reranker
            sci["retrieval"].update(
                {
                    "source_method": "hybrid_rrf",
                    "source_rrf_k": cfg.retrieval.rrf_k,
                    "experiment_id": r.experiment_id,
                    "preregistration_sha256": r.preregistration_sha256,
                    "candidate_k": rr.candidate_k,
                    "adapter": rr.adapter,
                    "model_id": rr.model_id,
                    "model_revision": rr.model_revision,
                    "tokenizer_id": rr.tokenizer_id,
                    "tokenizer_revision": rr.tokenizer_revision,
                    "max_length": rr.max_length,
                    "device": rr.device,
                    "dtype": rr.dtype,
                }
            )
    elif stage == "generate":
        from rag_evidence.generation.prompts import prompt_hash

        g = cfg.generation
        sci["generation"] = {
            "model_id": g.model_id,
            "model_revision": g.model_revision,
            "tokenizer_id": g.tokenizer_id,
            "tokenizer_revision": g.tokenizer_revision,
            "local_artifact_sha256": g.local_artifact_sha256,
            "runtime": g.runtime,
            "runtime_version": g.runtime_version,
            "backend": g.backend,
            "dtype": g.dtype,
            "quantization": g.quantization,
            "max_new_tokens": g.max_new_tokens,
            "context_source": g.context_source,
            "retrieval_run": g.retrieval_run,
            "retrieval_results_raw": g.retrieval_results_raw,
            "top_k_context": g.top_k_context,
            "prompt_version": g.prompt_version,
            "prompt_hash": prompt_hash(g.prompt_version),
            "do_sample": g.do_sample,
            "num_beams": g.num_beams,
        }
    elif stage == "attribute":
        g = cfg.generation
        sci["generation_model"] = {
            "model_id": g.model_id,
            "model_revision": g.model_revision,
            "tokenizer_id": g.tokenizer_id,
            "tokenizer_revision": g.tokenizer_revision,
            "local_artifact_sha256": g.local_artifact_sha256,
            "runtime": g.runtime,
            "runtime_version": g.runtime_version,
            "backend": g.backend,
            "dtype": g.dtype,
            "quantization": g.quantization,
            "prompt_version": g.prompt_version,
            "do_sample": g.do_sample,
            "num_beams": g.num_beams,
        }
        sci["attribution"] = {
            "faithfulness_enabled": cfg.attribution.faithfulness.enabled,
            "faithfulness_k": cfg.attribution.faithfulness.k,
            "embedding_model_id": cfg.attribution.embedding.model_id,
            "controls_retrieval_run": cfg.attribution.controls.retrieval_run,
            "controls_retrieval_results_raw": (cfg.attribution.controls.retrieval_results_raw),
            "controls_shuffled_source": cfg.attribution.controls.shuffled_source,
            "run_namespace": cfg.attribution.run_namespace,
            "gold_context_source": cfg.attribution.gold_context_source,
        }
    sci.update(extra)
    return sci


def config_hash(sci: dict[str, Any]) -> str:
    canonical = json.dumps(sci, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _diff(old: dict[str, Any], new: dict[str, Any]) -> str:
    fo, fn = _flatten(old), _flatten(new)
    lines = []
    for key in sorted(set(fo) | set(fn)):
        if fo.get(key, "<absent>") != fn.get(key, "<absent>"):
            lines.append(f"  {key}: {fo.get(key, '<absent>')!r} -> {fn.get(key, '<absent>')!r}")
    return "\n".join(lines) or "  (hash differs but no field-level diff found)"


def start_or_resume_run(
    run_dir: Path,
    cfg: AppConfig,
    *,
    stage: str,
    name: str,
    resume: bool,
    execution_kind: Literal["real", "mock"],
    expected_count: int,
    limit: int | None = None,
    sci_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create or resume a run manifest; enforces the resume-compatibility policy."""
    sci = scientific_config(cfg, stage, **(sci_extra or {}))
    new_hash = config_hash(sci)
    meta_path = run_dir / RUN_META_FILE
    records_path = run_dir / RECORDS_FILE

    if meta_path.exists():
        meta = read_json(meta_path)
        if not resume:
            if records_path.exists() and records_path.stat().st_size > 0:
                raise ArtifactError(
                    f"{run_dir} already contains records; pass --resume to continue it "
                    "or delete the directory to start over (refusing to mix runs)"
                )
        else:
            if meta.get("config_hash") != new_hash:
                raise ResumeConflictError(
                    f"--resume refused for {run_dir}: scientific config changed since the "
                    f"original run.\n{_diff(meta.get('config_scientific', {}), sci)}\n"
                    "Start a new run (delete the directory or change the run name) or "
                    "restore the original config. There is no --force."
                )
            if meta.get("execution_kind") != execution_kind:
                raise ResumeConflictError(
                    f"--resume refused for {run_dir}: execution_kind changed "
                    f"({meta.get('execution_kind')} -> {execution_kind})"
                )
            meta.setdefault("resume_events", []).append(
                {"at": _utc_now(), "env": device_info(), "libs": library_versions()}
            )
            meta["status"] = "running"
            meta["limit"] = limit
            write_json_atomic(meta_path, meta)
            logger.info("resuming run %s (%s)", run_dir, meta.get("run_id"))
            return dict(meta)

    run_id = f"{cfg.split}_{stage}_{name}_{_utc_now().replace(':', '').replace('-', '')}"
    meta = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "stage": stage,
        "name": name,
        "split": cfg.split,
        "run_name": cfg.run_name,
        "execution_kind": execution_kind,
        "config_hash": new_hash,
        "config_scientific": sci,
        "package_version": rag_evidence.__version__,
        "started_utc": _utc_now(),
        "finished_utc": None,
        "status": "running",
        "expected_count": expected_count,
        "limit": limit,
        "n_attempted": 0,
        "n_success": 0,
        "n_failed": 0,
        "n_skipped": 0,
        "env": device_info(),
        "libs": library_versions(),
        "model": {},
        "run_peak_vram_mb": None,
        "total_wall_s": None,
        "resume_events": [],
        "notes": "",
    }
    write_json_atomic(meta_path, meta)
    logger.info("started run %s", run_id)
    return meta


def check_dtype_compatible(meta: dict[str, Any], effective: dict[str, Any], run_dir: Path) -> None:
    """Scoring/generation resumes must not silently change numerics mid-run.

    `effective` example: {"dtype_effective": "float16", "quantization_effective": "none"}.
    First session records it; later resumes must match exactly.
    """
    recorded = meta.get("model") or {}
    for key, value in effective.items():
        if recorded.get(key) is not None and recorded[key] != value:
            raise ResumeConflictError(
                f"--resume refused for {run_dir}: {key} changed mid-run "
                f"({recorded[key]} -> {value}); logprobs/generations would not be "
                "comparable within one run. Start a new run on this hardware instead."
            )


def update_model_info(run_dir: Path, meta: dict[str, Any], model_info: dict[str, Any]) -> None:
    meta["model"] = {**meta.get("model", {}), **model_info}
    write_json_atomic(run_dir / RUN_META_FILE, meta)


def finalize_run(
    run_dir: Path,
    meta: dict[str, Any],
    *,
    status: str,
    n_attempted: int,
    n_success: int,
    n_failed: int,
    n_skipped: int = 0,
    run_peak_vram_mb: float | None = None,
    total_wall_s: float | None = None,
) -> None:
    meta.update(
        finished_utc=_utc_now(),
        status=status,
        n_attempted=n_attempted,
        n_success=n_success,
        n_failed=n_failed,
        n_skipped=n_skipped,
        run_peak_vram_mb=run_peak_vram_mb,
        total_wall_s=total_wall_s,
    )
    write_json_atomic(run_dir / RUN_META_FILE, meta)
