"""Stage dispatch. The CLI calls these; each lazily imports its stage module so
that `--help`, config validation, and precomputed serving never pull torch/HF."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_evidence.config import AppConfig


def run_data_prepare(cfg: AppConfig) -> None:
    from rag_evidence.data.hotpot import prepare_data

    prepare_data(cfg)


def run_data_challenge(cfg: AppConfig) -> None:
    from rag_evidence.data.challenge_prepare import prepare_challenge

    prepare_challenge(cfg)


def run_retrieve(cfg: AppConfig, *, method: str, resume: bool, limit: int | None) -> None:
    from rag_evidence.retrieval.run import run_retrieval_stage

    run_retrieval_stage(cfg, method=method, resume=resume, limit=limit)


def run_generate(cfg: AppConfig, *, resume: bool, limit: int | None) -> None:
    from rag_evidence.generation.run import run_generation_stage

    run_generation_stage(cfg, resume=resume, limit=limit)


def run_attribute(
    cfg: AppConfig,
    *,
    method: str,
    mode: str | None,
    resume: bool,
    limit: int | None,
    retry_failures: bool = False,
) -> None:
    from rag_evidence.attribution.run import run_attribution_stage

    run_attribution_stage(
        cfg,
        method=method,
        mode=mode,
        resume=resume,
        limit=limit,
        retry_failures=retry_failures,
    )


def run_evaluate(cfg: AppConfig, *, allow_partial: bool = False) -> None:
    from rag_evidence.evaluation.evaluate import evaluate_all

    evaluate_all(cfg, allow_partial=allow_partial)


def run_report(cfg: AppConfig) -> None:
    from rag_evidence.reporting.report import build_report

    build_report(cfg)


def run_serve(cfg: AppConfig) -> None:
    from rag_evidence.api import serve_app

    serve_app(cfg)


def run_status(cfg: AppConfig) -> None:
    from rag_evidence.storage.status import print_status

    print_status(cfg)


def run_export(cfg: AppConfig, *, out: Path | None) -> None:
    from rag_evidence.storage.transfer import export_results

    export_results(cfg, out=out)


def run_import_results(cfg: AppConfig, *, zip_path: Path) -> None:
    from rag_evidence.storage.transfer import import_results

    import_results(cfg, zip_path=zip_path)
