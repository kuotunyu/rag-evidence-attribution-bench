"""Command-line interface.

Fixed surface (do not change without updating README + tests):

    python -m rag_evidence.cli data prepare        --config configs/smoke.yaml
    python -m rag_evidence.cli data challenge      --config configs/v2/eval.yaml
    python -m rag_evidence.cli retrieve
        --method bm25|dense|hybrid_rrf|hybrid_rrf_rerank --config …
    python -m rag_evidence.cli generate            --config …
    python -m rag_evidence.cli attribute  --method citations|embedding|leave_one_out|…
    python -m rag_evidence.cli evaluate            --config …
    python -m rag_evidence.cli report              --config …
    python -m rag_evidence.cli serve               --config …

Support commands (checkpoint / Colab round-trip): status, export, import-results.

All heavy imports (torch, transformers, datasets) happen lazily inside stage
modules — `--help` and `serve` in precomputed mode never import them.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer

import rag_evidence

app = typer.Typer(
    name="rag-evidence",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    help="RAG evidence attribution benchmark (HotpotQA distractor, Qwen3).",
)
data_app = typer.Typer(no_args_is_help=True, help="Dataset preparation commands.")
app.add_typer(data_app, name="data")
reranking_app = typer.Typer(
    no_args_is_help=True,
    help="Controlled cross-encoder reranking extension commands.",
)
app.add_typer(reranking_app, name="reranking")
challenge_app = typer.Typer(
    no_args_is_help=True,
    help="Human-gated answerability challenge execution (generate/attribute/evaluate/report).",
)
app.add_typer(challenge_app, name="challenge")
annotation_app = typer.Typer(
    no_args_is_help=True,
    help="Build blind packages or run the offline human annotation console.",
)
app.add_typer(annotation_app, name="annotation")

ConfigOpt = Annotated[
    Path,
    typer.Option("--config", exists=True, dir_okay=False, readable=True, help="YAML config path."),
]
ResumeOpt = Annotated[
    bool, typer.Option("--resume", help="Resume an interrupted run (skip completed samples).")
]
LimitOpt = Annotated[
    int | None,
    typer.Option("--limit", min=1, help="Debug: process at most N samples (recorded in run_meta)."),
]


class RetrievalMethod(StrEnum):
    bm25 = "bm25"
    dense = "dense"
    hybrid_rrf = "hybrid_rrf"
    hybrid_rrf_rerank = "hybrid_rrf_rerank"


class RerankingArm(StrEnum):
    bm25 = "bm25"
    dense = "dense"
    hybrid_rrf = "hybrid_rrf"
    hybrid_rrf_rerank = "hybrid_rrf_rerank"


class ChallengePhase(StrEnum):
    pilot = "pilot"
    confirmatory = "confirmatory"


class ChallengeVariant(StrEnum):
    missing_hop = "missing_hop"
    evidence_swap = "evidence_swap"


class HandoffWheelMode(StrEnum):
    byte_identical = "byte-identical"
    per_build_hash_verified = "per-build-hash-verified"


def _challenge_config(
    cfg: Any,
    *,
    phase: ChallengePhase,
    variant: ChallengeVariant,
    assignments: Path,
    eligibility: Path,
) -> Any:
    from rag_evidence.data.challenge_execution import build_challenge_execution_config

    return build_challenge_execution_config(
        cfg,
        phase=phase.value,
        variant=variant.value,
        challenge_records_path=(
            cfg.results_raw_dir / cfg.split / "challenge" / "samples" / "records.jsonl"
        ),
        assignment_manifest_path=assignments,
        eligibility_path=eligibility,
        natural_samples_path=cfg.results_raw_dir / cfg.split / "samples" / "records.jsonl",
    )


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"rag-evidence-attribution-bench {rag_evidence.__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version_callback, is_eager=True)
    ] = False,
) -> None:
    """RAG evidence attribution benchmark."""


def _run(stage: Callable[..., Any], config_path: Path, /, **kwargs: Any) -> None:
    """Load config, set up logging, run a stage, map expected errors to exit 1."""
    from rag_evidence.config import load_config
    from rag_evidence.errors import RagEvidenceError
    from rag_evidence.logging_utils import setup_logging

    cfg = load_config(config_path)
    setup_logging(cfg.runtime.log_level)
    try:
        stage(cfg, **kwargs)
    except RagEvidenceError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@data_app.command("prepare")
def data_prepare(config: ConfigOpt) -> None:
    """Download/verify HotpotQA distractor, build split manifest + prepared JSONL."""
    from rag_evidence.pipeline import run_data_prepare

    _run(run_data_prepare, config)


@data_app.command("challenge")
def data_challenge(config: ConfigOpt) -> None:
    """Validate v2 source artifacts and build answerability challenge v1."""
    from rag_evidence.pipeline import run_data_challenge

    _run(run_data_challenge, config)


@app.command()
def retrieve(
    config: ConfigOpt,
    method: Annotated[RetrievalMethod, typer.Option("--method")],
    resume: ResumeOpt = False,
    limit: LimitOpt = None,
) -> None:
    """Rank each question's candidate passages with the chosen retriever."""
    from rag_evidence.pipeline import run_retrieve

    _run(run_retrieve, config, method=method.value, resume=resume, limit=limit)


@app.command()
def generate(
    config: ConfigOpt,
    resume: ResumeOpt = False,
    limit: LimitOpt = None,
) -> None:
    """Generate cited answers with the configured backend (GPU stage — Colab)."""
    from rag_evidence.pipeline import run_generate

    _run(run_generate, config, resume=resume, limit=limit)


@app.command()
def attribute(
    config: ConfigOpt,
    method: Annotated[
        str,
        typer.Option(
            "--method",
            help="citations | embedding | leave_one_out | control_* | registered adapter name",
        ),
    ],
    mode: Annotated[
        str | None,
        typer.Option("--mode", help="gold | generated (default: both modes from config)"),
    ] = None,
    resume: ResumeOpt = False,
    retry_failures: Annotated[
        bool,
        typer.Option(
            "--retry-failures",
            help="With --resume, append retries only for records whose latest attempt failed.",
        ),
    ] = False,
    limit: LimitOpt = None,
) -> None:
    """Score how much each passage contributed to the target answer."""
    from rag_evidence.pipeline import run_attribute

    _run(
        run_attribute,
        config,
        method=method,
        mode=mode,
        resume=resume,
        retry_failures=retry_failures,
        limit=limit,
    )


@app.command()
def evaluate(
    config: ConfigOpt,
    allow_partial: Annotated[
        bool,
        typer.Option(
            "--allow-partial",
            help="Include unfinished runs; their summary entries are marked partial.",
        ),
    ] = False,
) -> None:
    """Aggregate raw records into derived metrics (pure CPU, no model access)."""
    from rag_evidence.pipeline import run_evaluate

    _run(run_evaluate, config, allow_partial=allow_partial)


@app.command()
def report(config: ConfigOpt) -> None:
    """Render summary.json into report.md, figures, and the README results block."""
    from rag_evidence.pipeline import run_report

    _run(run_report, config)


@challenge_app.command("generate")
def challenge_generate(
    config: ConfigOpt,
    phase: Annotated[ChallengePhase, typer.Option("--phase")],
    variant: Annotated[ChallengeVariant, typer.Option("--variant")],
    assignments: Annotated[
        Path, typer.Option("--assignments", exists=True, dir_okay=False, readable=True)
    ],
    eligibility: Annotated[
        Path, typer.Option("--eligibility", exists=True, dir_okay=False, readable=True)
    ],
    resume: ResumeOpt = False,
    limit: LimitOpt = None,
) -> None:
    """Generate on one eligible challenge arm; transformation labels are not truth."""
    from rag_evidence.generation.run import run_generation_stage

    _run(
        lambda cfg, **kwargs: run_generation_stage(
            _challenge_config(
                cfg,
                phase=phase,
                variant=variant,
                assignments=assignments,
                eligibility=eligibility,
            ),
            **kwargs,
        ),
        config,
        resume=resume,
        limit=limit,
    )


@challenge_app.command("attribute")
def challenge_attribute(
    config: ConfigOpt,
    phase: Annotated[ChallengePhase, typer.Option("--phase")],
    variant: Annotated[ChallengeVariant, typer.Option("--variant")],
    assignments: Annotated[
        Path, typer.Option("--assignments", exists=True, dir_okay=False, readable=True)
    ],
    eligibility: Annotated[
        Path, typer.Option("--eligibility", exists=True, dir_okay=False, readable=True)
    ],
    method: Annotated[str, typer.Option("--method")],
    mode: Annotated[str, typer.Option("--mode")] = "generated",
    resume: ResumeOpt = False,
    retry_failures: Annotated[bool, typer.Option("--retry-failures")] = False,
    limit: LimitOpt = None,
) -> None:
    """Attribute one eligible challenge arm (generated-answer mode by default)."""
    from rag_evidence.attribution.run import run_attribution_stage

    _run(
        lambda cfg, **kwargs: run_attribution_stage(
            _challenge_config(
                cfg,
                phase=phase,
                variant=variant,
                assignments=assignments,
                eligibility=eligibility,
            ),
            **kwargs,
        ),
        config,
        method=method,
        mode=mode,
        resume=resume,
        retry_failures=retry_failures,
        limit=limit,
    )


@challenge_app.command("evaluate")
def challenge_evaluate(
    config: ConfigOpt,
    phase: Annotated[ChallengePhase, typer.Option("--phase")],
    variant: Annotated[ChallengeVariant, typer.Option("--variant")],
    assignments: Annotated[
        Path, typer.Option("--assignments", exists=True, dir_okay=False, readable=True)
    ],
    eligibility: Annotated[
        Path, typer.Option("--eligibility", exists=True, dir_okay=False, readable=True)
    ],
    allow_partial: Annotated[bool, typer.Option("--allow-partial")] = False,
) -> None:
    """Evaluate one eligible challenge arm in its isolated derived root."""
    from rag_evidence.evaluation.evaluate import evaluate_all

    _run(
        lambda cfg, **kwargs: evaluate_all(
            _challenge_config(
                cfg,
                phase=phase,
                variant=variant,
                assignments=assignments,
                eligibility=eligibility,
            ),
            **kwargs,
        ),
        config,
        allow_partial=allow_partial,
    )


@challenge_app.command("report")
def challenge_report(
    config: ConfigOpt,
    phase: Annotated[ChallengePhase, typer.Option("--phase")],
    variant: Annotated[ChallengeVariant, typer.Option("--variant")],
    assignments: Annotated[
        Path, typer.Option("--assignments", exists=True, dir_okay=False, readable=True)
    ],
    eligibility: Annotated[
        Path, typer.Option("--eligibility", exists=True, dir_okay=False, readable=True)
    ],
) -> None:
    """Report one challenge arm without changing public README result blocks."""
    from rag_evidence.reporting.report import build_report

    _run(
        lambda cfg: build_report(
            _challenge_config(
                cfg,
                phase=phase,
                variant=variant,
                assignments=assignments,
                eligibility=eligibility,
            )
        ),
        config,
    )


@annotation_app.command("collect")
def annotation_collect(
    manifest: Annotated[
        Path,
        typer.Option("--manifest", exists=True, dir_okay=False, readable=True),
    ],
    submission: Annotated[
        list[Path],
        typer.Option("--submission", exists=True, dir_okay=False, readable=True),
    ],
    amendment: Annotated[
        list[Path],
        typer.Option("--amendment", exists=True, dir_okay=False, readable=True),
    ],
    out: Annotated[Path, typer.Option("--out", file_okay=False)],
) -> None:
    """Collect two paired human streams and resolve append-only amendments."""
    from rag_evidence.annotation.coordinator import collect_annotation_streams
    from rag_evidence.errors import RagEvidenceError

    try:
        result = collect_annotation_streams(manifest, submission, amendment, out)
    except RagEvidenceError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"collected {result.completed_tasks}/{result.assigned_tasks} tasks into {result.output_dir}"
    )
    if not result.complete:
        typer.secho(
            "blocked: collection is incomplete; no disagreement queue was created",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=2)


@annotation_app.command("adjudicate")
def annotation_adjudicate(
    manifest: Annotated[
        Path,
        typer.Option("--manifest", exists=True, dir_okay=False, readable=True),
    ],
    effective: Annotated[
        Path,
        typer.Option("--effective", exists=True, dir_okay=False, readable=True),
    ],
    state: Annotated[Path, typer.Option("--state", file_okay=False)],
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 8002,
) -> None:
    """Serve the coordinator-only third-human disagreement console."""
    from rag_evidence.annotation.runtime import serve_adjudication
    from rag_evidence.errors import RagEvidenceError

    try:
        serve_adjudication(manifest, effective, state, host=host, port=port)
    except RagEvidenceError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@annotation_app.command("finalize-pilot")
def annotation_finalize_pilot(
    manifest: Annotated[
        Path,
        typer.Option("--manifest", exists=True, dir_okay=False, readable=True),
    ],
    originals: Annotated[
        Path,
        typer.Option("--originals", exists=True, dir_okay=False, readable=True),
    ],
    amendments: Annotated[
        Path,
        typer.Option("--amendments", exists=True, dir_okay=False, readable=True),
    ],
    adjudications: Annotated[
        Path,
        typer.Option("--adjudications", exists=True, dir_okay=False, readable=True),
    ],
    protocol: Annotated[
        Path,
        typer.Option("--protocol", exists=True, dir_okay=False, readable=True),
    ],
    out: Annotated[Path, typer.Option("--out", file_okay=False)],
) -> None:
    """Write the fixed pilot accounting, IAA, privacy, and verdict artifact set."""
    from rag_evidence.annotation.finalize import (
        PilotVerdictName,
        finalize_pilot,
    )
    from rag_evidence.errors import RagEvidenceError

    try:
        result = finalize_pilot(
            manifest,
            originals,
            amendments,
            adjudications,
            protocol,
            out,
        )
    except RagEvidenceError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(result.verdict.value)
    if result.verdict is not PilotVerdictName.READY_FOR_HUMAN_FREEZE_REVIEW:
        raise typer.Exit(code=2)


@annotation_app.command("build-handoff")
def annotation_build_handoff(
    spec: Annotated[
        Path,
        typer.Option("--spec", exists=True, dir_okay=False, readable=True),
    ],
    wheel: Annotated[
        Path,
        typer.Option("--wheel", exists=True, dir_okay=False, readable=True),
    ],
    clean_install: Annotated[
        Path,
        typer.Option("--clean-install", exists=True, dir_okay=False, readable=True),
    ],
    source_commit: Annotated[str, typer.Option("--source-commit")],
    build_time: Annotated[str, typer.Option("--build-time")],
    wheel_reproducibility: Annotated[
        HandoffWheelMode,
        typer.Option("--wheel-reproducibility"),
    ],
    out: Annotated[Path, typer.Option("--out", file_okay=False)],
) -> None:
    """Build two disjoint, checksum-bound annotator kits outside the repository."""
    from rag_evidence.annotation.handoff import (
        CleanInstallVerification,
        build_handoff,
    )
    from rag_evidence.errors import RagEvidenceError

    try:
        verification = CleanInstallVerification.model_validate_json(
            clean_install.read_text(encoding="utf-8")
        )
        receipt = build_handoff(
            spec_path=spec,
            wheel_path=wheel,
            external_root=out,
            source_commit=source_commit,
            build_time=build_time,
            clean_install=verification,
            wheel_reproducibility=wheel_reproducibility.value,
        )
    except (RagEvidenceError, ValueError) as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"built isolated kits at {out}; wheel={receipt.wheel.reproducibility}; "
        f"commit={receipt.source_commit_sha}"
    )


@annotation_app.command("rehearse-synthetic")
def annotation_rehearse_synthetic(
    out: Annotated[Path, typer.Option("--out", file_okay=False)],
    repository_root: Annotated[
        Path,
        typer.Option("--repository-root", exists=True, file_okay=False, readable=True),
    ] = Path("."),
) -> None:
    """Run an invented 40-task operational rehearsal twice outside the repository."""
    from rag_evidence.annotation.rehearsal import run_synthetic_rehearsal
    from rag_evidence.errors import RagEvidenceError

    try:
        result = run_synthetic_rehearsal(out, repository_root)
    except RagEvidenceError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"synthetic tasks={result.tasks}; amendments={result.amendments}; "
        f"disagreements={result.disagreements}; adjudications={result.adjudications}; "
        f"defect_exclusions={result.dataset_defect_exclusions}; "
        f"verdict={result.verdict}; byte_identical={result.repeat_byte_identical}"
    )


@annotation_app.command("package-pilot")
def annotation_package_pilot(
    config: ConfigOpt,
    out: Annotated[Path, typer.Option("--out", file_okay=False)],
) -> None:
    """Create the deterministic 20-parent, decision-free pilot assignment package."""
    from rag_evidence.annotation.package import build_pilot_package

    _run(lambda cfg: build_pilot_package(cfg, out), config)


@annotation_app.command("build-coordinator-manifest")
def annotation_build_coordinator_manifest(
    challenge_records: Annotated[
        Path,
        typer.Option("--challenge-records", exists=True, dir_okay=False, readable=True),
    ],
    package_a: Annotated[
        Path, typer.Option("--package-a", exists=True, dir_okay=False, readable=True)
    ],
    package_b: Annotated[
        Path, typer.Option("--package-b", exists=True, dir_okay=False, readable=True)
    ],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
) -> None:
    """Rebuild the coordinator-only v2 manifest outside the source repository."""
    from rag_evidence.annotation.package import rebuild_coordinator_manifest
    from rag_evidence.errors import RagEvidenceError

    try:
        manifest = rebuild_coordinator_manifest(
            challenge_records,
            package_a,
            package_b,
            output,
            repository_root=Path.cwd(),
        )
    except RagEvidenceError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"wrote coordinator manifest with {len(manifest.coordinator_tasks)} tasks to {output}"
    )


@annotation_app.command("serve")
def annotation_serve(
    package: Annotated[Path, typer.Option("--package", exists=True, dir_okay=False, readable=True)],
    state: Annotated[Path, typer.Option("--state", file_okay=False)],
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 8001,
) -> None:
    """Serve one annotator package locally; state remains outside the clean package."""
    from rag_evidence.annotation.runtime import serve_annotation
    from rag_evidence.errors import RagEvidenceError

    try:
        serve_annotation(package, state, host=host, port=port)
    except RagEvidenceError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def serve(config: ConfigOpt) -> None:
    """Serve the FastAPI API with the Gradio explorer mounted at /app."""
    from rag_evidence.pipeline import run_serve

    _run(run_serve, config)


@app.command()
def status(config: ConfigOpt) -> None:
    """Show per-stage completed/expected counts for the config's split."""
    from rag_evidence.pipeline import run_status

    _run(run_status, config)


@app.command()
def export(
    config: ConfigOpt,
    out: Annotated[Path | None, typer.Option("--out", help="Output zip path.")] = None,
) -> None:
    """Package results for the config's split into a single validated zip."""
    from rag_evidence.pipeline import run_export

    _run(run_export, config, out=out)


@app.command("import-results")
def import_results(
    config: ConfigOpt,
    zip_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Validate and import a results zip produced by `export` (e.g. from Colab)."""
    from rag_evidence.pipeline import run_import_results

    _run(run_import_results, config, zip_path=zip_path)


@reranking_app.command("generate")
def reranking_generate(
    config: ConfigOpt,
    arm: Annotated[RerankingArm, typer.Option("--arm")],
    resume: ResumeOpt = False,
    limit: LimitOpt = None,
) -> None:
    """Generate one preregistered top-k context arm."""
    from rag_evidence.generation.run import run_generation_stage
    from rag_evidence.reranking.experiment import arm_config

    _run(
        lambda cfg, **kwargs: run_generation_stage(arm_config(cfg, arm.value), **kwargs),
        config,
        resume=resume,
        limit=limit,
    )


@reranking_app.command("attribute")
def reranking_attribute(
    config: ConfigOpt,
    arm: Annotated[RerankingArm, typer.Option("--arm")],
    method: Annotated[str, typer.Option("--method")],
    mode: Annotated[
        str | None,
        typer.Option("--mode", help="gold | generated (default: both modes from config)"),
    ] = None,
    resume: ResumeOpt = False,
    retry_failures: Annotated[
        bool,
        typer.Option(
            "--retry-failures",
            help="With --resume, append retries only for records whose latest attempt failed.",
        ),
    ] = False,
    limit: LimitOpt = None,
) -> None:
    """Run an attribution method in one namespaced reranking arm."""
    from rag_evidence.attribution.run import run_attribution_stage
    from rag_evidence.reranking.experiment import arm_config

    _run(
        lambda cfg, **kwargs: run_attribution_stage(arm_config(cfg, arm.value), **kwargs),
        config,
        method=method,
        mode=mode,
        resume=resume,
        retry_failures=retry_failures,
        limit=limit,
    )


@reranking_app.command("evaluate")
def reranking_evaluate(
    config: ConfigOpt,
    allow_partial: Annotated[
        bool,
        typer.Option("--allow-partial", help="Include visibly marked partial extension runs."),
    ] = False,
) -> None:
    """Aggregate extension raw records and build comparison/error artifacts."""
    from rag_evidence.evaluation.evaluate import evaluate_all
    from rag_evidence.reranking.report import build_reranking_comparison

    def stage(cfg: Any, **kwargs: Any) -> None:
        evaluate_all(cfg, **kwargs)
        build_reranking_comparison(cfg)

    _run(stage, config, allow_partial=allow_partial)


@reranking_app.command("report")
def reranking_report(config: ConfigOpt) -> None:
    """Render machine-readable extension comparisons into Markdown/README."""
    from rag_evidence.reranking.report import build_reranking_report

    _run(build_reranking_report, config)


if __name__ == "__main__":
    app()
