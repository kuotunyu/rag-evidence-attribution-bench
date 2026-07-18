"""Command-line interface.

Fixed surface (do not change without updating README + tests):

    python -m rag_evidence.cli data prepare        --config configs/smoke.yaml
    python -m rag_evidence.cli retrieve            --method bm25|dense|hybrid_rrf --config …
    python -m rag_evidence.cli generate            --config …
    python -m rag_evidence.cli attribute           --method citations|embedding|leave_one_out|… --config …
    python -m rag_evidence.cli evaluate            --config …
    python -m rag_evidence.cli report              --config …
    python -m rag_evidence.cli serve               --config …

Support commands (checkpoint / Colab round-trip): status, export, import-results.

All heavy imports (torch, transformers, datasets) happen lazily inside stage
modules — `--help` and `serve` in precomputed mode never import them.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
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


class RetrievalMethod(str, Enum):
    bm25 = "bm25"
    dense = "dense"
    hybrid_rrf = "hybrid_rrf"


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
    limit: LimitOpt = None,
) -> None:
    """Score how much each passage contributed to the target answer."""
    from rag_evidence.pipeline import run_attribute

    _run(run_attribute, config, method=method, mode=mode, resume=resume, limit=limit)


@app.command()
def evaluate(config: ConfigOpt) -> None:
    """Aggregate raw records into derived metrics (pure CPU, no model access)."""
    from rag_evidence.pipeline import run_evaluate

    _run(run_evaluate, config)


@app.command()
def report(config: ConfigOpt) -> None:
    """Render summary.json into report.md, figures, and the README results block."""
    from rag_evidence.pipeline import run_report

    _run(run_report, config)


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


if __name__ == "__main__":
    app()
