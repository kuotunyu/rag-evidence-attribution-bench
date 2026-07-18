"""FastAPI app over precomputed results (spec-fixed endpoint surface).

Precomputed mode serves stored records only; nothing here can trigger a model download
or GPU work. `serve.mode: live` is reserved for the optional GPU compose profile and is
not implemented in v0.1 — selecting it fails fast with a clear message.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

import rag_evidence
from rag_evidence.config import AppConfig
from rag_evidence.errors import ConfigError, SampleNotFoundError
from rag_evidence.store import ResultsStore

logger = logging.getLogger(__name__)


class RetrieveRequest(BaseModel):
    sample_id: str
    method: str = "bm25"
    k: int | None = Field(default=None, ge=1)


class AnswerRequest(BaseModel):
    sample_id: str
    generation: str | None = None  # generation run name; default = first available


class AttributeRequest(BaseModel):
    sample_id: str
    method: str
    mode: str = "gold"


def create_app(cfg: AppConfig) -> FastAPI:
    if cfg.serve.mode != "precomputed":
        raise ConfigError(
            "serve.mode 'live' is not implemented in v0.1 — the explorer serves "
            "precomputed results; GPU live mode is a future optional compose profile"
        )
    store = ResultsStore(cfg.results_raw_dir, cfg.results_derived_dir, cfg.split)
    app = FastAPI(
        title="rag-evidence-attribution-bench",
        version=rag_evidence.__version__,
        description="Precomputed benchmark explorer API (HotpotQA distractor, Qwen3).",
    )
    app.state.store = store

    def _wrap(fn: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except SampleNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc.args[0])) from exc

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "mode": cfg.serve.mode,
            "split": cfg.split,
            "n_samples": len(store.samples),
            "version": rag_evidence.__version__,
        }

    @app.get("/methods")
    def methods() -> dict[str, Any]:
        return store.methods()

    @app.get("/benchmark/{sample_id}")
    def benchmark(sample_id: str) -> Any:
        return _wrap(store.benchmark_record, sample_id)

    @app.post("/retrieve")
    def retrieve(req: RetrieveRequest) -> Any:
        return _wrap(store.retrieve, req.sample_id, req.method, req.k)

    @app.post("/answer")
    def answer(req: AnswerRequest) -> Any:
        return _wrap(store.answer, req.sample_id, req.generation)

    @app.post("/attribute")
    def attribute(req: AttributeRequest) -> Any:
        return _wrap(store.attribute, req.sample_id, req.method, req.mode)

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/app")

    return app


def serve_app(cfg: AppConfig) -> None:
    import uvicorn

    app = create_app(cfg)
    import gradio as gr

    from rag_evidence.explorer import build_demo  # lazy: pulls gradio

    demo = build_demo(app.state.store)
    gr.mount_gradio_app(app, demo, path="/app")
    logger.info(
        "serving on http://%s:%d (API at /, explorer at /app)", cfg.serve.host, cfg.serve.port
    )
    uvicorn.run(app, host=cfg.serve.host, port=cfg.serve.port, log_level="info")
