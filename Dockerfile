# syntax=docker/dockerfile:1
# CPU-first explorer image: serves PRECOMPUTED results only.
# No torch, no transformers, no model weights — HF_HUB_OFFLINE is baked in so the
# container is structurally unable to download models.

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# dependency layer (cache-friendly: lock + manifest only)
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra app
# project layer
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra app

FROM python:3.11-slim-bookworm AS runtime
RUN useradd --create-home appuser
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY configs ./configs
COPY results ./results
COPY data/manifests ./data/manifests
COPY app ./app
ENV PATH="/app/.venv/bin:$PATH" \
    RAG_EVIDENCE_HOST=0.0.0.0 \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=25s --retries=5 \
  CMD ["python", "-c", "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3).status==200 else 1)"]
CMD ["python", "-m", "rag_evidence.cli", "serve", "--config", "configs/full.yaml"]
