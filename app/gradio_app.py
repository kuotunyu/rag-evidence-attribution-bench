"""Standalone Gradio launcher (the API-mounted route is `rag-evidence serve`).

Usage: uv run python app/gradio_app.py [config_path]
"""

from __future__ import annotations

import sys
from pathlib import Path

from rag_evidence.config import load_config
from rag_evidence.explorer import build_demo
from rag_evidence.logging_utils import setup_logging
from rag_evidence.store import ResultsStore


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs/full.yaml")
    cfg = load_config(config_path)
    setup_logging(cfg.runtime.log_level)
    store = ResultsStore(cfg.results_raw_dir, cfg.results_derived_dir, cfg.split)
    build_demo(store).launch(server_name=cfg.serve.host, server_port=cfg.serve.port)


if __name__ == "__main__":
    main()
