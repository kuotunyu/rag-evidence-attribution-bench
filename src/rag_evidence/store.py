"""ResultsStore: read-only, in-memory view over precomputed results for one split.

Shared by the FastAPI endpoints and the Gradio explorer (same process, same object).
Torch-free by design — the Docker explorer image has no ML stack at all.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rag_evidence.errors import SampleNotFoundError
from rag_evidence.storage.artifacts import RECORDS_FILE, read_json, read_records

logger = logging.getLogger(__name__)


def _load_jsonl_by_qid(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {rec["question_id"]: rec for rec in read_records(path)}


class ResultsStore:
    def __init__(self, results_raw: Path, results_derived: Path, split: str) -> None:
        self.split = split
        raw = results_raw / split

        self.samples = _load_jsonl_by_qid(raw / "samples" / RECORDS_FILE)
        self.retrieval: dict[str, dict[str, dict[str, Any]]] = {}
        for run_dir in sorted((raw / "retrieve").glob("*/")) if (raw / "retrieve").exists() else []:
            if (run_dir / RECORDS_FILE).exists():
                self.retrieval[run_dir.name] = _load_jsonl_by_qid(run_dir / RECORDS_FILE)
        self.generation: dict[str, dict[str, dict[str, Any]]] = {}
        for run_dir in sorted((raw / "generate").glob("*/")) if (raw / "generate").exists() else []:
            if (run_dir / RECORDS_FILE).exists():
                self.generation[run_dir.name] = _load_jsonl_by_qid(run_dir / RECORDS_FILE)
        self.attribution: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
        attribute_dir = raw / "attribute"
        if attribute_dir.exists():
            for mode_dir in sorted(attribute_dir.glob("*/")):
                methods = {}
                for run_dir in sorted(mode_dir.glob("*/")):
                    if (run_dir / RECORDS_FILE).exists():
                        methods[run_dir.name] = _load_jsonl_by_qid(run_dir / RECORDS_FILE)
                if methods:
                    self.attribution[mode_dir.name] = methods

        summary_path = results_derived / "summary.json"
        self.summary: dict[str, Any] | None = (
            read_json(summary_path) if summary_path.exists() else None
        )
        logger.info(
            "store[%s]: %d samples, retrieval=%s, generation=%s, attribution=%s",
            split,
            len(self.samples),
            sorted(self.retrieval),
            sorted(self.generation),
            {m: sorted(v) for m, v in self.attribution.items()},
        )

    # ------------------------------------------------------------------ access

    def list_ids(self) -> list[str]:
        return sorted(self.samples)

    def sample(self, sample_id: str) -> dict[str, Any]:
        try:
            return self.samples[sample_id]
        except KeyError:
            raise SampleNotFoundError(f"sample {sample_id!r} not in split {self.split!r}") from None

    def methods(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "retrieval": sorted(self.retrieval),
            "generation": sorted(self.generation),
            "attribution": {m: sorted(v) for m, v in self.attribution.items()},
            "n_samples": len(self.samples),
        }

    def retrieve(self, sample_id: str, method: str, k: int | None = None) -> dict[str, Any]:
        self.sample(sample_id)
        run = self.retrieval.get(method)
        if run is None:
            raise KeyError(f"no retrieval run {method!r}; available: {sorted(self.retrieval)}")
        rec = run.get(sample_id)
        if rec is None or rec.get("error") is not None:
            raise SampleNotFoundError(f"no successful {method} record for {sample_id}")
        ranking = rec["ranking"][:k] if k else rec["ranking"]
        return {**rec, "ranking": ranking}

    def answer(self, sample_id: str, generation: str | None = None) -> dict[str, Any]:
        self.sample(sample_id)
        if not self.generation:
            raise KeyError("no generation runs present (pending Colab execution)")
        name = generation or sorted(self.generation)[0]
        run = self.generation.get(name)
        if run is None:
            raise KeyError(f"no generation run {name!r}; available: {sorted(self.generation)}")
        rec = run.get(sample_id)
        if rec is None:
            raise SampleNotFoundError(f"no generation record for {sample_id}")
        return rec

    def attribute(self, sample_id: str, method: str, mode: str = "gold") -> dict[str, Any]:
        self.sample(sample_id)
        modes = self.attribution.get(mode)
        if modes is None:
            raise KeyError(
                f"no attribution runs for mode {mode!r}; available: {sorted(self.attribution)}"
            )
        run = modes.get(method)
        if run is None:
            raise KeyError(f"no attribution method {method!r}; available: {sorted(modes)}")
        rec = run.get(sample_id)
        if rec is None:
            raise SampleNotFoundError(f"no {mode}/{method} record for {sample_id}")
        return rec

    def benchmark_record(self, sample_id: str) -> dict[str, Any]:
        """Everything known about one sample, merged for display."""
        sample = self.sample(sample_id)
        return {
            "sample": sample,
            "retrieval": {
                m: run.get(sample_id) for m, run in self.retrieval.items() if sample_id in run
            },
            "generation": {
                n: run.get(sample_id) for n, run in self.generation.items() if sample_id in run
            },
            "attribution": {
                mode: {
                    meth: run.get(sample_id) for meth, run in methods.items() if sample_id in run
                }
                for mode, methods in self.attribution.items()
            },
        }
