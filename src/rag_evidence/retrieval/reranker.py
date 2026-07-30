"""Replaceable cross-encoder reranker adapters and an append-only score cache.

The benchmark runner depends only on :class:`RerankerAdapter`. The initial adapter uses
plain Hugging Face Transformers so this repository never imports runtime code from the
Longcare RAG repository whose model choice motivated the controlled extension.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from rag_evidence.config import RerankerConfig
from rag_evidence.errors import ArtifactError, GpuRequiredError
from rag_evidence.storage.artifacts import append_record, read_records

logger = logging.getLogger(__name__)


class RerankerAdapter(Protocol):
    """Minimal replaceable interface used by the retrieval runner."""

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]: ...

    def info(self) -> dict[str, Any]: ...


class HFCrossEncoderReranker:
    """Scalar sequence-classification logits for `(query, passage)` pairs."""

    def __init__(self, cfg: RerankerConfig) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if cfg.device == "cuda" and not torch.cuda.is_available():
            raise GpuRequiredError(
                "reranking.reranker.device=cuda but CUDA is unavailable; do not silently "
                "fall back because dtype/device are preregistered scientific settings"
            )
        self.cfg = cfg
        self._torch = torch
        dtype = getattr(torch, cfg.dtype)
        logger.info(
            "loading reranker %s@%s tokenizer=%s@%s on %s/%s",
            cfg.model_id,
            cfg.model_revision,
            cfg.tokenizer_id,
            cfg.tokenizer_revision,
            cfg.device,
            cfg.dtype,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer_id,
            revision=cfg.tokenizer_revision,
            trust_remote_code=False,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_id,
            revision=cfg.model_revision,
            trust_remote_code=False,
            dtype=dtype,
        ).to(cfg.device)
        self._model.eval()
        self._effective_dtype = str(next(self._model.parameters()).dtype).removeprefix("torch.")
        self._num_parameters = sum(p.numel() for p in self._model.parameters())

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        scores: list[float] = []
        for start in range(0, len(pairs), self.cfg.batch_size):
            batch = pairs[start : start + self.cfg.batch_size]
            queries = [pair[0] for pair in batch]
            passages = [pair[1] for pair in batch]
            inputs = self._tokenizer(
                queries,
                passages,
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
                return_tensors="pt",
            ).to(self.cfg.device)
            with self._torch.inference_mode():
                logits = self._model(**inputs, return_dict=True).logits
            flat = logits.reshape(-1).float().cpu().tolist()
            if len(flat) != len(batch):
                raise ArtifactError(
                    f"reranker returned {len(flat)} logits for {len(batch)} input pairs"
                )
            scores.extend(float(value) for value in flat)
        return scores

    def info(self) -> dict[str, Any]:
        return {
            "adapter": self.cfg.adapter,
            "model_id": self.cfg.model_id,
            "model_revision": self.cfg.model_revision,
            "tokenizer_id": self.cfg.tokenizer_id,
            "tokenizer_revision": self.cfg.tokenizer_revision,
            "tokenizer_class": type(self._tokenizer).__name__,
            "max_length": self.cfg.max_length,
            "device": self.cfg.device,
            "dtype_requested": self.cfg.dtype,
            "dtype_effective": self._effective_dtype,
            "batch_size": self.cfg.batch_size,
            "num_parameters": self._num_parameters,
        }


def build_reranker(cfg: RerankerConfig) -> RerankerAdapter:
    if cfg.adapter == "hf_sequence_classification":
        return HFCrossEncoderReranker(cfg)
    raise ValueError(f"unknown reranker adapter {cfg.adapter!r}")


def cache_identity(cfg: RerankerConfig) -> dict[str, Any]:
    """Only fields that can change a cached score."""
    return {
        "adapter": cfg.adapter,
        "model_id": cfg.model_id,
        "model_revision": cfg.model_revision,
        "tokenizer_id": cfg.tokenizer_id,
        "tokenizer_revision": cfg.tokenizer_revision,
        "max_length": cfg.max_length,
        "dtype": cfg.dtype,
    }


def score_cache_key(
    identity: Mapping[str, Any],
    *,
    question: str,
    passage_id: str,
    passage_text: str,
) -> str:
    material = {
        "identity": dict(identity),
        "question": question,
        "passage_id": passage_id,
        "passage_sha256": hashlib.sha256(passage_text.encode("utf-8")).hexdigest(),
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class RerankScoreCache:
    """Append-only scalar cache; duplicate keys are harmless and last-write-wins."""

    def __init__(self, path: Path | None, identity: Mapping[str, Any]) -> None:
        self.path = path
        self.identity = dict(identity)
        self._scores: dict[str, float] = {}
        if path is not None and path.exists():
            for rec in read_records(path):
                if rec.get("identity") != self.identity:
                    raise ArtifactError(
                        f"rerank cache identity mismatch in {path}; refusing mixed model scores"
                    )
                self._scores[str(rec["key"])] = float(rec["score"])
            logger.info("rerank cache %s: %d scores loaded", path, len(self._scores))

    def get(self, key: str) -> float | None:
        return self._scores.get(key)

    def put(
        self,
        key: str,
        score: float,
        *,
        question_id: str,
        passage_id: str,
    ) -> None:
        self._scores[key] = float(score)
        if self.path is not None:
            append_record(
                self.path,
                {
                    "key": key,
                    "identity": self.identity,
                    "question_id": question_id,
                    "passage_id": passage_id,
                    "score": float(score),
                },
            )
