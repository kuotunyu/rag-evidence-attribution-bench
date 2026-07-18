"""Qwen3-Embedding wrapper + on-disk embedding cache.

Qwen3-Embedding is asymmetric: QUERIES get the instruction prompt (the model ships a
"query" prompt in its sentence-transformers config), DOCUMENTS are embedded plain.
Mixing these up silently degrades retrieval — this wrapper is the only embedding
entry point in the package.

The cache stores passage embeddings keyed by passage_id in a compressed .npz under
results/raw/cache/ (gitignored); dense retrieval and embedding attribution share it.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rag_evidence.config import resolve_device
from rag_evidence.errors import ArtifactError

logger = logging.getLogger(__name__)


def model_tag(model_id: str) -> str:
    return model_id.replace("/", "__")


class Embedder:
    """Lazy-loaded sentence-transformers model with the Qwen3 query/document asymmetry."""

    def __init__(self, model: Any, model_id: str, device: str, normalize: bool) -> None:
        self._model = model
        self.model_id = model_id
        self.device = device
        self.normalize = normalize

    @classmethod
    def load(
        cls,
        model_id: str,
        *,
        device: str = "auto",
        max_length: int = 1024,
        normalize: bool = True,
    ) -> Embedder:
        resolved = resolve_device(device)
        logger.info("loading embedder %s on %s", model_id, resolved)
        from sentence_transformers import SentenceTransformer  # lazy heavy import

        model = SentenceTransformer(model_id, device=resolved)
        model.max_seq_length = max_length
        return cls(model, model_id, resolved, normalize)

    def embed_queries(self, texts: Sequence[str], *, batch_size: int = 8) -> np.ndarray:
        return self._encode(texts, batch_size=batch_size, is_query=True)

    def embed_passages(self, texts: Sequence[str], *, batch_size: int = 8) -> np.ndarray:
        return self._encode(texts, batch_size=batch_size, is_query=False)

    def _encode(self, texts: Sequence[str], *, batch_size: int, is_query: bool) -> np.ndarray:
        kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "normalize_embeddings": self.normalize,
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }
        if is_query:
            kwargs["prompt_name"] = "query"  # instruction applies to queries ONLY
        out = self._model.encode(list(texts), **kwargs)
        return np.asarray(out, dtype=np.float32)


class EmbeddingCache:
    """Append-only vector cache: {key: vector} persisted as .npz (keys + matrix)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._store: dict[str, np.ndarray] = {}
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as data:
                    keys = [str(k) for k in data["keys"]]
                    matrix = data["vectors"]
            except Exception as exc:
                raise ArtifactError(f"corrupt embedding cache {path}: {exc}") from exc
            self._store = {k: matrix[i] for i, k in enumerate(keys)}
            logger.info("embedding cache %s: %d vectors loaded", path.name, len(self._store))

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self._store)
        matrix = np.stack([self._store[k] for k in keys]) if keys else np.zeros((0, 0), np.float32)
        tmp = self.path.with_suffix(f".tmp.{os.getpid()}.npz")
        try:
            np.savez_compressed(tmp, keys=np.array(keys), vectors=matrix)
            os.replace(tmp, self.path)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise ArtifactError(f"failed to write embedding cache {self.path}: {exc}") from exc

    def get_or_compute(
        self,
        keys: Sequence[str],
        texts: Sequence[str],
        fn: Callable[[Sequence[str]], np.ndarray],
    ) -> np.ndarray:
        if len(keys) != len(texts):
            raise ValueError("keys and texts length mismatch")
        missing = [i for i, k in enumerate(keys) if k not in self._store]
        if missing:
            vectors = fn([texts[i] for i in missing])
            if len(vectors) != len(missing):
                raise ArtifactError("embedding fn returned wrong number of vectors")
            for j, i in enumerate(missing):
                self._store[keys[i]] = np.asarray(vectors[j], dtype=np.float32)
            self._save()
            logger.info("embedding cache %s: +%d vectors", self.path.name, len(missing))
        return np.stack([self._store[k] for k in keys])
