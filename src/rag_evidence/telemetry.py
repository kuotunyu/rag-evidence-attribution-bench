"""Timing, peak-VRAM, environment capture, deterministic seeding.

CUDA rules: every function here is safe without CUDA (and without torch installed at
all — the explorer image has no torch). On CPU, VRAM readings are None, never 0.0:
zero would be a fabricated measurement.
"""

from __future__ import annotations

import hashlib
import platform
import sys
import time
from types import TracebackType
from typing import Any


def _torch() -> Any | None:
    try:
        import torch

        return torch
    except ImportError:
        return None


def _cuda_available() -> bool:
    torch = _torch()
    return bool(torch is not None and torch.cuda.is_available())


class SampleTimer:
    """Monotonic timer; synchronizes CUDA on exit so GPU work is fully counted."""

    def __init__(self) -> None:
        self._start = 0.0
        self.elapsed_s = 0.0

    def __enter__(self) -> SampleTimer:
        if _cuda_available():
            _torch().cuda.synchronize()  # type: ignore[union-attr]
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if _cuda_available():
            _torch().cuda.synchronize()  # type: ignore[union-attr]
        self.elapsed_s = time.perf_counter() - self._start

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_s * 1000.0


def reset_peak_vram() -> None:
    if _cuda_available():
        _torch().cuda.reset_peak_memory_stats()  # type: ignore[union-attr]


def peak_vram_mb() -> float | None:
    """Allocator-level peak since the last reset; None when CUDA is unavailable."""
    if not _cuda_available():
        return None
    torch = _torch()
    return float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)  # type: ignore[union-attr]


def device_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": sys.platform,
        "machine": platform.machine(),
    }
    torch = _torch()
    if torch is None:
        info.update({"torch": None, "cuda": None, "gpu_name": None, "gpu_total_vram_mb": None})
        return info
    info["torch"] = torch.__version__
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["cuda"] = torch.version.cuda
        info["gpu_name"] = props.name
        info["gpu_total_vram_mb"] = round(props.total_memory / (1024.0 * 1024.0), 1)
    else:
        info.update({"cuda": None, "gpu_name": None, "gpu_total_vram_mb": None})
    return info


def library_versions() -> dict[str, str | None]:
    """Versions of the libraries that can change benchmark numbers."""
    out: dict[str, str | None] = {}
    for mod in ("torch", "transformers", "tokenizers", "accelerate", "bitsandbytes", "datasets"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def derive_seed(global_seed: int, *parts: str) -> int:
    """Deterministic per-(sample, method, …) seed. Never uses built-in hash()
    (randomized per process); sha256 is stable across machines and Python versions."""
    material = f"{global_seed}:" + ":".join(parts)
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")
