"""Optional third-party attribution adapters (registered only if importable)."""

import contextlib

with contextlib.suppress(ImportError):
    from rag_evidence.attribution.adapters import contextcite  # noqa: F401
