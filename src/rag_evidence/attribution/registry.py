"""Method registry — the single extension point for ContextCite / ARC-JSD adapters."""

from __future__ import annotations

from rag_evidence.attribution.base import AttributionMethod
from rag_evidence.errors import ConfigError

_REGISTRY: dict[str, type[AttributionMethod]] = {}


def register(cls: type[AttributionMethod]) -> type[AttributionMethod]:
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls.__name__} has no `name`")
    if name in _REGISTRY:
        raise ValueError(f"attribution method {name!r} registered twice")
    _REGISTRY[name] = cls
    return cls


def get_method(name: str) -> AttributionMethod:
    try:
        return _REGISTRY[name]()
    except KeyError as exc:
        raise ConfigError(
            f"unknown attribution method {name!r}; available: {sorted(_REGISTRY)}"
        ) from exc


def available_methods() -> list[str]:
    return sorted(_REGISTRY)
