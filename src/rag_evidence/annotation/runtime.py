"""Fail-closed loopback hosting for annotation and adjudication consoles."""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rag_evidence.errors import DataError

Resolver = Callable[..., list[tuple[Any, ...]]]


def validate_loopback_host(
    host: str,
    *,
    resolver: Resolver = socket.getaddrinfo,
) -> str:
    """Accept only exact loopback literals or localhost resolving entirely to loopback."""
    if host in {"127.0.0.1", "::1"}:
        return host
    if host != "localhost":
        raise DataError("annotation runtime host must be loopback")
    try:
        rows = resolver(host, None)
        addresses = {
            ipaddress.ip_address(str(row[4][0]))
            for row in rows
            if len(row) >= 5 and isinstance(row[4], tuple) and row[4]
        }
    except (OSError, TypeError, ValueError) as exc:
        raise DataError("localhost must resolve only to loopback addresses") from exc
    if not addresses or not all(address.is_loopback for address in addresses):
        raise DataError("localhost must resolve only to loopback addresses")
    return host


def serve_annotation(
    package: Path,
    state: Path,
    *,
    host: str,
    port: int,
) -> None:
    """Validate the bind target before loading a package or touching state."""
    validated_host = validate_loopback_host(host)
    import uvicorn

    from rag_evidence.annotation.app import create_annotation_app

    application = create_annotation_app(package, state)
    uvicorn.run(application, host=validated_host, port=port)


def serve_adjudication(
    manifest: Path,
    effective: Path,
    state: Path,
    *,
    host: str,
    port: int,
) -> None:
    """Validate the bind target before loading coordinator inputs or touching state."""
    validated_host = validate_loopback_host(host)
    import uvicorn

    from rag_evidence.annotation.app import create_adjudication_app

    application = create_adjudication_app(manifest, effective, state)
    uvicorn.run(application, host=validated_host, port=port)
