"""Annotation servers fail closed before state creation on non-loopback hosts."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from rag_evidence.annotation.runtime import (
    serve_adjudication,
    serve_annotation,
    validate_loopback_host,
)
from rag_evidence.errors import DataError

AddressRows = list[tuple[object, object, object, object, tuple[str, int]]]


@pytest.mark.parametrize("host", ["127.0.0.1", "::1"])
def test_exact_loopback_literals_are_accepted(host: str) -> None:
    assert validate_loopback_host(host) == host


def test_localhost_requires_every_resolved_address_to_be_loopback() -> None:
    def loopback_only(*_args: object) -> AddressRows:
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 0)),
        ]

    def mixed(*_args: object) -> AddressRows:
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("203.0.113.5", 0)),
        ]

    assert validate_loopback_host("localhost", resolver=loopback_only) == "localhost"
    with pytest.raises(DataError, match="loopback"):
        validate_loopback_host("localhost", resolver=mixed)


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "::", "127.0.0.2", "192.168.1.5", "8.8.8.8", "example.com"],
)
def test_non_loopback_annotation_host_fails_before_state_creation(
    tmp_path: Path,
    host: str,
) -> None:
    state = tmp_path / "state"

    with pytest.raises(DataError, match="loopback"):
        serve_annotation(tmp_path / "missing-package.json", state, host=host, port=8001)

    assert not state.exists()


def test_non_loopback_adjudication_host_fails_before_state_creation(tmp_path: Path) -> None:
    state = tmp_path / "state"

    with pytest.raises(DataError, match="loopback"):
        serve_adjudication(
            tmp_path / "missing-manifest.json",
            tmp_path / "missing-effective.jsonl",
            state,
            host="0.0.0.0",
            port=8002,
        )

    assert not state.exists()


def test_committed_launchers_fix_host_without_a_host_override(repo_root: Path) -> None:
    launcher_root = repo_root / "pilot/v0.2/launchers"

    for path in launcher_root.iterdir():
        text = path.read_text(encoding="utf-8")
        assert "--host 127.0.0.1" in text
        assert "[string]$Host" not in text
        assert "HOST=${" not in text
