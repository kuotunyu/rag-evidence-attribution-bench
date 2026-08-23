"""Distribution, protocol, and schema identities remain explicit and distinct."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from typer.testing import CliRunner

import rag_evidence
from rag_evidence.cli import app


def test_distribution_identity_is_synchronized(repo_root: Path) -> None:
    metadata = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["version"] == "0.2.0.dev0"
    assert rag_evidence.__version__ == "0.2.0.dev0"
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "rag-evidence-attribution-bench 0.2.0.dev0"


def test_protocol_schema_and_distribution_are_distinct(repo_root: Path) -> None:
    protocol = (repo_root / "PILOT_PROTOCOL.md").read_text(encoding="utf-8")
    package = json.loads(
        (repo_root / "pilot/v0.2/packages/ann-pilot-a.json").read_text(encoding="utf-8")
    )

    assert "pilot-v0.2.2-draft" in protocol
    assert package["instruction_version"] == "pilot-v0.2.2-draft"
    assert package["schema_version"] == "assignment-package-v2"
    assert rag_evidence.__version__ == "0.2.0.dev0"
