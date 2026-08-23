"""Platform evidence is complete, sanitized, identity-bound, and byte-verified."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from rag_evidence.annotation.platform import (
    REQUIRED_COMMAND_KINDS,
    PlatformIdentityV2,
    PlatformVerificationReceiptV2,
    verify_platform_receipt,
)
from rag_evidence.errors import DataError

ZERO_HASH = hashlib.sha256(b"").hexdigest()
WHEEL_HASH = hashlib.sha256(b"wheel").hexdigest()
LOCK_HASH = hashlib.sha256(b"lock").hexdigest()
PROTOCOL_HASH = hashlib.sha256(b"protocol").hexdigest()
SPEC_HASH = hashlib.sha256(b"spec").hexdigest()
SCHEMA_HASH = hashlib.sha256(b"schema").hexdigest()


def _identity() -> PlatformIdentityV2:
    return PlatformIdentityV2(
        source_commit_sha="1" * 40,
        git_tree_sha="2" * 40,
        wheel_filename="rag_evidence_attribution_bench-0.2.0.dev0-py3-none-any.whl",
        wheel_byte_size=5,
        wheel_sha256=WHEEL_HASH,
        dependency_lock_sha256=LOCK_HASH,
        protocol_sha256=PROTOCOL_HASH,
        handoff_spec_sha256=SPEC_HASH,
        handoff_schema_sha256=SCHEMA_HASH,
    )


def _payload(platform: str = "Windows") -> dict[str, Any]:
    timestamp = "2026-08-24T00:00:00Z"
    commands = [
        {
            "kind": kind,
            "argv": ["python", "-m", "rag_evidence.cli", "--version"],
            "logical_cwd": "verification-root",
            "exit_code": 0,
            "stdout_sha256": ZERO_HASH,
            "stderr_sha256": ZERO_HASH,
            "started_at": timestamp,
            "ended_at": timestamp,
            "semantic_result": "passed",
        }
        for kind in REQUIRED_COMMAND_KINDS
    ]
    return {
        **_identity().model_dump(mode="json"),
        "schema_version": "platform-verification-receipt-v2",
        "verifier_version": "annotation-platform-verifier-v2",
        "verifier_sha256": "3" * 64,
        "platform": platform,
        "python_version": "3.11.9",
        "os_version": "fixture-os",
        "architecture": "AMD64",
        "pip_version": "24.0",
        "index_configuration": ["https://pypi.org/simple"],
        "bootstrap_argv": [
            "python",
            "-m",
            "pip",
            "install",
            "--require-hashes",
            "--only-binary=:all:",
            "-r",
            "annotation-requirements-py311.lock",
        ],
        "wheel_install_argv": [
            "python",
            "-m",
            "pip",
            "install",
            "--no-deps",
            "verified-wheel.whl",
        ],
        "installed_distributions": [
            {"name": "fastapi", "version": "1.0"},
            {"name": "rag-evidence-attribution-bench", "version": "0.2.0.dev0"},
        ],
        "commands": commands,
        "smoke_artifact_sha256": {"smoke/tasks.json": ZERO_HASH},
        "started_at": timestamp,
        "ended_at": timestamp,
        "overall_result": "passed",
    }


@pytest.mark.parametrize("missing", REQUIRED_COMMAND_KINDS)
def test_receipt_rejects_each_missing_command(missing: str) -> None:
    payload = _payload()
    payload["commands"] = [row for row in payload["commands"] if row["kind"] != missing]

    with pytest.raises(ValidationError, match="every required command"):
        PlatformVerificationReceiptV2.model_validate(payload)


def test_receipt_binds_exact_dev_distribution_and_binary_only_bootstrap() -> None:
    receipt = PlatformVerificationReceiptV2.model_validate(_payload())

    assert receipt.python_distribution == "0.2.0.dev0"
    assert receipt.wheel_sha256 == WHEEL_HASH
    assert "--require-hashes" in receipt.bootstrap_argv
    assert "--only-binary=:all:" in receipt.bootstrap_argv
    assert "--no-deps" in receipt.wheel_install_argv
    assert receipt.overall_result == "passed"


def test_receipt_rejects_secret_or_absolute_command_material() -> None:
    payload = _payload()
    payload["commands"][0]["argv"] = ["python", "token=do-not-record"]
    with pytest.raises(ValidationError, match="secret"):
        PlatformVerificationReceiptV2.model_validate(payload)

    payload = _payload()
    payload["commands"][0]["argv"] = [r"C:\Users\private\python.exe"]
    with pytest.raises(ValidationError, match="private absolute path"):
        PlatformVerificationReceiptV2.model_validate(payload)


def test_verify_platform_receipt_hashes_retained_smoke_bytes(tmp_path: Path) -> None:
    smoke = tmp_path / "smoke/tasks.json"
    smoke.parent.mkdir()
    smoke.write_bytes(b"")
    receipt_path = tmp_path / "platform-verification-receipt.json"
    receipt_path.write_text(json.dumps(_payload()), encoding="utf-8")

    receipt = verify_platform_receipt(receipt_path, expected=_identity())
    assert receipt.platform == "Windows"

    smoke.write_bytes(b"poison")
    with pytest.raises(DataError, match="hash mismatch"):
        verify_platform_receipt(receipt_path, expected=_identity())


def test_verify_platform_receipt_rejects_wrong_expected_wheel(tmp_path: Path) -> None:
    smoke = tmp_path / "smoke/tasks.json"
    smoke.parent.mkdir()
    smoke.write_bytes(b"")
    receipt_path = tmp_path / "platform-verification-receipt.json"
    receipt_path.write_text(json.dumps(_payload()), encoding="utf-8")
    expected_payload = _identity().model_dump(mode="python")
    expected_payload["wheel_sha256"] = hashlib.sha256(b"other").hexdigest()

    with pytest.raises(DataError, match="wheel_sha256"):
        verify_platform_receipt(
            receipt_path,
            expected=PlatformIdentityV2.model_validate(expected_payload),
        )


def test_receipt_times_are_real_utc_values() -> None:
    receipt = PlatformVerificationReceiptV2.model_validate(_payload())
    assert receipt.started_at == dt.datetime(2026, 8, 24, tzinfo=dt.UTC)
