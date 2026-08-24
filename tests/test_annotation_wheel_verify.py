"""Wheel evidence is byte-identical, internally valid, and bound to Git blobs."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import shutil
import subprocess
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rag_evidence.annotation.source_verify import verify_checkout
from rag_evidence.annotation.wheel_verify import (
    BuildEnvironmentV2,
    WheelSourceSpecV2,
    build_four_wheels,
    replay_wheel_build,
    require_byte_identity,
    verify_wheel_payload,
)
from rag_evidence.errors import DataError

DIST_INFO = "rag_evidence_attribution_bench-0.2.0.dist-info"
WHEEL_NAME = "rag_evidence_attribution_bench-0.2.0-py3-none-any.whl"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _record_bytes(entries: dict[str, bytes]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for name, payload in sorted(entries.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
        writer.writerow((name, f"sha256={digest}", str(len(payload))))
    writer.writerow((f"{DIST_INFO}/RECORD", "", ""))
    return output.getvalue().encode()


def _write_wheel(path: Path, entries: dict[str, bytes]) -> None:
    complete = dict(entries)
    complete[f"{DIST_INFO}/RECORD"] = _record_bytes(entries)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in complete.items():
            info = zipfile.ZipInfo(name, date_time=(2000, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload)


@pytest.fixture
def source_bound_wheel(tmp_path: Path) -> tuple[Path, object, WheelSourceSpecV2]:
    checkout_root = tmp_path / "checkout"
    package_root = checkout_root / "src/rag_evidence"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text('__version__ = "0.2.0"\n', encoding="utf-8")
    (checkout_root / "pyproject.toml").write_text(
        '[project]\nname = "rag-evidence-attribution-bench"\nversion = "0.2.0"\n',
        encoding="utf-8",
    )
    _git(checkout_root, "init")
    _git(checkout_root, "config", "user.name", "Test Owner")
    _git(checkout_root, "config", "user.email", "owner@example.invalid")
    _git(checkout_root, "config", "core.autocrlf", "false")
    _git(checkout_root, "add", ".")
    _git(checkout_root, "commit", "-m", "wheel fixture")
    commit = _git(checkout_root, "rev-parse", "HEAD")
    _git(checkout_root, "switch", "--detach", commit)
    checkout = verify_checkout(
        checkout_root,
        commit,
        phase="pre-build",
        logical_checkout="fixture",
    )
    wheel = tmp_path / WHEEL_NAME
    _write_wheel(
        wheel,
        {
            "rag_evidence/__init__.py": (package_root / "__init__.py").read_bytes(),
            f"{DIST_INFO}/METADATA": (
                b"Metadata-Version: 2.4\nName: rag-evidence-attribution-bench\nVersion: 0.2.0\n\n"
            ),
            f"{DIST_INFO}/WHEEL": (
                b"Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\n"
                b"Tag: py3-none-any\n\n"
            ),
        },
    )
    return (
        wheel,
        checkout,
        WheelSourceSpecV2(
            distribution_name="rag-evidence-attribution-bench",
            distribution_version="0.2.0",
            package_source_root="src/rag_evidence",
        ),
    )


def _mutate(source: Path, destination: Path, mutation: str) -> None:
    with zipfile.ZipFile(source) as archive:
        entries = {info.filename: archive.read(info) for info in archive.infolist()}
    record_name = f"{DIST_INFO}/RECORD"
    entries.pop(record_name)
    if mutation == "metadata-version":
        entries[f"{DIST_INFO}/METADATA"] = entries[f"{DIST_INFO}/METADATA"].replace(
            b"0.2.0", b"0.1.0"
        )
    elif mutation == "unsafe-path":
        entries["../escape.py"] = b"escape\n"
    elif mutation == "missing-source":
        entries.pop("rag_evidence/__init__.py")
    elif mutation == "unexpected-package-source":
        entries["rag_evidence/extra.py"] = b"unexpected\n"
    elif mutation == "source-byte":
        entries["rag_evidence/__init__.py"] = b'__version__ = "changed"\n'
    _write_wheel(destination, entries)
    if mutation in {"record-hash", "record-size"}:
        with zipfile.ZipFile(destination) as archive:
            rewritten = {info.filename: archive.read(info) for info in archive.infolist()}
        record = rewritten[record_name].decode()
        if mutation == "record-hash":
            record = record.replace("sha256=", "sha256=" + "0" * 43, 1)
        else:
            rows = list(csv.reader(io.StringIO(record)))
            rows[0][2] = str(int(rows[0][2]) + 1)
            stream = io.StringIO(newline="")
            csv.writer(stream, lineterminator="\n").writerows(rows)
            record = stream.getvalue()
        rewritten[record_name] = record.encode()
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, payload in rewritten.items():
                archive.writestr(name, payload)
    elif mutation == "duplicate-entry":
        with zipfile.ZipFile(destination, "a") as archive:
            archive.writestr("rag_evidence/__init__.py", b"duplicate\n")


def test_valid_wheel_is_bound_to_exact_source_and_metadata(
    source_bound_wheel: tuple[Path, object, WheelSourceSpecV2],
) -> None:
    wheel, checkout, spec = source_bound_wheel

    identity = verify_wheel_payload(wheel, checkout, spec)

    assert identity.filename == WHEEL_NAME
    assert identity.distribution_version == "0.2.0"
    assert identity.package_files == ("rag_evidence/__init__.py",)


@pytest.mark.parametrize(
    "mutation",
    [
        "record-hash",
        "record-size",
        "metadata-version",
        "duplicate-entry",
        "unsafe-path",
        "missing-source",
        "unexpected-package-source",
        "source-byte",
    ],
)
def test_mutated_wheel_fails_source_binding(
    source_bound_wheel: tuple[Path, object, WheelSourceSpecV2],
    tmp_path: Path,
    mutation: str,
) -> None:
    wheel, checkout, spec = source_bound_wheel
    poisoned = tmp_path / f"poison-{mutation}.whl"
    _mutate(wheel, poisoned, mutation)

    with pytest.raises(DataError):
        verify_wheel_payload(poisoned, checkout, spec)


def test_four_wheels_must_be_byte_identical(
    source_bound_wheel: tuple[Path, object, WheelSourceSpecV2],
    tmp_path: Path,
) -> None:
    wheel, _checkout, _spec = source_bound_wheel
    copies = []
    for index in range(4):
        path = tmp_path / f"copy-{index}" / wheel.name
        path.parent.mkdir()
        shutil.copyfile(wheel, path)
        copies.append(path)
    assert require_byte_identity(copies).sha256 == hashlib.sha256(wheel.read_bytes()).hexdigest()
    copies[-1].write_bytes(copies[-1].read_bytes() + b"changed")

    with pytest.raises(DataError, match="byte-identical"):
        require_byte_identity(copies)


def _fake_uv_builder(
    monkeypatch: pytest.MonkeyPatch,
    checkout_root: Path,
    *,
    poison_after_build: bool = False,
) -> dict[str, Any]:
    import rag_evidence.annotation.wheel_verify as module

    real_run = subprocess.run
    observed: dict[str, Any] = {}

    def run(args: tuple[str, ...], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        if args[0] != "uv":
            return real_run(args, **kwargs)
        observed["argv"] = args
        observed["env"] = kwargs["env"]
        output = Path(args[-1])
        output.mkdir(parents=True, exist_ok=True)
        _write_wheel(
            output / WHEEL_NAME,
            {
                "rag_evidence/__init__.py": (
                    checkout_root / "src/rag_evidence/__init__.py"
                ).read_bytes(),
                f"{DIST_INFO}/METADATA": (
                    b"Metadata-Version: 2.4\nName: rag-evidence-attribution-bench\n"
                    b"Version: 0.2.0\n\n"
                ),
                f"{DIST_INFO}/WHEEL": (
                    b"Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\n"
                    b"Tag: py3-none-any\n\n"
                ),
            },
        )
        if poison_after_build:
            (checkout_root / "post-build-poison.txt").write_text("poison", encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, b"built", b"")

    monkeypatch.setattr(module.subprocess, "run", run)
    return observed


def _external_environment(tmp_path: Path, label: str = "one") -> BuildEnvironmentV2:
    return BuildEnvironmentV2(
        cache_root=tmp_path / f"{label}-cache",
        build_root=tmp_path / f"{label}-build",
        output_root=tmp_path / f"{label}-dist",
        pycache_root=tmp_path / f"{label}-pycache",
    )


def test_replay_build_records_external_reproducible_environment(
    source_bound_wheel: tuple[Path, object, WheelSourceSpecV2],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wheel, checkout, spec = source_bound_wheel
    observed = _fake_uv_builder(monkeypatch, checkout.root)

    record = replay_wheel_build(
        checkout.root,
        checkout.commit_sha,
        _external_environment(tmp_path / "external"),
        logical_checkout="A",
        build_label="supplied",
        spec=spec,
    )

    assert record.pre_build.phase == "pre-build"
    assert record.post_build.phase == "post-build"
    assert record.wheel.distribution_version == "0.2.0"
    assert observed["argv"][:4] == ("uv", "build", "--wheel", "--out-dir")
    assert observed["env"]["SOURCE_DATE_EPOCH"] == "1787443200"
    assert observed["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert Path(observed["env"]["UV_CACHE_DIR"]).is_relative_to(tmp_path)


def test_replay_build_rejects_post_build_checkout_pollution(
    source_bound_wheel: tuple[Path, object, WheelSourceSpecV2],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wheel, checkout, spec = source_bound_wheel
    _fake_uv_builder(monkeypatch, checkout.root, poison_after_build=True)

    with pytest.raises(DataError, match="clean"):
        replay_wheel_build(
            checkout.root,
            checkout.commit_sha,
            _external_environment(tmp_path / "external"),
            logical_checkout="A",
            build_label="replay",
            spec=spec,
        )


def test_replay_build_rejects_environment_inside_checkout(
    source_bound_wheel: tuple[Path, object, WheelSourceSpecV2],
) -> None:
    _wheel, checkout, spec = source_bound_wheel

    with pytest.raises(DataError, match="outside"):
        replay_wheel_build(
            checkout.root,
            checkout.commit_sha,
            _external_environment(checkout.root / "generated"),
            logical_checkout="A",
            build_label="replay",
            spec=spec,
        )


def test_four_build_orchestrator_writes_bound_verification(
    source_bound_wheel: tuple[Path, object, WheelSourceSpecV2],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wheel, checkout, _spec = source_bound_wheel
    second = tmp_path / "checkout-copy"
    shutil.copytree(checkout.root, second)
    _fake_uv_builder(monkeypatch, checkout.root)

    # The fake builder must read from whichever checkout subprocess receives.
    import rag_evidence.annotation.wheel_verify as module

    first_fake: Callable[..., subprocess.CompletedProcess[bytes]] = module.subprocess.run
    real_run = subprocess.run

    def checkout_aware_run(
        args: tuple[str, ...], **kwargs: Any
    ) -> subprocess.CompletedProcess[bytes]:
        if args[0] != "uv":
            return real_run(args, **kwargs)
        current = Path(kwargs["cwd"])
        output = Path(args[-1])
        output.mkdir(parents=True, exist_ok=True)
        _write_wheel(
            output / WHEEL_NAME,
            {
                "rag_evidence/__init__.py": (current / "src/rag_evidence/__init__.py").read_bytes(),
                f"{DIST_INFO}/METADATA": (
                    b"Metadata-Version: 2.4\nName: rag-evidence-attribution-bench\n"
                    b"Version: 0.2.0\n\n"
                ),
                f"{DIST_INFO}/WHEEL": (
                    b"Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\n"
                    b"Tag: py3-none-any\n\n"
                ),
            },
        )
        return subprocess.CompletedProcess(args, 0, b"built", b"")

    del first_fake
    monkeypatch.setattr(module.subprocess, "run", checkout_aware_run)
    result = build_four_wheels(
        checkout.root,
        second,
        checkout.commit_sha,
        tmp_path / "evidence",
    )

    assert len(result.builds) == 4
    assert len({record.wheel.sha256 for record in result.builds}) == 1
    assert (tmp_path / "evidence/wheel-verification.json").is_file()
