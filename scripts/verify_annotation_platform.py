#!/usr/bin/env python3
"""Stdlib-first Windows/Linux verifier for the breaking-v2 annotation handoff.

Bootstrap always uses ``--require-hashes --only-binary=:all:``. The project wheel is
installed separately with ``--no-deps``. A passed receipt is emitted by the installed wheel only
after all runtime and loopback probes succeed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform as platform_module
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

SOURCE_DATE_EPOCH = "1787443200"
WHEEL_NAME = "rag_evidence_attribution_bench-0.2.0.dev0-py3-none-any.whl"
PROTOCOL_VERSION = "pilot-v0.2.2-draft"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def bootstrap_argv(python: str, lock: str) -> tuple[str, ...]:
    return (
        python,
        "-m",
        "pip",
        "install",
        "--require-hashes",
        "--only-binary=:all:",
        "-r",
        lock,
    )


def wheel_install_argv(python: str, wheel: str) -> tuple[str, ...]:
    return (python, "-m", "pip", "install", "--no-deps", wheel)


def _sanitize_url(value: str) -> str | None:
    try:
        parsed = urllib.parse.urlsplit(value.strip().strip("'\""))
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    port = f":{parsed.port}" if parsed.port is not None else ""
    path = parsed.path.rstrip("/") or "/"
    return urllib.parse.urlunsplit((parsed.scheme, f"{parsed.hostname}{port}", path, "", ""))


def sanitize_index_configuration(environment: dict[str, str]) -> tuple[str, ...]:
    """Return only normalized, non-secret index locations and trusted-host policy."""
    records: list[str] = []
    mapping = (
        ("PIP_INDEX_URL", "index-url"),
        ("PIP_EXTRA_INDEX_URL", "extra-index-url"),
    )
    for variable, label in mapping:
        for raw in environment.get(variable, "").split():
            sanitized = _sanitize_url(raw)
            if sanitized is not None:
                records.append(f"{label}={sanitized}")
    trusted = environment.get("PIP_TRUSTED_HOST", "")
    for host in trusted.split():
        if all(character.isalnum() or character in ".-_:" for character in host):
            records.append(f"trusted-host={host.casefold()}")
    if not records:
        records.append("index-url=https://pypi.org/simple")
    return tuple(sorted(set(records)))


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(
    records: list[dict[str, Any]],
    *,
    kind: str,
    actual_argv: tuple[str, ...],
    logical_argv: tuple[str, ...],
    cwd: Path,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[bytes]:
    started = _utc_now()
    try:
        completed = subprocess.run(
            actual_argv,
            cwd=cwd,
            env=environment,
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise RuntimeError(f"platform verification command could not start: {kind}") from exc
    ended = _utc_now()
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"platform verification command failed: {kind}: {stderr}")
    records.append(
        {
            "kind": kind,
            "argv": logical_argv,
            "logical_cwd": "verification-root",
            "exit_code": completed.returncode,
            "stdout_sha256": _sha256_bytes(completed.stdout),
            "stderr_sha256": _sha256_bytes(completed.stderr),
            "started_at": started,
            "ended_at": ended,
            "semantic_result": "passed",
        }
    )
    return completed


def _semantic_record(
    records: list[dict[str, Any]],
    *,
    kind: str,
    argv: tuple[str, ...],
    stdout: bytes = b"",
    started: str | None = None,
) -> None:
    timestamp = started or _utc_now()
    records.append(
        {
            "kind": kind,
            "argv": argv,
            "logical_cwd": "verification-root",
            "exit_code": 0,
            "stdout_sha256": _sha256_bytes(stdout),
            "stderr_sha256": EMPTY_SHA256,
            "started_at": timestamp,
            "ended_at": _utc_now(),
            "semantic_result": "passed",
        }
    )


def _require_git_identity(checkout: Path, commit: str, tree: str) -> None:
    if len(commit) != 40 or len(tree) != 40:
        raise RuntimeError("source commit and tree must be full Git object IDs")
    commands = (
        (("git", "rev-parse", "HEAD"), commit),
        (("git", "rev-parse", f"{commit}^{{tree}}"), tree),
    )
    for argv, expected in commands:
        result = subprocess.run(argv, cwd=checkout, check=True, capture_output=True, text=True)
        if result.stdout.strip() != expected:
            raise RuntimeError("platform verifier source Git identity mismatch")
    for clean_argv in (
        ("git", "status", "--porcelain=v1", "--untracked-files=all", "-z"),
        ("git", "ls-files", "--others", "--ignored", "--exclude-standard", "-z"),
    ):
        clean_result = subprocess.run(
            clean_argv,
            cwd=checkout,
            check=True,
            capture_output=True,
        )
        if clean_result.stdout:
            raise RuntimeError("platform verifier requires a clean checkout with no ignored paths")


def _venv_python(venv: Path) -> Path:
    relative = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return venv / relative


def _environment(work_root: Path, venv: Path) -> dict[str, str]:
    environment = os.environ.copy()
    bin_dir = _venv_python(venv).parent
    environment.update(
        {
            "PATH": str(bin_dir) + os.pathsep + environment.get("PATH", ""),
            "PIP_CACHE_DIR": str(work_root / "pip-cache"),
            "PYTHONPYCACHEPREFIX": str(work_root / "pycache"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "SOURCE_DATE_EPOCH": SOURCE_DATE_EPOCH,
        }
    )
    environment.pop("PYTHONPATH", None)
    return environment


def _http_get(url: str, *, timeout: float = 2.0) -> bytes:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"unexpected HTTP status: {response.status}")
        return bytes(response.read())


def _wait_ready(base_url: str, process: subprocess.Popen[bytes]) -> bytes:
    deadline = time.monotonic() + 45
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("annotation launcher exited before becoming ready")
        try:
            return _http_get(f"{base_url}/api/progress")
        except (OSError, urllib.error.URLError, RuntimeError) as exc:
            last_error = exc
            time.sleep(0.2)
    raise RuntimeError("annotation launcher readiness timed out") from last_error


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _launcher_argv(platform_name: str, launcher: Path, state: Path, port: int) -> tuple[str, ...]:
    if platform_name == "Windows":
        shell = shutil.which("pwsh") or shutil.which("powershell")
        if shell is None:
            raise RuntimeError("PowerShell is required for Windows launcher verification")
        return (
            shell,
            "-NoProfile",
            "-File",
            str(launcher),
            "-StateRoot",
            str(state),
            "-Port",
            str(port),
        )
    return ("sh", str(launcher), str(state), str(port))


def _probe_runtime(
    records: list[dict[str, Any]],
    *,
    platform_name: str,
    launcher: Path,
    state: Path,
    work_root: Path,
    output_root: Path,
    environment: dict[str, str],
) -> None:
    port = _free_port()
    argv = _launcher_argv(platform_name, launcher, state, port)
    stdout_path = work_root / "launcher.stdout.log"
    stderr_path = work_root / "launcher.stderr.log"
    started = _utc_now()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process_options: dict[str, Any] = {}
        if os.name == "nt":
            process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            process_options["start_new_session"] = True
        process = subprocess.Popen(
            argv,
            cwd=launcher.parent,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            **process_options,
        )
        try:
            base_url = f"http://127.0.0.1:{port}"
            initial_progress = _wait_ready(base_url, process)
            _semantic_record(
                records,
                kind="launcher-runtime",
                argv=(launcher.name, "<external-state>", "<ephemeral-port>"),
                started=started,
            )
            probes = (
                ("tasks-probe", "/api/tasks", "smoke/tasks.json"),
                ("progress-probe", "/api/progress", "smoke/progress.json"),
                ("empty-export", "/api/export/submissions.jsonl", "smoke/submissions.jsonl"),
            )
            for kind, endpoint, relative in probes:
                payload = (
                    initial_progress
                    if kind == "progress-probe"
                    else _http_get(base_url + endpoint)
                )
                target = output_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
                _semantic_record(
                    records,
                    kind=kind,
                    argv=("http-get", endpoint),
                    stdout=payload,
                )
            amendments = _http_get(base_url + "/api/export/amendments.jsonl")
            (output_root / "smoke/amendments.jsonl").write_bytes(amendments)
            tasks = json.loads((output_root / "smoke/tasks.json").read_text(encoding="utf-8"))
            progress = json.loads((output_root / "smoke/progress.json").read_text(encoding="utf-8"))
            if (
                tasks.get("schema_version") != "assignment-package-v2"
                or len(tasks.get("tasks", [])) != 40
            ):
                raise RuntimeError("launcher tasks probe did not return one 40-task v2 package")
            if progress != {"total": 40, "submitted": 0, "remaining": 40}:
                raise RuntimeError("launcher progress probe is not an empty 40-task state")
            if (output_root / "smoke/submissions.jsonl").read_bytes() or amendments:
                raise RuntimeError("platform smoke unexpectedly created a human decision artifact")
        finally:
            if process.poll() is None and os.name == "nt":
                subprocess.run(
                    ("taskkill", "/PID", str(process.pid), "/T", "/F"),
                    check=False,
                    capture_output=True,
                )
            elif process.poll() is None:
                kill_process_group = getattr(os, "killpg", None)
                if kill_process_group is None:
                    raise RuntimeError("POSIX process-group termination is unavailable")
                kill_process_group(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def _loopback_probes(
    records: list[dict[str, Any]],
    python: Path,
    work_root: Path,
    environment: dict[str, str],
) -> None:
    acceptance = {
        "loopback-ipv4": "127.0.0.1",
        "loopback-ipv6": "::1",
        "loopback-localhost": "localhost",
    }
    for kind, host in acceptance.items():
        code = (
            "from rag_evidence.annotation.runtime import validate_loopback_host; "
            f"assert validate_loopback_host({host!r}) == {host!r}"
        )
        _run(
            records,
            kind=kind,
            actual_argv=(str(python), "-c", code),
            logical_argv=("python", "-c", f"accept-loopback:{host}"),
            cwd=work_root,
            environment=environment,
        )
    rejections = {
        "reject-unspecified-ipv4": "0.0.0.0",
        "reject-unspecified-ipv6": "::",
        "reject-lan": "192.168.1.10",
        "reject-public": "8.8.8.8",
        "reject-hostname": "example.invalid",
    }
    for kind, host in rejections.items():
        code = (
            "from rag_evidence.annotation.runtime import validate_loopback_host; "
            "from rag_evidence.errors import DataError; "
            f"host={host!r}; rejected=False; "
            "\ntry: validate_loopback_host(host)\n"
            "except DataError: rejected=True\n"
            "assert rejected"
        )
        _run(
            records,
            kind=kind,
            actual_argv=(str(python), "-c", code),
            logical_argv=("python", "-c", f"reject-non-loopback:{host}"),
            cwd=work_root,
            environment=environment,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify annotation runtime with --require-hashes --only-binary=:all: and --no-deps"
        )
    )
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--handoff-spec", type=Path, required=True)
    parser.add_argument("--handoff-schema", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--platform", choices=("Windows", "Linux"), required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    checkout = args.checkout.resolve()
    work_root = args.work_root.resolve()
    output_root = args.output.resolve()
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("platform verifier requires Python 3.11")
    actual_platform = platform_module.system()
    if actual_platform != args.platform:
        raise RuntimeError(f"requested platform {args.platform} does not match {actual_platform}")
    if any(root.is_relative_to(checkout) for root in (work_root, output_root)):
        raise RuntimeError("platform work and evidence roots must remain outside the checkout")
    if (
        work_root == output_root
        or work_root.is_relative_to(output_root)
        or output_root.is_relative_to(work_root)
    ):
        raise RuntimeError("platform work and evidence roots must be distinct")
    for root in (work_root, output_root):
        if root.exists() and any(root.iterdir()):
            raise RuntimeError("platform work and evidence roots must be absent or empty")
        root.mkdir(parents=True, exist_ok=True)
    _require_git_identity(checkout, args.source_commit, args.source_tree)

    inputs = (
        args.wheel,
        args.package,
        args.lock,
        args.launcher,
        args.handoff_spec,
        args.handoff_schema,
        args.protocol,
    )
    if any(not path.resolve().is_file() for path in inputs):
        raise RuntimeError("platform verifier input is missing")
    if args.wheel.name != WHEEL_NAME:
        raise RuntimeError("platform verifier requires the exact 0.2.0.dev0 wheel filename")
    protocol_text = args.protocol.read_text(encoding="utf-8")
    if f"`{PROTOCOL_VERSION}`" not in protocol_text:
        raise RuntimeError("protocol source does not declare pilot-v0.2.2-draft")

    started = _utc_now()
    records: list[dict[str, Any]] = []
    venv = work_root / "venv"
    subprocess.run((sys.executable, "-m", "venv", str(venv)), check=True, cwd=work_root)
    python = _venv_python(venv)
    environment = _environment(work_root, venv)
    kit = work_root / "verification-kit"
    kit.mkdir()
    staged = {
        "wheel": kit / WHEEL_NAME,
        "package": kit / args.package.name,
        "lock": kit / "annotation-requirements-py311.lock",
        "launcher": kit / ("start.ps1" if args.platform == "Windows" else "start.sh"),
    }
    for source, destination in (
        (args.wheel, staged["wheel"]),
        (args.package, staged["package"]),
        (args.lock, staged["lock"]),
        (args.launcher, staged["launcher"]),
    ):
        shutil.copyfile(source, destination)

    pip_version_result = _run(
        records,
        kind="pip-version",
        actual_argv=(str(python), "-m", "pip", "--version"),
        logical_argv=("python", "-m", "pip", "--version"),
        cwd=work_root,
        environment=environment,
    )
    pip_version = pip_version_result.stdout.decode("utf-8").split()[1]
    _run(
        records,
        kind="index-configuration",
        actual_argv=(str(python), "-m", "pip", "config", "list"),
        logical_argv=("python", "-m", "pip", "config", "list"),
        cwd=work_root,
        environment=environment,
    )
    bootstrap_actual = bootstrap_argv(str(python), str(staged["lock"]))
    bootstrap_logical = bootstrap_argv("python", "annotation-requirements-py311.lock")
    _run(
        records,
        kind="dependency-bootstrap",
        actual_argv=bootstrap_actual,
        logical_argv=bootstrap_logical,
        cwd=kit,
        environment=environment,
    )
    install_actual = wheel_install_argv(str(python), str(staged["wheel"]))
    install_logical = wheel_install_argv("python", WHEEL_NAME)
    _run(
        records,
        kind="wheel-install",
        actual_argv=install_actual,
        logical_argv=install_logical,
        cwd=kit,
        environment=environment,
    )
    distributions_code = (
        "import importlib.metadata as m,json; "
        "print(json.dumps(sorted(([d.metadata['Name'],d.version] for d in m.distributions()),"
        "key=lambda x:x[0].casefold())))"
    )
    distributions_result = _run(
        records,
        kind="installed-distributions",
        actual_argv=(str(python), "-c", distributions_code),
        logical_argv=("python", "-c", "list-installed-distributions"),
        cwd=work_root,
        environment=environment,
    )
    installed = [
        {"name": name, "version": version}
        for name, version in json.loads(distributions_result.stdout.decode("utf-8"))
    ]
    _run(
        records,
        kind="cli-version",
        actual_argv=(str(python), "-m", "rag_evidence.cli", "--version"),
        logical_argv=("python", "-m", "rag_evidence.cli", "--version"),
        cwd=work_root,
        environment=environment,
    )
    _run(
        records,
        kind="cli-help",
        actual_argv=(str(python), "-m", "rag_evidence.cli", "annotation", "--help"),
        logical_argv=("python", "-m", "rag_evidence.cli", "annotation", "--help"),
        cwd=work_root,
        environment=environment,
    )
    package_literal = str(staged["package"])
    app_state_literal = str(work_root / "app-probe-state")
    app_code = (
        "from pathlib import Path; from rag_evidence.annotation.app import create_annotation_app; "
        f"app=create_annotation_app(Path({package_literal!r}),Path({app_state_literal!r})); "
        "assert app.version == 'annotation-ui-v2'"
    )
    _run(
        records,
        kind="app-creation",
        actual_argv=(str(python), "-c", app_code),
        logical_argv=("python", "-c", "create-annotation-app-without-state-write"),
        cwd=work_root,
        environment=environment,
    )
    if (work_root / "app-probe-state").exists():
        raise RuntimeError("app creation probe wrote annotation state")
    _loopback_probes(records, python, work_root, environment)
    _probe_runtime(
        records,
        platform_name=args.platform,
        launcher=staged["launcher"],
        state=work_root / "launcher-state",
        work_root=work_root,
        output_root=output_root,
        environment=environment,
    )

    smoke_hashes = {
        path.relative_to(output_root).as_posix(): _sha256_file(path)
        for path in sorted((output_root / "smoke").rglob("*"))
        if path.is_file()
    }
    payload = {
        "schema_version": "platform-verification-receipt-v2",
        "verifier_version": "annotation-platform-verifier-v2",
        "verifier_sha256": _sha256_file(Path(__file__)),
        "platform": args.platform,
        "source_commit_sha": args.source_commit,
        "git_tree_sha": args.source_tree,
        "python_distribution": "0.2.0.dev0",
        "wheel_filename": args.wheel.name,
        "wheel_byte_size": args.wheel.stat().st_size,
        "wheel_sha256": _sha256_file(args.wheel),
        "dependency_lock_sha256": _sha256_file(args.lock),
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": _sha256_file(args.protocol),
        "handoff_spec_sha256": _sha256_file(args.handoff_spec),
        "handoff_schema_sha256": _sha256_file(args.handoff_schema),
        "python_version": platform_module.python_version(),
        "os_version": platform_module.platform(),
        "architecture": platform_module.machine(),
        "pip_version": pip_version,
        "index_configuration": sanitize_index_configuration(environment),
        "bootstrap_argv": bootstrap_logical,
        "wheel_install_argv": install_logical,
        "installed_distributions": installed,
        "commands": records,
        "smoke_artifact_sha256": smoke_hashes,
        "started_at": started,
        "ended_at": _utc_now(),
        "overall_result": "passed",
    }
    candidate = work_root / "platform-receipt-candidate.json"
    candidate.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    receipt_path = output_root / "platform-verification-receipt.json"
    emit_argv = (
        str(python),
        "-m",
        "rag_evidence.annotation.platform",
        "--payload",
        str(candidate),
        "--output",
        str(receipt_path),
        "--evidence-root",
        str(output_root),
    )
    completed = subprocess.run(emit_argv, cwd=work_root, env=environment, check=False)
    if completed.returncode != 0 or not receipt_path.is_file():
        raise RuntimeError("installed project did not validate and emit the platform receipt")
    print(str(receipt_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
