"""The outer platform verifier is stdlib-first and preserves the exact bootstrap contract."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType


def _load(repo_root: Path) -> ModuleType:
    path = repo_root / "scripts/verify_annotation_platform.py"
    spec = importlib.util.spec_from_file_location("verify_annotation_platform", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_platform_verifier_help_is_available_without_project_import(repo_root: Path) -> None:
    path = repo_root / "scripts/verify_annotation_platform.py"
    completed = subprocess.run(
        (sys.executable, str(path), "--help"),
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--only-binary" in completed.stdout
    assert "--platform" in completed.stdout


def test_bootstrap_and_wheel_install_argv_are_not_weakened(repo_root: Path) -> None:
    module = _load(repo_root)

    bootstrap = module.bootstrap_argv("python", "annotation-requirements-py311.lock")
    wheel_install = module.wheel_install_argv("python", "verified.whl")

    assert bootstrap == (
        "python",
        "-m",
        "pip",
        "install",
        "--require-hashes",
        "--only-binary=:all:",
        "-r",
        "annotation-requirements-py311.lock",
    )
    assert wheel_install == (
        "python",
        "-m",
        "pip",
        "install",
        "--no-deps",
        "verified.whl",
    )


def test_index_configuration_sanitizer_removes_credentials(repo_root: Path) -> None:
    module = _load(repo_root)

    sanitized = module.sanitize_index_configuration(
        {
            "PIP_INDEX_URL": "https://user:secret@example.invalid/simple?token=bad",
            "PIP_EXTRA_INDEX_URL": "https://mirror.invalid/simple",
            "PIP_TRUSTED_HOST": "mirror.invalid",
        }
    )

    assert sanitized == (
        "extra-index-url=https://mirror.invalid/simple",
        "index-url=https://example.invalid/simple",
        "trusted-host=mirror.invalid",
    )
    joined = " ".join(sanitized)
    assert "user" not in joined
    assert "secret" not in joined
    assert "token" not in joined
