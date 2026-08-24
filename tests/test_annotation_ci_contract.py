"""CI verifies the exact PR head on symmetric Windows and Linux annotation platforms."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

PRERELEASE_VERSION = "0.2.0" + ".dev0"


def _workflow(repo_root: Path) -> dict[str, Any]:
    payload: dict[str, Any] = yaml.safe_load(
        (repo_root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    )
    return payload


def test_ci_has_symmetric_annotation_platform_jobs(repo_root: Path) -> None:
    workflow = _workflow(repo_root)
    jobs = workflow["jobs"]
    windows = jobs["annotation-windows"]
    linux = jobs["annotation-linux"]

    assert windows["runs-on"] == "windows-latest"
    assert linux["runs-on"] == "ubuntu-latest"
    assert windows["env"]["VERIFICATION_PLATFORM"] == "Windows"
    assert linux["env"]["VERIFICATION_PLATFORM"] == "Linux"
    assert windows["env"]["SOURCE_DATE_EPOCH"] == linux["env"]["SOURCE_DATE_EPOCH"]
    assert windows["env"]["SOURCE_COMMIT"] == linux["env"]["SOURCE_COMMIT"]
    serialized = json.dumps((windows, linux))
    assert "verify_annotation_platform.py" in serialized
    assert "--only-binary=:all:" in serialized
    assert "annotation-requirements-py311.lock" in serialized
    assert "rag_evidence_attribution_bench-0.2.0-py3-none-any.whl" in serialized
    assert PRERELEASE_VERSION not in serialized
    assert "checkout@v4" in serialized
    assert "pull_request.head.sha" in serialized
    for job in (windows, linux):
        assert "runner.temp" not in json.dumps(job["env"])
        build_step = next(
            step for step in job["steps"] if step.get("name", "").startswith("Build exact")
        )
        assert "runner.temp" in json.dumps(build_step["env"])


def test_platform_jobs_upload_only_coordinator_verification_evidence(repo_root: Path) -> None:
    jobs = _workflow(repo_root)["jobs"]
    names: set[str] = set()
    for job_name in ("annotation-windows", "annotation-linux"):
        job = jobs[job_name]
        upload = [
            step
            for step in job["steps"]
            if step.get("uses", "").startswith("actions/upload-artifact@")
        ]
        assert len(upload) == 1
        names.add(upload[0]["with"]["name"])
        assert "platform-output" in upload[0]["with"]["path"]
        assert "handoff" not in upload[0]["with"]["path"].casefold()
    assert names == {"platform-verification-windows", "platform-verification-linux"}


def test_existing_quality_and_docker_jobs_remain(repo_root: Path) -> None:
    jobs = _workflow(repo_root)["jobs"]
    assert {"checks", "docker", "annotation-windows", "annotation-linux"} <= set(jobs)


def test_ci_runs_for_main_and_version_tags(repo_root: Path) -> None:
    workflow = _workflow(repo_root)
    triggers = workflow.get("on", workflow.get(True))

    assert triggers["push"]["branches"] == ["main"]
    assert triggers["push"]["tags"] == ["v*"]
