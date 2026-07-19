"""Export / import-results round-trip and its integrity guards."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from rag_evidence.config import AppConfig
from rag_evidence.errors import ArtifactError
from rag_evidence.generation.run import run_generation_stage
from rag_evidence.retrieval.run import run_retrieval_stage
from rag_evidence.storage.transfer import export_results, import_results


@pytest.fixture()
def exported_env(tiny_env: AppConfig) -> tuple[AppConfig, Path]:
    cfg = tiny_env
    run_retrieval_stage(cfg, method="bm25", resume=False, limit=None)
    run_generation_stage(cfg, resume=False, limit=None)
    out = Path("results/export/bundle.zip")
    export_results(cfg, out=out)
    return cfg, out


def test_export_creates_manifest_with_canonical_paths(exported_env: tuple[AppConfig, Path]) -> None:
    _cfg, out = exported_env
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert "export_manifest.json" in names
        payload_names = [n for n in names if n != "export_manifest.json"]
        assert all(
            n.startswith(("results/raw/smoke/", "results/derived/", "data/manifests/"))
            for n in payload_names
        )
        assert "results/raw/smoke/generate/fake/records.jsonl" in names


def test_roundtrip_into_fresh_tree(
    exported_env: tuple[AppConfig, Path],
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, out = exported_env
    bundle = out.resolve()
    fresh = tmp_path_factory.mktemp("fresh_repo")
    monkeypatch.chdir(fresh)
    import_results(cfg, zip_path=bundle)
    assert (fresh / "results/raw/smoke/generate/fake/records.jsonl").exists()
    assert (fresh / "data/manifests/split_manifest.json").exists()
    # re-import of identical content is a no-op success
    import_results(cfg, zip_path=bundle)


def test_import_refuses_conflicting_raw(
    exported_env: tuple[AppConfig, Path],
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, out = exported_env
    bundle = out.resolve()
    fresh = tmp_path_factory.mktemp("fresh_repo2")
    monkeypatch.chdir(fresh)
    conflict = fresh / "results/raw/smoke/generate/fake/records.jsonl"
    conflict.parent.mkdir(parents=True)
    conflict.write_text('{"question_id": "different-content"}\n', encoding="utf-8")
    with pytest.raises(ArtifactError, match="DIFFERENT"):
        import_results(cfg, zip_path=bundle)


def test_import_refuses_non_bundle_zip(tiny_env: AppConfig, tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.zip"
    with zipfile.ZipFile(bogus, "w") as zf:
        zf.writestr("results/raw/smoke/whatever.jsonl", "{}")
    with pytest.raises(ArtifactError, match="export_manifest"):
        import_results(tiny_env, zip_path=bogus)


def test_import_refuses_zip_slip(tiny_env: AppConfig, tmp_path: Path) -> None:
    import json as _json

    evil = tmp_path / "evil.zip"
    manifest = {"schema_version": 1, "files": {"results/raw/../../evil.txt": "0" * 64}}
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("results/raw/../../evil.txt", "pwned")
        zf.writestr("export_manifest.json", _json.dumps(manifest))
    with pytest.raises(ArtifactError, match=r"suspicious|missing"):
        import_results(tiny_env, zip_path=evil)
