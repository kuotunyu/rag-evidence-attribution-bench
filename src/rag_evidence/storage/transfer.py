"""Results round-trip: `export` (Colab → zip) and `import-results` (zip → repo).

import-results is the integrity gate that turns "some zip" into "results this repo will
report on": manifest present, per-file sha256 verified, no zip-slip paths, and existing
local runs are never silently overwritten with different content.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import zipfile
from pathlib import Path
from typing import Any

import rag_evidence
from rag_evidence.config import AppConfig
from rag_evidence.errors import ArtifactError
from rag_evidence.storage.artifacts import read_records

logger = logging.getLogger(__name__)

EXPORT_MANIFEST = "export_manifest.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_files(cfg: AppConfig) -> list[tuple[Path, str]]:
    """(file, canonical arcname) pairs. Arcnames are ALWAYS the repo-relative form
    configured in YAML (for example results/v2/raw/<split>/…) even when the live
    results dirs are absolute (Colab writes checkpoints to a Drive mount via
    RAG_EVIDENCE_RESULTS_* env overrides) — import always lands in the repo layout."""
    def canonical_root(configured: str, *, fallback: str) -> str:
        path = Path(configured)
        if not path.is_absolute():
            return path.as_posix()
        parts = list(path.parts)
        result_indices = [index for index, part in enumerate(parts) if part == "results"]
        if result_indices:
            return "/".join(parts[result_indices[-1] :])
        return fallback

    pairs: list[tuple[Path, str]] = []
    raw_arc_root = canonical_root(cfg.paths.results_raw, fallback="results/raw")
    derived_arc_root = canonical_root(
        cfg.paths.results_derived, fallback="results/derived"
    )
    raw_split = cfg.results_raw_dir / cfg.split
    if not raw_split.exists():
        raise ArtifactError(f"nothing to export: {raw_split} does not exist")
    for f in raw_split.rglob("*"):
        if f.is_file():
            rel = f.relative_to(raw_split).as_posix()
            pairs.append((f, f"{raw_arc_root}/{cfg.split}/{rel}"))
    if cfg.results_derived_dir.exists():
        for f in cfg.results_derived_dir.rglob("*"):
            if f.is_file():
                rel = f.relative_to(cfg.results_derived_dir).as_posix()
                pairs.append((f, f"{derived_arc_root}/{rel}"))
    if cfg.manifest_file.exists():
        pairs.append((cfg.manifest_file, Path(cfg.data.manifest_path).as_posix()))
    return pairs


def export_results(cfg: AppConfig, *, out: Path | None) -> None:
    pairs = _collect_files(cfg)
    # validate every JSONL parses before shipping (truncated tails are fine, mid-file
    # corruption is not — same rule as resume)
    for f, _arc in pairs:
        if f.suffix == ".jsonl":
            for _ in read_records(f):
                pass

    stamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out = out or Path("results/export") / f"results_{cfg.split}_{stamp}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "split": cfg.split,
        "run_name": cfg.run_name,
        "package_version": rag_evidence.__version__,
        "created_utc": stamp,
        "files": {},
    }
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f, arcname in sorted(pairs, key=lambda p: p[1]):
            manifest["files"][arcname] = _sha256_file(f)
            zf.write(f, arcname)
        zf.writestr(EXPORT_MANIFEST, json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info("exported %d files -> %s", len(pairs), out)


def import_results(cfg: AppConfig, *, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        if EXPORT_MANIFEST not in names:
            raise ArtifactError(
                f"{zip_path} has no {EXPORT_MANIFEST} — not an `export` bundle; refusing"
            )
        manifest = json.loads(zf.read(EXPORT_MANIFEST))
        declared: dict[str, str] = manifest["files"]

        missing = sorted(set(declared) - names)
        if missing:
            raise ArtifactError(f"zip is missing {len(missing)} declared files: {missing[:3]}")

        for arcname in declared:
            # zip-slip / absolute-path guard: only forward, repo-relative paths
            if arcname.startswith(("/", "\\")) or ".." in Path(arcname).parts or ":" in arcname:
                raise ArtifactError(f"refusing suspicious path in zip: {arcname!r}")
            if not arcname.startswith(("results/", "data/manifests/")):
                raise ArtifactError(f"refusing path outside results area: {arcname!r}")

        conflicts: list[str] = []
        for arcname, digest in declared.items():
            target = Path(arcname)
            if target.exists() and _sha256_file(target) != digest:
                conflicts.append(arcname)
        # summary.json / derived outputs are regenerable — conflicts there are expected;
        # RAW records conflicting is the dangerous case.
        raw_conflicts = [c for c in conflicts if "/raw/" in c or c.startswith("results/raw")]
        if raw_conflicts:
            raise ArtifactError(
                f"{len(raw_conflicts)} raw files already exist locally with DIFFERENT "
                f"content (e.g. {raw_conflicts[:2]}). Refusing to overwrite raw results — "
                "move the local run away first if the import should win."
            )

        n_written = 0
        for arcname, digest in declared.items():
            target = Path(arcname)
            target.parent.mkdir(parents=True, exist_ok=True)
            data = zf.read(arcname)
            if hashlib.sha256(data).hexdigest() != digest:
                raise ArtifactError(f"sha256 mismatch inside zip for {arcname} — corrupt bundle")
            target.write_bytes(data)
            n_written += 1

    logger.info(
        "imported %d files from %s (split=%s, exported %s by package %s)",
        n_written,
        zip_path,
        manifest.get("split"),
        manifest.get("created_utc"),
        manifest.get("package_version"),
    )
    logger.info("next: `python -m rag_evidence.cli evaluate --config <config>` then `report`")
