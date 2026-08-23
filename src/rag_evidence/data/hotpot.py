"""HotpotQA distractor: download, normalize, build manifest + prepared JSONL.

HF schema gotcha handled here: `context` and `supporting_facts` are column-oriented
parallel arrays ({title: [...], sentences: [[...]]} / {title: [...], sent_id: [...]}),
not lists of records — we normalize to row-oriented structures immediately.
"""

from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path
from typing import Any

from rag_evidence.config import AppConfig
from rag_evidence.data import ids
from rag_evidence.data.manifest_v2 import build_manifest_v2, validate_manifest_v2
from rag_evidence.data.schema import Example, Passage
from rag_evidence.data.splits import (
    build_manifest,
    check_manifest_matches_config,
    load_manifest,
    save_manifest,
    verify_examples,
)
from rag_evidence.errors import DataError
from rag_evidence.storage.artifacts import append_record

logger = logging.getLogger(__name__)


def verify_resolved_dataset_revision(repo_id: str, requested_revision: str) -> str:
    """Resolve a pinned Hub revision and reject any server-side mismatch."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise DataError("the `huggingface-hub` package is required for manifest schema v2") from exc

    try:
        info = HfApi().dataset_info(repo_id=repo_id, revision=requested_revision)
    except Exception as exc:
        raise DataError(f"failed to resolve dataset revision for {repo_id}: {exc}") from exc

    resolved_revision = str(info.sha)
    if resolved_revision != requested_revision:
        raise DataError(
            f"dataset revision mismatch for {repo_id}: "
            f"requested {requested_revision}, resolved {resolved_revision}"
        )
    return resolved_revision


def normalize_hf_example(row: dict[str, Any]) -> dict[str, Any]:
    """HF parallel-array row → canonical raw dict (row-oriented)."""
    ctx = row["context"]
    sf = row["supporting_facts"]
    return {
        "question_id": row["id"],
        "question": row["question"],
        "answer": row["answer"],
        "type": row["type"],
        "level": row["level"],
        "context": [
            [title, list(sents)]
            for title, sents in zip(ctx["title"], ctx["sentences"], strict=True)
        ],
        "supporting_facts": [
            [title, int(sid)] for title, sid in zip(sf["title"], sf["sent_id"], strict=True)
        ],
    }


def load_raw_examples(cfg: AppConfig) -> list[dict[str, Any]]:
    """Download/load the configured HF split and normalize every example."""
    try:
        from datasets import load_dataset  # lazy heavy import
    except ImportError as exc:
        raise DataError(
            "the `datasets` package is required for data prepare — "
            "install with `uv sync --extra ml`"
        ) from exc
    logger.info(
        "loading %s config=%s split=%s (first run downloads ~600 MB)",
        cfg.data.hf_path,
        cfg.data.hf_config,
        cfg.data.hf_split,
    )
    try:
        dset = load_dataset(
            cfg.data.hf_path,
            cfg.data.hf_config,
            split=cfg.data.hf_split,
            revision=cfg.data.hf_revision,
        )
    except Exception as exc:
        raise DataError(f"failed to load HotpotQA from the HF Hub: {exc}") from exc
    logger.info("loaded %d examples; normalizing", len(dset))
    return [normalize_hf_example(row) for row in dset]


def resolve_supporting_facts(
    raw: dict[str, Any],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """(title, sent_id) pairs → (gold_passage_ids, sentence_ids, dropped) via stable IDs.

    Known dataset noise: a sent_id can exceed the paragraph's sentence count, and
    (rarely) a title may not appear in the context. Both are dropped with a warning and
    recorded; the gold PASSAGE set is unaffected by out-of-range sentence indices as long
    as the title resolves.
    """
    qid = raw["question_id"]
    title_to_index = {title: i for i, (title, _sents) in enumerate(raw["context"])}
    if len(title_to_index) != len(raw["context"]):
        raise DataError(f"duplicate context titles in question {qid}")

    gold_pids: list[str] = []
    sentence_ids: list[str] = []
    dropped: list[dict[str, Any]] = []
    for title, sent_idx in raw["supporting_facts"]:
        p_index = title_to_index.get(title)
        if p_index is None:
            dropped.append({"title": title, "sent_id": sent_idx, "reason": "title_not_in_context"})
            logger.warning("q=%s: supporting-fact title %r not in context — dropped", qid, title)
            continue
        pid = ids.passage_id(qid, p_index)
        if pid not in gold_pids:
            gold_pids.append(pid)
        n_sents = len(raw["context"][p_index][1])
        if not 0 <= sent_idx < n_sents:
            dropped.append({"title": title, "sent_id": sent_idx, "reason": "sent_id_out_of_range"})
            logger.warning(
                "q=%s: supporting-fact sent_id %d out of range for %r (%d sentences) — "
                "sentence dropped, passage kept as gold",
                qid,
                sent_idx,
                title,
                n_sents,
            )
            continue
        sid = ids.sentence_id(pid, sent_idx)
        if sid not in sentence_ids:
            sentence_ids.append(sid)
    return gold_pids, sentence_ids, dropped


def build_example(raw: dict[str, Any]) -> Example:
    qid = raw["question_id"]
    gold_pids, sentence_ids, dropped = resolve_supporting_facts(raw)
    passages = tuple(
        Passage(
            passage_id=ids.passage_id(qid, i),
            index=i,
            title=title,
            sentences=tuple(sents),
            is_gold=ids.passage_id(qid, i) in gold_pids,
        )
        for i, (title, sents) in enumerate(raw["context"])
    )
    return Example(
        question_id=qid,
        question=raw["question"],
        answer=raw["answer"],
        level=raw["level"],
        qtype=raw["type"],
        passages=passages,
        gold_passage_ids=tuple(gold_pids),
        supporting_fact_sentence_ids=tuple(sentence_ids),
        dropped_supporting_facts=tuple(dropped),
    )


def load_prepared(path: Path) -> list[Example]:
    from rag_evidence.storage.artifacts import read_records

    if not path.exists():
        raise DataError(f"prepared file {path} not found — run `data prepare` first")
    return [Example.from_json(rec) for rec in read_records(path)]


def load_prepared_verified(cfg: AppConfig) -> list[Example]:
    """Load the config's prepared split and verify it against the committed manifest.

    Every stage runner calls this at startup: qid set must equal the manifest split and
    each record's stored raw_fingerprint must match the manifest — a stale or tampered
    prepared file fails loudly before any compute is spent.
    """
    from rag_evidence.errors import FingerprintMismatchError
    from rag_evidence.storage.artifacts import read_records

    manifest = load_manifest(cfg.manifest_file)
    check_manifest_matches_config(
        manifest, seed=cfg.data.split_seed, sizes=dict(cfg.data.split_sizes)
    )
    path = cfg.prepared_file
    if not path.exists():
        raise DataError(f"prepared file {path} not found — run `data prepare` first")
    records = list(read_records(path))
    expected = manifest["splits"][cfg.split]["question_ids"]
    got = [r["question_id"] for r in records]
    if sorted(got) != sorted(expected):
        raise DataError(
            f"{path} does not match manifest split {cfg.split!r} "
            f"({len(got)} records vs {len(expected)} expected qids) — re-run `data prepare`"
        )
    hashes = manifest["example_hashes"]
    for rec in records:
        qid = rec["question_id"]
        if rec.get("raw_fingerprint") != hashes[qid]:
            raise FingerprintMismatchError(qid, hashes[qid], str(rec.get("raw_fingerprint")))
    return [Example.from_json(rec) for rec in records]


def load_execution_examples(cfg: AppConfig) -> list[Example]:
    """Load natural examples or fail-closed human-eligible challenge examples."""
    if cfg.execution.track == "challenge":
        from rag_evidence.data.challenge_execution import load_eligible_challenge_examples

        return list(load_eligible_challenge_examples(cfg).examples)
    return load_prepared_verified(cfg)


def load_execution_metadata(cfg: AppConfig) -> dict[str, dict[str, str]]:
    if cfg.execution.track == "challenge":
        from rag_evidence.data.challenge_execution import load_eligible_challenge_examples

        return load_eligible_challenge_examples(cfg).metadata
    return {}


def prepare_data(cfg: AppConfig) -> None:
    """The `data prepare` stage. Idempotent; never regenerates an existing manifest."""
    resolved_revision = cfg.data.hf_revision
    if cfg.data.manifest_schema_version == 2:
        assert cfg.data.hf_revision is not None  # enforced by DataConfig validation
        resolved_revision = verify_resolved_dataset_revision(cfg.data.hf_path, cfg.data.hf_revision)
    raw_examples = load_raw_examples(cfg)
    raw_by_qid = {r["question_id"]: r for r in raw_examples}

    manifest_path = cfg.manifest_file
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        check_manifest_matches_config(
            manifest, seed=cfg.data.split_seed, sizes=dict(cfg.data.split_sizes)
        )
        if cfg.data.manifest_schema_version != manifest["schema_version"]:
            raise DataError(
                f"config requests manifest schema {cfg.data.manifest_schema_version}, "
                f"but {manifest_path} is schema {manifest['schema_version']}"
            )
        if manifest["schema_version"] == 2:
            if manifest["dataset"]["requested_revision"] != cfg.data.hf_revision:
                raise DataError("manifest v2 requested revision does not match config")
            if manifest["dataset"]["resolved_revision"] != resolved_revision:
                raise DataError("manifest v2 resolved revision does not match current source")
            validate_manifest_v2(manifest, raw_examples)
        logger.info("using existing committed manifest %s", manifest_path)
    else:
        if cfg.data.manifest_schema_version == 2:
            assert cfg.data.hf_revision is not None
            assert resolved_revision is not None
            manifest = build_manifest_v2(
                raw_examples,
                seed=cfg.data.split_seed,
                requested_sizes=dict(cfg.data.split_sizes),
                dataset_info={
                    "hf_path": cfg.data.hf_path,
                    "hf_config": cfg.data.hf_config,
                    "hf_split": cfg.data.hf_split,
                    "requested_revision": cfg.data.hf_revision,
                    "resolved_revision": resolved_revision,
                },
            )
            validate_manifest_v2(manifest, raw_examples)
        else:
            manifest = build_manifest(
                raw_examples,
                seed=cfg.data.split_seed,
                sizes=dict(cfg.data.split_sizes),
                dataset_info={
                    "hf_path": cfg.data.hf_path,
                    "hf_config": cfg.data.hf_config,
                    "hf_split": cfg.data.hf_split,
                    "hf_revision": cfg.data.hf_revision,
                },
                created_utc=_dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
        save_manifest(manifest_path, manifest)
        logger.info("wrote new split manifest %s — COMMIT THIS FILE", manifest_path)

    prepared_dir = Path(cfg.data.prepared_dir)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    total_dropped = 0
    for split_name, split_info in manifest["splits"].items():
        verify_examples(raw_by_qid, manifest, split_name)
        out_path = prepared_dir / f"{split_name}.jsonl"
        if out_path.exists():
            out_path.unlink()  # prepared files are derived; regenerating is always safe
        # samples.jsonl mirrors the prepared data into results/raw so the precomputed
        # explorer (Docker image, no dataset download) can display questions/passages.
        # It embeds HotpotQA text → distributed under CC BY-SA 4.0, see DATA_CARD.md.
        samples_path = Path(cfg.paths.results_raw) / split_name / "samples" / "records.jsonl"
        if samples_path.exists():
            samples_path.unlink()
        n = 0
        for qid in split_info["question_ids"]:
            example = build_example(raw_by_qid[qid])
            total_dropped += len(example.dropped_supporting_facts)
            record = example.to_json()
            record["raw_fingerprint"] = manifest["example_hashes"][qid]
            append_record(out_path, record)
            append_record(samples_path, record)
            n += 1
        logger.info("prepared %s: %d examples -> %s (+ samples.jsonl)", split_name, n, out_path)
    if total_dropped:
        logger.warning(
            "%d supporting-fact entries dropped across all splits (recorded per-example)",
            total_dropped,
        )
    logger.info("data prepare complete")
