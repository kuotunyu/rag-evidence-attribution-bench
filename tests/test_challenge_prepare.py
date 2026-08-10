"""Challenge preparation validates source data before atomic versioned writes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.core import TyperGroup, TyperOption
from typer.main import get_command

from rag_evidence.cli import app
from rag_evidence.config import AppConfig, load_config
from rag_evidence.data.challenge_prepare import prepare_challenge
from rag_evidence.data.hotpot import build_example
from rag_evidence.data.manifest_v2 import build_manifest_v2
from rag_evidence.data.splits import example_fingerprint
from rag_evidence.errors import DataError
from rag_evidence.storage.artifacts import (
    read_json,
    read_records,
    write_json_atomic,
    write_records_atomic,
)

REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"


def test_data_challenge_cli_is_registered() -> None:
    root = get_command(app)
    assert isinstance(root, TyperGroup)
    data = root.commands["data"]
    assert isinstance(data, TyperGroup)
    challenge = data.commands["challenge"]
    config = next(parameter for parameter in challenge.params if parameter.name == "config")
    assert isinstance(config, TyperOption)
    assert "--config" in config.opts


def _raw(index: int) -> dict[str, Any]:
    qid = f"q-{index}"
    return {
        "question_id": qid,
        "question": f"Who founded Project {index}?",
        "answer": f"Founder {index}",
        "type": "bridge" if index % 2 else "comparison",
        "level": "hard",
        "context": [
            [f"Project {index}", [f"Project {index} was founded by Founder {index}."]],
            [f"Link {index}", [f"Link {index} identifies Project {index}."]],
            [f"Distractor {index}", [f"Distractor {index} hosts a local festival."]],
        ],
        "supporting_facts": [[f"Project {index}", 0], [f"Link {index}", 0]],
    }


def _raws() -> list[dict[str, Any]]:
    return [_raw(index) for index in range(6)]


def _config(tmp_path: Path, *, schema_version: int = 2) -> AppConfig:
    path = tmp_path / "challenge.yaml"
    revision = f"  hf_revision: {REVISION}\n" if schema_version == 2 else ""
    path.write_text(
        f"""\
run_name: challenge-test
split: eval
seed: 7
data:
  manifest_schema_version: {schema_version}
{revision}  manifest_path: data/manifests/split_manifest_v2.json
  prepared_dir: data/v2/prepared
  split_seed: 19
  split_sizes: {{smoke: 2, dev: 2, eval: 2}}
challenge:
  schema_version: 1
  seed: 20260810
  transform_version: challenge-v1
  manifest_path: data/manifests/challenge_manifest_v1.json
  prepared_dir: data/v2/challenge
paths:
  data_dir: data
  results_raw: results/v2/raw
  results_derived: results/v2/derived
  assets_dir: results/v2/assets
""",
        encoding="utf-8",
    )
    return load_config(path)


def _write_source(cfg: AppConfig) -> dict[str, Any]:
    raws = _raws()
    manifest = build_manifest_v2(
        raws,
        seed=cfg.data.split_seed,
        requested_sizes=dict(cfg.data.split_sizes),
        dataset_info={
            "hf_path": "hotpotqa/hotpot_qa",
            "hf_config": "distractor",
            "hf_split": "validation",
            "requested_revision": REVISION,
            "resolved_revision": REVISION,
        },
    )
    write_json_atomic(cfg.manifest_file, manifest)
    raw_by_qid = {raw["question_id"]: raw for raw in raws}
    for split, split_info in manifest["splits"].items():
        records = []
        for qid in split_info["question_ids"]:
            record = build_example(raw_by_qid[qid]).to_json()
            record["raw_fingerprint"] = example_fingerprint(raw_by_qid[qid])
            records.append(record)
        write_records_atomic(Path(cfg.data.prepared_dir) / f"{split}.jsonl", records)
    return manifest


def _snapshot() -> dict[str, bytes]:
    roots = (Path("data/manifests"), Path("data/v2/challenge"), Path("results/v2/raw"))
    return {
        path.as_posix(): path.read_bytes()
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_prepare_challenge_writes_three_per_parent_and_is_byte_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = _config(tmp_path)
    source = _write_source(cfg)
    natural_before = {
        path.as_posix(): path.read_bytes() for path in Path("data/v2/prepared").glob("*.jsonl")
    }

    prepare_challenge(cfg)

    manifest = read_json(cfg.challenge_manifest_file)
    assert manifest["source"]["resolved_revision"] == REVISION
    for split in ("smoke", "dev", "eval"):
        prepared = Path(cfg.challenge.prepared_dir) / f"{split}.jsonl"
        samples = cfg.results_raw_dir / split / "challenge/samples/records.jsonl"
        rows = list(read_records(prepared))
        assert len(rows) == 3 * source["splits"][split]["size"]
        assert samples.read_bytes() == prepared.read_bytes()
    assert not Path("data/prepared").exists()
    assert not Path("results/raw").exists()
    first = _snapshot()

    prepare_challenge(cfg)

    assert _snapshot() == first
    assert {
        path.as_posix(): path.read_bytes() for path in Path("data/v2/prepared").glob("*.jsonl")
    } == natural_before


def test_prepare_challenge_rejects_source_fingerprint_before_any_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = _config(tmp_path)
    _write_source(cfg)
    path = Path(cfg.data.prepared_dir) / "eval.jsonl"
    rows = list(read_records(path))
    rows[0]["raw_fingerprint"] = "0" * 64
    write_records_atomic(path, rows)

    with pytest.raises(DataError, match="fingerprint"):
        prepare_challenge(cfg)

    assert not cfg.challenge_manifest_file.exists()
    assert not cfg.challenge_prepared_dir.exists()
    assert not any(Path("results/v2/raw").glob("*/challenge"))


def test_prepare_challenge_validates_existing_manifest_before_replacing_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = _config(tmp_path)
    _write_source(cfg)
    prepare_challenge(cfg)
    before = _snapshot()
    manifest = read_json(cfg.challenge_manifest_file)
    manifest["source"]["manifest_sha256"] = "0" * 64
    write_json_atomic(cfg.challenge_manifest_file, manifest)
    corrupted_manifest = cfg.challenge_manifest_file.read_bytes()

    with pytest.raises(DataError, match="source"):
        prepare_challenge(cfg)

    after = _snapshot()
    assert after[cfg.challenge_manifest_file.as_posix()] == corrupted_manifest
    for path, content in before.items():
        if path != cfg.challenge_manifest_file.as_posix():
            assert after[path] == content


def test_prepare_challenge_requires_all_splits_and_manifest_schema_v2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = _config(tmp_path)
    _write_source(cfg)
    (Path(cfg.data.prepared_dir) / "dev.jsonl").unlink()
    with pytest.raises(DataError, match=r"dev|prepared"):
        prepare_challenge(cfg)

    schema_v1 = _config(tmp_path, schema_version=1)
    with pytest.raises(DataError, match="schema v2"):
        prepare_challenge(schema_v1)
