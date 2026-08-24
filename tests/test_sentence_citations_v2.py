"""Prompt v2 binds stable sentence aliases without changing passage-citation v1."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_evidence.config import GenerationConfig, load_config, validate_v2_runtime_binding
from rag_evidence.data.schema import Passage
from rag_evidence.errors import ConfigError
from rag_evidence.generation.backends import directory_artifact_sha256
from rag_evidence.generation.citations import (
    parse_citations,
    parse_sentence_citation_response,
)
from rag_evidence.generation.prompts import (
    build_messages,
    build_sentence_alias_map,
    prompt_hash,
)
from rag_evidence.storage.runmeta import scientific_config


def _passages() -> list[Passage]:
    return [
        Passage("q-p00", 0, "Bridge", ("Designed by Noor.", "Opened in 1999."), True),
        Passage("q-p01", 1, "Noor", ("Noor is from Riverton.",), True),
    ]


def test_v1_prompt_hash_and_passage_parser_remain_unchanged() -> None:
    assert prompt_hash("v1") == "sha256:2a65bb7f97966405"
    parsed = parse_citations("Riverton [P1][P2]", {"P1": "q-p00", "P2": "q-p01"})
    assert parsed.cited_passage_ids == ("q-p00", "q-p01")


def test_v2_prompt_has_exact_registry_hash_and_stable_sentence_aliases() -> None:
    passages = _passages()
    alias_map = {"P3": "q-p00", "P7": "q-p01"}
    sentence_map = build_sentence_alias_map(passages, alias_map)

    assert prompt_hash("v2") == "sha256:7bf172cbaef00537"
    assert sentence_map == {
        "P3.S1": "q-p00-s00",
        "P3.S2": "q-p00-s01",
        "P7.S1": "q-p01-s00",
    }
    messages = build_messages("Where is Noor from?", passages, alias_map, version="v2")
    assert "[P3] Bridge" in messages[1]["content"]
    assert "[P3.S1] Designed by Noor." in messages[1]["content"]
    assert "[P7.S1] Noor is from Riverton." in messages[1]["content"]

    ablated = build_messages("Where is Noor from?", [passages[1]], alias_map, version="v2")
    assert "[P7.S1] Noor is from Riverton." in ablated[1]["content"]
    assert "[P1" not in ablated[1]["content"]


def test_sentence_parser_separates_answer_valid_invalid_and_missing_citations() -> None:
    passage_map = {"P1": "q-p00", "P2": "q-p01"}
    sentence_map = {
        "P1.S1": "q-p00-s00",
        "P1.S2": "q-p00-s01",
        "P2.S1": "q-p01-s00",
    }
    parsed = parse_sentence_citation_response(
        "Riverton [P2.S1][P9.S2][P1]", passage_map, sentence_map
    )

    assert parsed.answer_text == "Riverton"
    assert parsed.raw_aliases == ("P2.S1", "P9.S2", "P1")
    assert parsed.cited_sentence_ids == ("q-p01-s00",)
    assert parsed.invalid_aliases == ("P9.S2",)
    assert parsed.cited_passage_ids == ("q-p00",)
    assert parsed.missing_citations is False

    missing = parse_sentence_citation_response("Riverton", passage_map, sentence_map)
    assert missing.answer_text == "Riverton"
    assert missing.missing_citations is True
    abstained = parse_sentence_citation_response("INSUFFICIENT EVIDENCE", passage_map, sentence_map)
    assert abstained.missing_citations is False


def test_v2_generation_config_binds_revisions_runtime_and_decoding(tmp_path: Path) -> None:
    path = tmp_path / "v2.yaml"
    path.write_text(
        """\
run_name: v2-test
split: smoke
generation:
  backend: fake
  model_revision: cdbee75f17c01a7cc42f958dc650907174af0554
  tokenizer_id: Qwen/Qwen3-4B-Instruct-2507
  tokenizer_revision: cdbee75f17c01a7cc42f958dc650907174af0554
  local_artifact_sha256: null
  runtime: transformers
  runtime_version: 5.14.1
  prompt_version: v2
  do_sample: false
  num_beams: 1
""",
        encoding="utf-8",
    )
    cfg = load_config(path)
    scientific = scientific_config(cfg, "generate")

    assert scientific["generation"]["model_revision"] == (
        "cdbee75f17c01a7cc42f958dc650907174af0554"
    )
    assert scientific["generation"]["tokenizer_revision"] == (
        "cdbee75f17c01a7cc42f958dc650907174af0554"
    )
    assert scientific["generation"]["local_artifact_sha256"] is None
    assert scientific["generation"]["runtime_version"] == "5.14.1"
    assert scientific["generation"]["do_sample"] is False
    assert scientific["generation"]["num_beams"] == 1
    assert scientific["generation"]["prompt_hash"] == "sha256:7bf172cbaef00537"
    # Fake runs are synthetic verification and never require a model snapshot.
    validate_v2_runtime_binding(cfg.generation)


def test_real_v2_run_fails_before_model_load_when_local_hash_is_unbound() -> None:
    config = GenerationConfig(
        prompt_version="v2",
        model_revision="cdbee75f17c01a7cc42f958dc650907174af0554",
        tokenizer_id="Qwen/Qwen3-4B-Instruct-2507",
        tokenizer_revision="cdbee75f17c01a7cc42f958dc650907174af0554",
        local_artifact_sha256=None,
        runtime="transformers",
        runtime_version="5.14.1",
    )

    with pytest.raises(ConfigError, match="local_artifact_sha256"):
        validate_v2_runtime_binding(config)


@pytest.mark.parametrize("field", ["model_revision", "tokenizer_revision"])
def test_revision_fields_require_exact_commit_hash(field: str) -> None:
    with pytest.raises(ValueError, match="40-character"):
        GenerationConfig.model_validate({field: "main"})


def test_local_artifact_hash_requires_full_sha256() -> None:
    with pytest.raises(ValueError, match="64-character"):
        GenerationConfig(local_artifact_sha256="abc")


def test_local_snapshot_digest_binds_relative_paths_and_file_bytes(tmp_path: Path) -> None:
    (tmp_path / "weights").mkdir()
    (tmp_path / "config.json").write_bytes(b"{}")
    (tmp_path / "weights" / "part.bin").write_bytes(b"abc")

    assert directory_artifact_sha256(tmp_path) == (
        "b24bdc218e1fa48d940a6ac45cb4229e6a9c0f711dbae8b8398f4abe33f7f95c"
    )
