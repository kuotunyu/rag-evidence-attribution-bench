"""Human eligibility is a hard precondition for every challenge execution stage."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from rag_evidence.annotation.assignment import build_dual_assignments
from rag_evidence.annotation.blinding import project_challenge
from rag_evidence.annotation.models import EligibilityArtifact, EligibilityRecord
from rag_evidence.config import AppConfig
from rag_evidence.data.challenge_execution import (
    build_challenge_execution_config,
    load_eligible_challenge_examples,
)
from rag_evidence.data.challenge_schema import (
    PENDING_REVIEW,
    ChallengeRecord,
    make_challenge_id,
    record_content_hash,
    remap_example,
)
from rag_evidence.data.hotpot import load_prepared_verified
from rag_evidence.data.schema import Example
from rag_evidence.errors import DataError
from rag_evidence.storage.artifacts import write_json_atomic, write_records_atomic


def _challenge(parent: Example, transformation: str = "missing_hop") -> ChallengeRecord:
    challenge_id = make_challenge_id(
        parent.question_id,
        "smoke",
        transformation,
        transform_version="challenge-v1",
    )
    example = remap_example(
        parent,
        challenge_id,
        passages=parent.passages,
        gold_slots={p.index for p in parent.passages if p.is_gold},
        supporting_slots={
            (int(sid.rsplit("-p", 1)[1].split("-s", 1)[0]), int(sid.rsplit("-s", 1)[1]))
            for sid in parent.supporting_fact_sentence_ids
        },
    )
    unhashed = {
        "schema_version": 1,
        "challenge_id": challenge_id,
        "parent_question_id": parent.question_id,
        "source_split": "smoke",
        "transformation": transformation,
        "transform_version": "challenge-v1",
        "seed": 20260810,
        "expected_answerability": "unanswerable",
        "review": PENDING_REVIEW.to_json(),
        "changed_fields": ["/example/passages/0"],
        "provenance": {"fixture": True},
        "parent_fingerprint": "a" * 64,
        "example": example.to_json(),
    }
    return ChallengeRecord.from_json({**unhashed, "content_hash": record_content_hash(unhashed)})


def eligible_challenge_config(
    tiny_env: AppConfig,
    tmp_path: Path,
    *,
    eligible: bool = True,
    phase: str = "pilot",
    protocol_version: str = "pilot-v0.2-draft",
    transformation: str = "missing_hop",
) -> AppConfig:
    parent = load_prepared_verified(tiny_env)[0]
    record = _challenge(parent, transformation)
    task = project_challenge(
        record,
        instruction_version=protocol_version,
        instruction_hash="1" * 64,
        batch="fixture-batch",
        namespace=f"{phase}-fixture",
    )
    assignments = build_dual_assignments([task], ("ann-a1", "ann-b2"), seed=19)
    eligibility = EligibilityArtifact(
        phase=phase,  # type: ignore[arg-type]
        protocol_version=protocol_version,
        protocol_hash="2" * 64,
        generated_at=dt.datetime(2026, 8, 23, tzinfo=dt.UTC),
        records=(
            EligibilityRecord(
                challenge_id=record.challenge_id,
                task_content_hash=task.task_content_hash,
                source_annotation_hashes=("3" * 64, "4" * 64),
                source_answerabilities=("unanswerable", "unanswerable"),
                adjudication_hash=None,
                final_answerability="unanswerable",
                eligible=eligible,
                exclusion_reason=None if eligible else "synthetic pending review",
            ),
        ),
    )
    source = tmp_path / f"{transformation}.jsonl"
    assignments_path = tmp_path / f"{transformation}-assignments.json"
    eligibility_path = tmp_path / f"{transformation}-eligibility.json"
    write_records_atomic(source, [record.to_json()])
    write_json_atomic(assignments_path, assignments.model_dump(mode="json"))
    write_json_atomic(eligibility_path, eligibility.model_dump(mode="json"))
    return build_challenge_execution_config(
        tiny_env,
        phase=phase,  # type: ignore[arg-type]
        variant=transformation,  # type: ignore[arg-type]
        challenge_records_path=source,
        assignment_manifest_path=assignments_path,
        eligibility_path=eligibility_path,
        natural_samples_path=Path("results/raw/smoke/samples/records.jsonl"),
    )


def test_human_eligibility_and_source_hashes_are_required_before_output(
    tiny_env: AppConfig, tmp_path: Path
) -> None:
    cfg = eligible_challenge_config(tiny_env, tmp_path, eligible=False)

    with pytest.raises(DataError, match="no human-eligible"):
        load_eligible_challenge_examples(cfg)
    assert not cfg.results_raw_dir.exists()


def test_expected_transform_label_cannot_override_human_decision(
    tiny_env: AppConfig, tmp_path: Path
) -> None:
    cfg = eligible_challenge_config(tiny_env, tmp_path)
    loaded = load_eligible_challenge_examples(cfg)

    assert len(loaded.examples) == 1
    metadata = loaded.metadata[loaded.examples[0].question_id]
    assert metadata["human_answerability"] == "unanswerable"
    assert "expected_answerability" not in metadata


def test_confirmatory_execution_refuses_a_draft_protocol(
    tiny_env: AppConfig, tmp_path: Path
) -> None:
    cfg = eligible_challenge_config(
        tiny_env,
        tmp_path,
        phase="confirmatory",
        protocol_version="v2-confirmatory-draft",
    )

    with pytest.raises(DataError, match="frozen"):
        load_eligible_challenge_examples(cfg)


def test_execution_roots_separate_phase_and_variant(tiny_env: AppConfig, tmp_path: Path) -> None:
    pilot = eligible_challenge_config(tiny_env, tmp_path)
    confirmatory = eligible_challenge_config(
        tiny_env,
        tmp_path,
        phase="confirmatory",
        protocol_version="v2-confirmatory-frozen",
    )
    swap = eligible_challenge_config(
        tiny_env,
        tmp_path,
        transformation="evidence_swap",
    )

    assert pilot.results_raw_dir != confirmatory.results_raw_dir
    assert pilot.results_raw_dir != swap.results_raw_dir
    assert "pilot/challenge/missing_hop/raw" in pilot.results_raw_dir.as_posix()
    assert "confirmatory/challenge/missing_hop/raw" in confirmatory.results_raw_dir.as_posix()
