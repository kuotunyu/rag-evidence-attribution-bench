"""Declared attribution rows distinguish absence, unsupported modes, and exploration."""

from __future__ import annotations

from rag_evidence.evaluation.protocol import build_experiment_matrix


def test_experiment_matrix_never_treats_discovered_subset_as_complete_protocol() -> None:
    matrix = build_experiment_matrix(
        {
            "gold": {"leave_one_out", "embedding", "control_lexical", "contextcite"},
            "generated": {"leave_one_out", "embedding", "control_lexical", "citations"},
        }
    )

    assert matrix["gold"]["leave_one_out"]["status"] == "complete"
    assert matrix["gold"]["embedding_question"]["status"] == "not_run"
    assert matrix["gold"]["oracle_gold"]["status"] == "not_run"
    assert matrix["gold"]["citations"]["status"] == "unsupported"
    assert matrix["gold"]["contextcite"]["status"] == "experimental_unvalidated"
    assert matrix["generated"]["citations"]["status"] == "complete"
    assert matrix["generated"]["control_random"]["status"] == "not_run"


def test_optional_adapter_is_not_run_when_no_artifact_exists() -> None:
    matrix = build_experiment_matrix({"gold": set(), "generated": set()})

    assert matrix["gold"]["arc_jsd"]["status"] == "not_run"
    assert matrix["generated"]["contextcite"]["status"] == "not_run"


def test_matrix_records_analysis_tiers_before_results_are_viewed() -> None:
    matrix = build_experiment_matrix(
        {
            "gold": {"leave_one_out", "embedding", "control_random"},
            "generated": {"citations"},
        }
    )

    assert matrix["gold"]["leave_one_out"]["analysis_tier"] == "confirmatory"
    assert matrix["gold"]["embedding"]["analysis_tier"] == "secondary"
    assert matrix["gold"]["control_random"]["analysis_tier"] == "control"
    assert matrix["generated"]["citations"]["analysis_tier"] == "secondary"
