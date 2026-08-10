"""Per-question method comparisons across incomplete attribution runs."""

from __future__ import annotations

from rag_evidence.evaluation.comparisons import compare_attribution_methods


def _row(
    qid: str,
    *,
    f1: float,
    ap: float,
    ndcg: float,
    sufficiency: float,
    comprehensiveness: float,
    in_subset: bool = True,
) -> dict[str, object]:
    return {
        "question_id": qid,
        "in_agreement_subset": in_subset,
        "f1_at": {"2": f1},
        "ap": ap,
        "ndcg_at_10": ndcg,
        "sufficiency": sufficiency,
        "comprehensiveness": comprehensiveness,
    }


def test_comparison_aligns_each_estimand_on_complete_question_pairs() -> None:
    candidate = {
        "q1": _row("q1", f1=0.8, ap=0.9, ndcg=0.9, sufficiency=1.0, comprehensiveness=4.0),
        "q2": _row(
            "q2",
            f1=0.7,
            ap=0.8,
            ndcg=0.8,
            sufficiency=1.5,
            comprehensiveness=3.5,
            in_subset=False,
        ),
        "q3": _row("q3", f1=0.6, ap=0.7, ndcg=0.7, sufficiency=2.0, comprehensiveness=3.0),
    }
    comparator = {
        "q1": _row("q1", f1=0.5, ap=0.6, ndcg=0.6, sufficiency=2.0, comprehensiveness=2.0),
        "q2": _row("q2", f1=0.4, ap=0.5, ndcg=0.5, sufficiency=2.5, comprehensiveness=1.5),
        "q4": _row("q4", f1=0.3, ap=0.4, ndcg=0.4, sufficiency=3.0, comprehensiveness=1.0),
    }

    result = compare_attribution_methods(
        candidate,
        comparator,
        split="eval",
        mode="generated",
        candidate_method="leave_one_out",
        comparator_method="control_lexical",
        analysis_tier="confirmatory",
        primary_k=2,
        global_seed=42,
        resamples=1_000,
        confidence=0.95,
        tolerance=0.0,
    )

    assert result["candidate_method"] == "leave_one_out"
    assert result["comparator_method"] == "control_lexical"
    assert result["analysis_tier"] == "confirmatory"
    agreement = result["metrics"]["f1_at_2"]
    assert agreement["n_pairs"] == 1
    assert agreement["exclusions"] == {
        "candidate_missing": 1,
        "comparator_missing": 1,
        "candidate_null": 1,
        "comparator_null": 0,
    }
    assert result["metrics"]["mean_average_precision"]["n_pairs"] == 1
    assert result["metrics"]["sufficiency"]["n_pairs"] == 2
    assert result["metrics"]["comprehensiveness"]["favorable_direction"] == "higher"


def test_comparison_seed_depends_on_metric_identity() -> None:
    rows = {
        "q1": _row("q1", f1=0.8, ap=0.8, ndcg=0.8, sufficiency=1.0, comprehensiveness=4.0),
        "q2": _row("q2", f1=0.7, ap=0.7, ndcg=0.7, sufficiency=1.5, comprehensiveness=3.5),
    }
    controls = {
        "q1": _row("q1", f1=0.5, ap=0.5, ndcg=0.5, sufficiency=2.0, comprehensiveness=2.0),
        "q2": _row("q2", f1=0.4, ap=0.4, ndcg=0.4, sufficiency=2.5, comprehensiveness=1.5),
    }

    result = compare_attribution_methods(
        rows,
        controls,
        split="dev",
        mode="gold",
        candidate_method="embedding",
        comparator_method="control_random",
        analysis_tier="secondary",
        primary_k=2,
        global_seed=7,
        resamples=1_000,
        confidence=0.95,
        tolerance=0.0,
    )
    seeds = {metric["seed_uint64"] for metric in result["metrics"].values()}

    assert len(seeds) == len(result["metrics"])
