"""Public scientific language stays inside the approved validity boundary."""

from __future__ import annotations

import re
from pathlib import Path

from rag_evidence.reporting.report import render_results_block

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (REPO_ROOT / name).read_text(encoding="utf-8")


def test_homepage_readmes_are_compact_and_link_detailed_evidence() -> None:
    for name, detail in (
        ("README.md", "docs/HISTORICAL_V01_EVIDENCE_ZH.md"),
        ("README_en.md", "docs/HISTORICAL_V01_EVIDENCE_EN.md"),
    ):
        text = _read(name)
        assert 250 <= len(text.splitlines()) <= 350
        assert detail in text
        assert "PILOT_PROTOCOL.md" in text
        assert "PREREGISTRATION_V2_CONFIRMATORY_DRAFT.md" in text
        assert "v0.1" in text
        assert "v2" in text


def test_public_documents_reject_retired_causal_attribution_claims() -> None:
    public = "\n".join(_read(name) for name in ("README.md", "README_en.md", "MODEL_CARD.md"))
    retired = (
        r"leave[_ -]one[_ -]out[^\n|]{0,80}causal",
        r"causal[^\n|]{0,80}leave[_ -]one[_ -]out",
        r"arc[-_ ]?jsd[^\n|]{0,80}causal",
        r"causal[^\n|]{0,80}arc[-_ ]?jsd",
        r"causal \(primary baseline\)",
    )
    lowered = public.casefold()
    for pattern in retired:
        assert re.search(pattern, lowered) is None, pattern

    assert "deletion-based teacher-forced target-dependence diagnostic" in lowered
    assert "experimental distributional-dependence diagnostic" in lowered


def test_reference_agreement_is_not_described_as_complete_faithfulness() -> None:
    public = "\n".join(_read(name).casefold() for name in ("README.md", "README_en.md"))
    assert "reference agreement" in public
    assert "not causal ground truth" in public
    assert "not complete faithfulness" in public


def test_generated_report_uses_diagnostic_and_construct_language() -> None:
    block = render_results_block(
        {
            "schema_version": 2,
            "package_version": "0.2.0",
            "dataset_hash": "a" * 64,
            "source_snapshot_utc": "2026-08-23T00:00:00Z",
            "primary_k": 2,
            "splits": {
                "smoke": {
                    "n_questions": 1,
                    "retrieval": {},
                    "generation": {},
                    "attribution": {
                        "gold": {
                            "m": {
                                "schema_version": 2,
                                "is_control": False,
                                "agreement": {
                                    "n": 1,
                                    "prf_at": {"2": {"f1": 1.0}},
                                    "mean_average_precision": 1.0,
                                    "ndcg_at_10": 1.0,
                                },
                                "causal_dependence": {
                                    "n": 1,
                                    "sufficiency_mean": 0.1,
                                    "comprehensiveness_mean": 0.2,
                                },
                                "execution": {
                                    "seconds_per_sample": 0.1,
                                    "failure_rate": 0.0,
                                },
                            },
                            "causal_validation": {
                                "status": "not_run",
                                "missing_methods": ["oracle_gold"],
                            },
                        }
                    },
                }
            },
        }
    )

    assert "sufficiency diagnostic" in block
    assert "comprehensiveness diagnostic" in block
    assert "Construct validation: **NOT RUN**" in block
    assert "Causal-dependence validation" not in block
