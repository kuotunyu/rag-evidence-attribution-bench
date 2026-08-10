"""Declared Gate A attribution experiment matrix."""

from __future__ import annotations

from collections.abc import Mapping, Set
from typing import Any

_CORE_ALL_MODES = (
    "leave_one_out",
    "embedding",
    "control_random",
    "control_retrieval",
    "control_lexical",
    "control_length",
    "control_shuffled",
)
_QUERY_VARIANTS = (
    "embedding_question",
    "embedding_answer",
    "embedding_question_answer",
    "control_lexical_question",
    "control_lexical_answer",
    "control_lexical_question_answer",
)
_CONSTRUCT_CONTROLS = ("oracle_gold", "control_answer_string")
_OPTIONAL_ADAPTERS = ("arc_jsd", "contextcite")


def _analysis_tier(method: str) -> str:
    if method == "leave_one_out":
        return "confirmatory"
    if method.startswith("control_") or method == "oracle_gold":
        return "control"
    if method in _OPTIONAL_ADAPTERS:
        return "exploratory"
    return "secondary"


def build_experiment_matrix(discovered: Mapping[str, Set[str]]) -> dict[str, Any]:
    """Label every declared row independently of discovered-run completeness."""
    matrix: dict[str, Any] = {}
    declared = (*_CORE_ALL_MODES, *_QUERY_VARIANTS, *_CONSTRUCT_CONTROLS, *_OPTIONAL_ADAPTERS)
    for mode in ("gold", "generated"):
        available = set(discovered.get(mode, set()))
        rows: dict[str, Any] = {}
        for method in declared:
            if method in _OPTIONAL_ADAPTERS and method in available:
                status = "experimental_unvalidated"
            else:
                status = "complete" if method in available else "not_run"
            rows[method] = {
                "status": status,
                "analysis_tier": _analysis_tier(method),
            }
        rows["citations"] = {
            "status": (
                "unsupported"
                if mode == "gold"
                else ("complete" if "citations" in available else "not_run")
            ),
            "analysis_tier": "secondary",
        }
        matrix[mode] = rows
    return matrix
