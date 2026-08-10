"""Canonical query text construction for attribution ablations."""

from __future__ import annotations

from typing import Literal

QueryConvention = Literal["question", "answer", "question_answer"]


def compose_query(question: str, answer: str, convention: QueryConvention) -> str:
    """Build the exact query text declared by an attribution method."""
    question = question.strip()
    answer = answer.strip()
    if convention == "question":
        return question
    if convention == "answer":
        return answer
    if convention == "question_answer":
        return " ".join(part for part in (question, answer) if part)
    raise ValueError(f"unknown query convention: {convention!r}")
