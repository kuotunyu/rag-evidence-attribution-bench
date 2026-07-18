"""Stable ID construction and prompt-alias mapping.

Scheme (positional, protected by the manifest's per-example content fingerprints):
    question : HotpotQA `_id` (stable 24-hex string shipped with the dataset)
    passage  : {question_id}-p{index:02d}   (index = position in the example's context list)
    sentence : {passage_id}-s{jdx:02d}      (jdx = sentence index inside the passage)

Prompt aliases [P1]..[P10] are prompt-local and 1-based; every artifact persists its
alias_map {"P1": "<passage_id>", ...}. Aliases are bound to passage identity: when a
passage is ablated from a prompt, the remaining passages KEEP their original aliases.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_PASSAGE_RE = re.compile(r"^(?P<qid>.+)-p(?P<idx>\d{2,})$")
_ALIAS_RE = re.compile(r"^P(?P<n>[1-9]\d*)$")


def passage_id(question_id: str, index: int) -> str:
    if index < 0:
        raise ValueError(f"passage index must be >= 0, got {index}")
    return f"{question_id}-p{index:02d}"


def sentence_id(pid: str, sent_index: int) -> str:
    if sent_index < 0:
        raise ValueError(f"sentence index must be >= 0, got {sent_index}")
    return f"{pid}-s{sent_index:02d}"


def parse_passage_id(pid: str) -> tuple[str, int]:
    m = _PASSAGE_RE.match(pid)
    if m is None:
        raise ValueError(f"not a passage id: {pid!r}")
    return m.group("qid"), int(m.group("idx"))


def make_alias_map(passage_ids: Sequence[str]) -> dict[str, str]:
    """['pid_a', 'pid_b', …] (prompt order) → {'P1': 'pid_a', 'P2': 'pid_b', …}."""
    return {f"P{i + 1}": pid for i, pid in enumerate(passage_ids)}


def invert_alias_map(alias_map: dict[str, str]) -> dict[str, str]:
    return {pid: alias for alias, pid in alias_map.items()}


def is_alias(token: str) -> bool:
    return _ALIAS_RE.match(token) is not None
