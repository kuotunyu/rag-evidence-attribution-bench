"""Parse and strip [P#] citations from model responses.

Accepted shapes (models mix them freely): [P1], [P1][P2], [P1, P2], [P1 P2], [p3].
Anything inside brackets that is not exclusively P-references is left untouched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_evidence.generation.prompts import ABSTAIN_TEXT

_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")
_PREF_RE = re.compile(r"^[Pp]([1-9]\d*)$")
_SPLIT_RE = re.compile(r"[,\s;/]+")


@dataclass(frozen=True)
class ParsedCitations:
    raw_aliases: tuple[str, ...]  # ordered, deduplicated, normalized to "P<n>"
    cited_passage_ids: tuple[str, ...]  # resolved through the alias_map
    invalid_aliases: tuple[str, ...]  # cited but not in the alias_map (e.g. [P11])


def _is_pure_citation_group(content: str) -> list[str] | None:
    tokens = [t for t in _SPLIT_RE.split(content.strip()) if t]
    if not tokens:
        return None
    normalized = []
    for tok in tokens:
        m = _PREF_RE.match(tok)
        if m is None:
            return None
        normalized.append(f"P{m.group(1)}")
    return normalized


def parse_citations(text: str, alias_map: dict[str, str]) -> ParsedCitations:
    seen: list[str] = []
    for match in _BRACKET_RE.finditer(text):
        aliases = _is_pure_citation_group(match.group(1))
        if aliases is None:
            continue
        for alias in aliases:
            if alias not in seen:
                seen.append(alias)
    valid = [a for a in seen if a in alias_map]
    invalid = [a for a in seen if a not in alias_map]
    return ParsedCitations(
        raw_aliases=tuple(seen),
        cited_passage_ids=tuple(alias_map[a] for a in valid),
        invalid_aliases=tuple(invalid),
    )


def strip_citations(text: str) -> str:
    """Remove pure-citation bracket groups; collapse the whitespace they leave behind."""

    def _sub(match: re.Match[str]) -> str:
        return "" if _is_pure_citation_group(match.group(1)) is not None else match.group(0)

    stripped = _BRACKET_RE.sub(_sub, text)
    return re.sub(r"\s+", " ", stripped).strip(" \t\n.,;:")


def is_abstention(text: str) -> bool:
    return ABSTAIN_TEXT.lower() in text.lower()
