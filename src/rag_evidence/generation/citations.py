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
_ANY_ALIAS_RE = re.compile(r"^[Pp]([1-9]\d*)(?:\.[Ss]([1-9]\d*))?$")


@dataclass(frozen=True)
class ParsedCitations:
    raw_aliases: tuple[str, ...]  # ordered, deduplicated, normalized to "P<n>"
    cited_passage_ids: tuple[str, ...]  # resolved through the alias_map
    invalid_aliases: tuple[str, ...]  # cited but not in the alias_map (e.g. [P11])


@dataclass(frozen=True)
class ParsedSentenceCitationResponse:
    answer_text: str
    raw_aliases: tuple[str, ...]
    cited_sentence_ids: tuple[str, ...]
    cited_passage_ids: tuple[str, ...]
    invalid_aliases: tuple[str, ...]
    missing_citations: bool


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


def _all_citation_aliases(text: str) -> tuple[str, ...]:
    aliases: list[str] = []
    for match in _BRACKET_RE.finditer(text):
        tokens = [token for token in _SPLIT_RE.split(match.group(1).strip()) if token]
        if not tokens:
            continue
        normalized: list[str] = []
        for token in tokens:
            alias_match = _ANY_ALIAS_RE.fullmatch(token)
            if alias_match is None:
                normalized = []
                break
            passage = f"P{alias_match.group(1)}"
            sentence = alias_match.group(2)
            normalized.append(f"{passage}.S{sentence}" if sentence else passage)
        for alias in normalized:
            if alias not in aliases:
                aliases.append(alias)
    return tuple(aliases)


def _strip_all_citation_aliases(text: str) -> str:
    def remove(match: re.Match[str]) -> str:
        tokens = [token for token in _SPLIT_RE.split(match.group(1).strip()) if token]
        return (
            ""
            if tokens and all(_ANY_ALIAS_RE.fullmatch(token) for token in tokens)
            else match.group(0)
        )

    stripped = _BRACKET_RE.sub(remove, text)
    return re.sub(r"\s+", " ", stripped).strip(" \t\n.,;:")


def parse_sentence_citation_response(
    text: str,
    passage_alias_map: dict[str, str],
    sentence_alias_map: dict[str, str],
) -> ParsedSentenceCitationResponse:
    """Parse prompt-v2 output while reporting sentence and legacy passage citations separately."""
    raw = _all_citation_aliases(text)
    valid_sentences = [alias for alias in raw if alias in sentence_alias_map]
    valid_passages = [alias for alias in raw if alias in passage_alias_map]
    invalid = [
        alias for alias in raw if alias not in sentence_alias_map and alias not in passage_alias_map
    ]
    abstained = is_abstention(text)
    return ParsedSentenceCitationResponse(
        answer_text="" if abstained else _strip_all_citation_aliases(text),
        raw_aliases=raw,
        cited_sentence_ids=tuple(sentence_alias_map[alias] for alias in valid_sentences),
        cited_passage_ids=tuple(passage_alias_map[alias] for alias in valid_passages),
        invalid_aliases=tuple(invalid),
        missing_citations=not abstained and not raw,
    )
