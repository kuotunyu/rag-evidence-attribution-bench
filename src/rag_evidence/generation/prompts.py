"""Versioned prompt templates and message building.

The prompt template is part of the scientific config: its version string is hashed into
every run manifest (prompt_hash), so any wording change is visible as a config change.

Alias rule (load-bearing for attribution): aliases are bound to passage IDENTITY, not
position. `build_messages` takes an explicit alias_map; when a passage is ablated from
the context, the remaining passages keep their original aliases — prompts stay maximally
comparable across ablations.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence

from rag_evidence.data.schema import Passage
from rag_evidence.errors import ConfigError

logger = logging.getLogger(__name__)

ABSTAIN_TEXT = "INSUFFICIENT EVIDENCE"

_SYSTEM_V1 = (
    "You answer multi-hop questions strictly from the provided passages. "
    "Keep the answer short: a word, a name, yes/no, or a brief phrase. "
    "After the answer, cite the IDs of the passages that support it, like [P3] or [P2][P7]. "
    f"If the passages do not contain enough evidence, reply exactly: {ABSTAIN_TEXT}"
)

_USER_TEMPLATE_V1 = (
    "Passages:\n{passages_block}\n\n"
    "Question: {question}\n\n"
    "Short answer with citations:"
)

PROMPT_VERSIONS: dict[str, dict[str, str]] = {
    "v1": {"system": _SYSTEM_V1, "user_template": _USER_TEMPLATE_V1},
}


def prompt_hash(version: str) -> str:
    try:
        parts = PROMPT_VERSIONS[version]
    except KeyError as exc:
        raise ConfigError(
            f"unknown prompt_version {version!r}; known: {sorted(PROMPT_VERSIONS)}"
        ) from exc
    material = parts["system"] + "\n---\n" + parts["user_template"]
    return "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def build_messages(
    question: str,
    passages: Sequence[Passage],
    alias_map: dict[str, str],
    *,
    version: str = "v1",
) -> list[dict[str, str]]:
    """Chat messages for the given passages (prompt order = `passages` order).

    alias_map maps alias -> passage_id and MUST cover every passage given; passages not
    in the map are a programming error (aliases are assigned once per sample, upstream).
    """
    parts = PROMPT_VERSIONS.get(version)
    if parts is None:
        raise ConfigError(f"unknown prompt_version {version!r}")
    by_pid = {pid: alias for alias, pid in alias_map.items()}
    lines = []
    for p in passages:
        alias = by_pid.get(p.passage_id)
        if alias is None:
            raise ValueError(f"passage {p.passage_id} missing from alias_map")
        lines.append(f"[{alias}] {p.title}: {p.text}")
    user = parts["user_template"].format(
        passages_block="\n".join(lines), question=question.strip()
    )
    return [
        {"role": "system", "content": parts["system"]},
        {"role": "user", "content": user},
    ]
