"""ContextCite adapter (optional; `uv sync --extra contextcite`).

Wraps MadryLab's `context-cite` (surrogate-model attribution over context ablations)
through its designed extension points — no library internals are rewritten:
- a custom passage-level partitioner (sources = our passages, matching the benchmark's
  attribution unit),
- the existing model/tokenizer from our QwenBackend (no second copy in VRAM).

Honest scoping (documented, not hidden): ContextCite attributes the response IT
generates with its own prompt template — not an arbitrary teacher-forced target. Mode A
(gold answers) is therefore unsupported, and in mode B the attributed text is
ContextCite's own regeneration, which may differ from the stored generation-run answer;
the adapter records both and flags mismatches in metadata.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from rag_evidence.attribution.base import AttributionMethod, AttributionResult, ModelResources
from rag_evidence.attribution.registry import register
from rag_evidence.data.schema import Passage
from rag_evidence.errors import ConfigError, UpstreamMissingError
from rag_evidence.metrics.text import normalize_answer

_SEPARATOR = "\n\n"


def _build_partitioner(passage_texts: list[str]) -> Any:
    from context_cite.context_partitioner import BaseContextPartitioner

    class PassagePartitioner(BaseContextPartitioner):  # type: ignore[misc]
        """Sources are whole passages; ablation drops passages, mirroring our LOO unit."""

        def __init__(self, parts: list[str]) -> None:
            super().__init__(_SEPARATOR.join(parts))
            self.parts = parts

        @property
        def num_sources(self) -> int:
            return len(self.parts)

        def split_context(self) -> None:  # parts are fixed at construction
            return None

        def get_source(self, index: int) -> str:
            return self.parts[index]

        def get_context(self, mask: Any = None) -> str:
            if mask is None:
                return str(self.context)
            kept = [p for p, keep in zip(self.parts, mask, strict=True) if keep]
            return _SEPARATOR.join(kept)

    return PassagePartitioner(passage_texts)


@register
class ContextCiteAttribution(AttributionMethod):
    name = "contextcite"
    requires_generator = True
    supports_teacher_forced = False  # attributes its own generated response only

    def attribute(
        self,
        question: str,
        passages: Sequence[Passage],
        target_answer: str,
        model: ModelResources | None,
        method_config: Mapping[str, Any],
    ) -> AttributionResult:
        try:
            from context_cite import ContextCiter
        except ImportError as exc:
            raise UpstreamMissingError(
                "context-cite is not installed — `uv sync --extra contextcite`"
            ) from exc
        if model is None or model.generator is None:
            raise UpstreamMissingError("contextcite needs the generator backend")
        hf_model = getattr(model.generator, "hf_model", None)
        hf_tokenizer = getattr(model.generator, "hf_tokenizer", None)
        if hf_model is None or hf_tokenizer is None:
            raise ConfigError(
                "contextcite requires the real qwen backend (generation.backend: qwen); "
                "the fake test backend has no transformers model to wrap"
            )

        num_ablations = int(method_config.get("num_ablations", 64))
        passage_texts = [f"{p.title}: {p.text}" for p in passages]
        partitioner = _build_partitioner(passage_texts)

        citer = ContextCiter(
            hf_model,
            hf_tokenizer,
            context=partitioner.context,
            query=question,
            partitioner=partitioner,
            num_ablations=num_ablations,
        )
        scores = np.asarray(citer.get_attributions(as_dataframe=False, verbose=False))
        if scores.shape[0] != len(passages):
            raise UpstreamMissingError(
                f"contextcite returned {scores.shape[0]} scores for {len(passages)} passages"
            )
        own_response = str(citer.response)
        mismatch = normalize_answer(own_response) != normalize_answer(target_answer)
        return AttributionResult(
            raw_scores={p.passage_id: float(s) for p, s in zip(passages, scores, strict=True)},
            metadata={
                "num_ablations": num_ablations,
                "contextcite_response": own_response[:500],
                "pipeline_target_answer": target_answer[:500],
                "response_differs_from_pipeline_target": mismatch,
                "note": (
                    "scores attribute ContextCite's own regeneration under its prompt "
                    "template, not the stored generation-run answer"
                ),
            },
            num_model_calls=num_ablations + 1,
        )
