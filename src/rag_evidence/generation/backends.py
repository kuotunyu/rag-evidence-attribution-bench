"""Generation backends: the real Qwen wrapper and the deterministic FakeLM.

Both implement GeneratorBackend:
    generate(messages, max_new_tokens)  -> GenerationOutput
    target_logprob(messages, target)    -> TargetScore   (teacher-forced, deterministic)
    info()                              -> effective dtype/quantization/device (manifest)

FakeLM is selected only by config `generation.backend: fake`, which stamps
execution_kind="mock" on the run manifest — mock output can never reach a README.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

from rag_evidence.errors import GpuRequiredError, RagEvidenceError

logger = logging.getLogger(__name__)

# HotpotQA 10-passage prompts are ~1.2-1.8k tokens; anything near this cap is anomalous
# and gets logged loudly (we deliberately do not silently truncate).
PROMPT_TOKEN_WARN_THRESHOLD = 6000


@dataclass(frozen=True)
class GenerationOutput:
    text: str
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class TargetScore:
    sum_logprob: float
    num_target_tokens: int

    @property
    def mean_logprob(self) -> float:
        return self.sum_logprob / self.num_target_tokens if self.num_target_tokens else 0.0

    def to_json(self) -> dict[str, float | int]:
        return {
            "sum_logprob": self.sum_logprob,
            "num_target_tokens": self.num_target_tokens,
            "mean_logprob": self.mean_logprob,
        }


class GeneratorBackend(Protocol):
    model_id: str

    def generate(
        self, messages: list[dict[str, str]], *, max_new_tokens: int
    ) -> GenerationOutput: ...

    def target_logprob(self, messages: list[dict[str, str]], target: str) -> TargetScore: ...

    def target_distributions(self, messages: list[dict[str, str]], target: str) -> Any:
        """Teacher-forced next-token LOG-prob distributions at each target position:
        float32 array [n_target_tokens, vocab] on CPU (numpy). Used by ARC-JSD."""
        ...

    def info(self) -> dict[str, Any]: ...


# --------------------------------------------------------------------------- Qwen (real)


class QwenBackend:
    """Qwen3-4B-Instruct via transformers. GPU stage — runs on Colab in this project.

    Deterministic benchmark decoding: a fresh GenerationConfig with do_sample=False and
    num_beams=1 fully replaces the model's shipped sampling defaults.
    """

    def __init__(
        self,
        model_id: str,
        *,
        device: str,
        dtype: str,
        quantization: str,
        fallback_4bit: bool,
        seed: int,
    ) -> None:
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            raise GpuRequiredError(
                "generation.backend=qwen resolved to device=cuda but no CUDA device is "
                "available — run this stage on Colab (or set runtime.device: cpu explicitly "
                "for a tiny debugging run; a 4B model on CPU is impractically slow)"
            )
        self.model_id = model_id
        self.device = device
        self.requested_dtype = dtype
        self.requested_quantization = quantization
        torch.manual_seed(seed)

        self._torch = torch
        self._tokenizer, self._model, self.effective_quantization = self._load(
            quantization, fallback_4bit
        )
        self._model.eval()
        self.effective_dtype = str(next(self._model.parameters()).dtype).removeprefix("torch.")
        logger.info(
            "loaded %s dtype=%s quant=%s device=%s",
            model_id,
            self.effective_dtype,
            self.effective_quantization,
            self.device,
        )

    def _load(self, quantization: str, fallback_4bit: bool) -> tuple[Any, Any, str]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        torch_dtype = getattr(torch, self.requested_dtype)

        def load(quant: str) -> Any:
            kwargs: dict[str, Any] = {"dtype": torch_dtype}
            if quant == "4bit":
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                )
                kwargs["device_map"] = {"": 0}
                return AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
            model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
            return model.to(self.device)  # type: ignore[arg-type]  # transformers stub quirk

        try:
            return tokenizer, load(quantization), quantization
        except torch.cuda.OutOfMemoryError:
            if quantization == "none" and fallback_4bit and self.device == "cuda":
                logger.warning(
                    "OOM loading %s in %s — falling back to 4-bit NF4 (recorded in run_meta)",
                    self.model_id,
                    self.requested_dtype,
                )
                torch.cuda.empty_cache()
                return tokenizer, load("4bit"), "4bit"
            raise

    def info(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "device": self.device,
            "dtype_requested": self.requested_dtype,
            "dtype_effective": self.effective_dtype,
            "quantization_requested": self.requested_quantization,
            "quantization_effective": self.effective_quantization,
        }

    @property
    def hf_model(self) -> Any:
        """The underlying transformers model — for adapters (ContextCite) that need it."""
        return self._model

    @property
    def hf_tokenizer(self) -> Any:
        return self._tokenizer

    def _render(self, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> Any:
        return self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self._model.device)

    def generate(self, messages: list[dict[str, str]], *, max_new_tokens: int) -> GenerationOutput:
        from transformers import GenerationConfig

        inputs = self._render(messages, add_generation_prompt=True)
        n_prompt = int(inputs["input_ids"].shape[-1])
        if n_prompt > PROMPT_TOKEN_WARN_THRESHOLD:
            logger.warning("anomalously long prompt: %d tokens", n_prompt)
        gen_config = GenerationConfig(  # type: ignore[no-untyped-call]
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
        )
        with self._torch.inference_mode():
            out = self._model.generate(**inputs, generation_config=gen_config)
        completion_ids = out[0, n_prompt:]
        text = self._tokenizer.decode(completion_ids, skip_special_tokens=True)
        return GenerationOutput(
            text=text.strip(),
            prompt_tokens=n_prompt,
            completion_tokens=int(completion_ids.shape[-1]),
        )

    def target_logprob(self, messages: list[dict[str, str]], target: str) -> TargetScore:
        """Teacher-forced sum of token logprobs of `target` after the chat prompt."""
        torch = self._torch
        prompt_ids = self._render(messages, add_generation_prompt=True)["input_ids"]
        target_ids = self._tokenizer(target, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].to(prompt_ids.device)
        if target_ids.shape[-1] == 0:
            raise RagEvidenceError("empty target for teacher-forced scoring")
        input_ids = torch.cat([prompt_ids, target_ids], dim=-1)
        with torch.inference_mode():
            logits = self._model(input_ids).logits
        n_prompt = prompt_ids.shape[-1]
        # logits at position i predict token i+1
        pred_slice = logits[0, n_prompt - 1 : -1, :].float()
        log_probs = torch.log_softmax(pred_slice, dim=-1)
        token_lp = log_probs.gather(-1, target_ids[0].unsqueeze(-1)).squeeze(-1)
        return TargetScore(
            sum_logprob=float(token_lp.sum().item()),
            num_target_tokens=int(target_ids.shape[-1]),
        )

    def target_distributions(self, messages: list[dict[str, str]], target: str) -> Any:
        torch = self._torch
        prompt_ids = self._render(messages, add_generation_prompt=True)["input_ids"]
        target_ids = self._tokenizer(target, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].to(prompt_ids.device)
        if target_ids.shape[-1] == 0:
            raise RagEvidenceError("empty target for distribution scoring")
        input_ids = torch.cat([prompt_ids, target_ids], dim=-1)
        with torch.inference_mode():
            logits = self._model(input_ids).logits
        n_prompt = prompt_ids.shape[-1]
        pred_slice = logits[0, n_prompt - 1 : -1, :].float()
        return torch.log_softmax(pred_slice, dim=-1).cpu().numpy()


# ------------------------------------------------------------------------------- FakeLM


def _stable_unit(material: str) -> float:
    """Deterministic float in [0, 1) from a string — stable across machines/processes."""
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


_QUESTION_RE = re.compile(r"^Question:\s*(.+?)\s*$", re.MULTILINE)
_PASSAGE_LINE_RE = re.compile(r"^\[(P\d+)\]\s*([^:]+):\s*(.*)$", re.MULTILINE)


class FakeLM:
    """Deterministic stub for tests and mocked end-to-end runs. Zero torch.

    generate(): if the question (parsed from the prompt) is in `script`, returns the
    scripted response verbatim; otherwise emits a deterministic pseudo-answer citing the
    first two passage aliases.

    target_logprob(): sum over passages present in the prompt of a per-passage weight —
    2.0 when the passage title appears in the target or the target appears in the passage
    text (an "important" passage), else a tiny stable hash noise in [0, 0.1). Removing an
    important passage therefore drops the score by ~2.0, so leave-one-out recovers
    exactly the important passages. Tests can compute the same weights via
    `FakeLM.passage_weight`.
    """

    model_id = "fake"

    def __init__(self, script: dict[str, str] | None = None, *, fail_marker: str = "") -> None:
        self.script = script or {}
        self.fail_marker = fail_marker  # question substring that triggers a synthetic error

    def info(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "device": "cpu",
            "dtype_requested": "fake",
            "dtype_effective": "fake",
            "quantization_requested": "none",
            "quantization_effective": "none",
        }

    @staticmethod
    def _prompt_text(messages: list[dict[str, str]]) -> str:
        return "\n".join(m["content"] for m in messages)

    @staticmethod
    def passage_weight(title: str, passage_text: str, target: str) -> float:
        t = target.lower().strip()
        if t and (title.lower() in t or t in passage_text.lower()):
            return 2.0
        return 0.1 * _stable_unit(f"w:{title}:{t}")

    def _parse(self, messages: list[dict[str, str]]) -> tuple[str, list[tuple[str, str, str]]]:
        prompt = self._prompt_text(messages)
        qmatch = _QUESTION_RE.search(prompt)
        question = qmatch.group(1) if qmatch else ""
        passages = [(m[0], m[1].strip(), m[2]) for m in _PASSAGE_LINE_RE.findall(prompt)]
        return question, passages

    def generate(self, messages: list[dict[str, str]], *, max_new_tokens: int) -> GenerationOutput:
        question, passages = self._parse(messages)
        if self.fail_marker and self.fail_marker in question:
            raise RuntimeError(f"FakeLM synthetic failure for question: {question[:40]}")
        if question in self.script:
            text = self.script[question]
        else:
            aliases = "".join(f"[{a}]" for a, _t, _x in passages[:2]) or "[P1]"
            text = f"fake-answer-{_stable_unit('gen:' + question):.4f} {aliases}"
        prompt_tokens = len(self._prompt_text(messages).split())
        return GenerationOutput(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(text.split()),
        )

    def target_logprob(self, messages: list[dict[str, str]], target: str) -> TargetScore:
        _question, passages = self._parse(messages)
        base = -10.0 - 5.0 * _stable_unit(f"t:{target}")
        score = base + sum(self.passage_weight(title, text, target) for _a, title, text in passages)
        n_tokens = max(1, len(target.split()))
        return TargetScore(sum_logprob=score, num_target_tokens=n_tokens)

    _FAKE_VOCAB = 32

    def target_distributions(self, messages: list[dict[str, str]], target: str) -> Any:
        """Deterministic fake distributions with the same design property as
        target_logprob: removing an 'important' passage (weight ~2.0) shifts the
        distribution strongly, so ARC-JSD recovers important passages."""
        import numpy as np

        _question, passages = self._parse(messages)
        n_tokens = max(1, len(target.split()))

        def direction(material: str) -> Any:
            seed = int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self._FAKE_VOCAB)
            return v / np.linalg.norm(v)

        rows = []
        for i in range(n_tokens):
            logits = direction(f"base:{target}:{i}")
            for _alias, title, text in passages:
                weight = self.passage_weight(title, text, target)
                logits = logits + weight * direction(f"dir:{title}:{i}")
            logits = logits - logits.max()
            log_probs = logits - np.log(np.exp(logits).sum())
            rows.append(log_probs.astype(np.float32))
        return np.stack(rows)


# ------------------------------------------------------------------------------ factory


def build_backend(
    cfg_generation: Any, *, device: str, dtype: str, seed: int, explicit_cpu: bool = False
) -> GeneratorBackend:
    """cfg_generation is config.GenerationConfig (typed Any to keep this module light).

    explicit_cpu: True only when runtime.device is literally "cpu" in the config. With
    device=auto resolving to cpu we refuse to load the 4B model — that is a
    "you meant to run this on Colab" situation, not a fallback.
    """
    if cfg_generation.backend == "fake":
        return FakeLM()
    if device == "cpu" and not explicit_cpu:
        raise GpuRequiredError(
            "no CUDA device available and runtime.device is 'auto' — the Qwen backend "
            "runs on Colab in this project (or set runtime.device: cpu explicitly for a "
            "tiny debugging run; a 4B model on CPU is impractically slow)"
        )
    return QwenBackend(
        cfg_generation.model_id,
        device=device,
        dtype=dtype,
        quantization=cfg_generation.quantization,
        fallback_4bit=cfg_generation.fallback_4bit,
        seed=seed,
    )
