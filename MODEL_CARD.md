# MODEL CARD

## Models

| Role | Model | License | Notes |
|---|---|---|---|
| Generator | `Qwen/Qwen3-4B-Instruct-2507` | Apache-2.0 | non-thinking instruct variant; official chat template; transformers ≥ 4.51 (repo locks 5.14.1) |
| Embedder (dense retrieval + embedding attribution) | `Qwen/Qwen3-Embedding-0.6B` | Apache-2.0 | 1024-dim, last-token pooling; instruction prompt applied to QUERIES only (asymmetric use) |
| Reranker (extension only) | `BAAI/bge-reranker-v2-m3` revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | Apache-2.0 | multilingual 0.6B cross-encoder; raw scalar classification logit; tokenizer pinned to the same revision |

## Controlled reranking extension

The original benchmark does not use a reranker. The isolated extension under
`results/reranking/` compares BM25, dense, hybrid RRF, and hybrid RRF + cross-encoder
while holding every downstream setting fixed. Its locked protocol is
[PREREGISTRATION_RERANKING.md](PREREGISTRATION_RERANKING.md). Paired uncertainty,
per-question transfer taxonomy, subgroup descriptions, and cache-cost decomposition are
locked separately in
[SECONDARY_ANALYSIS_RERANKING.md](SECONDARY_ANALYSIS_RERANKING.md); they cannot replace
the primary decision rule.

- Adapter: in-repo `RerankerAdapter`; the initial implementation uses Transformers
  `AutoTokenizer` and `AutoModelForSequenceClassification` with `trust_remote_code=False`.
- Pair rendering: `(question, title + "\n" + passage)`.
- Candidate-k: 10 (the complete fixed Hotpot distractor candidate set); final context-k: 5.
- Max pair length 512, batch size 16, exact revision, raw logits, hybrid-rank tie-break.
- Smoke: CPU/float32. Dev and locked eval: CUDA/float16.
- Every run records effective dtype/device, tokenizer class, model load time, peak VRAM,
  pair throughput, and cache hit/miss counts.
- This repository does not import Longcare RAG runtime code or model objects.

## Decoding (benchmark mode)

- Deterministic greedy: `do_sample=False`, `num_beams=1`, fresh `GenerationConfig`
  (Qwen's shipped sampling defaults are fully replaced), `max_new_tokens: 256`.
- The prompt (versioned `v1`, hash recorded in every run manifest) instructs: short
  answer, inline citations `[P#]`, and the exact abstention string
  `INSUFFICIENT EVIDENCE` when the passages do not support an answer.
- dtype auto: BF16 where supported, FP16 otherwise (Colab T4 has no BF16); 4-bit NF4
  only as an automatic OOM fallback. The dtype/quantization actually used is recorded
  in every run manifest and shown in every report table.

## Attribution methods shipped

| method | kind | modes | cost/sample | notes |
|---|---|---|---|---|
| `citations` | model self-report | B only | 0 model calls | parses `[P#]` from the generated answer; gold answers performed no citation act, hence no mode A |
| `embedding` | similarity | A + B | 0 generator calls | cosine(Qwen3-embed(question + answer), passage); shares the passage-vector cache with dense retrieval |
| `leave_one_out` | **causal (primary baseline)** | A + B | 11 teacher-forced passes | Δ sum-logprob of the target when each passage is removed; aliases stable under ablation; citation markers stripped from targets |
| `arc_jsd` | causal, **EXPERIMENTAL** | A + B | 11 passes (full distributions) | mean positional JSD between answer-token distributions, full vs minus-passage (arXiv:2505.16415); unvalidated against the official Qwen2.5 implementation — see `legacy/arc_jsd/` |
| `contextcite` | surrogate model (optional extra) | B only | ~65 generations | MadryLab context-cite with a passage-level partitioner, reusing the loaded model; **attributes its own regeneration** under its own prompt template, not the stored answer (mismatch flagged in metadata) |
| `control_random` / `control_retrieval` / `control_lexical` / `control_length` / `control_shuffled` | controls | A + B | 0 | flow through the identical pipeline, faithfulness passes included |

Faithfulness (sufficiency ↓ / comprehensiveness ↑, ERASER conventions on mean per-token
teacher-forced logprob, k=2) is computed for every method inside the `attribute` stage
(GPU-resident) and stored as raw numbers; `evaluate`/`report` are pure CPU arithmetic.

## Evaluation modes

- **A (teacher-forced)**: target = official gold answer; all samples.
- **B (generated)**: target = the model's own answer; all non-abstained samples are
  attributed (faithfulness is ground-truth-free), but agreement-with-supporting-facts
  metrics aggregate ONLY over the correct subset (EM==1 by default) — supporting facts
  are never treated as causal ground truth for a wrong answer. Subset sizes are always
  reported.

## Determinism policy

Promised: fixed versioned prompts, greedy decoding, batch-1 sorted-order scoring, pinned
library versions (lock + notebook install cell), full environment manifest per run.
Not promised (documented): bit-exact logits across GPU architectures/dtypes — cuBLAS
kernel selection and FP16 vs BF16 rounding differ by hardware. Cross-dtype numbers are
never merged into one table row without a dtype label; leave-one-out and ARC-JSD deltas
are always within-run comparisons.

## VRAM / latency measurement

- Peak VRAM: `torch.cuda.max_memory_allocated` — per-stage reset, per-sample running
  snapshots; allocator-level (≈0.5–1 GB below `nvidia-smi`); ±100–300 MB run-to-run noise
  is normal. CPU runs record `null`, never 0.
- Latency: monotonic clock with CUDA synchronization; generation throughput =
  completion tokens / end-to-end generation seconds.
- Attribution cost: seconds-per-sample and mean model calls per sample, per method.
