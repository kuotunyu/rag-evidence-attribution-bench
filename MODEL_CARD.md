# MODEL CARD

> Skeleton — completed at milestone M12.

## Models

| Role | Model | License | Notes |
|---|---|---|---|
| Generator | `Qwen/Qwen3-4B-Instruct-2507` | Apache-2.0 | non-thinking instruct variant; official chat template; transformers ≥ 4.51 |
| Embedder (dense retrieval + embedding attribution) | `Qwen/Qwen3-Embedding-0.6B` | Apache-2.0 | 1024-dim, last-token pooling, instruction prefix on queries only |

## Decoding (benchmark mode)

- Deterministic greedy: `do_sample=False`, `num_beams=1`; Qwen's shipped sampling defaults
  explicitly removed from the GenerationConfig.
- Answers must cite passage aliases (`[P1]`…); abstaining is allowed when evidence is
  insufficient.
- dtype: auto — BF16 where supported, FP16 otherwise (e.g. Colab T4); 4-bit NF4 only as an
  automatic OOM fallback. The dtype actually used is recorded in every run manifest and
  printed in every report table.

## Determinism policy

Promised: fixed versioned prompts, greedy decoding, batch-1 sorted-order scoring, pinned
library versions, full environment manifest per run.
Not promised (and documented): bit-exact logits across GPU architectures/dtypes — cuBLAS
kernel selection and FP16 vs BF16 rounding differ by hardware. Cross-dtype numbers are never
merged into a single table row without a dtype label; leave-one-out deltas are always
within-run comparisons.

## VRAM / latency measurement _(numbers pending real runs)_

- Peak VRAM: `torch.cuda.max_memory_allocated` (per-stage reset, per-sample snapshots);
  allocator-level, ~0.5–1 GB below `nvidia-smi`. CPU runs record `null`, never 0.
- Latency: monotonic clock with CUDA synchronization; generation throughput =
  output tokens / end-to-end generation seconds.
