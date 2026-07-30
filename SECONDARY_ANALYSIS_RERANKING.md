# Locked secondary analysis plan: reranking transfer diagnostics

Status: **LOCKED BEFORE ANY RERANKING DOWNSTREAM, DEV, OR EVAL RESULT**

- Analysis-plan ID: `reranking-transfer-secondary-v1-2026-07-29`
- Locked on: 2026-07-29 (Asia/Taipei)
- The retrieval-only CPU smoke result was visible before this secondary plan was written.
- No reranking generation, attribution, dev, or locked-eval result was visible.
- This plan supplements `PREREGISTRATION_RERANKING.md`. It does not change its primary
  endpoints, tolerances, decision gate, or one-shot locked-eval policy.

These analyses are descriptive/secondary. They may explain a confirmatory result but may
not replace a failed primary endpoint or justify changing the reranker after eval.

## 1. Paired uncertainty

The experimental unit is a question. Every comparison is
`hybrid_rrf_rerank - hybrid_rrf` on aligned question IDs.

- Resampling: paired non-parametric bootstrap over question IDs.
- Replicates: 10,000.
- Interval: two-sided 95% percentile interval.
- Seed: deterministic SHA-256 derivation from the existing global seed and metric name.
- Missingness: a question is included only when both arms have a valid value for that
  metric; the paired sample count is always reported.
- Interpretation: intervals are descriptive. No multiple-comparison-adjusted hypothesis
  claim is added after seeing the results.

Metrics:

- retrieval nDCG@5 and complete necessary-passage coverage@5;
- answer EM and token F1;
- citation coverage over all successfully processed questions;
- generated-answer attribution F1@2 for citations, embedding, and leave-one-out;
- leave-one-out sufficiency (negative delta is favorable) and comprehensiveness
  (positive delta is favorable).

## 2. Per-question transfer taxonomy

Direction uses an absolute numerical tolerance of `1e-12`:

- delta greater than tolerance: `up`;
- delta less than negative tolerance: `down`;
- otherwise: `unchanged`;
- unavailable paired values: `pending`.

For retrieval nDCG@5 crossed with generated leave-one-out F1@2, each question is assigned
one of:

- `retrieval_up_attribution_up`;
- `retrieval_up_attribution_down`;
- `retrieval_down_attribution_up`;
- `retrieval_down_attribution_down`;
- the corresponding `*_unchanged` combinations;
- `pending`.

The same directional fields are retained for citation and embedding attribution, but
leave-one-out is the primary transfer taxonomy because it is the registered causal
baseline. Counts never substitute for the preregistered aggregate metrics.

## 3. Consolidation and subgroup descriptions

For HotpotQA type and difficulty level, report:

- paired sample count;
- mean retrieval, answer, citation, and attribution deltas;
- transfer-taxonomy counts;
- complete-evidence gains and losses at context K=5.

Subgroups remain descriptive: no minimum-performing subgroup is used as a new decision
gate, and no configuration may be selected from subgroup results.

## 4. Systems accounting

Do not mix model loading, scoring, cache reuse, and downstream generation:

- model-load latency is reported once per process and amortized per evaluated question;
- `cold` observations have one or more cache misses;
- `warm` observations have cache hits and zero misses;
- mixed-hit observations are reported separately;
- scoring throughput uses newly scored pairs only;
- retrieval-plus-generation latency is explicitly an estimate formed from aligned stage
  measurements;
- warm latency remains null unless it was actually observed;
- local-model cost is latency/VRAM/throughput, with no fabricated currency amount.

CPU smoke systems numbers are diagnostics. Only the declared locked-eval CUDA hardware
is evaluated against the preregistered latency and VRAM gate.

## 5. Reporting discipline

The machine-readable comparison must include this file's SHA-256, bootstrap settings,
paired counts, interval endpoints, transfer-taxonomy counts, and systems observation
status. README and Markdown tables remain generated from artifacts. Negative, null, and
mixed results must remain visible.
