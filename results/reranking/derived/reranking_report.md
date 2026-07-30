# Controlled reranking extension report

**Controlled reranking extension**

Confirmatory configuration: candidate-k 10 (the full fixed Hotpot distractor candidate set), final context-k 5. Smoke/dev are not formal eval.

| split | arm | nDCG@5 | complete evidence@5 | EM | F1 | citation coverage | rerank p95 ms | retrieval e2e p95 ms | retrieval+generation p95 ms (est.) | pipeline peak VRAM MB |
|---|---|---|---|---|---|---|---|---|---|---|
| smoke | bm25 | 0.724 | 0.550 | 0.350 | 0.509 | 0.500 | — | 1.4 | 3661.7 | 7726.4 |
| smoke | dense | 0.790 | 0.750 | 0.350 | 0.493 | 0.450 | — | 135512.8 | 141259.9 | 8523.3 |
| smoke | hybrid_rrf | 0.760 | 0.750 | 0.400 | 0.559 | 0.575 | — | 135512.9 | 136758.9 | 8519.2 |
| smoke | hybrid_rrf_rerank | 0.895 | 0.850 | 0.500 | 0.633 | 0.475 | 7368.4 | 143017.2 | 143960.6 | 8601.0 |
| dev | bm25 | 0.724 | 0.517 | — | — | — | — | 1.4 | — | — |
| dev | dense | 0.820 | 0.700 | — | — | — | — | 605.1 | — | — |
| dev | hybrid_rrf | 0.825 | 0.700 | — | — | — | — | 605.2 | — | — |
| dev | hybrid_rrf_rerank | 0.954 | 0.967 | — | — | — | 42.9 | 633.2 | — | 1204.7 |

_All values are generated from `results/reranking/derived/*/reranking_comparison.json`; missing downstream runs remain `—`._

## Research questions — smoke

- `1_retrieval`: nDCG@5 hybrid_rrf_rerank minus hybrid_rrf: delta=+0.134.
- `2_answer_quality`: answer F1 delta: delta=+0.074.
- `3_citation_attribution`: citation coverage delta: delta=-0.100. generated leave-one-out F1@2 delta: delta=-0.125. sufficiency delta (lower is better): delta=+0.291. comprehensiveness delta (higher is better): delta=+1.329.
- `4_retrieval_better_attribution_worse`: 1 per-sample cases detected.
- `5_system_cost`: smoke diagnostic rerank p95=7368.4 ms; the 250 ms gate is evaluated only on locked-eval CUDA hardware.
- `6_question_types`: See group_analysis.json; group deltas are machine-generated.
- `7_multi_hop_consolidation`: 1 questions lost complete necessary-passage coverage at K=5.
- `evidentiary_status`: smoke artifacts are incomplete or non-eval; not formal eval.

### Paired bootstrap (secondary, descriptive)

| metric | subset | n pairs | mean delta | 95% CI | favorable direction |
|---|---|---:|---:|---:|---|
| retrieval_ndcg_at_5 | all_valid_paired_questions | 20 | 0.134 | [0.046, 0.229] | higher |
| complete_evidence_at_5 | all_valid_paired_questions | 20 | 0.100 | [-0.100, 0.300] | higher |
| answer_em | all_valid_paired_questions | 20 | 0.100 | [0.000, 0.250] | higher |
| answer_f1 | all_valid_paired_questions | 20 | 0.074 | [-0.067, 0.232] | higher |
| citation_coverage | all_successfully_processed_paired_questions | 20 | -0.100 | [-0.275, 0.075] | higher |
| leave_one_out_sufficiency | all_paired_attributed_answers | 16 | -0.248 | [-0.872, 0.112] | lower |
| leave_one_out_comprehensiveness | all_paired_attributed_answers | 16 | 0.967 | [-0.599, 2.752] | higher |
| citations_attribution_f1_at_2 | both_arms_correct_answer_subset | 8 | 0.062 | [0.000, 0.188] | higher |
| embedding_attribution_f1_at_2 | both_arms_correct_answer_subset | 8 | 0.062 | [0.000, 0.188] | higher |
| leave_one_out_attribution_f1_at_2 | both_arms_correct_answer_subset | 8 | -0.125 | [-0.375, 0.125] | higher |

### Primary transfer taxonomy

```json
{
  "pending": 12,
  "retrieval_unchanged_attribution_down": 2,
  "retrieval_unchanged_attribution_unchanged": 3,
  "retrieval_unchanged_attribution_up": 1,
  "retrieval_up_attribution_down": 1,
  "retrieval_up_attribution_unchanged": 1
}
```

### Systems decomposition

```json
{
  "hybrid_rrf_rerank": {
    "retrieval_end_to_end_latency_ms": {
      "mean": 72224.10759999999,
      "p50": 56782.344,
      "p95": 143017.222
    },
    "generation_latency_ms": {
      "mean": 749.34805,
      "p50": 448.115,
      "p95": 943.398
    },
    "estimated_retrieval_plus_generation_latency_ms": {
      "mean": 72973.45564999999,
      "p50": 57230.458999999995,
      "p95": 143960.62
    },
    "retrieval_peak_vram_mb": null,
    "generation_peak_vram_mb": 8601.0068359375,
    "pipeline_peak_vram_mb": 8601.0068359375,
    "rerank_latency_ms": {
      "mean": 3300.3174999999997,
      "p50": 2573.036,
      "p95": 7368.442
    },
    "rerank_scoring_latency_ms": {
      "mean": 3277.9778999999994,
      "p50": 2549.302,
      "p95": 7338.564
    },
    "rerank_model_load_s": 24.229469,
    "rerank_model_load_amortized_ms_per_question": 1211.47345,
    "cold_rerank_latency_with_amortized_load_ms": {
      "mean": 4511.79095,
      "p50": 3784.50945,
      "p95": 8579.91545
    },
    "rerank_throughput_pairs_per_s": 3.050661201834217,
    "cache": {
      "hits": 0,
      "misses": 200,
      "hit_rate": 0.0
    },
    "cache_observations": {
      "cold": {
        "n": 20,
        "latency_ms": {
          "mean": 3300.3174999999997,
          "p50": 2573.036,
          "p95": 7368.442
        }
      },
      "warm": {
        "n": 0,
        "latency_ms": null
      },
      "mixed": {
        "n": 0,
        "latency_ms": null
      }
    }
  },
  "incremental_cost_vs_hybrid_rrf": {
    "retrieval_latency_ms": {
      "mean": 3300.3174999999997,
      "p50": 2573.036,
      "p95": 7368.442
    },
    "estimated_retrieval_plus_generation_latency_ms": {
      "mean": 3435.5869999999704,
      "p50": 2531.9069999999992,
      "p95": 7201.727999999974
    },
    "pipeline_peak_vram_mb": 81.7724609375,
    "local_api_fee": 0.0,
    "currency": null,
    "note": "Local-model compute cost; no monetary API charge is fabricated."
  }
}
```

### Decision

```json
{
  "split": "smoke",
  "formal": false,
  "complete_retrieval": true,
  "complete_generation": true,
  "complete_attribution": true,
  "attribution_complete_by_mode": {
    "gold": true,
    "generated": true
  },
  "required_attribution_methods": {
    "gold": [
      "control_length",
      "control_lexical",
      "control_random",
      "control_retrieval",
      "control_shuffled",
      "embedding",
      "leave_one_out"
    ],
    "generated": [
      "citations",
      "control_length",
      "control_lexical",
      "control_random",
      "control_retrieval",
      "control_shuffled",
      "embedding",
      "leave_one_out"
    ]
  },
  "quality_gate": true,
  "guardrail_gate": false,
  "cost_gate": null,
  "retain_as_default": null,
  "publishable": false,
  "cost_summary": "smoke diagnostic rerank p95=7368.4 ms; the 250 ms gate is evaluated only on locked-eval CUDA hardware."
}
```

## Research questions — dev

- `1_retrieval`: nDCG@5 hybrid_rrf_rerank minus hybrid_rrf: delta=+0.129.
- `2_answer_quality`: answer F1 delta: pending.
- `3_citation_attribution`: citation coverage delta: pending. generated leave-one-out F1@2 delta: pending. sufficiency delta (lower is better): pending. comprehensiveness delta (higher is better): pending.
- `4_retrieval_better_attribution_worse`: Pending downstream attribution artifacts.
- `5_system_cost`: dev diagnostic rerank p95=42.9 ms; the 250 ms gate is evaluated only on locked-eval CUDA hardware.
- `6_question_types`: See group_analysis.json; group deltas are machine-generated.
- `7_multi_hop_consolidation`: 1 questions lost complete necessary-passage coverage at K=5.
- `evidentiary_status`: dev artifacts are incomplete or non-eval; not formal eval.

### Paired bootstrap (secondary, descriptive)

| metric | subset | n pairs | mean delta | 95% CI | favorable direction |
|---|---|---:|---:|---:|---|
| retrieval_ndcg_at_5 | all_valid_paired_questions | 60 | 0.129 | [0.079, 0.183] | higher |
| complete_evidence_at_5 | all_valid_paired_questions | 60 | 0.267 | [0.150, 0.383] | higher |
| answer_em | all_valid_paired_questions | 0 | — | ?? | higher |
| answer_f1 | all_valid_paired_questions | 0 | — | ?? | higher |
| citation_coverage | all_successfully_processed_paired_questions | 0 | — | ?? | higher |
| leave_one_out_sufficiency | all_paired_attributed_answers | 0 | — | ?? | lower |
| leave_one_out_comprehensiveness | all_paired_attributed_answers | 0 | — | ?? | higher |
| citations_attribution_f1_at_2 | both_arms_correct_answer_subset | 0 | — | ?? | higher |
| embedding_attribution_f1_at_2 | both_arms_correct_answer_subset | 0 | — | ?? | higher |
| leave_one_out_attribution_f1_at_2 | both_arms_correct_answer_subset | 0 | — | ?? | higher |

### Primary transfer taxonomy

```json
{
  "pending": 60
}
```

### Systems decomposition

```json
{
  "hybrid_rrf_rerank": {
    "retrieval_end_to_end_latency_ms": {
      "mean": 426.35956666666664,
      "p50": 407.912,
      "p95": 633.237
    },
    "generation_latency_ms": null,
    "estimated_retrieval_plus_generation_latency_ms": null,
    "retrieval_peak_vram_mb": 1204.73779296875,
    "generation_peak_vram_mb": null,
    "pipeline_peak_vram_mb": 1204.73779296875,
    "rerank_latency_ms": {
      "mean": 34.69110000000001,
      "p50": 27.848,
      "p95": 42.869
    },
    "rerank_scoring_latency_ms": {
      "mean": 26.509233333333334,
      "p50": 20.239,
      "p95": 35.252
    },
    "rerank_model_load_s": 4.887099,
    "rerank_model_load_amortized_ms_per_question": 81.45165,
    "cold_rerank_latency_with_amortized_load_ms": {
      "mean": 116.14275,
      "p50": 109.29965,
      "p95": 124.32065
    },
    "rerank_throughput_pairs_per_s": 377.2270542213594,
    "cache": {
      "hits": 0,
      "misses": 600,
      "hit_rate": 0.0
    },
    "cache_observations": {
      "cold": {
        "n": 60,
        "latency_ms": {
          "mean": 34.69110000000001,
          "p50": 27.848,
          "p95": 42.869
        }
      },
      "warm": {
        "n": 0,
        "latency_ms": null
      },
      "mixed": {
        "n": 0,
        "latency_ms": null
      }
    }
  },
  "incremental_cost_vs_hybrid_rrf": {
    "retrieval_latency_ms": {
      "mean": 34.69110000000001,
      "p50": 27.848,
      "p95": 42.869
    },
    "estimated_retrieval_plus_generation_latency_ms": null,
    "pipeline_peak_vram_mb": null,
    "local_api_fee": 0.0,
    "currency": null,
    "note": "Local-model compute cost; no monetary API charge is fabricated."
  }
}
```

### Decision

```json
{
  "split": "dev",
  "formal": false,
  "complete_retrieval": true,
  "complete_generation": false,
  "complete_attribution": false,
  "attribution_complete_by_mode": {
    "gold": false,
    "generated": false
  },
  "required_attribution_methods": {
    "gold": [
      "control_length",
      "control_lexical",
      "control_random",
      "control_retrieval",
      "control_shuffled",
      "embedding",
      "leave_one_out"
    ],
    "generated": [
      "citations",
      "control_length",
      "control_lexical",
      "control_random",
      "control_retrieval",
      "control_shuffled",
      "embedding",
      "leave_one_out"
    ]
  },
  "quality_gate": null,
  "guardrail_gate": null,
  "cost_gate": null,
  "retain_as_default": null,
  "publishable": false,
  "cost_summary": "dev diagnostic rerank p95=42.9 ms; the 250 ms gate is evaluated only on locked-eval CUDA hardware."
}
```
