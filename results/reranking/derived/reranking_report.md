# Controlled reranking extension report

**Controlled reranking extension**

Confirmatory configuration: candidate-k 10 (the full fixed Hotpot distractor candidate set), final context-k 5. Smoke/dev are not formal eval.

| split | arm | nDCG@5 | complete evidence@5 | EM | F1 | citation coverage | rerank p95 ms | retrieval e2e p95 ms | retrieval+generation p95 ms (est.) | pipeline peak VRAM MB |
|---|---|---|---|---|---|---|---|---|---|---|
| smoke | bm25 | 0.724 | 0.550 | 0.350 | 0.509 | 0.500 | — | 1.4 | 3661.7 | 7726.4 |
| smoke | dense | 0.790 | 0.750 | 0.350 | 0.493 | 0.450 | — | 135512.8 | 141259.9 | 8523.3 |
| smoke | hybrid_rrf | 0.760 | 0.750 | 0.400 | 0.559 | 0.575 | — | 135512.9 | 136758.9 | 8519.2 |
| smoke | hybrid_rrf_rerank | 0.895 | 0.850 | 0.500 | 0.633 | 0.475 | 7368.4 | 143017.2 | 143960.6 | 8601.0 |
| dev | bm25 | 0.724 | 0.517 | 0.350 | 0.417 | 0.500 | — | 1.4 | 4772.9 | 8709.7 |
| dev | dense | 0.820 | 0.700 | 0.400 | 0.479 | 0.592 | — | 605.1 | 3945.4 | 8294.3 |
| dev | hybrid_rrf | 0.825 | 0.700 | 0.383 | 0.473 | 0.625 | — | 605.2 | 5863.9 | 8709.7 |
| dev | hybrid_rrf_rerank | 0.954 | 0.967 | 0.500 | 0.597 | 0.675 | 42.9 | 633.2 | 5768.7 | 8332.5 |
| eval | bm25 | 0.762 | 0.679 | 0.346 | 0.454 | 0.496 | — | 1.7 | 5250.4 | 9535.2 |
| eval | dense | 0.843 | 0.779 | 0.346 | 0.462 | 0.525 | — | 2011.5 | 5692.6 | 8476.5 |
| eval | hybrid_rrf | 0.842 | 0.787 | 0.362 | 0.484 | 0.529 | — | 2011.5 | 7705.9 | 8577.9 |
| eval | hybrid_rrf_rerank | 0.938 | 0.917 | 0.375 | 0.509 | 0.562 | 67.7 | 2054.4 | 6429.3 | 8723.5 |

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
- `2_answer_quality`: answer F1 delta: delta=+0.124.
- `3_citation_attribution`: citation coverage delta: delta=+0.050. generated leave-one-out F1@2 delta: delta=+0.078. sufficiency delta (lower is better): delta=+0.094. comprehensiveness delta (higher is better): delta=+0.110.
- `4_retrieval_better_attribution_worse`: 0 per-sample cases detected.
- `5_system_cost`: dev diagnostic rerank p95=42.9 ms; the 250 ms gate is evaluated only on locked-eval CUDA hardware.
- `6_question_types`: See group_analysis.json; group deltas are machine-generated.
- `7_multi_hop_consolidation`: 1 questions lost complete necessary-passage coverage at K=5.
- `evidentiary_status`: dev artifacts are complete but non-eval; not formal eval.

### Paired bootstrap (secondary, descriptive)

| metric | subset | n pairs | mean delta | 95% CI | favorable direction |
|---|---|---:|---:|---:|---|
| retrieval_ndcg_at_5 | all_valid_paired_questions | 60 | 0.129 | [0.079, 0.183] | higher |
| complete_evidence_at_5 | all_valid_paired_questions | 60 | 0.267 | [0.150, 0.383] | higher |
| answer_em | all_valid_paired_questions | 60 | 0.117 | [0.017, 0.217] | higher |
| answer_f1 | all_valid_paired_questions | 60 | 0.124 | [0.020, 0.234] | higher |
| citation_coverage | all_successfully_processed_paired_questions | 60 | 0.050 | [-0.050, 0.150] | higher |
| leave_one_out_sufficiency | all_paired_attributed_answers | 43 | -0.005 | [-0.125, 0.106] | lower |
| leave_one_out_comprehensiveness | all_paired_attributed_answers | 43 | -0.443 | [-1.482, 0.315] | higher |
| citations_attribution_f1_at_2 | both_arms_correct_answer_subset | 21 | 0.048 | [0.000, 0.119] | higher |
| embedding_attribution_f1_at_2 | both_arms_correct_answer_subset | 21 | 0.000 | [0.000, 0.000] | higher |
| leave_one_out_attribution_f1_at_2 | both_arms_correct_answer_subset | 21 | 0.095 | [0.024, 0.190] | higher |

### Primary transfer taxonomy

```json
{
  "pending": 39,
  "retrieval_down_attribution_unchanged": 2,
  "retrieval_unchanged_attribution_unchanged": 9,
  "retrieval_unchanged_attribution_up": 1,
  "retrieval_up_attribution_unchanged": 6,
  "retrieval_up_attribution_up": 3
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
    "generation_latency_ms": {
      "mean": 1085.689566666667,
      "p50": 504.443,
      "p95": 5135.504
    },
    "estimated_retrieval_plus_generation_latency_ms": {
      "mean": 1512.0491333333334,
      "p50": 912.355,
      "p95": 5768.741
    },
    "retrieval_peak_vram_mb": 1204.73779296875,
    "generation_peak_vram_mb": 8332.52587890625,
    "pipeline_peak_vram_mb": 8332.52587890625,
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
    "estimated_retrieval_plus_generation_latency_ms": {
      "mean": -627.1743333333334,
      "p50": -397.75700000000006,
      "p95": -95.1260000000002
    },
    "pipeline_peak_vram_mb": -377.1552734375,
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
  "guardrail_gate": true,
  "cost_gate": null,
  "retain_as_default": null,
  "publishable": false,
  "cost_summary": "dev diagnostic rerank p95=42.9 ms; the 250 ms gate is evaluated only on locked-eval CUDA hardware."
}
```

## Research questions — eval

- `1_retrieval`: nDCG@5 hybrid_rrf_rerank minus hybrid_rrf: delta=+0.096.
- `2_answer_quality`: answer F1 delta: delta=+0.025.
- `3_citation_attribution`: citation coverage delta: delta=+0.033. generated leave-one-out F1@2 delta: delta=+0.011. sufficiency delta (lower is better): delta=+0.068. comprehensiveness delta (higher is better): delta=-0.166.
- `4_retrieval_better_attribution_worse`: 5 per-sample cases detected.
- `5_system_cost`: eval rerank p95=67.7 ms; preregistered 250 ms gate passed.
- `6_question_types`: See group_analysis.json; group deltas are machine-generated.
- `7_multi_hop_consolidation`: 8 questions lost complete necessary-passage coverage at K=5.
- `evidentiary_status`: Formal locked eval.

### Paired bootstrap (secondary, descriptive)

| metric | subset | n pairs | mean delta | 95% CI | favorable direction |
|---|---|---:|---:|---:|---|
| retrieval_ndcg_at_5 | all_valid_paired_questions | 240 | 0.096 | [0.072, 0.120] | higher |
| complete_evidence_at_5 | all_valid_paired_questions | 240 | 0.129 | [0.075, 0.183] | higher |
| answer_em | all_valid_paired_questions | 240 | 0.013 | [-0.025, 0.054] | higher |
| answer_f1 | all_valid_paired_questions | 240 | 0.025 | [-0.019, 0.069] | higher |
| citation_coverage | all_successfully_processed_paired_questions | 240 | 0.033 | [-0.008, 0.077] | higher |
| leave_one_out_sufficiency | all_paired_attributed_answers | 161 | 0.097 | [-0.265, 0.481] | lower |
| leave_one_out_comprehensiveness | all_paired_attributed_answers | 161 | -0.211 | [-0.845, 0.406] | higher |
| citations_attribution_f1_at_2 | both_arms_correct_answer_subset | 77 | 0.039 | [0.006, 0.078] | higher |
| embedding_attribution_f1_at_2 | both_arms_correct_answer_subset | 77 | 0.013 | [-0.013, 0.039] | higher |
| leave_one_out_attribution_f1_at_2 | both_arms_correct_answer_subset | 77 | -0.006 | [-0.058, 0.045] | higher |

### Primary transfer taxonomy

```json
{
  "pending": 163,
  "retrieval_down_attribution_down": 1,
  "retrieval_down_attribution_unchanged": 2,
  "retrieval_unchanged_attribution_down": 4,
  "retrieval_unchanged_attribution_unchanged": 30,
  "retrieval_unchanged_attribution_up": 6,
  "retrieval_up_attribution_down": 4,
  "retrieval_up_attribution_unchanged": 28,
  "retrieval_up_attribution_up": 2
}
```

### Systems decomposition

```json
{
  "hybrid_rrf_rerank": {
    "retrieval_end_to_end_latency_ms": {
      "mean": 1167.788804166667,
      "p50": 1161.234,
      "p95": 2054.373
    },
    "generation_latency_ms": {
      "mean": 1361.1663458333326,
      "p50": 832.336,
      "p95": 4374.939
    },
    "estimated_retrieval_plus_generation_latency_ms": {
      "mean": 2528.95515,
      "p50": 1993.57,
      "p95": 6429.312
    },
    "retrieval_peak_vram_mb": 1203.82568359375,
    "generation_peak_vram_mb": 8723.48779296875,
    "pipeline_peak_vram_mb": 8723.48779296875,
    "rerank_latency_ms": {
      "mean": 46.80237500000001,
      "p50": 42.659,
      "p95": 67.744
    },
    "rerank_scoring_latency_ms": {
      "mean": 35.33453333333335,
      "p50": 30.704,
      "p95": 53.547
    },
    "rerank_model_load_s": 3.464391,
    "rerank_model_load_amortized_ms_per_question": 14.434962500000001,
    "cold_rerank_latency_with_amortized_load_ms": {
      "mean": 61.23733750000001,
      "p50": 57.0939625,
      "p95": 82.1789625
    },
    "rerank_throughput_pairs_per_s": 283.0092562894092,
    "cache": {
      "hits": 0,
      "misses": 2400,
      "hit_rate": 0.0
    },
    "cache_observations": {
      "cold": {
        "n": 240,
        "latency_ms": {
          "mean": 46.80237500000001,
          "p50": 42.659,
          "p95": 67.744
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
      "mean": 46.80237500000001,
      "p50": 42.659,
      "p95": 67.744
    },
    "estimated_retrieval_plus_generation_latency_ms": {
      "mean": -346.11131250000017,
      "p50": -47.096000000000004,
      "p95": -1276.5389999999998
    },
    "pipeline_peak_vram_mb": 145.6279296875,
    "local_api_fee": 0.0,
    "currency": null,
    "note": "Local-model compute cost; no monetary API charge is fabricated."
  }
}
```

### Decision

```json
{
  "split": "eval",
  "formal": true,
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
  "guardrail_gate": true,
  "cost_gate": true,
  "retain_as_default": true,
  "publishable": true,
  "cost_summary": "eval rerank p95=67.7 ms; preregistered 250 ms gate passed."
}
```
