# Benchmark report

**Retrieval**

| split | method | R@2 | R@5 | MRR | nDCG@10 | p50 ms | n |
|---|---|---|---|---|---|---|---|
| smoke | bm25 (cpu) | 0.650 | 0.725 | 0.852 | 0.832 | 1.0 | 20 |
| smoke | dense (cpu) | 0.625 | 0.850 | 0.874 | 0.849 | 54265.5 | 20 |
| smoke | hybrid_rrf (cpu) | 0.625 | 0.850 | 0.814 | 0.819 | 0.0 | 20 |
| dev | bm25 (cpu) | 0.600 | 0.742 | 0.861 | 0.821 | 1.1 | 60 |
| dev | dense (NVIDIA A100-SXM4-40GB) | 0.683 | 0.842 | 0.928 | 0.880 | 377.5 | 60 |
| dev | hybrid_rrf (NVIDIA A100-SXM4-40GB) | 0.675 | 0.842 | 0.957 | 0.885 | 0.1 | 60 |
| eval | bm25 (cpu) | 0.598 | 0.823 | 0.862 | 0.831 | 1.1 | 240 |
| eval | dense (NVIDIA A100-SXM4-40GB) | 0.694 | 0.885 | 0.930 | 0.888 | 1101.2 | 240 |
| eval | hybrid_rrf (NVIDIA A100-SXM4-40GB) | 0.667 | 0.887 | 0.927 | 0.886 | 0.1 | 240 |

**Generation** (deterministic greedy)

| split | model | EM | F1 | cite-P | cite-R | cite-F1 | abstain | tok/s | peak VRAM MB | n |
|---|---|---|---|---|---|---|---|---|---|---|
| smoke | qwen3-4b (bfloat16) | 0.500 | 0.636 | 0.912 | 0.737 | 0.786 | 0.050 | 2.5 | 12120 | 20 |
| dev | qwen3-4b (bfloat16) | 0.433 | 0.547 | 0.892 | 0.802 | 0.816 | 0.117 | 20.0 | 8197 | 60 |
| eval | qwen3-4b (bfloat16) | 0.362 | 0.496 | 0.940 | 0.745 | 0.801 | 0.200 | 19.7 | 8437 | 240 |

**Attribution — mode A (teacher-forced gold answer), split `smoke`**
| method | F1@2 | AUPRC | nDCG@10 | sufficiency ↓ | comprehensiveness ↑ | s/sample | fail % | n |
|---|---|---|---|---|---|---|---|---|
| embedding | 0.750 | 0.841 | 0.906 | 1.551 | 4.939 | 13.105 | 0.0 | 20 |
| leave_one_out | 0.675 | 0.775 | 0.869 | 0.641 | 5.773 | 79.457 | 0.0 | 20 |
| control_length _(control)_ | 0.025 | 0.249 | 0.450 | 6.551 | 0.026 | 7.551 | 0.0 | 20 |
| control_lexical _(control)_ | 0.650 | 0.744 | 0.850 | 1.563 | 3.914 | 2.592 | 0.0 | 20 |
| control_random _(control)_ | 0.250 | 0.411 | 0.595 | 4.904 | 1.556 | 7.780 | 0.0 | 20 |
| control_retrieval _(control)_ | 0.650 | 0.734 | 0.832 | 1.562 | 4.272 | 4.129 | 0.0 | 20 |
| control_shuffled _(control)_ | 0.250 | 0.427 | 0.592 | 5.038 | 2.125 | 7.233 | 0.0 | 20 |

**Attribution — mode B (generated answer), split `smoke`**
_Agreement metrics over the correct subset only: n_correct=10 of n_total=20 (criterion: em; abstained: 1)._

| method | F1@2 | AUPRC | nDCG@10 | sufficiency ↓ | comprehensiveness ↑ | s/sample | fail % | n |
|---|---|---|---|---|---|---|---|---|
| citations | 0.650 | 0.748 | 0.862 | 1.477 | 6.646 | 16.485 | 0.0 | 10 |
| contextcite | 0.750 | 0.836 | 0.908 | 0.974 | 5.841 | 78.280 | 5.3 | 10 |
| embedding | 0.700 | 0.823 | 0.902 | 2.124 | 5.290 | 2.277 | 0.0 | 10 |
| leave_one_out | 0.800 | 0.877 | 0.938 | 1.055 | 6.371 | 41.120 | 0.0 | 10 |
| control_length _(control)_ | 0.050 | 0.268 | 0.469 | 7.053 | 0.313 | 3.756 | 0.0 | 10 |
| control_lexical _(control)_ | 0.650 | 0.751 | 0.845 | 2.457 | 5.016 | 1.129 | 0.0 | 10 |
| control_random _(control)_ | 0.150 | 0.352 | 0.547 | 5.054 | 1.690 | 3.382 | 0.0 | 10 |
| control_retrieval _(control)_ | 0.600 | 0.665 | 0.765 | 2.226 | 5.275 | 2.097 | 0.0 | 10 |
| control_shuffled _(control)_ | 0.200 | 0.397 | 0.570 | 6.483 | 0.949 | 4.431 | 0.0 | 10 |

**Attribution — mode A (teacher-forced gold answer), split `dev`**
| method | F1@2 | AUPRC | nDCG@10 | sufficiency ↓ | comprehensiveness ↑ | s/sample | fail % | n |
|---|---|---|---|---|---|---|---|---|
| embedding | 0.817 | 0.885 | 0.938 | 0.669 | 5.742 | 0.241 | 0.0 | 60 |
| leave_one_out | 0.758 | 0.823 | 0.899 | -0.359 | 6.637 | 1.050 | 0.0 | 60 |
| control_length _(control)_ | 0.117 | 0.287 | 0.483 | 6.065 | 0.354 | 0.135 | 0.0 | 60 |
| control_lexical _(control)_ | 0.658 | 0.737 | 0.831 | 1.672 | 4.972 | 0.068 | 0.0 | 60 |
| control_random _(control)_ | 0.167 | 0.330 | 0.523 | 5.695 | 1.322 | 0.148 | 0.0 | 60 |
| control_retrieval _(control)_ | 0.600 | 0.707 | 0.821 | 2.315 | 4.006 | 0.096 | 0.0 | 60 |
| control_shuffled _(control)_ | 0.167 | 0.346 | 0.536 | 6.302 | 0.611 | 0.144 | 0.0 | 60 |

**Attribution — mode B (generated answer), split `dev`**
_Agreement metrics over the correct subset only: n_correct=26 of n_total=60 (criterion: em; abstained: 7)._

| method | F1@2 | AUPRC | nDCG@10 | sufficiency ↓ | comprehensiveness ↑ | s/sample | fail % | n |
|---|---|---|---|---|---|---|---|---|
| citations | 0.788 | 0.868 | 0.926 | 1.854 | 6.742 | 0.262 | 0.0 | 26 |
| embedding | 0.808 | 0.890 | 0.944 | 1.982 | 6.971 | 0.086 | 0.0 | 26 |
| leave_one_out | 0.788 | 0.866 | 0.932 | 1.449 | 6.884 | 0.574 | 0.0 | 26 |
| control_length _(control)_ | 0.038 | 0.229 | 0.436 | 7.661 | 0.674 | 0.072 | 0.0 | 26 |
| control_lexical _(control)_ | 0.635 | 0.724 | 0.823 | 2.680 | 5.802 | 0.027 | 0.0 | 26 |
| control_random _(control)_ | 0.173 | 0.343 | 0.536 | 6.786 | 1.400 | 0.080 | 0.0 | 26 |
| control_retrieval _(control)_ | 0.673 | 0.753 | 0.851 | 3.465 | 5.350 | 0.047 | 0.0 | 26 |
| control_shuffled _(control)_ | 0.173 | 0.341 | 0.530 | 6.604 | 0.833 | 0.078 | 0.0 | 26 |

**Attribution — mode A (teacher-forced gold answer), split `eval`**
| method | F1@2 | AUPRC | nDCG@10 | sufficiency ↓ | comprehensiveness ↑ | s/sample | fail % | n |
|---|---|---|---|---|---|---|---|---|
| embedding | 0.812 | 0.894 | 0.940 | 1.604 | 6.249 | 0.254 | 0.0 | 240 |
| leave_one_out | 0.727 | 0.813 | 0.892 | -0.262 | 7.180 | 1.072 | 0.0 | 240 |
| control_length _(control)_ | 0.104 | 0.281 | 0.478 | 7.559 | 0.707 | 0.131 | 0.0 | 240 |
| control_lexical _(control)_ | 0.619 | 0.731 | 0.833 | 2.728 | 5.384 | 0.072 | 0.0 | 240 |
| control_random _(control)_ | 0.206 | 0.374 | 0.560 | 6.259 | 1.731 | 0.143 | 0.0 | 240 |
| control_retrieval _(control)_ | 0.598 | 0.722 | 0.831 | 3.330 | 5.099 | 0.100 | 0.0 | 240 |
| control_shuffled _(control)_ | 0.179 | 0.372 | 0.558 | 6.368 | 1.104 | 0.139 | 0.0 | 240 |

**Attribution — mode B (generated answer), split `eval`**
_Agreement metrics over the correct subset only: n_correct=87 of n_total=240 (criterion: em; abstained: 48)._

| method | F1@2 | AUPRC | nDCG@10 | sufficiency ↓ | comprehensiveness ↑ | s/sample | fail % | n |
|---|---|---|---|---|---|---|---|---|
| citations | 0.753 | 0.844 | 0.915 | 1.582 | 6.727 | 0.257 | 0.0 | 87 |
| embedding | 0.816 | 0.900 | 0.945 | 2.521 | 6.555 | 0.091 | 0.0 | 87 |
| leave_one_out | 0.793 | 0.857 | 0.921 | 0.860 | 7.156 | 0.650 | 0.0 | 87 |
| control_length _(control)_ | 0.075 | 0.262 | 0.462 | 7.765 | 0.939 | 0.081 | 0.0 | 87 |
| control_lexical _(control)_ | 0.649 | 0.758 | 0.851 | 2.720 | 5.842 | 0.035 | 0.0 | 87 |
| control_random _(control)_ | 0.207 | 0.381 | 0.565 | 6.604 | 1.873 | 0.088 | 0.0 | 87 |
| control_retrieval _(control)_ | 0.615 | 0.742 | 0.846 | 3.473 | 5.547 | 0.062 | 0.0 | 87 |
| control_shuffled _(control)_ | 0.172 | 0.363 | 0.551 | 7.057 | 1.252 | 0.083 | 0.0 | 87 |

_Generated by `report` from `results/derived/summary.json` at 2026-07-25T08:30:49Z (package 0.1.0; dataset sha256:aa57b60128d8…). Do not edit by hand._
