# Controlled reranking extension runbook

Read and keep [PREREGISTRATION_RERANKING.md](../PREREGISTRATION_RERANKING.md) and
[SECONDARY_ANALYSIS_RERANKING.md](../SECONDARY_ANALYSIS_RERANKING.md) unchanged before
running anything. The runner verifies both SHA-256 values and refuses a drifted design.
The secondary plan adds descriptive uncertainty/transfer diagnostics but never changes
the confirmatory gate.

## Completion status

The preregistered 60-question dev run and the single 240-question formal locked eval were
completed on 2026-07-30. The eval has 65 completed run metadata files and zero failures.
Do not start a second formal eval or change the locked configuration in response to the
result. CPU-only re-derivation with `reranking evaluate` and `reranking report` is allowed.
See `results/reranking/derived/eval/` for the machine-readable formal artifacts.

## Output isolation

- Existing benchmark (read-only input): `results/raw/{split}/retrieve/{method}`
- Extension raw checkpoints: `results/reranking/raw/{split}/`
- Extension derived artifacts: `results/reranking/derived/{split}/`
- Reranker cache (gitignored): `results/reranking/raw/cache/reranker/`

The extension reuses the already verified BM25/dense/hybrid RRF rankings as first-stage
inputs. It does not recalculate or overwrite them.

## 1. CPU smoke: reranker only

This is safe while another repository owns the RTX 4090. It uses the smoke config's
explicit CPU/float32 reranker and processes 20 × 10 query-passage pairs.

```powershell
python -m rag_evidence.cli retrieve `
  --method hybrid_rrf_rerank `
  --config configs/reranking/smoke.yaml `
  --resume

python -m rag_evidence.cli reranking evaluate --config configs/reranking/smoke.yaml
python -m rag_evidence.cli reranking report   --config configs/reranking/smoke.yaml
```

The resulting table is retrieval-only until the downstream GPU arms are run. Missing
values are emitted as `—`; they are never hand-filled.

## 2. GPU ownership check

Before any CUDA command:

```powershell
nvidia-smi
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'SafeSynth|run_eval.py|longcare' } |
  Select-Object ProcessId, Name, CommandLine
```

If SafeSynth or another declared owner is running, stop here. Do not kill, pause, or
modify that process. This repository's Windows lock installs CPU-only torch by default;
follow [TRANSFER.md](../TRANSFER.md) to opt into CUDA only after the GPU is free.

## 3. Dev retrieval

The dev/eval configs lock the reranker to CUDA/float16, revision
`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, max length 512, and batch size 16.

```powershell
python -m rag_evidence.cli retrieve `
  --method hybrid_rrf_rerank `
  --config configs/reranking/dev.yaml `
  --resume
```

## 4. Four fair downstream arms

Run the same commands for `smoke` first, then `dev`. The published full benchmark used
these attribution methods on dev/eval: citations, embedding, leave-one-out, and five
controls. `control_shuffled` must run after `leave_one_out`.

```powershell
$cfg = "configs/reranking/dev.yaml"
$arms = @("bm25", "dense", "hybrid_rrf", "hybrid_rrf_rerank")
$methods = @(
  "leave_one_out",
  "embedding",
  "citations",
  "control_random",
  "control_retrieval",
  "control_lexical",
  "control_length",
  "control_shuffled"
)

foreach ($arm in $arms) {
  python -m rag_evidence.cli reranking generate `
    --arm $arm --config $cfg --resume
  foreach ($method in $methods) {
    python -m rag_evidence.cli reranking attribute `
      --arm $arm --method $method --config $cfg --resume
  }
}

python -m rag_evidence.cli reranking evaluate --config $cfg
python -m rag_evidence.cli reranking report   --config $cfg
```

Do not tune on the output. Dev verifies the preregistered fixed configuration; it does
not select candidate-k or context-k.
If an attribution process writes failure records because of an infrastructure error,
never delete or edit them. Retry only failed keys in a fresh process with both flags:

```powershell
python -m rag_evidence.cli reranking attribute `
  --arm <arm> --method <method> --mode <mode> --config $cfg `
  --resume --retry-failures
```

The evaluator preserves attempt/failure counters and uses the final record per question.
`--retry-failures` without `--resume` is rejected.

## 5. Locked eval — once

Run only after smoke/dev, unit tests, Ruff, mypy, and publication checks pass. The eval
commands are identical to section 4 with:

```powershell
$cfg = "configs/reranking/eval.yaml"
```

First create the eval rerank retrieval run:

```powershell
python -m rag_evidence.cli retrieve `
  --method hybrid_rrf_rerank `
  --config configs/reranking/eval.yaml `
  --resume
```

Then run the four-arm loop once, followed by `reranking evaluate` and `reranking report`.
Resume is allowed under the same scientific hash; changing any locked field is refused.

## 6. Machine-readable deliverables

For each completed split:

- `reranking_comparison.json`: all arm metrics, deltas, systems cost, stability, control
  separation, paired bootstrap intervals, transfer-taxonomy counts, seven
  research-question answers, and the preregistered retain/publish decision;
- `reranking_error_analysis.jsonl`: per-question retrieval, generation, citation,
  attribution, paradox, and consolidation-loss flags;
- `reranking_group_analysis.json`: grouped deltas by HotpotQA type and difficulty level.

The combined `results/reranking/derived/reranking_report.md` and both README extension
blocks are generated from those files.
