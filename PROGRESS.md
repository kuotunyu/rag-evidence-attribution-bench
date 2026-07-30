# PROGRESS

進度追蹤檔。每個 milestone 結束時更新；隔一段時間回來，先讀這份檔案再繼續。
(Progress tracker. Updated at every milestone boundary — read this first when resuming.)

## Milestone status

| # | Milestone | Status | Verify with |
|---|-----------|--------|-------------|
| M0 | Repo scaffold (pyproject/uv, CLI skeleton, configs, docs) | ✅ done | `uv sync --extra ml --extra app` + `uv run python -m rag_evidence.cli --help` |
| M1 | storage / config / telemetry + tests | ✅ done (31 tests green) | `uv run pytest tests/test_config.py tests/test_artifacts.py tests/test_runmeta.py` |
| M2 | data layer + REAL HotpotQA prepare + split manifest | ✅ done — real run, 7405 examples, manifest committed | `uv run python -m rag_evidence.cli data prepare --config configs/smoke.yaml` |
| M3 | retrieval (bm25 / dense / hybrid) + REAL CPU runs | ✅ done — real: bm25 all splits, dense+hybrid smoke (20/20 each, 0 failures) | `uv run python -m rag_evidence.cli status --config configs/smoke.yaml` |
| M4 | generation module (QwenBackend + FakeLM, checkpoint/--resume) | ✅ done (mocked tests; real gen = Colab) | `uv run pytest tests/test_resume.py tests/test_citation_parser.py` |
| M5 | attribution (3 methods + 5 controls + faithfulness) | ✅ done (FakeLM-verified; real numbers = Colab) | `uv run pytest tests/test_attribution_runner.py tests/test_sufficiency.py` |
| M6 | evaluate + report (mode A/B, summary.json, README injection) | ✅ done — real retrieval numbers in README; PENDING blocks machine-generated; partial-run guard + mock gating tested | `uv run python -m rag_evidence.cli evaluate --config configs/smoke.yaml` |
| M7 | FastAPI + Gradio explorer + serve | ✅ done (TestClient 10/10; store torch-free) | `uv run pytest tests/test_api.py tests/test_store.py` |
| M8 | Docker (CPU, no torch) + CI | ✅ done — real build + /health probe passed (240 eval samples served) | `docker build -t rag-evidence . && docker run -d -p 8000:8000 rag-evidence` → GET /health |
| M9 | Colab notebooks + export / import-results / status | ✅ done — 3 notebooks nbformat-valid; bundle 1.9MB built; transfer round-trip tested | `uv run python scripts/make_colab_bundle.py` |
| M10 | ContextCite adapter attempt (4h stop-loss) | ✅ SUCCEEDED (~1h) — works vs transformers 5.14; passage-level partitioner; real CPU run verified | `uv run python -m rag_evidence.cli attribute --method contextcite …` (Colab) |
| M11 | ARC-JSD legacy env + experimental native method | ✅ done — native `arc_jsd` (FakeLM-tested, EXPERIMENTAL) + legacy/arc_jsd/ env + official-repro notebook | `uv run pytest tests/test_arc_jsd.py` |
| M12 | docs finalization + final commit | ✅ done | read README.md / DATA_CARD.md / MODEL_CARD.md |
| M13 | Preregistered cross-encoder reranking extension | 🟡 CPU scaffold/smoke done; GPU dev/eval pending because SafeSynth owns RTX 4090 | `python -m rag_evidence.cli reranking evaluate --config configs/reranking/smoke.yaml` |

## Acceptance checklist (from spec)

| Item | State | Notes |
|------|-------|-----------|
| 20-question smoke end-to-end success | ✅ **VERIFIED REAL** (2026-07-25) | Colab T4, full pipeline: generate → 6 methods + 5 controls (2 modes) → contextcite → evaluate → report; imported + re-evaluated locally, EM cross-check passed |
| Locked eval (240) full run | ✅ **VERIFIED REAL** (2026-07-25) | Colab A100, 240/240 every stage, imported + re-evaluated locally, EM cross-check passed — the statistically meaningful headline numbers |
| Dev (60) full run | ✅ **VERIFIED REAL** (2026-07-25) | Colab A100, 60/60 every stage, imported + re-evaluated locally from raw, EM cross-check passed; re-derived numbers matched the earlier eval-zip copy exactly |
| BM25 + dense retrieval working | ✅ verified-local (REAL) | bm25 20/60/240, dense+hybrid smoke 20/20, 0 failures; real numbers in README |
| ≥ 3 attribution methods | ✅ **VERIFIED REAL** | 6 methods (citations/embedding/leave_one_out/contextcite + experimental arc_jsd not run this pass) + 5 controls, real Qwen3-4B numbers now in README |
| Teacher-forced vs generated-correct reported separately | ✅ **VERIFIED REAL** | mode A (n=20) vs mode B (n_correct=10 of 20, 1 abstained) — separate tables in README, subset size reported |
| Quality / latency / VRAM comparison | ✅ **VERIFIED REAL** | EM 0.500, F1 0.636, citation F1 0.786, peak VRAM 12120 MB (bfloat16, T4), 2.5 tok/s — all in README |
| Docker precomputed explorer boots | ✅ verified-local (REAL) | container /health → 240 eval samples; /methods honest-empty for pending stages |
| All README numbers generated from results/derived/summary.json | ✅ verified-local (REAL) | retrieval + generation + attribution tables all injected by `report`; no PENDING blocks left for smoke |
| No fabricated results | ✅ enforced by design + tested | `execution_kind: mock` gated out of README; partial-run guard; sha256 import gate; this real import had zero raw conflicts |

## How to resume

1. `cd rag-evidence-attribution-bench && uv sync --extra ml --extra app`
2. Read "Session log" below for the last stopping point.
3. `uv run pytest -m "not gpu and not slow"` to confirm the tree is green.
4. Continue the first non-✅ milestone above; plan details in
   `C:\Users\USER\.claude\plans\text-repository-steady-dusk.md` (local machine only — not in repo).

## Colab handoff (when M9 is done)

1. Build bundle: `uv run python -m rag_evidence.cli export --config configs/smoke.yaml` (bundle zip incl. prepared data).
2. Upload zip to Google Drive `MyDrive/reab/`.
3. Run `notebooks/00_colab_smoke.ipynb` top to bottom (GPU runtime).
4. Download `results_<run_id>.zip`, then locally:
   `uv run python -m rag_evidence.cli import-results results_<run_id>.zip --config configs/smoke.yaml`
5. `evaluate` + `report` locally → README results flip from PENDING to real tables.

## Session log (append-only)

### 2026-07-18
- Plan approved (13 milestones M0–M12). Key decisions: GPU generation only on Colab (user-executed);
  splits from HotpotQA validation; transformers ≥4.51 main env vs ARC-JSD legacy env (4.43.3).
- M0 started: git init, pyproject (torch-cpu uv index), CLI skeleton (typer, lazy imports),
  configs smoke/dev/full, doc skeletons, this file.
- Hit + resolved: torch 2.13/2.9 CPU builds fail DLL init on this Win10 machine (WinError 1114,
  c10.dll; reproduced with minimal PATH). Fallback: `torch<2.7` pinned **win32-only** (2.6.0 works);
  Linux/Colab unaffected. Recorded in FAILURES.md. Lock: transformers 5.14.1 kept.
- M0+M1 verified: `uv sync` clean, CLI help/version OK, **pytest 31 passed** (artifacts/config/
  runmeta/ids/supporting-facts/splits). Committed. M2 real `data prepare` started (~613MB download).
- M2 real run done: 7405 validation examples, manifest committed (seed 20260718, 320 fingerprints,
  eval locked), idempotent rerun verified. M3 real runs: bm25 20/60/240 all 0-fail; dense smoke
  20/20 on CPU (median ~54s/q — honest CPU number); hybrid_rrf smoke 20/20.
- M4+M5+M6+M7 done, **pytest 82 passed, ruff clean, mypy clean (51 files)**. Real retrieval tables
  live in README via `report`; generation/attribution show machine-generated PENDING.
- Integrity hardening from a live near-miss: evaluate initially aggregated the still-running dense
  run → added partial-run guard (`--allow-partial` + `partial` flags + ⚠ labels in tables).
- samples.jsonl now emitted per split (explorer works without dataset download; CC BY-SA noted).
- M8 real verification: docker build (2 fixes: Docker Desktop engine was off; editable-install
  venv needed --no-editable) → container /health = 240 eval samples, /methods honest-empty for
  generation/attribution. M9: notebooks validated, bundle 1.9MB, export/import sha256 round-trip
  + zip-slip/conflict guards tested. Env-override paths may now be absolute (Colab Drive), YAML
  paths still must be relative; export normalizes arcnames to repo-relative form.
- M10 ContextCite: SUCCESS inside stop-loss — resolves+runs against transformers 5.14.1/torch 2.6;
  adapter reuses the loaded backend model (no double VRAM) + custom passage partitioner; caveat
  documented: it attributes its own regeneration (mode B only, mismatch flag recorded).
- M11 ARC-JSD: native experimental `arc_jsd` method (mean positional JSD, teacher-forced;
  target_distributions on both backends; JSD math unit-tested) + legacy/arc_jsd/ isolated env
  (transformers 4.43.3) with official-repro notebook and validation protocol. EXPERIMENTAL label
  stays until the user validates against the official Qwen2.5 flow on Colab.
- M12: READMEs (en/zh-TW) finalized with real retrieval tables preserved; DATA_CARD carries the
  real manifest fingerprints; MODEL_CARD lists all 6 methods + 5 controls with caveats.
- 2026-07 hotfix: user's first real smoke run hit a Drive-path AssertionError (bundle
  wasn't at the hardcoded `MyDrive/reab/reab_bundle.zip`). Root cause: manual "place a
  file at an exact path" step is inherently error-prone. Fixed in both notebooks:
  bundle upload now uses `google.colab.files.upload()` (native picker, no folder/path
  to get right); 00_colab_smoke drops Drive entirely (short run, browser download at
  the end is already the real transport home); 01_colab_run keeps Drive ONLY for
  checkpoint durability across the long run, which still needs zero manual setup
  (`os.makedirs(..., exist_ok=True)`).
- 2026-07 perf fix (found live, during user's actual smoke run): `run_attribution_stage`
  called `_build_resources` (full Qwen3-4B + Qwen3-Embedding load) once PER MODE inside
  `_run_one_mode`, so every `attribute --method X` invocation reloaded both models twice
  (gold, then generated) even within one process — and since faithfulness is on by
  default, this hit every method including cheap ones (embedding, all 5 controls), not
  just leave_one_out. Fixed: resources/scorer built ONCE per CLI invocation in
  `run_attribution_stage`, passed into `_run_one_mode` for both modes to share. No
  behavior/output change (same records, same manifests) — pure fixed-cost reduction.
  Matters far more for `01_colab_run.ipynb` (240q = 12x the redundant reloads avoided
  vs smoke's 20q). 92 tests still green after the fix; bundle rebuilt (133 files, 2.0MB).
  Did NOT affect the user's already-running Colab session (old code already unzipped
  there) — applies to the next upload/run.
### 2026-07-25 — REAL smoke run complete, acceptance verified
- User ran `00_colab_smoke.ipynb` on Colab T4 (Pro+). Hit two real issues, both fixed live
  and documented above: the Drive-path bundle placement (fixed → files.upload() picker)
  and the double model-reload-per-mode perf issue (fixed, bundle rebuilt). This run itself
  used the OLD pre-perf-fix code (already unzipped before the fix landed), so it was slower
  than a fresh run would be — not a data-quality issue, just wall-clock.
- Export zip `results_smoke_20260724T181501Z.zip` (330KB, 81 files) downloaded to
  `D:\Downloads\` (not the default C:\ Downloads — worth remembering for next time).
  `import-results` succeeded: 0 raw conflicts, all sha256 verified.
- `status` confirms 20/20 real for every stage: retrieve (bm25/dense/hybrid_rrf), generate
  (qwen3-4b), attribute × (citations, embedding, leave_one_out, contextcite, 5 controls) ×
  2 modes. contextcite: 18/20 ok (2 failures/skips — optional method, not a blocker).
- Local `evaluate` re-derivation passed the EM cross-check (stored vs recomputed answer
  correctness agree) — imported data is self-consistent, not corrupted in transit.
- **Headline real numbers** (see README.md for full tables): generation EM 0.500, F1 0.636,
  citation F1 0.786, abstain rate 5%, peak VRAM 12120 MB (bfloat16, T4), 2.5 tok/s.
  Attribution mode A: leave_one_out F1@2=0.675/AUPRC=0.775, embedding F1@2=0.750 — both
  clearly beat control_random/control_shuffled (~0.25) and control_length (~0.03),
  confirming the benchmark discriminates real signal from noise. Mode B (n_correct=10/20):
  leave_one_out F1@2=0.800 (highest), contextcite 0.750.
- **Acceptance: all 8 checklist items now ✅ VERIFIED REAL** — no more pending-Colab items
  for the smoke split. README/README_zh-TW/report.md/summary.json/assets all regenerated
  and committed.
- **Remaining optional work (not blocking, not started this session):** `01_colab_run.ipynb`
  (dev60 + locked eval240 — needs the rebuilt bundle re-uploaded to pick up the perf fix);
  legacy ARC-JSD validation against the official Qwen2.5 implementation (drops the
  `experimental` label on the native `arc_jsd` method).

### 2026-07-25 (later) — preparing the full run (01)
- Derived REAL per-sample costs from the smoke `run_meta.json` files instead of guessing:
  mode A ≈ 122 s/sample (leave_one_out 79.5 dominates), mode B ≈ 75 s/sample
  (leave_one_out 41.1, citations 16.5), generation 7.7 s/sample → **≈ 3.4 min/sample on T4**.
  Projection: dev60 ≈ 3.5 h, eval240 ≈ 13–14 h, total ≈ 17 h on T4; ≈ 5 h on A100.
  → **A100 recommended for 01.** Rejected the obvious speedup (batching the 11 LOO passes):
  it would violate the batch-1 determinism promise in MODEL_CARD.md and make the numbers
  incomparable to the already-published smoke results.
- `01_colab_run.ipynb` intro rewritten with these measured numbers + a note that the dev
  and eval sections write to disjoint paths, so they MAY be run in two parallel Colab
  sessions (only `summary.json` is shared, and it is regenerable locally from raw records).
- `02_report.ipynb` upgraded from smoke-only to re-deriving ALL splits (smoke+dev+eval)
  and made cwd-robust; verified locally that `evaluate` on a split with no generation/
  attribution records yet exits cleanly rather than crashing.
- Bundle rebuilt: 202 files, 2.3 MB, verified to contain dev/eval prepared data, samples,
  committed bm25 runs, both configs, AND the model-reload perf fix. 92 tests green.

### 2026-07-25 (later still) — REAL locked eval (240) + dev (60) imported
- User ran `01_colab_run.ipynb` on Colab A100 (Pro+) end-to-end, both sections, in one
  session. Two export zips were produced (`results_dev_...zip`, `results_eval_...zip`);
  only the **eval** zip actually downloaded (same browser-blocked-download pattern as
  the smoke run) — dev's zip is still on Drive/undownloaded as of this entry.
- **eval (240, locked) — fully imported and independently verified**: `import-results`
  (141 files, 0 raw conflicts) → local `evaluate --config configs/full.yaml` recomputed
  from raw and passed the EM cross-check → `report`. 240/240 real on every stage
  (retrieve×3, generate, attribute × 8 methods × 2 modes — contextcite intentionally
  excluded from 01 per its own time-cost note).
- **dev (60) — real numbers present, NOT independently re-derivable locally yet**: the
  eval zip's copy of `results/derived/summary.json` already carried a fully-merged dev
  entry (written by Colab's own live `evaluate --config configs/dev.yaml` call earlier
  in the same session) — genuine real numbers, not fabricated. But dev's raw per-sample
  generate/attribute JSONL never arrived locally (only the dev zip has them), so
  `evaluate --config configs/dev.yaml` was deliberately NOT run locally this pass — doing
  so with dev's raw retrieval-only present would have overwritten the good imported dev
  entry with a regressed retrieval-only one. Confirmed by direct check:
  `results/raw/dev/{generate,attribute}` do not exist on this machine.
- **Headline real numbers, locked eval (n=240, the statistically meaningful split)**:
  generation EM 0.362, F1 0.496, citation F1 0.801, abstain 20.0%, peak VRAM 8437 MB
  (bfloat16, A100), 19.7 tok/s. Attribution mode A: embedding F1@2=0.812/AUPRC=0.894,
  leave_one_out F1@2=0.727/AUPRC=0.813 — both far above control_random/control_shuffled
  (~0.18–0.21) and control_length (~0.10). Mode B (n_correct=87/240, 48 abstained):
  embedding F1@2=0.816, leave_one_out 0.793, citations 0.753.
- **Cross-split consistency check (real signal, not noise)**: F1@2 for embedding/
  leave_one_out is stable within ~0.05–0.10 across smoke(20)/dev(60)/eval(240), and the
  real-method vs null-control separation (~0.6–0.8 vs ~0.15–0.21) holds at every scale —
  strong evidence the benchmark discriminates genuine causal/similarity signal, not an
  artifact of the small smoke sample. EM/F1 correctly DECREASES with split size
  (smoke 0.500 → dev 0.433 → eval 0.362 EM) — expected: 20-sample smoke accuracy is
  noisy and not representative; eval(240) is the number that matters for reporting.
  Device names confirm real hardware: dev/eval retrieval+generation ran on
  `NVIDIA A100-SXM4-40GB` (~8x the smoke run's T4 throughput, matching the notebook's
  own estimate).
- **dev zip recovered same session**: root cause found — the export cell ends with
  `files.download(zips[-1])`, which downloads only the LAST zip, so dev's was generated
  but never sent to the browser. Recovered by running a one-line `files.download(...)`
  cell against the still-live Colab session (the file would have been lost on runtime
  shutdown, costing a 1 h re-run).
- **dev (60) now fully imported and independently verified**: `import-results` (110 files,
  0 raw conflicts) → all three splits re-derived locally from raw
  (`evaluate` on smoke + dev + full, each passing the EM cross-check) → `report`.
  60/60 real on every stage. **Integrity result worth recording: the locally re-derived
  dev numbers matched the earlier eval-zip-carried copy EXACTLY** (EM 0.433, F1 0.547,
  embedding F1@2 0.817, leave_one_out 0.758, n_correct=26/60) — independent confirmation
  that those numbers were genuine and that the export/import round-trip is lossless.
- **All three splits (smoke 20 / dev 60 / locked eval 240) are now "real AND independently
  re-derived from raw records on this machine."** No split is in a weaker evidentiary
  state than any other.
### 2026-07-25 — pre-transfer hardening (moving to a Win11 + RTX 4090 box via USB)
- **Pre-GitHub audit passed**: 337 tracked files / 19 MB (all text JSONL, largest 1.8 MB);
  no tokens, no `.env` (only `.env.example`), no local absolute paths. The only absolute
  paths in tracked files are Colab's own `/content/drive/…` (in `01_colab_run.ipynb` and
  one test fixture asserting env overrides MAY be absolute) — correct and necessary.
- **Found a real blocker for the 4090 machine, documented rather than silently shipped**:
  `uv sync` on Windows resolves torch from the **pytorch-cpu index** (`[tool.uv.sources]`)
  AND caps it at `<2.7` (the Win10 `WinError 1114` workaround). Both are deliberate for
  CI/Docker/CPU reproducibility, but together they mean a CUDA workstation gets
  `torch 2.6.0+cpu` and cannot use its GPU. Failure mode is loud (`GpuRequiredError`),
  not silent. Written up in the new `TRANSFER.md` with the opt-in command and two honest
  caveats: the `<2.7` cap is a Win10-laptop workaround that may be unnecessary on Win11,
  and **the CUDA path has never been executed by this project** — every GPU number came
  from Colab, so the documented command is a starting point, not a tested procedure.
- **`TRANSFER.md` added**: what to copy (git bundle, not the folder — `.venv`/dataset/
  caches must not travel), setup + verification on the new machine, the CUDA opt-in, and
  the GitHub publish steps incl. the MIT-code / CC-BY-SA-results licensing split.
- **README/README_zh-TW freshness pass**: the results preamble no longer says GPU numbers
  are "pending Colab" (they landed); the Colab section documents the file-picker flow with
  measured times (T4 ~1.5 h smoke, A100 ~4.5 h dev+eval) instead of the stale
  "upload to Drive MyDrive/reab/" instructions.
- **Not re-verified this pass**: the Docker image (Docker Desktop was not running). It was
  verified earlier when `results/` held retrieval only; it now bakes in the full 19 MB of
  results. Nothing suggests it breaks — `ResultsStore` just loads more JSONL — but the
  boot+`/health` probe should be re-run once before publishing.

### 2026-07-25 — 02_report.ipynb verified end-to-end locally
- `02` needs no Colab and no GPU; added `jupyterlab` to the dev group so it is runnable
  from a clean checkout with one command (`uv run jupyter lab notebooks/02_report.ipynb`).
- Hardened the notebook before handing it over: replaced the `%cd` magic inside an
  `if` block (not portable across execution frontends) with plain `os.chdir`, and
  replaced bare `!python` shell-outs with `subprocess` on `sys.executable` so the CLI
  always runs on the same interpreter as the kernel. Failed commands now raise instead
  of silently continuing. The summary cell prints a readable per-split digest
  (generation metrics + best real method vs strongest control + mode-B subset size)
  instead of dumping 4000 chars of raw JSON.
- **Verified by actually executing it**, not by inspection: `jupyter nbconvert --execute`
  into a temp dir → **0 error outputs, 9 figures rendered**, digest correct for all three
  splits. The repo's copy stays output-free (executed to a temp path, never in place).
- **Reproducibility bonus**: that execution re-ran `evaluate` on all three splits and
  `report`; the resulting diff touched ONLY the generated-at timestamp — every metric in
  README/summary.json was bit-identical. Re-deriving from raw records is deterministic.
- **Notebook bug fixed (not just documented):** `01_colab_run.ipynb`'s dev section ran
  `export` but never downloaded, and the eval section downloaded only `zips[-1]` — so a
  single-session run of both sections silently stranded dev's zip on the Colab VM. Fixed:
  the dev cell now downloads its own zip immediately after exporting, and the eval cell
  loops over ALL export zips instead of taking the last one. Both notebooks re-validated.

### 2026-07-29 — preregistered reranking extension, CPU boundary reached
- Fully re-read the project contract/code/tests/results and parsed all 218 committed
  JSON/JSONL artifacts (10,901 objects, zero parse failures). Re-derived smoke/dev/eval
  into a temporary derived directory; the summary matched the committed
  `results/derived/summary.json` exactly after ignoring only `generated_utc`.
- Locked `PREREGISTRATION_RERANKING.md` before implementation or any new eval run
  (SHA-256 `fdef3b33d5a78ac939f0cab1a7523a1a29ed3ef1ee4d8b9705447045fbc2e42f`).
  Candidate-k is 10 because the committed distractor corpus has exactly 10 passages per
  question; final context-k is 5 for every arm.
- Added replaceable `RerankerAdapter`, pinned `BAAI/bge-reranker-v2-m3` model/tokenizer
  revision `953dc6f...`, append-only content cache, resume/config guards, systems telemetry,
  four namespaced downstream arms, per-arm Mode-A context, attribution stability/control
  separation, machine-generated comparison/error/group artifacts, and separate
  smoke/dev/eval configs under `configs/reranking/`.
- Existing 92 tests remained green; extension tests increased the suite to **105 passed**.
  Ruff and strict mypy are green.
- Observed and documented the CP950 editable `.pth` failure caused by this workspace's
  non-ASCII path; used an ASCII temp environment with a non-editable wheel instead.
- SafeSynth (`preflight_supervised_labeler_v18`) and Longcare production evaluation were
  both active on the RTX 4090. No CUDA process was interrupted or started.
- Real CPU/float32 smoke rerank completed: 20/20, 0 failures, 200 cache misses on the cold
  run, 3.05 scored pairs/s, model load 24.2 s, scoring 65.6 s, no VRAM claim. Exploratory
  retrieval-only deltas versus hybrid RRF: nDCG@5 `0.761 → 0.895`, Recall@5
  `0.850 → 0.925`, complete necessary-passage coverage@5 `0.750 → 0.850`. CPU rerank p95
  was 7.37 s and is explicitly not evaluated against the preregistered CUDA cost gate.
- Pending commands and the locked eval order are in `docs/RERANKING_RUNBOOK.md`. No
  reranking dev/eval or downstream Qwen/attribution result has been run.
- Locked a separate secondary analysis plan before any reranking downstream/dev/eval
  result (SHA-256 `4b60cc6f...6f48`). Machine artifacts now include 10,000-replicate
  paired bootstrap intervals, correct-answer-subset transfer taxonomy, type/level
  grouping, and observed cold/warm/mixed cache plus amortized model-load accounting.
- Final CPU publication checks: Ruff format/lint, strict mypy, 105 tests (77% coverage),
  226 JSON/JSONL artifacts parsed (11,146 objects), secret/machine-path scans, and
  original-result integrity all passed. Docker build/probe remains unverified because
  the Docker Desktop daemon was not running; do not run locked eval until that existing
  CI gate can be repeated.
