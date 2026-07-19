# PROGRESS

進度追蹤檔。每個 milestone 結束時更新；隔一段時間回來，先讀這份檔案再繼續。
(Progress tracker. Updated at every milestone boundary — read this first when resuming.)

## Milestone status

| # | Milestone | Status | Verify with |
|---|-----------|--------|-------------|
| M0 | Repo scaffold (pyproject/uv, CLI skeleton, configs, docs) | ✅ done | `uv sync --extra ml --extra app` + `uv run python -m rag_evidence.cli --help` |
| M1 | storage / config / telemetry + tests | ✅ done (31 tests green) | `uv run pytest tests/test_config.py tests/test_artifacts.py tests/test_runmeta.py` |
| M2 | data layer + REAL HotpotQA prepare + split manifest | 🔄 code done, real run in progress | `uv run python -m rag_evidence.cli data prepare --config configs/smoke.yaml` |
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

## Acceptance checklist (from spec)

| Item | State | Notes |
|------|-------|-----------|
| 20-question smoke end-to-end success | ⏳ **pending-Colab (user action)** | run `notebooks/00_colab_smoke.ipynb`, then `import-results` + `evaluate` + `report` locally |
| BM25 + dense retrieval working | ✅ verified-local (REAL) | bm25 20/60/240, dense+hybrid smoke 20/20, 0 failures; real numbers in README |
| ≥ 3 attribution methods | ✅ code + mocked-e2e verified (6 methods + 5 controls) | real GPU numbers pending-Colab |
| Teacher-forced vs generated-correct reported separately | ✅ verified (mock e2e + subset tests) | numbers pending-Colab |
| Quality / latency / VRAM comparison | ⏳ pending-Colab | tables/figures render automatically once real runs are imported |
| Docker precomputed explorer boots | ✅ verified-local (REAL) | container /health → 240 eval samples; /methods honest-empty for pending stages |
| All README numbers generated from results/derived/summary.json | ✅ verified-local (REAL) | retrieval tables injected by `report`; PENDING blocks machine-generated |
| No fabricated results | ✅ enforced by design + tested | `execution_kind: mock` gated out of README; partial-run guard; sha256 import gate |

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
- **SESSION END STATE: all 13 milestones done. Final gate: 92 tests passed, ruff clean,
  mypy clean. Remaining user actions: (1) rebuild bundle (`uv run python
  scripts/make_colab_bundle.py`) and upload to Drive MyDrive/reab/, (2) run
  00_colab_smoke.ipynb, (3) bring the export zip back for `import-results` — that flips the
  three pending-Colab acceptance items. Optional later: 01_colab_run (dev+locked eval),
  legacy ARC-JSD validation.**
