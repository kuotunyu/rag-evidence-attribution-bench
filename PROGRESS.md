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
| M8 | Docker (CPU, no torch) + CI | ⬜ pending | `docker compose up` → GET /health |
| M9 | Colab notebooks + export / import-results / status | ⬜ pending | `uv run pytest` (full CPU gate) |
| M10 | ContextCite adapter attempt (4h stop-loss) | ⬜ pending | — |
| M11 | ARC-JSD legacy env + experimental native method | ⬜ pending | — |
| M12 | docs finalization + final commit | ⬜ pending | — |

## Acceptance checklist (from spec)

| Item | State | Flips when |
|------|-------|-----------|
| 20-question smoke end-to-end success | ⏳ pending-Colab | user runs `notebooks/00_colab_smoke.ipynb`, results imported via `import-results` |
| BM25 + dense retrieval working | ⬜ pending-local | M3 real CPU runs |
| ≥ 3 attribution methods | ⬜ pending-local (code + mocked tests) | M5; real numbers pending-Colab |
| Teacher-forced vs generated-correct reported separately | ⬜ pending-local | M6 (structure) + Colab run (numbers) |
| Quality / latency / VRAM comparison | ⏳ pending-Colab | Colab smoke import |
| Docker precomputed explorer boots | ⬜ pending-local | M8 real `docker compose up` + /health |
| All README numbers generated from results/derived/summary.json | ⬜ pending-local | M6 report stage (PENDING block is already machine-generated) |
| No fabricated results | ✅ enforced by design | `execution_kind: mock` gated out of README; PENDING blocks machine-generated |

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
