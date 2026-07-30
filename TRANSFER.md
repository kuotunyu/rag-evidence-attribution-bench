# Moving this repo to another machine

Written for the actual move this project made: from the Win10 laptop that produced the
results to a Win11 + RTX 4090 workstation, via USB, before publishing to GitHub.

## 1. What to copy — and what NOT to

**Copy the git repository, not the folder.** The working directory contains large
machine-specific junk that must not travel:

| path | size | copy? | why |
|---|---|---|---|
| `.venv/` | GBs | ❌ **no** | machine-specific; recreate with `uv sync` |
| `data/prepared/`, `data/*` (dataset) | ~600 MB | ❌ no | regenerate with `data prepare` (fingerprints guarantee identical content) |
| `reab_bundle.zip` | 2.3 MB | ❌ no | regenerate with `scripts/make_colab_bundle.py` |
| `results/raw/cache/` | varies | ❌ no | embedding/logprob caches, gitignored |
| everything git-tracked | 19 MB | ✅ **yes** | code, tests, docs, **all benchmark results** |

The cleanest transfer is a **git bundle** — one file containing every commit and the full
history, with nothing untracked:

```bash
git bundle create reab.bundle --all
```

Copy `reab.bundle` to the USB drive. On the new machine:

```bash
git clone reab.bundle rag-evidence-attribution-bench
cd rag-evidence-attribution-bench
git remote remove origin      # origin currently points at the .bundle file
```

Verify nothing was lost — the clone should report the same HEAD commit:

```bash
git log --oneline -1
git fsck                      # integrity check of the transferred objects
```

## 2. Setting up on the new machine

```bash
uv sync --extra ml --extra app
uv run pytest -m "not gpu and not slow"     # expect: 106 passed
```

On a Traditional-Chinese Windows locale, an editable install can fail if the repository
path itself contains non-ASCII characters (`UnicodeDecodeError: cp950` while reading the
generated `.pth`). The verified fallback is an ASCII environment path plus a non-editable
install:

```powershell
$env:UV_PROJECT_ENVIRONMENT = "$env:TEMP\reab-env"
uv sync --no-editable --extra ml --extra app
& "$env:TEMP\reab-env\Scripts\python.exe" -m pytest -m "not gpu and not slow"
```

The current suite including the reranking extension has **106 tests**. See
[FAILURES.md](FAILURES.md) for the exact observed failure.

The test suite is fully offline (synthetic fixtures + a deterministic fake model), so a
green run here proves the transfer and the environment, not the network.

Regenerate the dataset-derived files (nothing is downloaded that isn't verified against
the committed fingerprint manifest):

```bash
uv run python -m rag_evidence.cli data prepare --config configs/smoke.yaml
```

Confirm the benchmark results survived the move — this re-derives every metric from the
raw per-sample records and should reproduce the committed tables **exactly** (only the
generated-at timestamp may differ):

```bash
uv run python -m rag_evidence.cli evaluate --config configs/smoke.yaml
uv run python -m rag_evidence.cli evaluate --config configs/dev.yaml
uv run python -m rag_evidence.cli evaluate --config configs/full.yaml
uv run python -m rag_evidence.cli report   --config configs/full.yaml
git diff --stat        # expect: only timestamp lines changed
```

## 3. ⚠ Using a local CUDA GPU (the one real gotcha)

**`uv sync` on Windows installs a CPU-only torch. On a CUDA workstation the GPU will be
unusable until you opt in.** Two separate settings cause this, both deliberate:

1. `pyproject.toml` → `[tool.uv.sources]` pins Windows torch to the **pytorch-cpu index**,
   so CI, Docker, and the CPU dev box all resolve to the same reproducible CPU build.
2. `ml = ["torch>=2.4,<2.7; sys_platform == 'win32'", …]` caps the version because
   **torch ≥ 2.7 fails to load on the original Win10 laptop** (`WinError 1114`, `c10.dll`
   — see [FAILURES.md](FAILURES.md)).

The failure mode is loud, not silent: the Qwen backend raises `GpuRequiredError` rather
than quietly running on CPU.

**To use a local CUDA GPU**, after `uv sync`, install a CUDA build over the CPU one:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
```

Then verify — do not assume:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Two honest caveats, neither verified on the new hardware:

- **The `<2.7` cap is a Win10-laptop workaround, not a known Win11 problem.** It may well
  be unnecessary on Win11. Test it there before assuming you must stay on 2.6; if torch
  ≥ 2.7 imports fine, the cap can be relaxed (and FAILURES.md updated to say so).
- **The base benchmark GPU numbers came from Colab, but the isolated reranking extension
  has now been executed locally on an RTX 4090.** Its locked runtime used
  torch `2.13.0+cu126`, transformers `5.14.1`, and sentence-transformers `5.6.1`.
  Keep the base and extension runtime claims distinct.

Once CUDA torch works, the GPU stages run locally with the same CLI the notebooks call —
no Colab, no bundle, no export/import round-trip:

```bash
uv run python -m rag_evidence.cli generate  --config configs/smoke.yaml --resume
uv run python -m rag_evidence.cli attribute --method leave_one_out --config configs/smoke.yaml --resume
```

For the isolated cross-encoder extension, use
[docs/RERANKING_RUNBOOK.md](docs/RERANKING_RUNBOOK.md). Never start its CUDA dev/eval
stages while SafeSynth or another declared GPU owner is active.
The 60-question GPU dev split and the single 240-question formal locked eval are complete.
Do not create a second formal eval run or tune the locked configuration from its results.
Re-derivation is CPU-only:

```bash
uv run python -m rag_evidence.cli reranking evaluate --config configs/reranking/eval.yaml
uv run python -m rag_evidence.cli reranking report   --config configs/reranking/eval.yaml
```

If an attribution process records infrastructure failures, preserve those rows and retry
only the latest failed keys with `--resume --retry-failures`; evaluation keeps the audit
trail and uses the final attempt per question.

Note that re-running a stage that already has results will refuse to overwrite them
(`--resume` skips completed samples; a changed config refuses to resume at all). To
produce a *new* comparison rather than continuing the published one, change `run_name`
in the config so it writes to its own directory.

## 4. Publishing to GitHub

Pre-flight checks already done on the source machine (re-run if anything changed):

- **Size**: 337 tracked files, 19 MB — well within limits.
- **Secrets**: no tokens, no `.env` (only `.env.example`), no local absolute paths.
  The only absolute paths in tracked files are Colab's own `/content/drive/…`, which are
  correct and necessary.
- **Never committed**: dataset, model weights, caches, `.venv`.

```bash
gh repo create rag-evidence-attribution-bench --public --source=. --remote=origin
git push -u origin main
```

Licensing is already declared and must stay accurate: code is MIT, but files under
`results/` embed HotpotQA passage text and are therefore **CC BY-SA 4.0** with
attribution to Yang et al. (2018). See [DATA_CARD.md](DATA_CARD.md).

After pushing, GitHub Actions runs [`.github/workflows/ci.yml`](.github/workflows/ci.yml)
for the first time — it has never executed on GitHub's runners, only been validated
locally. Expect to fix small CI-only issues on the first run.
