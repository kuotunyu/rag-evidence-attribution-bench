# ARC-JSD legacy environment (official reproduction)

The official ARC-JSD implementation ([ruizheliUOA/ARC_JSD](https://github.com/ruizheliUOA/ARC_JSD),
paper [arXiv:2505.16415](https://arxiv.org/abs/2505.16415)) pins **transformers==4.43.3** —
which predates Qwen3 support (≥4.51) — and targets Qwen2/2.5 + Gemma-2 with flash-attn.
It therefore cannot live in the main environment. This directory isolates it.

## Purpose and protocol (stop-loss: 2 days total for the Qwen3 port)

1. **Reproduce the official flow first** — run `10_arcjsd_official_repro.ipynb` on Colab
   (GPU): clones the official repo, installs its pinned dependencies, runs its attribution
   on Qwen2.5-0.5B-Instruct for a few HotpotQA-style examples.
2. **Validate the native port** — the main repo ships an EXPERIMENTAL native ARC-JSD
   (`attribute --method arc_jsd`, `src/rag_evidence/attribution/arc_jsd.py`): mean
   positional JSD between teacher-forced answer-token distributions with the full context
   vs. one passage removed. Run both implementations on the same examples with the SAME
   model (Qwen2.5-0.5B-Instruct, supported by both) and compare rankings/scores.
3. Only after step 2 agrees may the `experimental` label be dropped. Any divergence gets
   recorded in the main repo's FAILURES.md with the observed numbers.

Known deliberate differences of the native port (documented, not bugs):
- passage-level sources (the official RAG setting attributes retrieved documents),
- our benchmark prompt template,
- main-repo benchmark runs use Qwen3-4B-Instruct-2507 (official code has no Qwen3).

## Environment

Separate uv project (`pyproject.toml` here) — never install these pins into the main env:

```bash
cd legacy/arc_jsd
uv sync            # transformers 4.43.3, torch 2.4.x — Linux/Colab intended
```

flash-attn is intentionally NOT pinned here (Windows/Colab build pain); the official repo
used it for speed, not correctness. The reproduction notebook installs everything on Colab.

## Status

- [ ] official flow reproduced on Colab (user action — GPU required)
- [ ] native port validated against official scores on ≥5 examples
- Native port implemented + FakeLM-tested in the main repo (2026-07-18); labeled
  experimental until the boxes above are checked.
