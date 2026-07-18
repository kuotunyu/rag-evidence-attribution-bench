# DATA CARD

> Skeleton — completed at milestone M12. Facts below are already final; sections marked
> _(pending)_ are filled as the corresponding milestone lands.

## Source dataset

- **HotpotQA**, distractor setting. HF dataset `hotpotqa/hotpot_qa`, config `distractor`,
  split `validation` (7,405 examples; the distractor test split has no public answers).
- Reference: Yang et al., *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question
  Answering*, EMNLP 2018.
- License: **CC BY-SA 4.0**.

## What this repo stores (and what it never stores)

- Never committed: the downloaded dataset, prepared JSONL, model weights, caches.
- Committed: `data/manifests/split_manifest.json` (question IDs, seed, per-example SHA-256
  fingerprints), and benchmark results under `results/` — those results embed HotpotQA
  question/passage text and are therefore distributed under **CC BY-SA 4.0** with attribution,
  while the code remains MIT.

## Splits _(manifest committed at M2)_

- `smoke` = 20, `dev` = 60, `eval` = 240 questions; mutually disjoint; drawn from a seeded
  shuffle of validation question IDs (eval drawn first; `eval` is locked — the manifest is
  never regenerated once committed).

## Stable IDs

- question: HotpotQA `_id`; passage: `{qid}-p{idx:02d}` (position in the example's context
  list); sentence: `{passage_id}-s{jdx:02d}`.
- Positional IDs are safe because every stage verifies per-example SHA-256 fingerprints
  against the committed manifest before use; any content drift fails loudly.

## Known dataset quirks handled

- `context` / `supporting_facts` are parallel arrays (title-keyed join within an example).
- Out-of-range `sent_id` in supporting facts: dropped with a warning, recorded per example
  as `dropped_supporting_facts`; the gold passage set is unaffected.

## Fingerprint verification _(pending — numbers recorded after the real M2 run)_
