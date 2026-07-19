# DATA CARD

## Source dataset

- **HotpotQA**, distractor setting. HF dataset `hotpotqa/hotpot_qa`, config `distractor`,
  split `validation` (7,405 examples; the distractor test split has no public answers).
- Reference: Yang et al., *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question
  Answering*, EMNLP 2018.
- License: **CC BY-SA 4.0**.
- Each example: a question, a short answer, 10 context paragraphs (2 gold + 8 distractors),
  and sentence-level supporting facts for the gold paragraphs.

## What this repo stores (and what it never stores)

Never committed: the downloaded dataset, `data/prepared/` JSONL, model weights, HF caches,
embeddings/logprob caches (`results/raw/cache/` is gitignored).

Committed:
- `data/manifests/split_manifest.json` — question IDs, selection seed, per-example
  SHA-256 content fingerprints (all 320 selected examples), per-split and full-split
  hashes;
- benchmark results under `results/` including `results/raw/<split>/samples/records.jsonl`
  (the 320 selected questions with their passages, needed by the precomputed explorer).
  These files **embed HotpotQA text** and are therefore distributed under
  **CC BY-SA 4.0** with attribution, while the code remains MIT. 320 of 7,405 validation
  examples (~4%) — not the full dataset.

## Splits (built 2026-07-18, seed 20260718)

| split | size | locked | selection |
|---|---|---|---|
| smoke | 20 | no | third draw |
| dev | 60 | no | second draw |
| eval | 240 | **yes** | first draw |

Mutually disjoint, drawn from a seeded shuffle of sorted validation question IDs. The
locked eval split is drawn FIRST so nothing tuned on smoke/dev can leak into it.
`data prepare` never regenerates an existing manifest; the eval split is run once per
benchmark version.

## Fingerprints (real values from the committed manifest)

- Full validation split hash: `sha256:aa57b60128d8f13286ad4b72249e0b5c845ec4f670f78f528d59e99af650a6e9`
- Split hashes (first 20 hex): smoke `188bdc489a543e422529`, dev `a5fb5cba4071b178b9ea`,
  eval `6d4542d69d00e9691c2e`
- Canonicalization: per-example SHA-256 over compact sorted-key JSON of
  `{question_id, question, answer, type, level, context, supporting_facts}` (UTF-8).

Every pipeline stage re-verifies its split's fingerprints at startup; any re-download that
changes content fails loudly before any compute is spent (positional passage IDs are safe
because of exactly this check).

## Stable IDs

- question: HotpotQA `_id`; passage: `{qid}-p{idx:02d}` (position in the example's
  context list); sentence: `{passage_id}-s{jdx:02d}`.
- Prompt aliases `[P1]…[P10]` are prompt-local; every artifact stores its
  `alias_map {alias → passage_id}`. Aliases bind to passage identity — ablating a passage
  never renumbers the others.

## Known dataset quirks handled

- `context` / `supporting_facts` arrive as column-oriented parallel arrays; joined by
  title within each example (titles are unique per example; duplicates would abort).
- Out-of-range supporting-fact `sent_id`: the sentence reference is dropped with a warning
  and recorded per example (`dropped_supporting_facts`); the gold passage set is
  unaffected. Supporting-fact titles absent from the context are likewise dropped and
  recorded. In the 320 selected examples, zero entries were dropped in the real
  2026-07-18 prepare run.
