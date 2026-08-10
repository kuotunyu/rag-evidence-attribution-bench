# Dataset v2: Leakage-Resistant HotpotQA Splits

Dataset v2 is the pinned, group-disjoint successor to the historical schema-v1 split.
It freezes a public evaluation set for new experiments without changing or invalidating
any v1 artifact. It is not a permanently blind test set.

## Pinned source and integrity

- Dataset: `hotpotqa/hotpot_qa`, config `distractor`, split `validation`.
- Requested and resolved Hugging Face revision:
  `1908d6afbbead072334abe2965f91bd2709910ab`.
- Source examples: 7,405.
- Source dataset fingerprint:
  `sha256:aa57b60128d8f13286ad4b72249e0b5c845ec4f670f78f528d59e99af650a6e9`.
- Eligible-pool fingerprint:
  `sha256:d3627bd28efaee6ae05d1dec742215867ad2f3ee3dff63aebeb497d8a480a1ba`.
- Manifest fingerprint:
  `sha256:1faa7b1c5edd52e8cf8a4a259725025f195a4b530b19b3fc9986763d32f042a9`.

Preparation resolves the requested revision before loading or writing data. A different
resolved SHA, source fingerprint, selected-example fingerprint, group assignment, or
overlap count fails closed.

## Eligibility and exclusions

An example is eligible only when every supporting-fact title occurs exactly once in its
context and every supporting sentence index is in range. Of 7,405 examples, 7,404 are
eligible and one is excluded for `sent_id_out_of_range`. No selected example drops a
supporting fact.

The eligible pool contains 5,917 bridge (79.92%) and 1,487 comparison (20.08%) questions.
All 7,404 source-validation examples have HotpotQA level `hard`; dataset v2 therefore
cannot estimate performance by difficulty level within this source split.

## Leakage grouping and allocation

Questions are connected when any context paragraph has either the same normalized title
or the same canonical paragraph fingerprint. Title normalization applies Unicode NFKC,
whitespace collapse, trimming, and case folding. Paragraph fingerprints preserve sentence
order and hash NFKC/whitespace-normalized text. Connected components are allocated whole,
in eval/dev/smoke order, using seed `20260718`.

The eligible pool forms 3,184 groups. Its largest connected component contains 2,430
questions, which demonstrates why question-ID-only splitting is insufficient. That group
is not selected. The largest selected component contains 42 eval questions, 6 dev
questions, and 4 smoke questions.

| split | requested | realized | groups | bridge | comparison | level | frozen public eval |
|---|---:|---:|---:|---:|---:|---|---|
| smoke | 20 | 20 | 15 | 16 | 4 | hard: 20 | no |
| dev | 60 | 60 | 42 | 48 | 12 | hard: 60 | no |
| eval | 240 | 240 | 128 | 192 | 48 | hard: 240 | yes |

The selected set contains 185 complete groups and 320 questions. The remaining 2,999
groups and 7,084 questions are unselected. Each split is exactly 80% bridge and 20%
comparison, closely matching the eligible pool.

## Independent audit

The committed candidate was validated against a fresh load of the pinned source and then
rebuilt from the raw examples in reverse order. The rebuilt dictionary and serialized
bytes were identical to the candidate, including the manifest SHA-256 above.

| cross-split key | recorded overlaps | independently recomputed overlaps |
|---|---:|---:|
| question ID | 0 | 0 |
| normalized title | 0 | 0 |
| canonical paragraph | 0 | 0 |

The manifest is [split_manifest_v2.json](../data/manifests/split_manifest_v2.json).
Prepared files under `data/v2/prepared/` are reproducible local derivatives and remain
untracked. The committed explorer samples live under `results/v2/raw/`; v2 configs and
transfer bundles preserve this namespace and cannot overwrite historical v1 results.

## License and release boundary

Repository code is MIT licensed. Files under `results/v2/raw/` embed HotpotQA questions
and passages and are distributed under CC BY-SA 4.0 with attribution to Yang et al.
(2018), consistent with the repository data notice. This v2 dataset is ready for the next
benchmark experiments, but it is not by itself the final Zenodo package: software and
benchmark-data records still require separate release metadata and a machine-readable
third-party license inventory.
