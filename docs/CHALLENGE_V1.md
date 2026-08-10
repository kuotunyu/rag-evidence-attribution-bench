# Answerability Challenge v1

Answerability Challenge v1 is a deterministic, source-bound candidate set for testing
whether an attribution method reacts to controlled changes in answerability and evidence.
It extends the leakage-resistant [Dataset v2](DATASET_V2.md); it does not replace the
natural HotpotQA benchmark and is not yet a fully annotated confirmatory benchmark.

## Immutable source and generation contract

- Source manifest: `data/manifests/split_manifest_v2.json`
- Source manifest SHA-256:
  `1faa7b1c5edd52e8cf8a4a259725025f195a4b530b19b3fc9986763d32f042a9`
- Requested and resolved Hugging Face revision:
  `1908d6afbbead072334abe2965f91bd2709910ab`
- Challenge schema: `1`
- Transform version: `challenge-v1`
- Seed: `20260810`
- Parent questions: 320
- Challenge records: 960, exactly three per parent
- Generation exclusions or donor failures: 0

| split | parents | records | JSONL SHA-256 |
|---|---:|---:|---|
| smoke | 20 | 60 | `cd8c8d308093989d266a2e191caf47425596acd18cc618cf9a6a7e724302cf5d` |
| dev | 60 | 180 | `245389cb223416b61a3046923ef1c6571167c837efdb7a1adbb4c6d10ee2a613` |
| eval | 240 | 720 | `75be0dada86408c5fabfcdf6945a07b82e15a49ee18948f19df4cf172f0f342b` |

The challenge manifest is
[`challenge_manifest_v1.json`](../data/manifests/challenge_manifest_v1.json), SHA-256
`8b9bccf06de4de6b52ed95f91567bcb8e9a104d0777f3a4301b5f02310130f79`.
It binds every record ID to its content hash and binds the challenge to the source
manifest, dataset revision, dataset fingerprint, and split fingerprints.

## Transformations and review gates

| transformation | records | expected answerability | review status | confirmatory use |
|---|---:|---|---|---|
| `answer_bearing_distractor` | 320 | answerable | not required | eligible, subject to saturation stratification |
| `missing_hop` | 320 | unanswerable | two annotations pending | blocked |
| `evidence_swap` | 320 | unanswerable | two annotations pending | blocked |

The answer-bearing intervention retains the complete parent evidence and appends exactly
one controlled answer mention to a non-gold passage. Of its 320 records, 317 use a passage
that was answer-free before intervention. Three real-data saturation cases have no
answer-free non-gold passage; they are explicitly marked `selection_mode: salience_only`
and `preexisting_answer_mention: true`. Those three records must be excluded from, or
reported separately in, the primary answer-presence sensitivity estimate.

Each missing-hop record replaces one supporting passage with an answer-free passage from
a different parent in the same split. Each evidence-swap record edits exactly one
supporting sentence. The evidence-swap operator distribution is:

- `answer_substitution`: 139
- `negation_toggle`: 151
- `negation_prefix`: 30

The labels for all 640 missing-hop and evidence-swap records are deliberately provisional.
HotpotQA supporting facts may be redundant or non-exhaustive, and a synthetic edit may not
remove every valid reasoning route. Do not use these 640 records for confirmatory causal or
answerability claims until two independent annotators review them and a third reviewer
adjudicates disagreements.

## Independent audit

The committed candidate was checked without network or model access:

1. Parsed all 960 rows through the strict challenge schema and recomputed every content
   hash and manifest field.
2. Rebuilt the candidate after reversing split insertion order and parent order. The
   manifest and all three JSONL files were byte-identical.
3. Re-ran `data challenge`; the manifest and JSONL SHA-256 values were unchanged.
4. Verified 960 unique IDs, 320 complete three-transform parent groups, same-split and
   different-parent missing-hop donors, answer-free donor passages, full parent-evidence
   retention for answer-bearing records, and exactly one edited supporting sentence for
   every evidence swap.
5. Compared the historical v1 and natural v2 protected paths with commit `3ae961f`; no
   protected artifact changed.

Reproduction requires the ignored, reproducible `data/v2/prepared/*.jsonl` source files:

```powershell
uv sync --frozen --extra ml --extra app
uv run rag-evidence data challenge --config configs/v2/eval.yaml
uv run pytest -m "not gpu and not slow" -q
```

On Windows with Python 3.11, use an ASCII-only checkout or worktree path. An editable
installation below a non-ASCII parent directory can create a UTF-8 `.pth` file that the
CP950 startup locale cannot decode.

## License and release boundary

Repository code is MIT licensed. Challenge manifests and JSONL records embed or transform
HotpotQA questions and passages and remain under HotpotQA's CC BY-SA 4.0 terms, with
attribution to Yang et al. (2018). The upstream project and the Hugging Face dataset card
both state that dataset license:

- <https://github.com/hotpotqa/hotpot>
- <https://huggingface.co/datasets/hotpotqa/hotpot_qa>

A software release and a benchmark-data release must therefore remain separate. Do not
label the mixed repository contents as uniformly MIT, and do not publish a benchmark-data
Zenodo record until its CC BY-SA metadata, upstream attribution, checksums, and
machine-readable third-party license inventory are complete.
