# Answerability Challenge v1 Design

**Status:** approved under the existing Flagship Validity Promotion Design and the
user's delegated implementation authority on 2026-08-10

## Objective

Build Gate B3 as a deterministic, offline challenge namespace derived from the pinned,
group-disjoint HotpotQA manifest v2. Every selected smoke/dev/eval parent produces exactly
one `missing_hop`, one `answer_bearing_distractor`, and one `evidence_swap` record. The
natural HotpotQA splits and historical artifacts remain unchanged.

The challenge distinguishes generated expectations from validated ground truth.
`missing_hop` and `evidence_swap` records remain provisional and cannot enter a
confirmatory aggregate until two annotators agree or a third person adjudicates.

## Chosen approach

Use controlled transforms over prepared examples and same-split non-gold passages. This
keeps source provenance, transformation logic, and hashes reproducible without a remote
LLM or an NLI judge. A human gate handles the scientific question that automation cannot
settle reliably: whether the transformed context is truly unanswerable and minimally
sufficient evidence has actually been removed.

Rejected alternatives:

- Remote LLM rewriting is non-deterministic, adds evaluator bias, and violates the
  no-paid-API execution boundary.
- A hand-authored 100-example-only challenge would delay the engineering pipeline and
  would not cover every selected parent. Human review remains necessary, but it consumes
  a deterministic 960-record candidate set instead of defining the transformations.

## Artifacts and namespace

The v2 configs gain a strict `challenge` block:

```yaml
challenge:
  schema_version: 1
  seed: 20260810
  transform_version: challenge-v1
  manifest_path: data/manifests/challenge_manifest_v1.json
  prepared_dir: data/v2/challenge
```

The CLI command is:

```text
python -m rag_evidence.cli data challenge --config configs/v2/eval.yaml
```

It reads all three prepared v2 splits, irrespective of the config's active split, and
writes:

- tracked manifest: `data/manifests/challenge_manifest_v1.json`;
- ignored reproducible rows: `data/v2/challenge/{smoke,dev,eval}.jsonl`;
- tracked explorer/transport rows:
  `results/v2/raw/{smoke,dev,eval}/challenge/samples/records.jsonl`.

Placing samples below each existing v2 raw split makes current export/import transport
include them without a new transport protocol. Challenge rows never appear in natural
`samples/records.jsonl`, so natural aggregates cannot consume them accidentally.

## Record schema

Each JSONL row is a self-contained `ChallengeRecord`:

```json
{
  "schema_version": 1,
  "challenge_id": "ch-<24 lowercase hex>",
  "parent_question_id": "<HotpotQA id>",
  "source_split": "eval",
  "transformation": "missing_hop",
  "transform_version": "challenge-v1",
  "seed": 20260810,
  "expected_answerability": "unanswerable",
  "review": {
    "required": true,
    "status": "pending_two_annotators",
    "confirmatory_eligible": false
  },
  "changed_fields": ["/example/passages/3"],
  "provenance": {},
  "parent_fingerprint": "<64 lowercase hex>",
  "content_hash": "<64 lowercase hex>",
  "example": {}
}
```

`challenge_id` is the first 24 hex characters of SHA-256 over transform version, source
split, parent question ID, and transformation name. The nested `Example.question_id`
equals the challenge ID. Passage and sentence IDs are remapped from their stable slot
indices to the challenge ID so different transformations cannot collide.

`changed_fields` contains sorted unique JSON Pointer paths. `provenance` records donor,
removed passage/sentence IDs, the edit operator, and inserted text where applicable.
`content_hash` is SHA-256 over compact sorted-key canonical JSON of every other record
field. Any unknown field, malformed ID, hash mismatch, or inconsistent remapping fails.

## Deterministic transformations

Candidate ordering uses SHA-256 of `seed`, parent ID, transformation, and candidate ID.
It never depends on input iteration order.

### Missing hop

1. Select one complete gold passage by seeded order.
2. Select a donor from non-gold passages belonging to a different parent in the same
   split. Reject donors whose normalized title or canonical paragraph already occurs in
   the parent, or whose text contains the normalized target answer.
3. Replace the selected gold slot in place with donor title and sentences.
4. Remove the replaced passage and its sentences from the transformed gold labels; retain
   any remaining gold evidence.

The expected answerability is `unanswerable`. Review is required and confirmatory use is
blocked. If no valid donor exists, generation fails rather than weakening the filter or
inventing text.

### Answer-bearing distractor

1. Select a parent non-gold passage whose existing text does not contain the normalized
   target answer.
2. Append one controlled sentence: `<title> has also been associated with <answer>.`
3. Preserve every gold passage and supporting-sentence label.

The expected answerability is `answerable`, because the complete original evidence chain
is retained. The record is automatically confirmatory-eligible under Gate B3; Gate B4
still includes at least 40 such cases in its broader human audit. The exact inserted text
is stored, so its synthetic style remains visible and can be analyzed separately.

### Evidence swap

1. Select one annotated supporting sentence by seeded order and one non-gold title from
   the same parent as the replacement entity source.
2. If the raw answer occurs literally in the sentence, replace its first case-insensitive
   occurrence with the non-gold title.
3. Otherwise toggle the first auxiliary/copular negation (`is`, `are`, `was`, `were`,
   `has`, `have`, `had`, `can`, `could`, `will`, `would`, `do`, `does`, or `did`). If no
   such token occurs, prefix `It is not true that ` while retaining the original sentence.
4. Remove the edited sentence from supporting-sentence labels. Remove its passage from
   gold-passage labels only when no other supporting sentence remains in that passage.

The original surface text is preserved except for the recorded answer substitution or
negation operation. Expected answerability is `unanswerable`, but review is required and
confirmatory use is blocked because HotpotQA evidence may be redundant or non-exhaustive.

## Manifest and validation

`challenge_manifest_v1.json` records:

- source manifest path and SHA-256;
- pinned dataset revision and source dataset fingerprint;
- transform version and seed;
- parent and record counts by split, transformation, expected answerability, and review
  status;
- per-record content hashes and per-split aggregate hashes.

Validation reloads the source prepared records and independently recomputes all 960
challenge records. It requires exactly three transformations per parent, no duplicate
challenge IDs, valid parent fingerprints, byte-stable sorted ordering, correct
transformation-specific labels, and exact manifest statistics. Reversing parent and donor
input order must produce identical serialized bytes.

An existing manifest is immutable. Re-running the command validates it against the source
and regenerates only derived JSONL files. A manifest/source/config mismatch fails before
any output is replaced.

## Error handling

- The command requires manifest schema v2 and the exact pinned source revision.
- Missing or stale prepared splits, unknown parent IDs, dropped supporting facts, invalid
  stable IDs, missing gold/non-gold candidates, and donor-filter exhaustion raise
  `DataError`.
- Whole JSON manifests use atomic replacement only on first creation. Derived JSONL is
  written to temporary files and atomically replaced, avoiding half-written challenge
  sets after a crash.
- Pending human review is a visible scientific state, not an exception and not an
  automatic positive label.

## Testing and acceptance

Every production behavior starts with a failing test. Unit tests cover IDs and hashes,
all three transforms, donor filtering, evidence-label updates, strict schema rejection,
review gating, and source/config mismatch. Property-style tests compare reordered inputs
and verify exactly three unique records per parent. An end-to-end fixture exercises the
CLI preparation boundary, idempotence, crash-safe writes, v2-only paths, and historical
namespace isolation.

Gate B3 engineering acceptance requires:

- 960 deterministic records from the real 320-parent manifest;
- 320 rows for each transformation and no natural-split mutation;
- complete parent/change/provenance/hash fields;
- all 640 missing-hop/evidence-swap rows blocked from confirmatory use pending review;
- all 320 answer-bearing-distractor rows retain complete source evidence;
- reverse-order and repeated generation are byte-identical;
- formatting, lint, mypy, offline tests, coverage, and historical-artifact checks pass.

Gate B3 completion does not validate answerability and does not authorize a faithfulness
claim. That scientific promotion remains blocked on the Gate B4 annotation protocol.
