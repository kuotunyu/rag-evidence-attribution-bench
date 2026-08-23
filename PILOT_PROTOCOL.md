# Human Annotation Pilot Protocol

**Protocol ID:** `pilot-v0.2.2-draft`
**Status:** PILOT ONLY — mutable after pilot
**Scope:** 20 parent questions; missing-hop and evidence-swap tasks
**Primary-analysis eligibility:** permanently ineligible

The instruction identity is the SHA-256 of this entire file, byte for byte. Packages from
baseline `372096b` used the superseded `pilot-v0.2-draft` identity and are obsolete. Only
canonical packages regenerated from this file may enter a future owner-approved handoff.

## Purpose

The pilot tests whether independent annotators can understand the instructions, operate
the local annotation tool, identify minimal sufficient sentence evidence when it exists,
and explain disagreements. It estimates task time and surfaces ambiguous or defective
items. It does not estimate a model effect and cannot support a benchmark claim.

## Non-goals

- No confirmatory endpoint or model comparison is tested.
- No pilot decision enters the confirmatory primary analysis.
- No expected transformation label is shown or treated as a human label.
- No LLM, Codex, rule system, or model output supplies or adjudicates a decision.
- No GPU inference or formal 160-parent sample is opened.

## Sample

The package mechanically projects all 20 Dataset-v2 smoke parents and their missing-hop
and evidence-swap variants, for 40 tasks total. The sample is for tooling only and does
not stand in for the eval population. Each task is assigned independently to two human
annotator pseudonyms. Sibling variants are never adjacent in either annotator's order.

## Blinding

Annotators receive an export package, not repository access. Payloads contain only an
opaque task ID, opaque challenge ID, question, public passage aliases, public sentence aliases,
instruction version/hash, batch, and task-content hash. They exclude all parent/group/sibling
metadata and
transformation identity, expected answerability, changed fields, provenance, siblings,
gold/adjudicated labels, model/method identities, scores, and abstention expectations.

## Annotator instructions

1. Read only the question and the supplied passages.
2. Decide `answerable`, `unanswerable`, or `unclear` from those passages alone.
3. Use `answerable` only when the passages support a specific answer.
4. For `answerable`, enter the shortest supported answer text.
5. Select at least one minimal sufficient set of sentence aliases.
6. Add another evidence set only when it is independently sufficient.
7. Mark whether the listed sets appear exhaustive.
8. Mark ambiguity and dataset defect separately; do not use them as synonyms.
9. Report confidence on the fixed 1--5 scale.
10. Give a concise rationale grounded in the visible passages.
11. Use `unclear` when the instructions or item prevent a defensible answerability call.
12. Never infer the intended transformation or try to reverse-engineer expected labels.

### Minimal sufficient evidence

An evidence set is sufficient when a reader can derive the answer using that set and the
question, without relying on another supplied sentence. It is minimal when removing any
selected sentence makes the set insufficient. Different valid reasoning chains are stored
as separate sets rather than merged into one oversized set.

### Answerability examples

Synthetic onboarding examples demonstrate the three labels without using any formal pilot
item. The examples are deliberately unrelated to HotpotQA and carry no expected-label
field in the tool payload.

## Workflow

1. Each annotator completes onboarding and records only a pseudonym.
2. The local tool displays its exact instruction hash.
3. Drafts autosave locally and may be resumed.
4. Submission validates all required fields and becomes immutable.
5. A correction is a signed amendment that preserves the original submission.
6. The coordinator checks two-annotator completeness without opening expected labels.
7. Disagreements enter a queue containing both original judgments and rationales.
8. A third independent human adjudicator resolves or excludes disagreements.
9. Original annotations remain unchanged.

### Coordinator commands

Run collection only after receiving both independently exported streams and a present
amendment file from each annotator. An empty amendment file is valid; a missing file is not.

```text
rag-evidence annotation collect --manifest coordinator/manifest.json --submission returns/a/submissions.jsonl --submission returns/b/submissions.jsonl --amendment returns/a/amendments.jsonl --amendment returns/b/amendments.jsonl --out coordinator/collection
```

Incomplete collection writes its receipt and effective rows but exits blocked without a
disagreement queue. A complete collection may open the localhost-only third-human console:

```text
rag-evidence annotation adjudicate --manifest coordinator/manifest.json --effective coordinator/collection/effective-submissions.jsonl --state coordinator/private-adjudication-state --host 127.0.0.1 --port 8002
```

Export immutable adjudications from
`http://127.0.0.1:8002/api/export/adjudications.jsonl`, then finalize:

```text
rag-evidence annotation finalize-pilot --manifest coordinator/manifest.json --originals coordinator/collection/original-submissions.jsonl --amendments coordinator/collection/amendments.jsonl --adjudications coordinator/adjudications.jsonl --protocol PILOT_PROTOCOL.md --out coordinator/final
```

The coordinator stops at every `BLOCKED_*` verdict. Even
`READY_FOR_HUMAN_FREEZE_REVIEW` means only that the owner may review whether to freeze a
future protocol. It does not authorize the human pilot, confirmatory sampling, model/API
execution, merge, tag, release, or publication.

## Data captured for pilot learning

- task start and submit timestamps;
- active completion time where available;
- validation errors encountered;
- answerability and evidence-set disagreements;
- annotator confidence;
- ambiguity and dataset-defect flags;
- adjudicator-coded disagreement reason;
- free-form tooling/instruction feedback stored outside scientific labels.

Do not collect names, email addresses, employer identifiers, or other PII in committed
artifacts.

## Pilot completion rule

The pilot ends when all 40 tasks have two valid independent submissions, every disagreement
has an independent human adjudication or explicit exclusion, both mandatory IAA
coefficients pass, the export/privacy scan passes, and pilot feedback has been reviewed.
It does not end early because agreement is high, and it is not expanded because a model
effect appears small or large.

## Promotion gate

The confirmatory protocol remains DRAFT unless:

- raw agreement, prevalence, and evidence-set overlap have been reviewed as descriptive
  quantities;
- both Cohen's kappa and nominal Krippendorff's alpha are defined, finite, and at least
  0.70 over exactly 40 post-amendment, pre-adjudication pairs;
- guideline-driven disagreements have been resolved by revising instructions and, if
  needed, the schema/UI;
- revised instructions are versioned and hashed;
- blind-export and append-only integrity tests still pass;
- no formal eval assignment or model output has been opened.

Either IAA coefficient below 0.70, undefined, non-finite, or computed from fewer than 40
complete pairs blocks promotion. Raw agreement or prevalence cannot substitute for either
coefficient. Increasing the confirmatory sample is not a substitute for fixing
instructions. Any revision after this pilot makes the pilot permanently ineligible for
the confirmatory primary analysis, as already required by this protocol.

## Human responsibilities

Independent humans must perform all task judgments, amendment decisions, adjudications,
and the final guideline review. Codex may maintain schemas, UI behavior, exports, tests,
and deterministic accounting, but cannot fill, correct, or resolve a scientific label.

## Outputs

- two blinded annotator packages;
- immutable submissions and append-only amendments;
- a disagreement queue;
- independent adjudications;
- timing/validation summaries;
- IAA and evidence-set agreement summaries;
- a written guideline revision decision;
- a pilot completion checklist.

None of these outputs are formal confirmatory results.

## Privacy and repository boundary

Annotators receive only their external verified A or B delivery kit. A never receives B's
package, B never receives A's package, and neither receives the coordinator manifest or
repository access. State, submissions, amendments, adjudications, eligibility/IAA
artifacts, reports, receipts, checksums, synthetic rehearsal outputs, wheels, sdists, and
delivery copies remain outside Git. Only decision-free canonical package sources,
schemas, builder source/tests, runbooks, launchers, dependency lock, and hashes belong in
the repository.
