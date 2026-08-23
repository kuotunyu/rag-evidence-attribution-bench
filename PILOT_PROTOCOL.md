# Human Annotation Pilot Protocol

**Protocol ID:** `pilot-v0.2-draft`  
**Status:** PILOT ONLY — mutable after pilot  
**Scope:** 20 parent questions; missing-hop and evidence-swap tasks  
**Primary-analysis eligibility:** permanently ineligible

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
opaque task ID, opaque parent-group pseudonym, question, public passage aliases, public
sentence aliases, instruction version/hash, batch, and task-content hash. They exclude
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
has an independent human adjudication or explicit exclusion, the export/privacy scan
passes, and pilot feedback has been reviewed. It does not end early because agreement is
high, and it is not expanded because a model effect appears small or large.

## Promotion gate

The confirmatory protocol remains DRAFT unless:

- raw agreement, Cohen's kappa, Krippendorff's alpha, prevalence, and evidence-set overlap
  have been reviewed;
- the relevant IAA coefficient is at least 0.70;
- guideline-driven disagreements have been resolved by revising instructions and, if
  needed, the schema/UI;
- revised instructions are versioned and hashed;
- blind-export and append-only integrity tests still pass;
- no formal eval assignment or model output has been opened.

IAA below 0.70 blocks promotion. Increasing the confirmatory sample is not a substitute
for fixing instructions. Any revision after this pilot makes the pilot permanently
ineligible for the confirmatory primary analysis, as already required by this protocol.

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
