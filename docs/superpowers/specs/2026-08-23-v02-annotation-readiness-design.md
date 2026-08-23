# v0.2 Annotation-Ready Engineering Design

**Status:** APPROVED for B0 engineering on 2026-08-23  
**Base:** `main` at `935b39658633211a76c52b6bbfd4262353780e80`  
**Delivery branch:** `codex/v0.2-annotation-readiness`

## Boundary

This increment prepares the repository for an independent-human pilot. It does not run
the confirmatory benchmark, produce model outputs, manufacture labels, use a GPU or paid
API, create additional dataset tracks, or publish a release. Historical v0.1 artifacts
and tags remain immutable. Remote GitHub metadata is documented as a handoff suggestion
only.

## Public scientific claims

- Leave-one-out is a deletion-based, teacher-forced target-dependence diagnostic, not a
  causal primary baseline.
- ARC-JSD is an experimental distributional-dependence diagnostic, not a causal method.
- Sufficiency and comprehensiveness remain diagnostics until preregistered construct
  controls pass.
- Agreement with HotpotQA passages or supporting facts is reference agreement, not causal
  ground truth or complete faithfulness.
- The compact README distinguishes the historical v0.1 closed-candidate baseline from the
  unexecuted v2 confirmatory design and links detailed evidence instead of duplicating it.

Claim-contract tests must reject the retired causal wording in README, MODEL_CARD, and
generated report labels.

## Protocol layers

`PILOT_PROTOCOL.md` governs a 20-parent usability pilot. Pilot records test instructions,
UI, completion time, and disagreement reasons and are permanently excluded from the
confirmatory primary analysis. Instructions and schemas may change after the pilot.

`PREREGISTRATION_V2_CONFIRMATORY_DRAFT.md` remains visibly DRAFT. It proposes 160 eval
parents (128 bridge, 32 comparison), both missing-hop and evidence-swap variants per
parent, `leave_one_out - control_lexical` as the primary comparison, leakage groups as
bootstrap clusters, and variants nested within parents. It fixes candidate endpoints,
exclusions, a Holm family, 10,000 cluster-bootstrap resamples, IAA gates, and stopping
rules. It may become FROZEN only after the human pilot and final guideline revision, but
before opening the formal sample or producing formal model output; freezing records an
exact commit and document SHA-256. Model-effect estimates never trigger sample expansion.

## Annotation architecture

The annotation subsystem has five narrow boundaries:

1. Strict, versioned Pydantic records for answerability, sentence-citation review,
   amendments, and adjudication. Extra fields fail closed. Annotator IDs are pseudonyms;
   names and email addresses are invalid artifact content.
2. A server-side blind projector creates an explicit allowlisted task payload. It never
   serializes transformation identity, expected labels, changed fields, provenance,
   siblings, gold/adjudicated labels, model or method identities, scores, or abstention
   expectations. Stable public aliases replace internal passage and sentence IDs.
3. A deterministic assignment scheduler allocates each task to two independent annotator
   pseudonyms, randomizes order from a fixed seed, and prevents siblings from appearing
   adjacently for an annotator. Annotator packages contain projected tasks only and need
   no repository access.
4. A local FastAPI/HTML tool supports task display, evidence-set editing, autosave, resume,
   validation, progress, immutable submission, append-only amendments, JSON/JSONL export,
   and an adjudicator view. Originals are never overwritten.
5. Workflow code builds a two-annotator completeness gate, disagreement queue,
   adjudication records that embed both original decisions and reasons, and explicit flow
   counts for assigned, completed, disagreed, adjudicated, excluded, and eligible items.
   `unclear` is never coerced to a transformation's expected label.

## Annotation contracts

Answerability records bind `annotation_task_id`, `challenge_id`,
`blinded_parent_group`, annotator pseudonym, instruction version/hash, task content hash,
assignment batch, answerability (`answerable`, `unanswerable`, `unclear`), optional answer
text, one or more minimal sufficient evidence sets when applicable, exhaustiveness,
ambiguity/dataset-defect flags, confidence, concise rationale, timestamps, and schema
version.

Citation-review records additionally bind the generated answer, selected sentence citation
aliases, support judgment (`supported`, `partial`, `unsupported`, `invalid`), missing
evidence, answer correctness, and abstention correctness. Answer quality and citation
support are scored separately.

## Agreement and analysis

Synthetic, hand-checked fixtures cover raw agreement, Cohen's kappa, prevalence,
Krippendorff's alpha, exact evidence-set agreement, Jaccard, set-F1, Wilson intervals,
cluster bootstrap, parent-nested variants, and exclusion flow. Missing decisions are not
negative labels, and sibling variants are not independent observations. IAA below 0.70
blocks promotion. A `NOT RUN` construct status never becomes `PASS`.

## Sentence-citation generation contract

Prompt v2 assigns stable passage and sentence aliases and requests a strict answer plus
sentence-citation response. A strict parser distinguishes valid, invalid, and missing
citations, while preserving the v1 passage-level parser and all historical v0.1 artifacts.
Prompt registry entries have exact version/hash binding. Future v2 run metadata binds
generator and tokenizer revisions, local artifact SHA-256, runtime/library versions,
dtype, quantization, decoding, CUDA/GPU metadata, and prompt hash.

## Challenge execution boundary

Explicit challenge `generate`, `attribute`, `evaluate`, and `report` commands use a
prepared execution projection that retains parent/sibling linkage and variant identity.
Confirmatory commands fail closed unless human eligibility artifacts are complete;
provisional or pending rows cannot enter confirmatory analysis. Pilot and confirmatory
namespaces are physically separate, and natural, missing-hop, and evidence-swap outputs
remain distinguishable. This increment exercises that path only with the fake backend and
synthetic fixtures.

## Construct controls

The implementation supports `oracle_gold`, `control_random`, and
`control_answer_string` and exposes their eligibility gates. Reports use the exact states
`NOT RUN`, `FAILED`, or `PASSED`; absence of controls is not success. No formal construct
run occurs in B0.

## Pilot deliverable

Mechanically project and dual-assign 20 challenge parents without inspecting or judging
their answers. Deliver leak-scanned annotator packages, onboarding/instructions, a
completion checklist, and no annotations. Pilot data never enters confirmatory analysis.

## Verification gate

The delivery gate includes the full non-GPU test suite, formatting, lint, strict mypy,
package build, Docker health, schema and blind-payload tests, assignment determinism,
append-only/adjudication integrity, hand-checked IAA/statistics fixtures, synthetic
challenge end-to-end tests, claim consistency, and export/privacy scans.

The final handoff reports branch/worktree, commits/diff, README size reduction, corrected
claims, annotation/UI/workflow behavior, blind-projection evidence, challenge integration,
synthetic verification, pilot-package paths, required human actions, unexecuted formal-v2
work, and a GO / CONDITIONAL GO / NO-GO decision for the human pilot. It then stops without
push, merge, tag, release, GPU execution, or remote mutation.
