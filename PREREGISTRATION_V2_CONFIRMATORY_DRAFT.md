# Dataset v2 Confirmatory Preregistration

> **DRAFT — NOT FROZEN — NO CONFIRMATORY SAMPLE OPENED**

**Protocol ID:** `v2-confirmatory-draft`  
**Planned dataset:** pinned HotpotQA Dataset v2 eval candidate  
**Planned parent count:** 160  
**Status boundary:** design only; no formal model output exists

## Freeze conditions

This document may change in response to the human pilot. It may be relabeled `FROZEN`
only after all four conditions hold:

1. the 20-parent human pilot is complete;
2. the final annotation guideline and schema revisions are complete;
3. the formal 160-parent assignment has not been opened;
4. no formal model output has been produced for that sample.

Freezing records the exact Git commit and SHA-256 of this file, the annotation guideline,
schema, assignment algorithm, prompt registry, model/runtime configuration, and analysis
code. Any later change is a dated amendment, never a silent edit.

## Research question and claim boundary

The planned comparison asks whether the deletion-based teacher-forced target-dependence
diagnostic `leave_one_out` agrees with independently adjudicated evidence better than the
prespecified lexical comparator `control_lexical` under the frozen Dataset-v2 challenge.
It does not identify a general causal effect, certify complete faithfulness, or establish
open-corpus performance.

## Sampling frame

- Start from the 240 parents in the pinned Dataset-v2 eval candidate.
- Exclude every parent with an unresolved annotation, adjudication, hash, leakage-group,
  dataset-defect, or execution-eligibility problem before sampling.
- Sample 160 eligible parents without replacement: 128 bridge and 32 comparison.
- Use a recorded deterministic seed and a versioned assignment program.
- Include both missing-hop and evidence-swap variants for every selected parent.
- Do not substitute an ineligible sibling or parent after model effects are inspected.
- Pilot parents are permanently excluded.

If fewer than 160 eligible parents remain, stop and amend the protocol before opening any
model output. Do not silently reduce the sample or recruit from dev/smoke.

## Experimental hierarchy

The highest sampling cluster is the Dataset-v2 leakage connected component. Parents are
nested within leakage group; the two challenge variants are nested within parent. Variants
and siblings are not independent samples. Every estimate and bootstrap replicate retains
all observed parents and variants belonging to a sampled leakage group.

## Human eligibility

Each formal task requires two independent human annotations. An exact agreement may pass
the completeness gate; any answerability or minimal-evidence disagreement requires a third
independent human adjudicator. `unclear` is not converted to the transformation's expected
label. Excluded and unresolved rows remain visible in the flow accounting.

The annotation artifact must bind pseudonym, instruction version/hash, task-content hash,
batch, answerability, applicable answer text, one or more minimal sufficient evidence sets,
exhaustiveness, ambiguity/dataset defect, confidence, rationale, timestamps, and schema
version. Citation-review artifacts bind answer correctness and sentence-citation support
as distinct judgments.

## Analysis sets

- **Assigned set:** every sampled parent/variant task.
- **Completed set:** two valid independent annotations exist.
- **Adjudicated set:** all disagreements have a valid third-human record.
- **Eligible set:** completed/adjudicated, hash-valid, nondefective rows satisfying the
  frozen inclusion rules.
- **Model-analysis set:** eligible rows with complete paired method artifacts.

The report shows assigned, completed, disagreed, adjudicated, excluded, eligible, model
attempted, model successful, and complete-pair counts. Missing labels are never negative
labels; missing method outputs are never zero effects.

## Methods and target binding

- Confirmatory method: `leave_one_out`.
- Prespecified comparator: `control_lexical` using the frozen question+answer convention.
- Generator/target: pinned Qwen3-4B configuration and frozen sentence-citation prompt v2.
- Mode A and Mode B are reported separately and never pooled.
- Model, tokenizer, local artifact hash, runtime, dtype/quantization, decoding, CUDA/GPU,
  prompt version/hash, and source revisions must match the frozen manifest.

## Endpoints

The primary method family is `leave_one_out - control_lexical`, computed on aligned
eligible parents:

1. reference-agreement F1@2 difference;
2. mean per-parent average-precision difference (MAP).

Positive values favor `leave_one_out`. These are reference-agreement endpoints against
the adjudicated evidence contract, not complete faithfulness.

Secondary diagnostic endpoints are sufficiency and comprehensiveness differences. They
remain diagnostics in all wording unless prespecified `oracle_gold` versus
`control_random` and `control_answer_string` construct checks pass in the same frozen
model/dataset/mode cell. `NOT RUN` is not `PASSED`.

Answer correctness, abstention correctness, invalid/missing sentence citations, exact
evidence-set agreement, and bridge/comparison or transformation strata are secondary or
descriptive unless promoted by a pre-freeze amendment.

## Estimation

- Aggregate the two variants within parent before the top-level comparison unless the
  endpoint is explicitly variant-specific.
- Estimate candidate-minus-comparator mean differences.
- Use a deterministic nonparametric cluster bootstrap over leakage groups.
- Use 10,000 bootstrap replicates and two-sided 95% percentile intervals.
- Preserve all nested parents and variants in every sampled cluster.
- Report point estimate, interval, number of leakage groups, parents, variants, and every
  exclusion count.
- Report bridge/comparison and transformation strata descriptively without replacing the
  pooled primary estimate.

## Multiplicity

The two primary endpoint hypotheses form one family. If p-values are produced, apply Holm's
step-down adjustment across F1@2 and MAP. Diagnostic, citation, subgroup, and error analyses
are separate explicitly labeled secondary families; they cannot rescue a failed primary
family. No post-hoc endpoint or comparator replaces a planned endpoint.

## Inter-annotator agreement gate

Before freezing:

- report raw answerability agreement and class prevalence;
- report Cohen's kappa and nominal Krippendorff's alpha;
- report exact evidence-set-family agreement, best-match Jaccard, and set-F1;
- report complete/missing pair counts and ambiguity/defect prevalence;
- inspect disagreements by reason without reading model effects.

Any relevant IAA coefficient below 0.70 blocks promotion. The response is to revise and
re-pilot the instructions or tooling, not to inflate the confirmatory sample.

## Exclusions

Pre-model exclusions include invalid hashes, incomplete dual annotation, unresolved
disagreement, adjudicated `unclear`, dataset defect, non-exhaustive evidence that prevents
the endpoint, pilot membership, leakage-group/sample mismatch, and protocol-version
mismatch. Runtime failures are post-assignment exclusions and remain in failure counts.

No row is excluded because it weakens, reverses, or increases an effect. Saturated
answer-bearing cases, if analyzed, are kept in a prespecified separate stratum.

## Stopping rule

The formal run stops after all 160 selected parents and both variants have completed the
frozen attempt matrix, or after a documented systemic safety/runtime failure makes the
matrix impossible. There is no interim effect analysis and no sample-size increase based
on observed method effects. Hardware/runtime troubleshooting may repeat failed attempts
under the frozen retry policy but may not add parents.

## Promotion language

A completed result may say that one diagnostic showed higher reference agreement than the
lexical comparator only when the adjusted primary family and cluster intervals support
that direction. It may not say causal, faithful, open-corpus, universal, or independently
validated beyond the exact human protocol. Null, reversed, or inconclusive results are
reported without changing the endpoint, comparator, sample, or claim boundary.

## Required freeze record

The future freeze commit must include:

- this document and SHA-256;
- annotation instructions/schema hashes;
- pilot completion/IAA decision;
- formal sampling manifest and cluster mapping;
- eligibility artifact schema and flow counts;
- prompt registry/hash;
- generator/tokenizer revisions and local model artifact SHA-256;
- runtime, decoding, dtype/quantization, CUDA/GPU contract;
- analysis-code commit and dependency lock;
- explicit statement that the sample and model outputs were unopened at freeze time.

Until that record exists, this document is a design draft and authorizes no formal run.
