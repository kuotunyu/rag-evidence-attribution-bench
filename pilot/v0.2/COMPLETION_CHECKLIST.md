# Pilot Completion Checklist

This checklist is for a future separately approved human pilot. Completing B0.1
engineering does not authorize the human pilot. Packages from baseline `372096b` are
obsolete; only `pilot-v0.2.2-draft` packages with the current instruction hash are valid.

## Source and platform handoff gate

- [ ] The candidate is one exact pushed commit on the open PR; no later source edit reuses its
      evidence.
- [ ] Two new external disposable detached checkouts have the same candidate commit/tree and zero
      tracked, untracked, and ignored paths before and after every build.
- [ ] Supplied A/B and replay A/B wheels are byte-identical, report Python distribution
      `0.2.0.dev0`, and every packaged `rag_evidence` byte matches its candidate Git blob.
- [ ] The annotation requirements lock installed on Python 3.11 Windows and Linux using exact
      `--require-hashes --only-binary=:all:` bootstrap; neither platform used an sdist fallback.
- [ ] One verifier-generated Windows receipt and one verifier-generated Linux receipt bind the
      same source/tree/wheel/lock/protocol/spec identities and all retained smoke hashes.
- [ ] PR jobs `checks`, `docker`, `annotation-windows`, and `annotation-linux` are green for the
      exact candidate SHA.
- [ ] Final A/B kits have the closed source-controlled layout, verify their own `SHA256SUMS`, and
      contain neither the other package nor coordinator/private/generated decision artifacts.
- [ ] `handoff-receipt-v2` and root `SHA256SUMS` remain outside Git. No wheel, kit, platform
      receipt, state, synthetic output, or human decision has been committed.

Failure of any item is a feasibility stop for owner review. Do not weaken the binary-only rule,
reuse evidence from another commit/version, or begin a human pilot.

## Synthetic engineering rehearsal

Before any human handoff, run the explicitly synthetic rehearsal in a temporary directory:

```powershell
$RehearsalRoot = Join-Path $env:TEMP "reab-b01-synthetic-rehearsal"
uv run rag-evidence annotation rehearse-synthetic --out $RehearsalRoot --repository-root .
git status --short
```

```sh
REHEARSAL_ROOT=$(mktemp -d)/reab-b01-synthetic-rehearsal
uv run rag-evidence annotation rehearse-synthetic --out "$REHEARSAL_ROOT" --repository-root .
git status --short
```

The command must report 40 tasks, one amendment, complete adjudication, one dataset-defect
exclusion, `READY_FOR_HUMAN_FREEZE_REVIEW`, and byte-identical repeats while `git status`
shows no generated rehearsal artifact.

## Separately authorized human work

- [ ] Two independent humans completed onboarding with opaque pseudonyms.
- [ ] All 40 tasks have exactly two valid submissions.
- [ ] No names, emails, repository paths, or expected labels entered artifacts.
- [ ] Submitted originals remained immutable; every correction is an amendment.
- [ ] Every disagreement preserves both originals and both rationales.
- [ ] A third independent human adjudicated or explicitly excluded every disagreement.
- [ ] Flow counts reconcile: assigned/completed/disagreed/adjudicated/excluded/eligible.
- [ ] Raw agreement, prevalence, Cohen's kappa, Krippendorff's alpha, exact evidence-set
      agreement, Jaccard, and set-F1 were reviewed.
- [ ] Both Cohen's kappa and nominal Krippendorff's alpha are defined, finite, and at least
      0.70 over all 40 post-amendment, pre-adjudication pairs; otherwise promotion is
      blocked.
- [ ] Annotation time, validation failures, ambiguity, defects, and disagreement causes
      were reviewed.
- [ ] Guideline/schema/UI revision decision is written and versioned.
- [ ] Privacy scanner passes on the clean assignment package and collected exports.
- [ ] Pilot tasks are marked permanently ineligible for confirmatory primary analysis.
- [ ] Confirmatory protocol is still DRAFT and no formal eval assignment/model output exists.
- [ ] `rag-evidence annotation collect`, `rag-evidence annotation adjudicate`, and
      `rag-evidence annotation finalize-pilot` were run with the exact paths in the
      operational runbook.

Completion of this list permits a human decision about protocol freezing. It does not itself
freeze the protocol or authorize the confirmatory run.
