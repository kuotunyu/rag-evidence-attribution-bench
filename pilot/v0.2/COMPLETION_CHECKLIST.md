# Pilot Completion Checklist

This checklist is for a future separately approved human pilot. Completing B0.1
engineering does not authorize the human pilot. Packages from baseline `372096b` are
obsolete; only `pilot-v0.2.1-draft` packages with the current instruction hash are valid.

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
