# B0.1 Human Pilot Operational Closure Design

**Status:** APPROVED by owner on 2026-08-23
**Branch baseline:** `372096bc73dfb91b27e0249e1e29457a82af69bc`
**Scope:** operational closure for the synthetic rehearsal and future independent-human
pilot; no human decisions, model outputs, confirmatory sampling, release, or merge

## Outcome and boundaries

B0.1 closes the gap between the existing blind annotation UI and a coordinator-operated
pilot. It adds one auditable path from two separately exported annotator streams through
amendment resolution, completeness, disagreement review, third-human adjudication,
pre-adjudication agreement analysis, eligibility accounting, and a human-readable pilot
report. Every scientific decision still comes from humans; the software validates,
resolves append-only chains, computes declared statistics, and blocks unsafe promotion.

The repository will contain source, tests, protocol/runbook material, launchers, canonical
decision-free A/B packages, package and instruction hashes, and a handoff build
specification. It will not contain wheels, sdists, delivery-kit copies or archives,
coordinator-private copies, annotator state, human or synthetic decisions, eligibility/IAA
results, or rehearsal outputs. Generated handoff receipts and `SHA256SUMS` remain outside
Git. The package version remains `0.1.0`; B0.1 does not authorize a release.

## Verified root causes

The existing `annotation` CLI exposes only `package-pilot` and `serve`. The coordinator
workflow exists only as callable pieces, and `build_workflow_result()` consumes submitted
annotations directly without resolving `AnnotationAmendment` chains. Amendment validation
is local to one `AnnotationStore` append order, so independently returned files have no
aggregate fork, cycle, predecessor, original-hash, annotator, or task-binding audit. The
agreement primitive reports undefined kappa for a single class, but no finalization gate
checks defined/finite values, exact 40-task coverage, or the two-coefficient threshold.
Evidence agreement has no comparable-subset denominator and its low-level empty-family
identity behavior is unsafe for promotion aggregation.

## Architecture

### Coordinator collection

`rag_evidence.annotation.coordinator` will be the sole aggregate ingestion boundary. It
loads the existing `AssignmentManifest`, exactly two submission files, and exactly two
positionally paired amendment files. It validates that each submission file contains one
distinct assigned package pseudonym, every record binds to its blind task and assigned
annotator, and no task/annotator original is duplicated. Empty amendment files are valid;
missing files are not.

Amendments are resolved independently of file order. Each amendment is bound to a stored
original by canonical artifact hash, preserves task/annotator/instruction/content fields,
and points either to `null` for the chain head or the canonical hash of one amendment in
the same original chain. There must be exactly one head, at most one successor for every
node, and one traversal must consume the entire chain. Broken predecessors, forks,
cycles/disconnected components, duplicate IDs/hashes, and cross-original links fail.
The last replacement is the effective decision. Originals, all amendments, and effective
decisions are written separately.

Collection writes deterministic JSON/JSONL artifacts under `--out`: combined originals,
combined amendments, `effective-submissions.jsonl`, `collection-receipt.json`, and
`input-manifest.json` with basename, byte length, and SHA-256 for every consumed or emitted
artifact except the non-self-hashing manifest itself. Before two judgments exist for all
40 assigned tasks, collection writes the receipt/effective rows and exits blocked without
creating `disagreements.jsonl`. Complete collection writes the disagreement queue.

`build_workflow_result()` will accept optional amendments and use the same resolver so no
call path can silently ignore valid corrections. Downstream adjudication and finalization
consume effective judgments; collection preserves originals and the complete audit trail.

### Third-human adjudication

The existing `AdjudicationStore` and `create_adjudication_app()` remain authoritative.
The new `annotation adjudicate` CLI loads the assignment manifest and complete effective
submissions, then starts only the disagreement console. The API adds explicit status and
append-only JSONL export endpoints. Zero disagreements returns `complete=true`, zero
remaining, and an empty valid export.

An adjudicator pseudonym must differ from both effective annotators, as already enforced by
`AdjudicationRecord`. Submitted adjudications are immutable. B0.1 intentionally provides
no adjudication amendment schema: an erroneous submitted adjudication blocks finalization
and requires owner review; it is never silently overwritten or removed.

### Pilot finalization and verdicts

`rag_evidence.annotation.finalize` will write the complete fixed artifact set:

- `flow-accounting.json`
- `disagreements.jsonl`
- `eligibility.json`
- `iaa.json`
- `evidence-agreement.json`
- `timing-summary.json`
- `privacy-scan.json`
- `input-manifest.json`
- `pilot-verdict.json`
- `pilot-report.md`

Input and output records are privacy-scanned before public reporting. Artifacts contain
opaque pseudonyms and scientific decisions but no names, emails, private paths, hidden
transformation fields, expected labels, model outputs, or gold labels. Output timestamps
derive from the maximum validated source-artifact UTC timestamp rather than wall time, so
identical inputs produce identical bytes.

Verdict precedence is:

1. `BLOCKED_PRIVACY`
2. `BLOCKED_INTEGRITY`
3. `BLOCKED_INCOMPLETE`
4. `BLOCKED_UNRESOLVED_ADJUDICATION`
5. `BLOCKED_LOW_IAA`
6. `READY_FOR_HUMAN_FREEZE_REVIEW`

Blocked runs still write an auditable report and machine verdict where inputs can be read
safely. The tool never edits the confirmatory preregistration, opens a confirmatory sample,
or starts a model stage.

## Frozen IAA promotion rule

Answerability IAA uses all 40 tasks after valid amendments and before adjudication. The
pair order is defined by each manifest assignment; adjudicated decisions never enter these
pairs. The report must show total/complete/missing counts, raw agreement, each annotator's
prevalence, pooled prevalence, Cohen's kappa, and nominal Krippendorff's alpha.

Promotion requires both kappa and alpha to be defined, finite, and at least `0.70` with
exactly 40 complete pairs. Undefined, NaN/infinite, or sub-threshold values produce
`BLOCKED_LOW_IAA` after completeness and adjudication gates. Raw agreement and prevalence
are descriptive and cannot replace either coefficient.

Evidence agreement is computed only where both effective annotators chose `answerable`.
The artifact reports `n_total_tasks`, `n_comparable`, `n_excluded_not_both_answerable`,
exact set-family agreement rate, mean symmetric best-match Jaccard, and mean set-F1.
Empty evidence families never increment `n_comparable` or contribute a perfect score.
The strict annotation schema normally prevents an answerable empty family; aggregation
still fails closed if a corrupted input bypasses that invariant.

## Human handoff design

The committed handoff specification records only builder/spec/schema versions, supported
Python `3.11`, canonical A/B package SHA-256 values, instruction version/hash, dependency
lock hash, expected external layout, and the exact reproducible build recipe. It does not
predict a wheel hash.

The deterministic builder accepts one final wheel and canonical package sources and creates
two disjoint external directories. A receives the verified wheel, only
`ann-pilot-a.json`, the exact runbook/onboarding material, A's launcher, and A's
`SHA256SUMS`; B receives the analogous B-only files. Neither kit contains the coordinator
manifest or the other package. The external root receives `handoff-receipt.json` and a
coordinator `SHA256SUMS` containing source commit, builder, wheel, packages, runbook,
launchers, dependency lock, instruction identity, UTC build time, and clean-install
verification status.

The runbook gives exact Windows PowerShell and POSIX commands for Python 3.11 environment
creation, checksum verification, wheel installation with the app extra, private state
creation, localhost launch, JSONL export URLs, checksum generation, and safe return. It
forbids web search during annotation, PII, repository access, hidden metadata, and hash
mismatch continuation.

## Reproducibility policy

Final verification checks out the exact final commit into two different temporary
directories and builds twice with a fixed `SOURCE_DATE_EPOCH`. Canonical package bytes and
hashes must match. Wheel bytes and hashes are compared. If they differ, ZIP timestamps,
build metadata, and dependencies are isolated before any claim. If a reasonable fix cannot
make them byte-identical, receipts label the wheel `per-build-hash-verified`; source
reproducibility remains, but deterministic-wheel language is prohibited.

No generated wheel, sdist, kit, ZIP, receipt, `SHA256SUMS`, private state, decision, or
synthetic output is committed.

## Synthetic rehearsal

A shipped, explicitly synthetic rehearsal uses invented questions/passages unrelated to
the 40 pilot tasks. It builds two packages, 40 paired judgments, one valid amendment chain,
agreement and answerability/evidence disagreements, and a dataset-defect exclusion. It
exercises coordinator collection, effective resolution, disagreement artifacts,
adjudication API/status/export, finalization, IAA gates, privacy checks, artifact hashes,
and a second byte-identical run.

Negative tests cover broken/forked/cyclic amendment chains, duplicate annotators, a missing
second submission, original/adjudicator mismatch, reused original annotator as adjudicator,
undefined kappa/alpha, each coefficient below `0.70`, wrong evidence denominators, hidden
fields, and writes to the formal package directory. The rehearsal snapshots canonical
formal-package hashes before and after and refuses any output path inside `pilot/v0.2` or
`results/v2`.

## Documentation and package replacement

The protocol becomes `pilot-v0.2.1-draft`; its full-file SHA-256 binds regenerated
decision-free A/B packages. The canonical paths are safely replaced in place, and all
packages from baseline `372096b` are explicitly obsolete. README, onboarding, completion
checklist, and bilingual homepage boundaries use exact CLI commands, inputs, outputs, and
stop conditions. The confirmatory preregistration remains DRAFT.

## Verification and GitHub stop point

Verification includes targeted coordinator/workflow/CLI/API/report tests, full CPU/offline
pytest, Ruff, strict mypy, clean sdist/wheel builds, two clean wheel installations,
synthetic rehearsal, privacy scan, `git diff --check`, formal-output absence checks, and the
supported Docker precomputed smoke path. New work is committed without amending the prior
12 commits, pushed to `codex/v0.2-annotation-readiness`, and proposed to `main` in an open
PR. Work stops after GitHub CI is green; no merge, tag, release, or human pilot begins.
