# v0.2 Annotation-Ready Engineering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare the repository for a 20-parent independent-human annotation pilot while
leaving every formal v2 model run, human decision, and release action unexecuted.

**Architecture:** Add a strict `rag_evidence.annotation` boundary for blind projections,
append-only decisions, dual assignment, adjudication, agreement, and clustered analysis.
Reuse FastAPI for a minimal offline UI and wrap existing generation/attribution/evaluation
stages with a fail-closed challenge execution context. Preserve v1 prompt/artifact readers
and keep pilot, confirmatory, and natural output roots physically distinct.

**Tech Stack:** Python 3.11, Pydantic 2 strict models, FastAPI/TestClient, Typer, NumPy,
pytest, Ruff, strict mypy, Hatch/uv, Docker.

## Global Constraints

- Work only on `codex/v0.2-annotation-readiness` in the isolated worktree.
- Do not push, merge, tag, release, mutate remote metadata, use a GPU, call a paid API,
  or generate model-backed outputs.
- Do not create human labels, infer labels from transformations, or allow expected labels
  into annotator payloads.
- Keep the historical v0.1 tag and artifacts unchanged and keep prompt v1 readable.
- Use synthetic fixtures and the fake backend for execution tests.
- Every new behavior follows RED -> GREEN -> REFACTOR and receives a focused commit.

---

### Task 1: Public Claims, Compact READMEs, and Protocol Documents

**Files:**
- Modify: `MODEL_CARD.md`
- Replace: `README.md`, `README_en.md`
- Create: `docs/HISTORICAL_V01_EVIDENCE_ZH.md`, `docs/HISTORICAL_V01_EVIDENCE_EN.md`
- Create: `PILOT_PROTOCOL.md`, `PREREGISTRATION_V2_CONFIRMATORY_DRAFT.md`
- Create: `docs/GITHUB_HANDOFF.md`
- Test: `tests/test_claim_consistency.py`, `tests/test_publication_contract.py`

**Interfaces:**
- Produces the public vocabulary consumed by report templates and claim-contract tests:
  `deletion-based teacher-forced target-dependence diagnostic`, `experimental
  distributional-dependence diagnostic`, and `construct validation`.
- Produces protocol identifiers `pilot-v0.2-draft` and `v2-confirmatory-draft` used by
  annotation instruction hashes and package manifests.

- [ ] **Step 1: Write failing public-contract tests**

  Assert that current-branch public docs reject `causal (primary baseline)`, ARC-JSD as
  causal, and unqualified sufficiency/comprehensiveness claims; assert both homepage
  READMEs are 250--350 lines, contain the historical-v0.1 / unexecuted-v2 boundary, and
  link detailed evidence and both protocol layers.

- [ ] **Step 2: Run the focused tests and observe contract failures**

  Run: `uv run pytest tests/test_claim_consistency.py tests/test_publication_contract.py -q`

- [ ] **Step 3: Preserve detailed evidence and replace the homepage documents**

  Move the existing long READMEs to the two historical-evidence documents, create compact
  equivalent Chinese/English entrances in the required order, correct MODEL_CARD labels,
  and add the pilot, DRAFT preregistration, and GitHub description/topic handoff. The DRAFT
  must state all four freeze preconditions, 160-parent allocation, nested design,
  endpoints, exclusions, Holm family, 10,000 cluster bootstrap, IAA gate, and fixed
  stopping rule.

- [ ] **Step 4: Run claim/publication tests and repository link checks**

  Run: `uv run pytest tests/test_claim_consistency.py tests/test_publication_contract.py -q`

- [ ] **Step 5: Commit**

  Commit message: `docs: align public claims and separate pilot protocol`

### Task 2: Strict Annotation and Eligibility Schemas

**Files:**
- Create: `src/rag_evidence/annotation/__init__.py`
- Create: `src/rag_evidence/annotation/models.py`
- Test: `tests/test_annotation_schema.py`

**Interfaces:**
- Produces strict frozen Pydantic models `BlindTask`, `TaskPassage`, `TaskSentence`,
  `AnswerabilityAnnotation`, `CitationAnnotation`, `AnnotationAmendment`,
  `AdjudicationRecord`, `EligibilityRecord`, and `EligibilityArtifact`.
- `AnswerabilityAnnotation` requires one or more nonempty minimal evidence sets only for an
  answerable decision, forbids answer text/evidence for unanswerable decisions, accepts
  explicit `unclear`, validates pseudonyms, hashes, UTC timestamps, confidence, and
  distinct evidence aliases.
- `CitationAnnotation` keeps answer correctness and citation support separate.
- Eligibility requires two independent annotation hashes and an adjudication hash when
  decisions disagree; it never derives from `expected_answerability`.

- [ ] **Step 1: Write literal valid/invalid fixture tests**

  Cover unknown fields, PII-like annotator identities, invalid hashes/timestamps, evidence
  alias duplication, answerability-dependent fields, citation support states, immutable
  originals, and fail-closed eligibility.

- [ ] **Step 2: Verify the schema tests fail because the module is absent**

  Run: `uv run pytest tests/test_annotation_schema.py -q`

- [ ] **Step 3: Implement the minimal strict models and canonical SHA-256 helpers**

  Hash canonical JSON with sorted keys and UTF-8. Use `extra='forbid'`, frozen records,
  literal schema versions, and validators that reject email/name-shaped pseudonyms.

- [ ] **Step 4: Run the schema tests**

  Run: `uv run pytest tests/test_annotation_schema.py -q`

- [ ] **Step 5: Commit**

  Commit message: `feat: add fail-closed annotation schemas`

### Task 3: Server-Side Blind Projection and Deterministic Dual Assignment

**Files:**
- Create: `src/rag_evidence/annotation/blinding.py`
- Create: `src/rag_evidence/annotation/assignment.py`
- Test: `tests/test_annotation_blinding.py`, `tests/test_annotation_assignment.py`

**Interfaces:**
- `project_challenge(record, *, instruction_version, instruction_hash,
  batch, namespace) -> BlindTask` is an explicit allowlist projection.
- Passage aliases are `P1...`; sentence aliases are `P1.S1...`; internal IDs and gold
  flags never cross the boundary. Parent groups are one-way SHA-256 pseudonyms.
- `build_dual_assignments(tasks, annotators, seed) -> AssignmentManifest` gives every task
  to exactly two distinct annotators, produces assignment-specific task IDs, randomizes
  deterministically, and ensures siblings are not adjacent per annotator.
- `scan_blind_payload(payload)` recursively rejects forbidden key names, internal IDs,
  email addresses, model/method/score fields, or expected/adjudicated labels.

- [ ] **Step 1: Write leakage and scheduling tests against real-shaped synthetic records**

  Test both API JSON and exported JSONL, deterministic repeatability, changed-seed order,
  exact dual coverage, sibling non-adjacency, and an impossible-schedule error.

- [ ] **Step 2: Run both tests and observe missing-interface failures**

  Run: `uv run pytest tests/test_annotation_blinding.py tests/test_annotation_assignment.py -q`

- [ ] **Step 3: Implement allowlist projection, aliases, leak scan, and scheduler**

  Do not call `ChallengeRecord.to_json()` and delete keys; construct the blind payload
  field-by-field from question, public titles, and sentences.

- [ ] **Step 4: Run both test modules**

  Run: `uv run pytest tests/test_annotation_blinding.py tests/test_annotation_assignment.py -q`

- [ ] **Step 5: Commit**

  Commit message: `feat: build blind dual-annotation assignments`

### Task 4: Append-Only Store and Offline Annotation UI

**Files:**
- Create: `src/rag_evidence/annotation/store.py`
- Create: `src/rag_evidence/annotation/app.py`
- Create: `src/rag_evidence/annotation/ui.html`
- Test: `tests/test_annotation_store.py`, `tests/test_annotation_api.py`

**Interfaces:**
- `AnnotationStore(root, manifest)` permits mutable autosave drafts but append-only
  submissions, amendments, and adjudications; duplicate submission IDs and changes to
  bound task/instruction hashes fail.
- `create_annotation_app(package_dir, state_dir) -> FastAPI` exposes only blind task,
  draft, submit, amend, progress, export, disagreement, and adjudication endpoints.
- The HTML client renders passages/sentence checkboxes, multiple evidence sets, autosaves,
  resumes, validates, displays progress/instruction hash, and submits immutable records.

- [ ] **Step 1: Write store and TestClient failures**

  Assert restart/resume, autosave replacement, immutable submission, amendment chain,
  validation errors, progress counts, browser/API payload leak absence, offline static UI,
  and JSON/JSONL export.

- [ ] **Step 2: Run focused tests and observe failures**

  Run: `uv run pytest tests/test_annotation_store.py tests/test_annotation_api.py -q`

- [ ] **Step 3: Implement storage and minimal HTML/API behavior**

  All artifact writes use canonical JSON, atomic draft replacement, or append+fsync. The
  UI never receives source challenge records or repository paths.

- [ ] **Step 4: Run focused tests**

  Run: `uv run pytest tests/test_annotation_store.py tests/test_annotation_api.py -q`

- [ ] **Step 5: Commit**

  Commit message: `feat: add offline append-only annotation console`

### Task 5: Completeness, Disagreement, Adjudication, and Flow Accounting

**Files:**
- Create: `src/rag_evidence/annotation/workflow.py`
- Test: `tests/test_annotation_workflow.py`

**Interfaces:**
- `build_disagreement_queue(assignments, submissions) -> list[DisagreementCase]` refuses
  missing or duplicate judgments, preserves both originals/reasons, and compares
  answerability plus normalized minimal evidence sets.
- `build_eligibility_artifact(...)` emits `assigned/completed/disagreed/adjudicated/
  excluded/eligible` counts and requires independent human adjudication for disagreement.
- `unclear` remains unclear unless an adjudication record explicitly decides otherwise.

- [ ] **Step 1: Write hand-constructed flow and integrity tests**

  Include agreement without adjudication, disagreement blocked without adjudication,
  hash-mismatched adjudication, excluded defect, unresolved unclear, and exact flow totals.

- [ ] **Step 2: Verify failures**

  Run: `uv run pytest tests/test_annotation_workflow.py -q`

- [ ] **Step 3: Implement workflow gates**

  Compare submitted records, never challenge transformations. Bind every eligibility row
  to source task/submission/adjudication hashes.

- [ ] **Step 4: Run workflow tests**

  Run: `uv run pytest tests/test_annotation_workflow.py -q`

- [ ] **Step 5: Commit**

  Commit message: `feat: enforce dual-review adjudication workflow`

### Task 6: IAA, Evidence Agreement, and Cluster Statistics

**Files:**
- Create: `src/rag_evidence/annotation/agreement.py`
- Create: `src/rag_evidence/annotation/statistics.py`
- Test: `tests/test_annotation_agreement.py`, `tests/test_annotation_statistics.py`

**Interfaces:**
- `nominal_agreement(pairs)` returns pair count, complete/missing counts, label prevalence,
  raw agreement, Cohen kappa, and Krippendorff alpha without treating missing as negative.
- `evidence_agreement(left_sets, right_sets)` returns exact set-family agreement and
  symmetric best-match Jaccard/set-F1.
- `wilson_interval(successes, total, confidence)` uses the score interval.
- `cluster_bootstrap_difference(observations, ..., resamples=10_000)` samples leakage
  groups, retaining all parents and nested variants in sampled groups.
- `holm_adjust(p_values)` implements the fixed multiplicity family.

- [ ] **Step 1: Write hand-calculated fixtures**

  Use literal expected values for perfect agreement, imbalanced prevalence, missing pairs,
  partial evidence overlap, Wilson boundaries, cluster-vs-row bootstrap divergence,
  deterministic seed, and Holm monotonicity.

- [ ] **Step 2: Verify focused failures**

  Run: `uv run pytest tests/test_annotation_agreement.py tests/test_annotation_statistics.py -q`

- [ ] **Step 3: Implement metrics without sklearn estimators**

  Keep formulas auditable and validate probabilities, cluster/parent/variant identity,
  resample counts, and nonempty complete cases.

- [ ] **Step 4: Run focused tests**

  Run: `uv run pytest tests/test_annotation_agreement.py tests/test_annotation_statistics.py -q`

- [ ] **Step 5: Commit**

  Commit message: `feat: add auditable IAA and clustered statistics`

### Task 7: Sentence-Citation Prompt v2 and Runtime Binding

**Files:**
- Modify: `src/rag_evidence/generation/prompts.py`
- Modify: `src/rag_evidence/generation/citations.py`
- Modify: `src/rag_evidence/generation/run.py`
- Modify: `src/rag_evidence/config.py`, `src/rag_evidence/storage/runmeta.py`
- Modify: `configs/v2/smoke.yaml`, `configs/v2/dev.yaml`, `configs/v2/eval.yaml`
- Test: `tests/test_sentence_citations_v2.py`, `tests/test_config.py`, `tests/test_runmeta.py`

**Interfaces:**
- Prompt registry keeps v1 unchanged and adds v2 with `[P#.S#]` aliases and exact hash.
- `parse_sentence_citation_response(text, passage_alias_map, sentence_alias_map)` returns
  separate answer text, valid sentence IDs, invalid aliases, and `missing_citations`.
- Generation records retain legacy passage citations and add a versioned sentence-citation
  block for v2.
- v2 generation config binds exact generator/tokenizer revisions, optional local artifact
  SHA-256, runtime identity, decoding settings, dtype/quantization, prompt hash, and
  device/CUDA metadata in run_meta.

- [ ] **Step 1: Write failing prompt/parser/config/runmeta tests**

  Cover stable aliases under ablation, mixed valid/invalid/missing citations, answer/citation
  separation, unchanged v1 hash, exact 40-hex revisions, 64-hex local hash, deterministic
  decoding fields, and recorded runtime/device metadata.

- [ ] **Step 2: Observe focused failures**

  Run: `uv run pytest tests/test_sentence_citations_v2.py tests/test_config.py tests/test_runmeta.py -q`

- [ ] **Step 3: Implement the compatible v2 path and pin v2 configs**

  Do not mutate any historical v1 config or results. `local_artifact_sha256` may be an
  explicit future-run requirement rather than a fabricated current hash.

- [ ] **Step 4: Run focused tests**

  Run: `uv run pytest tests/test_sentence_citations_v2.py tests/test_config.py tests/test_runmeta.py -q`

- [ ] **Step 5: Commit**

  Commit message: `feat: bind sentence citations and v2 runtime provenance`

### Task 8: Fail-Closed Challenge Stage Integration

**Files:**
- Create: `src/rag_evidence/data/challenge_execution.py`
- Modify: `src/rag_evidence/config.py`, `src/rag_evidence/cli.py`
- Modify: `src/rag_evidence/data/hotpot.py`
- Modify: `src/rag_evidence/generation/run.py`, `src/rag_evidence/attribution/run.py`
- Modify: `src/rag_evidence/evaluation/evaluate.py`, `src/rag_evidence/reporting/report.py`
- Test: `tests/test_challenge_execution.py`, `tests/test_challenge_e2e.py`

**Interfaces:**
- `ExecutionConfig` selects `natural` or `challenge`, phase `pilot|confirmatory`, exactly
  one variant `missing_hop|evidence_swap`, and an eligibility artifact path.
- `challenge generate|attribute|evaluate|report` creates phase/variant-specific roots and
  delegates to existing stages after `load_eligible_challenge_examples` validates content
  hashes and dual-human eligibility.
- Pending/provisional records fail before run directories are created. Every stage record
  or linkage artifact retains blinded parent group, internal parent link for analysis, and
  variant; reports never merge natural or sibling variants as independent rows.

- [ ] **Step 1: Write preflight and fake-backend e2e tests**

  Assert pending rows are blocked, expected labels cannot satisfy eligibility, bad hashes
  fail, pilot/confirmatory paths differ, variants differ, parent linkage survives, and
  fake generate -> citations attribute -> evaluate -> report completes on synthetic rows.

- [ ] **Step 2: Run the new tests and observe failures**

  Run: `uv run pytest tests/test_challenge_execution.py tests/test_challenge_e2e.py -q`

- [ ] **Step 3: Implement execution config, loaders, CLI wrappers, and stage linkage**

  Reuse all existing run integrity behavior. Do not add a real-model invocation to tests
  and do not create formal-v2 output directories in the repository.

- [ ] **Step 4: Run focused and existing e2e tests**

  Run: `uv run pytest tests/test_challenge_execution.py tests/test_challenge_e2e.py tests/test_e2e_tiny.py -q`

- [ ] **Step 5: Commit**

  Commit message: `feat: integrate human-gated challenge stages`

### Task 9: Construct Gates and Fail-Closed Reporting

**Files:**
- Modify: `src/rag_evidence/evaluation/construct.py`
- Modify: `src/rag_evidence/evaluation/schema.py`
- Modify: `src/rag_evidence/reporting/report.py`
- Modify: `tests/test_construct_validation.py`, `tests/test_reporting_v2.py`

**Interfaces:**
- The existing `oracle_gold`, `control_random`, and `control_answer_string` inputs remain
  required; empty/ineligible/partial control artifacts return `not_run` or `failed`.
- Public output calls this `construct validation`, prints `NOT RUN`, `FAILED`, or `PASSED`,
  and labels sufficiency/comprehensiveness as diagnostics regardless of status.

- [ ] **Step 1: Add failing ineligible/partial/report wording tests**
- [ ] **Step 2: Run focused tests and observe failures**

  Run: `uv run pytest tests/test_construct_validation.py tests/test_reporting_v2.py -q`

- [ ] **Step 3: Implement explicit eligibility and status propagation**
- [ ] **Step 4: Run focused tests**

  Run: `uv run pytest tests/test_construct_validation.py tests/test_reporting_v2.py -q`

- [ ] **Step 5: Commit**

  Commit message: `fix: fail closed on construct validation`

### Task 10: Pilot Package, CLI, Privacy Scan, and Verification

**Files:**
- Create: `src/rag_evidence/annotation/package.py`
- Modify: `src/rag_evidence/cli.py`
- Create: `pilot/v0.2/README.md`, `pilot/v0.2/ONBOARDING.md`,
  `pilot/v0.2/COMPLETION_CHECKLIST.md`
- Generate: `pilot/v0.2/packages/manifest.json`, anonymized annotator JSON packages
- Create: `scripts/scan_annotation_export.py`
- Test: `tests/test_pilot_package.py`, `tests/test_clean_export.py`,
  `tests/test_cli_annotation.py`, `tests/test_repository_hygiene.py`

**Interfaces:**
- `annotation package-pilot` mechanically selects the 20 smoke parents and both reviewed
  challenge variants, projects and dual-assigns them, leak-scans every artifact, and writes
  zero annotation/adjudication/eligibility decisions.
- `annotation serve` starts the local console from a chosen package/state directory.
- The privacy scanner exits nonzero for hidden keys, internal IDs, PII, expected labels,
  annotations in a clean package, or repository paths.

- [ ] **Step 1: Write package/CLI/privacy failures**

  Assert 20 parents, 40 tasks, two packages, exact dual coverage, zero labels, zero hidden
  fields, deterministic byte output, instruction hash, no formal sample output, and scanner
  rejection of deliberately poisoned exports.

- [ ] **Step 2: Run focused tests and observe failures**

  Run: `uv run pytest tests/test_pilot_package.py tests/test_clean_export.py tests/test_cli_annotation.py -q`

- [ ] **Step 3: Implement package builder, commands, documents, and scanner**

  Generate packages from challenge records without reading answers into a decision
  function. Do not create an annotator-visible expected-label or provenance package.

- [ ] **Step 4: Generate the real 20-parent package and scan it**

  Run: `uv run rag-evidence annotation package-pilot --config configs/v2/smoke.yaml --out pilot/v0.2/packages`

  Run: `uv run python scripts/scan_annotation_export.py pilot/v0.2/packages`

- [ ] **Step 5: Run all required verification fresh**

  Run in order:

  ```powershell
  uv run ruff format --check .
  uv run ruff check .
  uv run mypy src
  uv run pytest -m "not gpu and not slow" -q
  uv build
  docker build -t rag-evidence:v02-b0 .
  docker run --rm -d --name rag-evidence-v02-b0 -p 18080:8000 rag-evidence:v02-b0
  # Probe /health and /methods, then stop the named container.
  ```

- [ ] **Step 6: Verify repository boundaries and commit**

  Confirm `main` remains at `935b396...`, no formal v2 model output was created, no GPU/API
  call occurred, `git diff --check` passes, the feature worktree is clean, and only the
  feature branch has new commits.

  Commit message: `feat: deliver annotation-ready pilot tooling`

## Handoff

Keep the branch and worktree intact. Report all twelve user-requested items, including
fresh command evidence and a conservative pilot readiness decision. Do not present merge,
push, tag, or release as an executed action.
