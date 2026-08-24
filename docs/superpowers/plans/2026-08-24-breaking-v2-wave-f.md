# Wave F: Regression, Rehearsal, and Candidate Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans,
> superpowers:test-driven-development for defects, superpowers:systematic-debugging for failures,
> and superpowers:verification-before-completion before any terminal-state claim.

**Goal:** Prove the full breaking-v2 implementation, select one exact source candidate, and create
the Git-external receipts and disjoint kits required for separate human-pilot authorization.

**Architecture:** Static regression and synthetic rehearsal close the source tree first. The final
candidate is then immutable: push, fresh checkouts, wheel verification, platform verification,
exact-SHA CI, and handoff assembly occur in strict order. Any source change restarts candidate
evidence from checkout creation.

**Tech Stack:** pytest, Ruff, mypy, uv/Hatch, Docker, Git worktrees, GitHub CLI/Actions, SHA-256,
PowerShell, Linux CI.

## Global constraints

All constraints in `2026-08-24-breaking-v2-plan-index.md` apply.

---

### Task F1: Close static privacy, v1 rejection, docs, and full regression

**Files:**
- Modify: `README.md`, `README_en.md`, `PILOT_PROTOCOL.md`,
  `pilot/v0.2/README.md`, `pilot/v0.2/ONBOARDING.md`, `pilot/v0.2/HANDOFF_RUNBOOK.md`,
  `pilot/v0.2/COMPLETION_CHECKLIST.md`
- Modify: `tests/test_claim_consistency.py`,
  `tests/test_publication_contract.py`, `tests/test_repository_hygiene.py`
- Modify: `src/rag_evidence/annotation/privacy.py`
- Modify: `scripts/scan_annotation_export.py`
- Create: `tests/test_annotation_delivery_static_scan.py`

**Interfaces:**
- `scan_delivery_sources(repository_root: Path) -> tuple[str, ...]` scans canonical packages,
  UI/JavaScript, API fixtures, v2 schema fixtures, docs, runbooks, launchers, and receipt fixtures;
  coordinator-only source and the approved spec are excluded by an explicit allowlist.
- Public text states feasibility-only status and the exact honest blinding boundary.
- Full v1 literal matrix fails at every loader/API/builder boundary.

- [ ] **Step 1: Write failing delivery/static-claim tests**

```python
def test_all_delivery_sources_pass_static_negative_scan() -> None:
    assert scan_delivery_sources(REPOSITORY_ROOT) == ()

def test_public_claim_boundary_is_exact() -> None:
    text = public_pilot_text().casefold()
    assert "metadata" in text and "transformation" in text and "expected-label" in text
    assert "semantic unlinkability" in text and "does not" in text
    assert "independent sibling perception" in text and "does not" in text
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/test_annotation_delivery_static_scan.py tests/test_annotation_v1_rejection.py tests/test_claim_consistency.py tests/test_publication_contract.py tests/test_repository_hygiene.py -q`

Expected: FAIL on stale v1 literals, stale bootstrap/runbook wording, or delivery-source leaks.

- [ ] **Step 3: Apply only evidence-backed source/doc corrections**

Remove stale v1 delivery literals and claims. Keep historical v0.1 evidence and coordinator-only
field names intact. Do not weaken scanner exclusions to make a real delivery leak pass.

Regenerate the committed source-only handoff spec and JSON schema after every package, instruction,
runbook, launcher, lock, builder, or verifier source is final:

```powershell
uv run rag-evidence annotation write-handoff-spec --repository-root . --output pilot/v0.2/handoff-manifest.json
```

- [ ] **Step 4: Run GREEN and full quality suite**

Run in order:

```powershell
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -m "not gpu and not slow" -q --cov=rag_evidence --cov-report=term
git diff --check
```

- [ ] **Step 5: Docker regression**

Run:

```powershell
docker build -t rag-evidence:v02-candidate .
docker run --rm rag-evidence:v02-candidate rag-evidence --version
```

Expected: image builds and reports `0.2.0`; no annotation state or decision is created.

- [ ] **Step 6: Commit boundary**

Commit: `test: close breaking v2 privacy regressions`

### Task F2: Upgrade and execute the synthetic full-path rehearsal

**Files:**
- Modify: `src/rag_evidence/annotation/rehearsal.py`
- Modify: `src/rag_evidence/cli.py`
- Modify: `tests/test_annotation_rehearsal.py`
- Modify: `tests/test_cli_annotation.py`

**Interfaces:**
- `run_synthetic_rehearsal(out: Path, repository_root: Path) -> RehearsalResult` creates invented
  v2 tasks and decisions only below an external root marked `SYNTHETIC-NOT-HUMAN-DATA.txt`.
- Rehearsal covers 40 tasks, two annotators, valid amendment chains, disagreements, third-human
  adjudication, defect exclusions, IAA gating, deterministic repeat, privacy scan, and unchanged
  canonical package hashes.
- Formal package directories, repository roots, and results roots are rejected.

- [ ] **Step 1: Write failing v2 rehearsal assertions**

```python
def test_rehearsal_is_v2_group_free_and_repeatable(tmp_path: Path) -> None:
    result = run_synthetic_rehearsal(tmp_path / "synthetic", REPOSITORY_ROOT)
    assert result.tasks == 40
    assert result.repeat_byte_identical is True
    assert result.schema_versions == EXPECTED_V2_REHEARSAL_SCHEMAS
    assert scan_delivery_tree(result.output_root) == ()

def test_rehearsal_never_changes_formal_packages(tmp_path: Path) -> None:
    before = canonical_package_hashes()
    run_synthetic_rehearsal(tmp_path / "synthetic", REPOSITORY_ROOT)
    assert canonical_package_hashes() == before
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/test_annotation_rehearsal.py tests/test_cli_annotation.py -q`

Expected: FAIL because the rehearsal still creates v1/group-bearing fixtures.

- [ ] **Step 3: Implement v2-only invented rehearsal data**

Use `BlindTaskV2`, external `AssignmentManifestV2`, v2 decision/amendment/adjudication records, and
all v2 aggregate schemas. Invented decisions are explicitly synthetic test fixtures, never human
labels or formal Dataset v2 decisions.

- [ ] **Step 4: Run GREEN**

Run: `uv run pytest tests/test_annotation_rehearsal.py tests/test_annotation_coordinator.py tests/test_annotation_finalize.py tests/test_adjudication_api.py -q`

- [ ] **Step 5: Execute the external rehearsal twice**

Run:

```powershell
$SyntheticRoot = Join-Path $env:TEMP ('reab-v02-synthetic-' + [guid]::NewGuid().ToString('N'))
uv run rag-evidence annotation rehearse-synthetic --out $SyntheticRoot
```

Expected: 40 tasks and byte-identical repeated output trees; formal package hashes are unchanged.

- [ ] **Step 6: Commit boundary**

Commit: `test: rehearse the complete v2 pilot workflow`

### Task F3: Select the final candidate and create external evidence

**Files:**
- No source file changes are expected.
- Generate outside Git: two detached checkouts, external environments/caches/build roots, supplied
  and replay wheel directories, coordinator manifest, Windows/Linux evidence downloads, final A/B
  kits, `handoff-receipt.json`, `SHA256SUMS`, and retained transcripts.

**Interfaces:**
- Consumes the exact committed source candidate and all Wave A-E commands.
- Produces four byte-identical wheel identities, two validated platform receipts, clean pre/post
  records, disjoint kits, and a final `handoff-receipt-v2`.

- [ ] **Step 0: Run RED against the ineligible active worktree**

Run:

```powershell
uv run rag-evidence annotation build-wheels --checkout-a . --checkout-b . --source-commit (git rev-parse HEAD) --output (Join-Path $env:TEMP 'reab-v02-negative-preflight')
```

Expected: FAIL before build because the two checkout paths are not distinct and the active worktree
contains ignored paths. No wheel-verification success record is written.

- [ ] **Step 1: Verify and commit the complete source tree**

Run the full Task F1 suite again, inspect `git diff`, commit any intended remaining source changes
with the formal identity, and require ordinary `git status --porcelain=v1 --untracked-files=all` to
be empty. The active worktree's ignored files remain untouched and make it ineligible for evidence.

- [ ] **Step 2: Push and bind the PR candidate**

```powershell
$Candidate = git rev-parse HEAD
git push origin codex/v0.2-annotation-readiness
$PrHead = gh pr view 3 --json headRefOid --jq .headRefOid
if ($PrHead -ne $Candidate) { throw 'PR head does not match candidate' }
```

- [ ] **Step 3: Create two fresh external detached checkouts**

```powershell
$EvidenceRoot = Join-Path $env:TEMP ("reab-v02-evidence-$Candidate")
$CheckoutA = Join-Path $EvidenceRoot 'checkout-a'
$CheckoutB = Join-Path $EvidenceRoot 'checkout-b'
$CoordinatorSource = Join-Path (git rev-parse --show-toplevel) 'results/v2/raw/smoke/challenge/samples/records.jsonl'
if (-not (Test-Path -LiteralPath $CoordinatorSource -PathType Leaf)) {
    throw 'Retained coordinator-only challenge source is missing'
}
git worktree add --detach $CheckoutA $Candidate
git worktree add --detach $CheckoutB $Candidate
```

Run both exact clean commands in each checkout before any build and retain argv/result records:

```text
git status --porcelain=v1 --untracked-files=all -z
git ls-files --others --ignored --exclude-standard -z
```

Both outputs must be zero bytes.

- [ ] **Step 4: Generate the external coordinator manifest and four-wheel evidence**

```powershell
uv run rag-evidence annotation build-wheels --checkout-a $CheckoutA --checkout-b $CheckoutB --source-commit $Candidate --output (Join-Path $EvidenceRoot 'wheel-evidence')
uv run rag-evidence annotation build-coordinator-manifest --challenge-records $CoordinatorSource --package-a (Join-Path $CheckoutA 'pilot/v0.2/packages/ann-pilot-a.json') --package-b (Join-Path $CheckoutA 'pilot/v0.2/packages/ann-pilot-b.json') --output (Join-Path $EvidenceRoot 'coordinator/assignment-manifest-v2.json')
```

Require supplied A/B and replay A/B wheels to share filename, size, SHA-256, and bytes. Re-run both
clean commands after builds and require zero-byte outputs.

- [ ] **Step 5: Execute platform verification and wait for exact-SHA CI**

Run native Windows verification against the verified wheel, then wait for PR jobs `checks`,
`docker`, `annotation-windows`, and `annotation-linux` for `$Candidate`:

```powershell
gh pr checks 3 --watch --fail-fast --interval 10
```

Locate the successful CI run whose `headSha` equals `$Candidate`, download both platform artifacts
below `$EvidenceRoot/platform-ci`, and validate their source/tree/wheel/lock/protocol/spec identities
and generated smoke hashes. If the CI wheel hash differs from the local verified wheel, stop the
handoff and debug reproducibility; do not rewrite a receipt.

- [ ] **Step 6: Run GREEN by assembling and revalidating final disjoint kits**

```powershell
uv run rag-evidence annotation build-handoff --spec pilot/v0.2/handoff-manifest.json --checkout-a $CheckoutA --checkout-b $CheckoutB --wheel-evidence (Join-Path $EvidenceRoot 'wheel-evidence/wheel-verification.json') --coordinator-manifest (Join-Path $EvidenceRoot 'coordinator/assignment-manifest-v2.json') --windows-receipt (Join-Path $EvidenceRoot 'platform-ci/windows/platform-verification-receipt.json') --linux-receipt (Join-Path $EvidenceRoot 'platform-ci/linux/platform-verification-receipt.json') --output (Join-Path $EvidenceRoot 'handoff')
```

Verify every `SHA256SUMS`, ensure kit A has only A package/launcher and kit B only B, scan both kits
for hidden/private data and decisions, and verify the root receipt against the exact Git candidate.

- [ ] **Step 7: Final repository and PR verification**

Require local/remote candidate equality, ahead/behind `0/0`, PR OPEN/non-draft, all exact-SHA checks
green, standard worktree status empty, and no prohibited generated artifact tracked. Report active
ignored-path presence honestly; it does not invalidate the external clean checkouts.

- [ ] **Step 8: Terminal boundary**

Do not remove worktrees or retained/superseded evidence without separate approval. Do not merge,
tag, release, publish, or start a human pilot. Stop only at:

The owner-approved stable closure supersedes this pre-release terminal. The v0.2.0 status is
`HUMAN_PILOT_NOT_CONDUCTED`; future human execution requires separate authorization.
