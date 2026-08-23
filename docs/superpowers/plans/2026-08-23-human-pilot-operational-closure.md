# Human Pilot Operational Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a decision-free B0.1 coordinator, adjudication, finalization, handoff, and synthetic-rehearsal path that makes the v0.2 pilot operationally ready without starting any human or model run.

**Architecture:** Keep aggregate ingestion, scientific reporting, external handoff construction, and synthetic fixtures in separate modules with strict Pydantic contracts and deterministic JSON/JSONL serialization. All downstream paths consume amendment-resolved effective submissions; the fixed 40-task pre-adjudication IAA gate blocks promotion unless both coefficients are finite and at least 0.70. Git stores only sources and canonical decision-free packages, while wheels, delivery kits, receipts, decisions, and rehearsal outputs remain external.

**Tech Stack:** Python 3.11, Pydantic 2, Typer, FastAPI, Hatchling/uv, pytest, Ruff, strict mypy, Docker precomputed smoke test

## Global Constraints

- Keep project package version at `0.1.0`; set only the annotation instruction identity to `pilot-v0.2.1-draft`.
- Do not generate human/model judgments, open a confirmatory sample, edit the frozen confirmatory preregistration, run Qwen, use a GPU, call large APIs, merge, tag, release, or publish to Hugging Face.
- Do not commit wheels, sdists, copied A/B delivery directories or ZIPs, coordinator-private manifests, state, submissions, amendments, adjudications, eligibility/IAA results, synthetic outputs, or any human decision.
- The committed handoff manifest records builder/spec/schema versions, canonical package hashes, instruction hash, expected layout, and reproducible recipe; it must not contain a predicted wheel hash.
- A and B delivery kits must be disjoint: each receives the same verified wheel, only its own package, shared onboarding/runbook, only its own launcher, and only its own `SHA256SUMS`; neither receives the other package or the coordinator manifest.
- Use exactly two submission files and exactly two positionally paired amendment files. A present empty amendment file is valid; a missing amendment file is invalid.
- Compute answerability IAA over all 40 post-amendment, pre-adjudication pairs. Promotion requires finite, defined Cohen's kappa and nominal Krippendorff's alpha, each `>= 0.70`.
- Compute evidence agreement only when both effective annotations are answerable and both evidence families are nonempty; empty families never contribute a perfect score.
- Write source-derived timestamps and canonical sorted JSON so identical inputs produce identical bytes.
- Make new independent commits only; do not amend the 12 pre-B0.1 commits.
- Push only `codex/v0.2-annotation-readiness`, open a PR to `main`, wait for green CI, and stop without merge.

---

## File map

- `src/rag_evidence/annotation/privacy.py`: public reusable recursive privacy scanner for coordinator and report artifacts.
- `src/rag_evidence/annotation/coordinator.py`: strict file loading, two-stream collection, aggregate amendment graph validation, effective-decision resolution, deterministic collection artifacts.
- `src/rag_evidence/annotation/workflow.py`: disagreement/eligibility workflow consuming optional amendments through the shared resolver.
- `src/rag_evidence/annotation/agreement.py`: aggregate evidence-denominator report in addition to the existing pairwise primitive.
- `src/rag_evidence/annotation/finalize.py`: fixed IAA gate, privacy/integrity/completeness verdict precedence, final artifact schemas and Markdown report.
- `src/rag_evidence/annotation/handoff.py`: committed-spec validation and external A/B kit/receipt construction.
- `src/rag_evidence/annotation/rehearsal.py`: invented 40-task synthetic end-to-end fixture and deterministic rehearsal runner.
- `src/rag_evidence/annotation/app.py`: adjudication status and canonical JSONL export endpoints.
- `src/rag_evidence/cli.py`: `collect`, `adjudicate`, `finalize-pilot`, `build-handoff`, and `rehearse-synthetic` commands.
- `pilot/v0.2/handoff-manifest.schema.json`: JSON Schema for the committed decision-free handoff manifest.
- `pilot/v0.2/handoff-manifest.json`: canonical hashes, instruction identity, supported environment, external layout, and recipe; no wheel hash.
- `pilot/v0.2/launchers/start-a.ps1`, `start-a.sh`, `start-b.ps1`, `start-b.sh`: kit-local, repository-independent launchers.
- `pilot/v0.2/HANDOFF_RUNBOOK.md`: exact Windows/POSIX installation, verification, export, return, privacy, and stop commands.
- `PILOT_PROTOCOL.md`, `pilot/v0.2/README.md`, `pilot/v0.2/ONBOARDING.md`, `pilot/v0.2/COMPLETION_CHECKLIST.md`, `README.md`, `README_en.md`: protocol identity and honest completion/public boundaries.
- `pilot/v0.2/packages/*.json`: regenerated decision-free package bytes bound to the new full instruction hash.
- `tests/test_annotation_coordinator.py`, `tests/test_annotation_finalize.py`, `tests/test_annotation_handoff.py`, `tests/test_annotation_rehearsal.py`: focused contracts and negative cases.
- `tests/test_annotation_workflow.py`, `tests/test_adjudication_api.py`, `tests/test_cli_annotation.py`, `tests/test_annotation_agreement.py`, `tests/test_annotation_store.py`, `tests/test_pilot_package.py`, `tests/test_repository_hygiene.py`: existing surfaces extended for B0.1.

### Task 1: Aggregate amendment resolution and coordinator collection

**Files:**
- Create: `src/rag_evidence/annotation/privacy.py`
- Create: `src/rag_evidence/annotation/coordinator.py`
- Modify: `src/rag_evidence/annotation/store.py`
- Modify: `src/rag_evidence/annotation/workflow.py`
- Modify: `src/rag_evidence/cli.py`
- Create: `tests/test_annotation_coordinator.py`
- Modify: `tests/test_annotation_store.py`
- Modify: `tests/test_annotation_workflow.py`
- Modify: `tests/test_cli_annotation.py`

**Interfaces:**
- Consumes: `AssignmentManifest`, `AnswerabilityAnnotation`, `AnnotationAmendment`, `artifact_hash()`, `build_disagreement_queue()`.
- Produces: `scan_private_payload(payload: object) -> tuple[str, ...]`, `resolve_amendments(manifest, originals, amendments) -> tuple[AnswerabilityAnnotation, ...]`, `collect_annotation_streams(manifest_path, submission_paths, amendment_paths, out) -> CollectionResult`, and `build_workflow_result(..., amendments: Sequence[AnnotationAmendment] = ())`.

- [ ] **Step 1: Write failing privacy and graph tests**

```python
def test_resolve_amendments_is_order_independent_and_uses_chain_tip() -> None:
    effective = resolve_amendments(manifest, originals, [second, first])
    assert effective_for(effective, original) == second.replacement

@pytest.mark.parametrize("kind", ["broken", "fork", "cycle", "disconnected"])
def test_resolve_amendments_rejects_invalid_graphs(kind: str) -> None:
    with pytest.raises(DataError, match=kind_pattern(kind)):
        resolve_amendments(manifest, originals, invalid_chain(kind))

def test_public_privacy_scanner_rejects_email_posix_and_windows_paths() -> None:
    assert scan_private_payload({"text": "x@y.test"})
    assert scan_private_payload({"text": "/Users/alice/private/state.json"})
    assert scan_private_payload({"text": r"C:\\Users\\alice\\state.json"})
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `uv run pytest tests/test_annotation_coordinator.py tests/test_annotation_store.py -q`

Expected: collection errors because `privacy.py`, `coordinator.py`, and the public scanner do not exist.

- [ ] **Step 3: Implement the public scanner and order-independent chain resolver**

```python
def resolve_amendments(
    manifest: AssignmentManifest,
    originals: Sequence[AnswerabilityAnnotation],
    amendments: Sequence[AnnotationAmendment],
) -> tuple[AnswerabilityAnnotation, ...]:
    indexed = index_submissions(manifest, originals)
    # Bind each amendment to exactly one canonical original; enforce identical
    # task, annotator, instruction, content, batch, and schema fields.
    # Build predecessor->successor maps per original, require one null head,
    # reject reused IDs/hashes, forks, cycles, broken/cross-chain links, then
    # traverse every node and return the final replacement in manifest order.
```

Move `_PRIVATE_KEYS` and the recursive scanner from `store.py` into `privacy.py`; detect private POSIX home paths as well as Windows drive paths. Keep `AnnotationStore._scan()` delegating to the public function so local behavior remains fail-closed.

- [ ] **Step 4: Run graph/privacy tests and verify GREEN**

Run: `uv run pytest tests/test_annotation_coordinator.py tests/test_annotation_store.py -q`

Expected: valid unordered chains pass; broken, forked, cyclic, disconnected, duplicate, cross-original, pseudonym, and binding cases fail with `DataError`.

- [ ] **Step 5: Write failing collection and workflow tests**

```python
def test_collect_requires_two_submissions_and_two_present_amendment_files(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="exactly two"):
        collect_annotation_streams(manifest_path, [a], [a_amendments], tmp_path / "out")

def test_incomplete_collection_writes_receipt_but_no_disagreement_queue(tmp_path: Path) -> None:
    result = collect_annotation_streams(manifest_path, [a, b_partial], [empty_a, empty_b], out)
    assert result.complete is False
    assert (out / "collection-receipt.json").exists()
    assert not (out / "disagreements.jsonl").exists()

def test_workflow_uses_effective_amendment() -> None:
    result = build_workflow_result(manifest, originals, [], amendments=[amendment], **meta)
    assert result.disagreements == ()
```

- [ ] **Step 6: Run collection/workflow tests and verify RED**

Run: `uv run pytest tests/test_annotation_coordinator.py tests/test_annotation_workflow.py -q`

Expected: failures because collection and workflow amendment parameters are absent.

- [ ] **Step 7: Implement deterministic collection and wire workflow/CLI**

```python
@annotation_app.command("collect")
def annotation_collect(
    manifest: Path,
    submission: list[Path] = typer.Option(..., "--submission"),
    amendment: list[Path] = typer.Option(..., "--amendment"),
    out: Path = typer.Option(..., "--out"),
) -> None:
    collect_annotation_streams(manifest, submission, amendment, out)
```

Canonicalize all records by manifest task order and assigned annotator order. Write `original-submissions.jsonl`, `amendments.jsonl`, `effective-submissions.jsonl`, `collection-receipt.json`, and `input-manifest.json`; write `disagreements.jsonl` only after all 40 tasks have two effective judgments. Manifest entries contain only basename, byte size, and SHA-256, never absolute paths.

- [ ] **Step 8: Run coordinator, workflow, and CLI tests**

Run: `uv run pytest tests/test_annotation_coordinator.py tests/test_annotation_workflow.py tests/test_cli_annotation.py -q`

Expected: PASS and CLI surface includes `collect` with four required option families.

- [ ] **Step 9: Commit Task 1**

```powershell
git add src/rag_evidence/annotation/privacy.py src/rag_evidence/annotation/coordinator.py src/rag_evidence/annotation/store.py src/rag_evidence/annotation/workflow.py src/rag_evidence/cli.py tests/test_annotation_coordinator.py tests/test_annotation_store.py tests/test_annotation_workflow.py tests/test_cli_annotation.py
git commit -m "feat: collect and resolve annotation amendments"
```

### Task 2: Third-human adjudication operations

**Files:**
- Modify: `src/rag_evidence/annotation/workflow.py`
- Modify: `src/rag_evidence/annotation/app.py`
- Modify: `src/rag_evidence/cli.py`
- Modify: `tests/test_adjudication_api.py`
- Modify: `tests/test_cli_annotation.py`

**Interfaces:**
- Consumes: complete `effective-submissions.jsonl`, existing `AdjudicationStore.submit()` and immutable `AdjudicationRecord`.
- Produces: `AdjudicationStore.status() -> dict[str, int | bool]`, `AdjudicationStore.export_jsonl() -> str`, `/api/status`, `/api/export/adjudications.jsonl`, and `annotation adjudicate`.

- [ ] **Step 1: Write failing status/export/API tests**

```python
def test_zero_disagreement_status_and_export_are_complete(client: TestClient) -> None:
    assert client.get("/api/status").json() == {
        "total": 0, "adjudicated": 0, "remaining": 0, "complete": True
    }
    response = client.get("/api/export/adjudications.jsonl")
    assert response.status_code == 200
    assert response.text == ""

def test_adjudication_export_is_canonical_append_only_jsonl(client: TestClient) -> None:
    client.post("/api/adjudications", json=record)
    assert client.get("/api/export/adjudications.jsonl").text.endswith("\n")
```

- [ ] **Step 2: Run adjudication tests and verify RED**

Run: `uv run pytest tests/test_adjudication_api.py -q`

Expected: `/api/status` and JSONL export return 404.

- [ ] **Step 3: Implement status/export and coordinator-only CLI**

```python
@annotation_app.command("adjudicate")
def annotation_adjudicate(
    manifest: Path = typer.Option(..., "--manifest"),
    effective: Path = typer.Option(..., "--effective"),
    state: Path = typer.Option(..., "--state"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8002, "--port", min=1, max=65535),
) -> None:
    uvicorn.run(create_adjudication_app(manifest, effective, state), host=host, port=port)
```

Status counts only the immutable disagreement queue, and export serializes sorted-key compact JSON in stored append order. Do not add update, delete, or correction endpoints.

- [ ] **Step 4: Run API and CLI tests**

Run: `uv run pytest tests/test_adjudication_api.py tests/test_cli_annotation.py -q`

Expected: PASS, including third-pseudonym, original-hash mismatch, duplicate, and zero-disagreement cases.

- [ ] **Step 5: Commit Task 2**

```powershell
git add src/rag_evidence/annotation/workflow.py src/rag_evidence/annotation/app.py src/rag_evidence/cli.py tests/test_adjudication_api.py tests/test_cli_annotation.py
git commit -m "feat: expose third-human adjudication operations"
```

### Task 3: Fixed IAA gate and pilot finalization

**Files:**
- Modify: `src/rag_evidence/annotation/agreement.py`
- Create: `src/rag_evidence/annotation/finalize.py`
- Modify: `src/rag_evidence/cli.py`
- Modify: `tests/test_annotation_agreement.py`
- Create: `tests/test_annotation_finalize.py`
- Modify: `tests/test_cli_annotation.py`

**Interfaces:**
- Consumes: manifest, originals, amendments, adjudications, protocol bytes, `resolve_amendments()`, `build_workflow_result()`, `nominal_agreement()`.
- Produces: `aggregate_evidence_agreement(pairs) -> EvidenceAgreementSummary`, `finalize_pilot(...) -> PilotVerdict`, ten fixed artifacts, and `annotation finalize-pilot`.

- [ ] **Step 1: Write failing evidence-denominator tests**

```python
def test_evidence_summary_counts_only_both_answerable_nonempty_pairs() -> None:
    summary = aggregate_evidence_agreement([both_answerable, one_unanswerable, corrupted_empty])
    assert summary.n_total_tasks == 3
    assert summary.n_comparable == 1
    assert summary.n_excluded_not_both_answerable == 1
    assert summary.n_invalid_empty_family == 1
```

- [ ] **Step 2: Run agreement test and verify RED**

Run: `uv run pytest tests/test_annotation_agreement.py -q`

Expected: aggregate type/function is absent.

- [ ] **Step 3: Implement aggregate evidence agreement**

```python
@dataclass(frozen=True)
class EvidenceAgreementSummary:
    n_total_tasks: int
    n_comparable: int
    n_excluded_not_both_answerable: int
    n_invalid_empty_family: int
    exact_set_family_agreement_rate: float | None
    mean_jaccard: float | None
    mean_set_f1: float | None
```

Call the existing pairwise primitive only for valid comparable pairs. When `n_comparable == 0`, return `None` for all three averages.

- [ ] **Step 4: Write failing finalization and verdict-precedence tests**

```python
@pytest.mark.parametrize(
    ("condition", "verdict"),
    [
        ("privacy", "BLOCKED_PRIVACY"),
        ("integrity", "BLOCKED_INTEGRITY"),
        ("incomplete", "BLOCKED_INCOMPLETE"),
        ("unresolved", "BLOCKED_UNRESOLVED_ADJUDICATION"),
        ("low_iaa", "BLOCKED_LOW_IAA"),
        ("ready", "READY_FOR_HUMAN_FREEZE_REVIEW"),
    ],
)
def test_finalization_verdict_precedence(condition: str, verdict: str) -> None:
    assert finalize_fixture(condition).verdict == verdict

def test_iaa_requires_exact_40_defined_finite_coefficients_at_point_70() -> None:
    assert gate(n=40, kappa=.70, alpha=.70).passed is True
    assert gate(n=40, kappa=None, alpha=1.0).passed is False
    assert gate(n=40, kappa=float("nan"), alpha=.9).passed is False
    assert gate(n=39, kappa=.9, alpha=.9).passed is False
```

- [ ] **Step 5: Run finalization tests and verify RED**

Run: `uv run pytest tests/test_annotation_finalize.py tests/test_annotation_agreement.py -q`

Expected: `finalize.py`, report models, and CLI do not exist.

- [ ] **Step 6: Implement finalization models, deterministic reports, and CLI**

```python
VERDICT_PRECEDENCE = (
    "BLOCKED_PRIVACY",
    "BLOCKED_INTEGRITY",
    "BLOCKED_INCOMPLETE",
    "BLOCKED_UNRESOLVED_ADJUDICATION",
    "BLOCKED_LOW_IAA",
    "READY_FOR_HUMAN_FREEZE_REVIEW",
)

@annotation_app.command("finalize-pilot")
def annotation_finalize_pilot(
    manifest: Path = typer.Option(..., "--manifest"),
    originals: Path = typer.Option(..., "--originals"),
    amendments: Path = typer.Option(..., "--amendments"),
    adjudications: Path = typer.Option(..., "--adjudications"),
    protocol: Path = typer.Option(..., "--protocol"),
    out: Path = typer.Option(..., "--out"),
) -> None:
    finalize_pilot(manifest, originals, amendments, adjudications, protocol, out)
```

Use manifest assignment order for left/right pairs. Derive `generated_at` from the maximum submitted/amended/adjudicated UTC source time. Always write `flow-accounting.json`, `disagreements.jsonl`, `eligibility.json`, `iaa.json`, `evidence-agreement.json`, `timing-summary.json`, `privacy-scan.json`, `input-manifest.json`, `pilot-verdict.json`, and `pilot-report.md` when inputs are safely parseable. Reports distinguish descriptive prevalence/raw agreement from the two mandatory coefficients.

- [ ] **Step 7: Run finalization, agreement, workflow, and CLI tests**

Run: `uv run pytest tests/test_annotation_finalize.py tests/test_annotation_agreement.py tests/test_annotation_workflow.py tests/test_cli_annotation.py -q`

Expected: PASS for exact threshold, undefined/single-class, NaN/infinite, sub-threshold, incomplete, unresolved, privacy, and deterministic second-run cases.

- [ ] **Step 8: Commit Task 3**

```powershell
git add src/rag_evidence/annotation/agreement.py src/rag_evidence/annotation/finalize.py src/rag_evidence/cli.py tests/test_annotation_agreement.py tests/test_annotation_finalize.py tests/test_cli_annotation.py
git commit -m "feat: finalize pilot with frozen IAA gates"
```

### Task 4: Protocol identity, exact runbooks, and canonical package replacement

**Files:**
- Modify: `src/rag_evidence/annotation/package.py`
- Modify: `PILOT_PROTOCOL.md`
- Modify: `pilot/v0.2/README.md`
- Modify: `pilot/v0.2/ONBOARDING.md`
- Modify: `pilot/v0.2/COMPLETION_CHECKLIST.md`
- Modify: `README.md`
- Modify: `README_en.md`
- Modify: `tests/test_pilot_package.py`
- Modify: `tests/test_publication_contract.py`
- Modify: `tests/test_repository_hygiene.py`

**Interfaces:**
- Consumes: all CLI signatures finalized in Tasks 1-3 and the existing deterministic package builder.
- Produces: instruction identity `pilot-v0.2.1-draft`, exact coordinator/annotator commands, explicit human-only boundaries, and regenerated package sources.

- [ ] **Step 1: Write failing protocol/publication contract tests**

```python
def test_protocol_and_canonical_packages_share_full_file_hash() -> None:
    protocol_hash = sha256(Path("PILOT_PROTOCOL.md").read_bytes()).hexdigest()
    for package in canonical_packages():
        assert package.instruction_version == "pilot-v0.2.1-draft"
        assert package.instruction_hash == protocol_hash

def test_docs_mark_baseline_packages_obsolete_and_forbid_pilot_start() -> None:
    text = public_docs_text()
    assert "372096b" in text
    assert "obsolete" in text.casefold()
    assert "does not authorize" in text.casefold()
```

- [ ] **Step 2: Run protocol/package tests and verify RED**

Run: `uv run pytest tests/test_pilot_package.py tests/test_publication_contract.py tests/test_repository_hygiene.py -q`

Expected: old `pilot-v0.2-draft` identity and missing B0.1 commands fail.

- [ ] **Step 3: Update protocol and bilingual operational documentation**

```text
Required command sequence:
1. rag-evidence annotation collect --manifest ... --submission A --submission B --amendment A --amendment B --out ...
2. rag-evidence annotation adjudicate --manifest ... --effective ... --state ... --host 127.0.0.1 --port 8002
3. rag-evidence annotation finalize-pilot --manifest ... --originals ... --amendments ... --adjudications ... --protocol PILOT_PROTOCOL.md --out ...
4. Stop at READY_FOR_HUMAN_FREEZE_REVIEW and request owner review; do not open confirmatory sampling.
```

Document that packages produced from baseline `372096b` are obsolete, that canonical files are safely replaced in the same paths, that state/decisions remain private and Git-ignored, and that B0.1 is annotation-ready engineering rather than a human pilot or v1.0 release.

- [ ] **Step 4: Regenerate canonical decision-free packages**

Run: `uv run rag-evidence annotation package-pilot --config configs/smoke.yaml --out pilot/v0.2/packages`

Expected: exactly `manifest.json`, `ann-pilot-a.json`, and `ann-pilot-b.json`; 40 tasks, two packages, zero decision fields; package instruction hashes equal the full protocol file hash.

- [ ] **Step 5: Run package/privacy/publication tests**

Run: `uv run pytest tests/test_pilot_package.py tests/test_clean_export.py tests/test_publication_contract.py tests/test_repository_hygiene.py -q`

Expected: PASS and `scan_clean_package()` reports `decisions: 0`.

- [ ] **Step 6: Commit Task 4**

```powershell
git add src/rag_evidence/annotation/package.py PILOT_PROTOCOL.md pilot/v0.2 README.md README_en.md tests/test_pilot_package.py tests/test_publication_contract.py tests/test_repository_hygiene.py
git commit -m "docs: operationalize the v0.2 human pilot protocol"
```

### Task 5: Source-controlled external handoff builder

**Files:**
- Create: `src/rag_evidence/annotation/handoff.py`
- Modify: `src/rag_evidence/cli.py`
- Create: `pilot/v0.2/handoff-manifest.schema.json`
- Create: `pilot/v0.2/handoff-manifest.json`
- Create: `pilot/v0.2/HANDOFF_RUNBOOK.md`
- Create: `pilot/v0.2/launchers/start-a.ps1`
- Create: `pilot/v0.2/launchers/start-a.sh`
- Create: `pilot/v0.2/launchers/start-b.ps1`
- Create: `pilot/v0.2/launchers/start-b.sh`
- Create: `tests/test_annotation_handoff.py`
- Modify: `tests/test_cli_annotation.py`
- Modify: `tests/test_repository_hygiene.py`

**Interfaces:**
- Consumes: final wheel path, exact clean source commit, committed manifest, canonical A/B package sources, runbook/onboarding, launchers, `uv.lock`, clean-install verification result, UTC build time.
- Produces: `build_handoff(spec_path, wheel_path, external_root, source_commit, build_time, clean_install) -> HandoffReceipt`, two disjoint kit directories, external `handoff-receipt.json`, coordinator `SHA256SUMS`, kit-local checksums, and `annotation build-handoff`.

- [ ] **Step 1: Write failing schema/boundary tests**

```python
def test_committed_handoff_manifest_has_no_wheel_hash() -> None:
    payload = read_json(Path("pilot/v0.2/handoff-manifest.json"))
    assert "wheel" not in json.dumps(payload).casefold()
    assert read_json(Path("pilot/v0.2/handoff-manifest.schema.json")) == HandoffSpec.model_json_schema()

def test_handoff_kits_are_disjoint_and_manifest_private(tmp_path: Path) -> None:
    receipt = build_handoff(spec, wheel, tmp_path / "external", sha, build_time, verified)
    assert kit_files("A") == {wheel.name, "ann-pilot-a.json", "ONBOARDING.md", "HANDOFF_RUNBOOK.md", "start.ps1", "start.sh", "SHA256SUMS"}
    assert kit_files("B") == {wheel.name, "ann-pilot-b.json", "ONBOARDING.md", "HANDOFF_RUNBOOK.md", "start.ps1", "start.sh", "SHA256SUMS"}
    assert not any("manifest" in name.casefold() for name in kit_files("A") | kit_files("B"))
```

- [ ] **Step 2: Run handoff tests and verify RED**

Run: `uv run pytest tests/test_annotation_handoff.py tests/test_repository_hygiene.py -q`

Expected: schema, builder, launchers, and runbook are absent.

- [ ] **Step 3: Implement strict committed spec and external builder**

```python
class HandoffSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["handoff-manifest-v1"]
    builder_version: Literal["handoff-builder-v1"]
    supported_python: Literal["3.11"]
    source_paths: HandoffSourcePaths
    canonical_sha256: CanonicalHashes
    instruction_version: Literal["pilot-v0.2.1-draft"]
    instruction_sha256: Hash64
    expected_layout: ExpectedLayout
    reproducible_build: ReproducibleBuildRecipe
```

Verify every committed source hash before copying. Refuse an existing nonempty external root. Hash the builder source itself. Use only relative names in checksums/receipt. The receipt records source commit, builder/wheel/package/runbook/launcher/lock hashes, instruction identity, UTC time, clean-install result, and `wheel_reproducibility` supplied by the final two-build verifier.

- [ ] **Step 4: Add repository-independent launchers and exact runbook**

```powershell
$KitRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
python -m rag_evidence annotation serve --package (Join-Path $KitRoot "ann-pilot-a.json") --state $StateRoot --host 127.0.0.1 --port 8001
```

```sh
KIT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python -m rag_evidence annotation serve --package "$KIT_ROOT/ann-pilot-a.json" --state "$STATE_ROOT" --host 127.0.0.1 --port 8001
```

Provide A/B variants with only the package name changed. The runbook gives Python 3.11 venv creation, `SHA256SUMS` verification, `pip install "./<wheel>[app]"`, launcher commands, localhost-only checks, JSONL export URLs, checksum creation, and safe return. It forbids web search, repository access, PII, hidden metadata, and continuation after any mismatch.

- [ ] **Step 5: Add build-handoff CLI and run focused tests**

Run: `uv run pytest tests/test_annotation_handoff.py tests/test_cli_annotation.py tests/test_repository_hygiene.py -q`

Expected: PASS; no generated wheel, archive, kit, receipt, or top-level `SHA256SUMS` appears in `git status`.

- [ ] **Step 6: Commit Task 5**

```powershell
git add src/rag_evidence/annotation/handoff.py src/rag_evidence/cli.py pilot/v0.2/handoff-manifest.schema.json pilot/v0.2/handoff-manifest.json pilot/v0.2/HANDOFF_RUNBOOK.md pilot/v0.2/launchers tests/test_annotation_handoff.py tests/test_cli_annotation.py tests/test_repository_hygiene.py
git commit -m "feat: build isolated annotation handoff kits"
```

### Task 6: Synthetic 40-task operational rehearsal

**Files:**
- Create: `src/rag_evidence/annotation/rehearsal.py`
- Modify: `src/rag_evidence/cli.py`
- Create: `tests/test_annotation_rehearsal.py`
- Modify: `tests/test_cli_annotation.py`
- Modify: `pilot/v0.2/COMPLETION_CHECKLIST.md`

**Interfaces:**
- Consumes: coordinator, adjudication store/app, finalizer, canonical formal-package hashes.
- Produces: `run_synthetic_rehearsal(out: Path, repository_root: Path) -> RehearsalResult` and `annotation rehearse-synthetic` with invented inputs and byte-identical repeat verification.

- [ ] **Step 1: Write failing rehearsal boundary and coverage tests**

```python
def test_rehearsal_refuses_formal_or_results_output_paths(tmp_path: Path) -> None:
    for forbidden in (Path("pilot/v0.2"), Path("results/v2")):
        with pytest.raises(DataError, match="formal"):
            run_synthetic_rehearsal(repository_root / forbidden, repository_root)

def test_rehearsal_covers_40_tasks_amendment_adjudication_and_repeat(tmp_path: Path) -> None:
    before = canonical_package_hashes(repository_root)
    result = run_synthetic_rehearsal(tmp_path / "rehearsal", repository_root)
    assert result.tasks == 40
    assert result.amendments >= 1
    assert result.disagreements >= 1
    assert result.dataset_defect_exclusions >= 1
    assert result.repeat_byte_identical is True
    assert canonical_package_hashes(repository_root) == before
```

- [ ] **Step 2: Run rehearsal tests and verify RED**

Run: `uv run pytest tests/test_annotation_rehearsal.py -q`

Expected: rehearsal module and command are absent.

- [ ] **Step 3: Implement invented fixture and full local path**

```python
def run_synthetic_rehearsal(out: Path, repository_root: Path) -> RehearsalResult:
    assert_safe_rehearsal_path(out, repository_root)
    before = canonical_package_hashes(repository_root)
    # Generate 40 deterministic invented question/passage tasks unrelated to
    # Dataset v2, two original streams, one valid amendment chain, multiple
    # disagreement types, third-human adjudications, and one defect exclusion.
    # Run collect -> adjudicate store/export/status -> finalize twice.
    assert canonical_package_hashes(repository_root) == before
    return compare_rehearsal_outputs(out / "run-1", out / "run-2")
```

Do not copy any formal task question, passage, ID, or decision. Mark every output root `SYNTHETIC-NOT-HUMAN-DATA.txt`; outputs exist only in the user-supplied external/temp path.

- [ ] **Step 4: Add negative cases and CLI contract**

Cover duplicate annotators, missing second stream, broken/forked/cyclic amendments, original/adjudicator mismatch, reused original annotator, undefined and sub-threshold IAA, invalid evidence denominator, private keys/paths, and formal-directory writes. Update the completion checklist with the exact temporary-directory command and required clean `git status` check.

- [ ] **Step 5: Run focused rehearsal suite twice**

Run: `uv run pytest tests/test_annotation_rehearsal.py tests/test_annotation_coordinator.py tests/test_annotation_finalize.py tests/test_adjudication_api.py -q`

Expected: PASS and both generated report trees are byte-identical.

- [ ] **Step 6: Commit Task 6**

```powershell
git add src/rag_evidence/annotation/rehearsal.py src/rag_evidence/cli.py tests/test_annotation_rehearsal.py tests/test_cli_annotation.py pilot/v0.2/COMPLETION_CHECKLIST.md
git commit -m "test: add synthetic pilot operational rehearsal"
```

### Task 7: Full verification, reproducible external build, and repository hygiene

**Files:**
- Modify only if a verification failure reveals an in-scope defect; use a new fix commit and rerun the failed check plus the full suite.
- Generate outside Git: two build roots, two wheel directories, two handoff roots, `handoff-receipt.json`, and `SHA256SUMS`.

**Interfaces:**
- Consumes: exact final clean commit and all prior task contracts.
- Produces: evidence that canonical packages and ideally wheels are byte-identical across two clean paths; otherwise an honest `per-build-hash-verified` external receipt.

- [ ] **Step 1: Run formatting, type, full CPU/offline, and artifact checks**

```powershell
uv run ruff check .
uv run mypy src
uv run pytest -m "not slow and not gpu" -q
git diff --check
git status --short
```

Expected: all checks pass; status contains only intended source/doc changes before their commit and no prohibited generated artifacts.

- [ ] **Step 2: Run a real external synthetic rehearsal**

Run: `uv run rag-evidence annotation rehearse-synthetic --out "$env:TEMP\reab-b01-synthetic"`

Expected: 40 tasks, at least one amendment/disagreement/adjudication/defect exclusion, byte-identical repeated outputs, and unchanged canonical formal-package hashes.

- [ ] **Step 3: Build from two clean exact-commit worktrees with fixed epoch**

```powershell
$env:SOURCE_DATE_EPOCH = "1787443200"
git worktree add --detach "$env:TEMP\reab-b01-build-a" HEAD
git worktree add --detach "$env:TEMP\reab-b01-build-b" HEAD
uv build --wheel --out-dir "$env:TEMP\reab-b01-dist-a" --directory "$env:TEMP\reab-b01-build-a"
uv build --wheel --out-dir "$env:TEMP\reab-b01-dist-b" --directory "$env:TEMP\reab-b01-build-b"
```

Expected: canonical A/B package bytes match between clean worktrees. Compare wheel bytes and SHA-256. If different, inspect ZIP entry timestamps, `.dist-info` metadata, and resolved build dependencies; fix reasonable nondeterminism and repeat. If still different, set only the external receipt status to `per-build-hash-verified` and prohibit deterministic-wheel wording.

- [ ] **Step 4: Clean-install both wheels and build external A/B kits**

```powershell
py -3.11 -m venv "$env:TEMP\reab-b01-venv-a"
py -3.11 -m venv "$env:TEMP\reab-b01-venv-b"
$WheelA = (Get-ChildItem -LiteralPath "$env:TEMP\reab-b01-dist-a" -Filter *.whl -File -ErrorAction Stop).FullName
$WheelB = (Get-ChildItem -LiteralPath "$env:TEMP\reab-b01-dist-b" -Filter *.whl -File -ErrorAction Stop).FullName
& "$env:TEMP\reab-b01-venv-a\Scripts\python.exe" -m pip install "${WheelA}[app]"
& "$env:TEMP\reab-b01-venv-b\Scripts\python.exe" -m pip install "${WheelB}[app]"
```

Run CLI help/import/app creation in both environments, then invoke `annotation build-handoff` outside the repository. Verify kit A has no B package/launcher/manifest and kit B has no A package/launcher/manifest. Verify every kit and coordinator checksum and inspect external receipts for exact source SHA, builder/wheel/package/runbook/launcher/lock hashes, instruction identity, UTC build time, and clean-install status.

- [ ] **Step 5: Run Docker and formal-boundary checks**

```powershell
docker build -t rag-evidence-b01 .
docker run --rm rag-evidence-b01 rag-evidence --help
Test-Path results/v2/pilot/confirmatory
git ls-files | rg "(?i)(\.whl$|\.tar\.gz$|handoff-receipt\.json$|SHA256SUMS$|submissions|amendments|adjudications|eligibility|iaa|synthetic)"
git status --short --branch
```

Expected: Docker precomputed path works; confirmatory output path is absent; the tracked-file scan contains only explicitly allowed source/schema/docs/tests and no generated artifacts or decisions; worktree is clean after the last independent fix commit.

- [ ] **Step 6: Push the existing branch and create or update an open PR**

```powershell
git push -u origin codex/v0.2-annotation-readiness
gh pr list --head codex/v0.2-annotation-readiness --base main --state open
gh pr create --base main --head codex/v0.2-annotation-readiness --title "Make v0.2 pilot annotation-ready" --body "B0.1 annotation-ready engineering only. No human pilot, confirmatory sample, model/API run, merge, tag, release, or Hugging Face publication. See commits and checks for tested coordinator, adjudication, fixed IAA gate, external handoff builder, and synthetic rehearsal."
```

If an open PR already exists, update its title/body instead of creating another. The PR must state B0.1 boundaries, tests, external build result, and that no human pilot, merge, tag, release, confirmatory sampling, API/model run, or HF publication occurred.

- [ ] **Step 7: Wait for CI and stop**

Run: `gh pr checks --watch --fail-fast`

Expected: all required checks green. Report the branch, final commit, PR URL, local verification, wheel reproducibility classification, and Git-external artifact locations. Do not merge.
