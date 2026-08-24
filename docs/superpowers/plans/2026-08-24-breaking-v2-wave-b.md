# Wave B: Submission, Adjudication, and IAA v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans and
> superpowers:test-driven-development task-by-task.

**Goal:** Upgrade every decision-bearing path to group-free v2 records while preserving immutable
submissions, order-independent amendments, third-human adjudication, and the fixed IAA gate.

**Architecture:** Local annotator state validates the seven-field v2 binding tuple. Coordinator
collection consumes the private manifest but emits only group-free streams and disagreement cases.
Finalization computes IAA over exactly 40 complete post-amendment/pre-adjudication pairs.

**Tech Stack:** Python 3.11, Pydantic 2, FastAPI/TestClient, canonical JSONL, NumPy, pytest.

## Global constraints

All constraints in `2026-08-24-breaking-v2-plan-index.md` apply.

---

### Task B1: Upgrade annotation state, API, and UI to v2 bindings

**Files:**
- Modify: `src/rag_evidence/annotation/store.py`
- Modify: `src/rag_evidence/annotation/app.py`
- Modify: `src/rag_evidence/annotation/ui.html`
- Modify: `tests/test_annotation_store.py`
- Modify: `tests/test_annotation_api.py`
- Modify: `tests/test_cli_annotation.py`

**Interfaces:**
- `AnnotationStore(root: Path, package: AssignmentPackageV2)` validates drafts, submissions, and
  amendments against task ID, challenge ID, content hash, instruction version/hash, batch, and
  annotator pseudonym.
- Draft schema is `annotation-draft-v2`; immutable decision schema is
  `answerability-annotation-v2`; amendments use `annotation-amendment-v2`.
- `create_annotation_app(package_path: Path, state_dir: Path) -> FastAPI` exposes group-free task,
  progress, draft, submission, amendment, and export payloads.

- [ ] **Step 1: Write failing API/state privacy tests**

```python
def test_submission_binding_never_requests_group(valid_submission_v2: dict[str, object]) -> None:
    response = client.post("/api/submissions", json=valid_submission_v2)
    assert response.status_code == 201
    assert "group" not in response.text.casefold()
    assert "bg-" not in response.text.casefold()

def test_v1_submission_is_rejected_before_state_append(v1_submission: dict[str, object]) -> None:
    before = state_tree_bytes()
    assert client.post("/api/submissions", json=v1_submission).status_code == 422
    assert state_tree_bytes() == before
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/test_annotation_store.py tests/test_annotation_api.py tests/test_cli_annotation.py -q`

Expected: FAIL because the store, JavaScript payload, and validators require
`blinded_parent_group` and v1 literals.

- [ ] **Step 3: Implement minimal v2 state and UI payloads**

```python
def _validate_task_binding(task: BlindTaskV2, record: AnswerabilityAnnotationV2) -> None:
    expected = (
        task.annotation_task_id, task.challenge_id, task.task_content_hash,
        task.instruction_version, task.instruction_hash, task.assignment_batch,
        package.annotator_pseudonym,
    )
    if annotation_binding(record) != expected:
        raise ArtifactError("submission binding does not match assigned v2 task")
```

Update UI literals to `annotation-ui-v2`, `annotation-draft-v2`, and
`answerability-annotation-v2`. Do not serialize a private manifest or hidden field into HTML state.

- [ ] **Step 4: Run GREEN**

Run: `uv run pytest tests/test_annotation_store.py tests/test_annotation_api.py tests/test_cli_annotation.py -q`

- [ ] **Step 5: Regression verification**

Run: `uv run pytest tests/test_annotation_blinding.py tests/test_annotation_privacy_v2.py -q`

- [ ] **Step 6: Commit boundary**

Commit: `feat: upgrade annotation console to group-free v2`

### Task B2: Upgrade collection and amendment resolution

**Files:**
- Modify: `src/rag_evidence/annotation/coordinator.py`
- Modify: `tests/test_annotation_coordinator.py`
- Modify: `tests/test_annotation_workflow.py`

**Interfaces:**
- `collect_annotation_streams(manifest_path, submission_paths, amendment_paths, out) -> CollectionResult`
  loads only `assignment-manifest-v2` and v2 decision streams.
- Receipt schemas are `annotation-collection-receipt-v2` and `annotation-input-manifest-v2`.
- `resolve_amendments(manifest, originals, amendments) -> tuple[AnswerabilityAnnotationV2, ...]`
  preserves order independence and rejects missing, forked, cyclic, cross-original, duplicate-ID,
  duplicate-hash, changed-binding, and v1 chains.

- [ ] **Step 1: Write failing v2 amendment/receipt tests**

```python
def test_amendment_resolution_is_order_independent(v2_chain: list[AnnotationAmendmentV2]) -> None:
    forward = resolve_amendments(manifest_v2, originals_v2, v2_chain)
    reverse = resolve_amendments(manifest_v2, originals_v2, list(reversed(v2_chain)))
    assert forward == reverse

def test_collection_receipts_are_v2_and_group_free(collected: CollectionResult) -> None:
    receipt = read_json(collected.output_dir / "collection-receipt.json")
    assert receipt["schema_version"] == "annotation-collection-receipt-v2"
    assert scan_delivery_payload(receipt, artifact_kind="collection_receipt") == ()
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/test_annotation_coordinator.py tests/test_annotation_workflow.py -q`

Expected: FAIL on v1 manifest/record parsing and v1 receipt literals.

- [ ] **Step 3: Implement v2-only loaders and binding checks**

```python
def _load_submissions(path: Path) -> tuple[AnswerabilityAnnotationV2, ...]:
    return tuple(AnswerabilityAnnotationV2.model_validate(row) for row in read_records(path))

def _same_binding(left: AnswerabilityAnnotationV2, right: AnswerabilityAnnotationV2) -> bool:
    return annotation_binding(left) == annotation_binding(right)
```

Preserve source file hashes and deterministic assignment order in output digests; never copy
coordinator group rows into returned artifacts.

- [ ] **Step 4: Run GREEN**

Run: `uv run pytest tests/test_annotation_coordinator.py tests/test_annotation_workflow.py -q`

- [ ] **Step 5: Regression verification**

Run: `uv run pytest tests/test_annotation_store.py tests/test_annotation_v1_rejection.py -q`

- [ ] **Step 6: Commit boundary**

Commit: `feat: collect and resolve v2 annotation streams`

### Task B3: Upgrade disagreement, adjudication, eligibility, and pilot finalization

**Files:**
- Modify: `src/rag_evidence/annotation/workflow.py`
- Modify: `src/rag_evidence/annotation/agreement.py`
- Modify: `src/rag_evidence/annotation/finalize.py`
- Modify: `src/rag_evidence/annotation/adjudicator_ui.html`
- Modify: `src/rag_evidence/annotation/app.py`
- Modify: `tests/test_adjudication_api.py`
- Modify: `tests/test_annotation_agreement.py`
- Modify: `tests/test_annotation_finalize.py`
- Modify: `tests/test_annotation_workflow.py`

**Interfaces:**
- `DisagreementCaseV2` and `AdjudicationV2` contain no group mapping.
- Their exact schema literals are `disagreement-case-v2` and `adjudication-v2`; eligibility uses
  `eligibility-record-v2` and `eligibility-artifact-v2`.
- `build_disagreement_queue(manifest, submissions) -> tuple[DisagreementCaseV2, ...]` requires two
  assigned originals per task.
- `create_adjudication_app(manifest_path, effective_path, state_dir) -> FastAPI` projects only the
  referenced `BlindTaskV2` plus the group-free disagreement case.
- Aggregate schemas are the exact `pilot-*-v2` and `eligibility-*-v2` names in the spec.
- The concrete aggregate literals are `pilot-iaa-v2`, `pilot-evidence-agreement-v2`,
  `pilot-privacy-scan-v2`, `pilot-verdict-v2`, `pilot-timing-summary-v2`,
  `pilot-flow-accounting-v2`, and `pilot-finalization-input-manifest-v2`.
- `evaluate_iaa_gate(agreement) -> IaaGateResult` requires 40 complete pre-adjudication pairs,
  finite Cohen kappa and nominal Krippendorff alpha, each at least `0.70`.

- [ ] **Step 1: Write failing adjudicator privacy and IAA tests**

```python
def test_adjudication_case_response_is_group_free() -> None:
    response = adjudication_client.get("/api/cases")
    assert response.status_code == 200
    assert scan_delivery_payload(response.json(), artifact_kind="adjudication_case") == ()

def test_iaa_uses_exactly_40_pre_adjudication_pairs() -> None:
    result = finalize_fixture(pair_count=40, kappa=.70, alpha=.70)
    assert result.verdict == "READY_FOR_HUMAN_FREEZE_REVIEW"
    assert finalize_fixture(pair_count=39, kappa=.99, alpha=.99).verdict == "BLOCKED_INCOMPLETE"
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/test_adjudication_api.py tests/test_annotation_agreement.py tests/test_annotation_finalize.py tests/test_annotation_workflow.py -q`

Expected: FAIL because v1 disagreement/adjudication fields and aggregate literals remain.

- [ ] **Step 3: Implement the group-free finalization path**

```python
class DisagreementCaseV2(_StrictModel):
    schema_version: Literal["disagreement-case-v2"] = "disagreement-case-v2"
    annotation_task_id: TaskId
    challenge_id: ChallengeId
    reasons: tuple[DisagreementReason, ...]
    left: AnswerabilityAnnotationV2
    right: AnswerabilityAnnotationV2

def evaluate_iaa_gate(agreement: NominalAgreement) -> IaaGateResult:
    passed = (
        agreement.n_complete == 40
        and finite(agreement.cohen_kappa)
        and finite(agreement.krippendorff_alpha)
        and agreement.cohen_kappa >= .70
        and agreement.krippendorff_alpha >= .70
    )
    return IaaGateResult(passed=passed, reasons=gate_reasons(agreement))
```

Adjudicated values remain excluded from IAA. Final reports preserve the feasibility-only and honest
blinding claim text from the approved spec.

- [ ] **Step 4: Run GREEN**

Run: `uv run pytest tests/test_adjudication_api.py tests/test_annotation_agreement.py tests/test_annotation_finalize.py tests/test_annotation_workflow.py -q`

- [ ] **Step 5: Full Wave B regression**

Run: `uv run pytest tests/test_annotation_*.py tests/test_adjudication_api.py -q`

- [ ] **Step 6: Commit boundary**

Commit: `feat: complete v2 adjudication and IAA pipeline`
