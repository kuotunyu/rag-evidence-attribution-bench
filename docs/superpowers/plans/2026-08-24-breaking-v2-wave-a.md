# Wave A: Breaking v2 Schemas and Privacy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans and
> superpowers:test-driven-development task-by-task.

**Goal:** Establish group-free v2 human artifacts and a deterministic coordinator-only assignment
mapping before any decision workflow is upgraded.

**Architecture:** Strict frozen Pydantic v2 models define the public boundary. Visible task
projection and scheduling consume an internal `(task, group, transformation)` record but serialize
group data only into an external coordinator manifest. A delivery scanner combines field allowlists
with hidden-key/value denial.

**Tech Stack:** Python 3.11, Pydantic 2, canonical JSON/SHA-256, pytest.

## Global constraints

All constraints in `2026-08-24-breaking-v2-plan-index.md` apply.

---

### Task A1: Replace human-visible records with strict v2 models

**Files:**
- Modify: `src/rag_evidence/annotation/models.py`
- Modify: `src/rag_evidence/annotation/__init__.py`
- Modify: `tests/test_annotation_schema.py`
- Create: `tests/test_annotation_v1_rejection.py`

**Interfaces:**
- Produces `BlindTaskV2`, `AnswerabilityAnnotationV2`, `CitationAnnotationV2`,
  `AnnotationAmendmentV2`, `AdjudicationV2`, `EligibilityRecordV2`, and
  `EligibilityArtifactV2`.
- Produces `BindingTuple = tuple[TaskId, ChallengeId, Hash64, str, Hash64, str, str]` through
  `annotation_binding(record) -> BindingTuple`.
- Retains `artifact_hash(record: BaseModel) -> str` over canonical UTF-8 JSON.
- Removes `BlindGroupId` and `blinded_parent_group` from every human model.

- [ ] **Step 1: Write failing strict-v2 and v1-rejection tests**

```python
def test_blind_task_v2_has_exact_group_free_fields(valid_task_payload: dict[str, object]) -> None:
    task = BlindTaskV2.model_validate(valid_task_payload)
    assert tuple(task.model_fields) == (
        "schema_version", "annotation_task_id", "challenge_id", "instruction_version",
        "instruction_hash", "question", "passages", "task_content_hash", "assignment_batch",
    )

@pytest.mark.parametrize("schema", V1_ANNOTATION_SCHEMA_LITERALS)
def test_every_v1_annotation_literal_fails_closed(schema: str) -> None:
    with pytest.raises(ValidationError):
        parse_annotation_v2({"schema_version": schema})
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/test_annotation_schema.py tests/test_annotation_v1_rejection.py -q`

Expected: FAIL because v2 classes/parsers do not exist and current models accept v1 group fields.

- [ ] **Step 3: Implement the minimal strict models**

```python
class BlindTaskV2(_StrictModel):
    schema_version: Literal["blind-task-v2"] = "blind-task-v2"
    annotation_task_id: TaskId
    challenge_id: ChallengeId
    instruction_version: str
    instruction_hash: Hash64
    question: Annotated[str, Field(min_length=1)]
    passages: tuple[TaskPassage, ...] = Field(min_length=1)
    task_content_hash: Hash64
    assignment_batch: str

def annotation_binding(record: AnswerabilityAnnotationV2) -> BindingTuple:
    return (
        record.annotation_task_id, record.challenge_id, record.task_content_hash,
        record.instruction_version, record.instruction_hash, record.assignment_batch,
        record.annotator_pseudonym,
    )
```

Use discriminated schema literals only for v2. `task_content_hash` excludes every group-like value.
Adjudications bind two complete originals and require a third pseudonym.

- [ ] **Step 4: Run GREEN**

Run: `uv run pytest tests/test_annotation_schema.py tests/test_annotation_v1_rejection.py -q`

Expected: PASS for v2 validation, binding immutability, third-human adjudication, and all v1 rejects.

- [ ] **Step 5: Run regression verification**

Run: `uv run pytest tests/test_annotation_statistics.py tests/test_sentence_citations_v2.py -q`

- [ ] **Step 6: Commit boundary**

Commit: `feat: replace annotation records with strict v2 schemas`

### Task A2: Implement allowlist projection and delivery scanning

**Files:**
- Modify: `src/rag_evidence/annotation/blinding.py`
- Modify: `src/rag_evidence/annotation/privacy.py`
- Modify: `scripts/scan_annotation_export.py`
- Modify: `tests/test_annotation_blinding.py`
- Create: `tests/test_annotation_privacy_v2.py`
- Modify: `tests/test_clean_export.py`

**Interfaces:**
- `project_challenge_v2(record, *, instruction_version, instruction_hash, batch, namespace) -> BlindTaskV2`
- `scan_delivery_payload(payload: object, *, artifact_kind: str) -> tuple[str, ...]`
- `scan_delivery_tree(root: Path, *, allowed_files: Collection[str] | None = None) -> tuple[str, ...]`
- Scanner denylist includes parent/group/sibling keys, `bg-*`, transformations, labels, provenance,
  internal IDs, paths, PII, seed, scores, and all v1 schema literals; model field allowlists are
  artifact-specific.

- [ ] **Step 1: Write failing leak and metadata-pair tests**

```python
@pytest.mark.parametrize("poison", [
    {"blinded_parent_group": "bg-0123456789abcdef01234567"},
    {"coordinator_group_id": "coord-0001"},
    {"transformation": "missing_hop"},
    {"expected_answerability": "unanswerable"},
])
def test_delivery_scanner_rejects_hidden_metadata(poison: object) -> None:
    assert scan_delivery_payload(poison, artifact_kind="submission")

def test_task_metadata_has_no_pair_reconstructing_value(package_v2: AssignmentPackageV2) -> None:
    task_values = per_task_metadata_values(package_v2)
    assert all(count != 2 for count in Counter(task_values).values())
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/test_annotation_blinding.py tests/test_annotation_privacy_v2.py tests/test_clean_export.py -q`

Expected: FAIL because current projection emits `blind-task-v1` plus `blinded_parent_group` and the
scanner does not reject all group-bearing keys/values.

- [ ] **Step 3: Implement field-by-field projection and allowlist scanning**

```python
content = {
    "challenge_id": record.challenge_id,
    "instruction_version": instruction_version,
    "instruction_hash": instruction_hash,
    "question": record.example.question,
    "passages": visible_passages(record),
}
payload = {
    "schema_version": "blind-task-v2",
    "annotation_task_id": opaque_task_id(namespace, batch, record.challenge_id),
    **content,
    "task_content_hash": canonical_hash(content),
    "assignment_batch": batch,
}
```

Ordinary prose containing words such as “parent” remains valid; only metadata keys and reserved
value patterns fail.

- [ ] **Step 4: Run GREEN**

Run: `uv run pytest tests/test_annotation_blinding.py tests/test_annotation_privacy_v2.py tests/test_clean_export.py -q`

- [ ] **Step 5: Run mutation-focused regressions**

Run: `uv run pytest tests/test_annotation_api.py tests/test_adjudication_api.py -q`

- [ ] **Step 6: Commit boundary**

Commit: `feat: enforce group-free v2 delivery projection`

### Task A3: Split public packages from the private coordinator manifest

**Files:**
- Modify: `src/rag_evidence/annotation/assignment.py`
- Modify: `src/rag_evidence/annotation/package.py`
- Modify: `src/rag_evidence/cli.py`
- Modify: `PILOT_PROTOCOL.md`
- Modify: `pilot/v0.2/README.md`
- Delete: `pilot/v0.2/packages/manifest.json`
- Modify: `pilot/v0.2/packages/ann-pilot-a.json`
- Modify: `pilot/v0.2/packages/ann-pilot-b.json`
- Create: `pilot/v0.2/assignment-manifest-v2.schema.json`
- Modify: `tests/test_annotation_assignment.py`
- Modify: `tests/test_pilot_package.py`
- Create: `tests/test_coordinator_manifest_v2.py`
- Modify: `tests/test_cli_annotation.py`

**Interfaces:**
- `AssignmentPackageV2` contains only package binding and ordered `BlindTaskV2` values.
- `CoordinatorTaskV2` contains task ID, internal group ID, transformation, two annotators, and two
  delivery positions.
- `AssignmentManifestV2` contains schema/protocol/batch/seed, `tasks`, coordinator rows, and
  `package_sha256`.
- `build_pilot_assignments(records, instruction_hash) -> tuple[AssignmentPackageV2,
  AssignmentPackageV2, AssignmentManifestV2]`.
- `write_coordinator_manifest_v2(path, manifest)` refuses repository-relative output.
- `annotation build-coordinator-manifest --challenge-records PATH --package-a PATH --package-b
  PATH --output PATH` rebuilds the private mapping from external challenge records and requires its
  package hashes to match the canonical A/B bytes.

- [ ] **Step 1: Write failing package/manifest boundary tests**

```python
def test_pilot_manifest_has_exact_private_shape(manifest_v2: AssignmentManifestV2) -> None:
    assert len(manifest_v2.tasks) == 40
    assert len({row.internal_group_id for row in manifest_v2.coordinator_tasks}) == 20
    assert all(len(set(row.annotators)) == 2 for row in manifest_v2.coordinator_tasks)
    assert manifest_v2.package_sha256 == literal_package_hashes()

def test_committed_package_directory_has_no_manifest() -> None:
    assert {p.name for p in Path("pilot/v0.2/packages").iterdir()} == {
        "ann-pilot-a.json", "ann-pilot-b.json"
    }
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/test_annotation_assignment.py tests/test_pilot_package.py tests/test_coordinator_manifest_v2.py -q`

Expected: FAIL because scheduling reads a group field from visible tasks and a private v1 manifest is
currently committed.

- [ ] **Step 3: Implement internal scheduling and exact pilot validation**

```python
@dataclass(frozen=True)
class SchedulableTask:
    task: BlindTaskV2
    internal_group_id: str
    transformation: Literal["missing_hop", "evidence_swap"]

def validate_pilot_manifest(manifest: AssignmentManifestV2) -> None:
    require_counts(manifest, tasks=40, groups=20, variants_per_group=2, annotators_per_task=2)
    require_non_adjacent_positions(manifest.coordinator_tasks)
    require_package_hash_binding(manifest)
```

Update the approved instruction document to `pilot-v0.2.2-draft`, then regenerate A/B package bytes
from the retained coordinator-only challenge records. Write the coordinator instance only to a
caller-provided external directory. Commit only its JSON schema and deterministic builder.

- [ ] **Step 4: Run GREEN**

Run: `uv run pytest tests/test_annotation_assignment.py tests/test_pilot_package.py tests/test_coordinator_manifest_v2.py -q`

- [ ] **Step 5: Run the static privacy scan**

Run: `uv run python scripts/scan_annotation_export.py pilot/v0.2/packages`

Expected: PASS with 2 packages, 40 tasks each, and zero decisions/hidden metadata.

- [ ] **Step 6: Commit boundary**

Commit: `feat: separate v2 packages from coordinator mapping`
