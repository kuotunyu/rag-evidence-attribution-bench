# Wave D: Git, Blob, and Wheel Source-Bound Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans and
> superpowers:test-driven-development task-by-task. Use superpowers:systematic-debugging for any
> reproducibility failure.

**Goal:** Make handoff construction fail closed unless two fresh exact-commit checkouts and four
wheel instances prove one byte-identical, Git-bound distribution.

**Architecture:** `source_verify.py` owns Git identity and clean gates. `wheel_verify.py` owns
reproducible replay, ZIP/RECORD/METADATA validation, and Git-blob payload comparison. `handoff.py`
owns the committed source manifest, receipt binding, and mutually exclusive kit assembly.

**Tech Stack:** Python subprocess, Git plumbing, zipfile/email metadata, SHA-256/base64, Pydantic 2,
pytest.

## Global constraints

All constraints in `2026-08-24-breaking-v2-plan-index.md` apply.

---

### Task D1: Verify exact Git identity and pre/post clean gates

**Files:**
- Create: `src/rag_evidence/annotation/source_verify.py`
- Create: `tests/test_annotation_source_verify.py`

**Interfaces:**
- `GitCommandRecordV2` records logical checkout, phase, argv, exit code, stdout byte count/hash,
  and stderr hash.
- `VerifiedCheckoutV2` records root, commit SHA, tree SHA, and immutable command records.
- `verify_checkout(root: Path, requested_commit: str, *, phase: Literal["pre-build",
  "post-build"]) -> VerifiedCheckoutV2` executes Git without a shell.
- `verify_tracked_blob(checkout, relative_path, expected_sha256) -> str` compares working bytes,
  `git cat-file` bytes, and committed manifest hash.

- [ ] **Step 1: Write failing Git adversarial tests**

```python
@pytest.mark.parametrize("poison", [
    "tracked-change", "staged-change", "untracked-file", "ignored-source-overlay",
    "ignored-build-config", "ignored-venv", "ignored-cache",
])
def test_checkout_poison_fails_closed(git_checkout: Path, poison: str) -> None:
    apply_poison(git_checkout, poison)
    with pytest.raises(DataError, match="clean"):
        verify_checkout(git_checkout, REAL_COMMIT, phase="pre-build")

def test_post_build_ignored_file_is_rejected(git_checkout: Path) -> None:
    verify_checkout(git_checkout, REAL_COMMIT, phase="pre-build")
    (git_checkout / "build/ignored.txt").parent.mkdir()
    (git_checkout / "build/ignored.txt").write_text("poison")
    with pytest.raises(DataError):
        verify_checkout(git_checkout, REAL_COMMIT, phase="post-build")
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/test_annotation_source_verify.py -q`

Expected: FAIL because exact Git/ignored verification interfaces do not exist.

- [ ] **Step 3: Implement shell-free Git verification**

```python
TRACKED_UNTRACKED_ARGV = (
    "git", "status", "--porcelain=v1", "--untracked-files=all", "-z",
)
IGNORED_ARGV = (
    "git", "ls-files", "--others", "--ignored", "--exclude-standard", "-z",
)

def _require_zero_output(record: GitCommandRecordV2) -> None:
    if record.exit_code != 0 or record.stdout_byte_count != 0:
        raise DataError("checkout clean gate failed")
```

Resolve the requested commit to a real 40-hex commit, require detached `HEAD` equality and matching
tree SHA, and never delete or repair a failed checkout.

- [ ] **Step 4: Run GREEN**

Run: `uv run pytest tests/test_annotation_source_verify.py -q`

- [ ] **Step 5: Regression verification**

Run: `uv run pytest tests/test_repository_hygiene.py -q`

- [ ] **Step 6: Commit boundary**

Commit: `feat: verify exact clean source checkouts`

### Task D2: Build and validate four byte-identical wheels

**Files:**
- Create: `src/rag_evidence/annotation/wheel_verify.py`
- Create: `tests/test_annotation_wheel_verify.py`
- Modify: `pyproject.toml`
- Modify: `src/rag_evidence/cli.py`
- Modify: `tests/test_cli_annotation.py`

**Interfaces:**
- `BuildEnvironmentV2` contains fixed `SOURCE_DATE_EPOCH`, external cache/build/output roots,
  `PYTHONDONTWRITEBYTECODE=1`, and an external `PYTHONPYCACHEPREFIX`.
- `replay_wheel_build(checkout, environment) -> WheelBuildRecordV2` brackets the exact committed
  argv with D1 pre/post verification.
- `verify_wheel_set(checkouts, supplied_wheels, replay_roots, spec) -> VerifiedWheelSetV2` requires
  supplied A, supplied B, replay A, and replay B bytes, names, sizes, and hashes to match.
- `verify_wheel_payload(wheel, checkout, spec) -> WheelIdentityV2` validates ZIP paths/duplicates,
  every RECORD hash/size, normalized name, `0.2.0.dev0` METADATA, and every packaged
  `rag_evidence` byte against a tracked `src/rag_evidence` blob.
- `annotation build-wheels --checkout-a PATH --checkout-b PATH --source-commit SHA --output PATH`
  writes two supplied and two replay wheels plus `wheel-verification.json` outside Git.

- [ ] **Step 1: Write failing wheel-integrity tests**

```python
@pytest.mark.parametrize("mutation", [
    "record-hash", "record-size", "metadata-version", "duplicate-entry", "unsafe-path",
    "missing-source", "unexpected-package-source", "source-byte",
])
def test_mutated_wheel_fails_source_binding(valid_wheel: Path, mutation: str) -> None:
    poisoned = mutate_wheel(valid_wheel, mutation)
    with pytest.raises(DataError):
        verify_wheel_payload(poisoned, checkout, handoff_spec)

def test_four_wheels_must_be_byte_identical(wheel_set_with_one_byte_changed: WheelFixture) -> None:
    with pytest.raises(DataError, match="byte-identical"):
        verify_wheel_set(**wheel_set_with_one_byte_changed.arguments)
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/test_annotation_wheel_verify.py -q`

Expected: FAIL because replay and source-bound wheel verification do not exist.

- [ ] **Step 3: Implement reproducible replay and wheel validation**

```python
BUILD_ARGV = ("uv", "build", "--wheel", "--out-dir", "{external_dist}")

def require_byte_identity(wheels: Sequence[Path]) -> WheelIdentityV2:
    identities = tuple(file_identity(path) for path in wheels)
    if len({(item.filename, item.byte_size, item.sha256) for item in identities}) != 1:
        raise DataError("all supplied and replay wheels must be byte-identical")
    return identities[0]
```

Pin Hatchling in `[build-system]` to the resolved version used by the candidate. Build paths,
virtual environments, uv/pip caches, and wheel outputs are external and distinct.

- [ ] **Step 4: Run GREEN**

Run: `uv run pytest tests/test_annotation_wheel_verify.py -q`

- [ ] **Step 5: Real two-directory reproducibility probe**

Run the source-controlled builder against two disposable detached test checkouts in `$env:TEMP` and
compare all four produced wheel hashes. If bytes differ, invoke systematic debugging and classify
the observed ZIP timestamp, metadata, line-ending, or build-dependency difference before changing
one variable.

- [ ] **Step 6: Commit boundary**

Commit: `feat: bind reproducible wheels to Git blobs`

### Task D3: Replace handoff manifest, receipt, and kit builder with v2

**Files:**
- Modify: `src/rag_evidence/annotation/handoff.py`
- Create: `src/rag_evidence/annotation/platform.py`
- Modify: `src/rag_evidence/cli.py`
- Modify: `pilot/v0.2/handoff-manifest.json`
- Modify: `pilot/v0.2/handoff-manifest.schema.json`
- Modify: `tests/test_annotation_handoff.py`
- Modify: `tests/test_cli_annotation.py`
- Modify: `tests/test_repository_hygiene.py`

**Interfaces:**
- `HandoffSpecV2` uses `handoff-manifest-v2`, `handoff-spec-v2`, `handoff-builder-v2`,
  `pilot-v0.2.2-draft`, `0.2.0.dev0`, `required_verification_platforms=("Windows", "Linux")`,
  canonical package/source hashes, expected layout, and exact build recipe. It contains no wheel
  hash or private coordinator-manifest instance/hash.
- `PlatformVerificationReceiptV2` model binds source/tree/distribution/wheel/lock/protocol/schema/
  spec, Python/pip/OS/architecture, command records, installed distributions, and smoke hashes.
- `build_handoff_v2(...) -> HandoffReceiptV2` consumes two verified checkouts, the four-wheel result,
  exactly one Windows and one Linux receipt, and an external coordinator manifest, then creates
  disjoint A/B kits and root receipt/checksums outside Git.
- `annotation write-handoff-spec --repository-root PATH --output
  pilot/v0.2/handoff-manifest.json` deterministically refreshes source hashes and the generated JSON
  schema before the final source commit.
- `annotation build-handoff` exposes the exact `build_handoff_v2` inputs used in Wave F.

- [ ] **Step 1: Write failing v2 spec/receipt/layout tests**

```python
def test_committed_handoff_manifest_is_source_only() -> None:
    payload = read_json(HANDOFF_MANIFEST)
    assert payload["schema_version"] == "handoff-manifest-v2"
    assert payload["python_distribution"] == "0.2.0.dev0"
    assert "wheel_sha256" not in json.dumps(payload)
    assert "assignment-manifest" not in json.dumps(payload)

def test_final_kits_are_mutually_exclusive(final_handoff: Path) -> None:
    assert kit_files(final_handoff / "kit-a") == expected_a_files()
    assert kit_files(final_handoff / "kit-b") == expected_b_files()
    assert "ann-pilot-b.json" not in kit_files(final_handoff / "kit-a")
    assert "ann-pilot-a.json" not in kit_files(final_handoff / "kit-b")
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/test_annotation_handoff.py tests/test_cli_annotation.py tests/test_repository_hygiene.py -q`

Expected: FAIL because current handoff permits v1, one wheel, fallback reproducibility, a committed
private manifest, and a generic clean-install boolean.

- [ ] **Step 3: Implement strict v2 assembly and receipt binding**

```python
def build_handoff_v2(inputs: HandoffInputsV2) -> HandoffReceiptV2:
    spec = load_tracked_spec_v2(inputs.spec_path)
    wheel = verify_wheel_set(inputs.checkouts, inputs.supplied_wheels, inputs.replay_roots, spec)
    receipts = validate_platform_pair(inputs.platform_receipts, wheel, spec)
    validate_private_manifest(inputs.coordinator_manifest, spec.canonical_package_sha256)
    copy_disjoint_kits(inputs.external_root, wheel.path, spec)
    return write_bound_receipt_and_checksums(inputs, wheel, receipts)
```

Receipt/transcript includes pre/post clean command argv/results. Output generation is atomic into a
new external root; any mismatch leaves no claim of a successful handoff.

- [ ] **Step 4: Run GREEN**

Run: `uv run pytest tests/test_annotation_handoff.py tests/test_cli_annotation.py tests/test_repository_hygiene.py -q`

- [ ] **Step 5: Full Wave D regression**

Run: `uv run pytest tests/test_annotation_source_verify.py tests/test_annotation_wheel_verify.py tests/test_annotation_handoff.py -q`

- [ ] **Step 6: Commit boundary**

Commit: `feat: build source-bound v2 handoff kits`
