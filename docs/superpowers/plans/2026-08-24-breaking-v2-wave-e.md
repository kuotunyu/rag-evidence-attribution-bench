# Wave E: Windows/Linux Platform Verification and CI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans and
> superpowers:test-driven-development task-by-task. Use superpowers:systematic-debugging for
> platform-specific failures.

**Goal:** Produce inspectable Windows and Linux execution receipts for the exact candidate wheel
and enforce equivalent verification jobs in the open pull request.

**Architecture:** A stdlib-only outer verifier records dependency bootstrap before importing the
installed project. The installed v2 platform module validates command/evidence completeness and
atomically emits the receipt. GitHub Actions uses the same script and uploads coordinator-only
evidence artifacts.

**Tech Stack:** Python 3.11 stdlib, pip, FastAPI/Uvicorn, PowerShell/POSIX, GitHub Actions,
artifact upload, pytest.

## Global constraints

All constraints in `2026-08-24-breaking-v2-plan-index.md` apply.

---

### Task E1: Implement the cross-platform verifier and receipt

**Files:**
- Modify: `src/rag_evidence/annotation/platform.py`
- Create: `scripts/verify_annotation_platform.py`
- Create: `tests/test_annotation_platform.py`
- Modify: `tests/test_annotation_runtime.py`
- Modify: `tests/test_annotation_handoff.py`

**Interfaces:**
- `CommandRecordV2` records argv, logical cwd, exit code, stdout/stderr digests, UTC start/end, and
  semantic result; secrets and absolute user paths are forbidden.
- `verify_platform_receipt(payload, *, expected: PlatformIdentityV2) -> PlatformVerificationReceiptV2`
  derives overall success from required command records and supplied smoke files.
- `scripts/verify_annotation_platform.py` accepts explicit checkout, wheel, package, lock, launcher,
  output, source commit/tree, and logical platform label. It creates venv/state/cache/output outside
  checkout and never writes a receipt after a failed command.
- Required evidence covers pip/index info, binary-only bootstrap, wheel `--no-deps` install,
  installed distribution list, CLI version/help, app creation, launcher runtime, tasks/progress,
  empty exports, loopback acceptance, and non-loopback rejection before state creation.

- [ ] **Step 1: Write failing receipt-integrity tests**

```python
@pytest.mark.parametrize("missing", REQUIRED_COMMAND_KINDS)
def test_receipt_rejects_missing_command(platform_payload: dict[str, object], missing: str) -> None:
    platform_payload["commands"] = [
        row for row in platform_payload["commands"] if row["kind"] != missing
    ]
    with pytest.raises(ValidationError):
        PlatformVerificationReceiptV2.model_validate(platform_payload)

def test_receipt_binds_exact_distribution_and_wheel(platform_receipt: PlatformVerificationReceiptV2) -> None:
    assert platform_receipt.python_distribution == "0.2.0.dev0"
    assert platform_receipt.wheel_sha256 == EXPECTED_WHEEL_SHA256
    assert platform_receipt.overall_result == "passed"
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/test_annotation_platform.py tests/test_annotation_runtime.py tests/test_annotation_handoff.py -q`

Expected: FAIL because the platform receipt model has no complete command/evidence contract and no
executable verifier exists.

- [ ] **Step 3: Implement the verifier command graph**

```python
BOOTSTRAP_ARGV = (
    python, "-m", "pip", "install", "--require-hashes", "--only-binary=:all:",
    "-r", str(lock),
)
WHEEL_INSTALL_ARGV = (python, "-m", "pip", "install", "--no-deps", str(wheel))

for step in required_steps(inputs):
    records.append(run_and_hash(step))
assert_all_passed(records)
write_platform_receipt_atomically(build_receipt(inputs, records, smoke_hashes))
```

Sanitize index configuration before serialization. Record distribution names/versions sorted by
normalized name. Use a condition-based HTTP readiness probe, stop the server in `finally`, and
retain group-free smoke artifacts until separate cleanup approval.

- [ ] **Step 4: Run GREEN**

Run: `uv run pytest tests/test_annotation_platform.py tests/test_annotation_runtime.py tests/test_annotation_handoff.py -q`

- [ ] **Step 5: Run a native Windows smoke with external directories**

Run `py -3.11 scripts/verify_annotation_platform.py --help`, then execute it against a locally built
test wheel and canonical A package using only paths below `$env:TEMP`. Confirm the receipt and smoke
hashes validate through `verify_platform_receipt`.

- [ ] **Step 6: Commit boundary**

Commit: `feat: attest annotation runtime on target platforms`

### Task E2: Add exact Windows/Linux verification CI jobs

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `pilot/v0.2/HANDOFF_RUNBOOK.md`
- Modify: `pilot/v0.2/COMPLETION_CHECKLIST.md`
- Create: `tests/test_annotation_ci_contract.py`
- Modify: `tests/test_repository_hygiene.py`

**Interfaces:**
- Jobs `annotation-windows` and `annotation-linux` run on `windows-latest` and `ubuntu-latest`.
- Both jobs use Python 3.11, external runner-temp venv/cache/build/output, fixed
  `SOURCE_DATE_EPOCH`, the exact lock/bootstrap, the same verifier semantics, and upload
  `platform-verification-windows` or `platform-verification-linux` evidence.
- Receipt artifacts bind `github.sha`; neither artifact is copied into annotator kits or committed.

- [ ] **Step 1: Write failing workflow-contract tests**

```python
def test_ci_has_symmetric_annotation_platform_jobs() -> None:
    workflow = load_ci_workflow()
    windows = workflow["jobs"]["annotation-windows"]
    linux = workflow["jobs"]["annotation-linux"]
    assert windows["runs-on"] == "windows-latest"
    assert linux["runs-on"] == "ubuntu-latest"
    assert semantic_verifier_steps(windows) == semantic_verifier_steps(linux)
    assert "--only-binary=:all:" in json.dumps((windows, linux))

def test_platform_jobs_upload_coordinator_only_receipts() -> None:
    assert uploaded_artifact_names() == {
        "platform-verification-windows", "platform-verification-linux"
    }
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/test_annotation_ci_contract.py tests/test_repository_hygiene.py -q`

Expected: FAIL because only generic Linux checks and Docker jobs exist.

- [ ] **Step 3: Implement symmetric jobs**

```yaml
annotation-windows:
  runs-on: windows-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with: {python-version: '3.11'}
    - name: Verify annotation platform
      shell: pwsh
      run: python scripts/verify_annotation_platform.py --platform Windows --source-commit $env:GITHUB_SHA

annotation-linux:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with: {python-version: '3.11'}
    - name: Verify annotation platform
      run: python scripts/verify_annotation_platform.py --platform Linux --source-commit "$GITHUB_SHA"
```

Supply all explicit external paths and artifact-upload steps in the actual workflow. The workflow
must fail on dirty/ignored checkout, missing binary wheel, receipt mismatch, privacy scan failure,
or non-loopback acceptance.

- [ ] **Step 4: Run GREEN**

Run: `uv run pytest tests/test_annotation_ci_contract.py tests/test_repository_hygiene.py -q`

- [ ] **Step 5: Local workflow regression**

Run: `uv run ruff check scripts/verify_annotation_platform.py tests/test_annotation_ci_contract.py`

Validate YAML parsing and confirm ordinary `checks` and `docker` jobs remain intact.

- [ ] **Step 6: Commit boundary**

Commit: `ci: verify v2 handoff on Windows and Linux`
