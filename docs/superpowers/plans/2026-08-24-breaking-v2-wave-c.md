# Wave C: Identity, Dependency Lock, and Runtime Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans and
> superpowers:test-driven-development task-by-task.

**Goal:** Establish the private `0.2.0` distribution identity, a binary-only hash-locked
annotation runtime, and fail-closed loopback hosting before source-bound builds begin.

**Architecture:** Version identity is asserted at package, CLI, wheel, receipt, and test boundaries.
A generated requirements lock contains only pinned third-party runtime dependencies. A dedicated
host validator runs before application or state creation and is shared by annotation and
adjudication commands.

**Tech Stack:** Python packaging/Hatch, uv export, pip, socket/ipaddress, Typer, pytest.

## Global constraints

All constraints in `2026-08-24-breaking-v2-plan-index.md` apply.

---

### Task C1: Synchronize distribution and protocol identity

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/rag_evidence/__init__.py`
- Modify: `src/rag_evidence/cli.py`
- Modify: `pilot/v0.2/README.md`
- Modify: `pilot/v0.2/ONBOARDING.md`
- Modify: `pilot/v0.2/COMPLETION_CHECKLIST.md`
- Modify: `tests/test_cli_annotation.py`
- Modify: `tests/test_publication_contract.py`
- Create: `tests/test_distribution_identity.py`

**Interfaces:**
- `rag_evidence.__version__ == "0.2.0"`.
- `rag-evidence --version` prints `rag-evidence-attribution-bench 0.2.0`.
- Protocol and packages bind `pilot-v0.2.2-draft`; v2 schemas retain independent identities.
- Docs state that final `0.2.0` needs separate approval and new exact-source evidence.

- [ ] **Step 1: Write failing identity synchronization tests**

```python
def test_distribution_identity_is_synchronized() -> None:
    assert project_version() == "0.2.0"
    assert rag_evidence.__version__ == "0.2.0"
    assert runner.invoke(app, ["--version"]).stdout.strip().endswith("0.2.0")

def test_protocol_schema_and_distribution_are_distinct() -> None:
    assert protocol_version() == "pilot-v0.2.2-draft"
    assert package_schema() == "assignment-package-v2"
    assert rag_evidence.__version__ == "0.2.0"
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/test_distribution_identity.py tests/test_cli_annotation.py tests/test_publication_contract.py -q`

Expected: FAIL because project/package distribution identity is still `0.1.0`.

- [ ] **Step 3: Apply the exact identities and honest claims**

```toml
[project]
version = "0.2.0"
```

```python
__version__ = "0.2.0"
```

Do not create a tag, release, publication configuration, or final `0.2.0` alias.

- [ ] **Step 4: Run GREEN**

Run: `uv run pytest tests/test_distribution_identity.py tests/test_cli_annotation.py tests/test_publication_contract.py -q`

- [ ] **Step 5: Regression verification**

Run: `uv run pytest tests/test_pilot_package.py tests/test_claim_consistency.py -q`

- [ ] **Step 6: Commit boundary**

Commit: `chore: set breaking v2 development identity`

### Task C2: Generate and validate the annotation requirements lock

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `pilot/v0.2/annotation-requirements-py311.lock`
- Create: `scripts/build_annotation_lock.py`
- Create: `tests/test_annotation_dependency_lock.py`
- Modify: `pilot/v0.2/HANDOFF_RUNBOOK.md`

**Interfaces:**
- Optional dependency group `annotation` contains FastAPI and Uvicorn runtime dependencies but no
  Gradio, ML/GPU dependency, project self-reference, or build dependency.
- `build_annotation_lock(repository_root: Path) -> bytes` runs a frozen uv export without emitting
  the root project and validates every requirement line.
- `validate_annotation_lock(text: str) -> None` rejects editable, local path, VCS, direct root,
  unpinned, unhashed, option-injected, and index-secret entries.
- Bootstrap is exactly `python -m pip install --require-hashes --only-binary=:all:
  -r annotation-requirements-py311.lock`, followed by project wheel `--no-deps` installation.

- [ ] **Step 1: Write failing lock purity tests**

```python
@pytest.mark.parametrize("bad_line", [
    "-e .", ".", "git+https://example.invalid/repo.git", "fastapi>=0.111",
    "rag-evidence-attribution-bench==0.2.0", "--extra-index-url https://user:secret@example.invalid",
])
def test_lock_validator_rejects_non_third_party_or_unpinned_entries(bad_line: str) -> None:
    with pytest.raises(DataError):
        validate_annotation_lock(valid_lock_text() + "\n" + bad_line)

def test_generated_lock_is_reproducible() -> None:
    assert build_annotation_lock(REPOSITORY_ROOT) == LOCK_PATH.read_bytes()
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/test_annotation_dependency_lock.py -q`

Expected: FAIL because the annotation extra, generator, lock, and validator do not exist.

- [ ] **Step 3: Implement lock generation and structural validation**

```python
EXPORT_ARGV = (
    "uv", "export", "--frozen", "--no-dev", "--extra", "annotation",
    "--no-emit-project", "--no-header", "--format", "requirements-txt",
)

def validate_requirement(requirement: Requirement) -> None:
    if requirement.name == "rag-evidence-attribution-bench" or requirement.url:
        raise DataError("annotation lock must contain third-party index distributions only")
    if len(list(requirement.specifier)) != 1 or "==" not in str(requirement.specifier):
        raise DataError("annotation dependency must be exactly pinned")
```

Generate the lock with hashes and platform markers. The runbook must never upgrade pip, resolve the
project extra, remove `--only-binary`, or install an sdist.

- [ ] **Step 4: Run GREEN**

Run: `uv run pytest tests/test_annotation_dependency_lock.py -q`

- [ ] **Step 5: Exercise the exact Windows bootstrap in an external venv**

Run:

```powershell
$Venv = Join-Path $env:TEMP 'reab-lock-smoke-win'
py -3.11 -m venv $Venv
& (Join-Path $Venv 'Scripts/python.exe') -m pip install --require-hashes --only-binary=:all: -r pilot/v0.2/annotation-requirements-py311.lock
```

Expected: every selected dependency installs from a compatible binary wheel. Missing binary is a
feasibility failure; do not alter the command.

- [ ] **Step 6: Commit boundary**

Commit: `build: lock binary-only annotation runtime`

### Task C3: Enforce loopback-only runtime before state creation

**Files:**
- Create: `src/rag_evidence/annotation/runtime.py`
- Modify: `src/rag_evidence/annotation/app.py`
- Modify: `src/rag_evidence/cli.py`
- Modify: `pilot/v0.2/launchers/start-a.ps1`
- Modify: `pilot/v0.2/launchers/start-a.sh`
- Modify: `pilot/v0.2/launchers/start-b.ps1`
- Modify: `pilot/v0.2/launchers/start-b.sh`
- Create: `tests/test_annotation_runtime.py`
- Modify: `tests/test_cli_annotation.py`

**Interfaces:**
- `validate_loopback_host(host: str, *, resolver=socket.getaddrinfo) -> str` accepts `127.0.0.1`,
  `::1`, and exact `localhost` only when all resolved addresses are loopback.
- `serve_annotation(...)` and `serve_adjudication(...)` validate host before package loading,
  application creation, state directory creation, or `uvicorn.run`.
- Launchers pass fixed `127.0.0.1` and offer no host override.

- [ ] **Step 1: Write failing host and no-side-effect tests**

```python
@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.5", "8.8.8.8", "example.com"])
def test_non_loopback_host_fails_before_state_creation(tmp_path: Path, host: str) -> None:
    state = tmp_path / "state"
    with pytest.raises(DataError, match="loopback"):
        serve_annotation(package, state, host=host, port=8001)
    assert not state.exists()

def test_localhost_rejects_mixed_resolution() -> None:
    with pytest.raises(DataError):
        validate_loopback_host("localhost", resolver=mixed_loopback_and_public_resolver)
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/test_annotation_runtime.py tests/test_cli_annotation.py -q`

Expected: FAIL because current CLI passes arbitrary host directly to Uvicorn.

- [ ] **Step 3: Implement the pre-state host gate**

```python
def validate_loopback_host(host: str, *, resolver: Resolver = socket.getaddrinfo) -> str:
    if host in {"127.0.0.1", "::1"}:
        return host
    if host != "localhost":
        raise DataError("annotation runtime host must be loopback")
    addresses = {ipaddress.ip_address(row[4][0]) for row in resolver(host, None)}
    if not addresses or not all(address.is_loopback for address in addresses):
        raise DataError("localhost must resolve only to loopback addresses")
    return host
```

- [ ] **Step 4: Run GREEN**

Run: `uv run pytest tests/test_annotation_runtime.py tests/test_cli_annotation.py -q`

- [ ] **Step 5: Regression verification**

Run: `uv run pytest tests/test_annotation_api.py tests/test_adjudication_api.py -q`

- [ ] **Step 6: Commit boundary**

Commit: `feat: enforce loopback-only annotation runtime`
