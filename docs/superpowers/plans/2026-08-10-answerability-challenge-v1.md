# Answerability Challenge v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic 960-record Gate B3 challenge set with explicit provisional-review gating, immutable provenance, and no mutation of the natural HotpotQA v2 benchmark.

**Architecture:** Add a strict challenge record schema, pure deterministic transforms, a source-bound manifest, and an atomic preparation boundary exposed as `data challenge`. Challenge JSONL lives below versioned v2 paths; missing-hop and evidence-swap records remain confirmatory-ineligible until human adjudication.

**Tech Stack:** Python 3.11, dataclasses, Pydantic 2 configuration, Typer, SHA-256 canonical JSON, pytest, Ruff, mypy.

## Global Constraints

- Preserve `data/manifests/split_manifest.json`, `data/manifests/split_manifest_v2.json`, `results/raw`, `results/derived`, and natural `results/v2/raw/*/samples/records.jsonl` byte-for-byte.
- Consume only manifest schema v2 and the pinned HotpotQA revision `1908d6afbbead072334abe2965f91bd2709910ab`.
- Generate exactly one `missing_hop`, `answer_bearing_distractor`, and `evidence_swap` record for every selected parent.
- Never describe automatically expected unanswerability as human-validated ground truth.
- Keep all missing-hop and evidence-swap records `pending_two_annotators` and `confirmatory_eligible: false`.
- Use no remote model, LLM judge, paid API, GPU, push, or publication action.
- Every production behavior follows RED-GREEN-REFACTOR TDD.

---

### Task 1: Atomic Whole-File JSONL Writes

**Files:**
- Modify: `src/rag_evidence/storage/artifacts.py`
- Modify: `tests/test_artifacts.py`

**Interfaces:**
- Produces: `write_records_atomic(path: Path, records: Iterable[Mapping[str, Any]]) -> None`.
- Guarantees: UTF-8, compact one-object-per-line JSON, flush/fsync, `os.replace`, and temp cleanup on serialization or filesystem failure.

- [ ] **Step 1: Write failing atomic-writer tests**

  Add tests that write two Unicode records, assert exact readback and no `*.tmp.*` file,
  replace an existing file atomically, and pass a non-JSON-serializable value to prove the
  old file survives and the temp file is removed.

- [ ] **Step 2: Run RED**

  ```powershell
  pytest tests/test_artifacts.py -q
  ```

  Expected: import failure because `write_records_atomic` does not exist.

- [ ] **Step 3: Implement the minimal writer**

  Serialize each mapping with `ensure_ascii=False` and `separators=(",", ":")` into a
  PID-suffixed sibling temp file, flush/fsync once after all rows, and replace the target.
  Catch serialization and `OSError` failures, delete the temp file, and raise
  `ArtifactError` while leaving an existing target untouched.

- [ ] **Step 4: Run GREEN and static checks**

  ```powershell
  pytest tests/test_artifacts.py -q
  ruff check src/rag_evidence/storage/artifacts.py tests/test_artifacts.py
  mypy src/rag_evidence/storage/artifacts.py
  ```

- [ ] **Step 5: Commit Task 1**

  ```powershell
  git add src/rag_evidence/storage/artifacts.py tests/test_artifacts.py
  git commit -m "feat: add atomic JSONL artifact writes"
  ```

### Task 2: Strict Challenge Record Schema

**Files:**
- Create: `src/rag_evidence/data/challenge_schema.py`
- Create: `tests/test_challenge_schema.py`

**Interfaces:**
- Produces constants: `CHALLENGE_SCHEMA_VERSION = 1`,
  `TRANSFORMATIONS = ("missing_hop", "answer_bearing_distractor", "evidence_swap")`.
- Produces: `ChallengeReview` and `ChallengeRecord` frozen dataclasses with strict
  `to_json()` / `from_json()` methods.
- Produces: `make_challenge_id(parent_question_id, source_split, transformation, *, transform_version) -> str`.
- Produces: `record_content_hash(payload_without_content_hash: Mapping[str, Any]) -> str`.
- Produces: `remap_example(example: Example, challenge_id: str, *, passages: Sequence[Passage], gold_slots: set[int], supporting_slots: set[tuple[int, int]]) -> Example`.

- [ ] **Step 1: Write failing schema tests**

  Assert an invented record round-trips exactly, IDs match `ch-[0-9a-f]{24}`, passage and
  sentence IDs use the challenge ID, and the content hash is stable under mapping key
  order. Parameterize mutations for unknown/missing fields, duplicate/unsorted
  `changed_fields`, an invalid review tuple, malformed stable IDs, and a one-byte payload
  change without hash update; each must raise `DataError` naming the invariant.

- [ ] **Step 2: Run RED**

  ```powershell
  pytest tests/test_challenge_schema.py -q
  ```

  Expected: import failure because `challenge_schema` does not exist.

- [ ] **Step 3: Implement strict parsing, hashing, and ID remapping**

  Review states are exactly:

  ```python
  NOT_REQUIRED = {"required": False, "status": "not_required", "confirmatory_eligible": True}
  PENDING = {"required": True, "status": "pending_two_annotators", "confirmatory_eligible": False}
  ```

  Reject any other combination. Hash compact sorted-key UTF-8 JSON. Preserve passage slot
  order and rebuild all passage/sentence labels from slot coordinates.

- [ ] **Step 4: Run GREEN and static checks**

  ```powershell
  pytest tests/test_challenge_schema.py -q
  ruff check src/rag_evidence/data/challenge_schema.py tests/test_challenge_schema.py
  mypy src/rag_evidence/data/challenge_schema.py
  ```

- [ ] **Step 5: Commit Task 2**

  ```powershell
  git add src/rag_evidence/data/challenge_schema.py tests/test_challenge_schema.py
  git commit -m "feat: define strict challenge record schema"
  ```

### Task 3: Deterministic Challenge Transformations

**Files:**
- Create: `src/rag_evidence/data/challenge_transforms.py`
- Create: `tests/test_challenge_transforms.py`

**Interfaces:**
- Consumes: `Example`, parent fingerprints, split membership, challenge schema helpers.
- Produces: `build_missing_hop(parent, split_parents, *, source_split, seed, transform_version, parent_fingerprint) -> ChallengeRecord`.
- Produces: `build_answer_bearing_distractor(parent, *, source_split, seed, transform_version, parent_fingerprint) -> ChallengeRecord`.
- Produces: `build_evidence_swap(parent, *, source_split, seed, transform_version, parent_fingerprint) -> ChallengeRecord`.
- Produces: `build_challenge_records(parents_by_split, parent_fingerprints, *, seed, transform_version) -> dict[str, tuple[ChallengeRecord, ...]]`.

- [ ] **Step 1: Write failing missing-hop tests**

  Invent two valid parents with two gold and at least two non-gold passages. Assert seeded
  order independence; donor comes from another parent in the same split; donor answer,
  title-overlap, and paragraph-overlap candidates are rejected; exactly one gold slot is
  replaced; removed sentence labels disappear; the parent object is unchanged; and donor
  exhaustion raises `DataError`.

- [ ] **Step 2: Run missing-hop RED, implement, and run GREEN**

  ```powershell
  pytest tests/test_challenge_transforms.py -q -k missing_hop
  ```

  Candidate rank is SHA-256 of
  `seed:parent_question_id:missing_hop:candidate_id`. Filter before ranking, never add an
  unsafe fallback.

- [ ] **Step 3: Write failing answer-bearing-distractor tests**

  Assert one non-gold passage gains exactly one sentence, normalized answer tokens occur
  in the appended sentence, complete gold passage/sentence labels remain unchanged after
  ID remapping, and the review state is `not_required`. When all non-gold passages already
  contain the answer, require the same one-sentence intervention plus explicit
  `salience_only` / `preexisting_answer_mention` provenance; only a parent with no non-gold
  passage raises `DataError`.

- [ ] **Step 4: Run answer-bearing RED, implement, and run GREEN**

  ```powershell
  pytest tests/test_challenge_transforms.py -q -k answer_bearing
  ```

  Insert exactly `<title> has also been associated with <answer>.` and store the inserted
  sentence and selected source slot in provenance.

- [ ] **Step 5: Write failing evidence-swap tests**

  Cover all three edit operators: literal answer replacement, auxiliary negation toggle,
  and prefix fallback. Assert only one sentence changes, the changed supporting sentence
  label is removed, a passage remains gold only when another supporting sentence remains,
  the original surface sentence is recorded, review remains pending, and no supporting or
  non-gold candidate raises `DataError`.

- [ ] **Step 6: Run evidence-swap RED, implement, and run GREEN**

  ```powershell
  pytest tests/test_challenge_transforms.py -q -k evidence_swap
  ```

- [ ] **Step 7: Write failing three-per-parent and reorder tests**

  Call `build_challenge_records` with normal and reversed split/parent order. Assert exact
  tuple equality, sorted unique challenge IDs, three transformation names per parent,
  answerability counts, and source objects unchanged.

- [ ] **Step 8: Implement collection construction and verify Task 3**

  ```powershell
  pytest tests/test_challenge_transforms.py -q
  ruff check src/rag_evidence/data/challenge_transforms.py tests/test_challenge_transforms.py
  mypy src/rag_evidence/data/challenge_transforms.py
  ```

- [ ] **Step 9: Commit Task 3**

  ```powershell
  git add src/rag_evidence/data/challenge_transforms.py tests/test_challenge_transforms.py
  git commit -m "feat: generate deterministic answerability challenges"
  ```

### Task 4: Source-Bound Challenge Manifest

**Files:**
- Create: `src/rag_evidence/data/challenge_manifest.py`
- Create: `tests/test_challenge_manifest.py`

**Interfaces:**
- Produces: `build_challenge_manifest(records_by_split, *, source_manifest_path, source_manifest_bytes, source_manifest, seed, transform_version) -> dict[str, Any]`.
- Produces: `validate_challenge_manifest(manifest, records_by_split, *, source_manifest_path, source_manifest_bytes, source_manifest, parents_by_split, parent_fingerprints) -> None`.
- Manifest schema: source path/SHA/revision/dataset hash, generation protocol, per-split
  parent/record distributions, per-record hashes, per-split hashes.

- [ ] **Step 1: Write failing manifest contract tests**

  Build a tiny three-split source manifest and challenge records. Assert deterministic
  output under reversed records, exact counts and hashes, and successful independent
  validation. Mutate source SHA, dataset revision, parent ID, record hash, split hash,
  transformation count, review count, duplicate ID, and one nested example byte
  separately; each must raise `DataError` naming the affected invariant.

- [ ] **Step 2: Run RED**

  ```powershell
  pytest tests/test_challenge_manifest.py -q
  ```

  Expected: import failure because `challenge_manifest` does not exist.

- [ ] **Step 3: Implement deterministic manifest build and independent recomputation**

  Recompute expected records with `build_challenge_records` during validation instead of
  trusting stored changed-field or review metadata. Aggregate hashes are newline-joined
  sorted `challenge_id:content_hash` pairs.

- [ ] **Step 4: Run GREEN and static checks**

  ```powershell
  pytest tests/test_challenge_manifest.py tests/test_challenge_transforms.py -q
  ruff check src/rag_evidence/data/challenge_manifest.py tests/test_challenge_manifest.py
  mypy src/rag_evidence/data/challenge_manifest.py
  ```

- [ ] **Step 5: Commit Task 4**

  ```powershell
  git add src/rag_evidence/data/challenge_manifest.py tests/test_challenge_manifest.py
  git commit -m "feat: bind challenge manifest to source data"
  ```

### Task 5: Configuration, CLI, and Atomic Preparation Boundary

**Files:**
- Modify: `src/rag_evidence/config.py`
- Modify: `src/rag_evidence/cli.py`
- Modify: `src/rag_evidence/pipeline.py`
- Create: `src/rag_evidence/data/challenge_prepare.py`
- Modify: `configs/v2/smoke.yaml`
- Modify: `configs/v2/dev.yaml`
- Modify: `configs/v2/eval.yaml`
- Modify: `.gitignore`
- Modify: `tests/test_config.py`
- Create: `tests/test_challenge_prepare.py`
- Modify: `tests/test_repository_hygiene.py`
- Modify: `tests/test_transfer.py`

**Interfaces:**
- Adds frozen `ChallengeConfig` to `AppConfig` with exact schema/version/seed and relative
  manifest/prepared paths.
- Adds: `prepare_challenge(cfg: AppConfig) -> None`.
- Adds: `run_data_challenge(cfg: AppConfig) -> None`.
- Adds CLI: `data challenge --config PATH`.

- [ ] **Step 1: Write failing configuration tests**

  Assert every v2 config has schema 1, seed 20260810, transform version `challenge-v1`,
  manifest `data/manifests/challenge_manifest_v1.json`, and prepared directory
  `data/v2/challenge`. Reject absolute paths, wrong versions, and invoking challenge
  preparation from a schema-v1 dataset config.

- [ ] **Step 2: Run config RED, implement the strict config, and run GREEN**

  ```powershell
  pytest tests/test_config.py -q -k challenge
  ```

- [ ] **Step 3: Write failing preparation tests**

  In a temporary directory, write invented source manifest/prepared files. Assert the
  stage verifies every parent QID and `raw_fingerprint`, builds or validates the immutable
  challenge manifest, writes only challenge paths, produces three rows per parent, and is
  byte-idempotent. Corrupt source fingerprints, an existing challenge manifest, and a
  missing split separately; each must fail before any challenge file is replaced.

- [ ] **Step 4: Implement the preparation stage with atomic JSONL writes**

  Load raw prepared dictionaries so `raw_fingerprint` remains available, then construct
  `Example` values. When a challenge manifest exists, validate it before writing derived
  files. When absent, write it once with `write_json_atomic`; do not overwrite it on later
  runs. Use `write_records_atomic` for prepared and sample JSONL.

- [ ] **Step 5: Add CLI/pipeline wiring and repository tracking tests**

  Add the Typer command and lazy pipeline import. Explicitly allow
  `data/manifests/challenge_manifest_v1.json` in `.gitignore`; keep `data/v2/challenge`
  ignored. Extend transfer testing to prove nested
  `results/v2/raw/eval/challenge/samples/records.jsonl` retains its canonical path.

- [ ] **Step 6: Verify Task 5**

  ```powershell
  pytest tests/test_config.py tests/test_challenge_prepare.py tests/test_repository_hygiene.py tests/test_transfer.py -q
  ruff check src/rag_evidence/config.py src/rag_evidence/cli.py src/rag_evidence/pipeline.py src/rag_evidence/data/challenge_prepare.py tests/test_challenge_prepare.py
  mypy src/rag_evidence/config.py src/rag_evidence/cli.py src/rag_evidence/pipeline.py src/rag_evidence/data/challenge_prepare.py
  git diff --exit-code 3ae961f -- data/manifests/split_manifest.json data/manifests/split_manifest_v2.json results/raw results/derived results/v2/raw/smoke/samples/records.jsonl results/v2/raw/dev/samples/records.jsonl results/v2/raw/eval/samples/records.jsonl
  ```

- [ ] **Step 7: Commit Task 5**

  ```powershell
  git add .gitignore configs/v2 src/rag_evidence/config.py src/rag_evidence/cli.py src/rag_evidence/pipeline.py src/rag_evidence/data/challenge_prepare.py tests/test_config.py tests/test_challenge_prepare.py tests/test_repository_hygiene.py tests/test_transfer.py
  git commit -m "feat: prepare versioned challenge artifacts"
  ```

### Task 6: Real Challenge Candidate Set and Audit

**Files:**
- Create: `data/manifests/challenge_manifest_v1.json`
- Create: `docs/CHALLENGE_V1.md`
- Generate locally: `data/v2/challenge/{smoke,dev,eval}.jsonl`
- Generate: `results/v2/raw/{smoke,dev,eval}/challenge/samples/records.jsonl`

**Interfaces:**
- Consumes the committed source manifest v2 and prepared v2 records.
- Produces the immutable 960-record candidate set and bounded audit documentation.

- [ ] **Step 1: Generate the real challenge set**

  ```powershell
  python -m rag_evidence.cli data challenge --config configs/v2/eval.yaml
  ```

  Expected: 320 parents and 960 rows; no network or model access.

- [ ] **Step 2: Independently validate and reverse-rebuild**

  Reload the source prepared files, call `validate_challenge_manifest`, rebuild with
  reversed split/parent order, and compare canonical manifest and JSONL bytes. Confirm
  exact three-per-parent coverage and no duplicate IDs.

- [ ] **Step 3: Audit transformation invariants**

  Recompute counts and assert 320 records per transformation; 640 pending review and
  confirmatory-ineligible; 320 answerable distractor rows retaining all parent evidence;
  missing-hop donors from the same split and different parent; answer-free missing-hop
  donors; and one edited supporting sentence per evidence swap.

- [ ] **Step 4: Write `docs/CHALLENGE_V1.md`**

  Record source/challenge hashes, exact counts, transform operators, donor failure count,
  answerability expectations, review status, known synthetic-style/evidence-exhaustivity
  limitations, license, and the explicit prohibition on confirmatory missing-hop or
  evidence-swap claims before annotation.

- [ ] **Step 5: Verify real idempotence and historical isolation**

  Hash manifest and all tracked challenge samples, rerun the command, and require identical
  hashes. Compare every protected natural/historical path to commit `3ae961f`.

- [ ] **Step 6: Commit Task 6**

  ```powershell
  git add data/manifests/challenge_manifest_v1.json docs/CHALLENGE_V1.md results/v2/raw/*/challenge
  git commit -m "data: add deterministic answerability challenge v1"
  ```

### Task 7: Final Verification and Human Handoff

**Files:**
- Modify only files required by a reproduced verification failure, with a failing regression test first.

**Interfaces:**
- Produces a local engineering-verification record and a precise human-review boundary.
- Does not create annotation judgments, push, publish, or reserve a DOI.

- [ ] **Step 1: Run full offline verification**

  ```powershell
  ruff format --check .
  ruff check .
  mypy src
  pytest -m "not gpu and not slow" -q --cov=rag_evidence --cov-report=term-missing
  ```

- [ ] **Step 2: Run challenge acceptance assertions**

  Confirm exact source/manifest SHA values, 960 unique rows, 320 per transform, exact
  review gating, stable-ID remapping, parent evidence retention/removal rules, reverse and
  repeated byte identity, and zero changes in every protected natural/historical path.

- [ ] **Step 3: Prepare the handoff**

  Report exact test/coverage results, challenge hash/counts/operator distribution, any
  generation exclusions, and the next required human action: two independent annotators
  plus a third adjudicator for missing-hop/evidence-swap validity and the later stratified
  Gate B4 audit. Keep all affected scientific claims explicitly blocked until that work is
  complete.
