# Gate B Dataset v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pinned, deterministic HotpotQA manifest v2 whose smoke/dev/eval splits have zero question-ID, normalized-title, and canonical-paragraph overlap while preserving complete supporting evidence.

**Architecture:** Keep schema-v1 manifests and artifacts immutable. Add focused grouping and manifest-v2 modules, expose them through the existing data preparation boundary, and write all new prepared/sample artifacts under versioned v2 paths. A seeded dynamic-programming allocator assigns whole connected components in eval/dev/smoke order and minimizes size error before bridge/comparison distribution error.

**Tech Stack:** Python 3.11, Pydantic 2, Hugging Face `datasets`/`huggingface_hub`, pytest, Ruff, mypy.

## Global Constraints

- Preserve `data/manifests/split_manifest.json`, `results/raw`, and `results/derived` byte-for-byte.
- Pin `hotpotqa/hotpot_qa` to `1908d6afbbead072334abe2965f91bd2709910ab`.
- New manifest: `data/manifests/split_manifest_v2.json`; new data artifacts: `data/v2/prepared` and `results/v2/raw`.
- Cross-split overlap must be zero for question ID, normalized title, and canonical paragraph fingerprint.
- Selected examples must retain every supporting-fact title and sentence index; malformed examples are excluded with counted reasons.
- Requested sizes are 20 smoke / 60 dev / 240 eval. Whole groups are never split to force cosmetic sizes; requested and realized sizes are both recorded.
- Selection order is eval, dev, smoke. Primary allocation objective is absolute size error; secondary objective is bridge/comparison distribution error; seeded order breaks exact ties.
- Manifest generation is byte-deterministic and contains no wall-clock timestamp.
- No paid API, model execution, publication, or push.
- Every behavior change follows red-green-refactor TDD.

---

### Task 1: Canonical Group Identities and Connected Components

**Files:**
- Create: `src/rag_evidence/data/grouping.py`
- Create: `tests/test_grouping.py`

**Interfaces:**
- Produces: `normalize_title(title: str) -> str`
- Produces: `paragraph_fingerprint(sentences: Sequence[str]) -> str`
- Produces: `supporting_fact_errors(raw: Mapping[str, Any]) -> tuple[str, ...]`
- Produces: `build_split_groups(raw_examples: Sequence[dict[str, Any]]) -> tuple[SplitGroup, ...]`
- `SplitGroup` exposes `group_id`, sorted `question_ids`, `size`, `bridge_count`, `comparison_count`, normalized titles, and paragraph hashes.

- [ ] **Step 1: Write failing canonicalization and grouping tests**

  Build invented raw examples where two questions share a title after NFKC/case/whitespace normalization, two share identical canonical paragraph text under different titles, and one is isolated. Assert the transitive closure forms exactly two groups, input order does not matter, and group IDs are stable SHA-256 values.

  ```python
  groups = build_split_groups([isolated, shared_paragraph, shared_title])
  assert sorted(group.size for group in groups) == [1, 2]
  assert build_split_groups(list(reversed(raws))) == groups
  ```

- [ ] **Step 2: Run the tests and verify RED**

  Run: `pytest tests/test_grouping.py -q`

  Expected: import failure because `rag_evidence.data.grouping` does not exist.

- [ ] **Step 3: Implement canonical identities and union-find grouping**

  Normalize titles with Unicode NFKC, collapsed whitespace, trimming, and `casefold()`. Canonicalize each paragraph as compact UTF-8 JSON over an ordered list of NFKC/collapsed-whitespace sentences, then SHA-256 it. Union questions sharing either key and derive `group_id` from newline-joined sorted question IDs.

- [ ] **Step 4: Add failing supporting-evidence eligibility cases**

  Assert `supporting_fact_errors` reports `title_not_in_context`, `sent_id_out_of_range`, and `duplicate_context_title`, and returns an empty tuple only when every supporting fact resolves.

- [ ] **Step 5: Implement eligibility checks and verify GREEN**

  Run:

  ```powershell
  pytest tests/test_grouping.py -q
  ruff check src/rag_evidence/data/grouping.py tests/test_grouping.py
  mypy src/rag_evidence/data/grouping.py
  ```

- [ ] **Step 6: Commit Task 1**

  ```powershell
  git add src/rag_evidence/data/grouping.py tests/test_grouping.py
  git commit -m "feat: define leakage-resistant split groups"
  ```

### Task 2: Deterministic Whole-Group Allocation

**Files:**
- Create: `src/rag_evidence/data/allocation.py`
- Create: `tests/test_group_allocation.py`

**Interfaces:**
- Consumes: `SplitGroup` from Task 1.
- Produces: `allocate_split_groups(groups, *, requested_sizes, seed) -> AllocationResult`.
- `AllocationResult` exposes split-to-groups, requested/realized sizes, type distributions, and unselected counts.

- [ ] **Step 1: Write failing allocation tests**

  Cover deterministic output under reversed input, no group reuse, eval/dev/smoke draw order, exact sizes when reachable, nearest size when no exact whole-group solution exists, bridge-count tie-breaking, and a changed selection under a different seed when objective values tie.

  ```python
  result = allocate_split_groups(groups, requested_sizes=SIZES, seed=17)
  selected = [group.group_id for rows in result.groups.values() for group in rows]
  assert len(selected) == len(set(selected))
  assert result.realized_sizes == {"smoke": 2, "dev": 3, "eval": 4}
  ```

- [ ] **Step 2: Run the tests and verify RED**

  Run: `pytest tests/test_group_allocation.py -q`

  Expected: import failure because the allocation module does not exist.

- [ ] **Step 3: Implement the allocator**

  Seed-sort groups by SHA-256 of `seed:group_id`. For each split in `("eval", "dev", "smoke")`, use a bounded 0/1 dynamic program over `(total_size, bridge_count)`. Choose the terminal state minimizing:

  ```python
  (
      abs(total_size - requested_size),
      abs(bridge_count - target_bridge_count),
      total_size > requested_size,
      seeded_terminal_order,
  )
  ```

  Remove every selected component before allocating the next split. Never split a component.

- [ ] **Step 4: Add property-style invariants and verify GREEN**

  Parametrize seeds and mixed group sizes; assert requested/realized accounting, type-count sums, and that the allocator never returns a partial group.

  Run:

  ```powershell
  pytest tests/test_group_allocation.py -q
  ruff check src/rag_evidence/data/allocation.py tests/test_group_allocation.py
  mypy src/rag_evidence/data/allocation.py
  ```

- [ ] **Step 5: Commit Task 2**

  ```powershell
  git add src/rag_evidence/data/allocation.py tests/test_group_allocation.py
  git commit -m "feat: allocate whole leakage groups deterministically"
  ```

### Task 3: Manifest v2 Construction and Fail-Closed Validation

**Files:**
- Create: `src/rag_evidence/data/manifest_v2.py`
- Create: `tests/test_manifest_v2.py`
- Modify: `src/rag_evidence/data/splits.py`

**Interfaces:**
- Consumes: grouping and allocation results from Tasks 1–2.
- Produces: `build_manifest_v2(raw_examples, *, seed, requested_sizes, dataset_info) -> dict[str, Any]`.
- Produces: `validate_manifest_v2(manifest, raw_examples) -> None`.
- `load_manifest` remains compatible with schema v1 and validates schema v2 structure before returning it.

- [ ] **Step 1: Write a failing manifest-v2 contract test**

  Assert schema version 2, exact 40-character source revision, deterministic bytes, requested and realized sizes, pool/exclusion counts, split group IDs, type/level distributions, example hashes, dataset hash, and overlap audit fields all equal to zero.

- [ ] **Step 2: Add failing corruption tests**

  Mutate a title, paragraph, group ID, source revision, supporting sentence index, split question ID, and overlap-audit count separately. Each mutation must make `validate_manifest_v2` raise `DataError` with the affected invariant in its message.

- [ ] **Step 3: Run tests and verify RED**

  Run: `pytest tests/test_manifest_v2.py -q`

  Expected: import failure because `manifest_v2` does not exist.

- [ ] **Step 4: Implement construction and independent validation**

  Filter malformed examples before grouping and record exclusion reasons. Store the complete source dataset hash separately from selected-example hashes. Recompute group keys and overlap counts during validation rather than trusting recorded audit values. Reject non-hex or non-40-character revisions.

- [ ] **Step 5: Make the existing manifest façade schema-aware**

  Preserve the current schema-v1 `build_manifest` behavior. Change `load_manifest` to accept versions 1 and 2, and change `check_manifest_matches_config` to compare v2 config sizes with `selection.requested_sizes` while preparation consumes `splits[*].size` as realized.

- [ ] **Step 6: Verify GREEN and compatibility**

  Run:

  ```powershell
  pytest tests/test_manifest_v2.py tests/test_splits.py tests/test_e2e_tiny.py -q
  ruff check src/rag_evidence/data tests/test_manifest_v2.py tests/test_splits.py
  mypy src/rag_evidence/data
  ```

- [ ] **Step 7: Commit Task 3**

  ```powershell
  git add src/rag_evidence/data/splits.py src/rag_evidence/data/manifest_v2.py tests/test_manifest_v2.py
  git commit -m "feat: add fail-closed manifest schema v2"
  ```

### Task 4: Pinned Data Preparation and Versioned Configuration

**Files:**
- Modify: `src/rag_evidence/config.py`
- Modify: `src/rag_evidence/data/hotpot.py`
- Modify: `src/rag_evidence/storage/transfer.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `tests/test_config.py`
- Modify: `tests/test_transfer.py`
- Create: `tests/test_data_prepare_v2.py`
- Create: `configs/v2/smoke.yaml`
- Create: `configs/v2/dev.yaml`
- Create: `configs/v2/eval.yaml`

**Interfaces:**
- Adds: `DataConfig.manifest_schema_version: Literal[1, 2] = 1`.
- Adds: `verify_resolved_dataset_revision(repo_id, requested_revision) -> str` at the data-loading boundary.
- Historical configs remain schema 1; v2 configs use the pinned revision and versioned paths.
- Transfer bundles retain configured v2 roots and filenames; v2 artifacts cannot be imported into historical `results/raw`, `results/derived`, or the v1 manifest path.

- [ ] **Step 1: Write failing configuration tests**

  Assert v2 configs load, use the exact pinned revision, write only to v2 paths, and reject null/short/symbolic revisions when `manifest_schema_version: 2`. Assert every historical config still loads unchanged as schema 1.

- [ ] **Step 2: Run config tests and verify RED**

  Run: `pytest tests/test_config.py -q`

  Expected: failure because `manifest_schema_version` is an unknown key and v2 configs do not exist.

- [ ] **Step 3: Implement strict configuration and add v2 YAMLs**

  Use `data/manifests/split_manifest_v2.json`, `data/v2/prepared`, `results/v2/raw`, `results/v2/derived`, and `results/v2/assets`. Preserve all scientific generation/retrieval/attribution settings from the matching v1 config.

  Declare `huggingface-hub>=0.30` directly in the `ml` optional dependency because production code calls `HfApi`, then refresh `uv.lock` without changing unrelated constraints.

- [ ] **Step 4: Write failing v2 preparation tests**

  Monkeypatch raw loading and revision resolution with invented local data. Assert v2 preparation builds/validates the v2 manifest, writes only v2 prepared/sample paths, refuses a resolved-revision mismatch before writing, and remains idempotent when the committed manifest exists.

- [ ] **Step 5: Implement the pinned preparation path**

  Resolve the dataset revision via `huggingface_hub.HfApi.dataset_info`; reject any resolved SHA unequal to the requested SHA. Dispatch to `build_manifest_v2` only for schema 2. Change transfer bundle paths to preserve configured result roots and the manifest filename instead of mapping v2 artifacts onto v1 paths.

- [ ] **Step 6: Verify GREEN and historical isolation**

  Run:

  ```powershell
  pytest tests/test_config.py tests/test_data_prepare_v2.py tests/test_transfer.py -q
  ruff check src/rag_evidence/config.py src/rag_evidence/data/hotpot.py src/rag_evidence/storage/transfer.py tests/test_data_prepare_v2.py
  mypy src/rag_evidence/config.py src/rag_evidence/data/hotpot.py src/rag_evidence/storage/transfer.py
  git diff --exit-code HEAD -- data/manifests/split_manifest.json results/raw results/derived
  ```

- [ ] **Step 7: Commit Task 4**

  ```powershell
  git add src/rag_evidence/config.py src/rag_evidence/data/hotpot.py src/rag_evidence/storage/transfer.py pyproject.toml uv.lock configs/v2 tests/test_config.py tests/test_data_prepare_v2.py tests/test_transfer.py
  git commit -m "feat: prepare pinned versioned dataset artifacts"
  ```

### Task 5: Build and Audit the Real HotpotQA Manifest v2

**Files:**
- Create: `data/manifests/split_manifest_v2.json`
- Create: `docs/DATASET_V2.md`
- Generate locally: `data/v2/prepared/*.jsonl`
- Generate: `results/v2/raw/{smoke,dev,eval}/samples/records.jsonl`

**Interfaces:**
- Consumes the pinned upstream revision through `configs/v2/eval.yaml`.
- Produces the immutable manifest and its human-readable leakage/eligibility audit.

- [ ] **Step 1: Run pinned data preparation**

  Run: `python -m rag_evidence.cli data prepare --config configs/v2/eval.yaml`

  Expected: the resolved SHA exactly matches the requested SHA before normalization or writes.

- [ ] **Step 2: Independently validate the generated manifest**

  Load the pinned raw examples again and call `validate_manifest_v2`. Assert zero recorded and recomputed overlap, all selected supporting facts resolve, all three realized sizes are positive, and every selected question belongs to exactly one whole group.

- [ ] **Step 3: Rebuild in a temporary directory and compare bytes**

  Call `build_manifest_v2` again with reversed raw input and compare canonical JSON bytes with the committed candidate. Expected: identical SHA-256.

- [ ] **Step 4: Write the bounded dataset audit**

  `docs/DATASET_V2.md` records pinned source, pool and exclusion counts, requested/realized sizes, bridge/comparison and level distributions, largest selected component, all three overlap counts, license, and the fact that public eval is frozen rather than blind.

- [ ] **Step 5: Verify historical namespaces are unchanged**

  Run:

  ```powershell
  git diff --exit-code codex/flagship-validity-design -- data/manifests/split_manifest.json results/raw results/derived
  pytest tests/test_grouping.py tests/test_group_allocation.py tests/test_manifest_v2.py tests/test_data_prepare_v2.py -q
  ```

- [ ] **Step 6: Commit Task 5**

  ```powershell
  git add data/manifests/split_manifest_v2.json docs/DATASET_V2.md results/v2/raw
  git commit -m "data: add leakage-resistant HotpotQA manifest v2"
  ```

### Task 6: Dataset-v2 Final Verification

**Files:**
- Modify only files required by a reproduced verification failure, with a failing regression test first.

**Interfaces:**
- Produces a local verification record; does not push or publish.

- [ ] **Step 1: Run formatting, lint, types, tests, and coverage**

  ```powershell
  ruff format --check .
  ruff check .
  mypy src
  pytest -m "not gpu and not slow" -q --cov=rag_evidence --cov-report=term-missing
  ```

- [ ] **Step 2: Run the acceptance slice**

  Confirm from the committed manifest and an independent recomputation:

  - exact source revision is the approved 40-character SHA;
  - question/title/paragraph overlap counts are zero;
  - every split is a union of complete connected components;
  - every selected supporting title and sentence resolves;
  - requested and realized sizes plus type/level distributions are explicit;
  - repeated generation is byte-identical;
  - historical manifest/raw/derived namespaces match the base tree.

- [ ] **Step 3: Prepare the local handoff**

  Report exact test and coverage output, manifest SHA-256, realized sizes, exclusions, distribution drift, remaining Gate B3/B4 work, and the later two-annotator requirement. Do not push or publish.
