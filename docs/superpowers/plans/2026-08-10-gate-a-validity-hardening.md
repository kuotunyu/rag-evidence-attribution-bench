# Gate A Validity Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing benchmark report evidence agreement, causal-dependence
diagnostics, uncertainty, denominators, and experiment completeness without misleading
terminology or requiring a new real-model run.

**Architecture:** Keep `results/raw` immutable and make evaluation a CPU-only projection
into schema v2. Add focused query, statistics, schema, and construct-validation modules;
the existing evaluator composes them and the reporter consumes both schema v1 and v2.
New real-model controls are runnable now but remain `not_run` for historical artifacts.

**Tech Stack:** Python 3.11, Pydantic 2, NumPy-free deterministic bootstrap arithmetic,
pytest, Ruff, mypy, YAML, JSON/JSONL.

## Global Constraints

- Do not modify any file under `results/raw` or `results/derived`.
- Gate A writes regenerated scientific artifacts only under `results/v1/derived`.
- Do not use a paid or remote API, publish, push, tag, or reserve a DOI.
- Use 10,000 paired non-parametric bootstrap replicates and two-sided 95% percentile
  intervals.
- Pair by question ID and report every exclusion reason.
- `leave_one_out` versus `control_lexical` is the sole confirmatory method comparison.
- `embedding` and `citations` are secondary; optional adapters are exploratory.
- Existing `embedding` and `control_lexical` names mean `question_answer` for compatibility.
- Historical causal-dependence diagnostics remain unvalidated until real oracle and
  answer-string control artifacts exist.
- Every behavior change follows a witnessed red-green-refactor cycle.

---

### Task 1: Deterministic Paired Bootstrap Primitive

**Files:**
- Create: `src/rag_evidence/evaluation/statistics.py`
- Create: `tests/test_paired_statistics.py`

**Interfaces:**
- Consumes: per-question mappings whose values are `float | None`.
- Produces: `paired_bootstrap_difference(candidate, comparator, *, global_seed,
  seed_parts, resamples, confidence, favorable_direction, tolerance) -> dict[str, Any]`.
- The result contains `n_pairs`, `n_candidate`, `n_comparator`, `exclusions`,
  `mean_delta`, `ci_low`, `ci_high`, `ci_excludes_zero`, `favorable_direction`,
  `favorable`, `confidence`, `resamples`, and `seed_uint64`.

- [ ] **Step 1: Write failing alignment and exclusion tests**

```python
def test_paired_bootstrap_aligns_question_ids_and_counts_exclusions() -> None:
    result = paired_bootstrap_difference(
        {"q1": 0.8, "q2": None, "q3": 0.4},
        {"q1": 0.5, "q2": 0.2, "q4": 0.1},
        global_seed=42,
        seed_parts=("eval", "gold", "leave_one_out", "control_lexical", "f1_at_2"),
        resamples=1000,
        confidence=0.95,
        favorable_direction="higher",
        tolerance=0.0,
    )
    assert result["n_pairs"] == 1
    assert result["mean_delta"] == pytest.approx(0.3)
    assert result["exclusions"] == {
        "candidate_missing": 1,
        "comparator_missing": 1,
        "candidate_null": 1,
        "comparator_null": 0,
    }
```

- [ ] **Step 2: Run the new test and verify RED**

Run: `uv run pytest tests/test_paired_statistics.py -q`

Expected: collection fails because `rag_evidence.evaluation.statistics` does not exist.

- [ ] **Step 3: Implement ID alignment, SHA-256 seed derivation, percentile interpolation,
  and favorable-direction evaluation**

```python
Direction = Literal["higher", "lower"]

def paired_bootstrap_difference(
    candidate: Mapping[str, float | None],
    comparator: Mapping[str, float | None],
    *,
    global_seed: int,
    seed_parts: Sequence[str],
    resamples: int = 10_000,
    confidence: float = 0.95,
    favorable_direction: Direction,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """Bootstrap mean(candidate - comparator) over complete question-ID pairs."""
```

Validate `resamples > 0`, `0 < confidence < 1`, and a non-negative tolerance. Derive the
seed from `global_seed` plus every ordered `seed_parts` value using SHA-256. Return null
interval fields when `n_pairs == 0`; never invent a zero effect.

- [ ] **Step 4: Add determinism, constant-difference, lower-is-better, empty-pair, and
  invalid-argument tests**

Use a constant positive delta to assert an exact `[delta, delta]` interval and repeat the
same call twice to assert byte-for-byte identical dictionaries. Use the reversed mapping to
prove that `favorable_direction="lower"` interprets a negative interval correctly.

- [ ] **Step 5: Verify and commit Task 1**

Run:

```powershell
uv run pytest tests/test_paired_statistics.py -q
uv run ruff check src/rag_evidence/evaluation/statistics.py tests/test_paired_statistics.py
uv run mypy src/rag_evidence/evaluation/statistics.py
git add src/rag_evidence/evaluation/statistics.py tests/test_paired_statistics.py
git commit -m "feat: add paired bootstrap statistics"
```

### Task 2: Explicit Query Conventions and Construct Controls

**Files:**
- Create: `src/rag_evidence/attribution/query.py`
- Modify: `src/rag_evidence/attribution/embedding.py`
- Modify: `src/rag_evidence/attribution/controls.py`
- Modify: `src/rag_evidence/attribution/__init__.py`
- Modify: `src/rag_evidence/attribution/run.py`
- Modify: `tests/test_attribution_runner.py`
- Create: `tests/test_query_conventions.py`
- Create: `tests/test_construct_controls.py`

**Interfaces:**
- Produces: `QueryConvention = Literal["question", "answer", "question_answer"]` and
  `compose_query(question, answer, convention) -> str`.
- Registers `embedding_question`, `embedding_answer`, `embedding_question_answer`,
  `control_lexical_question`, `control_lexical_answer`,
  `control_lexical_question_answer`, `oracle_gold`, and `control_answer_string`.
- Keeps `embedding` and `control_lexical` as `question_answer` aliases.
- Every query-based result stores `metadata["query_text_convention"]` using the literal
  value, not a prose rendering.
- Runner metadata records the method class's `is_control` value; evaluation does not infer
  control status solely from a `control_` name prefix.

- [ ] **Step 1: Write failing query-composition tests**

```python
@pytest.mark.parametrize(
    ("convention", "expected"),
    [
        ("question", "Who built it?"),
        ("answer", "Nora Finch"),
        ("question_answer", "Who built it? Nora Finch"),
    ],
)
def test_compose_query_is_explicit(convention: QueryConvention, expected: str) -> None:
    assert compose_query(" Who built it? ", " Nora Finch ", convention) == expected
```

Also assert an unsupported convention raises `ValueError` and empty components do not add
extra whitespace.

- [ ] **Step 2: Run the query test and verify RED**

Run: `uv run pytest tests/test_query_conventions.py -q`

Expected: collection fails because the query module does not exist.

- [ ] **Step 3: Implement `compose_query` and refactor embedding/lexical methods through
  small convention-specific base classes**

```python
class _EmbeddingByQuery(AttributionMethod):
    convention: ClassVar[QueryConvention]

class EmbeddingQuestion(_EmbeddingByQuery):
    name = "embedding_question"
    convention = "question"
```

Define all three registered variants and make `EmbeddingAttribution` inherit the
`question_answer` behavior. Follow the same pattern for lexical overlap. Do not use the
retrieval-rank control as an embedding ablation.

- [ ] **Step 4: Verify query variants with a recording embedder and lexical passages**

Extend `tests/test_query_conventions.py` with an embedder that records the exact query text.
Assert alias equivalence in scores and distinct `query_text_convention` metadata.

- [ ] **Step 5: Write failing oracle and answer-string control tests**

```python
def test_oracle_gold_scores_only_annotated_support() -> None:
    result = OracleGoldAttribution().attribute("q", passages, "Nora", None, {})
    assert result.raw_scores == {"p0": 1.0, "p1": 1.0, "p2": 0.0}

def test_answer_string_control_ignores_surface_free_support() -> None:
    result = AnswerStringControl().attribute("q", passages, "Nora Finch", None, {})
    assert result.raw_scores == {"p0": 0.0, "p1": 0.0, "p2": 1.0}
```

The answer-string control compares contiguous normalized answer tokens against normalized
title-plus-text tokens. An answer normalized to an empty string scores every passage zero.

- [ ] **Step 6: Run construct-control tests and verify RED**

Run: `uv run pytest tests/test_construct_controls.py -q`

Expected: import failure for both unimplemented control classes.

- [ ] **Step 7: Implement and register both controls, then extend runner coverage**

Add both names to `METHODS_ALL_MODES` in `tests/test_attribution_runner.py`. Assert each
successful record has valid scores, faithfulness output under FakeLM, and method metadata
identifying `positive_control` or `lexical_negative_control`. Set `oracle_gold.is_control`
to true even though its protocol name intentionally lacks the `control_` prefix, then prove
the run metadata preserves that classification.

- [ ] **Step 8: Verify and commit Task 2**

Run:

```powershell
uv run pytest tests/test_query_conventions.py tests/test_construct_controls.py tests/test_attribution_runner.py -q
uv run ruff check src/rag_evidence/attribution tests/test_query_conventions.py tests/test_construct_controls.py tests/test_attribution_runner.py
uv run mypy src/rag_evidence/attribution
git add src/rag_evidence/attribution tests/test_query_conventions.py tests/test_construct_controls.py tests/test_attribution_runner.py
git commit -m "feat: make attribution query conventions explicit"
```

### Task 3: Attribution Summary Schema v2

**Files:**
- Create: `src/rag_evidence/evaluation/schema.py`
- Create: `tests/test_evaluation_schema.py`

**Interfaces:**
- Produces strict frozen Pydantic models `AgreementSummary`, `CausalDependenceSummary`,
  `ExecutionSummary`, and `AttributionSummaryV2`.
- Produces `upgrade_attribution_metrics(legacy, *, agreement_subset,
  causal_validation_status) -> dict[str, Any]`.
- Keeps all original flattened values under `legacy_v1` and exposes no top-level `auprc`.

- [ ] **Step 1: Write failing schema-shape and denominator tests**

```python
def test_upgrade_separates_estimands_and_preserves_legacy() -> None:
    upgraded = upgrade_attribution_metrics(
        LEGACY,
        agreement_subset="generated_exact_match",
        causal_validation_status="not_run",
    )
    assert upgraded["agreement"]["n"] == 87
    assert upgraded["agreement"]["mean_average_precision"] == 0.857
    assert upgraded["causal_dependence"]["n"] == 192
    assert upgraded["execution"]["n_success"] == 192
    assert upgraded["legacy_v1"] == LEGACY
    assert "auprc" not in upgraded
```

Add rejection cases for a missing `n_agreement`, missing `n_faithfulness`, agreement count
above success count, and causal-dependence count above success count.

- [ ] **Step 2: Run schema tests and verify RED**

Run: `uv run pytest tests/test_evaluation_schema.py -q`

Expected: collection fails because the schema module does not exist.

- [ ] **Step 3: Implement strict v2 models and the compatibility upgrader**

```python
class AttributionSummaryV2(_StrictModel):
    schema_version: Literal[2] = 2
    run: str
    method: str
    agreement: AgreementSummary
    causal_dependence: CausalDependenceSummary
    execution: ExecutionSummary
    legacy_v1: dict[str, Any]
```

Use `mean_average_precision` for the macro mean of per-question AP. Name the causal fields
`sufficiency_mean` and `comprehensiveness_mean`; include `validation_status` with values
`passed`, `failed`, or `not_run`.

- [ ] **Step 4: Verify serialization, round-trip validation, and forbidden extra keys**

Use `AttributionSummaryV2.model_validate(payload).model_dump(mode="json")` and assert a
second validation produces the same dictionary.

- [ ] **Step 5: Verify and commit Task 3**

Run:

```powershell
uv run pytest tests/test_evaluation_schema.py -q
uv run ruff check src/rag_evidence/evaluation/schema.py tests/test_evaluation_schema.py
uv run mypy src/rag_evidence/evaluation/schema.py
git add src/rag_evidence/evaluation/schema.py tests/test_evaluation_schema.py
git commit -m "feat: add attribution summary schema v2"
```

### Task 4: Evaluator Integration, Pairwise Comparisons, and Construct Status

**Files:**
- Create: `src/rag_evidence/evaluation/construct.py`
- Modify: `src/rag_evidence/evaluation/evaluate.py`
- Modify: `tests/test_subset_filtering.py`
- Create: `tests/test_construct_validation.py`
- Create: `tests/test_attribution_comparisons.py`

**Interfaces:**
- `evaluate_attribution` returns v2 method summaries plus reserved mode keys
  `paired_comparisons`, `causal_validation`, and the existing generated-subset metadata.
- `evaluate_construct_validation(per_sample_by_method, *, bootstrap_options) -> dict`
  returns `status`, `missing_methods`, and oracle-versus-negative-control comparisons.
- `SUMMARY_SCHEMA_VERSION` becomes `2`.

- [ ] **Step 1: Change subset tests to demand v2 estimand blocks, then verify RED**

```python
assert loo_generated["agreement"]["n"] == 1
assert loo_generated["causal_dependence"]["n"] == 2
assert loo_generated["execution"]["n_success"] == 2
assert loo_generated["legacy_v1"]["n_agreement"] == 1
```

Run: `uv run pytest tests/test_subset_filtering.py -q`

Expected: assertion failure because the evaluator still emits flattened schema v1.

- [ ] **Step 2: Integrate the v2 upgrader while retaining per-sample output**

Keep a `per_sample_by_method` mapping in memory keyed by run name and question ID. Build
the legacy dictionary exactly once. Use a two-pass mode evaluation: first collect every
legacy aggregate and per-sample map; then compute the mode-level construct status and paired
comparisons; finally pass each legacy aggregate through `upgrade_attribution_metrics` with
that shared construct status. Write only the v2 object to new derived outputs.

- [ ] **Step 3: Write failing paired-comparison tests over intentionally misaligned runs**

Create temporary attribution records where the candidate and comparator have different
missing question IDs. Assert:

```python
comparison = mode["paired_comparisons"]["leave_one_out__vs__control_lexical"]
assert comparison["f1_at_2"]["n_pairs"] == 2
assert comparison["f1_at_2"]["exclusions"]["candidate_missing"] == 1
assert comparison["mean_average_precision"]["favorable_direction"] == "higher"
assert comparison["sufficiency"]["favorable_direction"] == "lower"
```

- [ ] **Step 4: Implement comparison generation**

Generate the primary comparison whenever both methods exist. Generate secondary method
versus available-control comparisons with `analysis_tier="secondary"`; optional adapters
use `analysis_tier="exploratory"`. Seed parts are exactly split, mode, candidate, comparator,
and metric. Agreement uses only rows in the agreement subset; causal diagnostics use every
row with that metric.

- [ ] **Step 5: Write failing construct-validation status tests**

Assert `not_run` with explicit missing methods for historical-style inputs. For constant
synthetic maps, assert `passed` only when oracle has lower sufficiency and higher
comprehensiveness than both `control_random` and `control_answer_string`, with every paired
interval excluding zero. Reverse one metric and assert `failed`.

- [ ] **Step 6: Implement construct validation without converting missing runs to failure**

`not_run` means at least one required method is absent. `failed` means all required methods
exist but any interval is null, crosses zero, or has the wrong direction. `passed` means all
four oracle-control metric comparisons pass.

- [ ] **Step 7: Verify and commit Task 4**

Run:

```powershell
uv run pytest tests/test_subset_filtering.py tests/test_attribution_comparisons.py tests/test_construct_validation.py -q
uv run ruff check src/rag_evidence/evaluation tests/test_subset_filtering.py tests/test_attribution_comparisons.py tests/test_construct_validation.py
uv run mypy src/rag_evidence/evaluation
git add src/rag_evidence/evaluation tests/test_subset_filtering.py tests/test_attribution_comparisons.py tests/test_construct_validation.py
git commit -m "feat: evaluate attribution estimands with paired uncertainty"
```

### Task 5: Truthful v1/v2 Reporting

**Files:**
- Modify: `src/rag_evidence/reporting/report.py`
- Modify: `tests/test_e2e_tiny.py`
- Create: `tests/test_reporting_v2.py`

**Interfaces:**
- `_attribution_tables` accepts v1 historical summaries and v2 summaries.
- V2 columns are `F1@k`, `MAP`, `nDCG@10`, `sufficiency diagnostic`,
  `comprehensiveness diagnostic`, `n agreement`, `n causal`, execution timing, and failure
  rate.
- A paired-comparison table prints mean delta, interval, pair count, exclusions, direction,
  and analysis tier.
- A causal-validation line is always one of `PASSED`, `FAILED`, or `NOT RUN`; only `PASSED`
  permits the word `causal` in an interpretive claim.

- [ ] **Step 1: Write failing v2 rendering tests**

Use a compact in-memory v2 summary and assert the block contains `MAP`, both denominators,
`95% CI`, `n_pairs`, and `Causal-dependence validation: NOT RUN`. Assert it does not contain
`AUPRC` or the phrase `faithfulness improvement`.

- [ ] **Step 2: Run the reporting test and verify RED**

Run: `uv run pytest tests/test_reporting_v2.py -q`

Expected: current renderer raises on missing flattened fields or emits AUPRC and one `n`.

- [ ] **Step 3: Add a small schema adapter and render v2 tables**

Keep historical v1 rendering readable by adapting it in memory and labeling its causal
status `NOT RUN`; do not rewrite historical JSON. Render intervals as
`mean [ci_low, ci_high]` and `—` when no complete pair exists.

- [ ] **Step 4: Update end-to-end assertions for schema v2 and mock gating**

Retain all existing guarantees that mock numbers never enter README results. Add an
assertion that generated-mode agreement and causal counts differ in the scripted fixture.

- [ ] **Step 5: Verify and commit Task 5**

Run:

```powershell
uv run pytest tests/test_reporting_v2.py tests/test_e2e_tiny.py -q
uv run ruff check src/rag_evidence/reporting/report.py tests/test_reporting_v2.py tests/test_e2e_tiny.py
uv run mypy src/rag_evidence/reporting/report.py
git add src/rag_evidence/reporting/report.py tests/test_reporting_v2.py tests/test_e2e_tiny.py
git commit -m "feat: report attribution validity without denominator mixing"
```

### Task 6: Versioned Gate A Configurations and Experiment Matrix

**Files:**
- Modify: `src/rag_evidence/config.py`
- Create: `src/rag_evidence/evaluation/protocol.py`
- Create: `configs/v1/smoke.yaml`
- Create: `configs/v1/dev.yaml`
- Create: `configs/v1/eval.yaml`
- Modify: `tests/test_config.py`
- Create: `tests/test_experiment_matrix.py`

**Interfaces:**
- `EvaluationConfig` gains locked defaults: `bootstrap_resamples=10000`,
  `bootstrap_confidence=0.95`, `bootstrap_tolerance=0.0`,
  `confirmatory_method="leave_one_out"`, and
  `primary_comparator="control_lexical"`.
- `build_experiment_matrix(discovered_methods) -> dict` labels every declared row
  `complete`, `not_run`, `unsupported`, or `experimental_unvalidated`.
- V1 configs read `results/raw` and write `results/v1/derived`; they never point at
  `results/derived`.

- [ ] **Step 1: Write failing locked-config tests**

Assert the three v1 configs validate, use the versioned output path, and contain the exact
bootstrap settings and confirmatory names. Assert invalid confidence and fewer than 1,000
resamples are rejected.

- [ ] **Step 2: Run config tests and verify RED**

Run: `uv run pytest tests/test_config.py -q`

Expected: missing v1 config files and evaluation fields.

- [ ] **Step 3: Implement config fields and add v1 YAML files**

Copy every scientific value from its matching historical config. Change only the derived
output path and the newly explicit Gate A evaluation fields.

- [ ] **Step 4: Write failing experiment-matrix tests**

Assert absent query variants are `not_run`, citations in gold mode are `unsupported`,
ContextCite is `experimental_unvalidated`, and a discovered required row is `complete`.

- [ ] **Step 5: Implement the declared matrix and attach it to each split summary**

The required historical rows are `leave_one_out`, `embedding`, `control_random`,
`control_retrieval`, `control_lexical`, `control_length`, and `control_shuffled` in both
modes, plus `citations` in generated mode. Query variants, `oracle_gold`, and
`control_answer_string` are declared `not_run` until artifacts exist.

- [ ] **Step 6: Verify and commit Task 6**

Run:

```powershell
uv run pytest tests/test_config.py tests/test_experiment_matrix.py -q
uv run ruff check src/rag_evidence/config.py src/rag_evidence/evaluation/protocol.py tests/test_config.py tests/test_experiment_matrix.py
uv run mypy src/rag_evidence/config.py src/rag_evidence/evaluation/protocol.py
git add src/rag_evidence/config.py src/rag_evidence/evaluation/protocol.py configs/v1 tests/test_config.py tests/test_experiment_matrix.py
git commit -m "feat: declare the Gate A experiment matrix"
```

### Task 7: README Claim Corrections and Offline Regeneration

**Files:**
- Modify: `README.md`
- Modify: `README_en.md`
- Modify: `src/rag_evidence/evaluation/evaluate.py`
- Create: `tests/test_readme_claims.py`
- Create: `tests/test_regeneration.py`
- Generate: `results/v1/derived/**`

**Interfaces:**
- README describes v0.1 as closed-candidate context attribution over all ten HotpotQA
  distractor passages in dataset order.
- Integrity language is `traceable and reproducible`, never tamper-proof.
- No hand-maintained test count appears.
- Regeneration starts from committed raw artifacts and writes only `results/v1/derived`.

- [ ] **Step 1: Write failing README claim tests**

```python
def test_readmes_bound_historical_claims(repo_root: Path) -> None:
    for name in ("README.md", "README_en.md"):
        text = (repo_root / name).read_text(encoding="utf-8")
        assert "closed-candidate" in text
        assert "all ten" in text or "全部十篇" in text
        assert not re.search(r"\b\d+ tests\b", text)
        assert "tamper" not in text.lower()
        assert "防篡改" not in text
```

- [ ] **Step 2: Run README tests and verify RED**

Run: `uv run pytest tests/test_readme_claims.py -q`

Expected: assertions fail on the current broad claims, test count, or tamper-proof wording.

- [ ] **Step 3: Correct the manual README prose**

Separate evidence agreement from causal-dependence diagnostics, state that v0.1 causal
construct validation was not run, and preserve the visible historical results as a labeled
legacy block until regenerated v1 outputs replace it.

- [ ] **Step 4: Write a failing deterministic regeneration test**

In a temporary tiny environment, run evaluation and reporting twice. Assert identical
summary JSON, per-sample JSONL, report Markdown, and README generated blocks. Assert the
historical `results/derived` path is never created.

- [ ] **Step 5: Run regeneration test and verify RED, then make the minimum timestamp or
  ordering changes required for deterministic comparison**

Run: `uv run pytest tests/test_regeneration.py -q`

The first failure must identify nondeterministic content or an incorrect output namespace,
not a fixture error. Replace wall-clock `generated_utc` with the maximum `finished_utc`
among the raw run metadata consumed by the current summary. Name that field
`source_snapshot_utc`; it is provenance from immutable inputs, not the report execution
time. This makes repeated raw-to-derived generation byte-stable.

- [ ] **Step 6: Regenerate Gate A outputs from immutable historical raw artifacts**

Run:

```powershell
uv run python -m rag_evidence.cli evaluate --config configs/v1/smoke.yaml
uv run python -m rag_evidence.cli evaluate --config configs/v1/dev.yaml
uv run python -m rag_evidence.cli evaluate --config configs/v1/eval.yaml
uv run python -m rag_evidence.cli report --config configs/v1/eval.yaml
git diff --exit-code -- results/raw results/derived
```

Confirm `results/v1/derived/summary.json` has schema version 2, historical cells have causal
validation `not_run`, paired primary comparisons have 10,000 resamples, and Mode B reports
distinct agreement and causal denominators.

- [ ] **Step 7: Verify and commit Task 7**

Run:

```powershell
uv run pytest tests/test_readme_claims.py tests/test_regeneration.py -q
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -m "not gpu and not slow" -q
git diff --check
git add README.md README_en.md tests/test_readme_claims.py tests/test_regeneration.py results/v1/derived
git commit -m "docs: publish bounded Gate A benchmark claims"
```

### Task 8: Gate A Final Verification

**Files:**
- Modify only files required by failures reproduced in this task, with a regression test
  written before each production correction.

**Interfaces:**
- Produces a local verification record in the task handoff; it does not publish or push.

- [ ] **Step 1: Run formatting, lint, types, tests, and coverage**

```powershell
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -m "not gpu and not slow" -q --cov=rag_evidence --cov-report=term-missing
```

- [ ] **Step 2: Verify immutable namespaces and deterministic v1 regeneration**

```powershell
git diff --exit-code codex/flagship-validity-design -- results/raw results/derived
uv run python -m rag_evidence.cli evaluate --config configs/v1/smoke.yaml
uv run python -m rag_evidence.cli evaluate --config configs/v1/dev.yaml
uv run python -m rag_evidence.cli evaluate --config configs/v1/eval.yaml
uv run python -m rag_evidence.cli report --config configs/v1/eval.yaml
git diff --exit-code
```

- [ ] **Step 3: Audit the acceptance slice**

Verify from generated files that:

- summary schema is 2;
- no v2 attribution object exposes top-level `auprc`;
- agreement, causal-dependence, and execution counts are separate;
- every available primary comparison reports 10,000 paired replicates, a 95% interval,
  pair count, and exclusions;
- missing query/control runs are visible as `not_run`;
- historical causal diagnostics are `not_run`, not `passed`;
- README says closed-candidate and makes no unsupported faithfulness claim;
- `results/raw` and `results/derived` match the design-branch tree exactly.

- [ ] **Step 4: Commit any verification-only artifact changes and prepare the local handoff**

If Step 2 leaves no diff, do not create an empty commit. Report exact command outputs,
coverage, commit hashes, remaining Gate B/C dependencies, and the human annotation work
that has not started.
