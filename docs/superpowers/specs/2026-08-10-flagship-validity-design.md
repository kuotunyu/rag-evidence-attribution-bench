# Flagship Validity Promotion Design

**Status:** approved in principle on 2026-08-10; written review gate pending

**Target release:** `1.0.0` only after every acceptance gate below passes

**Current release:** preserved as the immutable `0.1.0` historical baseline

## 1. Objective

Promote the repository from a reproducible HotpotQA/Qwen3 case study into a defensible
evidence-attribution benchmark without expanding its product surface. The promotion must
separate supporting-evidence agreement from causal faithfulness, quantify uncertainty,
remove avoidable split leakage, validate citation granularity and evaluator behavior,
demonstrate limited cross-model and cross-dataset transfer, and produce a legally coherent
Zenodo release.

The program is complete only when a fresh clone can reproduce every derived number and
report from pinned data, model, configuration, and raw artifacts without a paid API.

## 2. Constraints

- No paid or remote LLM-as-judge API.
- No new UI, agent loop, vector database, Graph RAG, PDF ingestion, or leaderboard.
- Do not overwrite or reinterpret artifacts under the existing `results/raw` and
  `results/derived` namespaces. They remain the `0.1.0` historical baseline.
- New scientific results live under a versioned `results/v1/` namespace.
- No publication or push is performed without a separate explicit instruction.
- Every model and dataset revision is a full 40-character commit hash.
- Every behavior change follows red-green-refactor TDD.
- Expensive GPU runs start only after the CPU pipeline, schemas, statistics, and release
  checks pass on synthetic fixtures and smoke data.

## 3. Scientific Claims and Terminology

The benchmark will make three distinct claims and will not collapse them into a single
"faithfulness" score.

1. **Evidence agreement:** whether a passage or sentence ranking agrees with annotated
   supporting evidence. Primary metrics are F1@2, mean per-question average precision
   (`MAP`), and nDCG. The current field named `auprc` is retained only as a deprecated
   schema-v1 compatibility alias; reports use `MAP`.
2. **Causal model dependence:** whether keeping or removing the attributed evidence changes
   the evaluated generator's teacher-forced target log-probability. Primary metrics are
   sufficiency and comprehensiveness with their own sample count.
3. **Citation correctness:** whether answer citations resolve to the annotated supporting
   sentence or passage. Passage-level and sentence-level results are reported separately.

No result may use the words "causal" or "faithfulness" when it is based only on lexical
overlap or agreement with HotpotQA supporting-fact labels.

## 4. Program Decomposition

The work is split into four independently testable gates. A later gate consumes artifacts
from earlier gates but may not silently alter them.

### Gate A: Metric, statistics, and report hardening

Gate A requires no new model run and re-derives improved reports from committed raw data.

#### A1. Summary schema v2

Each attribution entry is structured as:

```json
{
  "agreement": {
    "n": 87,
    "f1_at": {"2": 0.793},
    "mean_average_precision": 0.857,
    "ndcg_at_10": 0.921,
    "subset": "generated_exact_match"
  },
  "faithfulness": {
    "n": 192,
    "sufficiency_mean": 0.860,
    "comprehensiveness_mean": 7.156,
    "subset": "all_successful_non_abstained"
  },
  "execution": {
    "n_attempted": 192,
    "n_success": 192,
    "n_failed": 0,
    "seconds_per_sample": 0.650
  }
}
```

The schema preserves the historical flattened values in a `legacy_v1` object so consumers
can audit equivalence. A schema validator rejects a shared or missing denominator.

#### A2. Paired uncertainty

- Experimental unit: question ID.
- Comparison direction: candidate method minus comparator on aligned question IDs.
- Resampling: deterministic paired non-parametric bootstrap.
- Replicates: 10,000.
- Interval: two-sided 95% percentile interval.
- Seed: SHA-256 derivation from global seed, split, mode, method, comparator, and metric.
- Missingness: complete pairs only; every output reports `n_pairs` and exclusion counts.
- Primary comparator: `control_lexical` fixed before eval inspection.
- Secondary comparators: every registered control plus `leave_one_out` and `embedding`.
- Confirmatory attribution method: `leave_one_out`. Its agreement and causal-dependence
  comparisons are the only method-level promotion tests.
- `citations` and `embedding` are prespecified secondary analyses. `arc_jsd`, `contextcite`,
  and any newly registered adapter remain exploratory until a protocol amendment is
  committed before the relevant new-manifest eval artifacts are opened.
- No post-hoc "strongest eval control" selection is used for a confirmatory claim.
- P-values are not required. If later added, they are secondary and Holm-adjusted within
  each declared metric family.

Reports show paired intervals next to point estimates. Reranking retains its preregistered
point-threshold decision rule, but prose distinguishes retrieval improvement from
unresolved downstream transfer when a paired interval crosses zero.

#### A3. Causal-metric construct checks

Sufficiency and comprehensiveness are promoted from diagnostic outputs to causal-dependence
metrics only after passing fixed intervention checks at the same `k` and target sequence:

- **positive control:** keep/remove the annotated supporting evidence (`oracle_gold`);
- **uninformative negative control:** keep/remove seeded random distractors
  (`control_random`);
- **lexical negative control:** keep/remove answer-bearing non-supporting passages
  (`control_answer_string`);
- **synthetic counterfactual:** paired fixtures differ only in the evidence span that
  determines the target answer; swapping or deleting that span must reverse or attenuate
  target log-probability in the specified direction.

The evaluator passes a model/dataset cell only when the paired 95% interval for
`oracle_gold` versus each negative control excludes zero in the expected direction:
lower sufficiency loss and higher comprehensiveness. Synthetic counterfactual fixtures must
pass exactly. Failed construct checks do not invalidate evidence-agreement metrics, but
they prohibit describing that cell's sufficiency/comprehensiveness results as causal or
faithfulness evidence.

#### A4. Query-source ablations

Embedding and lexical scoring use the same three query conventions:

- `question`: question only;
- `answer`: target answer only;
- `question_answer`: question plus target answer.

The current `embedding` and `control_lexical` names remain aliases for their
`question_answer` variants. Metadata stores the exact query convention. A BM25 retrieval
rank is not described as a question-only embedding ablation.

#### A5. Reporting truthfulness

- "Complete" means that a declared experiment matrix is present, not merely that discovered
  runs finished.
- Absent optional methods appear as `not_run`, `unsupported`, or `experimental_unvalidated`.
- Mode B tables show `n_agreement` and `n_faithfulness` as separate columns.
- The README describes the historical experiment as closed-candidate context attribution:
  original generation used all ten dataset passages in dataset order.
- Integrity wording is "traceable and reproducible", never "impossible to tamper with".
- Test badges are generated from CI output or omitted; no hand-maintained pass count.

### Gate B: Dataset v1 and evaluator validation

#### B1. Pinned sources

Initial revisions recorded by this design are:

| resource | pinned revision |
|---|---|
| `hotpotqa/hotpot_qa` | `1908d6afbbead072334abe2965f91bd2709910ab` |
| `Qwen/Qwen3-4B-Instruct-2507` | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| `Qwen/Qwen3-Embedding-0.6B` | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |
| `BAAI/bge-reranker-v2-m3` | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` |

The pipeline records the requested revision and the resolved local snapshot commit. A
mismatch is fatal.

#### B2. Group-disjoint manifest v2

Questions are nodes in an undirected graph. Two questions are connected when any context
passage shares either a normalized title or an exact canonical paragraph fingerprint.
Connected components are indivisible split groups. A seeded deterministic allocator draws
eval groups first, then dev, then smoke, while minimizing deviation from the requested
240/60/20 sizes and preserving the available bridge/comparison distribution.

Manifest generation fails unless all of these invariants hold:

- no question ID overlap;
- no normalized title overlap across splits;
- no exact canonical paragraph overlap across splits;
- every selected example retains all supporting-fact passages and sentences;
- realized split sizes and type distributions are recorded;
- all source and per-example hashes verify.

If exact 240/60/20 sizes cannot satisfy group isolation, the allocator records the nearest
deterministic sizes and the preregistration uses those realized sizes. It never breaks a
group to preserve a cosmetic sample count.

The released eval set is described as a frozen public test set, not a permanently blind
leaderboard set.

#### B3. Answerability challenge set

A separate `challenge` namespace tests abstention and lexical robustness without changing
the natural HotpotQA aggregate.

For each selected source example, deterministic transformations produce:

1. **missing-hop:** remove one necessary supporting passage and replace it with a distractor;
2. **answer-bearing distractor:** insert a non-supporting passage containing the normalized
   answer string;
3. **evidence swap:** replace a supporting sentence with a contradictory or entity-swapped
   sentence while preserving surface form.

Every transformed sample stores its parent question ID, transformation, changed fields,
expected answerability, and content hash. Missing-hop and evidence-swap examples enter the
confirmatory challenge set only after two-person human validation with adjudication.

#### B4. Citation granularity and human audit

Prompt version `v2` renders sentence aliases such as `[P3:S2]` while retaining passage
aliases for backward compatibility. The parser resolves both formats. Reports provide:

- passage citation precision/recall/F1;
- sentence citation precision/recall/F1;
- complete evidence-chain citation rate;
- invalid citation rate;
- answerability-aware abstention precision/recall on the challenge set.

A stratified 100-example audit contains 70 bridge and 30 comparison examples, with at least
40 answer-bearing-distractor cases. Two annotators independently label answerability,
minimal sufficient evidence, citation correctness, and whether the dataset evidence is
exhaustive. Disagreements receive third-person adjudication. Cohen's kappa is reported for
binary fields and Krippendorff's alpha for multi-label evidence sets. No automatic metric
is promoted as validated when agreement is below 0.70.

### Gate C: Cross-model and cross-dataset confirmation

#### C1. Second generator family

The non-Qwen confirmation model is `microsoft/Phi-4-mini-instruct` pinned to
`cfbefacb99257ffa30c83adab238a50856ac3083` and distributed under MIT. It uses the same
semantic prompt contract and deterministic decoding, with a model-specific chat-template
adapter. Remote code is disabled unless the pinned model cannot run with the installed
Transformers implementation; enabling it requires a separate source audit and is recorded
as a protocol amendment before any eval result is viewed.

Primary cross-model claim: the direction of each method-minus-lexical paired effect for
F1@2/MAP and LOO sufficiency/comprehensiveness is reported for Qwen and Phi separately.
Pooling across models is secondary and never hides a sign reversal.

#### C2. Second dataset

The cross-dataset confirmation source is `Alab-NII/2wikimultihop` pinned to
`13800e5be57df1b4040b9b1588c6c811779e69e9` and distributed under Apache-2.0. A dedicated
adapter maps question, answer, context, supporting passages, and supporting sentences into
the existing canonical schema. Dataset-native identifiers remain in provenance metadata.

The confirmation subset is built with the same group-disjoint rule and contains 320
examples unless license/source validation or group structure makes that impossible; any
smaller realized count is preregistered before model execution. HotpotQA and 2Wiki results
are reported separately.

#### C3. Promotion criterion

The confirmatory effect is `leave_one_out - control_lexical`, paired by question. For
F1@2 and MAP, positive values favor `leave_one_out`. For sufficiency, negative values
favor `leave_one_out`; for comprehensiveness, positive values favor `leave_one_out`.

The benchmark may claim limited generalization or superior causal attribution only when:

- the paired 95% interval for the confirmatory F1@2 and MAP effects excludes zero in the
  favorable direction for each generator/dataset cell;
- every causal-metric construct check in A3 passes, and the paired 95% interval for the
  confirmatory sufficiency and comprehensiveness effects excludes zero in the favorable
  direction for each generator/dataset cell;
- on the answer-bearing-distractor challenge aggregate, both agreement effects retain the
  favorable sign and their paired intervals and pair counts are reported;
- failures and sign reversals remain visible in the report.

The criterion is directional and uncertainty-aware; it does not introduce an arbitrary
post-hoc minimum effect size. Failure of this criterion is a publishable benchmark result,
but the release must state that cross-model/cross-dataset superiority was not demonstrated.

### Gate D: Engineering taxonomy, CI, and release

#### D1. Engineering-action failure taxonomy

Every unsuccessful or paradoxical sample receives one primary category:

| category | owning component | default action |
|---|---|---|
| `retrieval_miss` | retriever | improve candidate recall or corpus indexing |
| `reranker_consolidation_loss` | reranker | diversity/coverage-aware reranking |
| `generator_ignored_available_evidence` | generator/prompt | prompt or decoding investigation |
| `parametric_answer_without_context_dependence` | generator/evaluator | separate answer correctness from context use |
| `citation_missing_or_wrong_granularity` | citation generation/parser | sentence alias or citation instruction fix |
| `lexical_distractor_capture` | attribution method | query convention or robustness fix |
| `gold_evidence_non_exhaustive_or_ambiguous` | dataset/evaluator | adjudicate or exclude from confirmatory set |
| `runtime_or_artifact_failure` | pipeline | retry, provenance, or environment fix |

The classifier is deterministic where artifact fields suffice and emits
`human_review_required` otherwise. At least 95% of audited failures must have an
adjudicated actionable category before release.

#### D2. CI release gates

CI runs offline and fails on any of the following:

- formatting, lint, mypy, unit, schema, or Docker health failure;
- total test coverage below 80%;
- `metrics`, `evaluation`, or `reporting` package coverage below 90%;
- raw-to-derived regeneration changes committed derived JSON/JSONL or generated README
  blocks;
- manifest, file checksum, model revision, or license inventory mismatch;
- a declared required experiment row is missing or partial;
- release metadata validation failure.

GPU reproducibility remains a separately recorded workflow because ordinary CI is offline
and CPU-only.

#### D3. Zenodo packaging

The GitHub repository gains validated `CITATION.cff`, `.zenodo.json`, `CHANGELOG.md`, and a
machine-readable `THIRD_PARTY_LICENSES.json`. Because Zenodo assigns one record-level
license while this repository contains MIT code and CC BY-SA HotpotQA derivatives, release
artifacts are split:

1. **software record:** source, tests, configs, schemas, documentation; MIT;
2. **benchmark-data record:** manifests, licensed samples, raw/derived result artifacts,
   annotations, and checksums; CC BY-SA 4.0 with upstream attributions.

The two records use reciprocal related identifiers. No model weights are redistributed.
The release checklist reserves but does not publish DOIs; publication remains a human gate.

## 5. Data and Artifact Flow

```text
pinned upstream data
  -> canonical records
  -> group-disjoint manifest v2
  -> versioned prepared data + challenge transformations
  -> retrieval/generation/attribution raw runs under results/v1
  -> schema-v2 evaluation + paired statistics
  -> engineering taxonomy + human-audit joins
  -> generated reports/README blocks
  -> checksum inventory
  -> separate software and benchmark-data release bundles
```

Raw artifacts are append-only. Derived artifacts may be deleted and regenerated. Report
generation never edits scientific values manually.

## 6. Error Handling

- Source revision drift, manifest overlap, schema mismatch, or missing denominators fail
  closed before model loading.
- Partial runs remain resumable but cannot satisfy a release matrix.
- Missing paired values are excluded pairwise and counted by reason.
- Human-review rows remain visible and block only the affected confirmatory challenge
  claim, not unrelated pipeline checks.
- An unavailable second model or dataset is a blocked Gate C, not permission to publish a
  flagship claim from Gate A/B alone.
- Release tooling creates local bundles and validation reports only; it has no publish or
  push command.

## 7. Testing Strategy

Every production change starts with a failing test. Required layers are:

1. unit tests for schema v2, paired bootstrap, query conventions, overlap grouping,
   citation aliases, taxonomy, revision validation, and license inventory;
2. property tests for split disjointness, bootstrap determinism, paired alignment, and
   report denominator consistency;
3. synthetic end-to-end fixtures covering correct, wrong, abstained, missing-hop,
   answer-bearing distractor, partial, and corrupt artifacts;
4. golden regeneration tests proving committed raw data recreates committed derived data
   and README blocks;
5. smoke runs before any full GPU matrix;
6. fresh-clone release-bundle verification with network-disabled evaluation/reporting.

## 8. Acceptance Gates

The `1.0.0` tag is permitted only when all conditions hold:

- `MAP`/AUPRC terminology and all estimands are correct.
- Mode B exposes distinct agreement and faithfulness denominators.
- Every primary comparison has a paired 95% interval and explicit pair count.
- Causal-dependence metrics pass the prespecified oracle, negative-control, and synthetic
  counterfactual construct checks, or affected cells are labeled unvalidated and make no
  faithfulness claim.
- Manifest v2 has zero cross-split normalized-title and paragraph overlap.
- All upstream revisions are pinned and verified.
- Query-source ablations, answerability challenges, and sentence citations are complete.
- The 100-example human audit meets the declared protocol and reports agreement.
- Both generator families and both datasets have complete, separately reported matrices.
- At least 95% of audited failures have an actionable adjudicated category.
- CI coverage, regeneration, checksum, schema, and release gates pass in a fresh clone.
- Software and benchmark-data bundles have coherent separate licenses and validated
  citation/Zenodo metadata.
- No result prose claims downstream faithfulness improvement when its paired interval does
  not support that direction.

## 9. Deliberate Non-Goals

- A hosted leaderboard or secret test server.
- More attribution algorithms before the validity gates pass.
- More rerankers, vector stores, chunkers, or generation UIs.
- Paid LLM judges or proprietary APIs.
- A universal RAG-quality scalar.
- Retrofitting the historical `0.1.0` raw records to look as though they were generated by
  the new protocol.

## 10. Execution Order

Implementation plans are produced and executed separately in this order:

1. Gate A: metrics/statistics/reporting and CI regeneration;
2. Gate B1-B2: pins and group-disjoint dataset construction;
3. Gate B3-B4: challenge set, sentence citations, and annotation tooling;
4. Gate D1-D2: taxonomy and release-grade CI;
5. Gate C: Phi and 2Wiki adapters, smoke, then full runs;
6. Gate D3: local release bundles and metadata validation;
7. human review of reports and explicit publication decision.

This order makes the cheap validity checks capable of stopping expensive model runs.
