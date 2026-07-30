# Preregistration: controlled cross-encoder reranking extension

Status: **LOCKED BEFORE IMPLEMENTATION AND BEFORE ANY NEW LOCKED-EVAL RUN**

- Preregistration ID: `reranking-v1-2026-07-29`
- Locked on: 2026-07-29 (Asia/Taipei)
- Existing benchmark version: `0.1.0`
- Existing locked-eval results were already published by the original project and are
  treated only as baseline provenance. No `hybrid_rrf_rerank` dev or eval result existed
  when this design was locked.
- Reference motivation: the consolidation failure described in
  <https://blog.aihao.tw/2026/07/26/beyond-rag-llamaindex-workshop/>: a reranker can
  improve the head of a ranking while dropping a passage that is jointly necessary for
  a multi-hop answer.

## 1. Confirmatory question

Holding the dataset split, question, candidate passages, generator, decoding, attribution
methods, answer metrics, and seeds fixed, does adding one multilingual cross-encoder
reranking stage after hybrid RRF improve:

1. retrieval ranking and complete multi-hop evidence coverage;
2. answer EM/F1 and citation coverage;
3. evidence sufficiency/comprehensiveness and attribution agreement;
4. without an unacceptable latency/VRAM cost or a consolidation failure that removes a
   necessary passage?

This is an extension of the evidence-attribution study, not a replacement benchmark.
The original artifacts under `results/raw/{smoke,dev,eval}` remain immutable controls.

## 2. Experimental unit, split policy, and leakage controls

- Dataset: the committed HotpotQA distractor manifest, unchanged.
- Units: smoke 20, dev 60, locked eval 240.
- The eval split remains locked. Implementation, adapter tests, cache behavior, thresholds,
  and any configuration decision must be completed using synthetic tests, smoke, and dev.
- Candidate configuration is selected/frozen on dev. Locked eval is executed once, with
  resume allowed only under the existing scientific-config hash.
- No dev result may be called a formal eval result.
- All four arms use the identical query-specific candidate corpus and passage identities.

### Candidate-count constraint

This benchmark's committed HotpotQA distractor corpus contains exactly **10 candidate
passages per question**. Therefore a top-30/top-50 reranking ablation would either be
identical to top-10 or would silently change the candidate corpus. The confirmatory
experiment fixes `candidate_k=10`; top-30/top-50 are declared **structurally inapplicable**,
not omitted based on model performance. A future corpus-level extension may study
10/30/50 only if BM25, dense, hybrid RRF, and reranked hybrid are all rerun over the same
new corpus.

## 3. Arms and the one allowed intervention

Confirmatory arms:

1. `bm25`
2. `dense`
3. `hybrid_rrf`
4. `hybrid_rrf_rerank`

For arm 4 only:

1. load the already stored `hybrid_rrf` order;
2. take its first 10 candidates (the complete candidate set here);
3. score each `(question, title + "\n" + passage)` pair with the fixed cross-encoder;
4. sort by descending raw cross-encoder logit, breaking exact ties by the original hybrid
   rank;
5. pass the first **5** passages to generation and attribution.

The three control arms also pass their first 5 passages. Thus only the ranking function
changes. `final_context_k=5` is fixed a priori rather than selected after dev.

## 4. Fixed model and inference configuration

Adapter interface: an in-repo, replaceable reranker protocol. The initial adapter uses
Hugging Face `AutoTokenizer` + `AutoModelForSequenceClassification`; it does not import
code or runtime state from Longcare RAG or any other repository.

| field | locked value |
|---|---|
| adapter | `hf_sequence_classification` |
| model | `BAAI/bge-reranker-v2-m3` |
| model revision | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` |
| tokenizer | `BAAI/bge-reranker-v2-m3` |
| tokenizer revision | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` |
| max pair length | 512 tokens |
| truncation | tokenizer pair truncation, `truncation=True` |
| score | raw scalar classification logit; no sigmoid |
| batch size | 16 pairs |
| smoke device/dtype | CPU / float32 |
| dev and eval device/dtype | CUDA / float16 |
| seed | existing global seed 42; inference is eval/no-grad |
| candidate rendering | `title + "\n" + passage_text` |
| tie-break | original hybrid RRF rank |

The exact effective dtype, device, library versions, model/tokenizer revisions, load
latency, and peak allocator VRAM must be written to `run_meta.json`.

## 5. Cache, resume, and provenance

- Reranker scores use an append-only cache keyed by model ID/revision, tokenizer
  ID/revision, max length, query content, passage ID, and passage content hash.
- Every retrieval record stores per-sample hit/miss counts. Run-level totals and hit rate
  are derived, never typed into a table.
- Partial final JSONL lines remain crash-tolerant; mid-file corruption remains fatal.
- Resume is refused if any scientific field above changes.
- The SHA-256 of this preregistration file is captured in the reranker run manifest.
- Original raw and derived benchmark artifacts are not overwritten. Extension outputs go
  under `results/reranking/`.

## 6. Outcomes and estimands

All metrics are macro averages unless explicitly described otherwise.

### Retrieval

- Recall@K for K in {2, 5, 10}
- MRR
- nDCG@K for K in {2, 5, 10}; primary retrieval endpoint: nDCG@5
- evidence coverage@K: fraction of gold supporting passages retained (numerically the
  same estimand as Recall@K, kept under the evidence terminology for the downstream
  analysis)
- complete/necessary-passage coverage@K: fraction of questions for which **all** gold
  passages are retained; primary consolidation guardrail: complete coverage@5

### Generation

- answer EM and token F1
- citation precision, recall, and F1 among answered samples
- citation coverage over all successfully processed questions (abstentions/no citations
  contribute zero)
- complete citation coverage: fraction citing every gold passage
- abstention rate
- faithfulness is represented by the existing teacher-forced sufficiency and
  comprehensiveness endpoints; no new unvalidated NLI scalar is introduced

### Attribution

- sufficiency (lower is better)
- comprehensiveness (higher is better)
- agreement with supporting passages: F1@2, AUPRC, nDCG@10
- stability under the retrieval intervention: per-question top-2 attribution Jaccard
  between `hybrid_rrf` and `hybrid_rrf_rerank`, reported per method
- control separation: each real method minus the strongest registered control on F1@2
  and AUPRC

Mode A uses the official gold answer with the arm's retrieved top-5 context. Mode B uses
the arm's generated non-abstained answer and exact generation context. As in the original
benchmark, Mode-B agreement is evaluated only on that arm's correct-answer subset, while
ground-truth-free sufficiency/comprehensiveness use all attributed answers.

### Systems

- rerank model-load latency
- rerank scoring latency (mean/p50/p95)
- estimated retrieval end-to-end latency:
  `max(BM25, dense) + RRF fusion + rerank` (explicitly labeled an estimate because the
  first-stage runs are stored artifacts)
- measured generation latency and estimated retrieval-plus-generation latency
- rerank and downstream peak VRAM
- rerank pair throughput
- cache hits, misses, and hit rate
- incremental latency and VRAM versus `hybrid_rrf`

No monetary API fee is incurred by the local model. “Incremental cost” means measured
compute latency/VRAM, not a fabricated currency value.

## 7. Dev analysis and locked decision rule

Dev may be used only to verify that the fixed experiment runs and to report the fixed
configuration above. Because the corpus cardinality and final context size are already
fixed, dev does not choose among 10/30/50 or among context sizes.

The reranker is considered worth retaining as the default extension only if locked eval
meets both:

1. nDCG@5 improves by at least 0.02 **or** complete evidence coverage@5 improves by at
   least 0.02 over `hybrid_rrf`; and
2. no guardrail degrades beyond tolerance: answer F1 −0.02, citation coverage −0.02,
   complete evidence coverage@5 −0.01, or generated-answer attribution F1@2 −0.02.

System cost is considered acceptable when rerank p95 incremental latency is at most
250 ms on the declared eval hardware and peak reranker VRAM fits independently on the
24 GB target GPU. If quality passes but cost fails, the result is publishable but the
reranker is not recommended as the default.

These thresholds and metrics must not be changed after seeing locked-eval results.
Negative and mixed results remain in the report.

## 8. Required error analysis and research-question answers

Machine-generated per-sample artifacts must classify:

- retrieval improved / unchanged / regressed;
- answer EM/F1 improved / regressed;
- citation coverage improved / regressed;
- complete-evidence gained / lost at context K=5;
- attribution agreement improved while retrieval regressed, and the reverse;
- results grouped by HotpotQA `type` and `level`.

The final generated report must answer:

1. Did reranking improve retrieval?
2. Did that improvement transfer to answer quality?
3. Did citation/evidence attribution improve?
4. Were there retrieval-better/attribution-worse cases?
5. Was latency/VRAM cost worth it?
6. Which question types improved or regressed?
7. Did context consolidation drop a necessary multi-hop passage?

## 9. Scope exclusions

This extension does not add or change PDF parsing, agent loops, Graph RAG, UI, chunking,
the candidate corpus, the generator, generation prompt/decoding, remote repositories, or
git remotes.
