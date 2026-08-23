# v0.2 Human Annotation Pilot Sources

This directory is the source-controlled, annotation-ready engineering contract, not a
delivery directory and not a completed study. The canonical blind package sources cover
20 Dataset-v2 smoke parents × two challenge variants = 40 tasks. Every task is assigned to
both independent annotators. No answerability label, adjudication, eligibility decision,
IAA result, synthetic decision, or model output is included.

Packages from baseline `372096b` are obsolete. The canonical sources in `packages/` are
regenerated in place against instruction identity `pilot-v0.2.2-draft` and the SHA-256 of
the complete `PILOT_PROTOCOL.md` file.

The task text derives from HotpotQA and retains its CC BY-SA 4.0 data obligations; project
code remains MIT licensed. Read the repository `DATA_CARD.md` and `PILOT_PROTOCOL.md`.

The committed `handoff-manifest.json` and `handoff-manifest.schema.json` bind builder/spec
versions, Python 3.11 support, the dependency lock, canonical source hashes, instruction
identity, expected Git-external layout, and the reproducible recipe. They intentionally do
not contain a predicted wheel hash. See `HANDOFF_RUNBOOK.md` for exact commands.

The private Python distribution identity is `0.2.0.dev0`. It is distinct from protocol identity
`pilot-v0.2.2-draft` and schema version `v2`; none of these creates a Git tag, release, or public
package. A future final `0.2.0` requires separate owner approval and newly built exact-source
evidence. A `dev0` wheel receipt cannot be reused for that final identity.

## B0.1 stop boundary

This source state does not authorize the human pilot. Owner approval is still required
before independent humans receive external kits. It also does not authorize protocol
freeze, confirmatory sampling, model/API execution, merge, tag, release, GitHub Release
assets, or Hugging Face publication.

The current execution boundary is `HUMAN_PILOT_NOT_STARTED`. Metadata blinding covers explicit
coordinator metadata, transformation identity, and expected-answerability labels. It does not
establish semantic unlinkability and does not establish independent sibling perception because
both annotators receive both non-adjacent variants. This pilot can produce tooling and instruction
feasibility evidence only.

## Future owner-approved coordinator workflow

1. Build Git-external, checksum-verified A and B kits from a final clean checkout. Each kit
   contains the same verified wheel but only its own package and launcher. Neither kit
   contains the coordinator-only `assignment-manifest-v2.json` or the other annotator's
   package. Rebuild that private manifest outside the repository with the exact canonical A/B
   package bytes before collection.
2. Each annotator uses a private state directory and returns `submissions.jsonl`, a present
   `amendments.jsonl` (which may be empty), and return checksums.
3. Collect both streams:

   `rag-evidence annotation collect --manifest coordinator/manifest.json --submission returns/a/submissions.jsonl --submission returns/b/submissions.jsonl --amendment returns/a/amendments.jsonl --amendment returns/b/amendments.jsonl --out coordinator/collection`

4. If complete, start third-human adjudication:

   `rag-evidence annotation adjudicate --manifest coordinator/manifest.json --effective coordinator/collection/effective-submissions.jsonl --state coordinator/private-adjudication-state --host 127.0.0.1 --port 8002`

5. Export `http://127.0.0.1:8002/api/export/adjudications.jsonl` and finalize:

   `rag-evidence annotation finalize-pilot --manifest coordinator/manifest.json --originals coordinator/collection/original-submissions.jsonl --amendments coordinator/collection/amendments.jsonl --adjudications coordinator/adjudications.jsonl --protocol PILOT_PROTOCOL.md --out coordinator/final`

6. Stop unless all 40 pairs are complete, all disagreements are adjudicated, privacy and
   integrity pass, and both Cohen's kappa and nominal Krippendorff's alpha are finite and
   at least 0.70. `READY_FOR_HUMAN_FREEZE_REVIEW` still requires owner review.
