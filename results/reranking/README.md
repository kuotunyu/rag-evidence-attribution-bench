# Reranking extension artifacts

This directory is isolated from the original benchmark results under `results/raw` and
`results/derived`.

- `raw/`: resumable stage outputs for the controlled reranking extension.
- `derived/`: machine-generated comparisons, error analysis, reports, and tables.
- `raw/cache/reranker/`: local append-only score cache; intentionally gitignored.

The currently committed extension run is a CPU retrieval smoke diagnostic only. It is
not a dev result or formal locked eval result. Missing downstream metrics remain null and
must not be filled by hand.

Future generation and attribution records can contain HotpotQA question and passage text.
HotpotQA is distributed under CC BY-SA 4.0; derived code and project-authored metadata
remain under this repository's MIT license. See `DATA_CARD.md` and
`PREREGISTRATION_RERANKING.md` for provenance, limitations, and the locked protocol.
Secondary paired-bootstrap, transfer-taxonomy, subgroup, and systems-decomposition
outputs follow the separately locked `SECONDARY_ANALYSIS_RERANKING.md`.
