# results/

Provenance rule: files under `raw/` are only ever created by a real pipeline execution on
this machine or imported from a real execution via `python -m rag_evidence.cli import-results`.
Nothing here is hand-written. Runs with `execution_kind: "mock"` (test backend) are never
rendered into the README.

Licensing: records embed HotpotQA question/passage text → these files are distributed under
CC BY-SA 4.0 with attribution (Yang et al., 2018), unlike the MIT-licensed code. See DATA_CARD.md.
