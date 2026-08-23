# v0.2 Human Annotation Pilot

This directory is an annotation-readiness handoff, not a completed study. It contains two
blind assignment packages covering 20 Dataset-v2 smoke parents × two challenge variants =
40 tasks. Every task is assigned to both independent annotators. No answerability label,
adjudication, eligibility decision, or model output is included.

The task text derives from HotpotQA and retains its CC BY-SA 4.0 data obligations; project
code remains MIT licensed. Read the repository `DATA_CARD.md` and `PILOT_PROTOCOL.md`.

## Coordinator workflow

1. Give `packages/ann-pilot-a.json` and `packages/ann-pilot-b.json` to two different humans.
   Do not give repository access or `manifest.json` unless operationally necessary.
2. Each human chooses a private local state directory and runs:

   `rag-evidence annotation serve --package <their-json> --state <private-state-dir>`

3. Keep annotator pseudonyms unchanged. Do not record names or email addresses.
4. Collect exported submissions and amendments separately from this clean package folder.
5. Run two-annotator completeness, then send only disagreements to a third human.
6. Review IAA, evidence overlap, timing, defects, and instruction feedback against the
   promotion gate in `PILOT_PROTOCOL.md`.

Re-run `python scripts/scan_annotation_export.py pilot/v0.2/packages` before delivery.
