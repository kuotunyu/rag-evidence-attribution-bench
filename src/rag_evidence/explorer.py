"""Gradio explorer over the same in-process ResultsStore as the API.

Default experience is precomputed: no model download, no GPU, ever.
"""

from __future__ import annotations

import logging
from typing import Any

from rag_evidence.store import ResultsStore

logger = logging.getLogger(__name__)


def _sample_label(store: ResultsStore, qid: str) -> str:
    q = store.samples[qid]["question"]
    return f"{qid[:8]}… — {q[:80]}"


def _passages_markdown(sample: dict[str, Any]) -> str:
    gold_sent_ids = set(sample["supporting_fact_sentence_ids"])
    lines = []
    for p in sample["passages"]:
        marker = "🟩" if p["is_gold"] else "▫️"
        lines.append(f"**{marker} `{p['passage_id'][-3:]}` {p['title']}**")
        for j, sent in enumerate(p["sentences"]):
            sid = f"{p['passage_id']}-s{j:02d}"
            if sid in gold_sent_ids:
                lines.append(f"> **{sent}** ⭐")
            else:
                lines.append(f"> {sent}")
        lines.append("")
    return "\n".join(lines)


def _generation_markdown(store: ResultsStore, qid: str) -> str:
    if not store.generation:
        return "_No generation run present yet (pending Colab execution)._"
    parts = []
    for name, run in sorted(store.generation.items()):
        rec = run.get(qid)
        if rec is None:
            continue
        if rec.get("error") is not None:
            parts.append(f"**{name}**: ❌ failed ({rec['error']['type']})")
            continue
        cites = " ".join(f"`[{a}]`" for a in rec["citations_raw"]) or "_none_"
        parts.append(
            f"**{name}** — EM {rec['em']}, F1 {rec['f1']:.2f}, "
            f"{'ABSTAINED' if rec['abstained'] else 'answered'}\n\n"
            f"> {rec['response_text']}\n\n"
            f"citations: {cites} | latency {rec['latency_ms']:.0f} ms | "
            f"peak VRAM {rec['peak_vram_mb'] or 'n/a (CPU)'}"
        )
    return "\n\n---\n\n".join(parts) or "_No record for this sample._"


def _attribution_rows(store: ResultsStore, qid: str, mode: str) -> list[list[Any]]:
    rows: list[list[Any]] = []
    methods = store.attribution.get(mode, {})
    sample = store.samples[qid]
    gold = set(sample["gold_passage_ids"])
    for method, run in sorted(methods.items()):
        rec = run.get(qid)
        if rec is None or rec.get("skipped") or rec.get("error") is not None:
            continue
        ranked = sorted(rec["raw_scores"].items(), key=lambda kv: -kv[1])
        top2 = [pid for pid, _ in ranked[:2]]
        hit = len(set(top2) & gold)
        faith = rec.get("faithfulness") or {}
        rows.append(
            [
                method,
                ", ".join(p[-3:] for p in top2),
                f"{hit}/2",
                round(faith.get("sufficiency"), 3) if faith else None,
                round(faith.get("comprehensiveness"), 3) if faith else None,
                rec.get("latency_s"),
                rec.get("peak_vram_mb") or "n/a",
            ]
        )
    return rows


def build_demo(store: ResultsStore) -> Any:
    import gradio as gr  # lazy heavy import

    ids = store.list_ids()
    choices = [(_sample_label(store, qid), qid) for qid in ids]

    with gr.Blocks(title="rag-evidence-attribution-bench") as demo:
        gr.Markdown(
            f"# RAG evidence attribution — precomputed explorer\n"
            f"Split **{store.split}** · {len(ids)} samples · "
            "🟩 = gold supporting passage, ⭐ = supporting-fact sentence"
        )
        with gr.Tab("Sample browser"):
            dropdown = gr.Dropdown(choices=choices, value=ids[0] if ids else None, label="Sample")
            question_md = gr.Markdown()
            answer_md = gr.Markdown()
            passages_md = gr.Markdown()
            generation_md = gr.Markdown()

            def show_sample(qid: str):
                if not qid:
                    return "", "", "", ""
                s = store.samples[qid]
                return (
                    f"### {s['question']}",
                    f"**Gold answer:** {s['answer']}  \ntype: {s['qtype']} · level: {s['level']}",
                    _passages_markdown(s),
                    _generation_markdown(store, qid),
                )

            dropdown.change(
                show_sample,
                inputs=dropdown,
                outputs=[question_md, answer_md, passages_md, generation_md],
            )
            if ids:
                question_md.value, answer_md.value, passages_md.value, generation_md.value = (
                    show_sample(ids[0])
                )

        with gr.Tab("Attribution comparison"):
            dd2 = gr.Dropdown(choices=choices, value=ids[0] if ids else None, label="Sample")
            mode_radio = gr.Radio(
                choices=sorted(store.attribution) or ["gold"],
                value=(sorted(store.attribution)[0] if store.attribution else "gold"),
                label="Mode (gold = teacher-forced A, generated = mode B)",
            )
            table = gr.Dataframe(
                headers=[
                    "method",
                    "top-2 passages",
                    "gold hits",
                    "sufficiency ↓",
                    "comprehensiveness ↑",
                    "s/sample",
                    "peak VRAM MB",
                ],
                interactive=False,
            )

            def show_attr(qid: str, mode: str):
                return _attribution_rows(store, qid, mode) if qid else []

            dd2.change(show_attr, inputs=[dd2, mode_radio], outputs=table)
            mode_radio.change(show_attr, inputs=[dd2, mode_radio], outputs=table)

        with gr.Tab("Benchmark summary"):
            if store.summary and store.summary.get("splits"):
                snapshot = store.summary.get("source_snapshot_utc") or store.summary.get(
                    "generated_utc"
                )
                gr.Markdown(
                    f"Source snapshot "
                    f"{snapshot}"
                    " · dataset "
                    f"`sha256:{str(store.summary.get('dataset_hash'))[:12]}…` — "
                    "see README for the full tables."
                )
                gr.JSON(value=store.summary)
            else:
                gr.Markdown("**RESULTS PENDING** — no summary.json yet.")

    return demo
