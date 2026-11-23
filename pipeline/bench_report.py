"""Render BENCHMARKS.md from data/scores.jsonl — the campaign's central
engine-score registry.

Aggregation rule (spec 2026-11-14-1b-bench-design): per (engine, ntrs_id, page)
the latest row by ts wins; only gt_status=="done" pages count; the norm column
shows the single norm_hash behind an engine's aggregate or "mixed" when its rows
span normalization eras (cross-era numbers are not comparable). All ts values are
ISO-8601 Z strings, so lexical max is chronological max.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

# Decodes every engine name that has ever appeared in data/scores.jsonl.
# Sources: naca-engine-decision notes, training/runs/v1/RESULTS.md, serve scripts.
ENGINE_LEGEND = {
    "baseline": "NTRS-provided 1990s PDF text layer (no new OCR)",
    "tesseract": "Tesseract 5 word-box OCR (campaign raw-OCR layer)",
    "vision": "Qwen3-VL-8B-Instruct, llama.cpp :8780, prompt v1 (six-engine trial)",
    "vision2": "Qwen3.8-27B, vLLM NVFP4 on 5090, prompt v1 (six-engine trial winner)",
    "vision3": "olmOCR-2 (trial; disqualified: silently abbreviates long tables)",
    "vision4": "chandra-2 (trial; loops/describes figure pages)",
    "vision_haiku": "Claude Haiku via API, prompt v1 (trial)",
    "vision_haiku2": "Claude Haiku via API, prompt v2 (trial)",
    "vision_or": "OpenRouter-hosted trial probe (8 pages only)",
    "vision_27bq4": "Qwen3.8-27B UD-Q4_K_M, llama.cpp :8781, thinking off",
    "vision_27b_v2": "27B escalation lane: Q4_K_M + prompt v2, hash 6a169a212aac (GT-gated)",
    "vision_235b": "Qwen3-VL-235B-A22B via OpenRouter, trial-era prompt/guard v1",
    "vision_8bstock": "Stock Qwen3-VL-8B sanity probe (6 pages)",
    "vision_8bvllm": "Qwen3-VL-8B via vLLM runtime comparison probe (6 pages)",
    "vision_8blora": "8B LoRA v1 (691 labels), Q8_0 llama.cpp :8782",
    "vision_8blora_lc25": "8B LoRA learning-curve run, 25% of training data",
    "vision_8blora_lc50": "8B LoRA learning-curve run, 50% of training data",
    "vision_8blora_c1024": "8B LoRA v1 at 1024px-capped input (resolution ablation)",
    "vision_8blora_c2048": "8B LoRA v1 at 2048px-capped input (resolution ablation)",
    "vision_8blora_v2": "8B LoRA v2 (grown label pool)",
    "vision_8blora_v2_1": "8B LoRA v2_1 — campaign engine, hash 6ecf7d0dbeb2",
    "vision_8bwin": "Campaign 8B via native Windows app (llama.cpp win-cuda, volunteer "
                    "profile) — packaging acceptance",
    "vision_glmocr": "GLM-OCR 0.9B (zai-org, MIT), official ggml-org GGUF Q8_0, llama.cpp",
    "vision_lighton08": "LightOnOCR-3-0.8B (Apache-2.0), transcription mode",
    "vision_paddlevl16": "PaddleOCR-VL-1.6 0.9B (Apache-2.0), official GGUF; vendor "
                         "notes llama.cpp path 'may have degraded performance'",
}


def aggregate(scores_path) -> list[dict]:
    latest = {}  # (engine, ntrs_id, page) -> (ts, norm_hash, metrics)
    with open(scores_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("gt_status") != "done":
                continue
            for eng, m in r["engines"].items():
                key = (eng, r["ntrs_id"], r["page"])
                if key not in latest or r["ts"] > latest[key][0]:
                    latest[key] = (r["ts"], r["norm_hash"], m)
    per_engine = {}
    for (eng, _, _), (ts, norm, m) in latest.items():
        per_engine.setdefault(eng, []).append((ts, norm, m))
    rows = []
    for eng, items in per_engine.items():
        norms = {norm for _, norm, _ in items}
        trs = [m["term_recall"] for _, _, m in items if m.get("term_recall") is not None]
        rows.append({
            "engine": eng,
            "n": len(items),
            "median_cer": round(statistics.median(m["cer"] for _, _, m in items), 4),
            "median_wer": round(statistics.median(m["wer"] for _, _, m in items), 4),
            "median_term_recall": round(statistics.median(trs), 4) if trs else None,
            "norm": norms.pop() if len(norms) == 1 else "mixed",
            "latest_ts": max(ts for ts, _, _ in items),
        })
    rows.sort(key=lambda r: r["median_cer"])
    return rows


def render(rows: list[dict]) -> str:
    out = ["| engine | n | median CER | median WER | term recall | norm | latest run | what it is |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        tr = "—" if r["median_term_recall"] is None else f'{r["median_term_recall"]:.3f}'
        legend = ENGINE_LEGEND.get(r["engine"], "(undocumented)")
        out.append(f'| {r["engine"]} | {r["n"]} | {r["median_cer"]:.4f} | '
                   f'{r["median_wer"]:.4f} | {tr} | {r["norm"]} | '
                   f'{r["latest_ts"][:10]} | {legend} |')
    return "\n".join(out)


def render_table_section(summary_path) -> str:
    p = Path(summary_path)
    if not p.exists():
        return "_Table second-reader bench: no data yet._"
    out = ["| reader pair | pages | cells compared | disagreement |",
           "|---|---|---|---|"]
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            rate = "—" if r["disagree_rate"] is None else f'{r["disagree_rate"] * 100:.1f}%'
            out.append(f'| {r["pair"]} | {r["pages"]} | {r["cells_compared"]} | {rate} |')
    return "\n".join(out)


def write_section(doc_path, marker: str, body: str) -> None:
    doc = Path(doc_path)
    text = doc.read_text()
    start, end = f"<!-- {marker}:start -->", f"<!-- {marker}:end -->"
    i, j = text.index(start), text.index(end)
    doc.write_text(text[: i + len(start)] + "\n" + body + "\n" + text[j:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="refresh leaderboard section")
    ap.add_argument("--table-write", action="store_true", help="refresh table-bench section")
    args = ap.parse_args()
    doc = common.ROOT / "BENCHMARKS.md"
    rows = aggregate(common.DATA / "scores.jsonl")
    if args.write:
        write_section(doc, "bench", render(rows))
        print(f"wrote leaderboard: {len(rows)} engines")
    if args.table_write:
        write_section(doc, "tablebench",
                      render_table_section(common.DATA / "bench" / "table_diff_summary.jsonl"))
        print("wrote table-bench section")
    if not (args.write or args.table_write):
        print(render(rows))


if __name__ == "__main__":
    main()
