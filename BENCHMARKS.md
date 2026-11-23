# NACA OCR engine benchmarks

Central registry of every engine scored against the project's ground-truth pages.
Generated from `data/scores.jsonl` by `pipeline/bench_report.py --write` — edit the
prose, never the tables.

**How to read it.** Scores are character/word error rate (lower is better) and
domain-term recall (higher is better) against hand-verified ground truth on real
NACA report scans, medians over the pages each engine was run on. The latest
scoring run put every engine on the same frozen 30-page GT set
(`data/gt_pages.jsonl`) under one normalization config, so the table is directly
comparable top to bottom; a smaller `n` means the engine was only ever run on a
subset (the 2-page vLLM/stock probes, the 1-page OpenRouter probe, the 22-page
235B trial). The `norm` column is the text-normalization config hash the scores
were computed under; **rows marked `mixed`, or rows with different norm hashes,
are not directly comparable** — normalization evolved during the project. The
campaign cascade bar is median CER ≤ 0.10; the 27B escalation lane's GT-gated
0.0975 is the production escalation config's reference score.

<!-- bench:start -->
| engine | n | median CER | median WER | term recall | norm | latest run | what it is |
|---|---|---|---|---|---|---|---|
| vision_27bq4 | 30 | 0.0835 | 0.1798 | 1.000 | 09600ef89b80 | 2026-11-16 | Qwen3.8-27B UD-Q4_K_M, llama.cpp :8781, thinking off |
| vision_8blora_v2_1 | 30 | 0.0940 | 0.1959 | 1.000 | 09600ef89b80 | 2026-11-16 | 8B LoRA v2_1 — campaign engine, hash 6ecf7d0dbeb2 |
| vision_27b_v2 | 30 | 0.0975 | 0.2172 | 1.000 | 09600ef89b80 | 2026-11-16 | 27B escalation lane: Q4_K_M + prompt v2, hash 6a169a212aac (GT-gated) |
| vision2 | 30 | 0.1002 | 0.1769 | 1.000 | 09600ef89b80 | 2026-11-16 | Qwen3.8-27B, vLLM NVFP4 on 5090, prompt v1 (six-engine trial winner) |
| vision_8blora | 30 | 0.1188 | 0.1993 | 1.000 | 09600ef89b80 | 2026-11-16 | 8B LoRA v1 (691 labels), Q8_0 llama.cpp :8782 |
| vision_8bwin | 30 | 0.1227 | 0.2147 | 1.000 | 09600ef89b80 | 2026-11-16 | Campaign 8B via native Windows app (llama.cpp win-cuda, volunteer profile) — packaging acceptance |
| vision_8blora_v2 | 30 | 0.1273 | 0.1959 | 1.000 | 09600ef89b80 | 2026-11-16 | 8B LoRA v2 (grown label pool) |
| vision_haiku | 30 | 0.1302 | 0.2140 | 1.000 | 09600ef89b80 | 2026-11-16 | Claude Haiku via API, prompt v1 (trial) |
| vision_haiku2 | 30 | 0.1389 | 0.2205 | 1.000 | 09600ef89b80 | 2026-11-16 | Claude Haiku via API, prompt v2 (trial) |
| vision_8blora_c2048 | 30 | 0.1517 | 0.2814 | 1.000 | 09600ef89b80 | 2026-11-16 | 8B LoRA v1 at 2048px-capped input (resolution ablation) |
| vision_8blora_lc50 | 30 | 0.1603 | 0.3120 | 1.000 | 09600ef89b80 | 2026-11-16 | 8B LoRA learning-curve run, 50% of training data |
| vision | 30 | 0.1663 | 0.3314 | 1.000 | 09600ef89b80 | 2026-11-16 | Qwen3-VL-8B-Instruct, llama.cpp :8780, prompt v1 (six-engine trial) |
| vision_lighton08 | 30 | 0.2029 | 0.3947 | 0.909 | 09600ef89b80 | 2026-11-16 | LightOnOCR-3-0.8B (Apache-2.0), transcription mode |
| vision_8blora_lc25 | 30 | 0.2127 | 0.3285 | 1.000 | 09600ef89b80 | 2026-11-16 | 8B LoRA learning-curve run, 25% of training data |
| vision_8blora_c1024 | 30 | 0.2195 | 0.4631 | 1.000 | 09600ef89b80 | 2026-11-16 | 8B LoRA v1 at 1024px-capped input (resolution ablation) |
| vision_235b | 22 | 0.2241 | 0.3110 | 1.000 | 09600ef89b80 | 2026-11-16 | Qwen3-VL-235B-A22B via OpenRouter, trial-era prompt/guard v1 |
| vision4 | 30 | 0.2652 | 0.5884 | 1.000 | 09600ef89b80 | 2026-11-16 | chandra-2 (trial; loops/describes figure pages) |
| vision_glmocr | 30 | 0.3279 | 0.4814 | 0.778 | 09600ef89b80 | 2026-11-16 | GLM-OCR 0.9B (zai-org, MIT), official ggml-org GGUF Q8_0, llama.cpp |
| vision_8bvllm | 2 | 0.3637 | 0.4827 | 1.000 | 09600ef89b80 | 2026-11-16 | Qwen3-VL-8B via vLLM runtime comparison probe (6 pages) |
| vision3 | 30 | 0.3672 | 0.6545 | 0.800 | 09600ef89b80 | 2026-11-16 | olmOCR-2 (trial; disqualified: silently abbreviates long tables) |
| vision_8bstock | 2 | 0.3976 | 0.6273 | 1.000 | 09600ef89b80 | 2026-11-16 | Stock Qwen3-VL-8B sanity probe (6 pages) |
| tesseract | 30 | 0.4113 | 0.8161 | 0.815 | 09600ef89b80 | 2026-11-16 | Tesseract 5 word-box OCR (campaign raw-OCR layer) |
| vision_or | 1 | 0.4489 | 0.5489 | 0.800 | 09600ef89b80 | 2026-11-16 | OpenRouter-hosted trial probe (8 pages only) |
| vision_paddlevl16 | 30 | 0.5170 | 0.6799 | 0.806 | 09600ef89b80 | 2026-11-16 | PaddleOCR-VL-1.6 0.9B (Apache-2.0), official GGUF; vendor notes llama.cpp path 'may have degraded performance' |
| baseline | 30 | 0.6959 | 0.9004 | 0.400 | 09600ef89b80 | 2026-11-16 | NTRS-provided 1990s PDF text layer (no new OCR) |
<!-- bench:end -->

## The 1B-specialist bench (2026-11-15)

Three sub-1B dedicated OCR models were benched as potential cheap cascade
tiers, all served with llama.cpp (official GGUF where the vendor publishes one),
greedy decoding, 16k context, pass4a's loop guard active:

- **GLM-OCR 0.9B** — median CER 0.3279 (mean 0.3638), 30/30 clean stops at 16k
  context (at 8k, 3 dense pages truncated at the context limit and were rerun).
  ~2.5 GB VRAM and the cleanest llama.cpp story of the three, but 3× the bar.
- **LightOnOCR-3-0.8B** — median CER 0.2029 (mean 0.2983), best of the three but
  2× the bar. 27/30 clean stops + 3 loop-guard retries. Thinking mode must be
  disabled (it emits zero output text otherwise). Caveat: the official repo
  ships no GGUF, so this ran a community conversion (prithivMLmods Q8_0 +
  bf16 mmproj) — an untested quantization path.
- **PaddleOCR-VL-1.6 0.9B** — median CER 0.517 (mean 1.92), eliminated. 26/30
  clean stops, 2 loop-guard retries, and two hallucination-loop pages at CER
  13.9 and 29.8. The vendor's own docs warn the llama.cpp path "may have
  degraded performance", and PaddleOCR's headline numbers come from its full
  pipeline (layout model + post-processing), not the bare VLM benched here.

**Verdict:** no 1B specialist clears the 0.10 transcription bar — the campaign's
fine-tuned 8B (0.0940) and the 27B lanes stay unchallenged, consistent with the
olmOCR Old-Scans finding that degraded historical scans are where small
specialists fall apart.

## Table second-reader bench

Numeric-table transcription is the one place CER is blind (one wrong digit in a
dense table barely moves it). Candidate second readers re-transcribe flagged
table pages independently; cells are parsed under NACA print conventions and
diffed positionally. Disagreement rate = differing cell pairs / compared pairs.

<!-- tablebench:start -->
| reader pair | pages | cells compared | disagreement |
|---|---|---|---|
| vision_glmocr vs vision_8blora_v2_1 | 58 | 3314 | 15.5% |
| vision_lighton08 vs vision_8blora_v2_1 | 58 | 9192 | 15.8% |
| vision_glmocr vs vision_235b | 10 | 440 | 8.4% |
| vision_lighton08 vs vision_235b | 10 | 1074 | 6.5% |
| vision_8blora_v2_1 vs vision_235b | 10 | 1924 | 3.7% |
<!-- tablebench:end -->

**Reading it.** The reference pair — the campaign 8B against the 235B pilot, the
two strongest readers — disagrees on 3.7% of numeric cells. A useful second
reader should land near that floor; instead the 1B candidates disagree with the
235B at 6.5% (LightOnOCR) and 8.4% (GLM-OCR), roughly double the strong-pair
baseline, and at ~15.5% against the campaign 8B over all 58 flagged pages.
GLM-OCR also yields far fewer comparable cells (3,314 vs LightOnOCR's 9,192
over the same pages) — not because it skips the tables, but because it often
emits them in formats the cell parser doesn't read (colon-separated rows, LaTeX
arrays, one value per line); 32 of its 58 pages parse to zero cells, so its
rates are measured only on the pages it renders as pipe/HTML tables. On this
run 8 of LightOnOCR's 58 pages and 3 of
GLM-OCR's ended in loop-guard retries or empty output. **Verdict:** neither 1B
model is a credible independent second reader for the table-consistency lane;
cross-checking stays with the 8B/27B/235B tiers.
