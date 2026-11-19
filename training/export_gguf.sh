#!/bin/bash
# Export a LoRA run: merge onto bf16 base -> GGUF Q8_0 -> stock mmproj alongside.
# Usage: export_gguf.sh [RUN_NAME]   (default v1)
# Text layers only were trained, so the stock 8B mmproj is reused unchanged.
set -euo pipefail
cd ~/data_normalizing
RUN="${1:-v1}"
OUT=models/qwen3-vl-8b-lora-$RUN
mkdir -p "$OUT"

echo "== stage 1: merge adapter onto bf16 base =="
training/.venv/bin/python training/merge_lora.py "$RUN"

echo "== stage 2: convert merged model to GGUF Q8_0 =="
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD/$OUT":/work ghcr.io/ggml-org/llama.cpp:full \
  --convert /work/merged-bf16 \
  --outfile "/work/Qwen3-VL-8B-lora-$RUN-Q8_0.gguf" --outtype q8_0
echo "gguf: $OUT/Qwen3-VL-8B-lora-$RUN-Q8_0.gguf"

echo "== stage 3: copy stock mmproj alongside =="
cp models/mmproj-Qwen3VL-8B-Instruct-F16.gguf "$OUT/"
echo "mmproj: $OUT/mmproj-Qwen3VL-8B-Instruct-F16.gguf"
ls -lh "$OUT"
