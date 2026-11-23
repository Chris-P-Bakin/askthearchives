#!/bin/bash
# Distilled 8B LoRA engine: Qwen3-VL-8B lora-v2_1 Q8_0 via llama.cpp CUDA, port 8782.
# v2_1 promoted 2026-11-12: first student to meet the <=0.10 cascade bar
# (GT median CER 0.0932 vs v1's 0.1188; see training/runs/v1/RESULTS.md).
# 3090 budget: ~8.5G weights + 0.9G mmproj + KV/compute; ~3.4G tenant stays.
# Fits alongside nothing else big - stop naca-vlm-27b first if it is running.
# Detached with restart policy: survives session death and daemon/host restarts.
# -c 32768 -np 2 = 16k per stream x 2 slots (pilot's single-stream envelope, preserved); KV ~4.7 GB fits only while naca-vlm-27b is stopped.
M=~/data_normalizing/models/qwen3-vl-8b-lora-v2_1
exec docker run -d --restart unless-stopped --name naca-vlm-8blora --runtime nvidia --gpus all \
  -p 127.0.0.1:8782:8080 \
  -v "$M":/models:ro \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m /models/Qwen3-VL-8B-lora-v2_1-Q8_0.gguf \
  --mmproj /models/mmproj-Qwen3VL-8B-Instruct-F16.gguf \
  -ngl 99 -c 32768 -np 2 --temp 0 --host 0.0.0.0 --port 8080
