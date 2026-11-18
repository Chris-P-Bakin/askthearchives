#!/bin/bash
# Pass 4a PRIMARY engine (the-gpu-box stand-in): Qwen3.8-27B UD-Q4_K_M via llama.cpp CUDA, port 8781.
# 3090 budget: ~16.5G weights + 0.9G mmproj + KV/compute in the rest; ~3.4G tenant stays.
# Stop naca-vlm (8B) first - both do not fit together.
# Detached with restart policy: survives session death and daemon/host restarts.
M=~/data_normalizing/models/qwen3.8-27b-gguf
exec docker run -d --restart unless-stopped --name naca-vlm-27b --runtime nvidia --gpus all \
  -p 127.0.0.1:8781:8080 \
  -v "$M":/models:ro \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m /models/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --mmproj /models/mmproj-F16.gguf \
  -ngl 99 -c 16384 --temp 0 --host 0.0.0.0 --port 8080
