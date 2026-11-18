#!/bin/bash
# Pass 4a model server: Qwen3-VL-8B-Instruct Q8_0 via llama.cpp CUDA, port 8780.
# GPU budget: ~10GB of the 3090, leaving room for the existing tenant (~3.4GB).
M=~/data_normalizing/models
exec docker run --rm --name naca-vlm --runtime nvidia --gpus all \
  -p 127.0.0.1:8780:8080 \
  -v "$M":/models:ro \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m /models/Qwen3VL-8B-Instruct-Q8_0.gguf \
  --mmproj /models/mmproj-Qwen3VL-8B-Instruct-F16.gguf \
  -ngl 99 -c 20480 --temp 0 --host 0.0.0.0 --port 8080
