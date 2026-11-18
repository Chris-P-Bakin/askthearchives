#!/bin/bash
# Spot-GPU worker bootstrap (vast.ai and similar container hosts).
#
# Launch the instance FROM the campaign's pinned llama.cpp CUDA image so the
# engine binary is byte-identical to the local workers:
#   ghcr.io/ggml-org/llama.cpp:server-cuda@sha256:4b554057c7cf0cb87e0724b6de9d7ca9bb24a3b72716af11e4c0fecad62c506b
# Requirements: 24 GB GPU (3090-class), ~32 GB disk, env BOINC_AUTHENTICATOR.
# This script replicates worker/Dockerfile on top of that image, pulls the
# model from the public data host, and runs the stock BOINC worker entry.
set -euo pipefail

REPO="${ARCHIVES_REPO:-https://github.com/Chris-P-Bakin/askthearchives}"
DATA_HOST="${ARCHIVES_DATA_HOST:-https://data.askthearchives.org}"
export BOINC_PROJECT_URL="${BOINC_PROJECT_URL:-https://boinc.askthearchives.org/naca/}"
: "${BOINC_AUTHENTICATOR:?set BOINC_AUTHENTICATOR in the instance environment}"

apt-get update
apt-get install -y --no-install-recommends python3 curl git wget boinc-client ca-certificates

# restart-idempotent: vast re-runs onstart on every container start
rm -rf /opt/archives
git clone --depth 1 "$REPO" /opt/archives
mkdir -p /worker
cp /opt/archives/worker/unit_worker.py /opt/archives/worker/entry.sh /worker/
cp /opt/archives/pipeline/guard.py /worker/guard.py
cp /opt/archives/pipeline/prompts/transcribe_v2.txt /worker/transcribe_v2.txt
cp -r /opt/archives/worker/boinc /worker/boinc
chmod +x /worker/entry.sh /worker/boinc/naca_vision

mkdir -p /models
cd /models
wget -c -q "$DATA_HOST/naca/models/Qwen3-VL-8B-lora-v2_1-Q8_0.gguf"
wget -c -q "$DATA_HOST/naca/models/mmproj-Qwen3VL-8B-Instruct-F16.gguf"
sha256sum -c - <<'SUMS'
2154df260d8154bec8ac84ca3118d555b0490d60492a4f2f05f1a4d21790bd45  Qwen3-VL-8B-lora-v2_1-Q8_0.gguf
ca524100ebf825c9a870db1c580d03879e0da0ab2541697e2458e64891cf9d38  mmproj-Qwen3VL-8B-Instruct-F16.gguf
SUMS

export NACA_VLM_URL=http://127.0.0.1:8080
# vast launches onstart outside the image entrypoint context; the llama.cpp
# binary needs its bundled libs on the path explicitly.
export LD_LIBRARY_PATH="/app:${LD_LIBRARY_PATH:-}"
exec /worker/entry.sh boinc
