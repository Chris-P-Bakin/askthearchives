#!/bin/bash
# Container entry: start one llama-server per GPU (same args as
# pipeline/serve_vlm_8blora.sh), wait for /health on each, then run the worker.
#   standalone  -> python3 unit_worker.py --standalone /work/units.jsonl --data-root /work/data [args]
#   boinc       -> BOINC client, anonymous platform (needs BOINC_PROJECT_URL + BOINC_AUTHENTICATOR)
# NGPUS overrides GPU detection (default: nvidia-smi -L count, min 1).
set -euo pipefail
MODE="${1:-standalone}"; shift || true

N="${NGPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
[ "$N" -ge 1 ] 2>/dev/null || N=1
PIDS=()
URLS=""
for i in $(seq 0 $((N - 1))); do
  # Only pin a card when there are several: overriding CUDA_VISIBLE_DEVICES
  # on single-GPU vast containers broke CUDA init (CPU fallback).
  if [ "$N" -gt 1 ]; then CVD="CUDA_VISIBLE_DEVICES=$i"; else CVD=""; fi
  env $CVD /app/llama-server \
    -m /models/Qwen3-VL-8B-lora-v2_1-Q8_0.gguf \
    --mmproj /models/mmproj-Qwen3VL-8B-Instruct-F16.gguf \
    -ngl 99 -c 32768 -np 2 --temp 0 --host 127.0.0.1 --port $((8080 + i)) \
    > >(tee /tmp/llama_$i.log) 2> >(tee -a /tmp/llama_$i.log >&2) &
  PIDS+=($!)
  URLS="${URLS}${URLS:+,}http://127.0.0.1:$((8080 + i))"
done
trap 'kill "${PIDS[@]}" 2>/dev/null || true' EXIT
for i in $(seq 0 $((N - 1))); do
  P=$((8080 + i))
  for _ in $(seq 180); do
    curl -sf "http://127.0.0.1:$P/health" >/dev/null && break
    kill -0 "${PIDS[$i]}" 2>/dev/null || { echo "llama-server :$P died" >&2; exit 1; }
    sleep 2
  done
  curl -sf "http://127.0.0.1:$P/health" >/dev/null || { echo "llama-server :$P never became healthy" >&2; exit 1; }
  # A healthy server on CPU is a silent money drain (2 t/s vs 45): die loudly
  # so the host's broken driver is visible instead of billed.
  if grep -q "failed to initialize CUDA" "/tmp/llama_$((P - 8080)).log"; then
    echo "GPU INIT FAILED on :$P (driver/CUDA mismatch on host) — refusing CPU fallback" >&2
    exit 1
  fi
done
export NACA_VLM_URLS="$URLS"
export NACA_VLM_URL="http://127.0.0.1:8080"

case "$MODE" in
  standalone) python3 /worker/unit_worker.py --standalone "${UNITS:-/work/units.jsonl}" --data-root "${DATA_ROOT:-/work/data}" "$@" ;;
  boinc)
    : "${BOINC_PROJECT_URL:?set BOINC_PROJECT_URL, e.g. http://<TAILNET_IP>/naca/}"
    : "${BOINC_AUTHENTICATOR:?set BOINC_AUTHENTICATOR (account key; keep it in env, never in files under git)}"
    URL="${BOINC_PROJECT_URL%/}/"
    KEY="$(echo "$URL" | sed -E 's#^https?://##; s#/+$##; s#/#_#g')"   # BOINC's url_to_filename
    D="${BOINC_DIR:-/var/lib/boinc}"
    mkdir -p "$D/projects/$KEY"
    A=naca_vision_1.00_x86_64-pc-linux-gnu
    cp /worker/boinc/naca_vision "$D/projects/$KEY/$A"
    cp /worker/boinc/app_info.xml "$D/projects/$KEY/app_info.xml"
    cp /worker/boinc/global_prefs_override.xml "$D/"
    # 2 slots per GPU (each llama-server runs -np 2)
    sed "s#<ncpus>2</ncpus>#<ncpus>$((2 * N))</ncpus>#" /worker/boinc/cc_config.xml > "$D/cc_config.xml"
    umask 077
    printf '<account>\n<master_url>%s</master_url>\n<authenticator>%s</authenticator>\n</account>\n' \
      "$URL" "$BOINC_AUTHENTICATOR" > "$D/account_$KEY.xml"
    # foreground (not exec) so the EXIT trap still stops llama-server
    boinc --dir "$D" &
    BOINC_PID=$!
    trap 'kill $BOINC_PID "${PIDS[@]}" 2>/dev/null || true' TERM INT
    wait $BOINC_PID ;;
  *) echo "unknown mode $MODE" >&2; exit 64 ;;
esac
