#!/bin/bash
# Container entry: start llama-server (same args as pipeline/serve_vlm_8blora.sh),
# wait for /health, then run the worker.
#   standalone  -> python3 unit_worker.py --standalone /work/units.jsonl --data-root /work/data [args]
#   boinc       -> BOINC client, anonymous platform (needs BOINC_PROJECT_URL + BOINC_AUTHENTICATOR)
set -euo pipefail
MODE="${1:-standalone}"; shift || true
/app/llama-server \
  -m /models/Qwen3-VL-8B-lora-v2_1-Q8_0.gguf \
  --mmproj /models/mmproj-Qwen3VL-8B-Instruct-F16.gguf \
  -ngl 99 -c 32768 -np 2 --temp 0 --host 127.0.0.1 --port 8080 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
for _ in $(seq 180); do
  curl -sf http://127.0.0.1:8080/health >/dev/null && break
  kill -0 $SERVER_PID 2>/dev/null || { echo "llama-server died" >&2; exit 1; }
  sleep 2
done
curl -sf http://127.0.0.1:8080/health >/dev/null || { echo "llama-server never became healthy" >&2; exit 1; }
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
    cp /worker/boinc/cc_config.xml /worker/boinc/global_prefs_override.xml "$D/"
    umask 077
    printf '<account>\n<master_url>%s</master_url>\n<authenticator>%s</authenticator>\n</account>\n' \
      "$URL" "$BOINC_AUTHENTICATOR" > "$D/account_$KEY.xml"
    # foreground (not exec) so the EXIT trap still stops llama-server
    boinc --dir "$D" &
    BOINC_PID=$!
    trap 'kill $BOINC_PID $SERVER_PID 2>/dev/null || true' TERM INT
    wait $BOINC_PID ;;
  *) echo "unknown mode $MODE" >&2; exit 64 ;;
esac
