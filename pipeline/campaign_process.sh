#!/bin/bash
# Campaign processor: free-running. Sweeps tesseract (pass3a) and layout
# (pass2) over every page on disk — both are idempotent per page, so pages
# are OCR'd as downloads land, not in end-of-batch bursts. Whenever a batch's
# ingest is complete (downloader marker) and not yet work-genned, mints its
# vision units and submits the 50/50 GPU split (odd lines -> 5090 batched
# app, even -> 3090 single-page app). Exits when batches 2-30 are all
# processed. Never waits on the downloader.
set -u
cd "$(dirname "$0")/.."
export NACA_REPORTS_CSV=data/roster.csv PYTHONUNBUFFERED=1
mkdir -p data/grind_markers
# 5090 share: the workstation's 5090 drops off the bus under load (2026-11-18);
# all vision work goes to the 3090 single-page app until it is stable.
# Set SPLIT_5090=1 to restore the 50/50 split.
SPLIT_5090=${SPLIT_5090:-0}
while :; do
  echo "=== $(date -u +%FT%TZ) sweep: pass3a (tesseract) ==="
  .venv/bin/python pipeline/pass3a.py --workers 8 || echo "pass3a sweep failed; retrying next cycle"
  echo "=== $(date -u +%FT%TZ) sweep: pass2 (layout) ==="
  .venv/bin/python pipeline/pass2.py --workers 8 || echo "pass2 sweep failed; retrying next cycle"
  done_all=1
  for b in $(seq 2 30); do
    if [ -f "data/grind_markers/batch_$b.processed.done" ]; then continue; fi
    if [ ! -f "data/grind_markers/batch_$b.pass1.done" ]; then done_all=0; continue; fi
    echo "=== $(date -u +%FT%TZ) batch $b workgen ==="
    .venv/bin/python pipeline/workgen.py --batch $b \
      || { echo "batch $b workgen FAILED; retrying next cycle"; done_all=0; continue; }
    u=$(printf 'data/units/batch_%03d_units.jsonl' $b)
    ok=1
    if [ "$SPLIT_5090" = 1 ]; then
      awk 'NR%2==1' "$u" > "$u.5090"
      awk 'NR%2==0' "$u" > "$u.3090"
      echo "=== $(date -u +%FT%TZ) batch $b submit (split 50/50) ==="
      .venv/bin/python pipeline/boinc_submit.py "$u.5090" --app naca_vision_b --batch 32 \
        || { echo "batch $b 5090 submit FAILED; retrying next cycle"; ok=0; }
      .venv/bin/python pipeline/boinc_submit.py "$u.3090" \
        || { echo "batch $b 3090 submit FAILED; retrying next cycle"; ok=0; }
    else
      echo "=== $(date -u +%FT%TZ) batch $b submit (all to 3090 single-page) ==="
      .venv/bin/python pipeline/boinc_submit.py "$u" \
        || { echo "batch $b 3090 submit FAILED; retrying next cycle"; ok=0; }
    fi
    if [ "$ok" = 1 ]; then
      touch "data/grind_markers/batch_$b.processed.done"
      echo "=== $(date -u +%FT%TZ) batch $b DONE ==="
    else
      done_all=0
    fi
  done
  [ "$done_all" = 1 ] && break
  sleep 300
done
echo "=== ALL PROCESSING DONE $(date -u +%FT%TZ) ==="
