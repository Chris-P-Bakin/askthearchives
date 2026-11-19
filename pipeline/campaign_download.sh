#!/bin/bash
# Campaign downloader: pass1 (download + raster) batch after batch, nothing
# else. Fully independent of the processor; drops data/grind_markers/
# batch_N.pass1.done as each batch's ingest completes. Idempotent — safe to
# restart anywhere (pass1 skips sha-verified docs).
set -u
cd "$(dirname "$0")/.."
export NACA_REPORTS_CSV=data/roster.csv PYTHONUNBUFFERED=1
mkdir -p data/grind_markers
for b in $(seq 2 30); do
  [ -f "data/grind_markers/batch_$b.pass1.done" ] && continue
  echo "=== $(date -u +%FT%TZ) batch $b pass1 (download+raster) ==="
  .venv/bin/python pipeline/pass1.py --batch $b --batch-size 500 --workers 8 \
    || { echo "batch $b pass1 FAILED"; exit 1; }
  touch "data/grind_markers/batch_$b.pass1.done"
done
echo "=== ALL DOWNLOADS DONE $(date -u +%FT%TZ) ==="
