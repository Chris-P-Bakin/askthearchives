#!/usr/bin/env bash
# Create the training venv (kept separate from the pipeline .venv per plan constraints).
set -euo pipefail
cd "$(dirname "$0")"
# uv-managed python ships C headers (system python3.12 lacks python3.12-dev; Triton JIT needs Python.h)
~/.local/bin/uv python install 3.12
[ -d .venv ] || ~/.local/bin/uv venv --seed --python-preference only-managed --python 3.12 .venv
.venv/bin/pip install -U pip
# Pinned to the v1-run stack (spec section 6: pin the month-tag release).
.venv/bin/pip install "unsloth==2026.11.3" "transformers==5.17.0" "trl==1.15.0" datasets pillow
mkdir -p logs runs
.venv/bin/python - <<'PY'
import torch, transformers, trl, unsloth
print("unsloth", unsloth.__version__)
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
print("transformers", transformers.__version__, "| trl", trl.__version__)
PY
