#!/bin/bash
# Pipeline dashboard in Docker, port 8777. --pid=host so pgrep sees host pass*.py.
# Detached with restart policy: survives session death and daemon/host restarts.
set -e
R=~/data_normalizing
docker build -q -t naca-dashboard -f "$R/pipeline/dashboard.Dockerfile" "$R/pipeline" >/dev/null
docker rm -f naca-dashboard 2>/dev/null || true
docker run -d --restart unless-stopped --name naca-dashboard --pid=host \
  --runtime nvidia --gpus all \
  -p 8777:8777 -v "$R":/app:ro naca-dashboard
echo "dashboard up on http://192.168.1.149:8777"
