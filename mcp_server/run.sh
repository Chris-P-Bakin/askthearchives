#!/bin/bash
# Build and run the NACA MCP server trial container on this workstation.
# Host port: NACA_MCP_PORT (default 8810; 8800 is taken on this box).
set -euo pipefail
cd "$(dirname "$0")/.."
DATA="${NACA_DATA_DIR:-$PWD/data}"
PORT="${NACA_MCP_PORT:-8810}"
docker build -q -t naca-mcp:trial -f mcp_server/Dockerfile .
docker rm -f naca-mcp 2>/dev/null || true
docker run -d --restart unless-stopped --name naca-mcp -p "$PORT:8800" \
  -v "$DATA/corpus.db:/index/corpus.db:ro" \
  -v "$DATA:/data:ro" \
  naca-mcp:trial
echo "naca-mcp up on http://$(hostname -I | awk '{print $1}'):$PORT/mcp"
