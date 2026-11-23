#!/bin/bash
# docker compose wrapper for the NACA BOINC stack (run from anywhere).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec docker compose --project-directory "$HERE/server" -p naca-boinc --env-file "$HERE/naca.env" \
  -f "$HERE/server/docker-compose.yml" -f "$HERE/docker-compose.override.yml" "$@"
