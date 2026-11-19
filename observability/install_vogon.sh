#!/usr/bin/env bash
# Idempotent installer: ClickHouse user + tables on clickhouse, Grafana datasource + dashboards.
# Never prints passwords. Safe to re-run.
set -euo pipefail
set +x

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
ENVF="$REPO/deploy/secrets-local/clickhouse.env"
HOST_IP="<TAILNET_IP>"
GRAFANA="http://$HOST_IP:3002"

# 1. credentials file (generate once, never overwrite)
mkdir -p "$(dirname "$ENVF")"
if [ ! -f "$ENVF" ]; then
  umask 077
  {
    echo "ARCHIVES_CH_URL=http://$HOST_IP:8123"
    echo "ARCHIVES_CH_USER=archives"
    echo "ARCHIVES_CH_PASSWORD=$(openssl rand -base64 18 | tr -d '/+=')"
  } > "$ENVF"
fi
chmod 600 "$ENVF"
CH_PW="$(sed -n 's/^ARCHIVES_CH_PASSWORD=//p' "$ENVF")"
[ -n "$CH_PW" ] || { echo "empty ARCHIVES_CH_PASSWORD in $ENVF" >&2; exit 1; }
case "$CH_PW" in *[!A-Za-z0-9]*) echo "password has unexpected characters" >&2; exit 1;; esac

# 2. ClickHouse: SQL-managed user, database, tables (admin client inside the container).
# Password travels on stdin, not argv, so it never shows in process listings.
ch() { docker exec -i clickhouse clickhouse-client --multiquery; }
ch <<SQL
CREATE DATABASE IF NOT EXISTS archives;
CREATE USER IF NOT EXISTS archives IDENTIFIED BY '$CH_PW';
ALTER USER archives IDENTIFIED BY '$CH_PW';
GRANT SELECT, INSERT, CREATE TABLE ON archives.* TO archives;
CREATE TABLE IF NOT EXISTS archives.samples (
  ts     DateTime,
  name   LowCardinality(String),
  labels Map(LowCardinality(String), String),
  value  Float64
) ENGINE = MergeTree
  PARTITION BY toYYYYMM(ts)
  ORDER BY (name, ts)
  TTL ts + INTERVAL 1 YEAR;
CREATE TABLE IF NOT EXISTS archives.requests (
  ts          DateTime64(3),
  tool        LowCardinality(String),
  duration_ms Float32,
  bytes       UInt32,
  results     UInt16,
  status      LowCardinality(String),
  degraded    UInt8
) ENGINE = MergeTree
  PARTITION BY toYYYYMM(ts)
  ORDER BY ts
  TTL toDateTime(ts) + INTERVAL 1 YEAR;
SQL
echo "clickhouse: archives user and tables ready"

# 3. Grafana: admin creds from the container env (never stored or echoed)
GF_PW="$(docker inspect grafana --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | sed -n 's/^GF_SECURITY_ADMIN_PASSWORD=//p')"
GF_USER="$(docker inspect grafana --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | sed -n 's/^GF_SECURITY_ADMIN_USER=//p')"
GF_USER="${GF_USER:-admin}"
[ -n "$GF_PW" ] || { echo "GF_SECURITY_ADMIN_PASSWORD not found on grafana" >&2; exit 1; }

# credentials via a curl config on stdin keep them out of argv
gf() { # method path [json-body-file]
  local method="$1" path="$2" body="${3:-}"
  local args=(-sS --fail-with-body -X "$method" -H 'Content-Type: application/json' -K -)
  [ -n "$body" ] && args+=(--data-binary "@$body")
  printf 'user = "%s:%s"\n' "$GF_USER" "$GF_PW" | curl "${args[@]}" "$GRAFANA$path"
}

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
chmod 700 "$TMP"

DS_JSON="$TMP/ds.json"
CH_PW="$CH_PW" python3 - "$DS_JSON" <<'PY'
import json, os, sys
json.dump({
    "uid": "archives-ch", "name": "archives-ch",
    "type": "grafana-clickhouse-datasource", "access": "proxy", "isDefault": False,
    "jsonData": {"host": "<TAILNET_IP>", "port": 8123, "protocol": "http",
                 "secure": False, "username": "archives", "defaultDatabase": "archives"},
    "secureJsonData": {"password": os.environ["CH_PW"]},
}, open(sys.argv[1], "w"))
PY

# datasource upsert by uid: PUT if it exists, else POST
if gf GET /api/datasources/uid/archives-ch >/dev/null 2>&1; then
  gf PUT /api/datasources/uid/archives-ch "$DS_JSON" >/dev/null
  echo "grafana: datasource archives-ch updated"
else
  gf POST /api/datasources "$DS_JSON" >/dev/null
  echo "grafana: datasource archives-ch created"
fi

# dashboards: import with overwrite
for name in campaign serving; do
  body="$TMP/$name.json"
  python3 - "$HERE/grafana/$name.json" "$body" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
json.dump({"dashboard": d, "overwrite": True, "message": "install_obs.sh"}, open(sys.argv[2], "w"))
PY
  gf POST /api/dashboards/db "$body" >/dev/null
  echo "grafana: dashboard $name imported"
done
echo "done"
