# Archives observability (tailnet-only)

Collector (host systemd user unit) -> ClickHouse on obs (`clickhouse`, :8123) -> Grafana on obs
(`grafana`, http://<TAILNET_IP>:3002). Everything binds the tailnet address only.

## Data flow

- `pipeline/obs_ship.py`, run by the `archives-metrics` systemd user unit (`Restart=always`),
  gathers campaign state (BOINC, escalation queues, layer tree, GPU, ingest log, cloudflared
  counters) every 30-120 s into `archives.samples`, and drains the MCP request spool
  (`data/mcp_requests.ndjson`) into `archives.requests` every 10 s. A failing source emits a
  `naca_collector_errors{source=...}` sample; the loop never dies.
- Tables: `archives.samples(ts, name, labels Map, value)` and
  `archives.requests(ts, tool, duration_ms, bytes, results, status, degraded)`; 1-year TTL.
  `status` is `ok`, `error` or `rejected`; latency and error-rate panels use only ok/error.

## Install (idempotent)

    observability/install_obs.sh

Run it from anywhere on a host that can `docker exec` into obs's containers and reach Grafana.
It:

1. creates `deploy/secrets-local/clickhouse.env` (mode 600, gitignored) with a generated password
   if absent; an existing file is kept;
2. through `docker exec clickhouse clickhouse-client` (SQL, no XML) creates database `archives`,
   user `archives` (password re-asserted with `ALTER USER`), `GRANT SELECT, INSERT, CREATE TABLE`
   on `archives.*`, and both tables with `IF NOT EXISTS`;
3. reads the Grafana admin password from `docker inspect grafana` (never stored or printed),
   upserts the `archives-ch` datasource (`grafana-clickhouse-datasource`) and imports
   `grafana/campaign.json` and `grafana/serving.json` with overwrite.

The Grafana ClickHouse datasource plugin must be installed on `grafana`
(`GF_INSTALL_PLUGINS=grafana-clickhouse-datasource`). The collector reads its credentials from
`deploy/secrets-local/clickhouse.env`.

## Dashboards

- `grafana/campaign.json` (uid `archives-campaign`): BOINC workunit states, escalation lanes
  pending, 235B spend, pages per layer, GPU memory, ingest MB/day and pages/day, collector errors.
- `grafana/serving.json` (uid `archives-serving`): requests/min by status (stacked),
  searches and pages served per hour, degraded-search rate, latency p50/p95, error rate, egress
  MB/h (result bytes and cloudflared muxer), tunnel request errors.

Cloudflared counters are stored raw; the panels derive per-bucket increases (counter resets
clamp to zero).

## Rollback

The old Prometheus stack (`naca-obs-prometheus`, `naca-obs-grafana`, compose project
`naca-observability` in `docker-compose.yml`) was stopped, not removed. To roll back:

    docker start naca-obs-prometheus naca-obs-grafana

and run the retired exporter again (`pipeline/metrics_exporter.py`, removed from the tree; restore
it from git history). Stop the `archives-metrics` unit first if you do not want both pipelines.
