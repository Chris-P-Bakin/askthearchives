# Public MCP endpoint: deploy runbook

Architecture: `archives-mcp` (systemd user service) binds `127.0.0.1:8800`; a
Cloudflare tunnel (`cloudflared`) exposes it as `https://mcp.<domain>/mcp`;
the static homepage in `site/` is served from Cloudflare Pages.

Host port note: port 8800 on the tailnet IP is held by an unrelated tenant
service. The MCP server binds `127.0.0.1:8800`, which is free, and the tunnel
targets `127.0.0.1:8800`. Do not change `ARCHIVES_HOST` to the tailnet IP.

Server env keys (set in `deploy/systemd/archives-mcp.service`): `ARCHIVES_HOST`,
`ARCHIVES_PORT` (default 8800), `ARCHIVES_DB`, `ARCHIVES_DATA`,
`ARCHIVES_RATE`, `ARCHIVES_BURST`, and `ARCHIVES_PUBLIC_HOST` — set it to the
tunnel hostname (`mcp.<domain>`) via a systemd drop-in (`systemctl --user edit archives-mcp`), or the server's
DNS-rebinding protection rejects tunneled requests with "Invalid Host header".

## 1. Register the domain in Cloudflare

Register or transfer `<domain>` into Cloudflare so its DNS is managed there.

## 2. Install the systemd user units and start the MCP service

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/archives-mcp.service deploy/systemd/archives-index.service \
   deploy/systemd/archives-index.timer deploy/systemd/cloudflared.service \
   ~/.config/systemd/user/
systemctl --user daemon-reload
loginctl enable-linger "$USER"           # keep user units running without a login
```

Start the service and the nightly index refresh timer. The pilot container
(below) holds host port 8810, not 8800, so both can run briefly side by side.

```bash
systemctl --user enable --now archives-mcp.service
systemctl --user enable --now archives-index.timer
systemctl --user status archives-mcp.service
```

`archives-index.service` runs `pipeline/index_refresh.py --restart-unit archives-mcp`
nightly at 03:30: it builds a staged index, refuses to swap an unreadable or
empty one, then restarts `archives-mcp`. Run it by hand with
`systemctl --user start archives-index.service`.

## 3. Verify locally, then retire the pilot container

Verify the service on loopback before anything else:

```bash
curl -sS http://127.0.0.1:8800/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"smoke","version":"0"}}}'
```

Only after `systemctl --user status archives-mcp` is healthy and the handshake
answers, retire the pilot container named `naca-mcp` (running on host port
8810; its name and purpose overlap the systemd service):

```bash
docker stop naca-mcp && docker rm naca-mcp
```

## 4. Create the tunnel and bring it up

```bash
mkdir -p deploy/secrets-local            # gitignored
cloudflared tunnel login
cloudflared tunnel create naca-mcp       # prints the tunnel UUID, writes the credentials JSON
mv ~/.cloudflared/<TUNNEL_UUID>.json deploy/secrets-local/
cp deploy/cloudflared/config.yml.example deploy/secrets-local/config.yml
# edit config.yml: replace every REPLACE-ME-* token (tunnel UUID, user, domain)
cloudflared tunnel route dns naca-mcp mcp.<domain>
systemctl --user enable --now cloudflared.service
systemctl --user status cloudflared.service
```

The unit runs `cloudflared tunnel --config deploy/secrets-local/config.yml run naca-mcp`
and is ordered after `archives-mcp.service`, so the tunnel never fronts a missing
server.

## 5. Cloudflare dashboard

On the `<domain>` zone:

- Security > WAF > Rate limiting rules: add a rule on host `mcp.<domain>`
  (for example 60 requests per minute per IP, action Block). This is the edge
  layer above the server's own `ARCHIVES_RATE` / `ARCHIVES_BURST` limiter.
- Security > Bots: enable Bot Fight Mode.

## 6. Deploy the homepage (Cloudflare Pages)

Create a Pages project (direct upload) and publish the `site/` directory:

```bash
npx wrangler pages deploy site --project-name naca-site
```

Attach `<domain>` as the custom domain. Do this only at launch (step 8); fill
the placeholders in `site/index.html` first.

## 7. Public smoke test

Handshake (Streamable HTTP, JSON-RPC):

```bash
curl -sS https://mcp.<domain>/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"smoke","version":"0"}}}'
```

A `search` call, interactively, with the MCP inspector:

```bash
npx @modelcontextprotocol/inspector
# transport: Streamable HTTP, URL: https://mcp.<domain>/mcp
# Tools > search > {"query": "boundary layer transition", "limit": 3}
```

The local check already ran in step 3; this confirms the tunnel and edge.

## 8. Launch gate

Do not announce or deploy the homepage until the batch-0 dataset publication is
done: `pipeline/publish.py` and the "Publication runbook (batch 0)" in
`boinc/README.md` produce the torrents. Launch checklist, all in
`site/index.html` unless noted:

- [x] torrent links filled (data.askthearchives.org R2 custom domain)
- [x] dataset counts filled (500 / 10,435 static fallback; /stats live-updates)
- [x] repo URL filled (github.com/Chris-P-Bakin/askthearchives)
- [x] domain: askthearchives.org is hardcoded in the page (endpoint already live)
- [x] page deployed (Cloudflare Pages project archives-site; deploy via data/publish helpers)

## Rename migration (live box)

The serving stack was renamed from `naca-*` to `archives-*` (env vars `NACA_*`
became `ARCHIVES_*`, no fallback shims). On the live box:

```bash
systemctl --user disable --now naca-mcp naca-index naca-index.timer
mv ~/.config/systemd/user/naca-mcp.service.d ~/.config/systemd/user/archives-mcp.service.d
sed -i 's/NACA_PUBLIC_HOST/ARCHIVES_PUBLIC_HOST/' ~/.config/systemd/user/archives-mcp.service.d/*.conf
rm ~/.config/systemd/user/naca-{mcp,index}.service ~/.config/systemd/user/naca-index.timer
# re-link the renamed units (see the install step above), then:
systemctl --user daemon-reload
systemctl --user enable --now archives-mcp archives-index.timer
```

Run this migration in the same session as deploying this code. The old units
cannot run the new server: it reads only `ARCHIVES_*`, so with the old
`NACA_*` env the next restart binds `0.0.0.0` (clashing with the tenant service
on 8800) and has no `ARCHIVES_PUBLIC_HOST`, so DNS-rebinding protection
rejects every tunneled request. Also expect the nightly `archives-index` to
exit 1 at the chunk step (the corpus swap still succeeds) until the chunk
rollout's step 2 is done; alternatively add `--no-chunks` to the installed
unit's `ExecStart` until then.

`cloudflared.service` is now ordered after `archives-mcp.service`; re-link it
and `systemctl --user restart cloudflared`. The Cloudflare tunnel itself keeps
its existing name `naca-mcp`.

## Chunk retrieval rollout

Chunk-granular search (Qwen3-Embedding + Qdrant + FTS5, with page provenance)
ships behind an acceptance gate. The page-level path stays untouched until the
cutover, so every step before step 5 is safe on the live box. Run from
`~/data_normalizing`.

1. **Verify the pinned embedding model.** The local GGUF must match
   `deploy/pins.json`:

   ```bash
   sha256sum data/models/qwen3-embed-0.6b/Qwen3-Embedding-0.6B-Q8_0.gguf
   grep -A2 embed_model deploy/pins.json    # sha256 must be identical
   ```

2. **Start the embed and vector-store units.** Link the units from
   `deploy/systemd/` (as in step 2 above), create the Qdrant storage dir
   yourself (docker would create it root-owned), then start:

   ```bash
   mkdir -p ~/data_normalizing/data/qdrant
   systemctl --user daemon-reload
   systemctl --user enable --now archives-embed archives-qdrant
   curl -s http://127.0.0.1:8201/health
   curl -s http://127.0.0.1:6333/collections
   ```

3. **Initial build** against the live `corpus.db` (CPU embedding of ~24k chunks,
   roughly 15-30 minutes; safe to re-run, it is incremental):

   ```bash
   .venv/bin/python pipeline/chunk_refresh.py
   ```

   (`pipeline/index_refresh.py` with `ARCHIVES_EMBED_URL` and
   `ARCHIVES_QDRANT_URL` set also runs it, as the nightly does.)

4. **Run the gate.** A degraded stack aborts with exit 2 and measures nothing:

   ```bash
   .venv/bin/python bench/retrieval/eval_production.py \
     --chunks-db data/chunks.db \
     --embed-url http://127.0.0.1:8201 --qdrant-url http://127.0.0.1:6333 \
     --queries bench/retrieval/queries.jsonl --limit 10
   ```

   Record the printed recall@{1,3,5,10} in `bench/retrieval/TRIAL-RESULTS.md`
   under "Production acceptance". Exit 0 means pass (recall@5 >= 0.70 and at
   least 0.10 above the 0.540 page-level baseline). **If it prints
   `GATE FAILED`, stop here: no cutover.**

5. **Cutover.** Add the chunk store to the archives-mcp override drop-in
   (`~/.config/systemd/user/archives-mcp.service.d/`, next to
   `ARCHIVES_PUBLIC_HOST`):

   ```ini
   [Service]
   Environment=ARCHIVES_CHUNKS_DB=%h/data_normalizing/data/chunks.db
   ```

   ```bash
   systemctl --user daemon-reload
   systemctl --user restart archives-mcp
   ```

   Verify a public search (step 7 above) returns results carrying `pages` and
   `original`.

6. **Post-cutover.** Add `--no-embed` to the `ExecStart` of
   `archives-index.service` so new builds stop writing `pages_vec`. Units are
   copied into `~/.config/systemd/user` at install, so edit the installed copy
   (or edit `deploy/systemd/archives-index.service` and re-copy it), then
   `systemctl --user daemon-reload`. Removing sqlite-vec from `corpus.db`
   ships with the next full rebuild.

7. **Rollback.** Remove the `ARCHIVES_CHUNKS_DB` line from the drop-in, then
   `systemctl --user daemon-reload && systemctl --user restart archives-mcp`.
   The page-level path is untouched, so this is fully clean up to step 6.
   After step 6, remove `--no-embed` and re-run the nightly first (otherwise
   `pages_vec` is stale for changed pages and semantic search quietly
   degrades). Not available after the sqlite-vec-removing rebuild:
   `validate_index` hard-fails.
