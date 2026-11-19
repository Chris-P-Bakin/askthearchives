# NACA BOINC project (alpha, tailnet-only)

A minimal BOINC server for the volunteer-compute plane. Trusted-workers phase:
`min_quorum=1, target_nresults=1`. Project URL: `http://<TAILNET_IP>/naca/`
(tailnet only; ops pages at `/naca_ops/`, user `admin`, password in
`boinc/server/secrets-local/ops.pass`).

Windows 5090 onboarding is `docs/worker-windows-wsl2.md` (BOINC mode section).
Spot/volunteer workers join only after a full batch has run end-to-end locally
(spec order-of-execution). The Task 8 standalone mode remains the fallback queue.

## Layout

| Path | What |
|---|---|
| `boinc/server/` | clone of `BOINC/boinc-server-docker` @ `1e77585` (gitignored), submodules pinned by gitlink: BOINC `679b7c8` (2026-06-12), PHPMailer `f0b2877` |
| `boinc/server/secrets-local/` | generated DB password, ops password, worker account key (gitignored, mode 600/644) |
| `boinc/naca.env` | compose variables (URL_BASE, PROJECT=naca, image tag `local`) |
| `boinc/docker-compose.override.yml` | tailnet-only port bind, no docker.sock, NACA mounts |
| `boinc/dc.sh` | `docker compose` wrapper with the right project dir / env / override |
| `boinc/install_project.sh` | idempotent post-start setup (ops password, glue, app, daemons) |
| `boinc/glue/` | WU/result templates and the validator/assimilator hook scripts |
| `boinc/stage/` | scratch dir `boinc_submit.py` stages inputs in (gitignored) |
| `pipeline/boinc_submit.py` | units jsonl -> staged inputs -> `create_work` |
| `worker/boinc/` | anonymous-platform app wrapper, `app_info.xml`, client config used by the worker container |

## Reproduce the stack (every command)

    git clone https://github.com/BOINC/boinc-server-docker boinc/server
    git -C boinc/server checkout 1e77585   # pin the top-level commit (full clone, not --depth 1)
    git -C boinc/server submodule update --init --recursive   # submodule SHAs come from the gitlinks at that commit
    # generate secrets (never committed)
    mkdir -p boinc/server/secrets-local
    printf 'DB_PASSWD=%s\nRECAPTCHA_PUBLIC_KEY=\nRECAPTCHA_PRIVATE_KEY=\nMAILPASS=\n' "$(openssl rand -hex 16)" > boinc/server/secrets-local/secrets.env
    openssl rand -base64 18 | tr -d '/+=' > boinc/server/secrets-local/ops.pass
    boinc/dc.sh build                  # ~15 min: compiles BOINC server from the pinned source
    boinc/dc.sh up -d mysql
    boinc/dc.sh run --rm makeproject   # = upstream `make post-makeproject`: builds project `naca`, creates DB
    boinc/dc.sh up -d apache
    boinc/install_project.sh
    curl http://<TAILNET_IP>/naca/   # 200

Account for the alpha worker (key lands only in `secrets-local/alpha-worker.key`):

    H=$(printf '%s' "${PASSWD}${EMAIL}" | md5sum | cut -d' ' -f1)
    curl "http://<TAILNET_IP>/naca/create_account.php?email_addr=$EMAIL&passwd_hash=$H&user_name=alpha-worker"

Submit and run:

    .venv/bin/python pipeline/boinc_submit.py data/units/batch_000_units.jsonl --limit 10
    docker run -d --name naca-alpha-worker --gpus all \
      -v $PWD/models/qwen3-vl-8b-lora-v2_1:/models:ro \
      -e BOINC_PROJECT_URL=http://<TAILNET_IP>/naca/ \
      -e BOINC_AUTHENTICATOR="$(cat boinc/server/secrets-local/alpha-worker.key)" \
      naca-worker boinc

## Non-obvious choices

* **Images are built locally from the pinned source, not pulled.** The upstream compose
  names `boinc/server_*:latest-defaultargs` Hub images; floating tags, unauditable. We
  build from the submodule commit and tag them `boinc/server_*:local-defaultargs`
  (`VERSION=local`). Base images (`php:apache-bookworm`, `mariadb:lts`, `debian:bookworm-slim`)
  are floating upstream tags: the built image ids at stand-up time were apache `6f59abfc3962`,
  makeproject `b3746dafbc35`, mysql `d23eab1167e5`.
* **No docker.sock in the apache container.** Upstream mounts it (root-equivalent on the
  host); only the boinc2docker variant needs it. Dropped via `volumes: !override`.
* **Default credentials replaced.** The upstream image bakes `DB_PASSWD=password` and a
  fixed ops `.htpasswd`. We bind-mount a generated `secrets.env` over it and
  `install_project.sh` rewrites the ops htpasswd from `ops.pass`.
* **Port 80 only, bound to `<TAILNET_IP>`** (`ports: !override`). MySQL publishes nothing.
  `ss -tlnp` shows only `<TAILNET_IP>:80` for the stack.
* **Glue is symlinked into the project tree after start**, not bind-mounted into it:
  mounting into the still-empty project volume leaves root-owned dirs that break
  `makeproject-step2`.
* **Validator/assimilator = `script_validator` / `script_assimilator` -> thin hooks**
  (`glue/bin/naca_validate`, `naca_assimilate`), configured with the BARE script name (`--init_script naca_validate`, `--script naca_assimilate`): the quoted two-token form (`"naca_validate files"`) passed no file paths in this build, the default tokens do. BOINC passes output-file paths only; the hooks derive the
  unit id from the output file name (`<wu>_<resultidx>_r<random>_<fileidx>`, WU = `naca_<unit_id>`) and find
  `unit.json` in the download hierarchy via `bin/dir_hier_path`, then call
  `pipeline/unit_validate.py` / `unit_assimilate.py` unchanged. They run inside the apache
  container with `pipeline/` mounted read-only at `/naca/pipeline` and `data/` read-write
  at `/naca/data` (the assimilator is the only writer; boincadm uid 1000 = host user).
  Infra errors in the validator exit 3 (`VAL_RESULT_TRANSIENT_ERROR`: retry later) rather
  than invalidating a good result.
* **Interim download host.** Inputs are staged into BOINC's own download hierarchy
  (`stage_file --copy`) and served by this project's Apache; R2 credentials are still
  pending. Staged copies are the unit json + the page PNG (sha256-checked against the unit
  before staging). Swap to R2 URLs (`--remote_file`) when creds land.
* **Anonymous platform in the worker container.** `worker/Dockerfile` installs
  `boinc-client`; `entry.sh boinc` starts llama-server first, then the client with an
  `account_*.xml` (authenticator from `BOINC_AUTHENTICATOR` env only) and an
  `app_info.xml` for `naca_vision`. The app `worker/boinc/naca_vision` is a plain script,
  not linked against the BOINC API, so it has two duties the API would normally cover:
  (1) resolve output paths from the `<soft_link>` files the client pre-creates for
  `result.txt` / `result.meta.json` in the slot dir; (2) write `boinc_finish_called`
  (otherwise the client reports "exited with zero status but no finished file").
  Inputs use `<copy_file/>` so `unit.json` / `page.png` are real files.
* **`global_prefs_override.xml` sets `suspend_cpu_usage=0`.** Without it the client
  suspends tasks whenever non-BOINC CPU load > 25% (this box is always busier).
* **2 concurrent tasks** (`ncpus=2` in `cc_config.xml`) to match llama-server `-np 2`.
* Worker `unit_worker.py` processing logic is untouched.

## Operating notes

* `boinc/dc.sh ps|logs|down` manage the stack. `down` keeps volumes; `down -v` wipes the project.
* Project-state queries: `docker exec naca-boinc-apache bash -c 'source /run/secrets/secrets.env; mysql -h mysql -uroot -p$DB_PASSWD naca -e "..."'`.
* `boinc_submit.py` is idempotent (skips WUs already in the DB). WU result states: `server_state` 5 = over, `validate_state` 1 = valid, workunit `assimilate_state` 2 = assimilated.
* **Cancelling WUs deletes their input files, with no cross-WU reference counting.** `file_deleter`
  removes a finished (or cancelled) WU's inputs from the download hierarchy even when another live
  WU references the same physical file. The single-page and batched apps share page PNG names, so
  cancelling superseded `naca_vision` WUs 404s the in-flight `naca_vision_b` downloads (hit live
  2026-11-16). Clients treat the 404s as transient and retry, so the fix is to re-stage the pages
  (`stage_unit` each unit, then `bin/stage_file --copy` the PNGs) — or cancel only after no live WU
  shares inputs with the cancelled set.
* Open account creation is on (tailnet-only alpha). Set `disable_account_creation` in `config.xml` before widening access.

## Alpha run (2026-11-14): 10 real batch-0 units, end to end

10 WUs submitted -> 10 results returned, 10 validated (0 invalid), 10 assimilated into
`data/19930091026/vision_8blora_v2_1/` with config hash `6ecf7d0dbeb2`; 3 digit-dense pages
(p9, p13, p14) queued as `table_flags` in `data/escalation_queue.jsonl`. Per-page worker seconds
(2 concurrent streams on the 3090): 10.0 10.9 26.7 10.9 18.8 16.0 203.4 22.3 26.1 39.9
(mean 38.5, median 20.5; p9 is a 14.5k-char table page). Wall 06:07:30 -> 06:13:51 including model load.

Gotchas hit (all fixed in the committed files): client "suspending - CPU is busy" (prefs override);
soft-link output files and the finished marker (app wrapper); two-token script args (above);
result file names carry `_r<random>`; and never hot-patch the app executable inside a running
client - it verifies the file and the tasks stick in "downloading" until the client restarts.
Persist `/var/lib/boinc` in a volume for any long-lived worker so the host id survives restarts.

## Escalation lanes

Pages that batch-0 could not trust are appended to `data/escalation_queue.jsonl` (kinds:
`table_flags`, `vision_failed`, `wu_error`). `pipeline/escalate.py` drains them by lane:

```
.venv/bin/python pipeline/escalate.py status                    # per-route pending counts + reconciliation
.venv/bin/python pipeline/escalate.py run-27b [--limit N] [--dry-run]       # vision_failed -> local Qwen 27B
.venv/bin/python pipeline/escalate.py run-235b [--limit N] [--budget 120] [--dry-run]  # table_flags -> hosted 235B
.venv/bin/python pipeline/escalate.py run-wu-error --units UNITS_FILE [--dry-run]  # resubmit errored BOINC workunits (suffix __r2..__r4)
```

Always `--dry-run` first and compare the planned count with `status`. All lanes are idempotent
and resumable; finished rows are recorded in `data/escalation_done.jsonl`.

* **GPU swap discipline (27B lane).** The 3090 cannot hold the 27B and the alpha worker's model.
  Stop the worker, start the 27B, drain, then restore: `docker stop naca-alpha-worker`,
  `docker start naca-vlm-27b` (wait healthy on :8781), `run-27b`, then `docker stop naca-vlm-27b`
  and, as a separate command, `docker start naca-alpha-worker`; confirm with `docker logs --tail`
  that it fetches tasks. Never touch the tenant `the tenant GPU service`. If the 27B dies mid-drain
  (tenant VRAM race), wait for free VRAM and rerun; run-27b picks up where it stopped.
* **Budget accounting (235B lane).** Hard cap $120 (`--budget`). Spend is read back from the
  per-page metas written by pass4a plus any `cost` extras in the done ledger, not estimated; work runs in
  chunks of 25 and a chunk is refused (`budget_stop`) if spent + (mean cost/page of this invocation's
  pages so far, floor $0.02 before any data) x chunk size would reach the cap.
  Needs the OpenRouter key in `NACA_VLM_KEY` (the lane refuses to dispatch without it).
* **Single operator.** Do not run two `run-235b` (or two lanes touching the same queue) at once;
  budget and done-file accounting assume one writer.
* **Review queue (`data/escalation_review.jsonl`).** Lane-owned, append-only (kept separate from
  `data/review_queue.jsonl`, which `pass5.py` rewrites on every run). Rows land here when a lane
  gave up on a page: `vision_failed_27b` / `vision_failed_235b` (loop guard fired), `empty_output_27b` /
  `empty_output_235b` (engine returned empty text, e.g. length-capped with no visible output; the
  loop guard skips text under `min_chars`, so the lane checks emptiness itself), and
  `table_flags_after_235b` (still digit-suspect after the 235B). They need a human or a different
  engine; they are not drain errors. Empty-output pages are not marked done in the ledger.
* **Status of live validation.** The 27B lane has been run live. The 235B lane's live run is
  PENDING the operator supplying the OpenRouter API key; it is covered by tests only so far.

## Publication runbook (batch 0)

Run only after batch 0 has drained and the plan-3 `spend()` fix gate has cleared.

1. `.venv/bin/python pipeline/publish.py export --batch 0` — builds and commits `data/publish/batch_000/`
   (manifest.jsonl is the commit marker). Re-running a committed batch only verifies it. `export`
   exits 0 even when a committed batch has drifted (mismatches are listed in the printed summary,
   `"verified": false`); the `torrent` step is the enforcing gate and refuses a drifted tree.
2. `.venv/bin/python pipeline/publish.py torrent --batch 0`
3. `.venv/bin/python pipeline/publish.py cumulative` — errata are regenerated implicitly for every
   committed batch (`gen_errata`, deterministic) before staging, so `errata/` always matches the
   `corpus.db` built in the same run. `corpus.db` is built WITH embeddings (hybrid FTS5 + sqlite-vec),
   which loads the embedding model and takes a while. Manual `publish.py errata --batch N` remains
   available to inspect overlays before cumulative.
4. `.venv/bin/python pipeline/ledger.py --batch 0 --write`
5. R2 sync, then hand the torrents to the operator for seeding. Once steps 1–5 are complete and the batch-0 publication is done, fill `site/index.html`'s launch placeholders (torrent links, `N_REPORTS`/`N_PAGES` counts, `NACA_REPO_URL`) and redeploy Cloudflare Pages (`deploy/README.md` step 8).
6. Which index serves the public endpoint depends on whether the systemd migration (`deploy/README.md`
   steps 2–3) has run.
   * **Before the migration**, the docker pilot is still serving. Swap naca-mcp onto the published db
     (documented only; not automated). The container mounts the host file `data/corpus.db` read-only at
     `/index/corpus.db` (see `mcp_server/run.sh`), so replace that file and restart (stop first so no
     reader has the old inode open). First back up the previous db (it holds the legacy hierarchy):
     `cp data/corpus.db data/corpus.db.pre-publish`, then, as separate commands:
     `docker stop naca-mcp` ; `cp data/publish/corpus.db data/corpus.db` ; `docker start naca-mcp`
     (or `NACA_DATA_DIR=<dir containing the published corpus.db> mcp_server/run.sh` to recreate it).
   * **After the migration, do NOT hand-swap the published db into `data/corpus.db`.** `archives-index.timer`
     rebuilds `data/corpus.db` from everything ingested under `data/` (the `--all-ingested` set) at 03:30
     and restarts `archives-mcp`, which silently supersedes a hand-swapped file. The published
     `data/publish/corpus.db` is a different, batch-scoped artifact (committed batches only) built for the
     torrent; the public index is meant to cover everything ingested. To rebuild the index on demand, run
     `systemctl --user start archives-index.service` (see `deploy/README.md` step 2).

## Windows native app (naca_vision_b)

Batched app for volunteer Windows + NVIDIA boxes (no Docker/WSL2): one WU = N
pages (default 32), signed `windows_x86_64` app version under plan class
`cuda_10g` (NVIDIA, >=10GB VRAM, driver >= 551.61). Spec:
`docs/superpowers/specs/2026-11-15-windows-native-worker-design.md`; acceptance
runbook: `docs/worker-windows-native.md`.

* **Bundle**: official BOINC wrapper 26016 (main program) + job.xml →
  embeddable CPython 3.12 runs `worker/win/launcher.py`, which starts the
  bundled llama.cpp b12199 win-cuda server (volunteer profile `-c 16384 -np 1`)
  and loops the manifest pages through `unit_worker.py` with per-page
  checkpointing. Pins (versions + sha256 + the Linux engine build they match):
  `boinc/win/pins.json`; artifacts cached in gitignored `boinc/vendor/`.
* **Build/sign/register**: `.venv/bin/python boinc/win/package.py fetch`,
  `assemble`, `stage` (copies into `apps/naca_vision_b/2.00/`, signs every file
  with `boinc/server/secrets-local/code_sign_private` — generated once with
  `crypt_prog -genkey`, never in the container except during signing — then
  `update_versions`).
* **Models**: both GGUFs staged once as sticky files
  (`qwen3vl8b_lora_v2_1_q8.gguf`, `qwen3vl8b_mmproj_f16.gguf`); every batch WU
  references them, clients download once.
* **Submit**: `.venv/bin/python pipeline/boinc_submit.py <units.jsonl> --app
  naca_vision_b --batch 32 [--gt]`. `--gt` tags the WU name `naca_b_gt_*`; the
  assimilator routes those to `data/<id>/vision_8bwin/` (acceptance runs),
  everything else to the campaign layer.
* **Server glue**: `script_validator`/`script_assimilator` daemons for
  `naca_vision_b` (hooks `naca_validate_b`/`naca_assimilate_b` → pipeline
  `batch_validate.py`/`batch_assimilate.py`; whole-result invalid on bad hash /
  malformed line / silently missing page, honest per-page errors escalate only
  those pages). Daemon entries need distinct `<output>`/`<pid_file>` names or
  BOINC's per-name lock keeps the second validator instance from starting.

## Ops gotcha: feeder shared-memory cache

The scheduler serves workunits from the feeder's shared-memory cache, not
the DB. Any direct DB edit to `workunit.xml_doc` / `result.xml_doc_in`
(e.g. the 2026-11-18 tailnet→public URL rewrite) is invisible until the
project daemons restart (`bin/stop && bin/start`). Clients that already
hold stale copies keep retrying them until deadline; new assignments are
clean after the restart.

During the R2 input migration (2026-11-19): migrated unsent results get
priority 100 so remote workers drain CDN-backed work first; unmigrated
(tunnel-URL) units are served last and mostly to the tailnet-local worker.
Re-run the priority UPDATE + daemon restart if the migrator is rerun.
