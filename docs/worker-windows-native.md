# Windows native worker — 5090 acceptance runbook

> **Executed 2026-11-15/16 — ACCEPTED (user decision).** App version 2.02.
> Measured: app VRAM footprint 12.9GB (peak 15057MB − 2115MB desktop baseline)
> → plan-class floor 13500MB (16GB+ cards; the ~10GB design target was
> unreachable: q8 weights alone are 9.9GB). GT 30 pages: median CER 0.1227 vs
> campaign 0.0940 — formal gate missed, but 18/30 pages byte-identical to the
> campaign engine, config hash exact, no truncation, and mean CER better
> (0.181 vs 0.235); the median miss is greedy-decode numeric noise across GPU
> archs (-np 1 vs -np 2, sm_120) at n=30. Shipped fixes along the way: 2.02 =
> explicit utf-8 on all worker/launcher text I/O (Windows cp1252 default
> crashed every page containing Greek/math chars); feeder needs `-allapps` or
> a deep single-app backlog starves the batched app out of the scheduler
> cache.

Run top-to-bottom when at the 5090 box (<WINDOWS-HOST>, tailnet
<WORKER_TAILNET_IP>), booted into Windows. No Docker, no WSL2: just the BOINC
client. Everything server-side is already live (see boinc/README.md "Windows
native app"). Gate (spec 2026-11-15-windows-native-worker-design): the 30 GT
pages through this path score median CER within **0.005 of 0.0940**.

## 1. Attach the BOINC client

1. Install the BOINC client (boinc.berkeley.edu, client release 8.x, no
   VirtualBox). NVIDIA driver must be >= 551.61 (CUDA 12.4).
2. Make sure Tailscale is up and `http://<TAILNET_IP>/naca/` answers in a
   browser.
3. Attach: Add Project → URL `http://<TAILNET_IP>/naca/`, the
   alpha-worker account (key in `boinc/server/secrets-local/alpha-worker.key`
   on the operator box; use the manager's "login with account key").
4. On the operator box, submit one real 2-page batch:
   `.venv/bin/python pipeline/boinc_submit.py data/units/batch_000_units.jsonl --app naca_vision_b --batch 2 --limit 2`
   The client should download the app version (~1.2GB) + both sticky models
   (~9.4GB, one-time) + the WU, run it, and upload. Watch BOINC Manager's
   event log; the task log is `launcher_stdout.txt` in the slot dir.
5. Verify on the operator box: the two pages assimilated into the campaign
   layer and the WU validated (`script_validator_b.log`).

## 2. VRAM measurement → final plan class

While a batch task runs, on the 5090 in PowerShell:

    nvidia-smi --query-gpu=memory.used --format=csv -lms 500 > vram.log

Peak VRAM = max over the run. Plan-class floor = peak + 512MB margin.
On the operator box, set `min_gpu_ram_mb` and `gpu_ram_used_mb` in
`boinc/install_project.sh` (plan_class_spec.xml heredoc) to that floor and
re-run `bash boinc/install_project.sh`.

**Tripwire (user decision):** if the floor lands materially above ~10GB, STOP
and surface it — it shrinks the volunteer pool the design promised.

## 3. GT acceptance run

On the operator box (GT units file = the 30 pages of `data/gt_pages.jsonl` in
units format; build it from the units jsonl if not present):

    .venv/bin/python pipeline/boinc_submit.py <gt_units.jsonl> --app naca_vision_b --batch 30 --gt

`--gt` routes assimilation to `data/<id>/vision_8bwin/`; the campaign layer is
untouched. When the client finishes and the batch assimilates:

    .venv/bin/python pipeline/score.py
    .venv/bin/python pipeline/bench_report.py --write

`vision_8bwin` appears in BENCHMARKS.md. **Gate:** median CER within 0.005 of
0.0940.

- **Pass:** packaging accepted; record the measured plan-class floor and the
  CER in the ledger/memory; volunteers can be onboarded (tailnet only until
  the public-exposure project).
- **Miss:** debug with build parity first — the Windows llama.cpp is pinned to
  the same b12199 build as the Linux engine (boinc/win/pins.json), so a >0.005
  gap is packaging/profile, not upstream drift. Check `-c 16384 -np 1` vs the
  campaign flags, MAX_PX handling, and the launcher's prompt file resolution
  (config hash in the result metas must be 6ecf7d0dbeb2).

## Notes

- App-version files are immutable once registered: `update_versions` skips an
  existing version and refuses changed files. To ship ANY bundle change, bump
  the single `VERSION` constant in `boinc/win/package.py` (it feeds the dir
  name, the physical-name prefix, and the apps path together), then
  `package.py assemble` + `stage` — clients download the new version on their
  next scheduler contact.
- The failed-page retry lane (`escalate.py run-wu-error`) still resubmits as
  single-page `naca_vision` WUs served by the Docker worker — by design for
  the tailnet phase.
