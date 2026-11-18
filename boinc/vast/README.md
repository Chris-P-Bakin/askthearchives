# Spot-GPU workers (vast.ai)

Rent interruptible 3090-class instances and attach them to the campaign as
ordinary BOINC workers. Work units are page-granular, checkpointed and
reissue-safe, so spot eviction only costs the in-flight pages.

## Prerequisites (one-time)

- `boinc.askthearchives.org` public hostname on the Cloudflare tunnel →
  `http://<TAILNET_IP>:80`, and the project's `config.xml` URLs migrated to
  the public hostname (see `deploy/README.md`).
- Models published at `data.askthearchives.org/naca/models/` (done 2026-11-18).
- A dedicated BOINC account for spot workers (revocable without touching the
  trusted workers): on the server,
  `docker exec naca-boinc-apache bash -c 'cd /home/boincadm/project && bin/make_user ...'`
  and keep the authenticator in `boinc/server/secrets-local/spot-worker.key`.
  Spot hosts can read instance env, so never give them the alpha key.

## Launch

Pick 24 GB interruptible offers (floor ≈ $0.09–0.15/hr for a 3090) and create
with the pinned engine image, ~32 GB disk, and the bootstrap script:

```bash
vastai search offers 'gpu_name=RTX_3090 num_gpus=1 inet_down>200' --interruptible -o 'dph'
vastai create instance <OFFER_ID> \
  --image ghcr.io/ggml-org/llama.cpp:server-cuda@sha256:4b554057c7cf0cb87e0724b6de9d7ca9bb24a3b72716af11e4c0fecad62c506b \
  --disk 32 \
  --env '-e BOINC_AUTHENTICATOR=<spot-worker key>' \
  --onstart-cmd 'wget -qO /tmp/onstart.sh https://raw.githubusercontent.com/Chris-P-Bakin/askthearchives/main/boinc/vast/onstart.sh && bash /tmp/onstart.sh'
```

First boot downloads ~10 GB of model, then the worker runs two concurrent
page streams (≈380 pages/hour on a 3090). Progress shows up on the campaign
dashboard (BOINC in-progress counts) within minutes.

## Economics (measured, 2026-11)

3090 at $0.10–0.15/hr ÷ 380 pages/hr ≈ **$0.0003–0.0004 per page** — about
10–15× cheaper than the hosted 27B API ($0.0041/page). The remaining ~485k
campaign vision pages ≈ $130–190 of compute.

## Decommission

`vastai destroy instance <id>`. Unfinished WUs hit their 48 h deadline and
reissue automatically. To revoke all spot workers at once, disable the
spot-worker account server-side.
