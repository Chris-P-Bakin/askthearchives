#!/usr/bin/env python3
"""Collector that ships campaign + serving telemetry to ClickHouse (stdlib only).

Gathers BOINC DB counts (docker exec), escalation queue/spend, per-layer meta-file counts,
GPU memory, batch ingest state, cloudflared tunnel counters and grind download volume into
`archives.samples`, and drains the MCP request spool into `archives.requests`.
Every source degrades independently: a failure emits naca_collector_errors{source} and the
loop carries on. ClickHouse outages are absorbed by bounded in-memory buffers (oldest dropped).
Run on the host (needs docker + the venv): `python pipeline/obs_ship.py [--once]`."""
import argparse
import os
import re
import subprocess
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path

import common
import escalate
import units
from ch_client import CHClient, CHUnavailable
from obs_spool import read_spool

DATA_DIR = common.DATA
BUFFER_CAP = 10_000
SOURCES = ("boinc", "queue", "tree", "gpu", "ingest", "cloudflared", "grind")
REFRESH_S = {"boinc": 30, "gpu": 30, "cloudflared": 30, "grind": 30,
             "ingest": 30, "queue": 60, "tree": 120}
SPOOL_S = 10
LAYERS = (units.CAMPAIGN_LAYER, escalate.LAYER_27B, escalate.LAYER_235B)
CF_PREFIXES = ("cloudflared_tunnel_", "cloudflared_muxer_")
REQUEST_FIELDS = ("ts", "tool", "duration_ms", "bytes", "results", "status", "degraded")
BOINC_Q = ("select (select count(*) from workunit),"
           "(select count(*) from result where validate_state=1),"
           "(select count(*) from result where validate_state=2),"
           "(select count(*) from result where server_state=4),"
           "(select count(*) from workunit where assimilate_state=2)")
_PROM_LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)')
_PROM_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')
_MB = re.compile(r"mb=([0-9.]+)")


class Shipper:
    def __init__(self, ch, data=None, run=subprocess.run, clock=time.time):
        self.ch = ch
        self.data = Path(data) if data is not None else DATA_DIR
        self.run, self.clock = run, clock
        self.spool = Path(os.environ.get("ARCHIVES_SPOOL", "data/mcp_requests.ndjson"))
        self.cf_url = os.environ.get("ARCHIVES_CF_METRICS_URL", "http://127.0.0.1:2000/metrics")
        self.errors = {s: 0 for s in SOURCES}
        self.grind_mb = 0.0
        self.buf_samples = deque(maxlen=BUFFER_CAP)
        self.buf_requests = deque(maxlen=BUFFER_CAP)
        self._due = {s: 0.0 for s in SOURCES}
        self._spool_due = 0.0

    # -- sources: each returns a list of (name, labels, value) -------------------
    def _boinc(self):
        cmd = ["docker", "exec", "naca-boinc-apache", "bash", "-c",
               'source /run/secrets/secrets.env; mysql -h mysql -uroot -p$DB_PASSWD naca -N -e "'
               + BOINC_Q + '"']
        p = self.run(cmd, capture_output=True, text=True, timeout=5)
        if p.returncode != 0:
            raise RuntimeError("rc")
        v = [int(x) for x in p.stdout.split()]
        if len(v) != 5:
            raise ValueError("shape")
        names = ("naca_wus_total", "naca_wus_valid", "naca_wus_invalid",
                 "naca_wus_inflight", "naca_pages_assimilated")
        return [(n, {}, x) for n, x in zip(names, v)]

    def _gpu(self):
        p = self.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
        if p.returncode != 0:
            raise RuntimeError("rc")
        used = total = 0
        rows = [l for l in p.stdout.splitlines() if l.strip()]
        if not rows:
            raise ValueError("empty")
        for l in rows:
            u, t = (int(x) for x in l.split(","))
            used, total = used + u, total + t
        return [("naca_gpu_mem_mb", {"kind": "used"}, used),
                ("naca_gpu_mem_mb", {"kind": "total"}, total)]

    def _queue(self):
        data = self.data
        rows, _ = escalate.read_queue_ex(data / "escalation_queue.jsonl")
        routed = escalate.route(rows, data)
        out = []
        for lane, lrows in routed.items():
            if lane == "27b":
                lrows = escalate.pending(lrows, escalate.LAYER_27B, escalate.ESC_27B_HASH, data=data)
            if lane == "wu_error":
                lrows = [r for r in lrows if escalate.wu_error_open(r, data)]
            else:
                lrows = [r for r in lrows if not escalate.is_done(r, data)]
            out.append(("naca_queue_pending", {"lane": lane}, len(lrows)))
        out.append(("naca_235b_spend_usd", {}, round(escalate.spend(data), 6)))
        for q, f in (("pass5", "review_queue.jsonl"), ("escalation", "escalation_review.jsonl")):
            p = data / f
            n = sum(1 for l in p.open("rb") if l.strip()) if p.exists() else 0
            out.append(("naca_review_rows", {"queue": q}, n))
        return out

    def _tree(self):
        counts = {l: 0 for l in LAYERS}
        with os.scandir(self.data) as it:
            dirs = [e.path for e in it if e.is_dir(follow_symlinks=False) and e.name.isdigit()]
        for d in dirs:
            for layer in LAYERS:
                try:
                    with os.scandir(os.path.join(d, layer)) as it:
                        counts[layer] += sum(1 for e in it if e.name.endswith(".meta.json"))
                except OSError:
                    pass
        return [("naca_pages_meta", {"layer": l}, n) for l, n in counts.items()]

    def _ingest(self):
        out = []
        for p in sorted(self.data.glob("batch[0-9]*_ingest.log")):
            m = re.match(r"batch(\d+)_ingest\.log$", p.name)
            with open(p, "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - 4096))
                tail = f.read().decode("utf-8", "replace").splitlines()
            last = next((l for l in reversed(tail) if l.strip()), "")
            ev = re.match(r"\[[^\]]*\]\s+(\w+)", last)
            if not (m and ev):
                raise ValueError("unparsable log tail")
            out.append(("naca_ingest_batch", {"batch": m.group(1), "state": ev.group(1)}, 1))
        return out

    def _cloudflared(self):
        with urllib.request.urlopen(self.cf_url, timeout=5) as r:
            text = r.read().decode("utf-8", "replace")
        counters = set()
        out = []
        for line in text.splitlines():
            if line.startswith("# TYPE "):
                parts = line.split()
                if len(parts) >= 4 and parts[3] == "counter":
                    counters.add(parts[2])
                continue
            if not line or line.startswith("#"):
                continue
            m = _PROM_LINE.match(line)
            if not m or m.group(1) not in counters or not m.group(1).startswith(CF_PREFIXES):
                continue
            labels = {k: v.replace('\\"', '"').replace("\\\\", "\\")
                      for k, v in _PROM_LABEL.findall(m.group(2) or "")}
            out.append((m.group(1), labels, float(m.group(3))))
        return out

    def _grind(self):
        log = self.data / "campaign_grind.log"
        off_file = self.data / "campaign_grind.offset"
        offset = 0
        try:
            parts = off_file.read_text().split()
            offset = int(parts[0])
            self.grind_mb = float(parts[1]) if len(parts) > 1 else self.grind_mb
        except (OSError, ValueError, IndexError):
            pass
        try:
            size = log.stat().st_size
            if offset < 0 or offset > size:      # log truncated/rotated
                offset = 0
            with log.open("rb") as f:
                f.seek(offset)
                chunk = f.read()
        except OSError:
            return [("naca_ingest_downloaded_mb_total", {}, self.grind_mb)]
        end = chunk.rfind(b"\n") + 1             # complete lines only
        for m in _MB.finditer(chunk[:end].decode("utf-8", "replace")):
            try:
                self.grind_mb += float(m.group(1))
            except ValueError:
                pass
        if end:
            tmp = off_file.with_name(off_file.name + ".tmp")
            tmp.write_text(f"{offset + end} {self.grind_mb!r}")
            os.replace(tmp, off_file)
        return [("naca_ingest_downloaded_mb_total", {}, self.grind_mb)]

    # -- buffering ---------------------------------------------------------------
    def add_samples(self, samples, ts=None):
        ts = self.clock() if ts is None else ts
        for n, labels, v in samples:
            self.buf_samples.append({"ts": int(ts), "name": n, "labels": labels, "value": float(v)})

    def gather(self, name):
        try:
            res = getattr(self, "_" + name)()
        except Exception as e:
            print(f"source {name} failed: {type(e).__name__}", file=sys.stderr, flush=True)
            self.errors[name] += 1
            res = [("naca_collector_errors", {"source": name}, self.errors[name])]
        self.add_samples(res)

    def drain_spool(self):
        try:
            events = read_spool(self.spool, self.spool.with_name(self.spool.name + ".offset"))
        except Exception as e:
            print(f"spool failed: {type(e).__name__}", file=sys.stderr, flush=True)
            return
        for ev in events:
            row = {k: ev[k] for k in REQUEST_FIELDS if k in ev}
            # ClickHouse (JSONEachRow, basic input format) rejects fractional
            # unix floats for DateTime64; send a UTC string it parses exactly.
            if isinstance(row.get("ts"), (int, float)):
                t = float(row["ts"])
                row["ts"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t)) \
                    + f".{int(t * 1000) % 1000:03d}"
            self.buf_requests.append(row)

    def _flush_one(self, table, buf):
        if not buf:
            return
        rows = list(buf)
        try:
            self.ch.insert(table, rows)
        except CHUnavailable as e:
            print(f"insert {table} failed ({len(rows)} rows kept): {e}", file=sys.stderr, flush=True)
            return
        for _ in rows:
            buf.popleft()

    def flush(self):
        self._flush_one("archives.samples", self.buf_samples)
        self._flush_one("archives.requests", self.buf_requests)

    # -- scheduling --------------------------------------------------------------
    def run_once(self):
        for s in SOURCES:
            self.gather(s)
        self.drain_spool()
        self.flush()

    def cycle(self, now=None):
        now = time.monotonic() if now is None else now
        for s in SOURCES:
            if now >= self._due[s]:
                self.gather(s)
                self._due[s] = now + REFRESH_S[s]
        if now >= self._spool_due:
            self.drain_spool()
            self._spool_due = now + SPOOL_S
        self.flush()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--once", action="store_true", help="one gather+drain+ship cycle, then exit")
    args = ap.parse_args(argv)
    ch = CHClient(os.environ.get("ARCHIVES_CH_URL", "http://127.0.0.1:8123"),
                  os.environ.get("ARCHIVES_CH_USER", "default"),
                  os.environ.get("ARCHIVES_CH_PASSWORD", ""))
    s = Shipper(ch, data=DATA_DIR, run=subprocess.run)
    if args.once:
        s.run_once()
        return 0
    print("obs_ship running", flush=True)
    while True:
        try:
            s.cycle()
        except Exception as e:
            print(f"cycle failed: {type(e).__name__}", file=sys.stderr, flush=True)
        time.sleep(1.0)


if __name__ == "__main__":
    sys.exit(main())
