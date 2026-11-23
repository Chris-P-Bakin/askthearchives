import io
import json
import subprocess

import pytest

import obs_ship
from ch_client import CHUnavailable


class FakeCH:
    def __init__(self):
        self.inserts = []      # (table, rows)
        self.fail = 0

    def insert(self, table, rows):
        if self.fail:
            self.fail -= 1
            raise CHUnavailable("down")
        self.inserts.append((table, list(rows)))

    def rows(self, table):
        return [r for t, rs in self.inserts if t == table for r in rs]


def _w(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(obj if isinstance(obj, str) else json.dumps(obj))


def _fake_run(out_by_cmd):
    def run(cmd, **kw):
        key = "docker" if cmd[0] == "docker" else "nvidia"
        o = out_by_cmd[key]
        if isinstance(o, Exception):
            raise o
        return subprocess.CompletedProcess(cmd, 0, stdout=o, stderr="")
    return run


OK_RUN = {"docker": "10\t8\t1\t3\t7\n", "nvidia": "18275, 24576\n"}


@pytest.fixture
def data(tmp_path):
    d = tmp_path / "data"
    _w(d / "100" / "vision_8blora_v2_1" / "1.meta.json", {})
    _w(d / "100" / "vision_8blora_v2_1" / "2.meta.json", {})
    _w(d / "100" / "vision_235b" / "1.meta.json", {"usage": {"cost": 1.25}})
    _w(d / "100" / "pages" / "1.png", "x")
    _w(d / "escalation_queue.jsonl",
       json.dumps({"unit_id": "100-p1", "kind": "table_flags", "ntrs_id": "100", "page": 1}) + "\n")
    _w(d / "review_queue.jsonl", "a\nb\n")
    _w(d / "batch1_ingest.log", "[t] report_done x=1\n[t] pass3a_start pages=3\n")
    return d


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("ARCHIVES_SPOOL", str(tmp_path / "spool.ndjson"))
    monkeypatch.delenv("ARCHIVES_CF_METRICS_URL", raising=False)


def _no_cf(monkeypatch, body=""):
    monkeypatch.setattr(obs_ship.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(body.encode()))


def _ship(data, run=None, ch=None, clock=lambda: 1000.0):
    ch = ch or FakeCH()
    return obs_ship.Shipper(ch, data=data, run=run or _fake_run(OK_RUN), clock=clock), ch


def _by(samples):
    return {(n, tuple(sorted(l.items()))): v for n, l, v in samples}


def _sample_rows(ch):
    return [(r["name"], r["labels"], r["value"]) for r in ch.rows("archives.samples")]


def test_values_ported(data, monkeypatch):
    _no_cf(monkeypatch)
    s, ch = _ship(data)
    s.run_once()
    got = _by(_sample_rows(ch))
    assert got[("naca_wus_total", ())] == 10
    assert got[("naca_wus_valid", ())] == 8
    assert got[("naca_wus_invalid", ())] == 1
    assert got[("naca_wus_inflight", ())] == 3
    assert got[("naca_pages_assimilated", ())] == 7
    assert got[("naca_gpu_mem_mb", (("kind", "used"),))] == 18275
    assert got[("naca_gpu_mem_mb", (("kind", "total"),))] == 24576
    assert got[("naca_235b_spend_usd", ())] == 1.25
    assert got[("naca_queue_pending", (("lane", "235b"),))] == 1
    assert got[("naca_queue_pending", (("lane", "27b"),))] == 0
    assert got[("naca_review_rows", (("queue", "pass5"),))] == 2
    assert got[("naca_pages_meta", (("layer", "vision_8blora_v2_1"),))] == 2
    assert got[("naca_ingest_batch", (("batch", "1"), ("state", "pass3a_start")))] == 1
    assert all(r["ts"] == 1000.0 for r in ch.rows("archives.samples"))


def test_all_sources_down_emit_errors_and_loop_survives(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("refused")
    monkeypatch.setattr(obs_ship.urllib.request, "urlopen", boom)
    s, ch = _ship(tmp_path / "nodata", _fake_run({"docker": subprocess.TimeoutExpired("d", 1),
                                                  "nvidia": FileNotFoundError("x")}))
    s.run_once()
    got = _by(_sample_rows(ch))
    for src in ("boinc", "gpu", "cloudflared"):
        assert got[("naca_collector_errors", (("source", src),))] == 1
    assert not any(n == "naca_wus_total" for n, _, _ in _sample_rows(ch))


def test_error_counter_cumulative(data, monkeypatch):
    _no_cf(monkeypatch)
    s, ch = _ship(data, _fake_run({"docker": "garbage", "nvidia": "x"}))
    s.run_once()
    s.run_once()
    vals = [v for n, l, v in _sample_rows(ch)
            if n == "naca_collector_errors" and l == {"source": "boinc"}]
    assert vals == [1, 2]


def test_no_secret_values_in_output(data, monkeypatch):
    _no_cf(monkeypatch)
    monkeypatch.setenv("DB_PASSWD", "sk-hunter2")
    s, ch = _ship(data, _fake_run({"docker": "password: sk-x cfat_y", "nvidia": "sk-q"}))
    s.run_once()
    t = json.dumps(ch.inserts).lower()
    for bad in ("sk-", "cfat_", "password", "hunter2"):
        assert bad not in t


PROM = """\
# HELP cloudflared_tunnel_total_requests Total
# TYPE cloudflared_tunnel_total_requests counter
cloudflared_tunnel_total_requests 42
# TYPE cloudflared_tunnel_response_by_code counter
cloudflared_tunnel_response_by_code{status_code="200"} 40
cloudflared_tunnel_response_by_code{status_code="502"} 2
# TYPE cloudflared_muxer_rtt gauge
cloudflared_muxer_rtt 5
# TYPE cloudflared_muxer_writes counter
cloudflared_muxer_writes{connection_id="0"} 7
# TYPE go_goroutines gauge
go_goroutines 12
# TYPE process_cpu_seconds_total counter
process_cpu_seconds_total 3.5
# TYPE cloudflared_other_thing counter
cloudflared_other_thing 1
"""


def test_cloudflared_parse_whitelist(data, monkeypatch):
    seen = []

    def fake(req, *a, **k):
        seen.append(getattr(req, "full_url", req))
        return io.BytesIO(PROM.encode())
    monkeypatch.setattr(obs_ship.urllib.request, "urlopen", fake)
    s, _ = _ship(data)
    out = s._cloudflared()
    assert seen == ["http://127.0.0.1:2000/metrics"]
    assert _by(out) == {
        ("cloudflared_tunnel_total_requests", ()): 42,
        ("cloudflared_tunnel_response_by_code", (("status_code", "200"),)): 40,
        ("cloudflared_tunnel_response_by_code", (("status_code", "502"),)): 2,
        ("cloudflared_muxer_writes", (("connection_id", "0"),)): 7,
    }


def test_cloudflared_source_fails_independently(data, monkeypatch):
    def boom(*a, **k):
        raise OSError("refused")
    monkeypatch.setattr(obs_ship.urllib.request, "urlopen", boom)
    s, ch = _ship(data)
    s.run_once()
    names = {(r["name"], tuple(r["labels"].items())) for r in ch.rows("archives.samples")}
    assert ("naca_wus_total", ()) in names
    assert ("naca_gpu_mem_mb", (("kind", "used"),)) in names
    assert ("naca_collector_errors", (("source", "cloudflared"),)) in names


def test_grind_mb_accumulates_across_calls(data):
    log = data / "campaign_grind.log"
    _w(log, "x mb=1.5 y\nnoise\nz mb=2 w\n")
    s, _ = _ship(data)
    assert s._grind() == [("naca_ingest_downloaded_mb_total", {}, 3.5)]
    assert s._grind() == [("naca_ingest_downloaded_mb_total", {}, 3.5)]
    with open(log, "a") as f:
        f.write("a mb=0.5\npartial mb=9")      # partial line not consumed yet
    assert s._grind()[0][2] == 4.0
    with open(log, "a") as f:
        f.write("\n")
    assert s._grind()[0][2] == 13.0


def test_sample_ts_is_integer(data):
    s, ch = _ship(data, clock=lambda: 1000.75)
    s.add_samples([("n", {}, 1)])
    assert s.buf_samples[0]["ts"] == 1000 and isinstance(s.buf_samples[0]["ts"], int)


def test_grind_total_survives_restart(data):
    log = data / "campaign_grind.log"
    _w(log, "x mb=1.5\nz mb=2\n")
    s, _ = _ship(data)
    assert s._grind()[0][2] == 3.5
    s2, _ = _ship(data)
    assert s2._grind()[0][2] == 3.5
    with open(log, "a") as f:
        f.write("a mb=0.5\n")
    assert s2._grind()[0][2] == 4.0
    s3, _ = _ship(data)
    assert s3._grind()[0][2] == 4.0


def test_grind_old_format_offset_file(data):
    log = data / "campaign_grind.log"
    _w(log, "x mb=1\n")
    (data / "campaign_grind.offset").write_text("6")
    s, _ = _ship(data)
    assert s._grind()[0][2] == 0.0


def test_grind_missing_log_is_not_an_error(data):
    s, _ = _ship(data)
    assert s._grind() == [("naca_ingest_downloaded_mb_total", {}, 0.0)]


def test_buffer_cap_drops_oldest_and_recovers(data):
    ch = FakeCH()
    s, _ = _ship(data, ch=ch)
    ch.fail = 2
    for i in range(obs_ship.BUFFER_CAP + 500):
        s.add_samples([("n", {"i": str(i)}, i)])
    assert len(s.buf_samples) == obs_ship.BUFFER_CAP
    assert s.buf_samples[0]["value"] == 500            # oldest dropped first
    s.flush()
    s.flush()
    assert len(s.buf_samples) == obs_ship.BUFFER_CAP    # bounded, retained across failures
    s.flush()
    assert len(s.buf_samples) == 0
    assert len(ch.rows("archives.samples")) == obs_ship.BUFFER_CAP


def test_requests_rows_match_spool_events(data, tmp_path):
    ev = {"ts": 1700000000.123, "tool": "search", "duration_ms": 12, "bytes": 300,
          "results": 5, "status": "ok", "degraded": 0}
    _w(tmp_path / "spool.ndjson", json.dumps(ev) + "\n")
    s, ch = _ship(data)
    s.drain_spool()
    s.flush()
    # ts is shipped as a UTC string: DateTime64 rejects fractional unix floats.
    want = dict(ev, ts="2023-11-14 22:13:20.123")
    assert ch.rows("archives.requests") == [want]
    s.drain_spool()
    s.flush()
    assert ch.rows("archives.requests") == [want]     # offset persisted, no duplicates


def test_requests_kept_on_insert_failure(data, tmp_path):
    ev = {"ts": 1.0, "tool": "t", "duration_ms": 1, "bytes": 1, "results": 0, "status": "ok", "degraded": 0}
    _w(tmp_path / "spool.ndjson", json.dumps(ev) + "\n")
    ch = FakeCH()
    s, _ = _ship(data, ch=ch)
    ch.fail = 1
    s.drain_spool()
    s.flush()
    assert ch.rows("archives.requests") == []
    s.flush()
    assert ch.rows("archives.requests") == [dict(ev, ts="1970-01-01 00:00:01.000")]


def test_cadence(data, monkeypatch):
    _no_cf(monkeypatch)
    s, ch = _ship(data)
    s.cycle(now=0.0)                       # everything due on first cycle
    n0 = len(ch.rows("archives.samples"))
    s.cycle(now=10.0)                      # nothing due
    assert len(ch.rows("archives.samples")) == n0
    s.cycle(now=30.0)                      # boinc/gpu/cloudflared/grind only
    new = {r["name"] for r in ch.rows("archives.samples")[n0:]}
    assert "naca_wus_total" in new and "naca_gpu_mem_mb" in new
    assert "naca_queue_pending" not in new and "naca_pages_meta" not in new
    n1 = len(ch.rows("archives.samples"))
    s.cycle(now=60.0)
    assert "naca_queue_pending" in {r["name"] for r in ch.rows("archives.samples")[n1:]}


def test_once_cycle_exit_zero(data, monkeypatch, tmp_path):
    _no_cf(monkeypatch)
    ch = FakeCH()
    monkeypatch.setattr(obs_ship, "CHClient", lambda *a, **k: ch)
    monkeypatch.setattr(obs_ship, "DATA_DIR", data)
    monkeypatch.setattr(obs_ship.subprocess, "run", _fake_run(OK_RUN))
    _w(tmp_path / "spool.ndjson", json.dumps({"ts": 1.0, "tool": "t", "duration_ms": 1, "bytes": 1,
                                              "results": 0, "status": "ok", "degraded": 0}) + "\n")
    assert obs_ship.main(["--once"]) == 0
    assert ch.rows("archives.samples") and len(ch.rows("archives.requests")) == 1
