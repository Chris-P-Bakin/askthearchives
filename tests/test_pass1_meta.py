import json
import subprocess

import common
import pass1

ROW = {"ntrs_id": "42", "report_number": "TN-42"}


def _setup(tmp_path, monkeypatch, meta):
    monkeypatch.setattr(common, "DATA", tmp_path)
    d = tmp_path / "42"; d.mkdir()
    (d / "source.pdf").write_bytes(b"%PDF fake")
    (d / "meta.json").write_text(json.dumps(meta))
    return d


def _events(name):
    return [json.loads(l) for l in common.LOG.read_text().splitlines()
            if json.loads(l).get("event") == name]


def test_rerun_preserves_first_run_timings(tmp_path, monkeypatch):
    first = {"timings": {"rasterize_s": 99.0, "baseline_s": 5.0, "pages": 3}, "deskewed": 2}
    d = _setup(tmp_path, monkeypatch, dict(first))
    monkeypatch.setattr(pass1, "rasterize", lambda i: (3, {}))   # all pages skipped
    monkeypatch.setattr(pass1, "baseline", lambda i, n, w: ({"baseline_absent": False, "pages": {}}, 0))
    assert pass1.process_report((ROW, set()))[1] == "ok"
    meta = json.loads((d / "meta.json").read_text())
    assert meta["timings"] == first["timings"]
    assert meta["deskewed"] == 2
    assert _events("report_done")[-1]["deskewed"] == 2


def test_first_run_writes_timings_and_deskewed(tmp_path, monkeypatch):
    d = _setup(tmp_path, monkeypatch, {})
    monkeypatch.setattr(pass1, "rasterize", lambda i: (2, {1: 0.5, 2: 0.0}))
    monkeypatch.setattr(pass1, "baseline", lambda i, n, w: ({"baseline_absent": False, "pages": {}}, 0))
    pass1.process_report((ROW, set()))
    meta = json.loads((d / "meta.json").read_text())
    assert meta["deskewed"] == 1 and meta["timings"]["pages"] == 2


def test_raster_timeout_marks_retryable(tmp_path, monkeypatch):
    d = _setup(tmp_path, monkeypatch, {"sha256": "x"})
    monkeypatch.setenv("NACA_RASTER_TIMEOUT", "5")
    calls = []

    def fake_run(cmd, **kw):
        if cmd[0] == "pdfinfo":
            return subprocess.CompletedProcess(cmd, 0, stdout="Pages: 3\n")
        calls.append(kw.get("timeout"))
        (d / "_raster").mkdir(exist_ok=True)
        (d / "_raster" / "p-1.png").write_bytes(b"partial")
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))
    monkeypatch.setattr(pass1.subprocess, "run", fake_run)
    assert pass1.process_report((ROW, set()))[1] == "error"
    assert calls == [5]
    assert _events("report_failed")[-1]["reason"] == "raster_timeout"
    assert not (d / "pages").exists() and not (d / "_raster").exists()
    assert json.loads((d / "meta.json").read_text())["pdf_status"] == "raster_timeout"
    # next run retries rather than skipping
    assert pass1.process_report((ROW, set()))[1] == "error"
    assert calls == [5, 5]
