import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "worker" / "win"))
import launcher  # noqa: E402

STUB = str(ROOT / "tests" / "stub_llama_server.py")
# a wrapper the launcher can exec as the "server binary"
STUB_BIN = [sys.executable, STUB]


def png(tag: bytes = b""):
    """Minimal PNG that passes unit_worker.png_size (magic, IHDR, IEND)."""
    import struct
    import zlib

    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", 8, 6, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"tEXt", b"tag\x00" + tag) + chunk(b"IEND", b""))


def make_workdir(tmp_path, n=3):
    units = []
    for i in range(n):
        img = png(str(i).encode())
        (tmp_path / f"page_{i:03d}.png").write_bytes(img)
        units.append({"unit_id": f"19930001-p{i}", "ntrs_id": "19930001", "page": i,
                      "image_key": f"k{i}", "image_sha256": hashlib.sha256(img).hexdigest(),
                      "model_id": "m", "prompt_sha": "p", "config_hash": "6ecf7d0dbeb2"})
    (tmp_path / "manifest.json").write_text(json.dumps(units))
    # model soft-links (content irrelevant to the stub)
    for name in ("model.gguf", "mmproj.gguf"):
        (tmp_path / name).write_bytes(b"gguf")
    return units


def run_launcher(tmp_path, env_extra=None):
    env = {**os.environ, "NACA_SERVER_BIN": json.dumps(STUB_BIN),
           "NACA_HEALTH_DEADLINE": "30", **(env_extra or {})}
    old = dict(os.environ)
    os.environ.update(env)
    try:
        return launcher.main(str(tmp_path))
    finally:
        os.environ.clear()
        os.environ.update(old)


def read_results(tmp_path):
    return [json.loads(l) for l in (tmp_path / "results.jsonl").read_text().splitlines()]


def test_resolve_softlink(tmp_path):
    plain = tmp_path / "a.txt"
    plain.write_text("x")
    assert launcher.resolve(str(plain)) == str(plain)
    link = tmp_path / "b.txt"
    link.write_text("<soft_link>/abs/target</soft_link>\n")
    assert launcher.resolve(str(link)) == "/abs/target"


def test_launcher_full_batch_writes_results_and_meta(tmp_path):
    units = make_workdir(tmp_path, 3)
    assert run_launcher(tmp_path) == 0
    rows = read_results(tmp_path)
    assert [r["unit_id"] for r in rows] == [u["unit_id"] for u in units]
    assert all(r["status"] == "ok" and r["text"] and
               r["meta"]["config_hash"] == "6ecf7d0dbeb2" for r in rows)
    bm = json.loads((tmp_path / "batch.meta.json").read_text())
    assert bm["pages"] == 3 and bm["ok"] == 3 and bm["errors"] == 0
    assert (tmp_path / "boinc_finish_called").read_text().startswith("0")


def test_launcher_resumes_from_checkpoint(tmp_path):
    units = make_workdir(tmp_path, 3)
    done = {"unit_id": units[0]["unit_id"], "status": "ok", "text": "already here",
            "meta": {"config_hash": "6ecf7d0dbeb2"}}
    (tmp_path / "checkpoint.jsonl").write_text(json.dumps(done) + "\n")
    log = tmp_path / "stublog"
    assert run_launcher(tmp_path, {"STUB_LOG": str(log)}) == 0
    assert len(set(log.read_text().split())) == 2  # only the 2 remaining pages hit the server
    rows = read_results(tmp_path)
    assert len(rows) == 3 and rows[0]["text"] == "already here"


def test_launcher_server_start_failure_is_infra_exit(tmp_path):
    make_workdir(tmp_path, 2)
    rc = run_launcher(tmp_path, {"NACA_SERVER_BIN": json.dumps(["/bin/false"]),
                                 "NACA_HEALTH_DEADLINE": "3"})
    assert rc == 1
    assert not (tmp_path / "results.jsonl").exists()


def test_launcher_page_failure_recorded_not_fatal(tmp_path):
    make_workdir(tmp_path, 3)
    assert run_launcher(tmp_path, {"STUB_FAIL_NTH_UNIQUE": "2"}) == 0
    rows = read_results(tmp_path)
    assert rows[1]["status"] == "error" and "server_http_404" in rows[1]["error"]
    assert rows[0]["status"] == rows[2]["status"] == "ok"
    bm = json.loads((tmp_path / "batch.meta.json").read_text())
    assert bm["ok"] == 2 and bm["errors"] == 1


def test_truncated_checkpoint_line_tolerated(tmp_path):
    units = make_workdir(tmp_path, 3)
    done = {"unit_id": units[0]["unit_id"], "status": "ok", "text": "kept",
            "meta": {"config_hash": "6ecf7d0dbeb2"}}
    # preemption mid-append: a complete line, then a truncated one
    (tmp_path / "checkpoint.jsonl").write_text(
        json.dumps(done) + "\n" + '{"unit_id": "19930001-p1", "status": "o')
    assert run_launcher(tmp_path) == 0
    rows = read_results(tmp_path)
    assert len(rows) == 3 and rows[0]["text"] == "kept"
    assert rows[1]["status"] == "ok"  # truncated page was redone


def test_server_death_mid_batch_is_infra_exit(tmp_path):
    make_workdir(tmp_path, 3)
    rc = run_launcher(tmp_path, {"STUB_DIE_AFTER": "1"})
    assert rc == 1
    assert not (tmp_path / "results.jsonl").exists()


def test_complete_checkpoint_needs_no_server(tmp_path):
    units = make_workdir(tmp_path, 2)
    lines = [json.dumps({"unit_id": u["unit_id"], "status": "ok", "text": "t",
                         "meta": {"config_hash": "6ecf7d0dbeb2"}}) for u in units]
    (tmp_path / "checkpoint.jsonl").write_text("\n".join(lines) + "\n")
    rc = run_launcher(tmp_path, {"NACA_SERVER_BIN": json.dumps(["/bin/false"])})
    assert rc == 0 and len(read_results(tmp_path)) == 2


def test_non_ascii_transcription_survives_ascii_locale(tmp_path):
    # Windows embeddable Python defaults text I/O to cp1252; LC_ALL=C (with
    # coercion and UTF-8 mode disabled) reproduces that hostile default here.
    # Greek/math characters on a page must not crash the worker or launcher.
    import subprocess
    units = make_workdir(tmp_path, 2)
    env = {**os.environ, "NACA_SERVER_BIN": json.dumps(STUB_BIN),
           "NACA_HEALTH_DEADLINE": "30", "STUB_NONASCII": "1",
           "LC_ALL": "C", "LANG": "C",
           "PYTHONCOERCECLOCALE": "0", "PYTHONUTF8": "0"}
    r = subprocess.run([sys.executable, str(ROOT / "worker" / "win" / "launcher.py"),
                        str(tmp_path)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    rows = read_results(tmp_path)
    assert [row["status"] for row in rows] == ["ok", "ok"], rows
    assert "α" in rows[0]["text"]
