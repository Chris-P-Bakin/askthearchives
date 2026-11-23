"""BOINC batch task: transcribe the workunit's pages against a local llama-server.

Runs under the BOINC wrapper (job.xml) in the slot dir, stdlib only, OS-agnostic.
Reads manifest.json (ordered unit records) + page_NNN.png; starts the bundled
llama-server (volunteer profile), runs unit_worker.py per page as a subprocess,
checkpoints each finished page, and writes results.jsonl + batch.meta.json
through BOINC's output soft-links. Exit 0 = batch done (per-page errors are
recorded lines, not failures); exit 1 = infrastructure failure (server never
became healthy) so BOINC reissues the workunit.
"""
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

PROFILE_FLAGS = ["-ngl", "99", "-c", "16384", "-np", "1", "--temp", "0"]
_SOFT = re.compile(r"<soft_link>(.*?)</soft_link>", re.S)


def resolve(path: str) -> str:
    """BOINC pre-creates app/input/output files as <soft_link> XML; follow it."""
    try:
        head = open(path, encoding="utf-8", errors="replace").read(4096)
    except OSError:
        return path
    m = _SOFT.search(head)
    return m.group(1).strip() if m else path


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_healthy(url: str, deadline_s: float, proc) -> bool:
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(1.0)
    return False


def server_cmd(workdir: Path, port: int) -> list[str]:
    """NACA_SERVER_BIN is a JSON argv list (tests) or a path; default bundled exe."""
    bin_env = os.environ.get("NACA_SERVER_BIN", "")
    if bin_env.startswith("["):
        argv = json.loads(bin_env)
    else:
        argv = [resolve(bin_env or str(workdir / "llama-server.exe"))]
    return argv + ["-m", resolve(str(workdir / "model.gguf")),
                   "--mmproj", resolve(str(workdir / "mmproj.gguf")),
                   "--host", "127.0.0.1", "--port", str(port), *PROFILE_FLAGS]


def find_unit_worker(workdir: Path) -> str:
    for cand in (workdir / "unit_worker.py",
                 Path(__file__).resolve().parent / "unit_worker.py",
                 Path(__file__).resolve().parent.parent / "unit_worker.py"):
        if cand.exists():
            return resolve(str(cand))
    raise FileNotFoundError("unit_worker.py not found")


def load_checkpoint(workdir: Path) -> dict:
    done = {}
    cp = workdir / "checkpoint.jsonl"
    if cp.exists():
        for line in cp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue   # preemption mid-append truncated this line: redo the page
            if isinstance(row, dict) and "unit_id" in row and "status" in row:
                done[row["unit_id"]] = row
    return done


def run_page(unit: dict, page_png: str, url: str, worker: str, workdir: Path) -> dict:
    with tempfile.TemporaryDirectory() as td:
        uj = Path(td) / "unit.json"
        uj.write_text(json.dumps(unit), encoding="utf-8")
        env = dict(os.environ, NACA_VLM_URL=url)
        prompt = workdir / "transcribe_v2.txt"
        if prompt.exists():
            env["NACA_PROMPT_FILE"] = resolve(str(prompt))
        r = subprocess.run([sys.executable, worker, str(uj), resolve(page_png), td,
                            "--server", url], env=env,
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        meta_p = Path(td) / "result.meta.json"
        if r.returncode == 0:
            return {"unit_id": unit["unit_id"], "status": "ok",
                    "text": (Path(td) / "result.txt").read_text(encoding="utf-8"),
                    "meta": json.loads(meta_p.read_text(encoding="utf-8"))}
        try:
            reason = json.loads(meta_p.read_text(encoding="utf-8")).get(
                "error", f"worker_rc_{r.returncode}")
        except (OSError, ValueError):
            reason = f"worker_rc_{r.returncode}: {r.stderr.strip()[:2000]}"
        return {"unit_id": unit["unit_id"], "status": "error", "error": reason}


def main(workdir: str = ".") -> int:
    wd = Path(workdir)
    t0 = time.time()
    manifest = json.loads(Path(resolve(str(wd / "manifest.json"))).read_text(encoding="utf-8"))
    done = load_checkpoint(wd)
    todo = [u["unit_id"] for u in manifest if u["unit_id"] not in done]
    cmd = ["(checkpoint complete, server not started)"]
    if todo:
        port = free_port()
        url = f"http://127.0.0.1:{port}"
        cmd = server_cmd(wd, port)
        try:
            proc = subprocess.Popen(cmd, cwd=wd, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
        except OSError as e:
            print(f"launcher: cannot start server {cmd[0]}: {e}", file=sys.stderr)
            return 1
        try:
            deadline = float(os.environ.get("NACA_HEALTH_DEADLINE", "600"))
            if not wait_healthy(url, deadline, proc):
                print("launcher: server never became healthy", file=sys.stderr)
                return 1
            worker = find_unit_worker(wd)
            for i, unit in enumerate(manifest):
                if unit["unit_id"] in done:
                    continue
                row = run_page(unit, str(wd / f"page_{i:03d}.png"), url, worker, wd)
                if proc.poll() is not None:
                    # server died: this (and every later) page would fail for infra
                    # reasons, not page content — reissue instead of mass "errors"
                    print("launcher: server died mid-batch", file=sys.stderr)
                    return 1
                done[unit["unit_id"]] = row
                with open(wd / "checkpoint.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
        finally:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
    rows = [done[u["unit_id"]] for u in manifest]
    ok = sum(r["status"] == "ok" for r in rows)
    out = Path(resolve(str(wd / "results.jsonl")))
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    meta = {"pages": len(rows), "ok": ok, "errors": len(rows) - ok,
            "elapsed_s": round(time.time() - t0, 1),
            "server": {"bin": cmd[0], "flags": PROFILE_FLAGS}}
    Path(resolve(str(wd / "batch.meta.json"))).write_text(json.dumps(meta, indent=1),
                                                          encoding="utf-8")
    (wd / "boinc_finish_called").write_text("0\n\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
