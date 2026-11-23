"""Shared helpers for the BOINC script_validator / script_assimilator hooks.

Runs inside the apache container (python3 stdlib only). The repo's pipeline/
and data/ are bind-mounted at /naca/pipeline (ro) and /naca/data (rw).
"""
import os, re, subprocess, sys

PIPELINE = os.environ.get("NACA_PIPELINE", "/naca/pipeline")
DATA_ROOT = os.environ.get("NACA_DATA_ROOT", "/naca/data")
WU_PREFIX = "naca_"
VAL_RESULT_TRANSIENT_ERROR = 3   # sched/validate_util2.h


def project_root() -> str:
    return os.environ.get("PROJECT_ROOT") or os.path.abspath(os.path.join(os.getcwd(), ".."))


def unit_id_from_output_path(path: str) -> str:
    """Output files are <wu_name>_<resultidx>_r<random>_<fileidx>; wu_name = naca_<unit_id>."""
    wu = re.sub(r"_\d+_r\d+_\d+$", "", os.path.basename(path))
    if not wu.startswith(WU_PREFIX):
        raise ValueError(f"unexpected output file name {path!r}")
    return wu[len(WU_PREFIX):]


def wu_name_from_output_path(path: str) -> str:
    """Full workunit name (prefix intact) from a result output file name."""
    return re.sub(r"_\d+_r\d+_\d+$", "", os.path.basename(path))


def _dir_hier_path(phys_name: str) -> str:
    root = project_root()
    out = subprocess.run([os.path.join(root, "bin", "dir_hier_path"), phys_name],
                         cwd=root, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def run_batch_cli(script: str, wu_name: str, results: str, meta: str, extra=()) -> int:
    """Run a batched-pipeline CLI: <manifest> <results.jsonl> <batch.meta.json> [extra]."""
    manifest = _dir_hier_path(f"{wu_name}.manifest.json")
    for f in (manifest, results, meta):
        if not os.path.isfile(f):
            print(f"missing input {f!r}", file=sys.stderr)
            return CRASH_RC
    for f in (results, meta):
        try:
            with open(f, "rb") as fh:
                fh.read().decode("utf-8")
        except (UnicodeDecodeError, OSError) as e:
            print(f"undecodable result file {f!r}: {e}", file=sys.stderr)
            return 1
    r = subprocess.run([sys.executable, os.path.join(PIPELINE, script),
                        manifest, results, meta, *extra],
                       stderr=subprocess.PIPE, text=True)
    if r.stderr:
        sys.stderr.write(r.stderr)
    return classify_rc(r.returncode, r.stderr)


def unit_json_path(unit_id: str) -> str:
    root = project_root()
    out = subprocess.run([os.path.join(root, "bin", "dir_hier_path"), f"{unit_id}.unit.json"],
                         cwd=root, capture_output=True, text=True, check=True)
    p = out.stdout.strip()
    if os.path.isfile(p):
        return p
    # R2-submitted WUs never stage files into the download hierarchy; fetch
    # the (tiny) unit.json from the CDN into a local cache instead.
    cache = os.path.join(root, "r2_unit_cache")
    os.makedirs(cache, exist_ok=True)
    cp = os.path.join(cache, f"{unit_id}.unit.json")
    if not os.path.isfile(cp):
        base = os.environ.get("ARCHIVES_WU_BASE",
                              "https://data.askthearchives.org/naca/wu")
        import urllib.request
        try:
            req = urllib.request.Request(
                f"{base}/{unit_id}.unit.json",
                headers={"User-Agent": "askthearchives-glue/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r,                  open(cp + ".tmp", "wb") as w:
                w.write(r.read())
            os.replace(cp + ".tmp", cp)
        except Exception:
            return p   # caller reports missing input -> transient, retried
    return cp


CRASH_RC = 99   # "could not evaluate": tooling/infra failure, never a verdict


def classify_rc(rc: int, stderr: str) -> int:
    """Map a pipeline CLI exit to rc. A Python crash also exits 1, but leaves a
    traceback on stderr; a real verdict (valid/invalid) never does."""
    if rc in (0, 1) and "Traceback" not in (stderr or ""):
        return rc
    return CRASH_RC


def run_cli(script: str, unit_id: str, txt: str, meta: str, extra=()) -> int:
    unit = unit_json_path(unit_id)
    for f in (unit, txt, meta):
        if not os.path.isfile(f):
            print(f"missing input {f!r}", file=sys.stderr)
            return CRASH_RC
    # Worker-produced artifacts that cannot decode are a bad RESULT (verdict INVALID: rc 1),
    # not an infra failure; unit.json problems stay transient (checked above).
    for f in (txt, meta):
        try:
            with open(f, "rb") as fh:
                fh.read().decode("utf-8")
        except (UnicodeDecodeError, OSError) as e:
            print(f"undecodable result file {f!r}: {e}", file=sys.stderr)
            return 1
    r = subprocess.run([sys.executable, os.path.join(PIPELINE, script), unit, txt, meta, *extra],
                       stderr=subprocess.PIPE, text=True)
    if r.stderr:
        sys.stderr.write(r.stderr)
    return classify_rc(r.returncode, r.stderr)


def record_wu_error(argv) -> bool:
    """`--error N wu_name wu_id batch_id`: give the failed WU a recorded disposition as a
    wu_error row in escalation_queue.jsonl (deduped per failed WU: unit_id+kind+wu_name, so a resubmission that errors again reopens the unit). True if newly queued."""
    import re
    import time
    code, wu_name = argv[1], argv[2]
    uid = wu_name[len(WU_PREFIX):] if wu_name.startswith(WU_PREFIX) else wu_name
    uid = re.sub(r"__r\d+$", "", uid)   # errored resubmission naca_<uid>__rN is still unit <uid>
    ntrs, _, page = uid.rpartition("-p")
    sys.path.insert(0, PIPELINE)
    from unit_assimilate import _enqueue
    row = {"unit_id": uid, "ntrs_id": ntrs, "page": int(page) if page.isdigit() else None,
           "kind": "wu_error", "error_code": code, "wu_name": wu_name,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    from pathlib import Path
    return _enqueue(Path(DATA_ROOT), row, key=("unit_id", "kind", "wu_name"))
