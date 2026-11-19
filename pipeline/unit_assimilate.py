"""Lands validated worker results into the data tree (BOINC script_assimilator entry).

    python pipeline/unit_assimilate.py <unit.json> <result.txt> <result.meta.json> [--data-root DIR]

Idempotent: a page already holding an accepted-hash result is left untouched.
Pages needing human/engine review are appended to <data_root>/escalation_queue.jsonl.
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import table_verify
from unit_validate import validate_result
from units import ACCEPTED_HASHES, CAMPAIGN_LAYER, unit_doc_dir

DIGIT_THRESHOLD = 0.15  # same gate as the table_verify CLI


def _atomic_write(path: Path, data: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data)
    os.replace(tmp, path)


def _already_done(txt: Path, meta_p: Path) -> bool:
    if not (txt.exists() and meta_p.exists()):
        return False
    try:
        return json.loads(meta_p.read_text()).get("config_hash") in ACCEPTED_HASHES
    except (ValueError, OSError):
        return False


def _enqueue(root: Path, row: dict, key=("unit_id", "kind")) -> bool:
    """Append row unless an existing row matches on `key` fields (default unit_id+kind)."""
    q = root / "escalation_queue.jsonl"
    if q.exists():
        for line in q.read_text().splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if all(e.get(k) == row.get(k) for k in key):
                return False
    root.mkdir(parents=True, exist_ok=True)
    with open(q, "a") as f:
        f.write(json.dumps(row) + "\n")
    return True


def _enqueue_flags(root: Path, unit: dict, text: str, meta: dict, ts: str) -> list:
    base = {"unit_id": unit["unit_id"], "ntrs_id": unit["ntrs_id"], "page": unit["page"],
            "config_hash": meta.get("config_hash"), "ts": ts}
    if unit.get("collection", "naca") != "naca":  # absent means naca
        base["collection"] = unit["collection"]
    queued = []
    if (meta.get("loop_guard") or {}).get("vision_failed") is True:
        row = dict(base, kind="vision_failed", flags=[meta["loop_guard"]])
        if _enqueue(root, row):
            queued.append("vision_failed")
    elif (table_verify.digit_fraction(text) > DIGIT_THRESHOLD
          or table_verify.unrolled(text)):
        flags = table_verify.verify_text(text)["flags"]
        if flags and _enqueue(root, dict(base, kind="table_flags", flags=flags)):
            queued.append("table_flags")
    return queued


def assimilate(unit: dict, text: str, meta: dict, data_root=None, layer=None) -> dict:
    ok, reason = validate_result(unit, text, meta)
    if not ok:
        return {"status": "rejected", "reason": reason}
    root = Path(data_root) if data_root is not None else Path(common.DATA)
    vdir = unit_doc_dir(unit, root) / (layer or CAMPAIGN_LAYER)
    n = unit["page"]
    txt, meta_p = vdir / f"{n}.txt", vdir / f"{n}.meta.json"
    if _already_done(txt, meta_p):
        # Self-heal the crash window (page written, queue row lost): re-derive from the
        # published copy; _enqueue's dedupe makes this idempotent.
        pub_text, pub_meta = txt.read_text(), json.loads(meta_p.read_text())
        _enqueue_flags(root, unit, pub_text, pub_meta, pub_meta.get("assimilated_ts", ""))
        return {"status": "duplicate"}

    vdir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    full = dict(meta, assimilated_ts=ts, unit_id=unit["unit_id"],
                worker=meta.get("worker", ""))
    _atomic_write(txt, text)
    _atomic_write(meta_p, json.dumps(full, indent=1))  # meta last: it is the commit marker

    return {"status": "assimilated", "queued": _enqueue_flags(root, unit, text, meta, ts)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("unit")
    ap.add_argument("text")
    ap.add_argument("meta")
    ap.add_argument("--data-root")
    a = ap.parse_args(argv)
    res = assimilate(json.loads(Path(a.unit).read_text()), Path(a.text).read_text(),
                     json.loads(Path(a.meta).read_text()), a.data_root)
    print(json.dumps(res))
    return 1 if res["status"] == "rejected" else 0


if __name__ == "__main__":
    sys.exit(main())
