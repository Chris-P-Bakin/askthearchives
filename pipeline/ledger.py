#!/usr/bin/env python3
"""Public per-batch ledger: one reconciled row per batch -> data/publish/ledger.jsonl and a
markdown table in RESULTS.md between <!-- ledger:start --> / <!-- ledger:end -->.

Arithmetic that does not reconcile never crashes: it lands in the row's "warnings"."""
import argparse
import json
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import escalate
import units
import workgen

START, END = "<!-- ledger:start -->", "<!-- ledger:end -->"
BOINC_Q = ("select (select count(*) from workunit),"
           "(select count(*) from result where validate_state=1),"
           "(select count(*) from result where validate_state=2)")


@contextmanager
def _data_root(data):
    """workgen.gated_pages reads common.DATA via report_dir; point it at `data` for the call."""
    old = common.DATA
    common.DATA = Path(data)
    try:
        yield
    finally:
        common.DATA = old


def _meta(p):
    try:
        m = json.loads(Path(p).read_text())
        return m if isinstance(m, dict) else None
    except (OSError, ValueError):
        return None


def _page_of(row):
    nid, n = row.get("ntrs_id"), row.get("page")
    if nid is None and isinstance(row.get("unit_id"), str) and "-p" in row["unit_id"]:
        nid, _, n = row["unit_id"].rpartition("-p")
    try:
        return str(nid), int(n)
    except (TypeError, ValueError):
        return None


def _boinc():
    cmd = ["docker", "exec", "naca-boinc-apache", "bash", "-c",
           'source /run/secrets/secrets.env; mysql -h mysql -uroot -p$DB_PASSWD naca -N -e "'
           + BOINC_Q + '"']
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if p.returncode != 0:
            raise RuntimeError(f"rc={p.returncode}")
        a, b, c = (int(x) for x in p.stdout.split())
        return {"workunits": a, "results_valid": b, "results_invalid": c}, None
    except Exception as e:
        return None, f"boinc counts unavailable: {type(e).__name__}"


def batch_row(batch: int, data=None, now=None, boinc=True) -> dict:
    data = Path(data) if data is not None else common.DATA
    ids = [str(i) for i in json.loads((data / "batches" / f"batch_{batch:03d}.json").read_text())["ids"]]
    idset, warnings = set(ids), []

    ocr = {(nid, int(oj.stem)) for nid in ids for oj in (data / nid / "ocr").glob("[0-9]*.json")}
    try:
        with _data_root(data):
            gated = set(workgen.gated_pages(ids))
    except Exception as e:  # unreadable ocr/layout json: degrade, don't crash
        gated = set()
        warnings.append(f"gate computation failed: {type(e).__name__}: {e}")
    ungated = ocr - gated
    pages = len(ocr)
    if gated - ocr or len(gated) + len(ungated) != pages:
        warnings.append(f"gated({len(gated)}) + ungated({len(ungated)}) != ocr_pages({pages})")

    failed, stamps, campaign = 0, [], 0
    for nid in ids:
        for mp in (data / nid / units.CAMPAIGN_LAYER).glob("*.meta.json"):
            m = _meta(mp)
            if not m or m.get("config_hash") not in units.ACCEPTED_HASHES:
                continue
            campaign += 1
            if (m.get("loop_guard") or {}).get("vision_failed") is True:
                failed += 1
            if m.get("assimilated_ts"):
                stamps.append(m["assimilated_ts"])

    esc = {escalate.LAYER_27B: set(), escalate.LAYER_235B: set()}
    for layer in esc:
        for nid in ids:
            for mp in (data / nid / layer).glob("*.meta.json"):
                esc[layer].add((nid, int(mp.name.split(".")[0])))
    for e in escalate._read_jsonl(escalate._done_path(data)):
        pg = _page_of(e)
        if e.get("layer") in esc and pg and pg[0] in idset:
            esc[e["layer"]].add(pg)
    e27, e235 = esc[escalate.LAYER_27B], esc[escalate.LAYER_235B]
    if (e27 | e235) - gated:
        warnings.append(f"escalated pages not in gated set: {len((e27 | e235) - gated)}")
    if len(e27) > len(gated) or len(e235) > len(gated):
        warnings.append("escalated count exceeds gated")

    review = sum(1 for r in escalate._read_jsonl(data / "escalation_review.jsonl")
                 if (_page_of(r) or ("", 0))[0] in idset)
    flagged = {pg for r in escalate._read_jsonl(data / "escalation_queue.jsonl")
               if r.get("kind") == "table_flags" and (pg := _page_of(r)) and pg[0] in idset}

    def pct(x, d):
        return round(100.0 * x / d, 2) if d else None

    pph = None
    if len(stamps) >= 2:
        t = [time.mktime(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ")) for s in stamps]
        if max(t) > min(t):
            pph = round(len(stamps) / ((max(t) - min(t)) / 3600), 2)

    b, bw = _boinc() if boinc else (None, None)
    if bw:
        warnings.append(bw)
    return {
        "batch": batch, "docs": len(ids), "pages": pages, "ocr_pages": pages,
        "gated": len(gated), "ungated": len(ungated), "gate_pct": pct(len(gated), pages),
        "campaign_pages": campaign,
        "vision_failed_pct": pct(failed, len(gated)), "flag_pct": pct(len(flagged), len(gated)),
        "escalated_27b": len(e27), "escalated_235b": len(e235), "review_rows": review,
        "spend_usd_cumulative": escalate.spend(data), "pages_per_hour": pph,
        "boinc": b, "warnings": warnings,
        "published_ts": now or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


COLS = [("batch", "batch"), ("docs", "docs"), ("pages", "pages"), ("gate_pct", "gate %"),
        ("vision_failed_pct", "vision-failed %"), ("flag_pct", "table-flag %"),
        ("escalated_27b", "27B"), ("escalated_235b", "235B"), ("review_rows", "review"),
        ("spend_usd_cumulative", "spend $ (cumulative)"), ("pages_per_hour", "pages/h"), ("published_ts", "published")]


def _cell(v):
    return "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def _warn_cell(ws):
    ws = [str(w).replace("|", "\\|").replace("\n", " ") for w in ws or []]
    if not ws:
        return "-"
    txt = "; ".join(ws[:2])
    return txt + (f" (+{len(ws) - 2} more, see ledger.jsonl)" if len(ws) > 2 else "")


FOOTNOTE = ("*spend $ (cumulative) is the campaign-wide 235B meter (`escalate.spend()`) at "
            "publish time, not a per-batch figure.*")


def table(rows) -> str:
    out = ["| " + " | ".join(h for _, h in COLS) + " | warnings |",
           "|" + "---|" * (len(COLS) + 1)]
    for r in rows:
        out.append("| " + " | ".join(_cell(r.get(k)) for k, _ in COLS)
                   + f" | {_warn_cell(r.get('warnings'))} |")
    return "\n".join(out) + "\n\n" + FOOTNOTE


def write(row, ledger_path=None, results_path=None):
    ledger_path = Path(ledger_path or common.DATA / "publish" / "ledger.jsonl")
    results_path = Path(results_path or common.ROOT / "RESULTS.md")
    rows = {r["batch"]: r for r in escalate._read_jsonl(ledger_path) if "batch" in r}
    rows[row["batch"]] = row
    rows = [rows[k] for k in sorted(rows)]
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    block = f"{START}\n{table(rows)}\n{END}"
    text = results_path.read_text() if results_path.exists() else ""
    pat = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
    text = pat.sub(lambda _: block, text) if pat.search(text) else text.rstrip("\n") + "\n\n" + block + "\n"
    results_path.write_text(text)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    row = batch_row(a.batch)
    print(json.dumps(row, indent=1))
    for w in row["warnings"]:
        print("WARNING:", w, file=sys.stderr)
    if a.write:
        write(row)


if __name__ == "__main__":
    main()
