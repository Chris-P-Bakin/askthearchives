"""Full NACA roster from the NTRS search API.

The search endpoint 400s past offset ~10,000, so the corpus (~14.6k records)
is sliced into published-date windows; any window reporting more than
MAX_WINDOW hits is split at its midpoint and re-fetched. Records are deduped
by NTRS id. A windowed pull alone misses records (odd/missing published dates),
so fetch_all then unions in four sorted sweeps (published/created, asc/desc,
up to the 10k offset limit) and asserts the result equals the unfiltered
total. `--from-jsonl PATH` rebuilds the CSV offline from saved raw records.
`--probe` sends one windowed request and prints its total, to
verify the date filter narrows the result set.
"""
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from common import DATA, load_reports, log

UA = "naca-ocr-pilot/0.1 (research pipeline, homelab)"
SEARCH_URL = "https://ntrs.nasa.gov/api/citations/search"
BASE_URL = "https://ntrs.nasa.gov"
THROTTLE = 4.0            # seconds between search API requests
PAGE_SIZE = 100
MAX_WINDOW = 9500         # API refuses offsets past ~10,000
UNION_WINDOW = 10000      # offsets reachable by one sorted sweep
UNION_SORTS = (("published", "asc"), ("published", "desc"),
               ("created", "asc"), ("created", "desc"))

SERIES = ("TR", "TN", "TM", "RM", "WR")
_series_re = re.compile(r"NACA-(TR|TN|TM|RM|WR)\b")
FIELDS = ["ntrs_id", "report_number", "title", "series", "published_year",
          "pdf_url", "no_pdf", "in_pilot", "pub_date", "decade"]


def classify_series(report_numbers: list) -> str:
    for num in report_numbers:
        m = _series_re.search(num)
        if m:
            return m.group(1)
    return "other"


def _report_numbers(rec: dict) -> list:
    out = []
    for n in rec.get("otherReportNumbers") or []:
        n = re.sub(r"^Report Number:\s*", "", n).strip()
        if n and n not in out:
            out.append(n)
    return out


def _pdf_url(rec: dict) -> str:
    for d in rec.get("downloads") or []:
        links = d.get("links") or {}
        link = links.get("pdf") or links.get("original")
        if link and (d.get("mimetype") == "application/pdf"
                     or link.lower().split("?")[0].endswith(".pdf")):
            return link if link.startswith("http") else BASE_URL + link
    return ""


def extract_row(rec: dict) -> dict:
    nums = _report_numbers(rec)
    year, pub_date = None, ""
    for p in rec.get("publications") or []:
        m = re.match(r"(\d{4})(-\d\d-\d\d)?", p.get("publicationDate") or "")
        if m:
            year = int(m.group(1))
            pub_date = m.group(0) if m.group(2) else f"{year}-01-01"
            break
    url = _pdf_url(rec)
    return {"ntrs_id": str(rec["id"]), "title": rec.get("title", ""),
            "series": classify_series(nums), "report_number": ";".join(nums),
            "published_year": year, "pdf_url": url, "no_pdf": not url,
            "pub_date": pub_date,
            "decade": f"{year // 10 * 10}s" if year else ""}


def year_windows(lo=1900, hi=1976, step=10) -> list:
    """Contiguous [gte, lt) ISO date windows covering years lo..hi."""
    out, y = [], lo
    while y < hi:
        out.append((f"{y}-01-01", f"{min(y + step, hi)}-01-01"))
        y += step
    return out


def split_window(w: tuple) -> tuple:
    lo, hi = w
    a, b = date.fromisoformat(lo), date.fromisoformat(hi)
    mid = date.fromordinal((a.toordinal() + b.toordinal()) // 2).isoformat()
    return (lo, mid), (mid, hi)


def _search_body(window, offset):
    return {"q": "NACA",
            "published": {"gte": window[0], "lt": window[1]},
            "page": {"size": PAGE_SIZE, "from": offset}}


def post_search(body: dict, tries=4) -> dict:
    """POST to the search API with global throttle and retry/backoff."""
    data = json.dumps(body).encode()
    for attempt in range(tries):
        common.ntrs_wait(THROTTLE)
        req = urllib.request.Request(
            SEARCH_URL, data=data, method="POST",
            headers={"User-Agent": UA, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 400:
                raise                     # malformed body; retrying cannot help
            log("search_retry", code=e.code, attempt=attempt)
        except Exception as e:
            log("search_retry", error=str(e)[:120], attempt=attempt)
        time.sleep(2 ** attempt * 5)
    raise RuntimeError(f"gave up after {tries} tries: {body}")


def _fetch_window(post, window, found: dict):
    first = post(_search_body(window, 0))
    total = first["stats"]["total"]
    if total > MAX_WINDOW:
        for half in split_window(window):
            _fetch_window(post, half, found)
        return
    page, off = first, 0
    while True:
        for rec in page["results"]:
            found[rec["id"]] = rec
        off += PAGE_SIZE
        if off >= total or not page["results"]:
            break
        page = post(_search_body(window, off))
    log("window_done", window=f"{window[0]}..{window[1]}", total=total, seen=len(found))


def _sorted_sweep(post, field, order, found: dict):
    off = 0
    while off < UNION_WINDOW:
        size = min(PAGE_SIZE, UNION_WINDOW - off)
        page = post({"q": "NACA", "sort": {"field": field, "order": order},
                     "page": {"size": size, "from": off}})
        if not page["results"]:
            break
        for rec in page["results"]:
            found[rec["id"]] = rec
        off += size


def fetch_all(post, union=True) -> list:
    found = {}
    for window in year_windows():
        _fetch_window(post, window, found)
    if union:
        windowed = len(found)
        for field, order in UNION_SORTS:
            _sorted_sweep(post, field, order, found)
        total = post({"q": "NACA", "page": {"size": 1, "from": 0}})["stats"]["total"]
        log("roster_union", windowed=windowed, final=len(found), total=total)
        if len(found) != total:
            raise RuntimeError(f"roster incomplete: have {len(found)} ids but "
                               f"unfiltered stats.total is {total}")
    return list(found.values())


def _csv_rows(path) -> int:
    with open(path, newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def _check_not_shrinking(csv_path: Path, new_rows: int, force: bool):
    if force or not csv_path.exists():
        return
    old = _csv_rows(csv_path)
    if new_rows < old:
        raise RuntimeError(f"refusing to overwrite {csv_path}: {old} existing rows > "
                           f"{new_rows} new rows (pass --force to override)")


def write_roster(records: list, csv_path, jsonl_path=None, force=False) -> dict:
    """Write roster CSV (and raw-record JSONL unless jsonl_path is None) atomically."""
    pilot = {r["ntrs_id"] for r in load_reports(str(common.CSV))}
    rows = [extract_row(r) for r in records]
    csv_path = Path(csv_path)
    _check_not_shrinking(csv_path, len(rows), force)
    csv_tmp = csv_path.with_name(csv_path.name + ".tmp")
    with open(csv_tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({**row, "in_pilot": row["ntrs_id"] in pilot})
    if jsonl_path is not None:
        jsonl_path = Path(jsonl_path)
        jsonl_tmp = jsonl_path.with_name(jsonl_path.name + ".tmp")
        with open(jsonl_tmp, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        os.replace(csv_tmp, csv_path)
        os.replace(jsonl_tmp, jsonl_path)
    else:
        os.replace(csv_tmp, csv_path)
    return {
        "total": len(rows),
        "by_series": dict(Counter(r["series"] for r in rows)),
        "by_decade": dict(Counter(f"{r['published_year'] // 10 * 10}s"
                                  if r["published_year"] else "unknown" for r in rows)),
        "no_pdf": sum(r["no_pdf"] for r in rows),
        "pilot_overlap": sum(r["ntrs_id"] in pilot for r in rows),
    }


def run(post, csv_path, jsonl_path, force=False) -> dict:
    return write_roster(fetch_all(post), csv_path, jsonl_path, force=force)


def rebuild_from_jsonl(jsonl_path, csv_path, force=False) -> dict:
    """Offline rebuild of the roster CSV from saved raw NTRS records (no network)."""
    with open(jsonl_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    return write_roster(records, csv_path, None, force=force)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", action="store_true",
                    help="one windowed request (1930s); print the total")
    ap.add_argument("--out", default=str(DATA / "roster.csv"))
    ap.add_argument("--jsonl", default=str(DATA / "roster_records.jsonl"))
    ap.add_argument("--from-jsonl", metavar="PATH",
                    help="rebuild --out offline from saved raw records; no network")
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting a roster CSV with fewer rows")
    args = ap.parse_args()
    if args.from_jsonl:
        print(json.dumps(rebuild_from_jsonl(args.from_jsonl, args.out, args.force), indent=1))
        return
    if args.probe:
        resp = post_search(_search_body(("1930-01-01", "1940-01-01"), 0))
        print("probe total:", resp["stats"]["total"])
        return
    print(json.dumps(run(post_search, args.out, args.jsonl, args.force), indent=1))


if __name__ == "__main__":
    main()
