"""Pass 1: fetch, rasterize, baseline-extract. Idempotent per (ntrs_id, page).

Spec: download with throttle + descriptive UA + retry/backoff + sha256;
render 300 dpi keeping the untouched render as the canonical coordinate
space; extract the 1990s text layer unchanged; record per-page stats.
"""
# Launch batch runs with PYTHONUNBUFFERED=1 (block-buffered log lesson from batch 0).
import argparse, hashlib, json, os, pathlib, re, shutil, subprocess, sys, time, urllib.request, urllib.error
from multiprocessing import Pool
import common
from common import log, report_dir

UA = "naca-ocr-pilot/0.1 (research pipeline, homelab)"
THROTTLE = 4.0          # seconds between any two HTTP requests

def _throttled_get(url: str, binary_path=None, tries=4):
    """GET with global throttle, retry/backoff; returns (bytes|path, final_url, status)."""
    for attempt in range(tries):
        common.ntrs_wait(THROTTLE)
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                final = resp.geturl()
                if binary_path:
                    with open(binary_path, "wb") as f:
                        shutil.copyfileobj(resp, f)
                    return binary_path, final, resp.status
                return resp.read(), final, resp.status
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise                     # not retryable; caller logs it
            log("http_retry", url=url, code=e.code, attempt=attempt)
        except Exception as e:
            log("http_retry", url=url, error=str(e)[:120], attempt=attempt)
        time.sleep(2 ** attempt * 5)
    raise RuntimeError(f"gave up after {tries} tries: {url}")

def _record_pdf_link(rec):
    """Actual download URL from the NTRS citation record, when the CSV pattern 404s."""
    for d in (rec or {}).get("downloads") or []:
        links = d.get("links") or {}
        link = links.get("pdf") or links.get("original")
        if link:
            return "https://ntrs.nasa.gov" + link if link.startswith("/") else link
    return None

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# ---------------------------------------------------------------- download
def download(row):
    d = report_dir(row["ntrs_id"]); d.mkdir(parents=True, exist_ok=True)
    meta_p, pdf_p = d / "meta.json", d / "source.pdf"
    if meta_p.exists() and pdf_p.exists():
        meta = json.loads(meta_p.read_text())
        if meta.get("sha256") == sha256_file(pdf_p):
            return meta
    meta = {"csv_row": row}
    meta["ntrs_record"] = None
    def fetch_record():
        try:
            rec, _, _ = _throttled_get(f"https://ntrs.nasa.gov/api/citations/{row['ntrs_id']}")
            meta["ntrs_record"] = json.loads(rec)
        except Exception as e:
            log("ntrs_record_failed", ntrs_id=row["ntrs_id"], error=str(e)[:120])
    def missing(code, url):
        log("pdf_missing", ntrs_id=row["ntrs_id"], code=code, url=url)
        meta.update(pdf_status=code, sha256=None)
        meta_p.write_text(json.dumps(meta, indent=1))
        return meta
    url = row.get("pdf_url") or ""
    if url:   # roster URL: one NTRS request instead of two
        try:
            _, final_url, status = _throttled_get(url, binary_path=pdf_p)
        except urllib.error.HTTPError as e:
            if e.code != 404:
                return missing(e.code, url)
            url = ""   # fall back to the citation record once
        except RuntimeError:
            return missing(None, url)
    if not url:
        fetch_record()
        url = _record_pdf_link(meta["ntrs_record"])
        if not url:
            return missing(404, row.get("pdf_url") or "")
        log("pdf_fallback_link", ntrs_id=row["ntrs_id"], url=url)
        try:
            _, final_url, status = _throttled_get(url, binary_path=pdf_p)
        except (urllib.error.HTTPError, RuntimeError) as e2:
            return missing(getattr(e2, "code", None), url)
    meta.update(pdf_status=status, pdf_url_requested=url,
                pdf_url_final=final_url if final_url != url else None,
                sha256=sha256_file(pdf_p),
                download_time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                pdf_bytes=pdf_p.stat().st_size)
    if meta["pdf_url_final"]:
        log("pdf_redirected", ntrs_id=row["ntrs_id"], final=final_url)
    meta_p.write_text(json.dumps(meta, indent=1))
    log("downloaded", ntrs_id=row["ntrs_id"], mb=round(meta["pdf_bytes"] / 1e6, 1))
    return meta

# ---------------------------------------------------------------- rasterize
def pdf_pages(pdf) -> int:
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    m = re.search(r"^Pages:\s+(\d+)", out, re.M)
    return int(m.group(1)) if m else 0

def deskew_and_normalize(src, dst):
    """Projection-profile deskew (±3°) + autocontrast. Angle < 0.1° -> no rotation."""
    from PIL import Image, ImageOps
    import numpy as np
    im = Image.open(src).convert("L")
    small = im.resize((max(1, im.width * 800 // im.height), 800))
    a = np.asarray(small, dtype=np.float32)
    ink = a < a.mean() * 0.75
    best_angle, best_score = 0.0, -1.0
    for coarse in np.arange(-3.0, 3.01, 0.5):
        for ang in (coarse,):
            rot = np.asarray(Image.fromarray(ink.astype("uint8") * 255).rotate(ang, fillcolor=0))
            score = float(np.var(rot.sum(axis=1)))
            if score > best_score:
                best_angle, best_score = ang, score
    for ang in np.arange(best_angle - 0.4, best_angle + 0.41, 0.1):
        rot = np.asarray(Image.fromarray(ink.astype("uint8") * 255).rotate(float(ang), fillcolor=0))
        score = float(np.var(rot.sum(axis=1)))
        if score > best_score:
            best_angle, best_score = float(ang), score
    out = Image.open(src)
    if abs(best_angle) >= 0.1:
        out = out.rotate(best_angle, resample=Image.BICUBIC, fillcolor="white", expand=False)
    ImageOps.autocontrast(out.convert("L"), cutoff=1).save(dst)
    return round(best_angle, 2)

def ink_fraction(png):
    from PIL import Image
    import numpy as np
    a = np.asarray(Image.open(png).convert("L").resize((400, 520)), dtype=np.uint8)
    return round(float((a < 128).mean()), 5)

class RasterTimeout(Exception):
    pass

def rasterize(ntrs_id):
    d = report_dir(ntrs_id)
    pdf = d / "source.pdf"
    n = pdf_pages(pdf)
    pages = d / "pages"
    have_raw = len(list(pages.glob("*.raw.png")))
    if have_raw < n:
        tmp = d / "_raster"; shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()
        timeout = int(os.environ.get("NACA_RASTER_TIMEOUT", "7200"))
        try:   # subprocess.run kills the child on expiry
            subprocess.run(["pdftoppm", "-r", "300", "-png", str(pdf), str(tmp / "p")],
                           check=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            shutil.rmtree(tmp, ignore_errors=True)
            raise RasterTimeout(f"pdftoppm exceeded {timeout}s")
        pages.mkdir(exist_ok=True)
        for f in sorted(tmp.glob("p-*.png")):
            page = int(f.stem.split("-")[-1])
            f.rename(pages / f"{page}.raw.png")
        shutil.rmtree(tmp)
    pages.mkdir(exist_ok=True)
    angles = {}
    for raw in sorted(pages.glob("*.raw.png"), key=lambda p: int(p.name.split(".")[0])):
        page = int(raw.name.split(".")[0])
        dsk = pages / f"{page}.png"
        if not dsk.exists():
            angles[page] = deskew_and_normalize(raw, dsk)
    return n, angles

# ---------------------------------------------------------------- baseline
def baseline(ntrs_id, n_pages, words):
    d = report_dir(ntrs_id)
    pdf = d / "source.pdf"
    b = d / "baseline"; b.mkdir(exist_ok=True)
    fonts = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True).stdout
    font_rows = [l for l in fonts.splitlines()[2:] if l.strip()]
    stats = {"ntrs_id": ntrs_id, "baseline_absent": not font_rows, "fonts": len(font_rows),
             "pdf_pages": n_pages, "pages": {}}
    for p in range(1, n_pages + 1):
        txt_p, bbox_p = b / f"{p}.txt", b / f"{p}.bbox.html"
        if not txt_p.exists():
            subprocess.run(["pdftotext", "-layout", "-f", str(p), "-l", str(p), str(pdf), str(txt_p)], check=True)
        if not bbox_p.exists():
            subprocess.run(["pdftotext", "-bbox-layout", "-f", str(p), "-l", str(p), str(pdf), str(bbox_p)], check=True)
        text = txt_p.read_text(errors="replace")
        toks, known, share = common.dict_share(text, words)
        ps = {"chars": len(text.strip()), "words": toks, "known_words": known, "dict_share": share}
        if ps["chars"] < 50:
            ink = ink_fraction(d / "pages" / f"{p}.raw.png")
            ps["flag"] = "blank_page" if ink < 0.005 else "image_only"
            ps["ink_fraction"] = ink
        stats["pages"][str(p)] = ps
    flagged = sum(1 for v in stats["pages"].values() if "flag" in v)
    (b / "stats.json").write_text(json.dumps(stats, indent=1))
    return stats, flagged

# ---------------------------------------------------------------- per report
def process_report(args):
    row, words = args
    try:
        d = report_dir(row["ntrs_id"])
        if not (d / "source.pdf").exists():
            return (row["ntrs_id"], "no_pdf", 0)
        import time as _time
        _t0 = _time.time()
        n, angles = rasterize(row["ntrs_id"])
        _t_raster = _time.time() - _t0
        _t0 = _time.time()
        stats, flagged = baseline(row["ntrs_id"], n, words)
        _t_base = _time.time() - _t0
        _mp = d / "meta.json"
        n_deskewed = sum(1 for a in angles.values() if abs(a) >= 0.1)
        if _mp.exists():
            # timings/deskewed are write-once: a skip re-run only measures the skipped work
            _meta = json.loads(_mp.read_text())
            if _meta.get("pdf_status") == "raster_timeout":
                _meta["pdf_status"] = 200
            _meta.setdefault("timings", {"rasterize_s": round(_t_raster, 1),
                                         "baseline_s": round(_t_base, 1), "pages": n})
            _meta.setdefault("deskewed", n_deskewed)
            n_deskewed = _meta["deskewed"]
            _mp.write_text(json.dumps(_meta, indent=1))
        shares = [v["dict_share"] for v in stats["pages"].values() if v["words"] > 0]
        log("report_done", ntrs_id=row["ntrs_id"], report=row["report_number"], pages=n,
            baseline_absent=stats["baseline_absent"], low_text_pages=flagged,
            mean_dict_share=round(sum(shares) / len(shares), 3) if shares else 0,
            deskewed=n_deskewed)
        return (row["ntrs_id"], "ok", n)
    except RasterTimeout as e:
        _mp = report_dir(row["ntrs_id"]) / "meta.json"
        if _mp.exists():   # sha256 stays, so download() won't refetch; process_report retries raster
            _m = json.loads(_mp.read_text()); _m["pdf_status"] = "raster_timeout"
            _mp.write_text(json.dumps(_m, indent=1))
        log("report_failed", ntrs_id=row["ntrs_id"], reason="raster_timeout", error=str(e))
        return (row["ntrs_id"], "error", 0)
    except Exception as e:
        log("report_failed", ntrs_id=row["ntrs_id"], error=repr(e)[:200])
        return (row["ntrs_id"], "error", 0)

def record_no_pdf(row):
    """Disposition for roster rows whose NTRS record has no PDF: meta only, no download."""
    d = report_dir(row["ntrs_id"]); d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(
        {"pdf_status": "no_pdf_in_record", "csv_row": row}, indent=1))

def split_no_pdf(rows):
    return ([r for r in rows if r.get("no_pdf") != "True"],
            [r for r in rows if r.get("no_pdf") == "True"])

def select_batch(rows, batch, batch_size):
    """Campaign population (year <= 1975 or series != other), ordered by
    (published_year as int, ntrs_id) with undated rows last, then slice batch N."""
    if rows and "published_year" not in rows[0]:
        sys.exit("--batch needs a roster CSV with a published_year column")
    def year(r):
        y = r["published_year"]
        return int(y) if y else 9999
    pop = [r for r in rows if (r["published_year"] and int(r["published_year"]) <= 1975)
           or r["series"] != "other"]
    pop.sort(key=lambda r: (year(r), r["ntrs_id"]))
    return pop[batch * batch_size:(batch + 1) * batch_size]

POPULATION_RULE = "published_year<=1975 or series!=other"

def resolve_batch(rows, batch, batch_size, roster_path, batches_dir=None, rebatch=False):
    """select_batch + a persisted manifest (data/batches/batch_NNN.json). Membership is
    the manifest's id list; a recomputed selection that differs is refused unless rebatch."""
    batches_dir = pathlib.Path(batches_dir or common.DATA / "batches")
    mpath = batches_dir / f"batch_{batch:03d}.json"
    sel = select_batch(rows, batch, batch_size)
    ids = [r["ntrs_id"] for r in sel]
    if mpath.exists() and not rebatch:
        recorded = json.loads(mpath.read_text())["ids"]
        if recorded != ids:
            sys.exit(f"batch {batch}: recomputed selection ({len(ids)} ids) differs from "
                     f"{mpath} ({len(recorded)} ids); roster or population changed. "
                     f"Pass --rebatch to overwrite the manifest.")
        return sel
    batches_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"batch": batch, "batch_size": batch_size, "ids": ids,
                "roster_sha256": sha256_file(roster_path),
                "population_rule": POPULATION_RULE,
                "created_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    tmp = mpath.with_name(mpath.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=1))
    os.replace(tmp, mpath)
    return sel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", help="comma-separated ntrs_ids (default: all)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--batch", type=int, help="roster batch index (use with NACA_REPORTS_CSV=data/roster.csv)")
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--rebatch", action="store_true",
                    help="overwrite the batch manifest if the recomputed selection differs")
    args = ap.parse_args()
    rows = common.load_reports()
    if args.batch is not None:
        rows = resolve_batch(rows, args.batch, args.batch_size,
                             os.environ.get("NACA_REPORTS_CSV") or common.CSV,
                             rebatch=args.rebatch)
        rows, no_pdf = split_no_pdf(rows)
        for r in no_pdf:
            record_no_pdf(r)
        log("batch_selected", batch=args.batch, with_pdf=len(rows), no_pdf=len(no_pdf))
    if args.ids:
        want = set(args.ids.split(","))
        rows = [r for r in rows if r["ntrs_id"] in want]
    if args.limit:
        rows = rows[: args.limit]
    log("pass1_start", reports=len(rows))
    if not args.skip_download:
        for row in rows:                       # downloads stay serial: global throttle
            download(row)
    words = common.known_words()
    with Pool(args.workers) as pool:           # raster + baseline parallel per report
        results = pool.map(process_report, [(r, words) for r in rows])
    ok = sum(1 for _, s, _ in results if s == "ok")
    pages = sum(n for _, _, n in results)
    log("pass1_done", reports_ok=ok, reports_failed=len(results) - ok, total_pages=pages)

if __name__ == "__main__":
    sys.exit(main())
