"""NARA RG 255 ephemeral ingest worker + reaper.

Per doc: download PDF -> sha256 -> raster -> tesseract word boxes -> pass2 layout
-> gate -> vision units appended to <root>/units_pending.jsonl -> meta.json ->
PDF deleted. Page PNGs live until `--reap` finds every gated page assimilated.
Submission of units_pending.jsonl is a separate operator step.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import units
from units import ACCEPTED_HASHES, CAMPAIGN_LAYER, make_unit

CID = "nara-rg255"
UA = "naca-ocr-pilot/0.1 (research pipeline, homelab)"  # same string as pass1.UA


def default_root() -> Path:
    return units.registry.root(CID)


def doc_id_of(na_id) -> str:
    return f"{units.registry.load()[CID]['doc_prefix']}{na_id}"


# ------------------------------------------------------------------ selection
def _pdf_object(rec: dict):
    for o in rec.get("digitalObjects") or []:
        if not isinstance(o, dict) or not o.get("objectUrl"):
            continue
        if "pdf" in str(o.get("objectType", "")).lower() or \
                str(o.get("objectFilename", "")).lower().endswith(".pdf"):
            return o
    return None


def select(meta_dir, done: set):
    """Return (selected records, counts). Pilot predicate: PDF object AND
    useRestriction.status == Unrestricted AND not in the done ledger."""
    counts = dict(scanned=0, malformed=0, no_pdf=0, excluded_restricted=0,
                  already_done=0, selected=0)
    out = []
    for f in sorted(Path(meta_dir).glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            counts["scanned"] += 1
            try:
                rec = json.loads(line)["record"]
                na_id = rec["naId"]
                if not isinstance(na_id, int) or isinstance(na_id, bool):
                    raise ValueError("naId")
                dobjs = rec.get("digitalObjects", [])
                if not isinstance(dobjs, list):
                    raise ValueError("digitalObjects")
            except (ValueError, KeyError, TypeError):
                counts["malformed"] += 1
                continue
            if _pdf_object(rec) is None:
                counts["no_pdf"] += 1
            elif (rec.get("useRestriction") or {}).get("status") != "Unrestricted":
                counts["excluded_restricted"] += 1
            elif doc_id_of(na_id) in done:
                counts["already_done"] += 1
            else:
                counts["selected"] += 1
                out.append(rec)
    return out, counts


def read_done(root) -> set:
    p = Path(root) / "ingest_done.jsonl"
    if not p.exists():
        return set()
    return {json.loads(l)["doc_id"] for l in p.read_text().splitlines() if l.strip()}


# ------------------------------------------------------------------ disk cap
def inflight_bytes(root) -> int:
    """Bytes held by page PNGs awaiting reaping (plus any in-progress PDFs)."""
    total = 0
    for d in Path(root).glob("*/"):
        for sub in ("pages", "_dl"):
            for f in (d / sub).rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
    return total


def cap_bytes() -> float:
    return float(os.environ.get("NARA_INFLIGHT_GB", "100")) * 1e9


# NARA moved object serving off the public S3 bucket (now 403) onto their own
# host; the open-data mirror's objectUrls are stale. Verified 2026-11-21.
_URL_REWRITES = [("https://s3.amazonaws.com/NARAprodstorage/lz/",
                  "https://catalog.archives.gov/medialz/")]


def rewrite_object_url(url: str) -> str:
    for old, new in _URL_REWRITES:
        if url.startswith(old):
            return new + url[len(old):]
    return url


# ------------------------------------------------------------------ per doc
def download(url: str, dest, tries: int = 3):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(rewrite_object_url(url),
                                         headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as w:
                shutil.copyfileobj(r, w)
            return
        except Exception as e:  # noqa: BLE001 - retried, then re-raised
            last = e
            time.sleep(2 ** i)
    raise last


def _default_ocr(png):
    import pass3a
    return pass3a.ocr_page(png)


def _default_layout(rdir, n):
    import pass2
    return pass2.process_page(rdir, n)


def gate_page(rdir: Path, n: int) -> bool:
    """Gate v1 (same thresholds as workgen.gated_pages): conf < 70 or figure frac > 0.2."""
    o = json.loads((rdir / "ocr" / f"{n}.json").read_text())
    confs = [w["conf"] for w in o["words"]]
    conf = sum(confs) / len(confs) if confs else 0.0
    frac = 0.0
    lay = rdir / "layout" / f"{n}.json"
    if lay.exists():
        frac = sum(f["area_frac"] for f in json.loads(lay.read_text())["figures"])
    return conf < 70 or frac > 0.2


def _date(rec, key):
    d = rec.get(key)
    return d.get("logicalDate") if isinstance(d, dict) else None


MIN_TEXT_LAYER_CHARS = 50   # a page with at least this much extractable text is born-digital


def extract_pdftext(pdf: Path) -> list[str]:
    """Per-page text layer via pdftotext -layout (1-based list); [] on failure."""
    r = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return []
    return r.stdout.split("\f")


def ingest_doc(rec: dict, root, fetch=download, ocr=_default_ocr,
               layout=_default_layout, extract=extract_pdftext) -> dict:
    import raster
    root = Path(root)
    doc_id = doc_id_of(rec["naId"])
    d = root / doc_id  # == units.doc_dir(doc_id, CID) when root is the registry root
    obj = _pdf_object(rec)
    dl = d / "_dl"
    dl.mkdir(parents=True, exist_ok=True)
    pdf = dl / "source.pdf"
    try:
        fetch(obj["objectUrl"], pdf)
        h = hashlib.sha256()
        with open(pdf, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        sha = h.hexdigest()
        import pass1
        n_pages = pass1.pdf_pages(pdf)
        texts = extract(pdf)
        textful = {n: texts[n - 1] for n in range(1, min(n_pages, len(texts)) + 1)
                   if len(texts[n - 1].strip()) >= MIN_TEXT_LAYER_CHARS}
        needs_ocr = [n for n in range(1, n_pages + 1) if n not in textful]
        if needs_ocr:
            raster.raster_pages(pdf, d / "pages", needs_ocr)
    finally:
        shutil.rmtree(dl, ignore_errors=True)  # PDF gone as soon as raster is done (or failed)
    if textful:
        (d / "pdftext").mkdir(parents=True, exist_ok=True)
        for n, t in textful.items():
            (d / "pdftext" / f"{n}.txt").write_text(t)
    gated = []
    if needs_ocr:
        (d / "ocr").mkdir(exist_ok=True)
        for n in needs_ocr:
            png = d / "pages" / f"{n}.png"
            o = ocr(png)
            (d / "ocr" / f"{n}.json").write_text(json.dumps(o))
            # Gate on OCR confidence alone: the figure-area trigger was calibrated
            # for NACA typescript and false-positives on modern page furniture
            # (2026-11-22 pilot: 99% of gating was figure-frac at median conf 93.5).
            confs = [w["conf"] for w in o["words"]]
            conf = sum(confs) / len(confs) if confs else 0.0
            if conf < 70:
                gated.append(n)
    new_units = [make_unit(doc_id, n, CID) for n in gated]
    if new_units:
        with open(root / "units_pending.jsonl", "a") as f:
            for u in new_units:
                f.write(json.dumps(u, sort_keys=True) + "\n")
    dobjs = rec.get("digitalObjects") or []
    meta = {"doc_id": doc_id, "naId": rec["naId"], "title": rec.get("title"),
            "start_date": _date(rec, "inclusiveStartDate") or _date(rec, "coverageStartDate"),
            "end_date": _date(rec, "inclusiveEndDate") or _date(rec, "coverageEndDate"),
            "objectUrl": obj["objectUrl"], "pdf_sha256": sha, "n_pages": n_pages,
            "pdftext_pages": len(textful), "ocr_pages": len(needs_ocr),
            "rights": (rec.get("useRestriction") or {}).get("status"),
            "has_extracted_text": any(isinstance(o, dict) and o.get("extractedText") for o in dobjs),
            "gated_pages": gated, "ingested_ts": time.time()}
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    with open(root / "ingest_done.jsonl", "a") as f:  # last: a crash earlier retries the doc
        f.write(json.dumps({"doc_id": doc_id, "ts": meta["ingested_ts"]}) + "\n")
    return meta


def run_cycle(meta_dir, root, limit=None, workers=4, fetch=download, ocr=_default_ocr,
              layout=_default_layout) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    sel, counts = select(meta_dir, read_done(root))
    if limit is not None:
        sel = sel[:limit]
    res = {**counts, "ingested": 0, "failed": 0, "paused": False}
    cap = cap_bytes()
    i = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        while i < len(sel):
            if inflight_bytes(root) >= cap:   # pause selection while over the cap
                res["paused"] = True
                break
            batch = sel[i:i + max(1, workers)]
            i += len(batch)
            futs = [ex.submit(ingest_doc, r, root, fetch, ocr, layout) for r in batch]
            for r, fu in zip(batch, futs):
                try:
                    fu.result()
                    res["ingested"] += 1
                except Exception as e:  # noqa: BLE001 - one bad doc must not stop the cycle
                    res["failed"] += 1
                    with open(root / "ingest_failed.jsonl", "a") as f:
                        f.write(json.dumps({"doc_id": doc_id_of(r["naId"]),
                                            "error": repr(e)[:300], "ts": time.time()}) + "\n")
    return res


# ------------------------------------------------------------------ reaper
def _accepted(d: Path, n: int) -> bool:
    p = d / CAMPAIGN_LAYER / f"{n}.meta.json"
    try:
        return json.loads(p.read_text()).get("config_hash") in ACCEPTED_HASHES
    except (ValueError, OSError):
        return False


def reap(root) -> dict:
    """Delete pages/ for docs whose gated pages are all assimilated (accepted-hash meta exists; flag-compute
    runs inside unit_assimilate, so it is implied) or that gated nothing. One retention-ledger line per reaped doc."""
    root = Path(root)
    reaped, held = [], []
    for d in sorted(root.glob("*/")):
        mp = d / "meta.json"
        if not mp.exists() or not (d / "pages").is_dir():
            continue
        meta = json.loads(mp.read_text())
        gated = meta.get("gated_pages", [])
        if gated and not all(_accepted(d, n) for n in gated):
            held.append(meta["doc_id"])
            continue
        line = {"doc_id": meta["doc_id"], "n_pages": meta["n_pages"],
                "pdf_sha256": meta["pdf_sha256"], "objectUrl": meta["objectUrl"],
                "deleted_ts": time.time()}
        with open(root / "retention.jsonl", "a") as f:   # ledger first: never delete unrecorded
            f.write(json.dumps(line) + "\n")
        shutil.rmtree(d / "pages")
        reaped.append(meta["doc_id"])
    return {"reaped": reaped, "held": held}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir", default="data/nara_spike/rg_255_meta")
    ap.add_argument("--root", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reap", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.root) if a.root else default_root()
    if a.reap:
        print(json.dumps(reap(root)))
        return
    if a.dry_run:
        sel, counts = select(a.meta_dir, read_done(root))
        print(json.dumps(counts))
        return
    print(json.dumps(run_cycle(a.meta_dir, root, a.limit, a.workers)))


if __name__ == "__main__":
    main()
