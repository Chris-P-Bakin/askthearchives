"""Pass 3a: cheap OCR on every page. Tesseract first; second engine added later.

Runs on the deskewed render (pages/{n}.png). Writes ocr/{n}.json per page:
engine, version, config + hash, reconstructed text, and word boxes with
confidences. Coordinates are deskewed-render pixels at 300 dpi; the mapping
back to the raw render is deterministic (the deskew is recomputable).
Idempotent per (report, page).
"""
import argparse, hashlib, json, os, subprocess, sys, time
from multiprocessing import Pool
import common
from common import log, report_dir

CONFIG = {"lang": "eng", "psm": 3, "dpi": 300, "oem": 3}
VERSION = subprocess.run(["tesseract", "--version"], capture_output=True,
                         text=True).stdout.split()[1]
CONFIG_HASH = hashlib.sha256(json.dumps({**CONFIG, "v": VERSION},
                             sort_keys=True).encode()).hexdigest()[:12]

def ocr_page(png_path):
    env = dict(os.environ, OMP_THREAD_LIMIT="1")
    out = subprocess.run(
        ["tesseract", str(png_path), "-", "-l", CONFIG["lang"],
         "--psm", str(CONFIG["psm"]), "--oem", str(CONFIG["oem"]),
         "--dpi", str(CONFIG["dpi"]), "tsv"],
        capture_output=True, text=True, env=env, check=True).stdout
    words, lines, key = [], {}, None
    for row in out.splitlines()[1:]:
        c = row.split("\t")
        if len(c) != 12:
            continue
        level, conf, text = int(c[0]), float(c[10]), c[11]
        if level != 5 or not text.strip():
            continue
        l, t, w, h = int(c[6]), int(c[7]), int(c[8]), int(c[9])
        words.append({"text": text, "conf": conf, "bbox": [l, t, l + w, t + h]})
        key = (c[2], c[3], c[4])          # block, par, line
        lines.setdefault(key, []).append(text)
    text = "\n".join(" ".join(ws) for ws in lines.values())
    return {"engine": "tesseract", "version": VERSION, "config": CONFIG,
            "config_hash": CONFIG_HASH, "coord_space": "deskewed_px_300dpi",
            "text": text, "words": words}

def process_page(args):
    nid, page, force = (args if len(args) == 3 else (*args, False))
    d = report_dir(nid)
    out = d / "ocr" / f"{page}.json"
    if not force and out.exists() and json.loads(out.read_text()).get("config_hash") == CONFIG_HASH:
        return 0
    out.parent.mkdir(exist_ok=True)
    try:
        t0 = time.time()
        rec = ocr_page(d / "pages" / f"{page}.png")
        rec["seconds"] = round(time.time() - t0, 2)
        out.write_text(json.dumps(rec))
        return 1
    except Exception as e:
        log("ocr_page_failed", ntrs_id=nid, page=page, error=repr(e)[:150])
        return -1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuning-only", action="store_true")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    jobs = []
    if args.tuning_only:
        import csv
        with open(common.DATA / "tuning_pages.csv", newline="") as f:
            jobs = [(r["ntrs_id"], int(r["pdf_page"])) for r in csv.DictReader(f)]
    else:
        for r in common.load_reports():
            d = report_dir(r["ntrs_id"]) / "pages"
            if d.is_dir():
                jobs += [(r["ntrs_id"], int(p.name.split(".")[0]))
                         for p in d.glob("*.png") if ".raw" not in p.name]
    log("pass3a_start", pages=len(jobs), engine=f"tesseract-{VERSION}",
        config_hash=CONFIG_HASH, tuning_only=args.tuning_only)
    with Pool(args.workers) as pool:
        res = pool.map(process_page, [(n, p, args.force) for n, p in jobs])
    log("pass3a_done", ocred=sum(1 for r in res if r == 1),
        skipped=sum(1 for r in res if r == 0), failed=sum(1 for r in res if r == -1))

if __name__ == "__main__":
    sys.exit(main())
