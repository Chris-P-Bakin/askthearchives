"""Pass 2: layout detection - find figure regions, crop them, write layout JSON.

CPU-only, no deps beyond numpy + PIL. Figure = connected non-text ink.
Text is masked using pass 3a word boxes, so pass 3a must run first.
"""
import argparse, hashlib, json, sys
from multiprocessing import Pool

import numpy as np
from PIL import Image

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import load_reports, report_dir, log

CONFIG = {
    "scale": 4,              # analyze at 1/4 resolution, block-min reduction
    "ink_thresh": 140,       # gray < thresh = ink (pages are autocontrasted)
    "text_pad": 3,           # px (reduced) padding around masked word boxes
    "text_conf": 20,         # mask words at/above this confidence
    "text_max_h": 120,       # full-scale px; taller "words" are not text
    "dilate": 5,             # reduced px, bridges dashed curves / halftone
    "min_side": 24,          # reduced px, min bbox side for a figure
    "min_ink": 150,          # reduced px of ink inside component
    "rule_aspect": 15,       # longer+thinner than this = rule, not figure
    "rule_max_thick": 8,     # reduced px
    "merge_gap": 12,         # reduced px, merge boxes closer than this
    "split_gap": 25,         # reduced px, split regions on blank gaps this tall/wide
    "label_reach": 60,       # full px, absorb OCR words within this of a figure
    "crop_margin": 15,       # full px margin on saved crops
    "border": 8,             # reduced px band at page edges zeroed (binding/scan edge)
    "despeckle": 10,         # drop undilated components with fewer ink px (dust)
}
CONFIG_HASH = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:12]
FORCE = False  # set by --force in main; forks inherit it


def block_min(a, s):
    h, w = a.shape
    h2, w2 = h // s, w // s
    return a[: h2 * s, : w2 * s].reshape(h2, s, w2, s).min(axis=(1, 3))


def dilate(mask, r):
    out = mask.copy()
    for ax in (0, 1):
        m = out.copy()
        for d in range(1, r + 1):
            m[tuple(slice(d, None) if i == ax else slice(None) for i in range(2))] |= \
                out[tuple(slice(None, -d) if i == ax else slice(None) for i in range(2))]
            m[tuple(slice(None, -d) if i == ax else slice(None) for i in range(2))] |= \
                out[tuple(slice(d, None) if i == ax else slice(None) for i in range(2))]
        out = m
    return out


def components(mask):
    """Connected components via row-run union-find. Returns list of (bbox, ink)."""
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    runs_prev = []  # (x0, x1, label)
    stats = {}      # label -> [x0, y0, x1, y1, ink]
    nxt = 0
    for y in range(mask.shape[0]):
        row = mask[y]
        xs = np.flatnonzero(np.diff(np.concatenate(([0], row.view(np.int8), [0]))))
        runs = []
        for i in range(0, len(xs), 2):
            x0, x1 = int(xs[i]), int(xs[i + 1])  # [x0, x1)
            lbl = None
            for px0, px1, pl in runs_prev:
                if px0 < x1 and x0 < px1:
                    if lbl is None:
                        lbl = find(pl)
                    else:
                        union(lbl, pl)
            if lbl is None:
                lbl = nxt
                parent[lbl] = lbl
                nxt += 1
            runs.append((x0, x1, lbl))
            s = stats.setdefault(lbl, [x0, y, x1, y + 1, 0])
            s[0] = min(s[0], x0); s[2] = max(s[2], x1); s[3] = y + 1
            s[4] += x1 - x0
        runs_prev = runs
    merged = {}
    for lbl, s in stats.items():
        r = find(lbl)
        if r in merged:
            m = merged[r]
            m[0] = min(m[0], s[0]); m[1] = min(m[1], s[1])
            m[2] = max(m[2], s[2]); m[3] = max(m[3], s[3]); m[4] += s[4]
        else:
            merged[r] = list(s)
    return [((m[0], m[1], m[2], m[3]), m[4]) for m in merged.values()]


def split_box(fig_ink, box, gap, depth=0):
    """Recursively split a region on blank row/column gaps >= gap."""
    x0, y0, x1, y1 = box
    sub = fig_ink[y0:y1, x0:x1]
    ys = np.flatnonzero(sub.any(axis=1))
    xs = np.flatnonzero(sub.any(axis=0))
    if len(ys) == 0:
        return []
    # shrink to content
    y0, y1 = y0 + int(ys[0]), y0 + int(ys[-1]) + 1
    x0, x1 = x0 + int(xs[0]), x0 + int(xs[-1]) + 1
    if depth >= 4:
        return [(x0, y0, x1, y1)]
    sub = fig_ink[y0:y1, x0:x1]
    for axis in (1, 0):  # rows first, then columns
        occ = np.flatnonzero(sub.any(axis=axis))
        gaps = np.flatnonzero(np.diff(occ) > gap)
        if len(gaps):
            cut = int(occ[gaps[0]]) + 1
            if axis == 1:
                a = (x0, y0, x1, y0 + cut); b = (x0, y0 + cut, x1, y1)
            else:
                a = (x0, y0, x0 + cut, y1); b = (x0 + cut, y0, x1, y1)
            return split_box(fig_ink, a, gap, depth + 1) + split_box(fig_ink, b, gap, depth + 1)
    return [(x0, y0, x1, y1)]


def boxes_touch(a, b, gap):
    return a[0] - gap < b[2] and b[0] - gap < a[2] and a[1] - gap < b[3] and b[1] - gap < a[3]


def merge_boxes(boxes, gap):
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        out = []
        while boxes:
            cur = boxes.pop()
            i = 0
            while i < len(boxes):
                if boxes_touch(cur, boxes[i], gap):
                    o = boxes.pop(i)
                    cur = [min(cur[0], o[0]), min(cur[1], o[1]),
                           max(cur[2], o[2]), max(cur[3], o[3])]
                    changed = True
                else:
                    i += 1
            out.append(cur)
        boxes = out
    return [tuple(b) for b in boxes]


def process_page(rdir, n, force=False):
    page = rdir / "pages" / f"{n}.png"
    ocr = rdir / "ocr" / f"{n}.json"
    ldir = rdir / "layout"
    fdir = rdir / "figures"
    out = ldir / f"{n}.json"
    if out.exists() and not (force or FORCE):
        try:
            if json.loads(out.read_text()).get("config_hash") == CONFIG_HASH:
                return "skipped"
        except Exception:
            pass
    _t0 = __import__("time").time()
    im = Image.open(page).convert("L")
    W, H = im.size
    s = CONFIG["scale"]
    a = block_min(np.asarray(im), s)
    ink = a < CONFIG["ink_thresh"]

    words = []
    if ocr.exists():
        words = json.loads(ocr.read_text()).get("words", [])
    tmask = np.zeros_like(ink)
    p = CONFIG["text_pad"]
    for w in words:
        l, t, r, b = w["bbox"]
        if b - t > CONFIG["text_max_h"] or w["conf"] < CONFIG["text_conf"]:
            continue
        tmask[max(0, t // s - p): b // s + p + 1, max(0, l // s - p): r // s + p + 1] = True

    fig_ink = ink & ~tmask
    B = CONFIG["border"]
    fig_ink[:B, :] = fig_ink[-B:, :] = False
    fig_ink[:, :B] = fig_ink[:, -B:] = False
    for (x0, y0, x1, y1), px in components(fig_ink):
        if px < CONFIG["despeckle"] and (x1 - x0) <= 5 and (y1 - y0) <= 5:
            fig_ink[y0:y1, x0:x1] = False
    grown = dilate(fig_ink, CONFIG["dilate"])

    figs, rules = [], []
    for (x0, y0, x1, y1), px in components(grown):
        w, h = x1 - x0, y1 - y0
        if w < 3 or h < 3 or px < CONFIG["min_ink"] // 3:
            continue
        thin = min(w, h) <= CONFIG["rule_max_thick"]
        if thin and max(w, h) / max(1, min(w, h)) >= CONFIG["rule_aspect"]:
            rules.append([x0 * s, y0 * s, x1 * s, y1 * s])
            continue
        if w < CONFIG["min_side"] or h < CONFIG["min_side"] or px < CONFIG["min_ink"]:
            continue
        figs.append((x0, y0, x1, y1))

    figs = merge_boxes(figs, CONFIG["merge_gap"])
    split = []
    for box in figs:
        for x0, y0, x1, y1 in split_box(fig_ink, box, CONFIG["split_gap"]):
            w, h, px = x1 - x0, y1 - y0, int(fig_ink[y0:y1, x0:x1].sum())
            if w >= CONFIG["min_side"] and h >= CONFIG["min_side"] and px >= CONFIG["min_ink"]:
                split.append((x0, y0, x1, y1))
    figs = [[x0 * s, y0 * s, x1 * s, y1 * s] for x0, y0, x1, y1 in split]

    # absorb nearby OCR words (axis/tick labels), then re-merge
    reach = CONFIG["label_reach"]
    for _ in range(2):
        for f in figs:
            for w in words:
                l, t, r, b = w["bbox"]
                if b - t > CONFIG["text_max_h"]:
                    continue
                if boxes_touch(f, (l, t, r, b), reach):
                    f[0] = min(f[0], l); f[1] = min(f[1], t)
                    f[2] = max(f[2], r); f[3] = max(f[3], b)
        figs = [list(b) for b in merge_boxes(figs, 8)]

    figs.sort(key=lambda f: (f[1], f[0]))
    fdir.mkdir(exist_ok=True)
    m = CONFIG["crop_margin"]
    recs = []
    for i, (l, t, r, b) in enumerate(figs, 1):
        l2, t2 = max(0, l - m), max(0, t - m)
        r2, b2 = min(W, r + m), min(H, b + m)
        im.crop((l2, t2, r2, b2)).save(fdir / f"{n}_{i}.png")
        recs.append({"index": i, "bbox": [l2, t2, r2, b2],
                     "area_frac": round((r2 - l2) * (b2 - t2) / (W * H), 4)})

    ldir.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "config_hash": CONFIG_HASH, "coord_space": "deskewed_px_300dpi",
        "page_size": [W, H], "figures": recs, "rules": rules,
        "n_words": len(words),
        "seconds": round(__import__("time").time() - _t0, 2),
    }, indent=1))
    return "ok"


def process_report(ntrs_id):
    rdir = report_dir(ntrs_id)
    pages = sorted((p for p in (rdir / "pages").glob("[0-9]*.png")
                    if not p.name.endswith(".raw.png")), key=lambda p: int(p.stem))
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    nfig = 0
    for p in pages:
        try:
            counts[process_page(rdir, int(p.stem))] += 1
        except Exception as e:
            counts["failed"] += 1
            log("pass2_page_error", report=ntrs_id, page=p.stem, error=repr(e))
    for lj in (rdir / "layout").glob("[0-9]*.json"):
        nfig += len(json.loads(lj.read_text())["figures"])
    return ntrs_id, counts, nfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuning", action="store_true", help="only the 30 tuning pages")
    ap.add_argument("--report", help="single NTRS id")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    global FORCE
    FORCE = args.force

    if args.tuning:
        import csv as _csv
        from common import DATA
        rows = list(_csv.DictReader(open(DATA / "tuning_pages.csv")))
        log("pass2_start", pages=len(rows), config_hash=CONFIG_HASH, tuning_only=True)
        counts = {"ok": 0, "skipped": 0, "failed": 0}
        nfig = 0
        for r in rows:
            try:
                counts[process_page(report_dir(r["ntrs_id"]), int(r["pdf_page"]), args.force)] += 1
                lj = report_dir(r["ntrs_id"]) / "layout" / f'{r["pdf_page"]}.json'
                nfig += len(json.loads(lj.read_text())["figures"])
            except Exception as e:
                counts["failed"] += 1
                log("pass2_page_error", report=r["ntrs_id"], page=r["pdf_page"], error=repr(e))
        log("pass2_done", figures=nfig, **counts, tuning_only=True)
        return

    ids = [args.report] if args.report else [r["ntrs_id"] for r in load_reports()]
    log("pass2_start", reports=len(ids), config_hash=CONFIG_HASH, tuning_only=False)
    total = {"ok": 0, "skipped": 0, "failed": 0}
    nfig = 0
    with Pool(args.workers) as pool:
        for rid, counts, nf in pool.imap_unordered(process_report, ids):
            for k in total:
                total[k] += counts[k]
            nfig += nf
            print(f"  {rid}: {counts} figures={nf}")
    log("pass2_done", figures=nfig, **total, tuning_only=False)


if __name__ == "__main__":
    main()
