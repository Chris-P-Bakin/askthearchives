"""Gate calibration: correlate free signals with measured tesseract CER on the
tuning set, sweep escalation rules, and project escalation volume corpus-wide."""
import json, sys
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import DATA, report_dir, dict_share, known_words, load_reports

WORDS = known_words()

BAD_CER = 0.25  # a page with tesseract CER above this should reach the vision tier


def signals(ntrs_id: str, n: int) -> dict:
    o = json.loads((report_dir(ntrs_id) / "ocr" / f"{n}.json").read_text())
    words = o["words"]
    confs = [w["conf"] for w in words]
    toks_t = {w.lower().strip(".,;:()") for w in o["text"].split() if len(w) > 2}
    bl = report_dir(ntrs_id) / "baseline" / f"{n}.txt"
    toks_b = set()
    if bl.exists():
        toks_b = {w.lower().strip(".,;:()") for w in bl.read_text(errors="replace").split() if len(w) > 2}
    agree = len(toks_t & toks_b) / max(1, len(toks_t | toks_b))
    return {
        "conf": float(np.mean(confs)) if confs else 0.0,
        "dict": dict_share(o["text"], WORDS)[2],
        "agree": agree,
        "chars": len(o["text"]),
    }


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d else 0.0


def main():
    scored = json.load(open(DATA / "scores_latest.json"))["pages"]
    rows = []
    for p in scored:
        if p["gt_status"] != "done" or "tesseract" not in p["engines"]:
            continue
        s = signals(p["ntrs_id"], p["page"])
        s["cer"] = p["engines"]["tesseract"]["cer"]
        s["id"] = f'{p["ntrs_id"]} p{p["page"]}'
        rows.append(s)
    cer = np.array([r["cer"] for r in rows])
    print(f"n={len(rows)} pages, {int((cer > BAD_CER).sum())} bad (CER>{BAD_CER})\n")
    print("signal correlation with tesseract CER (Spearman):")
    for k in ("conf", "dict", "agree", "chars"):
        print(f"  {k:<6} {spearman(np.array([r[k] for r in rows]), cer):+.3f}")

    print("\nrule sweep: escalate if conf<C or dict<D   (recall of bad pages | pages escalated on tuning set)")
    bad = cer > BAD_CER
    grid_c = [55, 60, 65, 70, 75, 80, 85]
    grid_d = [0.5, 0.6, 0.7, 0.8]
    best = []
    for C in grid_c:
        for D in grid_d:
            esc = np.array([(r["conf"] < C) or (r["dict"] < D) for r in rows])
            rec = (esc & bad).sum() / max(1, bad.sum())
            best.append((rec, esc.sum(), C, D))
    for rec, nesc, C, D in sorted(best, key=lambda t: (-t[0], t[1]))[:8]:
        print(f"  conf<{C:<3} or dict<{D:<4} -> recall {rec:.2f}, escalates {nesc}/{len(rows)}")

    # corpus-wide volume for a candidate rule
    C, D = 75, 0.7
    total = esc_n = 0
    for rep in load_reports():
        rdir = report_dir(rep["ntrs_id"]) / "ocr"
        for oj in rdir.glob("[0-9]*.json"):
            o = json.loads(oj.read_text())
            confs = [w["conf"] for w in o["words"]]
            c = float(np.mean(confs)) if confs else 0.0
            d = dict_share(o["text"], WORDS)[2]
            total += 1
            if c < C or d < D:
                esc_n += 1
    print(f"\ncorpus projection for conf<{C} or dict<{D}: {esc_n}/{total} pages "
          f"({esc_n/total:.1%}) would escalate to the vision tier")

    print("\npages the rule (conf<75 or dict<0.7) misses, with their CER:")
    for r in rows:
        if r["cer"] > BAD_CER and not (r["conf"] < 75 or r["dict"] < 0.7):
            print(f'  MISSED {r["id"]}: cer {r["cer"]:.2f}, conf {r["conf"]:.0f}, dict {r["dict"]:.2f}, agree {r["agree"]:.2f}')


if __name__ == "__main__":
    main()
