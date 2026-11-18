"""Per-page, per-step pipeline timing report.

Collects the "seconds" field each pass records:
  pass1  data/{id}/meta.json timings (per report: rasterize, baseline)
  pass2  layout/{n}.json
  pass3a ocr/{n}.json
  pass4a vision*/{n}.meta.json  (one row per engine dir; retries included)
Run: python3 pipeline/timing.py [--slowest N]
"""
import argparse, json, statistics as st, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import load_reports, report_dir


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slowest", type=int, default=5)
    args = ap.parse_args()

    steps = {}          # name -> list[(seconds, "id pN")]
    rep_timings = []    # pass1 per-report
    total_pages = 0
    for rep in load_reports():
        nid = rep["ntrs_id"]
        d = report_dir(nid)
        if not (d / "pages").is_dir():
            continue
        mp = d / "meta.json"
        if mp.exists():
            t = json.loads(mp.read_text()).get("timings")
            if t:
                rep_timings.append(t)
        pages = sorted(int(p.name.split(".")[0]) for p in (d / "pages").glob("*.png")
                       if ".raw" not in p.name)
        total_pages += len(pages)
        for n in pages:
            oj = d / "ocr" / f"{n}.json"
            if oj.exists():
                s = json.loads(oj.read_text()).get("seconds")
                if s is not None:
                    steps.setdefault("3a tesseract", []).append((s, f"{nid} p{n}"))
            lj = d / "layout" / f"{n}.json"
            if lj.exists():
                s = json.loads(lj.read_text()).get("seconds")
                if s is not None:
                    steps.setdefault("2 layout", []).append((s, f"{nid} p{n}"))
        for vd in sorted(d.glob("vision*")):
            if not vd.is_dir():
                continue
            for mj in vd.glob("*.meta.json"):
                m = json.loads(mj.read_text())
                if "seconds" in m:
                    steps.setdefault(f"4a {vd.name}", []).append(
                        (m["seconds"], f"{nid} p{mj.name.split('.')[0]}"))

    print(f"corpus: {total_pages} pages\n")
    if rep_timings:
        for key in ("rasterize_s", "baseline_s"):
            per = [t[key] / max(1, t["pages"]) for t in rep_timings]
            print(f"1 {key.removesuffix('_s'):<12} {len(rep_timings):>5} reports  "
                  f"median {st.median(per):6.2f}s/page")
    for name in sorted(steps):
        xs = [s for s, _ in steps[name]]
        cov = len(xs) / total_pages
        print(f"{name:<15} {len(xs):>5} pages ({cov:5.1%})  "
              f"median {st.median(xs):6.2f}s  mean {st.mean(xs):6.2f}s  "
              f"p90 {pct(xs, 0.90):6.2f}s  total {sum(xs)/3600:6.2f} h")
    for name in sorted(steps):
        rows = sorted(steps[name], reverse=True)[:args.slowest]
        print(f"\nslowest {name}: " + "; ".join(f"{i} {s:.1f}s" for s, i in rows))


if __name__ == "__main__":
    main()
