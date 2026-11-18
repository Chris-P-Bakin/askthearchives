"""Score engines on the staged By the People pages (see prep.py).

Same metric code and frozen NORM as pipeline/score.py (CER, WER), plus a small
GT-side pass for BTP markup that NACA GT never has:
  [?] / [illegible]      -> removed (like <?> in NACA GT)
  [word?]                -> word (volunteer's best reading kept)
Other brackets stay as text, and the share of GT pages that contain any are
reported so the effect can be judged. Term recall is skipped: the NACA term
list says nothing about legal reports.

Writes data/btp/scores.jsonl (append) and data/btp/scores_latest.json, never
data/scores.jsonl, so BENCHMARKS.md is untouched.

Usage: python bench/btp/score_btp.py [--pages data/btp/pages.jsonl]
"""
import argparse, json, pathlib, re, statistics, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))
from score import NORM_HASH, engine_texts, levenshtein, normalize, wer  # noqa: E402
from common import DATA, report_dir  # noqa: E402

OUT = DATA / "btp"
_ILLEG = re.compile(r"\[\s*(\?|illegible)\s*\]", re.I)
_UNSURE = re.compile(r"\[([^\[\]\n]{1,60}?)\?\]")


def btp_gt(text: str) -> str:
    t = _ILLEG.sub(" ", text)
    return _UNSURE.sub(r"\1", t)


def score_page(doc: str, n: int) -> dict:
    raw = (report_dir(doc) / "ground_truth" / f"{n}.txt").read_text(errors="replace")
    gt = normalize(btp_gt(raw))
    res = {"doc": doc, "page": n, "gt_chars": len(gt), "norm_hash": NORM_HASH,
           "gt_has_brackets": "[" in btp_gt(raw), "engines": {}}
    for eng, hyp_raw in engine_texts(doc, n).items():
        hyp = normalize(hyp_raw)
        res["engines"][eng] = {"cer": round(levenshtein(gt, hyp) / max(1, len(gt)), 4),
                               "wer": round(wer(gt, hyp), 4), "hyp_chars": len(hyp)}
    return res


def summarize(results: list) -> dict:
    by = {}
    for r in results:
        for eng, m in r["engines"].items():
            by.setdefault(eng, []).append(m)
    out = {}
    for eng, ms in sorted(by.items()):
        cers = sorted(m["cer"] for m in ms)
        out[eng] = {"n": len(ms), "median_cer": round(statistics.median(cers), 4),
                    "mean_cer": round(statistics.fmean(cers), 4),
                    "p75_cer": round(cers[min(len(cers) - 1, int(0.75 * len(cers)))], 4),
                    "median_wer": round(statistics.median(m["wer"] for m in ms), 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default=str(OUT / "pages.jsonl"))
    args = ap.parse_args()
    pages = [json.loads(l) for l in open(args.pages) if l.strip()]
    results = [score_page(p["doc"], int(p["page"])) for p in pages]
    summary = summarize(results)
    print(f'{"engine":<22} {"n":>3} {"med CER":>8} {"mean CER":>9} {"p75 CER":>8} {"med WER":>8}')
    for eng, s in summary.items():
        print(f'{eng:<22} {s["n"]:>3} {s["median_cer"]:>8.4f} {s["mean_cer"]:>9.4f}'
              f' {s["p75_cer"]:>8.4f} {s["median_wer"]:>8.4f}')
    nb = sum(r["gt_has_brackets"] for r in results)
    print(f"GT pages with other [brackets] left in: {nb}/{len(results)}  (norm {NORM_HASH})")
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "scores.jsonl", "a") as f:
        for r in results:
            f.write(json.dumps({"ts": ts, **r}) + "\n")
    (OUT / "scores_latest.json").write_text(json.dumps(
        {"ts": ts, "norm_hash": NORM_HASH, "summary": summary, "pages": results}, indent=1))


if __name__ == "__main__":
    main()
