"""Cross-engine agreement analysis between two vision transcription layers.

For every page transcribed by both engines, computes the inter-engine CER
(Levenshtein over normalized text / max length) and buckets pages:
  agree    -> pseudo-label training candidates / reconcile auto-accept
  minor    -> reconcile decides per page
  conflict -> human-review queue, worst first

Calibration mode (--calibrate) restricts to pages with ground truth and
prints inter-engine CER next to each engine's true CER, so the bucket
thresholds can be chosen from data rather than guessed.
"""

import argparse
import json
from pathlib import Path

from common import DATA
from score import NORM_HASH, levenshtein, normalize

DIR_A = "vision_27bq4"
DIR_B = "vision_haiku2"
T_AGREE = 0.02
T_CONFLICT = 0.10


def page_agreement(a_text: str, b_text: str) -> dict:
    na, nb = normalize(a_text), normalize(b_text)
    denom = max(len(na), len(nb))
    cer = levenshtein(na, nb) / denom if denom else 0.0
    return {"cer": round(cer, 4), "a_chars": len(na), "b_chars": len(nb)}


def bucket(cer: float, t_agree: float = T_AGREE, t_conflict: float = T_CONFLICT) -> str:
    if cer <= t_agree:
        return "agree"
    if cer <= t_conflict:
        return "minor"
    return "conflict"


def _ok_page(rdir: Path, vdir: str, n: int) -> str | None:
    """Return the page text if present and not vision_failed, else None."""
    txt = rdir / vdir / f"{n}.txt"
    meta = rdir / vdir / f"{n}.meta.json"
    if not txt.exists() or not meta.exists():
        return None
    try:
        m = json.loads(meta.read_text())
    except json.JSONDecodeError:
        return None
    if (m.get("loop_guard") or {}).get("vision_failed"):
        return None
    return txt.read_text()


def collect_pages(data_root: Path, dir_a: str = DIR_A, dir_b: str = DIR_B):
    """Yield (ntrs_id, page_n, text_a, text_b) for pages present in both engines."""
    for rdir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        adir = rdir / dir_a
        if not adir.is_dir() or not (rdir / dir_b).is_dir():
            continue
        for txt in sorted(adir.glob("*.txt"), key=lambda p: int(p.stem)):
            n = int(txt.stem)
            a = _ok_page(rdir, dir_a, n)
            b = _ok_page(rdir, dir_b, n)
            if a is not None and b is not None:
                yield rdir.name, n, a, b


def run(data_root: Path, out_path: Path | None, t_agree: float, t_conflict: float,
        dir_a: str = DIR_A, dir_b: str = DIR_B) -> dict:
    rows = []
    for ntrs_id, n, a, b in collect_pages(data_root, dir_a, dir_b):
        r = page_agreement(a, b)
        r.update(doc=ntrs_id, page=n,
                 bucket=bucket(r["cer"], t_agree, t_conflict))
        rows.append(r)
    if out_path:
        with out_path.open("w") as f:
            f.write(json.dumps({"norm_hash": NORM_HASH, "dir_a": dir_a, "dir_b": dir_b,
                                "t_agree": t_agree, "t_conflict": t_conflict}) + "\n")
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return summarize(rows)


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"pages": 0}
    cers = sorted(r["cer"] for r in rows)
    buckets = {}
    for r in rows:
        buckets[r["bucket"]] = buckets.get(r["bucket"], 0) + 1
    worst = sorted(rows, key=lambda r: -r["cer"])[:15]
    return {
        "pages": len(rows),
        "median_cer": cers[len(cers) // 2],
        "mean_cer": round(sum(cers) / len(cers), 4),
        "buckets": buckets,
        "worst": [(r["doc"], r["page"], r["cer"]) for r in worst],
    }


def calibrate(data_root: Path) -> list[dict]:
    """Pages with GT + both engines: inter-engine CER vs each engine's true CER."""
    out = []
    for ntrs_id, n, a, b in collect_pages(data_root):
        gt_p = data_root / ntrs_id / "ground_truth" / f"{n}.txt"
        if not gt_p.exists():
            continue
        gt = normalize(gt_p.read_text())
        na, nb = normalize(a), normalize(b)
        if not gt:
            continue
        out.append({
            "doc": ntrs_id, "page": n,
            "inter_cer": page_agreement(a, b)["cer"],
            "cer_a": round(levenshtein(gt, na) / len(gt), 4),
            "cer_b": round(levenshtein(gt, nb) / len(gt), 4),
        })
    return out


def resolve_out(out_arg: str | None, dir_a: str, dir_b: str) -> Path:
    """The canonical agreement.jsonl belongs to the default engine pair only."""
    if out_arg:
        return Path(out_arg)
    if (dir_a, dir_b) == (DIR_A, DIR_B):
        return DATA / "agreement.jsonl"
    return DATA / f"agreement_{dir_a}__{dir_b}.jsonl"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None,
                    help="output jsonl (default: canonical for default dirs, "
                         "agreement_<a>__<b>.jsonl for custom dirs)")
    ap.add_argument("--dir-a", default=DIR_A)
    ap.add_argument("--dir-b", default=DIR_B)
    ap.add_argument("--t-agree", type=float, default=T_AGREE)
    ap.add_argument("--t-conflict", type=float, default=T_CONFLICT)
    ap.add_argument("--calibrate", action="store_true",
                    help="GT pages only: inter-engine CER vs true CER per engine")
    args = ap.parse_args()

    if args.calibrate:
        rows = sorted(calibrate(DATA), key=lambda r: r["inter_cer"])
        print(f"{'doc':<13}{'pg':>5}  {'inter':>7}  {'cer_27b':>8}  {'cer_haiku':>9}")
        for r in rows:
            print(f"{r['doc']:<13}{r['page']:>5}  {r['inter_cer']:>7.4f}  "
                  f"{r['cer_a']:>8.4f}  {r['cer_b']:>9.4f}")
        return

    s = run(DATA, resolve_out(args.out, args.dir_a, args.dir_b),
            args.t_agree, args.t_conflict, dir_a=args.dir_a, dir_b=args.dir_b)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
