"""Stratified character-error analysis of the campaign 8B engine vs ground truth.

Decomposes normalized-char edits into substitutions / deletions / insertions
(difflib opcodes; inside an unequal replace span the overlap counts as S and
the surplus as D or I — an approximation of true editops, adequate for
aggregate rates and confusion mining), mines the top confusion pairs, breaks
out digit-involved substitutions, and stratifies by page class (table /
figure / prose, same thresholds the campaign gate uses).

Run from the repo root:  .venv/bin/python bench/error_analysis.py
"""
import difflib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from score import normalize  # noqa: E402
import table_verify  # noqa: E402

ENGINE = "vision_8blora_v2_1"
DIGIT_THRESHOLD = 0.15   # pipeline/unit_assimilate.py gate
FIGURE_THRESHOLD = 0.2   # workgen gate v1


def sdi(gt: str, hyp: str) -> dict:
    """Char-level S/D/I counts, confusion pairs, digit-involved subs."""
    a, b = list(gt), list(hyp)
    S = D = I = digit_S = 0
    confusions = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        la, lb = i2 - i1, j2 - j1
        if op == "replace":
            n = min(la, lb)
            S += n
            for ga, gb in zip(a[i1:i1 + n], b[j1:j1 + n]):
                confusions.append((ga, gb))
                if ga.isdigit() or gb.isdigit():
                    digit_S += 1
            if la > lb:
                D += la - lb
            else:
                I += lb - la
        elif op == "delete":
            D += la
        elif op == "insert":
            I += lb
    return {"S": S, "D": D, "I": I, "digit_S": digit_S,
            "confusions": confusions, "gt_chars": len(a)}


def page_class(gt_text: str, figure_area: float) -> str:
    if table_verify.digit_fraction(gt_text) > DIGIT_THRESHOLD:
        return "table"
    if figure_area > FIGURE_THRESHOLD:
        return "figure"
    return "prose"


def _figure_area(doc: str, page: int) -> float:
    p = ROOT / "data" / doc / "layout" / f"{page}.json"
    try:
        return sum(f["area_frac"] for f in json.loads(p.read_text())["figures"])
    except (OSError, ValueError, KeyError):
        return 0.0


def main() -> int:
    latest = {}
    for line in (ROOT / "data" / "scores.jsonl").open():
        r = json.loads(line)
        latest[(r["ntrs_id"], r["page"])] = r
    pages = []
    for (doc, page), row in sorted(latest.items()):
        gt_p = ROOT / "data" / doc / "ground_truth" / f"{page}.txt"
        hyp_p = ROOT / "data" / doc / ENGINE / f"{page}.txt"
        if not (gt_p.exists() and hyp_p.exists()):
            print(f"skip {doc} p{page}: missing text", file=sys.stderr)
            continue
        gt = normalize(gt_p.read_text())
        hyp = normalize(hyp_p.read_text())
        r = sdi(gt, hyp)
        r.update(doc=doc, page=page, gt_status=row["gt_status"],
                 cls=page_class(gt, _figure_area(doc, page)),
                 cer=(r["S"] + r["D"] + r["I"]) / max(1, r["gt_chars"]))
        pages.append(r)

    tot = {k: sum(p[k] for p in pages) for k in ("S", "D", "I", "digit_S", "gt_chars")}
    conf = Counter(c for p in pages for c in p["confusions"])
    digit_conf = Counter(c for c, n in conf.items()
                         if c[0].isdigit() or c[1].isdigit() for _ in range(n))

    out = [f"# 8B error analysis — {len(pages)} GT pages, engine {ENGINE}", ""]
    out.append(f"Totals over {tot['gt_chars']:,} GT chars: "
               f"S={tot['S']:,} D={tot['D']:,} I={tot['I']:,} "
               f"(rates {tot['S']/tot['gt_chars']:.4f} / "
               f"{tot['D']/tot['gt_chars']:.4f} / {tot['I']/tot['gt_chars']:.4f})")
    out.append(f"Digit-involved substitutions: {tot['digit_S']:,} "
               f"({tot['digit_S']/max(1,tot['S']):.0%} of S)")
    out.append("")
    out.append("| class | pages | GT chars | S rate | D rate | I rate | median CER |")
    out.append("|---|---|---|---|---|---|---|")
    for cls in ("prose", "table", "figure"):
        grp = [p for p in pages if p["cls"] == cls]
        if not grp:
            continue
        g = {k: sum(p[k] for p in grp) for k in ("S", "D", "I", "gt_chars")}
        med = sorted(p["cer"] for p in grp)[len(grp) // 2]
        out.append(f"| {cls} | {len(grp)} | {g['gt_chars']:,} "
                   f"| {g['S']/g['gt_chars']:.4f} | {g['D']/g['gt_chars']:.4f} "
                   f"| {g['I']/g['gt_chars']:.4f} | {med:.4f} |")
    out.append("")
    out.append("Top confusion pairs (gt→hyp):")
    for (ga, gb), n in conf.most_common(25):
        out.append(f"- {ga!r} → {gb!r} ×{n}")
    out.append("")
    out.append("Top digit confusions:")
    for (ga, gb), n in digit_conf.most_common(15):
        out.append(f"- {ga!r} → {gb!r} ×{n}")
    out.append("")
    out.append("Worst pages (CER):")
    for p in sorted(pages, key=lambda p: -p["cer"])[:8]:
        out.append(f"- {p['doc']} p{p['page']} [{p['cls']}] CER {p['cer']:.4f} "
                   f"(S={p['S']} D={p['D']} I={p['I']}, {p['gt_chars']} chars)")

    report = "\n".join(out) + "\n"
    dest = ROOT / "bench" / "ERROR-ANALYSIS-8B.md"
    dest.write_text(report)
    print(report)
    print(f"written: {dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
