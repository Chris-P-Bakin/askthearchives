"""Cross-engine numeric table diff — the second-reader bench.

Two engines transcribe the same flagged table page independently; cells are
parsed under NACA print conventions (table_verify.parse_cell) and compared
after dropout-tolerant row alignment. Disagreement rate approximates the
fraction of numeric cells where at least one reader is wrong — the quantity
CER is blind to (see naca-table-label-poison).

CLI: table_diff.py --pages data/bench/table_pages.jsonl --a vision_glmocr \
         --b vision_8blora_v2_1
writes data/bench/table_diff_<a>_vs_<b>.jsonl (one row per page) and appends
a summary row to data/bench/table_diff_summary.jsonl (shape consumed by
bench_report.render_table_section: {pair, pages, cells_compared, disagree_rate}).
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import table_verify  # noqa: E402

_TR = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_TD = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S | re.I)


def _html_to_pipes(text: str) -> str:
    """Rewrite HTML table rows as markdown pipe rows so one parser serves both."""
    def row(m):
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in _TD.findall(m.group(1))]
        return "| " + " | ".join(cells) + " |\n" if cells else ""
    return _TR.sub(row, text)


def extract_cells(text: str) -> list[list[float | None]]:
    """All numeric table rows in a transcription, row-major.

    Lighter than table_verify.parse_tables (no 3-row minimum, no modal-width
    filter): a diff can use any row with >= 2 numeric cells, and ragged rows
    are tolerated by the positional compare."""
    if "<tr" in text.lower():
        text = re.sub(r"<[^>]+>", "\n", _html_to_pipes(text))
    rows = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.split("|")]
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        if not cells or table_verify._is_separator(cells):
            continue
        parsed = [table_verify.parse_cell(c) for c in cells]
        if sum(v is not None for v in parsed) >= 2:
            rows.append(parsed)
    return rows


def _row_match(a: list, b: list) -> int:
    return sum(1 for x, y in zip(a, b) if x is not None and y is not None and x == y)


def diff(cells_a: list[list], cells_b: list[list]) -> dict:
    """Align rows (LCS over cell-match scores — a dropped row becomes a gap,
    never a shift) and compare aligned cells positionally."""
    n, m = len(cells_a), len(cells_b)
    # LCS-style DP: pairing rows costs nothing, scores by matching cells.
    score = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            pair = _row_match(cells_a[i], cells_b[j])
            score[i][j] = max(score[i + 1][j], score[i][j + 1],
                              (pair + score[i + 1][j + 1]) if pair else 0)
    compared = disagreements = 0
    i = j = 0
    while i < n and j < m:
        pair = _row_match(cells_a[i], cells_b[j])
        if pair and score[i][j] == pair + score[i + 1][j + 1]:
            for x, y in zip(cells_a[i], cells_b[j]):
                if x is not None and y is not None:
                    compared += 1
                    disagreements += x != y
            i += 1
            j += 1
        elif score[i + 1][j] >= score[i][j + 1]:
            i += 1
        else:
            j += 1
    return {"cells_compared": compared, "disagreements": disagreements,
            "disagree_rate": round(disagreements / compared, 4) if compared else None}


def _read_layer(ntrs_id: str, page: int, layer: str) -> str | None:
    p = common.DATA / ntrs_id / layer / f"{page}.txt"
    return p.read_text(errors="replace") if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", required=True, help="jsonl with doc|ntrs_id + page")
    ap.add_argument("--a", required=True, help="first engine layer dir name")
    ap.add_argument("--b", required=True, help="second engine layer dir name")
    args = ap.parse_args()
    bench = common.DATA / "bench"
    bench.mkdir(exist_ok=True)
    out_rows, tot_c, tot_d, pages = [], 0, 0, 0
    for line in Path(args.pages).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        nid, pg = r.get("ntrs_id") or r["doc"], int(r["page"])
        ta, tb = _read_layer(nid, pg, args.a), _read_layer(nid, pg, args.b)
        if ta is None or tb is None:
            out_rows.append({"ntrs_id": nid, "page": pg, "skipped": "missing_layer",
                             "have_a": ta is not None, "have_b": tb is not None})
            continue
        d = diff(extract_cells(ta), extract_cells(tb))
        out_rows.append({"ntrs_id": nid, "page": pg, **d})
        pages += 1
        tot_c += d["cells_compared"]
        tot_d += d["disagreements"]
    detail = bench / f"table_diff_{args.a}_vs_{args.b}.jsonl"
    detail.write_text("".join(json.dumps(r) + "\n" for r in out_rows))
    summary = {"pair": f"{args.a} vs {args.b}", "pages": pages,
               "cells_compared": tot_c,
               "disagree_rate": round(tot_d / tot_c, 4) if tot_c else None}
    with open(bench / "table_diff_summary.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(json.dumps(summary))
    skipped = sum(1 for r in out_rows if r.get("skipped"))
    if skipped:
        print(f"skipped {skipped} pages missing a layer", file=sys.stderr)


if __name__ == "__main__":
    main()
