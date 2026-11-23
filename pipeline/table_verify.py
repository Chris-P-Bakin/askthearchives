"""Table-consistency verifier for transcribed numeric tables.

The 2026-11-12 attestation showed CER-based agreement is blind to wrong table
digits (a column with exponents off x100 is one character in thousands), so
digit-dense pages need checks that read the *numbers*:

  monotonic  -> tabulated functions step smoothly; a wrong digit breaks the
                column's dominant direction
  magnitude  -> exponent slips jump an order of magnitude against the cell's
                neighbors in the same column
  identity   -> physics ties columns together (e.g. p/pt = (rho/rho_t)(T/T_t));
                relations are *discovered* (a ~ k*b*c, a ~ k*b, a*b ~ k hold for
                most rows) and the rows that break them are flagged, which also
                catches row-shift errors

Cells use NACA print conventions: plain decimals, comma thousands, and the
trailing-exponent form ".4859 -4" meaning 0.4859e-4.

CLI scans one engine layer's digit-dense pages (same >15% digit-fraction gate
as distill_dataset.table_heavy_pages) and writes data/table_verify_<dir>.jsonl.
"""

import argparse
import json
import math
import re
import sys
from statistics import median

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import DATA

_NUM = re.compile(r"^[+-]?(?:\d[\d,]*\.?\d*|\.\d+)$")
_EXP = re.compile(r"^[+-]?\d{1,2}$")


def _num(tok: str) -> float | None:
    if not _NUM.match(tok):
        return None
    try:
        return float(tok.replace(",", ""))
    except ValueError:
        return None


def parse_cell(s: str) -> float | None:
    """A table cell as a number, or None. Handles ".4859 -4" = 0.4859e-4."""
    parts = s.strip().split()
    if len(parts) == 1:
        return _num(parts[0])
    if len(parts) == 2 and _EXP.match(parts[1]):
        m = _num(parts[0])
        if m is not None and "." in parts[0]:  # mantissa is a decimal, not a count
            return m * 10.0 ** int(parts[1])
    return None


def _is_separator(cells: list[str]) -> bool:
    return all(set(c) <= set("-: ") for c in cells)


def parse_tables(text: str) -> list[list[list[float | None]]]:
    """Numeric tables as row-major cell matrices.

    A table is a run of consecutive lines with >= 2 pipes; markdown separator
    rows are dropped; of the remaining rows, those with >= 2 numeric cells and
    the block's modal cell count form the matrix (headers and ragged OCR rows
    fall out)."""
    tables, block = [], []

    def flush():
        nonlocal block
        rows = []
        for cells in block:
            parsed = [parse_cell(c) for c in cells]
            if sum(v is not None for v in parsed) >= 2:
                rows.append(parsed)
        block = []
        if len(rows) < 3:
            return
        widths = {}
        for r in rows:
            widths[len(r)] = widths.get(len(r), 0) + 1
        modal = max(widths, key=lambda w: widths[w])
        rows = [r for r in rows if len(r) == modal]
        if len(rows) >= 3 and modal >= 2:
            tables.append(rows)

    for line in text.splitlines():
        if "|" in line:
            cells = [c.strip() for c in line.split("|")]
            if cells and cells[0] == "":
                cells = cells[1:]
            if cells and cells[-1] == "":
                cells = cells[:-1]
            if not _is_separator(cells):
                block.append(cells)
        else:
            flush()
    flush()
    return tables


def check_monotonic(col: list[float | None], min_rows: int = 8,
                    dominance: float = 0.85) -> list[int]:
    """Row indices stepping against the column's dominant direction."""
    vals = [(i, v) for i, v in enumerate(col) if v is not None]
    if len(vals) < min_rows:
        return []
    steps = [(i2, v2 - v1) for (i1, v1), (i2, v2) in zip(vals, vals[1:]) if v2 != v1]
    if not steps:
        return []
    ups = sum(1 for _, d in steps if d > 0)
    if max(ups, len(steps) - ups) / len(steps) < dominance:
        return []
    against = (lambda d: d < 0) if ups > len(steps) - ups else (lambda d: d > 0)
    return [i for i, d in steps if against(d)]


def check_magnitude(col: list[float | None], window: int = 3,
                    jump: float = 1.0) -> list[int]:
    """Rows whose order of magnitude jumps >= `jump` decades vs. their
    same-column neighbors (local, so smooth decade-spanning columns pass)."""
    vals = [(i, math.log10(abs(v))) for i, v in enumerate(col)
            if v is not None and v != 0]
    if len(vals) < 2 * window:
        return []
    flags = []
    for k, (i, lg) in enumerate(vals):
        neigh = [l for _, l in vals[max(0, k - window):k] + vals[k + 1:k + 1 + window]]
        if neigh and abs(lg - median(neigh)) >= jump:
            flags.append(i)
    return flags


def _relation_outliers(ratios: list[tuple[int, float]], tol: float,
                       flag_tol: float, support: float,
                       min_rows: int) -> list[int] | None:
    if len(ratios) < min_rows:
        return None
    k = median(r for _, r in ratios)
    if k == 0:
        return None
    devs = [(i, abs(r / k - 1)) for i, r in ratios]
    inliers = sum(1 for _, d in devs if d <= tol)
    # flag only gross breaks: a wrong digit or exponent slip is never a few-%
    # effect, while a slowly drifting physical ratio is
    bad = [i for i, d in devs if d >= flag_tol]
    if bad and inliers / len(ratios) >= support:
        return bad
    return None


def check_identities(cols: list[list[float | None]], tol: float = 0.02,
                     flag_tol: float = 0.10, support: float = 0.6,
                     min_rows: int = 6) -> list[dict]:
    """Relations a ~ k*b*c, a ~ k*b, a*b ~ k that hold (within `tol`) for most
    rows; the flags are rows breaking them by >= `flag_tol` (wrong digits,
    exponent slips, row shifts)."""
    n = len(cols)
    flags = []

    def ratios(f, idxs):
        out = []
        for row in range(min(len(cols[i]) for i in idxs)):
            vs = [cols[i][row] for i in idxs]
            if any(v is None for v in vs):
                continue
            r = f(*vs)
            if r is not None:
                out.append((row, r))
        return out

    def add(kind, idxs, f):
        bad = _relation_outliers(ratios(f, idxs), tol, flag_tol, support, min_rows)
        if bad:
            flags.append({"kind": kind, "cols": tuple(idxs), "rows": bad})

    for a in range(n):
        for b in range(n):
            if b == a:
                continue
            add("ratio", (a, b), lambda x, y: x / y if y else None)
            if b > a:
                add("product", (a, b), lambda x, y: x * y)
            for c in range(b + 1, n):
                if c == a:
                    continue
                add("triple", (a, b, c),
                    lambda x, y, z: x / (y * z) if y * z else None)
    return flags


def unrolled(text: str, min_lines: int = 150, cap: int = 8, frac: float = 0.7) -> bool:
    """Cell-per-line serialization failure: the engine walked a table or a
    figure's tick labels emitting one value per line. Calibrated 2026-11-17
    on the GT bench and a corpus sweep: catastrophic unrolls run 1,200+
    lines; the 150-line floor excludes figure pages whose true content is
    legitimately sparse short labels (title blocks, tick marks), which a
    lower floor falsely flags."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < min_lines:
        return False
    return sum(1 for l in lines if len(l) <= cap) / len(lines) > frac


def verify_text(text: str) -> dict:
    """All flags for one page's transcript."""
    tables = parse_tables(text)
    flags = []
    if unrolled(text):
        flags.append({"check": "unrolled"})
    for ti, rows in enumerate(tables):
        cols = [list(col) for col in zip(*rows)]
        for ci, col in enumerate(cols):
            for check, fn in (("monotonic", check_monotonic),
                              ("magnitude", check_magnitude)):
                bad = fn(col)
                if bad:
                    flags.append({"table": ti, "check": check, "col": ci,
                                  "rows": bad})
        for rel in check_identities(cols):
            flags.append({"table": ti, "check": "identity", **rel})
    return {"tables": len(tables), "flags": flags}


def digit_fraction(text: str) -> float:
    return sum(c.isdigit() for c in text) / len(text) if text else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True,
                    help="engine layer under data/{id}/ (e.g. vision_27bq4)")
    ap.add_argument("--digit-threshold", type=float, default=0.15,
                    help="scan only pages above this digit fraction")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out = args.out or str(DATA / f"table_verify_{args.dir}.jsonl")

    scanned = flagged = 0
    by_check: dict[str, int] = {}
    with open(out, "w") as f:
        f.write(json.dumps({"dir": args.dir,
                            "digit_threshold": args.digit_threshold}) + "\n")
        for rdir in sorted(p for p in DATA.iterdir() if p.is_dir()):
            vdir = rdir / args.dir
            if not vdir.is_dir():
                continue
            for txt in sorted(vdir.glob("*.txt"), key=lambda p: int(p.stem)):
                meta = vdir / f"{txt.stem}.meta.json"
                if meta.exists():  # vision_failed pages are already review-queued
                    try:
                        lg = json.loads(meta.read_text()).get("loop_guard") or {}
                        if lg.get("vision_failed"):
                            continue
                    except json.JSONDecodeError:
                        pass
                t = txt.read_text()
                if digit_fraction(t) <= args.digit_threshold:
                    continue
                scanned += 1
                res = verify_text(t)
                if not res["flags"]:
                    continue
                flagged += 1
                for fl in res["flags"]:
                    by_check[fl["check"]] = by_check.get(fl["check"], 0) + 1
                f.write(json.dumps({"doc": rdir.name, "page": int(txt.stem),
                                    "tables": res["tables"],
                                    "flags": res["flags"]}) + "\n")
    print(json.dumps({"scanned": scanned, "flagged": flagged,
                      "flags_by_check": by_check, "out": out}, indent=1))


if __name__ == "__main__":
    main()
