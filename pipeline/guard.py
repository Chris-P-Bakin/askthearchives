"""Loop/degeneration guard for VLM page transcriptions.

Shared by pass4a, the validator and the worker. Stdlib only: must import
standalone (no data/ tree, no env, no imports from common).

LOOP_GUARD values are part of pass4a's CONFIG and therefore the campaign
config hash: never change a value here without a deliberate hash bump.
"""
import zlib

LOOP_GUARD = {
    "version": "v2",
    "min_chars": 400,            # shorter outputs are never judged
    "ratio_floor": 0.15,         # zlib(level 6) compressed/raw below this = loop
    "length_ratio_floor": 0.25,  # stricter floor when finish_reason == "length"
    "retry_temperature": 0.4,
    # v2: pseudo-table junk (chart-axis labels hallucinated into a giant table
    # or one runaway numeric line), judged only when finish_reason == "length"
    "pipe_line_floor": 0.4,      # fraction of non-empty lines containing "|" must exceed this
    "unique_floor": 0.6,         # AND unique-line fraction must be below this
    "runaway_line_chars": 2000,  # OR one numeric-only line at least this long...
    "runaway_line_share": 0.5,   # ...making up more than this share of the text
    "runaway_numeric_share": 0.95,  # ...and at least this share digits/.,- /space
}


def zratio(text: str, g: dict | None = None):
    b = text.encode()
    if len(b) < (g or LOOP_GUARD)["min_chars"]:
        return None
    return len(zlib.compress(b, 6)) / len(b)


def junk_reason(text: str, g: dict | None = None) -> str | None:
    """Chart-axis labels hallucinated into a table (or one runaway numeric line).

    Returns "pipe_table", "runaway" or None.

    Pipe form: most lines carry pipes yet few lines are unique (zlib can miss
    this when rows differ in digits). Raw lines, not digit-normalized: genuine
    dense numeric tables repeat line *shapes* (0.094 unique when normalized)
    but not line contents (1.0 raw), so normalizing would flag real tables.
    Runaway form: a single numeric/space-only line dominating the text.
    Every threshold comes from `g` (default LOOP_GUARD, hashed in CONFIG)."""
    g = g or LOOP_GUARD
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return None
    pipe_frac = sum("|" in l for l in lines) / len(lines)
    if pipe_frac > g["pipe_line_floor"] and len(set(lines)) / len(lines) < g["unique_floor"]:
        return "pipe_table"
    longest = max(lines, key=len)
    if (len(longest) >= g["runaway_line_chars"]
            and len(longest) / max(1, len(text.strip())) > g["runaway_line_share"]
            and sum(c in "0123456789.,- " for c in longest) / len(longest)
            > g["runaway_numeric_share"]):
        return "runaway"
    return None


def pseudo_table_junk(text: str, guard: dict | None = None) -> bool:
    return junk_reason(text, guard) is not None


def degenerate(text: str, finish_reason, g: dict | None = None) -> tuple[bool, float | None, str | None]:
    """(bad, zlib ratio, reason); reason is "zlib" | "pipe_table" | "runaway" | None."""
    g = g or LOOP_GUARD
    r = zratio(text, g)
    if r is None:
        return False, None, None
    r = round(r, 4)
    if r < g["ratio_floor"]:
        return True, r, "zlib"
    if finish_reason == "length" and r < g["length_ratio_floor"]:
        return True, r, "zlib"
    if finish_reason == "length":
        reason = junk_reason(text, g)
        if reason:
            return True, r, reason
    return False, r, None
