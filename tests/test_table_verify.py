"""Table-consistency verifier: the attestation showed CER agreement is blind to
wrong table digits, so numeric tables get their own checks (monotonicity,
exponent sanity, cross-column identities) before their labels are trusted."""

import pytest

from table_verify import (
    parse_cell,
    parse_tables,
    check_monotonic,
    check_magnitude,
    check_identities,
    unrolled,
    verify_text,
)


# ---------- cell parsing ----------

def test_parse_cell_plain_numbers():
    assert parse_cell("365.3") == 365.3
    assert parse_cell("-.16") == -0.16
    assert parse_cell(".92") == 0.92
    assert parse_cell("338,000") == 338000.0
    assert parse_cell("17.70") == 17.70


def test_parse_cell_trailing_exponent():
    # NACA print convention: ".4859 -4" means 0.4859e-4
    assert parse_cell(".4859 -4") == pytest.approx(0.4859e-4)
    assert parse_cell(".1962 -3") == pytest.approx(0.1962e-3)
    assert parse_cell("2.430 2") == pytest.approx(243.0)


def test_parse_cell_non_numeric():
    assert parse_cell("M2") is None
    assert parse_cell(r"$\rho/\rho_t$") is None
    assert parse_cell("") is None
    assert parse_cell("---") is None


# ---------- table extraction ----------

TABLE_TEXT = """REPORT 1135

M | p/pt | T/Tt
---|---|---
1.0 | .5283 | .8333
1.2 | .4124 | .7764
1.4 | .3142 | .7184

CONFIDENTIAL
"""


def test_parse_tables_extracts_pipe_block():
    tables = parse_tables(TABLE_TEXT)
    assert len(tables) == 1
    rows = tables[0]
    assert len(rows) == 3          # header kept out (non-numeric), separator dropped
    assert rows[0] == [1.0, 0.5283, 0.8333]
    assert rows[2][1] == 0.3142


def test_parse_tables_prose_has_none():
    assert parse_tables("The quick brown fox.\nNo tables here at all.\n") == []


def test_parse_tables_keeps_modal_width_rows():
    text = "a | b | c\n1 | 2 | 3\n4 | 5 | 6\n7 | 8\n9 | 10 | 11\n"
    tables = parse_tables(text)
    assert len(tables) == 1
    assert all(len(r) == 3 for r in tables[0])
    assert len(tables[0]) == 3     # the 2-cell row is dropped


# ---------- monotonicity ----------

def test_monotonic_spike_flagged():
    col = [10.0, 9.0, 8.0, 7.0, 60.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    assert 4 in check_monotonic(col)


def test_monotonic_clean_column_unflagged():
    col = [float(x) for x in range(20, 0, -1)]
    assert check_monotonic(col) == []


def test_monotonic_short_or_unordered_skipped():
    assert check_monotonic([3.0, 1.0, 2.0]) == []            # too short
    assert check_monotonic([5.0, 1.0, 4.0, 2.0, 6.0, 0.0, 3.0, 7.0]) == []  # no direction


# ---------- exponent sanity ----------

def test_magnitude_hundredfold_cell_flagged():
    col = [4.8e-4, 4.7e-4, 4.6e-4, 4.5e-2, 4.4e-4, 4.3e-4, 4.2e-4, 4.1e-4]
    assert check_magnitude(col) == [3]


def test_magnitude_gradual_decade_span_unflagged():
    # smooth geometric decay across several decades: no local jumps
    col = [10 ** (-0.25 * i) for i in range(16)]
    assert check_magnitude(col) == []


# ---------- cross-column identities ----------

def _identity_cols(bad_row=None):
    import random
    rng = random.Random(7)
    b = [round(rng.uniform(0.1, 0.9), 4) for _ in range(10)]
    c = [round(rng.uniform(0.1, 0.9), 4) for _ in range(10)]
    a = [round(x * y, 6) for x, y in zip(b, c)]
    if bad_row is not None:
        a[bad_row] = a[bad_row] * 100   # exponent slip on one row
    return [a, b, c]


def test_identity_broken_row_flagged():
    flags = check_identities(_identity_cols(bad_row=6))
    assert any(6 in f["rows"] for f in flags)


def test_identity_clean_columns_unflagged():
    assert check_identities(_identity_cols()) == []


def test_identity_drifting_ratio_unflagged():
    # two columns whose ratio drifts smoothly (1.00 -> 1.06): a real physical
    # trend, not a transcription error — end rows deviate a few % from the
    # median ratio but nothing is an order-of-digit slip
    b = [round(1.0 + 0.05 * i, 4) for i in range(12)]
    a = [round(v * (1.0 + 0.005 * i), 6) for i, v in enumerate(b)]
    filler = [round(0.5 + 0.001 * i, 4) for i in range(12)]
    assert check_identities([a, b, filler]) == []


def test_identity_row_shift_flagged():
    # the 235B's attested failure mode: value columns slip one row mid-table
    cols = _identity_cols()
    a, b, c = cols
    b2 = b[:6] + b[5:-1]          # rows 6.. read from the row above
    c2 = c[:6] + c[5:-1]
    flags = check_identities([a, b2, c2])
    assert flags and all(set(f["rows"]) & set(range(6, 10)) for f in flags)


# ---------- end to end ----------

def test_verify_text_flags_bad_page():
    rows = ["M | p/pt | r/rt | T/Tt", "---|---|---|---"]
    import random
    rng = random.Random(3)
    for i in range(12):
        r = round(rng.uniform(0.1, 0.9), 4)
        t = round(rng.uniform(0.1, 0.9), 4)
        p = round(r * t, 6)
        if i == 5:
            p *= 100
        rows.append(f"{1 + 0.1 * i:.1f} | {p} | {r} | {t}")
    res = verify_text("\n".join(rows))
    assert res["tables"] == 1
    assert any(f["check"] == "identity" and 5 in f["rows"] for f in res["flags"])


def test_verify_text_prose_clean():
    res = verify_text("Just a prose page.\nNothing numeric worth checking.\n")
    assert res["tables"] == 0 and res["flags"] == []


# ---- unrolled (cell-per-line serialization failure) ----

def test_unrolled_detects_cell_per_line_dump():
    text = "\n".join(str(n % 977) for n in range(300))   # 300 short numeric lines
    assert unrolled(text) is True


def test_unrolled_ignores_normal_prose_and_tables():
    prose = "\n".join("This is an ordinary sentence of reasonable length." for _ in range(300))
    table = "\n".join("0.25 | 1.37 | 204.1 | 0.0082 | 19.6" for _ in range(300))
    assert unrolled(prose) is False
    assert unrolled(table) is False


def test_unrolled_needs_minimum_lines():
    assert unrolled("\n".join(str(n) for n in range(100))) is False  # under the 150-line floor


def test_verify_text_carries_unrolled_flag():
    text = "\n".join(str(n % 977) for n in range(300))
    flags = verify_text(text)["flags"]
    assert {"check": "unrolled"} in flags
