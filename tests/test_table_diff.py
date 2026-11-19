import table_diff

PIPE = """\
| alpha | cl | cd |
|---|---|---|
| 0.1 | 0.205 | 0.0112 |
| 0.2 | 0.410 | 0.0134 |
| 0.3 | 0.612 | 0.0168 |
"""

PIPE_ONE_DIGIT = PIPE.replace("0.410", "0.419")

PIPE_DROPPED_ROW = """\
| alpha | cl | cd |
|---|---|---|
| 0.1 | 0.205 | 0.0112 |
| 0.3 | 0.612 | 0.0168 |
"""


def test_exponent_form_cells_parse():
    text = "| .4859 -4 | .5000 -4 | .5141 -4 |\n| .4859 -4 | .5000 -4 | .5141 -4 |\n| .1 | .2 | .3 |\n"
    rows = table_diff.extract_cells(text)
    assert rows, "expected at least one parsed row"
    assert abs(rows[0][0] - 0.4859e-4) < 1e-12


def test_identical_tables_zero_disagreements():
    a = table_diff.extract_cells(PIPE)
    d = table_diff.diff(a, table_diff.extract_cells(PIPE))
    assert d["cells_compared"] > 0
    assert d["disagreements"] == 0
    assert d["disagree_rate"] == 0.0


def test_single_wrong_digit_counted():
    d = table_diff.diff(table_diff.extract_cells(PIPE),
                        table_diff.extract_cells(PIPE_ONE_DIGIT))
    assert d["disagreements"] == 1


def test_dropped_row_does_not_misalign():
    d = table_diff.diff(table_diff.extract_cells(PIPE),
                        table_diff.extract_cells(PIPE_DROPPED_ROW))
    # rows 0.1 and 0.3 align cleanly; the missing 0.2 row is a gap, not a shift
    assert d["disagreements"] == 0
    assert d["cells_compared"] == 6


def test_empty_text_no_crash():
    d = table_diff.diff(table_diff.extract_cells(""), table_diff.extract_cells(PIPE))
    assert d["cells_compared"] == 0
    assert d["disagree_rate"] is None


def test_html_table_rows_extracted():
    html = ("<table><tr><td>0.1</td><td>0.205</td></tr>"
            "<tr><td>0.2</td><td>0.410</td></tr>"
            "<tr><td>0.3</td><td>0.612</td></tr></table>")
    rows = table_diff.extract_cells(html)
    assert len(rows) == 3
    assert rows[1][1] == 0.410
