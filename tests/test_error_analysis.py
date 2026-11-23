import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
from error_analysis import page_class, sdi  # noqa: E402


def test_pure_substitution():
    r = sdi("cat", "cot")
    assert (r["S"], r["D"], r["I"]) == (1, 0, 0)
    assert r["confusions"] == [("a", "o")]


def test_pure_deletion_and_insertion():
    assert (lambda r: (r["S"], r["D"], r["I"]))(sdi("abcd", "ad")) == (0, 2, 0)
    assert (lambda r: (r["S"], r["D"], r["I"]))(sdi("ad", "abcd")) == (0, 0, 2)


def test_mixed_replace_span_counts_extra_as_del_or_ins():
    # "abc" -> "x": replace span of unequal length = 1 S + 2 D
    r = sdi("abc", "x")
    assert (r["S"], r["D"], r["I"]) == (1, 2, 0)


def test_digit_substitutions_counted():
    r = sdi("CL = 0.85", "CL = 0.35")
    assert r["S"] == 1 and r["digit_S"] == 1
    assert ("8", "3") in r["confusions"]


def test_identical_is_zero():
    r = sdi("same text", "same text")
    assert (r["S"], r["D"], r["I"]) == (0, 0, 0) and r["gt_chars"] == 9


def test_page_class_table_beats_figure_and_prose():
    digits = "12.3 45.6 78.9 " * 20
    assert page_class(digits, figure_area=0.0) == "table"
    assert page_class("plain prose here", figure_area=0.5) == "figure"
    assert page_class("plain prose here", figure_area=0.0) == "prose"
