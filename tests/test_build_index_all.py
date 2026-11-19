import csv
import json
import sqlite3

import pytest

from build_index import build, ingested_ids, roster_meta

HDR = ["ntrs_id", "report_number", "title", "series", "published_year",
       "pdf_url", "no_pdf", "in_pilot", "pub_date", "decade"]


def _row(i, rn, title):
    return dict(ntrs_id=i, report_number=rn, title=title, series="TR",
                published_year="1919", pdf_url=f"http://x/{i}.pdf", no_pdf="False",
                in_pilot="False", pub_date="1919-01-01", decade="1910s")


def _doc(data, i, ocr=True, pages=2):
    pd = data / i / "pages"
    pd.mkdir(parents=True)
    for n in range(1, pages + 1):
        (pd / f"{n}.png").write_bytes(b"x")
        if ocr:
            (data / i / "ocr").mkdir(exist_ok=True)
            (data / i / "ocr" / f"{n}.json").write_text(
                json.dumps({"text": f"text {i} p{n}", "config_hash": "t"}))


@pytest.fixture
def tmp_tree(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    roster = data / "roster.csv"
    rows = [_row("111", "NACA-TR-1", "Alpha Title"), _row("222", "NACA-TR-2", "Beta Title"),
            _row("333", "NACA-TR-3", "Gamma Title")]
    with open(roster, "w", newline="") as f:
        w = csv.DictWriter(f, HDR)
        w.writeheader()
        w.writerows(rows)
    _doc(data, "111")
    _doc(data, "222")
    (data / "notadoc").mkdir()  # no pages/: not ingested
    return data, roster, tmp_path / "c.db"


def _go(tree):
    data, roster, db = tree
    ids = ingested_ids(data)
    return build(db, data, ids=ids, reports=roster_meta(roster, ids),
                 embed=False), db


def test_ingested_ids_sorted_pages_only(tmp_tree):
    assert ingested_ids(tmp_tree[0]) == ["111", "222"]


def test_all_ingested_uses_roster_metadata(tmp_tree):
    _, db = _go(tmp_tree)
    c = sqlite3.connect(db)
    rows = dict((r[0], r[1:]) for r in c.execute(
        "SELECT doc_id, report_number, title FROM reports"))
    assert rows == {"111": ("NACA-TR-1", "Alpha Title"),
                    "222": ("NACA-TR-2", "Beta Title")}
    assert c.execute("SELECT COUNT(*) FROM pages").fetchone()[0] == 4


def test_ingested_id_missing_from_roster_is_fatal(tmp_tree):
    data, roster, _ = tmp_tree
    (data / "999" / "pages").mkdir(parents=True)
    with pytest.raises(SystemExit, match="999"):
        roster_meta(roster, ingested_ids(data))


def test_doc_with_pages_but_no_text_builds_clean(tmp_tree):
    data, roster, db = tmp_tree
    _doc(data, "333", ocr=False)
    counts, db = _go(tmp_tree)
    c = sqlite3.connect(db)
    assert c.execute("SELECT COUNT(*) FROM pages WHERE doc_id='333'").fetchone()[0] == 0
    assert c.execute("SELECT COUNT(*) FROM pages WHERE doc_id='111'").fetchone()[0] == 2
