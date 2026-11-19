import importlib.util
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "catalog_build_mod", ROOT / "pipeline" / "catalog_build.py")
cb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cb)

RG = {"naId": 582, "title": "Records of NASA", "levelOfDescription": "recordGroup",
      "inclusiveStartDate": {"year": 1903}, "inclusiveEndDate": {"year": 2006}}


def _anc(*pairs):
    return [{"naId": n, "levelOfDescription": lv, "title": f"t{n}", "distance": d}
            for n, lv, d in pairs]


def _item(na_id, parent, digitized, restricted=False):
    r = {"naId": na_id, "title": f"Item {na_id}", "levelOfDescription": "item",
         "ancestors": _anc((parent, "series", 1), (582, "recordGroup", 2)),
         "useRestriction": {"status": "Restricted - Fully" if restricted else "Unrestricted"}}
    if digitized:
        r["digitalObjects"] = [{"objectType": "PDF", "objectUrl": "http://x/a.pdf"}]
    return r


def _write(d, recs, extra=()):
    d.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"record": r}) for r in recs] + list(extra)
    (d / "a.jsonl").write_text("\n".join(lines) + "\n")


def _fixture(d):
    s1 = {"naId": 10, "title": "Speech Files", "levelOfDescription": "series",
          "scopeAndContentNote": "Correspondence about wind tunnels.",
          "ancestors": _anc((582, "recordGroup", 1)),
          "inclusiveStartDate": {"year": 1968}, "inclusiveEndDate": {"year": 1969}}
    s2 = {"naId": 20, "title": "Budget Files", "levelOfDescription": "series",
          "ancestors": _anc((582, "recordGroup", 1))}
    _write(d, [RG, s1, s2, _item(101, 10, True), _item(102, 10, False),
               _item(103, 20, True), _item(104, 20, True, restricted=True)])


def _rows(db, sql):
    con = sqlite3.connect(db)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def test_build_levels_and_aggregates(tmp_path):
    _fixture(tmp_path / "m")
    db = tmp_path / "c.db"
    counts = cb.build(db, {"nara-rg255": tmp_path / "m"}, {"nara-rg255"})
    a = {r[0]: r for r in _rows(db, "select naId, level, n_items, n_digitized, parent_naId, "
                                    "collection, start_year, end_year from archives")}
    assert a[582][1:4] == ("recordGroup", 4, 3)
    assert a[10][1:5] == ("series", 2, 1, 582)
    assert a[20][2:4] == (2, 2)
    assert a[10][5] == "nara-rg255" and a[10][6:] == (1968, 1969)
    assert len(_rows(db, "select * from items")) == 4
    it = _rows(db, "select doc_id, naId, digitized, rights from items where naId=101")[0]
    assert it == ("nara255-101", 101, 1, "Unrestricted")
    assert counts["items"] == 4
    # naca collection-level row comes from the registry
    assert _rows(db, "select level, collection from archives where collection='naca'")[0][0] == "collection"


def test_items_only_for_ingested(tmp_path):
    _fixture(tmp_path / "m")
    db = tmp_path / "c.db"
    cb.build(db, {"nara-rg255": tmp_path / "m"}, set())
    assert _rows(db, "select count(*) from items") == [(0,)]
    assert _rows(db, "select n_items from archives where naId=582") == [(4,)]


def test_catalog_build_skips_corrupt_lines(tmp_path):
    d = tmp_path / "m"
    _write(d, [RG], extra=["{not json", json.dumps({"record": {"title": "no id"}}), ""])
    counts = cb.build(tmp_path / "c.db", {"nara-rg255": d}, {"nara-rg255"})
    assert counts["corrupt"] == 2
    assert _rows(tmp_path / "c.db", "select count(*) from archives where naId=582") == [(1,)]


def test_rebuild_is_idempotent(tmp_path):
    _fixture(tmp_path / "m")
    db = tmp_path / "c.db"
    for _ in range(2):
        cb.build(db, {"nara-rg255": tmp_path / "m"}, {"nara-rg255"})
    assert _rows(db, "select count(*) from items") == [(4,)]
    assert _rows(db, "select count(*) from archives where naId=10") == [(1,)]


def test_fts_search_scope_note(tmp_path):
    _fixture(tmp_path / "m")
    db = tmp_path / "c.db"
    cb.build(db, {"nara-rg255": tmp_path / "m"}, {"nara-rg255"})
    hit = _rows(db, "select a.naId from archives_fts f join archives a on a.rowid=f.rowid "
                    "where archives_fts match 'tunnels'")
    assert hit == [(10,)]
    assert _rows(db, "select count(*) from items_fts where items_fts match 'Item'") == [(4,)]
