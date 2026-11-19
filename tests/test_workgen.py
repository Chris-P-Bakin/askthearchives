import json

import pytest

import common
import pass4a
import units
import workgen


def _doc(root, nid, pages):
    """pages: {n: (conf, figure_frac_or_None)}"""
    for n, (conf, frac) in pages.items():
        (root / nid / "ocr").mkdir(parents=True, exist_ok=True)
        (root / nid / "pages").mkdir(parents=True, exist_ok=True)
        (root / nid / "pages" / f"{n}.png").write_bytes(f"{nid}{n}".encode())
        (root / nid / "ocr" / f"{n}.json").write_text(
            json.dumps({"words": [{"conf": conf}, {"conf": conf}]}))
        if frac is not None:
            (root / nid / "layout").mkdir(exist_ok=True)
            (root / nid / "layout" / f"{n}.json").write_text(
                json.dumps({"figures": [{"area_frac": frac}]}))


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "DATA", tmp_path)
    # D1: p1 low conf (in), p2 good (out), p3 big figure (in), p10 exact thresholds (out)
    _doc(tmp_path, "D1", {1: (50, None), 2: (90, 0.1), 3: (90, 0.5), 10: (70, 0.2)})
    # D2: p1 done with accepted hash, p2 stale hash, both gated
    _doc(tmp_path, "D2", {1: (10, None), 2: (10, None)})
    for n, h in ((1, "6ecf7d0dbeb2"), (2, "oldhash")):
        d = tmp_path / "D2" / units.CAMPAIGN_LAYER
        d.mkdir(exist_ok=True)
        (d / f"{n}.meta.json").write_text(json.dumps({"config_hash": h}))
    return tmp_path


def test_gated_pages_thresholds(tree):
    assert workgen.gated_pages(["D1"]) == [("D1", 1), ("D1", 3)]


def test_pending_units_skips_done(tree):
    us = workgen.pending_units({"ids": ["D1", "D2"]})
    keys = [(u["ntrs_id"], u["page"]) for u in us]
    assert ("D2", 1) not in keys
    assert ("D2", 2) in keys
    assert keys == [("D1", 1), ("D1", 3), ("D2", 2)]
    assert all(u["config_hash"] == "6ecf7d0dbeb2" for u in us)


def test_workgen_idempotent(tree):
    m = {"ids": ["D1", "D2"]}
    assert workgen.pending_units(m) == workgen.pending_units(m)


def test_escalated_pages_wrapper_unchanged(tree, tmp_path, monkeypatch):
    csv = tmp_path / "reports.csv"
    csv.write_text("ntrs_id\nD1\nD2\n")
    monkeypatch.setenv("NACA_REPORTS_CSV", str(csv))
    ids = [r["ntrs_id"] for r in common.load_reports()]
    assert pass4a.escalated_pages() == workgen.gated_pages(ids)
    assert pass4a.escalated_pages() == [("D1", 1), ("D1", 3), ("D2", 1), ("D2", 2)]


def test_cli_writes_jsonl(tree, monkeypatch):
    (tree / "batches").mkdir()
    (tree / "batches" / "batch_007.json").write_text(json.dumps({"batch": 7, "ids": ["D1"]}))
    workgen.main(["--batch", "7"])
    out = tree / "units" / "batch_007_units.jsonl"
    assert [u["unit_id"] for u in units.read_units(out)] == ["D1-p1", "D1-p3"]
