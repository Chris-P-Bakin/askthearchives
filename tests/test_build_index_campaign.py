import csv
import json
import sqlite3

import pytest

import escalate
import units
from build_index import best_text, build

H235 = {"config": {"prompt_version": "file:x", "model": None, "loop_guard": {"version": "v2"}},
        "config_hash": "h235"}


def _meta235():
    from escalate import ENV_235B
    m = json.loads(json.dumps(H235))
    m["config"]["model"] = ENV_235B["NACA_VLM_MODEL"]
    return m


def _put(rdir, layer, n, text, meta):
    d = rdir / layer
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{n}.txt").write_text(text)
    (d / f"{n}.meta.json").write_text(json.dumps(meta))


def _full_doc(tmp_path):
    rdir = tmp_path / "111"
    (rdir / "ground_truth").mkdir(parents=True)
    (rdir / "ground_truth" / "1.txt").write_text("GT")
    (rdir / "ground_truth" / "1.json").write_text(json.dumps({"status": "done"}))
    _put(rdir, "vision_235b", 1, "T235", _meta235())
    _put(rdir, "vision_27b_v2", 1, "T27", {"config_hash": escalate.ESC_27B_HASH})
    _put(rdir, "vision_8blora_v2_1", 1, "T8B", {"config_hash": sorted(units.ACCEPTED_HASHES)[0]})
    _put(rdir, "vision_27bq4", 1, "LEGACY", {"config_hash": "x"})
    (rdir / "ocr").mkdir()
    (rdir / "ocr" / "1.json").write_text(json.dumps({"text": "TESS", "config_hash": "t"}))
    return rdir


def test_campaign_priority_order(tmp_path):
    rdir = _full_doc(tmp_path)
    rv = tmp_path / "none.jsonl"
    order = [("GT", "ground_truth", "ground_truth/1.json"), ("T235", "vision_235b", "vision_235b/1.txt"),
             ("T27", "vision_27b_v2", "vision_27b_v2/1.txt"),
             ("T8B", "vision_8blora_v2_1", "vision_8blora_v2_1/1.txt")]
    for text, src, rm in order:
        for co in (False, True):
            pt = best_text(rdir, 1, campaign_only=co, review_path=rv)
            assert (pt.text, pt.source) == (text, src)
        (rdir / rm).unlink()
        if src == "ground_truth":
            (rdir / "ground_truth" / "1.txt").unlink()
    assert best_text(rdir, 1, review_path=rv).text == "LEGACY"
    pt = best_text(rdir, 1, campaign_only=True, review_path=rv)
    assert (pt.text, pt.quality) == ("TESS", "ocr_draft")


def test_campaign_only_excludes_legacy(tmp_path):
    rdir = tmp_path / "222"
    _put(rdir, "vision_27bq4", 1, "LEGACY", {"config_hash": "x"})
    (rdir / "ocr").mkdir()
    (rdir / "ocr" / "1.json").write_text(json.dumps({"text": "TESS", "config_hash": "t"}))
    pt = best_text(rdir, 1, campaign_only=True)
    assert pt.text == "TESS" and pt.quality == "ocr_draft"
    assert best_text(rdir, 1).text == "LEGACY"


def test_hash_gate(tmp_path):
    rdir = tmp_path / "333"
    _put(rdir, "vision_8blora_v2_1", 1, "T8B", {"config_hash": "wrong"})
    _put(rdir, "vision_27b_v2", 1, "T27", {"config_hash": "wrong"})
    _put(rdir, "vision_235b", 1, "T235", {"config_hash": "x", "config": {}})
    (rdir / "ocr").mkdir()
    (rdir / "ocr" / "1.json").write_text(json.dumps({"text": "TESS"}))
    assert best_text(rdir, 1, campaign_only=True).source == "tesseract"
    _put(rdir, "vision_8blora_v2_1", 1, "T8B", {"config_hash": sorted(units.ACCEPTED_HASHES)[0],
                                               "loop_guard": {"vision_failed": True}})
    assert best_text(rdir, 1, campaign_only=True).source == "tesseract"


def test_flagged_page_marked(tmp_path):
    rdir = tmp_path / "444"
    _put(rdir, "vision_27b_v2", 1, "a", {"config_hash": escalate.ESC_27B_HASH})
    _put(rdir, "vision_27b_v2", 2, "b", {"config_hash": escalate.ESC_27B_HASH})
    _put(rdir, "vision_27b_v2", 3, "c", {"config_hash": escalate.ESC_27B_HASH,
                                         "table_verify": {"flags": [{"k": 1}]}})
    rv = tmp_path / "rv.jsonl"
    rv.write_text(json.dumps({"ntrs_id": "444", "page": 2}) + "\n")
    assert [best_text(rdir, n, review_path=rv).flagged for n in (1, 2, 3)] == [False, True, True]
    assert best_text(rdir, 1, review_path=tmp_path / "missing.jsonl").flagged is False


def test_no_pdf_doc_gets_index_row(tmp_path, monkeypatch):
    csvp = tmp_path / "r.csv"
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, ["ntrs_id", "report_number", "title", "year"])
        w.writeheader()
        w.writerow({"ntrs_id": "555", "report_number": "R5", "title": "No pdf", "year": "1950"})
        w.writerow({"ntrs_id": "666", "report_number": "R6", "title": "Has pages", "year": "1951"})
    monkeypatch.setenv("NACA_REPORTS_CSV", str(csvp))
    data = tmp_path / "data"
    (data / "555").mkdir(parents=True)
    (data / "555" / "meta.json").write_text(json.dumps({"pdf_status": "no_pdf_in_record", "csv_row": {}}))
    (data / "666" / "pages").mkdir(parents=True)
    (data / "666" / "pages" / "1.png").write_bytes(b"x")
    (data / "666" / "ocr").mkdir()
    (data / "666" / "ocr" / "1.json").write_text(json.dumps({"text": "hello"}))
    db = tmp_path / "c.db"
    build(db, data, ids=["555", "666"], embed=False, campaign_only=True, shard_meta={"batch": 3})
    conn = sqlite3.connect(db)
    rows = {r[0]: r for r in conn.execute("SELECT ntrs_id, disposition, pages, title, year FROM docs")}
    assert rows["555"][1] == "no_pdf" and rows["555"][2] == 0 and rows["555"][3] == "No pdf"
    assert rows["666"][1] == "ok" and rows["666"][2] == 1
    assert dict(conn.execute("SELECT key, value FROM index_meta"))["shard_meta"] == '{"batch": 3}'


def test_pending_queue_rows_flag_until_resolved(tmp_path):
    rdir = tmp_path / "777"
    h8 = {"config_hash": sorted(units.ACCEPTED_HASHES)[0]}
    _put(rdir, "vision_8blora_v2_1", 1, "t", h8)
    _put(rdir, "vision_8blora_v2_1", 2, "t", h8)
    q = tmp_path / "q.jsonl"
    q.write_text(json.dumps({"ntrs_id": "777", "page": 1, "kind": "table_flags"}) + "\n"
                 + json.dumps({"ntrs_id": "777", "page": 2, "kind": "vision_failed"}) + "\n")
    rv = tmp_path / "rv.jsonl"
    kw = dict(review_path=rv, queue_path=q)
    assert best_text(rdir, 1, **kw).flagged is True          # (a)
    _put(rdir, "vision_235b", 1, "t235", _meta235())          # (b)
    assert best_text(rdir, 1, **kw).flagged is False
    _put(rdir, "vision_27b_v2", 2, "t27", {"config_hash": escalate.ESC_27B_HASH})  # (c)
    assert best_text(rdir, 2, **kw).flagged is False
    rv.write_text(json.dumps({"ntrs_id": "777", "page": 1}) + "\n")  # (d)
    assert best_text(rdir, 1, **kw).flagged is True
