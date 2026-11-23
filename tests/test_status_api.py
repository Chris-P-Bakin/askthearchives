import json

import status


def test_training_state_parses_and_downsamples(tmp_path):
    log = tmp_path / "v1_loss.jsonl"
    rows = [{"step": i, "loss": 1.0 / (i + 1), "epoch": i / 100} for i in range(1, 501)]
    rows[299]["eval_loss"] = 0.2
    log.write_text("\n".join(json.dumps(r) for r in rows))
    t = status.training_state(log, alive=True, total_steps=600)
    assert t["alive"] is True
    assert t["step"] == 500 and t["total_steps"] == 600
    assert t["pct"] == 83
    assert t["eval_loss"] == 0.2
    assert t["loss"] == rows[-1]["loss"]
    assert len(t["curve"]) <= status.CURVE_POINTS
    # curve preserves order and endpoints
    steps = [p[0] for p in t["curve"]]
    assert steps == sorted(steps) and steps[-1] == 500


def test_training_state_missing_log(tmp_path):
    t = status.training_state(tmp_path / "none.jsonl", alive=False, total_steps=0)
    assert t["alive"] is False and t["step"] == 0 and t["curve"] == []


def test_tier_counts_from_db(tmp_path):
    import sqlite3
    db = tmp_path / "corpus.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE pages(doc_id TEXT, quality TEXT)")
    conn.executemany("INSERT INTO pages VALUES(?,?)", [
        ("a", "verified"), ("a", "vision_agreed"), ("b", "ocr_draft"),
        ("b", "vision_conflict"), ("b", "vision_conflict")])
    conn.commit(); conn.close()
    tc = status.tier_counts(db)
    assert tc == {"verified": 1, "vision_agreed": 1, "vision_conflict": 2, "ocr_draft": 1}


def test_tier_counts_missing_db(tmp_path):
    assert status.tier_counts(tmp_path / "none.db") == {}


def test_collection_meta_reads_file(tmp_path):
    f = tmp_path / "collection.json"
    f.write_text(json.dumps({"name": "Alchemy Scrolls", "subtitle": "12 scrolls",
                             "short": "Scrolls"}))
    m = status.collection_meta(f)
    assert m["name"] == "Alchemy Scrolls" and m["short"] == "Scrolls"


def test_collection_meta_defaults_when_missing(tmp_path):
    m = status.collection_meta(tmp_path / "none.json")
    assert m["name"] == "Document Pipeline"
    assert "short" in m and "subtitle" in m
