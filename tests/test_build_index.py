import json
import sqlite3
from conftest import DATA

from build_index import best_text, build


def test_best_text_prefers_ground_truth():
    pt = best_text(DATA / "19930091059", 23)
    assert pt.quality == "verified"
    assert pt.source == "ground_truth"
    assert len(pt.text) > 100


def test_best_text_vision_failed_falls_back(tmp_path):
    rdir = tmp_path / "99999"
    (rdir / "vision2").mkdir(parents=True)
    (rdir / "ocr").mkdir()
    (rdir / "vision2" / "5.txt").write_text("looped garbage\n")
    (rdir / "vision2" / "5.meta.json").write_text(json.dumps(
        {"config_hash": "abc", "loop_guard": {"vision_failed": True}}))
    (rdir / "ocr" / "5.json").write_text(json.dumps(
        {"text": "tesseract text", "config_hash": "def", "words": []}))
    pt = best_text(rdir, 5)
    assert pt.quality == "ocr_draft"
    assert pt.text == "tesseract text"


def test_build_subset_fts(tmp_path):
    db = tmp_path / "corpus.db"
    counts = build(db, DATA, ids=["19930091059"], embed=False)
    assert counts["pages"] > 0
    # a distinctive long token from the verified p9 ground truth must be findable
    gt = (DATA / "19930091059" / "ground_truth" / "23.txt").read_text()
    token = max((w.strip('.,;:()"') for w in gt.split()), key=len)
    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT p.doc_id, p.page FROM pages_fts f JOIN pages p ON p.rowid = f.rowid "
        "WHERE pages_fts MATCH ?", (f'"{token}"',)).fetchall()
    assert ("19930091059", 23) in rows
    meta = dict(conn.execute("SELECT key, value FROM index_meta"))
    assert meta["schema_version"] == "1"


def test_build_embeds_and_is_incremental(tmp_path):
    import sqlite_vec
    db = tmp_path / "corpus.db"
    c1 = build(db, DATA, ids=["19930091059"])
    assert c1["indexed"] == c1["pages"] > 0
    conn = sqlite3.connect(db)
    conn.enable_load_extension(True); sqlite_vec.load(conn)
    nvec = conn.execute("SELECT count(*) FROM pages_vec").fetchone()[0]
    npages = conn.execute("SELECT count(*) FROM pages").fetchone()[0]
    assert nvec == npages
    meta = dict(conn.execute("SELECT key, value FROM index_meta"))
    assert meta["embed_model"] == "BAAI/bge-small-en-v1.5" and meta["embed_dim"] == "384"
    conn.close()
    c2 = build(db, DATA, ids=["19930091059"])
    assert c2["indexed"] == 0 and c2["skipped"] == c2["pages"]


def test_semantic_neighbors_sane(tmp_path):
    import sqlite_vec, struct
    from build_index import embed_texts
    db = tmp_path / "corpus.db"
    build(db, DATA, ids=["19930091059"])
    q = embed_texts(["supersonic flow tables and pressure ratios"])[0]
    conn = sqlite3.connect(db)
    conn.enable_load_extension(True); sqlite_vec.load(conn)
    rows = conn.execute(
        "SELECT p.doc_id, p.page FROM pages_vec v JOIN pages p ON p.rowid = v.rowid "
        "WHERE v.embedding MATCH ? AND k = 5 ORDER BY distance",
        (struct.pack("384f", *q),)).fetchall()
    assert len(rows) == 5 and all(r[0] == "19930091059" for r in rows)


def test_built_db_is_readonly_safe(tmp_path):
    """A read-only consumer (ro filesystem mount) must be able to open the db:
    requires journal_mode=delete and no -wal sidecar after build."""
    db = tmp_path / "corpus.db"
    build(db, DATA, ids=["19930093282"], embed=False)
    assert not (tmp_path / "corpus.db-wal").exists()
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    assert dict(conn.execute("SELECT key,value FROM index_meta"))["schema_version"] == "1"


def test_rebuild_no_embed_after_embed_does_not_crash(tmp_path):
    db = tmp_path / "corpus.db"
    build(db, DATA, ids=["19930093282"])
    counts = build(db, DATA, ids=["19930093282"], rebuild=True, embed=False)
    assert counts["indexed"] == counts["pages"]


def test_vec_backfill_after_no_embed_run(tmp_path):
    import sqlite_vec
    db = tmp_path / "corpus.db"
    build(db, DATA, ids=["19930093282"], embed=False)
    build(db, DATA, ids=["19930093282"], embed=True)  # pages unchanged, vectors missing
    conn = sqlite3.connect(db)
    conn.enable_load_extension(True); sqlite_vec.load(conn)
    nvec = conn.execute("SELECT count(*) FROM pages_vec").fetchone()[0]
    npages = conn.execute("SELECT count(*) FROM pages").fetchone()[0]
    assert nvec == npages > 0


def test_no_embed_incremental_removes_stale_vector(tmp_path):
    import os, sqlite_vec
    db = tmp_path / "corpus.db"
    build(db, DATA, ids=["19930093282"])
    conn = sqlite3.connect(db)
    src = conn.execute("SELECT doc_id, page, source FROM pages LIMIT 1").fetchone()
    conn.close()
    # touch the chosen page's source so it reindexes
    doc, page, source = src
    path = {"tesseract": DATA / doc / "ocr" / f"{page}.json",
            "vision_27bq4": DATA / doc / "vision_27bq4" / f"{page}.txt",
            "vision2": DATA / doc / "vision2" / f"{page}.txt",
            "vision_haiku2": DATA / doc / "vision_haiku2" / f"{page}.txt",
            "ground_truth": DATA / doc / "ground_truth" / f"{page}.txt"}[source]
    os.utime(path)
    build(db, DATA, ids=["19930093282"], embed=False)
    conn = sqlite3.connect(db)
    conn.enable_load_extension(True); sqlite_vec.load(conn)
    orphans = conn.execute("SELECT count(*) FROM pages_vec WHERE rowid NOT IN "
                           "(SELECT rowid FROM pages)").fetchone()[0]
    assert orphans == 0


def test_best_text_haiku_tier_between_vision2_and_ocr(tmp_path):
    rdir = tmp_path / "99999"
    (rdir / "vision_haiku2").mkdir(parents=True)
    (rdir / "ocr").mkdir()
    (rdir / "vision_haiku2" / "5.txt").write_text("haiku vision text\n")
    (rdir / "vision_haiku2" / "5.meta.json").write_text(json.dumps(
        {"config_hash": "hk1", "loop_guard": {"vision_failed": False}}))
    (rdir / "ocr" / "5.json").write_text(json.dumps(
        {"text": "tesseract text", "config_hash": "def", "words": []}))
    # no vision2 -> haiku beats ocr
    pt = best_text(rdir, 5)
    assert pt.source == "vision_haiku2" and pt.quality == "vision"
    assert pt.text == "haiku vision text\n"
    # vision2 present and healthy -> 27B still wins
    (rdir / "vision2").mkdir()
    (rdir / "vision2" / "5.txt").write_text("qwen vision text\n")
    (rdir / "vision2" / "5.meta.json").write_text(json.dumps(
        {"config_hash": "qw1", "loop_guard": {"vision_failed": False}}))
    pt = best_text(rdir, 5)
    assert pt.source == "vision2"
    # haiku vision_failed -> fall through to ocr
    import shutil; shutil.rmtree(rdir / "vision2")
    (rdir / "vision_haiku2" / "5.meta.json").write_text(json.dumps(
        {"config_hash": "hk1", "loop_guard": {"vision_failed": True}}))
    pt = best_text(rdir, 5)
    assert pt.quality == "ocr_draft"


def test_best_text_27bq4_tier_tops_vision(tmp_path):
    rdir = tmp_path / "99999"
    for vdir, txt in (("vision_27bq4", "q4 local text\n"),
                      ("vision2", "nvfp4 text\n"),
                      ("vision_haiku2", "haiku text\n")):
        (rdir / vdir).mkdir(parents=True)
        (rdir / vdir / "5.txt").write_text(txt)
        (rdir / vdir / "5.meta.json").write_text(json.dumps(
            {"config_hash": "x", "loop_guard": {"vision_failed": False}}))
    pt = best_text(rdir, 5)
    assert pt.source == "vision_27bq4" and pt.quality == "vision"
    assert pt.text == "q4 local text\n"
    # 27bq4 vision_failed -> next vision tier wins
    (rdir / "vision_27bq4" / "5.meta.json").write_text(json.dumps(
        {"config_hash": "x", "loop_guard": {"vision_failed": True}}))
    assert best_text(rdir, 5).source == "vision2"


def test_best_text_reads_reconcile_record(tmp_path):
    rdir = tmp_path / "88888"
    for vdir, txt in (("vision_27bq4", "q4 text\n"), ("vision_haiku2", "haiku text\n")):
        (rdir / vdir).mkdir(parents=True)
        (rdir / vdir / "5.txt").write_text(txt)
        (rdir / vdir / "5.meta.json").write_text(json.dumps(
            {"config_hash": "x", "loop_guard": {"vision_failed": False}}))
    (rdir / "reconcile").mkdir()

    def rec(conf, source="vision_27bq4"):
        (rdir / "reconcile" / "5.json").write_text(json.dumps(
            {"page_source": source, "confidence": conf,
             "inter_cer": 0.01, "decided_by": "agreement_v1"}))

    rec("agreed")
    pt = best_text(rdir, 5)
    assert pt.quality == "vision_agreed" and pt.source == "vision_27bq4"
    assert pt.text == "q4 text\n"
    rec("conflict")
    assert best_text(rdir, 5).quality == "vision_conflict"
    rec("single", source="vision_haiku2")
    pt = best_text(rdir, 5)
    assert pt.quality == "vision_single" and pt.text == "haiku text\n"
    # ground truth still outranks any reconcile record
    (rdir / "ground_truth").mkdir()
    (rdir / "ground_truth" / "5.txt").write_text("human text\n")
    (rdir / "ground_truth" / "5.json").write_text(json.dumps({"status": "done"}))
    assert best_text(rdir, 5).quality == "verified"
