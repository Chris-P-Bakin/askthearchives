import os
import sqlite3
import sys

import pytest

from index_refresh import refresh


def _mkdb(path, n=1, mtime=1.0):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS pages(doc_id, page, src_mtime REAL)")
    conn.execute("CREATE TABLE IF NOT EXISTS marker(v)")
    conn.execute("DELETE FROM pages")
    for i in range(n):
        conn.execute("INSERT INTO pages VALUES ('d', ?, ?)", (i, mtime))
    conn.commit()
    conn.close()


@pytest.fixture
def tmp_tree(tmp_path):
    db = tmp_path / "corpus.db"
    _mkdb(db)
    return db, tmp_path


def test_refresh_swaps_only_on_success(tmp_tree):
    db, data = tmp_tree
    before = db.read_bytes()

    def boom(staging):
        _mkdb(staging, n=5)
        raise RuntimeError("build failed")

    with pytest.raises(RuntimeError):
        refresh(db, data, build_fn=boom)
    assert db.read_bytes() == before


def test_refresh_is_atomic_rename_not_inplace(tmp_tree):
    db, data = tmp_tree
    ino = os.stat(db).st_ino
    seen = {}

    def build(staging):
        assert staging != db
        seen["live_during"] = sqlite3.connect(db).execute(
            "SELECT count(*) FROM pages").fetchone()[0]
        conn = sqlite3.connect(staging)
        conn.execute("INSERT INTO marker VALUES ('new')")
        conn.commit()
        conn.close()

    refresh(db, data, build_fn=build)
    assert seen["live_during"] == 1
    assert os.stat(db).st_ino != ino
    assert sqlite3.connect(db).execute("SELECT v FROM marker").fetchone() == ("new",)


def test_stale_staging_from_crash_is_overwritten(tmp_tree):
    db, data = tmp_tree
    staging = db.with_suffix(".staging")
    staging.write_bytes(b"garbage")
    for sfx in ("-wal", "-shm"):
        (db.parent / (staging.name + sfx)).write_bytes(b"junk")

    def build(p):
        _mkdb(p, n=3)

    assert refresh(db, data, build_fn=build) is True
    assert sqlite3.connect(db).execute("SELECT count(*) FROM pages").fetchone()[0] == 3
    assert not (db.parent / (staging.name + "-wal")).exists()


def test_restart_cmd_runs_only_when_changed(tmp_tree, tmp_path):
    db, data = tmp_tree
    log = tmp_path / "ran.txt"
    cmd = [sys.executable, "-c", f"open({str(log)!r}, 'a').write('x')"]

    assert refresh(db, data, build_fn=lambda p: None, restart_cmd=cmd) is False
    assert not log.exists()
    assert refresh(db, data, build_fn=lambda p: _mkdb(p, n=2), restart_cmd=cmd) is True
    assert log.read_text() == "x"
    assert refresh(db, data, build_fn=lambda p: _mkdb(p, n=2, mtime=9.0),
                   restart_cmd=cmd) is True
    assert log.read_text() == "xx"


def test_absent_live_db_builds_fresh(tmp_path):
    db = tmp_path / "new.db"
    assert refresh(db, tmp_path, build_fn=lambda p: _mkdb(p, n=1)) is True
    assert db.exists()


@pytest.mark.parametrize("payload", [b"", b"not a sqlite file" * 100, None])
def test_insane_staging_is_not_swapped(tmp_tree, payload):
    db, data = tmp_tree
    before = db.read_bytes()

    def build(staging):
        if payload is None:
            _mkdb(staging, n=0)  # valid schema, zero pages
        else:
            staging.write_bytes(payload)

    with pytest.raises(RuntimeError, match="staging"):
        refresh(db, data, build_fn=build)
    assert db.read_bytes() == before
    assert not db.with_suffix(".staging").exists()


def test_main_chunk_failure_exits_nonzero_after_swap(tmp_tree, monkeypatch):
    import index_refresh
    import chunk_refresh
    db, data = tmp_tree
    seen = {}
    monkeypatch.setattr(index_refresh, "refresh",
                        lambda *a, **k: seen.setdefault("embed", k["embed"]) or True)

    def boom(*a, **k):
        seen["roster"] = a[5] if len(a) > 5 else k.get("roster_csv")
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(chunk_refresh, "refresh_chunks", boom)
    monkeypatch.setattr(sys, "argv", ["x", "--db", str(db), "--data", str(data),
                                      "--chunks-db", str(data / "c.db"), "--no-embed"])
    with pytest.raises(SystemExit) as e:
        index_refresh.main()
    assert e.value.code == 1 and seen["embed"] is False
    assert seen["roster"] == data / "roster.csv"


def test_main_no_chunks_skips_chunk_step(tmp_tree, monkeypatch):
    import index_refresh
    import chunk_refresh
    db, data = tmp_tree
    monkeypatch.setattr(index_refresh, "refresh", lambda *a, **k: False)
    monkeypatch.setattr(chunk_refresh, "refresh_chunks",
                        lambda *a, **k: pytest.fail("should not run"))
    monkeypatch.setattr(sys, "argv", ["x", "--db", str(db), "--data", str(data),
                                      "--no-chunks"])
    index_refresh.main()
