import sqlite3

import pytest

import chunk_refresh
from chunk_store import ChunkStore
from chunk_refresh import COLLECTION, doc_content_hash, pdf_page_order_ok, refresh_chunks


class FakeEmbed:
    def __init__(self, fail_after=None):
        self.calls, self.fail_after = 0, fail_after

    def count_tokens(self, text):
        return len(text.split())

    def embed(self, texts):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("embed down")
        self.last_texts = texts
        return [[0.0, 1.0] for _ in texts]


class FakeQdrant:
    def __init__(self):
        self.log = []

    def ensure_collection(self, dim=1024):
        self.log.append(("ensure",))

    def upsert(self, items):
        self.log.append(("upsert", sorted({p["doc_id"] for _, _, p in items}),
                         [cid for cid, _, _ in items], [p for _, _, p in items]))

    def delete_by_doc(self, doc_id):
        self.log.append(("delete", doc_id))

    def ops(self, kind):
        return [e for e in self.log if e[0] == kind]


def _corpus(path, docs):
    conn = sqlite3.connect(path)
    conn.executescript("""
    CREATE TABLE reports(doc_id TEXT PRIMARY KEY, ntrs_id TEXT, report_number TEXT,
      title TEXT, series TEXT, decade TEXT, year TEXT, pages INTEGER, source_url TEXT);
    CREATE TABLE pages(doc_id TEXT, page INTEGER, text TEXT, source TEXT, quality TEXT,
      config_hash TEXT, chars INTEGER, src_mtime REAL, PRIMARY KEY(doc_id, page));""")
    for d, texts in docs.items():
        conn.execute("INSERT INTO reports VALUES (?,?,?,?,?,?,?,?,?)",
                     (d, d, f"R-{d}", f"Title {d}", "TN", "1940s", "1945",
                      len(texts), f"http://x/{d}.pdf"))
        for i, t in enumerate(texts, 1):
            conn.execute("INSERT INTO pages VALUES (?,?,?,?,?,?,?,?)",
                         (d, i, t, "best", "ok", "h1", len(t), 1.0))
    conn.commit()
    conn.close()


@pytest.fixture
def env(tmp_path):
    corpus = tmp_path / "corpus.db"
    _corpus(corpus, {"A": ["Alpha text one.", "More alpha."],
                     "B": ["Beta text."]})
    data = tmp_path / "data"
    data.mkdir()
    return corpus, tmp_path / "chunks.db", data


def run(env, embed=None, qdrant=None):
    corpus, chunks, data = env
    return refresh_chunks(corpus, chunks, data, embed or FakeEmbed(),
                          qdrant or FakeQdrant())


def test_first_run_chunks_all_docs(env):
    q = FakeQdrant()
    counts = run(env, qdrant=q)
    assert set(counts) == {"chunked_docs", "deleted_docs", "chunks", "skipped_docs"}
    assert counts["chunked_docs"] == 2 and counts["skipped_docs"] == 0
    assert counts["deleted_docs"] == 0 and counts["chunks"] >= 2
    store = ChunkStore(env[1], readonly=True)
    assert set(store.doc_hashes()) == {"A", "B"}
    assert store.doc_meta("A")["collection"] == COLLECTION
    assert store.doc_meta("A")["pdf_url"] == "http://x/A.pdf"
    ups = q.ops("upsert")
    assert sum(len(u[2]) for u in ups) == counts["chunks"]
    for u in ups:
        for p in u[3]:
            assert set(p) == {"chunk_id", "doc_id", "pages", "section", "collection"}


def test_embedding_prefix(env):
    e = FakeEmbed()
    run(env, embed=e)
    assert e.last_texts[0].startswith("Title B — ")
    assert ": Beta text." in e.last_texts[0]


def test_unchanged_doc_skipped(env):
    run(env)
    q = FakeQdrant()
    counts = run(env, qdrant=q)
    assert counts["skipped_docs"] == 2 and counts["chunked_docs"] == 0
    assert q.ops("upsert") == [] and q.ops("delete") == []


def test_changed_doc_rechunked_delete_before_upsert(env):
    run(env)
    conn = sqlite3.connect(env[0])
    conn.execute("UPDATE pages SET text='Changed words here.', src_mtime=2.0 "
                 "WHERE doc_id='A' AND page=1")
    conn.commit()
    conn.close()
    before = ChunkStore(env[1], readonly=True).get_chunks(["A:0000"])[0]["text"]
    q = FakeQdrant()
    counts = run(env, qdrant=q)
    assert counts["chunked_docs"] == 1 and counts["skipped_docs"] == 1
    kinds = [e[0] for e in q.log if e[0] in ("delete", "upsert")]
    assert kinds == ["delete", "upsert"]
    after = ChunkStore(env[1], readonly=True).get_chunks(["A:0000"])[0]["text"]
    assert "Changed" in after and after != before


def test_removed_doc_deleted_everywhere(env):
    run(env)
    conn = sqlite3.connect(env[0])
    conn.execute("DELETE FROM pages WHERE doc_id='B'")
    conn.execute("DELETE FROM reports WHERE doc_id='B'")
    conn.commit()
    conn.close()
    q = FakeQdrant()
    counts = run(env, qdrant=q)
    assert counts["deleted_docs"] == 1
    assert ("delete", "B") in q.log
    store = ChunkStore(env[1], readonly=True)
    assert set(store.doc_hashes()) == {"A"}
    assert store.get_chunks(["B:0000"]) == []


def test_pdf_mismatch_clears_anchor_flag(env):
    run(env)  # no source.pdf under data/A
    store = ChunkStore(env[1], readonly=True)
    assert store.doc_meta("A")["page_order_ok"] == 0


def test_pdf_page_order_ok_branches(tmp_path, monkeypatch):
    assert pdf_page_order_ok(tmp_path, 2) is False  # missing pdf
    (tmp_path / "source.pdf").write_bytes(b"x")

    class R:
        def __init__(self, out, rc=0):
            self.stdout, self.returncode = out, rc

    monkeypatch.setattr(chunk_refresh.subprocess, "run",
                        lambda *a, **k: R("Title: x\nPages:          2\n"))
    assert pdf_page_order_ok(tmp_path, 2) is True
    assert pdf_page_order_ok(tmp_path, 3) is False
    monkeypatch.setattr(chunk_refresh.subprocess, "run", lambda *a, **k: R("junk"))
    assert pdf_page_order_ok(tmp_path, 2) is False
    monkeypatch.setattr(chunk_refresh.subprocess, "run", lambda *a, **k: R("", 1))
    assert pdf_page_order_ok(tmp_path, 2) is False


def test_doc_content_hash_order_independent_and_sensitive():
    a = [(1, "best", "h", 1.0), (2, "best", "h", 1.0)]
    assert doc_content_hash(a) == doc_content_hash(list(reversed(a)))
    assert doc_content_hash(a) != doc_content_hash(
        [(1, "best", "h", 1.0), (2, "best", "h", 2.0)])


def test_staged_swap_leaves_live_db_untouched_on_failure(env):
    run(env)
    live = env[1]
    before = live.read_bytes()
    conn = sqlite3.connect(env[0])
    conn.execute("UPDATE pages SET text='New.', src_mtime=9.0")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError):
        run(env, embed=FakeEmbed(fail_after=1))
    assert live.read_bytes() == before
    assert not live.with_suffix(".staging").exists()


def test_roster_takes_precedence_over_corpus_reports(env, tmp_path):
    roster = tmp_path / "roster.csv"
    roster.write_text(
        "ntrs_id,report_number,title,series,published_year,pdf_url,no_pdf,"
        "in_pilot,pub_date,decade\n"
        "A,NACA-R-A,Roster Title A,TN,1946,http://roster/A.pdf,False,True,1946-01-01,1940s\n")
    corpus, chunks, data = env
    refresh_chunks(corpus, chunks, data, FakeEmbed(), FakeQdrant(), roster)
    store = ChunkStore(chunks, readonly=True)
    a = store.doc_meta("A")
    assert a["title"] == "Roster Title A"
    assert a["pdf_url"] == "http://roster/A.pdf"
    assert a["report_number"] == "NACA-R-A" and a["year"] == "1946"
    assert a["ntrs_id"] == "A"
    b = store.doc_meta("B")  # no roster row: corpus.db fallback
    assert b["title"] == "Title B" and b["pdf_url"] == "http://x/B.pdf"
    assert b["year"] == "1945"


def test_embed_input_prefix_omits_empty_section():
    from chunker import Chunk
    c = Chunk([1], "", "body", 0, 4)
    assert chunk_refresh._embed_input("T", c) == "T: body"
    assert chunk_refresh._embed_input("T", Chunk([1], "S", "body", 0, 4)) == "T — S: body"


def test_short_embed_response_is_an_error(env):
    class Short(FakeEmbed):
        def embed(self, texts):
            return []
    with pytest.raises(RuntimeError, match="vectors"):
        run(env, embed=Short())
    assert not env[1].exists()


def test_qdrant_name_mapping():
    from qdrant_client import Qdrant
    assert Qdrant.for_collection("http://q", "naca").collection == "chunks"
    assert (Qdrant.for_collection("http://q", "nara-rg255").collection
            == "chunks_nara-rg255")


def test_refresh_writes_collection_rows_and_target(tmp_path):
    corpus = tmp_path / "corpus.db"
    _corpus(corpus, {"nara-rg255-a": ["alpha beta gamma"]})
    chunks = tmp_path / "chunks.db"
    qd = FakeQdrant()
    refresh_chunks(corpus, chunks, tmp_path, FakeEmbed(), qd,
                   collection="nara-rg255")
    st = ChunkStore(chunks, readonly=True)
    assert st.doc_meta("nara-rg255-a")["collection"] == "nara-rg255"
    assert all(p["collection"] == "nara-rg255" for p in qd.ops("upsert")[0][3])


def test_stale_sweep_scoped_to_collection(env):
    corpus, chunks, data = env
    refresh_chunks(corpus, chunks, data, FakeEmbed(), FakeQdrant(), collection="other")
    mine1 = corpus.parent / "mine1.db"
    _corpus(mine1, {"A2": ["Alpha two."], "STALE": ["Gone soon."]})
    refresh_chunks(mine1, chunks, data, FakeEmbed(), FakeQdrant(), collection="mine")
    mine2 = corpus.parent / "mine2.db"
    _corpus(mine2, {"A2": ["Alpha two."]})
    q = FakeQdrant()
    counts = refresh_chunks(mine2, chunks, data, FakeEmbed(), q, collection="mine")
    store = ChunkStore(chunks, readonly=True)
    assert set(store.doc_hashes()) == {"A", "B", "A2"}
    assert q.ops("delete") == [("delete", "STALE")]
    assert counts["deleted_docs"] == 1
