import sqlite3

import pytest

from build_index import DDL
from chunk_store import ChunkStore
from chunker import Chunk
from embed_client import EmbedUnavailable
from qdrant_client import QdrantUnavailable
from mcp_server import db as dbmod
from mcp_server.retrieval import ChunkSearch


def _chunk(text, pages, section="S"):
    return Chunk(pages=list(pages), section=section, text=text,
                 char_start=0, char_end=len(text))


def _meta(**kw):
    m = dict(collection="naca", title="A Report", ntrs_id="19700001",
             pdf_url="https://x/y.pdf", report_number="TN D-1", year="1970",
             page_order_ok=1)
    m.update(kw)
    return m


@pytest.fixture
def corpus(tmp_path):
    p = tmp_path / "corpus.db"
    conn = sqlite3.connect(p)
    conn.executescript(DDL)
    conn.execute("INSERT INTO reports VALUES (?,?,?,?,?,?,?,?,?)",
                 ("d1", "19700001", "NACA-TN-1", "T", "TN", "1970s", "1970", 9, ""))
    for pg, (src, q) in {4: ("t4", "verified"), 5: ("t5", "ocr_draft")}.items():
        conn.execute("INSERT INTO pages VALUES (?,?,?,?,?,?,?,?)",
                     ("d1", pg, "x", src, q, "h", 1, 0.0))
    conn.execute("INSERT INTO index_meta VALUES ('embed_model','x')")
    conn.commit()
    conn.close()
    return dbmod.Corpus(p, tmp_path)


@pytest.fixture
def chunks_db(tmp_path):
    p = tmp_path / "chunks.db"
    st = ChunkStore(p)
    st.replace_doc("d1", "h", _meta(), [
        _chunk("wing flutter alpha", [4, 5], "Intro"),    # d1:0000
        _chunk("flutter only here", [5]),                 # d1:0001
        _chunk("unrelated semantic text", [6]),           # d1:0002
    ])
    st.replace_doc("d2", "h", _meta(page_order_ok=0), [_chunk("flutter nopdf", [1])])
    return p


class FakeEmbed:
    def __init__(self, exc=None):
        self.exc, self.calls = exc, 0

    def embed_query(self, q):
        self.calls += 1
        if self.exc:
            raise self.exc
        return [0.0]


class FakeQdrant:
    def __init__(self, hits=(), exc=None):
        self.hits, self.exc = list(hits), exc

    def search(self, vector, limit):
        if self.exc:
            raise self.exc
        return self.hits


def make(chunks_db, corpus, hits=(), embed=None, qd=None, rerank_url=None):
    return ChunkSearch(chunks_db, embed or FakeEmbed(),
                       qd or FakeQdrant(hits), rerank_url, corpus=corpus)


def test_hybrid_fuses_both_legs_rrf(chunks_db, corpus):
    cs = make(chunks_db, corpus, [("d1:0000", .9), ("d1:0002", .8)])
    out = cs.search("flutter", "hybrid", 10)
    assert out["degraded"] is False
    by = {r["chunk_id"]: r for r in out["results"]}
    r = dbmod.RRF_K
    both = by["d1:0000"]["score"]
    assert out["results"][0]["chunk_id"] == "d1:0000"
    assert both > by["d1:0001"]["score"] and both > by["d1:0002"]["score"]
    assert by["d1:0002"]["score"] == pytest.approx(round(1 / (r + 2), 5))
    kw = ChunkStore(chunks_db, readonly=True).fts_search(
        dbmod._fts_query("flutter"), 50)
    expect = 1 / (r + 1) + 1 / (r + kw.index("d1:0000") + 1)
    assert both == pytest.approx(round(expect, 5))


def test_result_entry_contract(chunks_db, corpus):
    out = make(chunks_db, corpus).search("alpha", "keyword", 5)
    e = out["results"][0]
    assert set(e) == {"collection", "doc_id", "page", "pages", "section",
                      "chunk_id", "snippet", "score", "quality", "source",
                      "original"}
    assert e["collection"] == "naca" and e["doc_id"] == "d1"
    assert e["page"] == e["pages"][0] == 4 and e["pages"] == [4, 5]
    assert e["section"] == "Intro" and e["chunk_id"] == "d1:0000"
    assert e["quality"] == "verified" and e["source"] == "t4"
    assert "[alpha]" in e["snippet"]
    assert e["original"] == {
        "record_url": "https://ntrs.nasa.gov/citations/19700001",
        "pdf_url": "https://x/y.pdf#page=4"}


def test_pdf_anchor_suppressed_when_flagged(chunks_db, corpus):
    out = make(chunks_db, corpus).search("nopdf", "keyword", 5)
    assert out["results"][0]["doc_id"] == "d2"
    assert out["results"][0]["original"]["pdf_url"] is None


def test_embed_down_degrades_to_fts(chunks_db, corpus):
    cs = make(chunks_db, corpus, embed=FakeEmbed(EmbedUnavailable("down")))
    out = cs.search("flutter", "hybrid", 10)
    assert out["degraded"] is True and len(out["results"]) == 3
    out = cs.search("flutter", "semantic", 10)
    assert out["degraded"] is True and len(out["results"]) == 3


def test_qdrant_down_and_malformed_degrade(chunks_db, corpus):
    for exc in (QdrantUnavailable("x"), KeyError("result"), TypeError("t"),
                ValueError("v")):
        cs = make(chunks_db, corpus, qd=FakeQdrant(exc=exc))
        out = cs.search("flutter", "hybrid", 10)
        assert out["degraded"] is True and out["results"]


def test_qdrant_chunk_missing_from_store_dropped(chunks_db, corpus):
    cs = make(chunks_db, corpus, [("ghost:0000", .9), ("d1:0002", .5)])
    ids = [r["chunk_id"] for r in cs.search("flutter", "hybrid", 10)["results"]]
    assert "ghost:0000" not in ids and "d1:0002" in ids


def test_keyword_mode_never_touches_embed(chunks_db, corpus):
    emb = FakeEmbed(AssertionError("must not be called"))
    out = make(chunks_db, corpus, embed=emb).search("flutter", "keyword", 5)
    assert emb.calls == 0 and out["degraded"] is False and out["results"]


def test_rerank_flag_reorders(chunks_db, corpus, monkeypatch):
    cs = make(chunks_db, corpus, rerank_url="http://r")
    base = [r["chunk_id"] for r in
            make(chunks_db, corpus).search("flutter", "keyword", 10)["results"]]
    posted = {}

    def fake_post(url, payload):
        posted["url"], posted["payload"] = url, payload
        n = len(payload["documents"])
        return {"results": [{"index": i, "relevance_score": float(i)}
                            for i in range(n)]}
    monkeypatch.setattr(cs, "_post_json", fake_post)
    got = [r["chunk_id"] for r in cs.search("flutter", "keyword", 10)["results"]]
    assert got == base[::-1]
    assert posted["url"] == "http://r/v1/rerank"
    assert posted["payload"]["model"] == "q"
    assert posted["payload"]["query"] == "flutter"


def test_rerank_failure_nonfatal(chunks_db, corpus, monkeypatch):
    cs = make(chunks_db, corpus, rerank_url="http://r")
    base = [r["chunk_id"] for r in
            make(chunks_db, corpus).search("flutter", "keyword", 10)["results"]]

    def boom(url, payload):
        raise OSError("down")
    monkeypatch.setattr(cs, "_post_json", boom)
    out = cs.search("flutter", "keyword", 10)
    assert [r["chunk_id"] for r in out["results"]] == base
    assert out["degraded"] is False


def test_fts_syntax_error_is_unparseable(chunks_db, corpus, monkeypatch):
    from mcp_server import retrieval
    monkeypatch.setattr(retrieval, "_fts_query", lambda q: q)
    with pytest.raises(dbmod.NotFoundError, match="unparseable query"):
        make(chunks_db, corpus).search('"', "keyword", 5)


def test_cite_page_span(corpus):
    base = corpus.cite("d1", 4)
    assert base["text"].endswith("p. 4. NTRS 19700001.")
    assert "page_end" not in base
    assert corpus.cite("d1", 4, None) == base
    assert corpus.cite("d1", 4, 4) == base
    sp = corpus.cite("d1", 4, 7)
    assert "pp. 4–7. NTRS" in sp["text"] and sp["page_end"] == 7


# ------------------------------------------------------------ scoped search
@pytest.fixture
def two_coll_db(tmp_path):
    p = tmp_path / "chunks2.db"
    st = ChunkStore(p)
    st.replace_doc("d1", "h", _meta(), [_chunk("flutter naca", [4])])
    st.replace_doc("nara-rg255-x", "h",
                   _meta(collection="nara-rg255", ntrs_id="x1"),
                   [_chunk("flutter nara", [2])])
    return p


def _scoped(db, resolver=None):
    qds = {"naca": FakeQdrant([("d1:0000", .9)]),
           "nara-rg255": FakeQdrant([("nara-rg255-x:0000", .9)])}
    return ChunkSearch(db, FakeEmbed(), qds["naca"], None,
                       qdrant_for=lambda cid: qds[cid],
                       corpus_resolver=resolver)


def test_scope_filters_keyword_and_semantic(two_coll_db):
    cs = _scoped(two_coll_db)
    for mode in ("keyword", "semantic", "hybrid"):
        out = cs.search("flutter", mode, 10, archives=["naca"])
        assert {r["collection"] for r in out["results"]} == {"naca"}
        out = cs.search("flutter", mode, 10, archives=["naca", "nara-rg255"])
        assert {r["collection"] for r in out["results"]} == {"naca", "nara-rg255"}


def test_scope_quality_from_hit_collection_and_missing_degrades(
        two_coll_db, corpus):
    seen = []

    def resolver(doc_id):
        seen.append(doc_id)
        if doc_id == "d1":
            return corpus
        raise RuntimeError("collection not yet serving")
    out = _scoped(two_coll_db, resolver).search(
        "flutter", "keyword", 10, archives=["naca", "nara-rg255"])
    by = {r["collection"]: r for r in out["results"]}
    assert by["naca"]["quality"] == "verified"
    assert by["nara-rg255"]["quality"] is None and by["nara-rg255"]["source"] is None
    assert set(seen) == {"d1", "nara-rg255-x"}


def test_partial_qdrant_failure_keeps_healthy_collection(two_coll_db):
    qds = {"naca": FakeQdrant(exc=QdrantUnavailable("no chunks")),
           "nara-rg255": FakeQdrant([("nara-rg255-x:0000", .9)])}
    cs = ChunkSearch(two_coll_db, FakeEmbed(), qds["naca"],
                     qdrant_for=lambda cid: qds[cid])
    both = ["naca", "nara-rg255"]
    out = cs.search("zzzunmatched", "semantic", 10, archives=both)
    assert out["degraded"] is True
    assert [r["chunk_id"] for r in out["results"]] == ["nara-rg255-x:0000"]
    out = cs.search("flutter", "hybrid", 10, archives=both)
    assert out["degraded"] is True and len(out["results"]) == 2
    qds["nara-rg255"].exc = QdrantUnavailable("down")
    out = cs.search("flutter", "semantic", 10, archives=both)
    assert out["degraded"] is True and len(out["results"]) == 2  # keyword fallback


def test_empty_default_scope_stays_empty(two_coll_db):
    cs = ChunkSearch(two_coll_db, FakeEmbed(), FakeQdrant(), default_scope=[])
    assert cs.search("flutter", "keyword", 10)["results"] == []
