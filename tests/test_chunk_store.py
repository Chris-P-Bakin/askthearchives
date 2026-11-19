import sqlite3

import pytest

from chunk_store import ChunkStore
from chunker import Chunk


def _meta(**kw):
    m = dict(collection="c1", title="A Report", ntrs_id="19700001",
             pdf_url="https://x/y.pdf", report_number="TN D-1", year="1970",
             page_order_ok=1)
    m.update(kw)
    return m


def _chunk(text, pages=(1,), section="S", start=0):
    return Chunk(pages=list(pages), section=section, text=text,
                 char_start=start, char_end=start + len(text))


def test_replace_doc_roundtrip(tmp_path):
    st = ChunkStore(tmp_path / "chunks.db")
    ids = st.replace_doc("d1", "h1", _meta(),
                         [_chunk("alpha beta", [1, 2]), _chunk("gamma", [3], "Intro", 11)])
    assert ids == ["d1:0000", "d1:0001"]
    rows = st.get_chunks(ids)
    assert rows[0]["text"] == "alpha beta"
    assert rows[0]["pages"] == [1, 2]
    assert (rows[0]["page_start"], rows[0]["page_end"]) == (1, 2)
    assert rows[1]["section"] == "Intro"
    assert (rows[1]["char_start"], rows[1]["char_end"]) == (11, 16)
    assert rows[1]["seq"] == 1 and rows[1]["doc_id"] == "d1"
    assert rows[0]["title"] == "A Report" and rows[0]["collection"] == "c1"
    assert st.doc_hashes() == {"d1": "h1"}


def test_get_chunks_order_and_unknown(tmp_path):
    st = ChunkStore(tmp_path / "chunks.db")
    st.replace_doc("d1", "h", _meta(), [_chunk("one"), _chunk("two")])
    rows = st.get_chunks(["d1:0001", "nope:0000", "d1:0000"])
    assert [r["chunk_id"] for r in rows] == ["d1:0001", "d1:0000"]


def test_replace_doc_shrinks_cleanly(tmp_path):
    st = ChunkStore(tmp_path / "chunks.db")
    old = st.replace_doc("d1", "h1", _meta(),
                         [_chunk("zebra one"), _chunk("zebra two"), _chunk("zebra three")])
    assert len(st.fts_search("zebra", 10)) == 3
    new = st.replace_doc("d1", "h2", _meta(), [_chunk("aardvark")])
    assert new == ["d1:0000"]
    assert [r["chunk_id"] for r in st.get_chunks(old)] == ["d1:0000"]
    assert st.get_chunks(old)[0]["text"] == "aardvark"
    assert st.fts_search("zebra", 10) == []
    assert st.fts_search("aardvark", 10) == ["d1:0000"]
    assert st.doc_hashes() == {"d1": "h2"}


def test_delete_doc(tmp_path):
    st = ChunkStore(tmp_path / "chunks.db")
    st.replace_doc("d1", "h", _meta(), [_chunk("quokka")])
    st.replace_doc("d2", "h", _meta(), [_chunk("quokka too")])
    st.delete_doc("d1")
    assert st.fts_search("quokka", 10) == ["d2:0000"]
    assert st.doc_meta("d1") is None
    assert st.doc_hashes() == {"d2": "h"}


def test_fts_search_ranks_match(tmp_path):
    st = ChunkStore(tmp_path / "chunks.db")
    st.replace_doc("d1", "h", _meta(), [_chunk("common words only"),
                                        _chunk("contains the rare term xylophonic here")])
    assert st.fts_search("xylophonic", 5) == ["d1:0001"]
    assert len(st.fts_search("common OR xylophonic", 1)) == 1


def test_meta_without_pdf_url(tmp_path):
    st = ChunkStore(tmp_path / "chunks.db")
    st.replace_doc("d1", "h", _meta(pdf_url=None), [_chunk("x")])
    m = st.doc_meta("d1")
    assert m["pdf_url"] is None
    assert m["title"] == "A Report" and m["page_order_ok"] == 1
    assert st.doc_meta("missing") is None


def test_readonly_rejects_write(tmp_path):
    p = tmp_path / "chunks.db"
    ChunkStore(p).replace_doc("d1", "h", _meta(), [_chunk("kiwi")])
    ro = ChunkStore(p, readonly=True)
    assert ro.fts_search("kiwi", 5) == ["d1:0000"]
    with pytest.raises(sqlite3.OperationalError):
        ro.replace_doc("d2", "h", _meta(), [_chunk("x")])
