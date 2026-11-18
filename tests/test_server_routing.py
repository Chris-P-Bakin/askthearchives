import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build_index import DDL  # noqa: E402
from mcp_server import db as dbmod, server  # noqa: E402
from mcp_server.server import ToolError  # noqa: E402

registry = dbmod.registry


def _make_db(path, docs):
    """docs: [(doc_id, ntrs_id, report_number, text)]"""
    conn = sqlite3.connect(path)
    conn.executescript(DDL)
    for d, ntrs, num, text in docs:
        conn.execute("INSERT INTO reports VALUES (?,?,?,?,?,?,?,?,?)",
                     (d, ntrs, num, "T", "TN", "1940s", "1945", 1, ""))
        conn.execute("INSERT INTO pages VALUES (?,?,?,?,?,?,?,?)",
                     (d, 1, text, "t", "ocr_draft", "h", 20, 0.0))
    conn.execute("INSERT INTO index_meta VALUES ('embed_model','x')")
    conn.commit()
    conn.close()


@pytest.fixture
def two(tmp_path, monkeypatch):
    naca_db = tmp_path / "naca.db"
    _make_db(naca_db, [("1001", "1001", "NACA-TN-1001", "naca page text")])
    nroot = tmp_path / "nara"
    nroot.mkdir()
    _make_db(nroot / "corpus.db",
             [("nara255-486937833", "nara255-486937833", "RG255-1", "nara page text")])
    monkeypatch.setattr(server, "MIDDLEWARE", [])
    monkeypatch.setattr(server, "_corpus", dbmod.Corpus(naca_db, tmp_path / "data"))
    monkeypatch.setattr(server, "_corpora", {})
    monkeypatch.setattr(server, "DB", naca_db)
    monkeypatch.setattr(registry, "root",
                        lambda cid: nroot if cid == "nara-rg255" else tmp_path)
    server._stats_cache.update(t=0.0, payload=None)
    return nroot


def test_routing_serves_both_collections(two):
    assert server.get_page("1001", 1)["text"] == "naca page text"
    assert server.get_page("nara255-486937833", 1)["text"] == "nara page text"


def test_routing_rejects_unknown_and_cross_collection_ids(two):
    with pytest.raises(ToolError, match="unknown doc_id"):
        server.get_page("nara255-999", 1)
    with pytest.raises(ToolError, match="unknown doc_id"):
        server.get_page("zz9-1", 1)


def test_nonserving_collection_is_tool_error(two):
    (two / "corpus.db").unlink()
    with pytest.raises(ToolError, match="not yet serving.*nara-rg255"):
        server.get_page("nara255-1", 1)


def test_cite_uses_registry_label_and_url(two):
    c = server.cite("nara255-486937833", 1)
    assert "NARA 486937833" in c["text"]
    assert c["record_url"] == "https://catalog.archives.gov/id/486937833"
    assert not c["text"].startswith("NACA")
    n = server.cite("1001", 1)
    assert n["text"] == 'NACA-TN-1001, "T" (1945), p. 1. NTRS 1001.'
    assert n["record_url"] == "https://ntrs.nasa.gov/citations/1001"


def test_stats_lists_nonserving_collection_status(two):
    (two / "corpus.db").unlink()
    assert server.stats_payload() == {"collections": [
        {"id": "naca", "reports": 1, "pages": 1},
        {"id": "nara-rg255", "status": "digitized"}]}


def test_stats_lists_serving_collection_counts(two):
    assert server.stats_payload()["collections"][1] == {
        "id": "nara-rg255", "reports": 1, "pages": 1}


def test_page_image_link_only_error(two):
    with pytest.raises(ToolError) as e:
        server.get_page_image("nara255-486937833", 1)
    assert str(e.value) == (
        "NASA records, RG 255 is a link-only collection: page images are "
        "served by National Archives (NARA); see "
        "https://catalog.archives.gov/id/486937833")


def test_hosted_collection_image_still_serves(two, tmp_path):
    from PIL import Image as PILImage
    d = tmp_path / "data" / "1001" / "pages"
    d.mkdir(parents=True)
    PILImage.new("RGB", (40, 40), "white").save(d / "1.png")
    img = server.get_page_image("1001", 1)
    assert img.data[:4] == b"\x89PNG"


def test_instructions_state_per_collection_citation_policy():
    assert "link-only" in server._INSTRUCTIONS_TMPL
