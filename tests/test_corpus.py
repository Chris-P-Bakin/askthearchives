import pytest
from conftest import DATA

from build_index import build
from mcp_server.db import Corpus, NotFoundError

IDS = ["19930090614", "19930092929", "19930093282", "19930091059"]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    db = tmp_path_factory.mktemp("idx") / "corpus.db"
    build(db, DATA, ids=IDS)
    return Corpus(db, DATA)


def test_search_keyword_finds_known_page(corpus):
    hits = corpus.search("aileron hinge-moment coefficient", mode="keyword")
    assert any(h["doc_id"] == "19930092929" for h in hits)


def test_search_hybrid_returns_scored_provenance(corpus):
    hits = corpus.search("hinge moment measurements", mode="hybrid", limit=5)
    assert hits
    assert {"doc_id", "page", "snippet", "score", "source", "quality"} <= set(hits[0])


def test_search_semantic_mode(corpus):
    assert corpus.search("tables of supersonic pressure ratios", mode="semantic", limit=5)


def test_search_fts_special_chars(corpus):
    assert isinstance(corpus.search('wing-flow "NACA (cm*', mode="keyword"), list)


def test_page_provenance_matches_disk(corpus):
    p = corpus.page("19930091059", 23)
    assert p["quality"] == "verified"
    assert p["text"] == (DATA / "19930091059/ground_truth/23.txt").read_text()
    assert "citation" in p


def test_page_empty_text_ok(corpus):
    p = corpus.page("19930093282", 2)
    assert len(p["text"]) < 200 and p["quality"]


def test_unknown_page_raises(corpus):
    with pytest.raises(NotFoundError) as e:
        corpus.page("19930090614", 9999)
    assert "valid" in e.value.detail
    with pytest.raises(NotFoundError):
        corpus.page("nonexistent", 1)


def test_report_availability(corpus):
    r = corpus.report("19930091059")
    assert r["record_id"] == "19930091059"
    av = r["availability"]
    allowed = {"verified", "vision", "vision_agreed", "vision_minor",
               "vision_conflict", "vision_blank", "vision_single", "ocr_draft"}
    assert set(av) <= allowed and "verified" in av and sum(av.values()) > 0


def test_figures_list(corpus):
    figs = corpus.figures("19930092929", 43)
    assert isinstance(figs, list)  # may be empty; never error


def test_cite_format(corpus):
    c = corpus.cite("19930092929", 43)
    assert c["text"].startswith("NACA") and c["doc_id"] == "19930092929"
    assert "ntrs" in c["record_url"]


def test_doc_id_traversal_blocked(corpus):
    for bad in ("../..", "/etc", "19930092929/../19930090614", "..%2F.."):
        with pytest.raises(NotFoundError):
            corpus.figures(bad)
        with pytest.raises(NotFoundError):
            corpus.page_image(bad, 1)


def test_cite_includes_year(corpus):
    assert "(19" in corpus.cite("19930092929", 43)["text"]


def test_connections_closed_on_error_paths(corpus):
    import os
    def db_fds():
        n = 0
        for f in os.listdir("/proc/self/fd"):
            try:
                if os.readlink(f"/proc/self/fd/{f}").endswith("corpus.db"):
                    n += 1
            except OSError:
                pass
        return n
    kept = []
    base = db_fds()
    for _ in range(25):
        try:
            corpus.page("19930092929", 9999)
        except NotFoundError as e:
            kept.append(e)  # hold tracebacks so refcounting can't save us
    assert db_fds() <= base + 1, f"leaked {db_fds() - base} fds on error path"
