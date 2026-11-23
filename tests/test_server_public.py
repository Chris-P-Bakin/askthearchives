import io
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build_index import DDL  # noqa: E402
from mcp_server import db as dbmod, server  # noqa: E402
from mcp_server.server import ToolError  # noqa: E402


@pytest.fixture
def tmp_db(tmp_path):
    p = tmp_path / "corpus.db"
    conn = sqlite3.connect(p)
    conn.executescript(DDL)
    for d, n in (("1001", 3), ("1002", 2)):
        conn.execute("INSERT INTO reports VALUES (?,?,?,?,?,?,?,?,?)",
                     (d, d, f"NACA-TN-{d}", "T", "TN", "1940s", "1945", n, ""))
        for pg in range(1, n + 1):
            conn.execute("INSERT INTO pages VALUES (?,?,?,?,?,?,?,?)",
                         (d, pg, f"wing flutter page {pg}", "t", "ocr_draft",
                          "h", 20, 0.0))
            rid = conn.execute("SELECT rowid FROM pages WHERE doc_id=? AND page=?",
                               (d, pg)).fetchone()[0]
            conn.execute("INSERT INTO pages_fts(rowid, text) VALUES (?,?)",
                         (rid, f"wing flutter page {pg}"))
    conn.execute("INSERT INTO index_meta VALUES ('embed_model','x')")
    conn.commit()
    conn.close()
    return p


@pytest.fixture
def tmp_corpus(tmp_db, tmp_path):
    from PIL import Image
    pdir = tmp_path / "data" / "1001" / "pages"
    pdir.mkdir(parents=True)
    Image.new("RGB", (3000, 4000), "white").save(pdir / "1.png")
    return dbmod.Corpus(tmp_db, tmp_path / "data")


def test_instructions_reports_live_counts(tmp_db):
    s = server.instructions_text(tmp_db)
    assert "2 reports" in s and "5 pages" in s and "100 scanned" not in s


def test_search_fts_operator_input_is_not_a_rebuild_error(tmp_corpus, monkeypatch):
    monkeypatch.setattr(server, "MIDDLEWARE", [])
    monkeypatch.setattr(server, "_corpus", tmp_corpus)
    # sanitized path: operator chars never error
    for q in ('"', 'AND OR NOT', 'wing*', '(', 'a:b'):
        server.search(q, mode="keyword")
    # raw unparseable MATCH (sanitizer bypassed) must say "query", not rebuild
    monkeypatch.setattr(dbmod, "_fts_query", lambda q: q)
    with pytest.raises(ToolError) as ei:
        server.search('"', mode="keyword")
    assert "unparseable query" in str(ei.value)
    assert "rebuild" not in str(ei.value)


def test_unreadable_index_still_reports_rebuild(tmp_corpus, monkeypatch):
    monkeypatch.setattr(server, "MIDDLEWARE", [])
    monkeypatch.setattr(server, "_corpus", tmp_corpus)

    def boom(self):
        raise sqlite3.OperationalError("unable to open database file")
    monkeypatch.setattr(dbmod.Corpus, "_conn", boom)
    with pytest.raises(ToolError, match="rebuild in progress"):
        server.search("wing", mode="keyword")


def test_page_image_max_px_clamped(tmp_corpus, monkeypatch):
    from PIL import Image
    monkeypatch.setattr(server, "MIDDLEWARE", [])
    monkeypatch.setattr(server, "_corpus", tmp_corpus)
    img = server.get_page_image("1001", 1, max_px=100000)
    im = Image.open(io.BytesIO(img.data))
    assert max(im.size) <= 2000


@pytest.mark.parametrize("msg", ["database is locked", "disk I/O error",
                                 "no such table: pages_fts"])
def test_match_time_non_syntax_error_is_rebuild_not_unparseable(
        tmp_corpus, monkeypatch, msg):
    monkeypatch.setattr(server, "MIDDLEWARE", [])
    monkeypatch.setattr(server, "_corpus", tmp_corpus)
    real = dbmod.Corpus._conn

    class Proxy:
        def __init__(self, c):
            self.c = c

        def execute(self, sql, *a):
            if "MATCH" in sql:
                raise sqlite3.OperationalError(msg)
            return self.c.execute(sql, *a)

        def close(self):
            self.c.close()
    monkeypatch.setattr(dbmod.Corpus, "_conn", lambda self: Proxy(real(self)))
    with pytest.raises(ToolError) as ei:
        server.search("wing", mode="keyword")
    assert "rebuild in progress" in str(ei.value)
    assert "unparseable" not in str(ei.value) and msg not in str(ei.value)


def test_transport_security_allows_public_host(monkeypatch):
    monkeypatch.setenv("ARCHIVES_PUBLIC_HOST", "mcp.example.org")
    ts = server.transport_security()
    assert ts.enable_dns_rebinding_protection is True
    assert "mcp.example.org" in ts.allowed_hosts
    assert "https://mcp.example.org" in ts.allowed_origins
    assert any(h.startswith("127.0.0.1") for h in ts.allowed_hosts)


def test_transport_security_loopback_only_by_default(monkeypatch):
    monkeypatch.delenv("ARCHIVES_PUBLIC_HOST", raising=False)
    ts = server.transport_security()
    assert ts.enable_dns_rebinding_protection is True
    assert all("example" not in h for h in ts.allowed_hosts)


def test_server_identity():
    assert server.app.name == "askthearchives"
    assert "Current collection" in server.INSTRUCTIONS
    assert "NACA Technical Reports" in server.INSTRUCTIONS


def test_search_routes_to_chunks_when_env_set(monkeypatch):
    class Stub:
        def search(self, query, mode, limit):
            return {"results": [{"chunk_id": "c"}], "degraded": False}
    monkeypatch.setattr(server, "MIDDLEWARE", [])
    monkeypatch.setattr(server, "_chunk_search", Stub())
    assert server.search_chunks("wing") == {"results": [{"chunk_id": "c"}],
                                     "degraded": False}


def test_search_unchanged_without_env(tmp_corpus, monkeypatch):
    monkeypatch.setattr(server, "MIDDLEWARE", [])
    monkeypatch.setattr(server, "_corpus", tmp_corpus)
    out = server.search_pages("wing", mode="keyword")
    assert isinstance(out, list) and out
    assert set(out[0]) == {"doc_id", "page", "snippet", "score", "source",
                           "quality"}


def test_validate_index_skips_pages_vec_when_chunks(tmp_path, monkeypatch):
    p = tmp_path / "c.db"
    conn = sqlite3.connect(p)
    conn.executescript("CREATE TABLE index_meta(key TEXT, value TEXT);"
                       "CREATE TABLE pages(x); CREATE TABLE pages_fts(x);"
                       "INSERT INTO index_meta VALUES(\"schema_version\",\"1\");"
                       "INSERT INTO index_meta VALUES"
                       "(\"embed_model\",\"BAAI/bge-small-en-v1.5\");")
    conn.commit()
    conn.close()
    monkeypatch.setattr(server, "CHUNKS_DB", None)
    with pytest.raises(SystemExit):
        server.validate_index(p)
    monkeypatch.setattr(server, "CHUNKS_DB", Path("x.db"))
    server.validate_index(p)


def test_instructions_mention_chunks_only_when_enabled(tmp_db, monkeypatch):
    monkeypatch.setattr(server, "CHUNKS_DB", None)
    off = server.instructions_text(tmp_db)
    monkeypatch.setattr(server, "CHUNKS_DB", Path("x.db"))
    on = server.instructions_text(tmp_db)
    assert "passage-level" in on and on != off
    assert "passage-level" not in off


def _search_schema(chunks):
    import asyncio
    from mcp.server.mcpserver import MCPServer
    scratch = MCPServer("scratch")
    server.register_search(scratch, chunks)
    tools = asyncio.run(scratch.list_tools())
    (t,) = [t for t in tools if t.name == "search"]
    return t.output_schema if hasattr(t, "output_schema") else t.outputSchema


def test_search_tool_schema_legacy_is_array():
    sch = _search_schema(False)
    assert sch and sch["properties"]["result"]["type"] == "array"


def test_search_tool_schema_chunks_is_dict():
    sch = _search_schema(True)
    assert not sch or "result" not in sch.get("properties", {})


def test_live_search_tool_matches_env():
    import asyncio
    (t,) = [t for t in asyncio.run(server.app.list_tools()) if t.name == "search"]
    sch = getattr(t, "output_schema", None) or getattr(t, "outputSchema", None)
    legacy = bool(sch) and "result" in sch["properties"]
    assert legacy == (not server.CHUNKS_DB)


# ---- /stats route ----

def test_stats_payload_counts(tmp_db, monkeypatch):
    monkeypatch.setattr(server, "DB", tmp_db)
    server._stats_cache.update(t=0.0, payload=None)
    assert server.stats_payload() == {"collections": [
        {"id": "naca", "reports": 2, "pages": 5},
        {"id": "nara-rg255", "status": "digitized"}]}


def test_stats_route_serves_json_with_cors(tmp_db, monkeypatch):
    from starlette.testclient import TestClient
    monkeypatch.setattr(server, "DB", tmp_db)
    server._stats_cache.update(t=0.0, payload=None)
    with TestClient(server.app.streamable_http_app()) as c:
        r = c.get("/stats")
    assert r.status_code == 200
    assert r.json() == {"collections": [{"id": "naca", "reports": 2, "pages": 5},
                                    {"id": "nara-rg255", "status": "digitized"}]}
    assert r.headers["access-control-allow-origin"] == "*"
    assert "max-age" in r.headers.get("cache-control", "")


def test_stats_cached_within_ttl(tmp_db, monkeypatch):
    monkeypatch.setattr(server, "DB", tmp_db)
    server._stats_cache.update(t=0.0, payload=None)
    first = server.stats_cached()
    monkeypatch.setattr(server, "DB", Path("/nonexistent/corpus.db"))
    assert server.stats_cached() == first      # served from cache, no re-read


def test_stats_route_unavailable_db_is_503(tmp_db, monkeypatch):
    from starlette.testclient import TestClient
    monkeypatch.setattr(server, "DB", Path("/nonexistent/corpus.db"))
    server._stats_cache.update(t=0.0, payload=None)
    with TestClient(server.app.streamable_http_app()) as c:
        r = c.get("/stats")
    assert r.status_code == 503


def test_scope_unknown_archive_rejected(monkeypatch):
    from mcp.server.mcpserver.exceptions import ToolError
    monkeypatch.setattr(server, "MIDDLEWARE", [])
    with pytest.raises(ToolError) as e:
        server.search_chunks("wing", archives=["nope"])
    assert "naca" in str(e.value) and "nara-rg255" in str(e.value)


def test_scope_passes_archives_through(monkeypatch):
    got = {}

    class Stub:
        def search(self, query, mode, limit, archives):
            got["a"] = archives
            return {"results": [], "degraded": False}
    monkeypatch.setattr(server, "MIDDLEWARE", [])
    monkeypatch.setattr(server, "_chunk_search", Stub())
    server.search_chunks("wing", archives=["naca"])
    assert got["a"] == ["naca"]
