"""Ask the Archives MCP server (trial iteration).

FastMCP over streamable HTTP. Read-only. Tools map 1:1 onto mcp_server.db.Corpus.
Config via env: ARCHIVES_DB, ARCHIVES_DATA, ARCHIVES_HOST, ARCHIVES_PORT;
chunk search (opt-in): ARCHIVES_CHUNKS_DB, ARCHIVES_EMBED_URL,
ARCHIVES_QDRANT_URL, ARCHIVES_RERANK, ARCHIVES_RERANK_URL.
"""
import os
import sqlite3
import sys
import time
from pathlib import Path

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from mcp_server import catalog, obs
from mcp_server.db import Corpus, NotFoundError, registry

DB = Path(os.environ.get("ARCHIVES_DB", "data/corpus.db"))
DATA_DIR = Path(os.environ.get("ARCHIVES_DATA", "data"))
HOST = os.environ.get("ARCHIVES_HOST", "0.0.0.0")
PORT = int(os.environ.get("ARCHIVES_PORT", "8800"))
_chunks_env = os.environ.get("ARCHIVES_CHUNKS_DB")
CHUNKS_DB = Path(_chunks_env) if _chunks_env else None  # unset = page-level search
CATALOG_DB = Path(os.environ.get("ARCHIVES_CATALOG_DB", "data/catalog.db"))
EMBED_URL = os.environ.get("ARCHIVES_EMBED_URL", "http://127.0.0.1:8201")
QDRANT_URL = os.environ.get("ARCHIVES_QDRANT_URL", "http://127.0.0.1:6333")
RERANK = os.environ.get("ARCHIVES_RERANK", "0") == "1"
RERANK_URL = os.environ.get("ARCHIVES_RERANK_URL", "http://127.0.0.1:8202")

EXPECTED_SCHEMA = "1"
EXPECTED_MODEL = "BAAI/bge-small-en-v1.5"

# Middleware slot for the public iteration (auth, rate limiting, blocklist).
# Each entry is called as fn(tool_name, kwargs) before the tool body.
MIDDLEWARE: list = []

_INSTRUCTIONS_TMPL = """\
Ask the Archives: open, machine-transcribed research archives.
Current collection: NACA Technical Reports (1920-1958), public domain (CC0),
{n_reports} reports ({n_pages} pages), scanned and machine-transcribed with
measured accuracy; some pages human-verified. Use responsibly:

- Cite what you quote: every page has a stable {doc_id, page}; use the
  `cite` tool for a formatted citation and a link to the source archive's
  record of the original.
- Citation policy is per collection: hosted collections serve page images
  that are the citable original; link-only collections do not serve images,
  and `cite` links the source archive's record as the citable original.
- `<?>` in text marks a span a transcriber could not read. Never invent
  text to fill one; report it as illegible.
- Check `quality` on every response: `verified` is human-checked ground
  truth; `vision` and `ocr_draft` are machine-generated - say so when
  quoting precise digits from them.
- Figure records with `verified: false` are machine-drafted descriptions.
{chunk_line}- For bulk use, download the published dataset dumps instead of crawling
  these tools.
"""

_CHUNK_LINE = """\
- `search` returns passage-level hits: `pages` is the span the passage covers
  and `original` links back to the source (archive record, PDF page anchor);
  `get_page_image` serves a page scan on demand.
- Consult the catalog/discovery first, then scope `search` with `archives`.
"""


def _fill(n_reports, n_pages) -> str:
    return (_INSTRUCTIONS_TMPL.replace("{chunk_line}",
                                       _CHUNK_LINE if CHUNKS_DB else "")
            .replace("{n_reports}", str(n_reports))
            .replace("{n_pages}", str(n_pages)))


def instructions_text(db_path: Path) -> str:
    """INSTRUCTIONS with live corpus counts read from the index."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        n_reports = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
        n_pages = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
    finally:
        conn.close()
    return _fill(n_reports, n_pages)


try:
    INSTRUCTIONS = instructions_text(DB)
except sqlite3.Error:  # index absent at import (tests); main() validates
    INSTRUCTIONS = _fill("Many", "many")

app = MCPServer("askthearchives", instructions=INSTRUCTIONS)
_corpus: Corpus | None = None  # naca (the env-configured ARCHIVES_DB)
_corpora: dict[str, Corpus] = {}  # other collections, by id
_chunk_search = None

# ---- public stats (homepage live counts) ----
STATS_TTL = 300.0
_stats_cache: dict = {"t": 0.0, "payload": None}


def stats_payload() -> dict:
    """Live corpus counts, per collection. Read-only; raises sqlite3.Error."""
    out = []
    for cid, entry in registry.load().items():
        db = DB if cid == "naca" else registry.root(cid) / "corpus.db"
        if cid != "naca" and not db.exists():
            out.append({"id": cid, "status": entry["status"]})
            continue
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            n_reports = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
            n_pages = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
        finally:
            conn.close()
        out.append({"id": cid, "reports": n_reports, "pages": n_pages})
    return {"collections": out}


def stats_cached() -> dict:
    """stats_payload under a TTL cache: one index read per STATS_TTL window."""
    now = time.monotonic()
    if _stats_cache["payload"] is None or now - _stats_cache["t"] > STATS_TTL:
        _stats_cache["payload"] = stats_payload()
        _stats_cache["t"] = now
    return _stats_cache["payload"]


@app.custom_route("/stats", methods=["GET"])
async def _stats_route(request):
    from starlette.responses import JSONResponse
    headers = {"Access-Control-Allow-Origin": "*",
               "Cache-Control": "public, max-age=300"}
    try:
        return JSONResponse(stats_cached(), headers=headers)
    except sqlite3.Error:
        return JSONResponse({"error": "index unavailable"}, status_code=503,
                            headers=headers)


def corpus() -> Corpus:
    global _corpus
    if _corpus is None:
        _corpus = Corpus(DB, DATA_DIR)
    return _corpus


def corpus_for(doc_id: str) -> Corpus:
    """Corpus serving doc_id, routed by registry doc_prefix. naca is the
    env-configured singleton; other collections open lazily from
    <root>/corpus.db and raise ToolError if that file is absent."""
    entry = registry.for_doc(doc_id)
    cid = entry["id"]
    if cid == "naca":
        return corpus()
    c = _corpora.get(cid)
    if c is None:
        root = registry.root(cid)
        if not (root / "corpus.db").exists():
            raise ToolError(f"collection not yet serving: {cid}")
        c = _corpora[cid] = Corpus(root / "corpus.db", root, entry)
    return c


def _transcribed_ids() -> list[str]:
    return [cid for cid, e in registry.load().items()
            if e.get("status") == "transcribed"]


def chunk_search():
    global _chunk_search
    if _chunk_search is None:
        from embed_client import EmbedClient
        from qdrant_client import Qdrant
        from mcp_server.retrieval import ChunkSearch
        _chunk_search = ChunkSearch(
            CHUNKS_DB, EmbedClient(EMBED_URL, timeout=5),
            Qdrant(QDRANT_URL, timeout=5),
            RERANK_URL if RERANK else None, corpus=corpus(),
            qdrant_for=lambda cid: Qdrant.for_collection(QDRANT_URL, cid, timeout=5),
            corpus_resolver=corpus_for, default_scope=_transcribed_ids())
    return _chunk_search


def validate_index(db_path: Path) -> None:
    if not Path(db_path).exists():
        raise SystemExit(f"corpus.db not found at {db_path}; "
                         "run pipeline/build_index.py first")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        meta = dict(conn.execute("SELECT key, value FROM index_meta"))
    except sqlite3.OperationalError as e:
        raise SystemExit(f"{db_path} is not a corpus index: {e}")
    finally:
        conn.close()
    if meta.get("schema_version") != EXPECTED_SCHEMA:
        raise SystemExit(f"index schema {meta.get('schema_version')!r} != "
                         f"{EXPECTED_SCHEMA!r}; rebuild with --rebuild")
    if meta.get("embed_model") != EXPECTED_MODEL:
        raise SystemExit(f"index embed model {meta.get('embed_model')!r} != "
                         f"{EXPECTED_MODEL!r}; rebuild with --rebuild")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        have = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
            "UNION SELECT name FROM sqlite_master WHERE type='table'")}
        need = {"pages", "pages_fts"} | (set() if CHUNKS_DB else {"pages_vec"})
        missing = need - have
    finally:
        conn.close()
    if missing:
        raise SystemExit(f"index at {db_path} is missing tables {sorted(missing)} "
                         "(built with --no-embed?); rebuild with embeddings")


def _guard(tool: str, kwargs: dict):
    for fn in MIDDLEWARE:
        fn(tool, kwargs)


def _run(tool: str, fn, **kwargs):
    t0 = time.monotonic()
    try:
        _guard(tool, kwargs)
    except BaseException:
        obs.record(tool, t0, "rejected", None)
        raise
    try:
        try:
            result = fn(**kwargs)
        except NotFoundError as e:
            raise ToolError(e.detail)
        except sqlite3.OperationalError:
            raise ToolError("corpus index temporarily unavailable (likely an "
                            "index rebuild in progress) - retry shortly")
    except BaseException:
        obs.record(tool, t0, "error", None)
        raise
    obs.record(tool, t0, "ok", result)
    return result


def search_pages(query: str, mode: str = "hybrid", limit: int = 10) -> list[dict]:
    """Search the corpus. mode: hybrid (default) | keyword | semantic.
    Returns hits with doc_id, page, snippet, score, source, quality."""
    return _run("search", corpus().search, query=query, mode=mode,
                limit=max(1, min(limit, 50)))


def search_chunks(query: str, mode: str = "hybrid", limit: int = 10,
                  archives: list[str] | None = None) -> dict:
    """Search the corpus. mode: hybrid (default) | keyword | semantic.
    archives: collection ids to search (default: all transcribed).
    Returns {results, degraded}; each hit has doc_id, page, pages, section,
    chunk_id, snippet, score, source, quality, original."""
    kwargs = {}
    if archives is not None:
        valid = list(registry.load())
        bad = [a for a in archives if a not in valid]
        if bad:
            raise ToolError(f"unknown archive(s) {bad}; valid ids: {valid}")
        kwargs["archives"] = list(archives)
    return _run("search", chunk_search().search, query=query, mode=mode,
                limit=max(1, min(limit, 50)), **kwargs)


def register_search(target, chunks: bool):
    """Register the page-level or chunk-level function as tool "search".
    Decided once at import so the legacy tool's MCP schema is untouched."""
    return target.tool(name="search")(search_chunks if chunks else search_pages)


search = register_search(app, bool(CHUNKS_DB))


@app.tool()
def get_page(doc_id: str, page: int) -> dict:
    """Full text of one page with provenance (source, quality, config_hash)
    and a ready-made citation."""
    return _run("get_page", corpus_for(doc_id).page, doc_id=doc_id, page=page)


@app.tool()
def get_report(doc_id: str) -> dict:
    """Document metadata, source-record link, and per-quality page availability."""
    return _run("get_report", corpus_for(doc_id).report, doc_id=doc_id)


@app.tool()
def get_figures(doc_id: str, page: int | None = None) -> list[dict]:
    """Structured figure records (charts, photos) for a report or one page.
    Records with verified=false are machine-drafted."""
    return _run("get_figures", corpus_for(doc_id).figures, doc_id=doc_id, page=page)


@app.tool()
def get_page_image(doc_id: str, page: int, bbox: list[int] | None = None,
                   max_px: int = 1200) -> Image:
    """The scanned page image (PNG), optionally cropped to bbox
    [left, top, right, bottom] in deskewed 300-dpi pixel coords
    (figure records and layout files use the same coords)."""
    entry = registry.for_doc(doc_id)
    if entry.get("citation") == "link-only":
        corpus_for(doc_id)  # not-yet-serving collections keep their error
        raise ToolError(
            f"{entry['title']} is a link-only collection: page images are "
            f"served by {entry['source']}; see "
            f"{registry.record_url(entry, doc_id)}")
    png = _run("get_page_image", corpus_for(doc_id).page_image, doc_id=doc_id,
               page=page, bbox=bbox,
               max_px=max(100, min(max_px, 2000)))
    return Image(data=png, format="png")


@app.tool()
def cite(doc_id: str, page: int, page_end: int | None = None) -> dict:
    """Formatted citation + stable identifiers + source-record URL for a page, or a
    page span when page_end is given."""
    return _run("cite", corpus_for(doc_id).cite, doc_id=doc_id, page=page,
                page_end=page_end)


def _statuses() -> dict[str, str]:
    return {cid: e["status"] for cid, e in registry.load().items()}


def _list_collections() -> list[dict]:
    try:
        live = {c["id"]: c for c in stats_cached()["collections"]}
    except sqlite3.Error:
        live = {}
    agg = catalog.aggregates(CATALOG_DB)
    out = []
    for cid, e in registry.load().items():
        row = {k: e.get(k) for k in ("title", "source", "status", "record_url",
                                      "citation")}
        row["id"] = cid
        row.update({k: v for k, v in live.get(cid, {}).items() if k != "id"})
        if cid in agg:
            row["catalog"] = agg[cid]
        out.append(row)
    return out


@app.tool()
def list_collections() -> list[dict]:
    """The archive collections this server covers: id, title, source, status
    (transcribed = searchable full text; digitized = catalog and scans only),
    record URL template, live counts for searchable collections, and catalog
    aggregates (record groups, series, items, digitized items) when built."""
    return _run("list_collections", _list_collections)


@app.tool()
def describe_archive(naId: int) -> dict:
    """One catalog entry (record group or series) by naId: title, scope note,
    date range, item and digitized counts, child entries, and the source
    catalog URL. Use search_catalog to find an naId."""
    return _run("describe_archive", catalog.describe, path=CATALOG_DB,
                na_id=naId, collection_status=_statuses())


@app.tool()
def search_catalog(query: str, level: str | None = None,
                   digitized_only: bool = False, limit: int = 20) -> list[dict]:
    """Full-text search of catalog descriptions (titles and scope notes)
    across all collections. level: recordGroup | series | collection.
    digitized_only keeps entries with digitized items. Each hit has naId,
    level, title, collection, years, n_items, n_digitized, our_status."""
    return _run("search_catalog", catalog.search, path=CATALOG_DB, query=query,
                level=level, digitized_only=digitized_only,
                limit=max(1, min(limit, 50)), collection_status=_statuses())


def transport_security():
    """DNS-rebinding protection stays ON; the tunnel's public hostname is
    allowed via ARCHIVES_PUBLIC_HOST (requests arrive on loopback with the
    public Host header)."""
    from mcp.server.transport_security import TransportSecuritySettings
    hosts = [f"127.0.0.1:{PORT}", f"localhost:{PORT}", "127.0.0.1", "localhost"]
    origins = [f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}"]
    pub = os.environ.get("ARCHIVES_PUBLIC_HOST", "").strip()
    if pub:
        hosts.append(pub)
        origins.append(f"https://{pub}")
    return TransportSecuritySettings(enable_dns_rebinding_protection=True,
                                     allowed_hosts=hosts, allowed_origins=origins)


def main():
    validate_index(DB)
    corpus()  # fail fast if data_dir is wrong too
    rate = float(os.environ.get("ARCHIVES_RATE", "5"))
    if rate > 0:  # ARCHIVES_RATE=0 disables (tests/dev)
        from mcp_server.ratelimit import install_limiter
        install_limiter(rate, int(os.environ.get("ARCHIVES_BURST", "20")))
    app.run(transport="streamable-http", host=HOST, port=PORT,
            transport_security=transport_security())


if __name__ == "__main__":
    main()
