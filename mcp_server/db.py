"""Read-only query layer over corpus.db + data/ for the Ask the Archives MCP server.

All data access lives here so tools and tests share one code path. Opens
SQLite read-only per connection. Never writes. Operating contract: the
indexer rebuilds corpus.db offline and the server restarts to pick it up;
while a rebuild runs, new connections can fail to open (the rebuild flips
the file to WAL, unreadable through a single-file ro mount) - the server
translates that into a clear retry-shortly tool error.
"""
import contextlib
import io
import json
import re
import sqlite3
import struct
import sys
from pathlib import Path

_PIPELINE = Path(__file__).resolve().parent.parent / "pipeline"
sys.path.insert(0, str(_PIPELINE))


def _load_registry():
    # pipeline/registry module (renamed from collections.py, which shadowed the stdlib name) once pipeline/ is on
    # sys.path, so load it by file path under a private module name.
    import importlib.util
    mod = sys.modules.get("collection_registry")
    if mod is None:
        spec = importlib.util.spec_from_file_location(
            "collection_registry", _PIPELINE / "collections_registry.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["collection_registry"] = mod
        spec.loader.exec_module(mod)
    return mod


registry = _load_registry()

RRF_K = 60
CAND = 50  # candidates taken from each ranker before fusion


class NotFoundError(Exception):
    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


_FTS_SYNTAX_MARKERS = ("fts5: syntax error", "unterminated string",
                       "no such column", "unknown special query")


def _fts_query(q: str) -> str:
    """User text -> safe FTS5 query: quoted terms ORed (no raw operators)."""
    terms = [t for t in re.split(r"[^0-9A-Za-z]+", q) if t]
    return " OR ".join(f'"{t}"' for t in terms) if terms else '""'


class Corpus:
    def __init__(self, db_path: Path, data_dir: Path, entry: dict | None = None):
        self.entry = entry if entry is not None else registry.load()["naca"]
        self.db_path = Path(db_path)
        self.data_dir = Path(data_dir)
        self._embedder = None
        with contextlib.closing(self._conn()) as conn:
            self.meta = dict(conn.execute("SELECT key, value FROM index_meta"))

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    def _embed_query(self, q: str) -> bytes:
        if self._embedder is None:
            from fastembed import TextEmbedding
            self._embedder = TextEmbedding(self.meta["embed_model"])
        v = next(iter(self._embedder.embed([q]))).tolist()
        return struct.pack(f"{len(v)}f", *v)

    # ---------------------------------------------------------------- search
    def search(self, query: str, mode: str = "hybrid", limit: int = 10) -> list[dict]:
        if mode not in ("hybrid", "keyword", "semantic"):
            raise NotFoundError("mode must be one of: hybrid, keyword, semantic")
        with contextlib.closing(self._conn()) as conn:
            return self._search(conn, query, mode, limit)

    def _search(self, conn, query: str, mode: str, limit: int) -> list[dict]:
        kw: list[int] = []
        sem: list[int] = []
        if mode in ("hybrid", "keyword"):
            # Probe first: open/locked failures propagate as OperationalError
            # (server maps them to "rebuild in progress"); only a failure of
            # the MATCH itself below is a query-syntax problem.
            conn.execute("SELECT 1 FROM pages LIMIT 1").fetchone()
            try:
                kw = [r[0] for r in conn.execute(
                    "SELECT rowid FROM pages_fts WHERE pages_fts MATCH ? "
                    "ORDER BY rank LIMIT ?", (_fts_query(query), CAND))]
            except sqlite3.OperationalError as e:
                if any(m in str(e).lower() for m in _FTS_SYNTAX_MARKERS):
                    raise NotFoundError("unparseable query: check quotes and "
                                        "special characters")
                raise  # locked / I/O / missing table: server maps to rebuild msg
        if mode in ("hybrid", "semantic"):
            sem = [r[0] for r in conn.execute(
                "SELECT rowid FROM pages_vec WHERE embedding MATCH ? AND k = ? "
                "ORDER BY distance", (self._embed_query(query), CAND))]
        scores: dict[int, float] = {}
        for lst in (kw, sem):
            for rank, rowid in enumerate(lst):
                scores[rowid] = scores.get(rowid, 0.0) + 1.0 / (RRF_K + rank + 1)
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:limit]
        kw_set = set(kw)
        out = []
        for rowid, score in top:
            row = conn.execute(
                "SELECT doc_id, page, text, source, quality FROM pages "
                "WHERE rowid=?", (rowid,)).fetchone()
            if not row:
                continue
            if rowid in kw_set:
                snip = conn.execute(
                    "SELECT snippet(pages_fts, 0, '[', ']', ' … ', 24) "
                    "FROM pages_fts WHERE rowid=? AND pages_fts MATCH ?",
                    (rowid, _fts_query(query))).fetchone()
                snippet = snip[0] if snip else row[2][:200]
            else:
                snippet = row[2][:200]
            out.append({"doc_id": row[0], "page": row[1], "snippet": snippet,
                        "score": round(score, 5), "source": row[3],
                        "quality": row[4]})
        return out

    def _check_doc(self, doc_id: str) -> None:
        """Reject doc_ids not in the index before any filesystem access
        (blocks path traversal: doc_id is used in data_dir joins)."""
        if not re.fullmatch(r"[0-9A-Za-z_-]+", doc_id or ""):
            raise NotFoundError(f"invalid doc_id {doc_id!r}")
        conn = self._conn()
        try:
            ok = conn.execute("SELECT 1 FROM reports WHERE doc_id=?",
                              (doc_id,)).fetchone()
        finally:
            conn.close()
        if not ok:
            raise NotFoundError(f"unknown doc_id {doc_id!r}")

    # ---------------------------------------------------------------- lookups
    def _page_row(self, conn, doc_id: str, page: int):
        row = conn.execute(
            "SELECT text, source, quality, config_hash, chars FROM pages "
            "WHERE doc_id=? AND page=?", (doc_id, page)).fetchone()
        if row is None:
            rng = conn.execute(
                "SELECT min(page), max(page) FROM pages WHERE doc_id=?",
                (doc_id,)).fetchone()
            if rng[0] is None:
                known = [r[0] for r in conn.execute(
                    "SELECT doc_id FROM reports LIMIT 5")]
                raise NotFoundError(
                    f"unknown doc_id {doc_id!r}; valid doc_ids look like {known}")
            raise NotFoundError(
                f"page {page} not found for {doc_id}; valid range {rng[0]}–{rng[1]}")
        return row

    def page(self, doc_id: str, page: int) -> dict:
        with contextlib.closing(self._conn()) as conn:
            text, source, quality, config_hash, chars = self._page_row(
                conn, doc_id, page)
        return {"doc_id": doc_id, "page": page, "text": text, "source": source,
                "quality": quality, "config_hash": config_hash, "chars": chars,
                "citation": self.cite(doc_id, page)}

    def report(self, doc_id: str) -> dict:
      with contextlib.closing(self._conn()) as conn:
        row = conn.execute(
            "SELECT doc_id, ntrs_id, report_number, title, series, decade, "
            "year, pages, source_url FROM reports WHERE doc_id=?",
            (doc_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"unknown doc_id {doc_id!r}")
        av = {"verified": 0, "vision": 0, "ocr_draft": 0}
        for q, n in conn.execute(
                "SELECT quality, count(*) FROM pages WHERE doc_id=? GROUP BY quality",
                (doc_id,)):
            av[q] = n
        keys = ("doc_id", "record_id", "report_number", "title", "series",
                "decade", "year", "pages", "source_url")
        return {**dict(zip(keys, row)), "availability": av,
                "record_url": registry.record_url(self.entry, row[1])}

    def figures(self, doc_id: str, page: int | None = None) -> list[dict]:
        self._check_doc(doc_id)
        fdir = self.data_dir / doc_id / "figures"
        if not fdir.is_dir():
            return []
        out = []
        pat = f"{page}_*.json" if page is not None else "*.json"
        for f in sorted(fdir.glob(pat)):
            try:
                rec = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            rec["_file"] = f.name
            out.append(rec)
        return out

    def page_image(self, doc_id: str, page: int, bbox: list[int] | None = None,
                   max_px: int = 1200) -> bytes:
        from PIL import Image
        self._check_doc(doc_id)
        png = self.data_dir / doc_id / "pages" / f"{page}.png"
        if not png.exists():
            with contextlib.closing(self._conn()) as conn:
                self._page_row(conn, doc_id, page)  # raises with range info
            raise NotFoundError(f"no image for {doc_id} p{page}")
        max_px = max(100, min(int(max_px), 2000))
        im = Image.open(png)
        if bbox is not None:
            if len(bbox) != 4:
                raise NotFoundError("bbox must be [left, top, right, bottom]")
            l, t, r, b = bbox
            l, t = max(0, min(l, im.width)), max(0, min(t, im.height))
            r, b = max(0, min(r, im.width)), max(0, min(b, im.height))
            if r - l < 1 or b - t < 1:
                raise NotFoundError(
                    f"bbox has zero area after clamping to page "
                    f"{im.width}x{im.height}")
            im = im.crop((l, t, r, b))
        if max(im.size) > max_px:
            scale = max_px / max(im.size)
            im = im.resize((max(1, int(im.width * scale)),
                            max(1, int(im.height * scale))))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    def cite(self, doc_id: str, page: int, page_end: int | None = None) -> dict:
        with contextlib.closing(self._conn()) as conn:
            row = conn.execute(
                "SELECT report_number, title, year, ntrs_id FROM reports "
                "WHERE doc_id=?", (doc_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"unknown doc_id {doc_id!r}")
        num, title, year, ntrs = row
        year_s = f" ({year})" if year else ""
        dp = self.entry["doc_prefix"]
        bare = ntrs[len(dp):] if dp and ntrs.startswith(dp) else ntrs
        pfx = self.entry.get("doc_label_prefix")
        label = (num if not pfx or num.upper().startswith(pfx)
                 else f"{pfx} {num}")
        span = (f"pp. {page}\u2013{page_end}" if page_end and page_end > page
                else f"p. {page}")
        out = {"text": f'{label}, "{title}"{year_s}, {span}. {self.entry["label"]} {bare}.',
               "doc_id": doc_id, "page": page,
               "record_url": registry.record_url(self.entry, ntrs)}
        if page_end and page_end > page:
            out["page_end"] = page_end
        return out
