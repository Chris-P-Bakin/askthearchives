"""chunks.db: authoritative chunk store beside the public corpus.db.

External-content FTS5 over chunk text; the index is kept consistent by
issuing an explicit FTS 'delete' before removing chunk rows (same approach
as build_index.py for pages_fts).
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from chunker import Chunk

SCHEMA_VERSION = "1"

DDL = """
CREATE TABLE IF NOT EXISTS chunk_docs(doc_id TEXT PRIMARY KEY, collection TEXT,
  content_hash TEXT, title TEXT, ntrs_id TEXT, pdf_url TEXT, report_number TEXT,
  year TEXT, page_order_ok INTEGER, chunked_at TEXT);
CREATE TABLE IF NOT EXISTS chunks(chunk_id TEXT PRIMARY KEY, doc_id TEXT,
  seq INTEGER, pages TEXT, page_start INTEGER, page_end INTEGER, section TEXT,
  text TEXT, char_start INTEGER, char_end INTEGER);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  text, content='chunks', content_rowid='rowid');
CREATE TABLE IF NOT EXISTS chunk_meta(key TEXT PRIMARY KEY, value TEXT);
"""

_META_COLS = ("collection", "title", "ntrs_id", "pdf_url", "report_number",
              "year", "page_order_ok")


class ChunkStore:
    def __init__(self, path: Path, readonly: bool = False):
        self.readonly = readonly
        if readonly:
            self.conn = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
        else:
            self.conn = sqlite3.connect(str(path))
            self.conn.executescript(DDL)
            self.conn.execute(
                "INSERT OR IGNORE INTO chunk_meta VALUES('schema_version', ?)",
                (SCHEMA_VERSION,))
            self.conn.commit()
        self.conn.row_factory = sqlite3.Row

    def doc_hashes(self, collection: str | None = None) -> dict[str, str]:
        if collection is not None:
            return {r["doc_id"]: r["content_hash"] for r in self.conn.execute(
                "SELECT doc_id, content_hash FROM chunk_docs WHERE collection=?",
                (collection,))}
        return {r["doc_id"]: r["content_hash"] for r in
                self.conn.execute("SELECT doc_id, content_hash FROM chunk_docs")}

    def _purge(self, doc_id: str) -> None:
        self.conn.execute(
            "INSERT INTO chunks_fts(chunks_fts, rowid, text) "
            "SELECT 'delete', rowid, text FROM chunks WHERE doc_id=?", (doc_id,))
        self.conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        self.conn.execute("DELETE FROM chunk_docs WHERE doc_id=?", (doc_id,))

    def replace_doc(self, doc_id: str, content_hash: str, meta: dict,
                    chunks: list[Chunk]) -> list[str]:
        ids = []
        with self.conn:
            self._purge(doc_id)
            self.conn.execute(
                "INSERT INTO chunk_docs VALUES (?,?,?,?,?,?,?,?,?,?)",
                (doc_id, meta["collection"], content_hash, meta["title"],
                 meta["ntrs_id"], meta.get("pdf_url"), meta["report_number"],
                 meta["year"], meta["page_order_ok"],
                 datetime.now(timezone.utc).isoformat()))
            for seq, c in enumerate(chunks):
                cid = f"{doc_id}:{seq:04d}"
                cur = self.conn.execute(
                    "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (cid, doc_id, seq, json.dumps(c.pages), min(c.pages),
                     max(c.pages), c.section, c.text, c.char_start, c.char_end))
                self.conn.execute(
                    "INSERT INTO chunks_fts(rowid, text) VALUES (?,?)",
                    (cur.lastrowid, c.text))
                ids.append(cid)
        return ids

    def delete_doc(self, doc_id: str) -> None:
        with self.conn:
            self._purge(doc_id)

    def fts_search(self, fts_query: str, limit: int,
                   collections: list[str] | None = None) -> list[str]:
        where, args = "", [fts_query]
        if collections is not None:
            where = (" AND c.doc_id IN (SELECT doc_id FROM chunk_docs "
                     "WHERE collection IN (%s))" % ",".join("?" * len(collections)))
            args += list(collections)
        rows = self.conn.execute(
            "SELECT c.chunk_id FROM chunks_fts JOIN chunks c "
            "ON c.rowid = chunks_fts.rowid WHERE chunks_fts MATCH ?" + where +
            " ORDER BY chunks_fts.rank LIMIT ?", (*args, limit))
        return [r["chunk_id"] for r in rows]

    def get_chunks(self, chunk_ids: list[str]) -> list[dict]:
        out = []
        for cid in chunk_ids:
            r = self.conn.execute(
                "SELECT c.*, d.collection, d.title, d.ntrs_id, d.pdf_url, "
                "d.report_number, d.year, d.page_order_ok "
                "FROM chunks c JOIN chunk_docs d ON d.doc_id = c.doc_id "
                "WHERE c.chunk_id=?", (cid,)).fetchone()
            if r is None:
                continue
            d = dict(r)
            d["pages"] = json.loads(d["pages"])
            out.append(d)
        return out

    def doc_meta(self, doc_id: str) -> dict | None:
        r = self.conn.execute(
            "SELECT doc_id, content_hash, chunked_at, " + ", ".join(_META_COLS)
            + " FROM chunk_docs WHERE doc_id=?", (doc_id,)).fetchone()
        return dict(r) if r else None
