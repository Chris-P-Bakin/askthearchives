"""Incremental chunk refresh: corpus.db best-text pages -> chunks.db + Qdrant.

A doc is re-chunked only when its content hash changes. chunks.db is never
written in place: changes land on `<db>.staging` and are os.replace'd on
success. Qdrant is not transactional, so per changed doc we delete_by_doc then
upsert; a failed run leaves the live chunks.db old, so the next run redoes the
same docs idempotently.
"""
import argparse
import hashlib
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunk_store import ChunkStore
from chunker import chunk_report
from qdrant_client import QdrantUnavailable

COLLECTION = "naca"
EMBED_BATCH = 32


def doc_content_hash(rows: list[tuple]) -> str:
    """sha256 over the doc's sorted (page, source, config_hash, src_mtime) tuples."""
    h = hashlib.sha256()
    for row in sorted(rows, key=lambda r: r[0]):
        h.update(repr(tuple(row)).encode())
    return h.hexdigest()


def pdf_page_order_ok(doc_dir: Path, rendered_pages: int) -> bool:
    """True only when source.pdf's page count equals the rendered page count."""
    pdf = Path(doc_dir) / "source.pdf"
    if not pdf.exists():
        return False
    try:
        r = subprocess.run(["pdfinfo", str(pdf)], capture_output=True,
                           text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    m = re.search(r"^Pages:\s+(\d+)", r.stdout or "", re.M)
    if r.returncode != 0 or not m:
        return False
    return int(m.group(1)) == rendered_pages


def _rm(p: Path):
    try:
        p.unlink()
    except FileNotFoundError:
        pass


def _embed_input(title: str, chunk) -> str:
    prefix = f"{title} — {chunk.section}: " if chunk.section else f"{title}: "
    return prefix + chunk.text


def _doc_meta(doc_id, roster, report, ok, collection=COLLECTION):
    r = roster.get(doc_id) or {}
    rep = report or {}
    return {
        "collection": collection,
        "title": r.get("title") or rep.get("title") or "",
        "ntrs_id": r.get("ntrs_id") or rep.get("ntrs_id") or doc_id,
        "pdf_url": r.get("pdf_url") or rep.get("source_url") or None,
        "report_number": r.get("report_number") or rep.get("report_number") or "",
        "year": r.get("published_year") or r.get("year") or rep.get("year") or "",
        "page_order_ok": int(ok),
    }


def refresh_chunks(corpus_db: Path, chunks_db: Path, data_dir: Path, embed,
                   qdrant, roster_csv: Path | None = None,
                   collection: str = COLLECTION) -> dict:
    corpus_db, chunks_db, data_dir = Path(corpus_db), Path(chunks_db), Path(data_dir)
    roster = {}
    if roster_csv is not None:
        from common import load_reports
        roster = {r["ntrs_id"]: r for r in load_reports(str(roster_csv))}

    staging = chunks_db.with_suffix(".staging")
    sidecars = [chunks_db.parent / (staging.name + s) for s in ("-wal", "-shm")]
    for p in (staging, *sidecars):
        _rm(p)
    if chunks_db.exists():
        shutil.copy2(chunks_db, staging)

    counts = {"chunked_docs": 0, "deleted_docs": 0, "chunks": 0, "skipped_docs": 0}
    conn = sqlite3.connect(f"file:{corpus_db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    store = None
    try:
        by_doc: dict[str, list] = {}
        for r in conn.execute(
                "SELECT doc_id, page, text, source, config_hash, src_mtime "
                "FROM pages ORDER BY doc_id, page"):
            by_doc.setdefault(r["doc_id"], []).append(r)
        reports = {r["doc_id"]: dict(r) for r in conn.execute("SELECT * FROM reports")}

        store = ChunkStore(staging)
        known = store.doc_hashes(collection)
        qdrant.ensure_collection()

        for doc_id in sorted(set(known) - set(by_doc)):
            store.delete_doc(doc_id)
            qdrant.delete_by_doc(doc_id)
            counts["deleted_docs"] += 1

        for doc_id, rows in by_doc.items():
            h = doc_content_hash([(r["page"], r["source"], r["config_hash"],
                                   r["src_mtime"]) for r in rows])
            if known.get(doc_id) == h:
                counts["skipped_docs"] += 1
                continue
            ok = pdf_page_order_ok(data_dir / doc_id, len(rows))
            meta = _doc_meta(doc_id, roster, reports.get(doc_id), ok, collection)
            chunks = chunk_report([(r["page"], r["text"] or "") for r in rows],
                                  meta["title"], count_tokens=embed.count_tokens)
            vectors = []
            for i in range(0, len(chunks), EMBED_BATCH):
                vectors += embed.embed([_embed_input(meta["title"], c)
                                        for c in chunks[i:i + EMBED_BATCH]])
            if len(vectors) != len(chunks):
                raise RuntimeError(f"{doc_id}: embed returned {len(vectors)} vectors "
                                   f"for {len(chunks)} chunks")
            ids = store.replace_doc(doc_id, h, meta, chunks)
            try:
                qdrant.delete_by_doc(doc_id)
                if ids:
                    qdrant.upsert([
                        (cid, vec, {"chunk_id": cid, "doc_id": doc_id, "pages": c.pages,
                                    "section": c.section, "collection": collection})
                        for cid, vec, c in zip(ids, vectors, chunks)])
            except QdrantUnavailable as e:
                raise QdrantUnavailable(f"doc {doc_id}: {e}") from e
            counts["chunked_docs"] += 1
            counts["chunks"] += len(ids)
    except BaseException:
        if store:
            store.conn.close()
        for p in (staging, *sidecars):
            _rm(p)
        raise
    finally:
        conn.close()

    store.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    store.conn.close()
    for p in sidecars:
        _rm(p)
    os.replace(staging, chunks_db)
    return counts


def main():
    from common import DATA
    from embed_client import EmbedClient
    from qdrant_client import Qdrant
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-db", default=str(DATA / "corpus.db"))
    ap.add_argument("--chunks-db", default=str(DATA / "chunks.db"))
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--roster", default=str(DATA / "roster.csv"))
    a = ap.parse_args()
    embed = EmbedClient(os.environ.get("ARCHIVES_EMBED_URL", "http://127.0.0.1:8201"))
    qdrant = Qdrant.for_collection(
        os.environ.get("ARCHIVES_QDRANT_URL", "http://127.0.0.1:6333"), a.collection)
    roster = Path(a.roster) if Path(a.roster).exists() else None
    print(refresh_chunks(Path(a.corpus_db), Path(a.chunks_db), Path(a.data),
                         embed, qdrant, roster, a.collection))


if __name__ == "__main__":
    main()
