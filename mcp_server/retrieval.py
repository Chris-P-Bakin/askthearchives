"""Chunk-granular search: RRF over Qdrant + chunk FTS, with provenance.

Hybrid = RRF(Qdrant top-CAND, chunk-FTS top-CAND); keyword = FTS only;
semantic = Qdrant only. A failing embed/qdrant dependency degrades to the
FTS leg (degraded=True) rather than erroring. Optional rerank reorders the
fused top-CAND via a llama.cpp /v1/rerank endpoint; its failure is non-fatal.
"""
import contextlib
import json
import sqlite3
import sys
import urllib.request
from pathlib import Path

from mcp_server.db import (CAND, RRF_K, _FTS_SYNTAX_MARKERS,
                           NotFoundError, _fts_query, registry)

from chunk_store import ChunkStore
from embed_client import EmbedUnavailable
from qdrant_client import QdrantUnavailable

# A broken dependency (including malformed 2xx bodies that surface as bare
# KeyError/TypeError/ValueError) must degrade to FTS-only, never 500.
_DEGRADE = (EmbedUnavailable, QdrantUnavailable, KeyError, TypeError, ValueError)
_RERANK_CHARS = 1100


def _log(msg: str) -> None:
    print(f"retrieval: {msg}", file=sys.stderr)


class ChunkSearch:
    def __init__(self, chunks_db: Path, embed, qdrant,
                 rerank_url: str | None = None, corpus=None,
                 qdrant_for=None, corpus_resolver=None, default_scope=None):
        self.chunks_db = Path(chunks_db)
        self.embed = embed
        self.qdrant = qdrant
        self.rerank_url = rerank_url.rstrip("/") if rerank_url else None
        self.corpus = corpus  # quality/source come from corpus.db pages
        # cid -> Qdrant (per-archive collections); without it only naca's
        # `qdrant` is queried.
        self.qdrant_for = qdrant_for
        # doc_id -> Corpus of the hit's own collection; failures degrade the
        # hit's quality/source to None.
        self.corpus_resolver = corpus_resolver
        self.default_scope = (list(default_scope)
                              if default_scope is not None else None)

    # ---------------------------------------------------------------- search
    def search(self, query: str, mode: str, limit: int,
               archives: list[str] | None = None) -> dict:
        if mode not in ("hybrid", "keyword", "semantic"):
            raise NotFoundError("mode must be one of: hybrid, keyword, semantic")
        store = ChunkStore(self.chunks_db, readonly=True)
        try:
            scope = list(archives) if archives is not None else self.default_scope
            return self._search(store, query, mode, limit, scope)
        finally:
            store.conn.close()

    def _fts(self, store, query: str, scope=None) -> list[str]:
        try:
            return store.fts_search(_fts_query(query), CAND, scope)
        except sqlite3.OperationalError as e:
            if any(m in str(e).lower() for m in _FTS_SYNTAX_MARKERS):
                raise NotFoundError("unparseable query: check quotes and "
                                    "special characters")
            raise

    def _semantic(self, query: str, scope=None) -> tuple[list[str], bool]:
        """Returns (candidates, partial). partial=True when some but not all
        scoped collections failed; raises only if every one failed."""
        vec = self.embed.embed_query(query)
        if scope is None or self.qdrant_for is None:
            return [cid for cid, _ in self.qdrant.search(vec, CAND)], False
        # Union of the scoped collections, each ranked list interleaved by
        # rank so RRF sees a single semantic leg.
        lists, errs = [], []
        for a in scope:
            try:
                lists.append([cid for cid, _ in
                              self.qdrant_for(a).search(vec, CAND)])
            except _DEGRADE as e:
                _log(f"semantic collection {a} unavailable: "
                     f"{type(e).__name__}: {e}")
                errs.append(e)
        if errs and not lists:
            raise errs[0]
        merged, seen = [], set()
        for rank in range(max((len(l) for l in lists), default=0)):
            for l in lists:
                if rank < len(l) and l[rank] not in seen:
                    seen.add(l[rank])
                    merged.append(l[rank])
        return merged, bool(errs)

    def _search(self, store, query: str, mode: str, limit: int,
                scope=None) -> dict:
        kw: list[str] = []
        sem: list[str] = []
        degraded = fell_back = False
        if mode in ("hybrid", "semantic"):
            try:
                sem, partial = self._semantic(query, scope)
                degraded = partial
            except _DEGRADE as e:
                _log(f"semantic leg unavailable, degrading to keyword: "
                     f"{type(e).__name__}: {e}")
                degraded = fell_back = True
        if mode in ("hybrid", "keyword") or fell_back:
            kw = self._fts(store, query, scope)
        scores: dict[str, float] = {}
        for lst in (kw, sem):
            for rank, cid in enumerate(lst):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:CAND]
        chunks = {c["chunk_id"]: c
                  for c in store.get_chunks([cid for cid, _ in top])}
        fused = [(chunks[cid], s) for cid, s in top if cid in chunks]
        if self.rerank_url and len(fused) > 1:
            fused = self._rerank(query, fused)
        fused = fused[:limit]
        kw_set = set(kw)
        results = [self._entry(store, c, s, query, c["chunk_id"] in kw_set)
                   for c, s in fused]
        return {"results": results, "degraded": degraded}

    # ---------------------------------------------------------------- rerank
    def _post_json(self, url: str, payload: dict) -> dict:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())

    def _rerank(self, query: str, fused: list) -> list:
        try:
            resp = self._post_json(
                f"{self.rerank_url}/v1/rerank",
                {"model": "q", "query": query,
                 "documents": [c["text"][:_RERANK_CHARS] for c, _ in fused]})
            order = sorted(resp["results"],
                           key=lambda r: -float(r["relevance_score"]))
            idx = [r["index"] for r in order]
            if sorted(idx) != list(range(len(fused))):
                raise ValueError("rerank returned an incomplete ranking")
            return [fused[i] for i in idx]
        except Exception as e:  # rerank is best-effort: keep RRF order
            _log(f"rerank failed, keeping RRF order: {type(e).__name__}: {e}")
            return fused

    # ---------------------------------------------------------------- output
    def _quality_source(self, doc_id: str, page: int):
        corpus = self.corpus
        if self.corpus_resolver is not None:
            try:
                corpus = self.corpus_resolver(doc_id)
            except Exception as e:  # collection not serving: degrade the hit
                _log(f"no corpus for {doc_id}: {type(e).__name__}: {e}")
                return None, None
        if corpus is None:
            return None, None
        try:
            with contextlib.closing(corpus._conn()) as conn:
                row = conn.execute(
                    "SELECT source, quality FROM pages WHERE doc_id=? AND page=?",
                    (doc_id, page)).fetchone()
        except sqlite3.Error:
            # per-hit resolved corpus may be unbuilt/mid-rebuild: degrade the hit;
            # the legacy single corpus keeps raising (surfaced by _run).
            if self.corpus_resolver is None:
                raise
            return None, None
        return (row[1], row[0]) if row else (None, None)

    def _entry(self, store, c: dict, score: float, query: str,
               from_kw: bool) -> dict:
        pages = c["pages"]
        first = pages[0]
        snippet = c["text"][:200]
        if from_kw:
            r = store.conn.execute(
                "SELECT snippet(chunks_fts, 0, '[', ']', ' … ', 24) "
                "FROM chunks_fts WHERE rowid=(SELECT rowid FROM chunks "
                "WHERE chunk_id=?) AND chunks_fts MATCH ?",
                (c["chunk_id"], _fts_query(query))).fetchone()
            if r:
                snippet = r[0]
        quality, source = self._quality_source(c["doc_id"], first)
        pdf = c.get("pdf_url")
        return {
            "collection": c["collection"], "doc_id": c["doc_id"], "page": first,
            "pages": pages, "section": c["section"], "chunk_id": c["chunk_id"],
            "snippet": snippet, "score": round(score, 5),
            "quality": quality, "source": source,
            "original": {
                "record_url": registry.record_url(
                    registry.load().get(c["collection"]) or registry.load()["naca"],
                    c["ntrs_id"]),
                "pdf_url": (f"{pdf}#page={first}"
                            if pdf and c.get("page_order_ok") != 0 else None),
            },
        }
