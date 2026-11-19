"""SPIKE: three-way retrieval eval — baseline page-index vs chunks vs chunks+rerank.

Metric: recall@k = fraction of queries whose gold (doc,page) appears in top-k
results (a chunk hit counts if the gold page falls in its page span).
"""
import json, sqlite3, struct, sys, time, urllib.request
from pathlib import Path
import numpy as np

S = Path(__file__).parent
ROOT = Path("~/data_normalizing")
EMBED_URL = "http://127.0.0.1:8201/v1/embeddings"
RERANK_URL = "http://127.0.0.1:8202/v1/rerank"
QUERY_INSTRUCT = ("Instruct: Given a web search query, retrieve relevant passages "
                  "that answer the query\nQuery: ")
KS = (1, 3, 5, 10)
RERANK_POOL = 50

queries = [json.loads(l) for l in open(S / "queries.jsonl")]
chunks = [json.loads(l) for l in open(S / "chunks.jsonl")]
embs = np.load(S / "embs.npy").astype(np.float32)
embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)

def embed_query(q):
    body = json.dumps({"input": [QUERY_INSTRUCT + q]}).encode()
    req = urllib.request.Request(EMBED_URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        v = np.asarray(json.load(r)["data"][0]["embedding"], dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)

def rerank(q, texts):
    body = json.dumps({"model": "q", "query": q, "documents": texts}).encode()
    req = urllib.request.Request(RERANK_URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        res = json.load(r)["results"]
    return [r_["index"] for r_ in sorted(res, key=lambda x: -x["relevance_score"])]

def hit(c, gold_doc, gold_page):
    return c["doc"] == gold_doc and gold_page in range(min(c["pages"]), max(c["pages"]) + 1)

# --- baseline: current page-level index (bge-small via fastembed, semantic mode)
sys.path.insert(0, str(ROOT / "mcp_server")); sys.path.insert(0, str(ROOT))
from mcp_server.db import Corpus  # noqa: E402
corpus = Corpus(ROOT / "data/corpus.db", ROOT / "data")

def eval_all():
    n = len(queries)
    base_hits = {k: 0 for k in KS}
    chunk_hits = {k: 0 for k in KS}
    rr_hits = {k: 0 for k in KS}
    for i, row in enumerate(queries):
        q, gd, gp = row["query"], row["doc"], row["page"]
        # baseline
        try:
            res = corpus.search(query=q, mode="semantic", limit=max(KS))
            for k in KS:
                if any(r["doc_id"] == gd and r["page"] == gp for r in res[:k]):
                    base_hits[k] += 1
        except Exception as e:
            print("baseline err", repr(e)[:80])
        # chunks
        qv = embed_query(q)
        sims = embs @ qv
        top = np.argsort(-sims)[:RERANK_POOL]
        for k in KS:
            if any(hit(chunks[j], gd, gp) for j in top[:k]):
                chunk_hits[k] += 1
        # + rerank
        texts = [chunks[j]["text"][:1100] for j in top]
        try:
            order = rerank(q, texts)
        except Exception:
            order = rerank(q[:300], [t[:700] for t in texts])
        rr = [top[j] for j in order]
        for k in KS:
            if any(hit(chunks[j], gd, gp) for j in rr[:k]):
                rr_hits[k] += 1
        if i % 20 == 0:
            print(f"{i}/{n}", flush=True)
    print(f"\nn={n} queries")
    print(f"{'k':>3} {'baseline(page)':>15} {'chunks 0.6B':>12} {'+rerank':>9}")
    for k in KS:
        print(f"{k:>3} {base_hits[k]/n:>15.3f} {chunk_hits[k]/n:>12.3f} {rr_hits[k]/n:>9.3f}")

if __name__ == "__main__":
    t0 = time.time()
    eval_all()
    print(f"({time.time()-t0:.0f}s)")
