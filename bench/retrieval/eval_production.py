"""Production acceptance gate for chunk retrieval.

Runs the 100 bench queries through the live ChunkSearch stack (hybrid mode) and
scores them with the normative hit rule from TRIAL-RESULTS.md: a query hits at k
if any of the top-k results has doc_id == gold doc and the gold page inside the
result's page span. Exits 0 iff the gate passes, 1 on gate failure, 2 if any
search came back degraded (the run would measure the wrong thing).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))

GATE_RECALL5 = 0.70
BASELINE_RECALL5 = 0.540
BASELINE_MARGIN = 0.10


class DegradedSearch(RuntimeError):
    """A search returned degraded:true; the gate cannot be measured."""


def _hit(result: dict, doc: str, page: int) -> bool:
    pages = result.get("pages") or []
    return (result.get("doc_id") == doc and bool(pages)
            and min(pages) <= page <= max(pages))


def evaluate(search_fn, queries, ks=(1, 3, 5, 10)) -> dict[int, float]:
    """search_fn(query) -> {"results": [...], "degraded": bool}."""
    hits = {k: 0 for k in ks}
    for row in queries:
        out = search_fn(row["query"])
        if out.get("degraded"):
            raise DegradedSearch(f"search degraded on query: {row['query'][:80]!r}")
        results = out["results"]
        for k in ks:
            if any(_hit(r, row["doc"], row["page"]) for r in results[:k]):
                hits[k] += 1
    return {k: hits[k] / len(queries) for k in ks}


def gate_passes(recall5: float) -> bool:
    return recall5 >= GATE_RECALL5 and recall5 >= BASELINE_RECALL5 + BASELINE_MARGIN


def main(argv=None) -> int:
    from common import DATA
    from embed_client import EmbedClient
    from qdrant_client import Qdrant
    from mcp_server.retrieval import ChunkSearch

    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-db", default=str(DATA / "chunks.db"))
    ap.add_argument("--embed-url", default="http://127.0.0.1:8201")
    ap.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    ap.add_argument("--queries", default=str(Path(__file__).parent / "queries.jsonl"))
    ap.add_argument("--limit", type=int, default=10)
    a = ap.parse_args(argv)

    queries = [json.loads(line) for line in open(a.queries) if line.strip()]
    cs = ChunkSearch(Path(a.chunks_db), EmbedClient(a.embed_url), Qdrant(a.qdrant_url))
    try:
        recalls = evaluate(lambda q: cs.search(q, "hybrid", a.limit), queries)
    except DegradedSearch as e:
        print(f"GATE ABORTED: {e}. Embed or Qdrant is unavailable; fix the stack "
              "and re-run. This run measured nothing.", file=sys.stderr)
        return 2
    for k, v in recalls.items():
        print(f"recall@{k}: {v:.3f}")
    r5 = recalls[5]
    if gate_passes(r5):
        print(f"GATE PASSED: recall@5 {r5:.3f} >= {GATE_RECALL5} and "
              f">= baseline {BASELINE_RECALL5} + {BASELINE_MARGIN}")
        return 0
    print(f"GATE FAILED: recall@5 {r5:.3f} (need >= {GATE_RECALL5} and >= "
          f"{BASELINE_RECALL5 + BASELINE_MARGIN:.2f})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
