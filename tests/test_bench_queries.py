import json
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent / "bench" / "retrieval"


def test_bench_queries_shape():
    rows = [json.loads(l) for l in open(BENCH / "queries.jsonl")]
    assert len(rows) == 100
    for r in rows:
        assert set(r) == {"query", "doc", "page"}
        assert isinstance(r["page"], int) and r["query"].strip()
