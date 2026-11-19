import importlib.util
from pathlib import Path

import pytest

_p = Path(__file__).resolve().parent.parent / "bench/retrieval/eval_production.py"
_spec = importlib.util.spec_from_file_location("eval_production", _p)
ep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ep)

QUERIES = [
    {"query": "q1", "doc": "A", "page": 5},
    {"query": "q2", "doc": "B", "page": 7},
    {"query": "q3", "doc": "C", "page": 9},
    {"query": "q4", "doc": "D", "page": 1},
]

RESULTS = {
    # gold at rank 1
    "q1": [{"doc_id": "A", "pages": [5]}],
    # gold page inside a multi-page span at rank 3; wrong-doc same page at 1
    "q2": [{"doc_id": "X", "pages": [7]},
           {"doc_id": "B", "pages": [1]},
           {"doc_id": "B", "pages": [6, 7, 8]}],
    # right doc, page outside span
    "q3": [{"doc_id": "C", "pages": [3, 4]}],
    # gold at rank 6
    "q4": [{"doc_id": "X", "pages": [1]}] * 5 + [{"doc_id": "D", "pages": [1]}],
}


def fake_search(query):
    return {"results": RESULTS[query], "degraded": False}


def test_evaluate_scores_span_hits():
    r = ep.evaluate(fake_search, QUERIES)
    assert r == {1: 0.25, 3: 0.5, 5: 0.5, 10: 0.75}


def test_evaluate_raises_on_degraded():
    with pytest.raises(ep.DegradedSearch):
        ep.evaluate(lambda q: {"results": [], "degraded": True}, QUERIES)


def test_gate_thresholds():
    assert ep.gate_passes(0.70) is True
    assert ep.gate_passes(0.64) is False
    assert ep.gate_passes(0.699) is False
