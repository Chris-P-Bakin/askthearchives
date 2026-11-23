import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from pass4a import bucket_pages


def _agreement(tmp_path):
    rows = [
        {"norm_hash": "x", "dir_a": "vision_27bq4", "dir_b": "vision_haiku2"},
        {"doc": "222", "page": 3, "bucket": "conflict", "cer": 0.5},
        {"doc": "111", "page": 9, "bucket": "conflict", "cer": 0.3},
        {"doc": "111", "page": 2, "bucket": "agree", "cer": 0.01},
        {"doc": "333", "page": 1, "bucket": "minor", "cer": 0.05},
    ]
    p = tmp_path / "agreement.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_bucket_pages_filters_and_sorts(tmp_path):
    p = _agreement(tmp_path)
    assert bucket_pages(p, "conflict") == [("111", 9), ("222", 3)]
    assert bucket_pages(p, "minor") == [("333", 1)]


def test_bucket_pages_unknown_bucket_is_empty(tmp_path):
    assert bucket_pages(_agreement(tmp_path), "nope") == []
