import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from pass4a import pages_file_pages


def test_pages_file_sorted_tuples(tmp_path):
    rows = [
        {"doc": "222", "page": 3},
        {"doc": "111", "page": 9},
        {"doc": "111", "page": 2},
    ]
    p = tmp_path / "pages.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert pages_file_pages(p) == [("111", 2), ("111", 9), ("222", 3)]


def test_pages_file_skips_blank_lines(tmp_path):
    p = tmp_path / "pages.jsonl"
    p.write_text('{"doc": "111", "page": 1}\n\n')
    assert pages_file_pages(p) == [("111", 1)]
