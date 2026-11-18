import json

import pytest

import common
import escalate
import ledger

L8 = "vision_8blora_v2_1"


def _w(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def _page(data, nid, n, conf=90, assim=None, failed=False, layer=L8, cost=None, h="6ecf7d0dbeb2"):
    _w(data / nid / "ocr" / f"{n}.json", {"words": [{"conf": conf}]})
    if assim is not None or layer != L8 or failed or cost is not None:
        m = {"config_hash": h, "loop_guard": {"vision_failed": failed}}
        if assim:
            m["assimilated_ts"] = assim
        if cost is not None:
            m["usage"] = {"cost": cost}
        _w(data / nid / layer / f"{n}.meta.json", m)


@pytest.fixture
def tree(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setattr(common, "DATA", data)
    _w(data / "batches" / "batch_003.json", {"batch": 3, "ids": ["A", "B"]})
    _page(data, "A", 1, 50, assim="2026-01-01T00:00:00Z")
    _page(data, "A", 2, 50, assim="2026-01-01T02:00:00Z", failed=True)
    _page(data, "A", 3, 95)
    _page(data, "B", 1, 50, assim="2026-01-01T01:00:00Z")
    _page(data, "B", 2, 50)
    _page(data, "B", 3, 95)
    _page(data, "A", 2, 50, layer="vision_27b_v2", h="6a169a212aac")
    _page(data, "A", 1, 50, layer="vision_235b", cost=0.25, h="x")
    (data / "escalation_done.jsonl").write_text(
        json.dumps({"layer": "vision_235b", "unit_id": "B-p1", "cost": 0.5}) + "\n")
    (data / "escalation_review.jsonl").write_text(
        json.dumps({"unit_id": "A-p2", "ntrs_id": "A", "page": 2}) + "\n"
        + json.dumps({"unit_id": "Z-p2", "ntrs_id": "Z", "page": 2}) + "\n")
    (data / "escalation_queue.jsonl").write_text(
        json.dumps({"unit_id": "A-p1", "ntrs_id": "A", "page": 1, "kind": "table_flags"}) + "\n")
    return data


def _nodocker(*a, **k):
    raise OSError("no docker")


def test_row_arithmetic_reconciles(tree, monkeypatch):
    monkeypatch.setattr(ledger.subprocess, "run", _nodocker)
    r = ledger.batch_row(3, data=tree, now="T")
    assert (r["docs"], r["pages"], r["gated"], r["ungated"]) == (2, 6, 4, 2)
    assert r["gated"] + r["ungated"] == r["pages"]
    assert r["gate_pct"] == pytest.approx(66.67, abs=0.01)
    assert r["escalated_27b"] == 1 and r["escalated_235b"] == 2
    assert r["escalated_27b"] + r["escalated_235b"] <= r["gated"]
    assert r["review_rows"] == 1
    assert r["vision_failed_pct"] == 25.0 and r["flag_pct"] == 25.0
    assert r["pages_per_hour"] == pytest.approx(3 / 2, abs=0.01)
    assert r["boinc"] is None and r["published_ts"] == "T"
    assert [w for w in r["warnings"] if "boinc" not in w] == []


def test_mismatch_warns_not_crashes(tree, monkeypatch):
    monkeypatch.setattr(ledger.subprocess, "run", _nodocker)
    _page(tree, "A", 3, 95, layer="vision_235b", cost=1, h="x")
    _page(tree, "B", 3, 95, layer="vision_27b_v2", h="x")
    r = ledger.batch_row(3, data=tree, now="T")
    assert any("escalated" in w for w in r["warnings"])
    (tree / "B" / "ocr" / "3.json").write_text("{not json")
    r = ledger.batch_row(3, data=tree, now="T")
    assert any("gate computation failed" in w for w in r["warnings"])


def test_spend_matches_escalate_spend(tree, monkeypatch):
    monkeypatch.setattr(ledger.subprocess, "run", _nodocker)
    r = ledger.batch_row(3, data=tree, now="T")
    assert r["spend_usd_cumulative"] == escalate.spend(tree) == 0.75


def test_boinc_counts_and_failure(tree, monkeypatch):
    class P:
        returncode = 0
        stdout = "10\t8\t1\n"
    monkeypatch.setattr(ledger.subprocess, "run", lambda *a, **k: P())
    r = ledger.batch_row(3, data=tree, now="T")
    assert r["boinc"] == {"workunits": 10, "results_valid": 8, "results_invalid": 1}
    monkeypatch.setattr(ledger.subprocess, "run", _nodocker)
    r = ledger.batch_row(3, data=tree, now="T")
    assert r["boinc"] is None and any("boinc" in w for w in r["warnings"])


def test_write_replaces_batch_row_and_markdown_idempotent(tree, tmp_path, monkeypatch):
    monkeypatch.setattr(ledger.subprocess, "run", _nodocker)
    lp, rp = tmp_path / "pub" / "ledger.jsonl", tmp_path / "RESULTS.md"
    rp.write_text("# R\n\nintro\n\n<!-- ledger:start -->\nold\n<!-- ledger:end -->\n\nfooter\n")
    ledger.write(ledger.batch_row(3, data=tree, now="T1"), lp, rp)
    ledger.write(ledger.batch_row(3, data=tree, now="T2"), lp, rp)
    _w(tree / "batches" / "batch_001.json", {"batch": 1, "ids": []})
    ledger.write(ledger.batch_row(1, data=tree, now="T3"), lp, rp)
    rows = [json.loads(x) for x in lp.read_text().splitlines()]
    assert [(r["batch"], r["published_ts"]) for r in rows] == [(1, "T3"), (3, "T2")]
    once = rp.read_text()
    ledger.write(ledger.batch_row(1, data=tree, now="T3"), lp, rp)
    assert rp.read_text() == once
    assert once.count("<!-- ledger:start -->") == 1 and "old" not in once
    assert once.startswith("# R") and once.rstrip().endswith("footer")
    assert "| 3 |" in once and "| 1 |" in once


def test_cumulative_spend_field_and_footnote(tree, tmp_path, monkeypatch):
    monkeypatch.setattr(ledger.subprocess, "run", _nodocker)
    r = ledger.batch_row(3, data=tree, now="T")
    assert "spend_usd" not in r and r["spend_usd_cumulative"] == escalate.spend(tree)
    out = ledger.table([r])
    assert "spend $ (cumulative)" in out and "campaign-wide" in out.splitlines()[-1]


def test_warning_text_rendered_escaped_and_truncated():
    row = {"batch": 1, "warnings": ["a | b", "second", "third"]}
    out = ledger.table([row])
    assert "a \\| b; second (+1 more, see ledger.jsonl)" in out
    assert "third" not in out
    out = ledger.table([{"batch": 1, "warnings": ["x | y"]}])
    assert "x \\| y" in out
