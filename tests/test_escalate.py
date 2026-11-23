import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import escalate  # noqa: E402


def _row(uid, kind, ts, **kw):
    ntrs, _, page = uid.rpartition("-p")
    base = {"unit_id": uid, "ntrs_id": ntrs, "page": int(page), "ts": ts, "kind": kind}
    if kind == "wu_error":
        base.update(error_code="1", wu_name="naca_" + uid)
    else:
        base.update(config_hash="6ecf7d0dbeb2",
                    flags=[{"table": 0, "check": "monotonic", "col": 1, "rows": [1]}])
    base.update(kw)
    return base


def _write_queue(data, rows):
    data.mkdir(parents=True, exist_ok=True)
    q = data / "escalation_queue.jsonl"
    q.write_text("".join(json.dumps(r) + "\n" for r in rows) + "not json\n")
    return q


def _png(data, ntrs, page):
    p = data / ntrs / "pages" / f"{page}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"png")


def test_dedupe_keeps_latest(tmp_path):
    q = _write_queue(tmp_path, [
        _row("100-p1", "table_flags", "2026-11-14T01:00:00Z", config_hash="old"),
        _row("100-p1", "table_flags", "2026-11-14T02:00:00Z", config_hash="new"),
        _row("100-p1", "vision_failed", "2026-11-14T00:00:00Z"),
    ])
    rows = escalate.read_queue(q)
    assert len(rows) == 2
    tf = [r for r in rows if r["kind"] == "table_flags"][0]
    assert tf["config_hash"] == "new"


def test_both_kinds_routes_235b_only(tmp_path):
    _png(tmp_path, "100", 1)
    _png(tmp_path, "100", 2)
    _png(tmp_path, "100", 3)
    rows = [_row("100-p1", "table_flags", "t"), _row("100-p1", "vision_failed", "t"),
            _row("100-p2", "vision_failed", "t"), _row("100-p3", "wu_error", "t")]
    r = escalate.route(rows, data=tmp_path)
    assert [x["unit_id"] for x in r["235b"]] == ["100-p1"]
    assert [x["kind"] for x in r["235b"]] == ["table_flags"]
    assert [x["unit_id"] for x in r["27b"]] == ["100-p2"]
    assert [x["unit_id"] for x in r["wu_error"]] == ["100-p3"]
    assert r["skipped_missing"] == []


def test_missing_page_routes_skipped(tmp_path):
    _png(tmp_path, "100", 1)
    rows = [_row("100-p1", "table_flags", "t"), _row("100-p2", "table_flags", "t")]
    r = escalate.route(rows, data=tmp_path)
    assert [x["unit_id"] for x in r["235b"]] == ["100-p1"]
    assert [x["unit_id"] for x in r["skipped_missing"]] == ["100-p2"]


def test_pending_drops_done_layer_pages(tmp_path):
    rows = [_row("100-p1", "vision_failed", "t"), _row("100-p2", "vision_failed", "t"),
            _row("100-p3", "vision_failed", "t")]
    for page, h in ((1, escalate.ESC_27B_HASH), (2, "stale")):
        d = tmp_path / "100" / escalate.LAYER_27B
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{page}.meta.json").write_text(json.dumps({"config_hash": h}))
    out = escalate.pending(rows, escalate.LAYER_27B, escalate.ESC_27B_HASH, data=tmp_path)
    assert [x["unit_id"] for x in out] == ["100-p2", "100-p3"]


def test_done_ledger_roundtrip(tmp_path):
    row = _row("100-p1", "wu_error", "t")
    other = _row("100-p1", "table_flags", "t")
    assert not escalate.is_done(row, data=tmp_path)
    escalate.mark_done(row, "resubmitted", data=tmp_path, wu="x")
    assert escalate.is_done(row, data=tmp_path)
    assert not escalate.is_done(other, data=tmp_path)
    line = json.loads((tmp_path / "escalation_done.jsonl").read_text().splitlines()[0])
    assert line["disposition"] == "resubmitted" and line["wu"] == "x"


def test_review_enqueue_dedupes(tmp_path):
    p = tmp_path / "review_queue.jsonl"
    row = _row("100-p1", "table_flags", "t")
    assert escalate.enqueue_review(row, "digits", "vision_235b", "h", path=p) is True
    assert escalate.enqueue_review(row, "digits", "vision_235b", "h", path=p) is False
    assert escalate.enqueue_review(row, "loop", "vision_235b", "h", path=p) is True
    lines = [json.loads(x) for x in p.read_text().splitlines()]
    assert len(lines) == 2
    assert set(lines[0]) == {"unit_id", "ntrs_id", "page", "reason", "source_layer",
                             "config_hash", "ts"}


def _acct(st, routed):
    return (st["dedupe_dropped"] + len(st["malformed"])
            + sum(len(v) for v in routed.values()))


def test_accounting_law_with_bad_rows(tmp_path):
    _png(tmp_path, "100", 1)
    _png(tmp_path, "100", 2)
    q = tmp_path / "escalation_queue.jsonl"
    lines = [
        json.dumps(_row("100-p1", "table_flags", "2026-01-01T00:00:00Z")),
        json.dumps(_row("100-p1", "table_flags", "2026-01-02T00:00:00Z")),  # dedupe
        json.dumps(_row("100-p1", "vision_failed", "t")),                   # suppressed
        json.dumps(_row("100-p2", "zz", "t")),                              # unknown kind
        json.dumps(_row("100-p2", "vision_failed", "t")),                   # 27b
        json.dumps({"unit_id": ["a"], "kind": "table_flags"}),              # list unit_id
        json.dumps({"unit_id": "x", "kind": ["table_flags"]}),              # list kind
        json.dumps({"kind": "table_flags", "page": 1}),                     # no unit_id
        json.dumps([1, 2]),                                                 # non-dict
        '{"unit_id": "100-p9", "kind": "tab',                               # torn
    ]
    q.write_text("\n".join(lines) + "\n\n")
    rows, st = escalate.read_queue_ex(q)
    assert st["lines"] == 10
    assert len(st["malformed"]) == 5 and st["malformed"][-1].startswith('{"unit_id": "100-p9"')
    assert st["dedupe_dropped"] == 1
    routed = escalate.route(rows, data=tmp_path)
    assert [r["unit_id"] for r in routed["unknown_kind"]] == ["100-p2"]
    assert [r["kind"] for r in routed["suppressed"]] == ["vision_failed"]
    assert [r["unit_id"] for r in routed["27b"]] == ["100-p2"]
    assert _acct(st, routed) == st["lines"]
    assert len(escalate.read_queue(q)) == len(rows)
