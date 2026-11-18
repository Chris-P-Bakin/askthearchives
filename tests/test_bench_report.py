import json

import bench_report


def _row(ts, nid, page, status, norm, engines):
    return {"ts": ts, "ntrs_id": nid, "page": page, "gt_status": status,
            "gt_chars": 100, "norm_hash": norm, "engines": engines}


def _eng(cer, wer=0.5, tr=1.0):
    return {"cer": cer, "wer": wer, "term_recall": tr, "n_terms": 3, "hyp_chars": 90}


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_latest_row_per_engine_page_wins(tmp_path):
    p = tmp_path / "scores.jsonl"
    _write(p, [
        _row("2026-11-01T00:00:00Z", "d1", 1, "done", "aaa", {"e1": _eng(0.9)}),
        _row("2026-11-10T00:00:00Z", "d1", 1, "done", "aaa", {"e1": _eng(0.1)}),
    ])
    rows = bench_report.aggregate(p)
    assert len(rows) == 1
    assert rows[0]["engine"] == "e1"
    assert rows[0]["n"] == 1
    assert rows[0]["median_cer"] == 0.1


def test_draft_gt_rows_excluded(tmp_path):
    p = tmp_path / "scores.jsonl"
    _write(p, [
        _row("2026-11-01T00:00:00Z", "d1", 1, "done", "aaa", {"e1": _eng(0.2)}),
        _row("2026-11-01T00:00:00Z", "d1", 2, "draft", "aaa", {"e1": _eng(0.8)}),
    ])
    rows = bench_report.aggregate(p)
    assert rows[0]["n"] == 1
    assert rows[0]["median_cer"] == 0.2


def test_mixed_norm_hashes_flagged(tmp_path):
    p = tmp_path / "scores.jsonl"
    _write(p, [
        _row("2026-11-01T00:00:00Z", "d1", 1, "done", "aaa", {"e1": _eng(0.2)}),
        _row("2026-11-02T00:00:00Z", "d1", 2, "done", "bbb", {"e1": _eng(0.4)}),
        _row("2026-11-02T00:00:00Z", "d1", 3, "done", "bbb", {"e2": _eng(0.3)}),
    ])
    by = {r["engine"]: r for r in bench_report.aggregate(p)}
    assert by["e1"]["norm"] == "mixed"
    assert by["e2"]["norm"] == "bbb"


def test_lexical_ts_max_assumption(tmp_path):
    # all ts in scores.jsonl are ISO-8601 Z; lexical max == chronological max
    p = tmp_path / "scores.jsonl"
    _write(p, [
        _row("2026-11-09T23:59:59Z", "d1", 1, "done", "aaa", {"e1": _eng(0.9)}),
        _row("2026-11-10T00:00:00Z", "d1", 1, "done", "aaa", {"e1": _eng(0.1)}),
        _row("2026-11-07T14:44:05Z", "d1", 1, "done", "aaa", {"e1": _eng(0.5)}),
    ])
    assert bench_report.aggregate(p)[0]["median_cer"] == 0.1


def test_render_legend_fallback():
    rows = [{"engine": "totally_new_engine", "n": 5, "median_cer": 0.1,
             "median_wer": 0.2, "median_term_recall": 1.0, "norm": "aaa",
             "latest_ts": "2026-11-14T00:00:00Z"}]
    out = bench_report.render(rows)
    assert "totally_new_engine" in out
    assert "(undocumented)" in out


def test_known_engines_have_legend_entries():
    # every engine present in the real scores.jsonl renders without the fallback,
    # i.e. the maintained legend covers the historical record
    import common
    real = common.DATA / "scores.jsonl"
    engines = set()
    with open(real) as f:
        for line in f:
            engines.update(json.loads(line)["engines"].keys())
    missing = engines - set(bench_report.ENGINE_LEGEND)
    assert not missing, f"legend missing: {sorted(missing)}"


def test_marker_replacement_idempotent(tmp_path):
    doc = tmp_path / "BENCHMARKS.md"
    doc.write_text("# B\n\n<!-- bench:start -->\nold\n<!-- bench:end -->\n\ntail\n")
    rows = [{"engine": "e1", "n": 1, "median_cer": 0.1, "median_wer": 0.2,
             "median_term_recall": 1.0, "norm": "aaa",
             "latest_ts": "2026-11-14T00:00:00Z"}]
    bench_report.write_section(doc, "bench", bench_report.render(rows))
    once = doc.read_text()
    bench_report.write_section(doc, "bench", bench_report.render(rows))
    assert doc.read_text() == once
    assert "old" not in once
    assert "tail" in once


def test_table_section_placeholder_when_absent(tmp_path):
    out = bench_report.render_table_section(tmp_path / "nope.jsonl")
    assert "no data yet" in out


def test_table_section_rows(tmp_path):
    p = tmp_path / "table_diff_summary.jsonl"
    _write(p, [{"pair": "vision_glmocr vs vision_8blora_v2_1", "pages": 10,
                "cells_compared": 500, "disagree_rate": 0.031}])
    out = bench_report.render_table_section(p)
    assert "vision_glmocr vs vision_8blora_v2_1" in out
    assert "3.1%" in out


def test_table_section_renders_none_disagree_rate(tmp_path):
    p = tmp_path / "summary.jsonl"
    p.write_text('{"pair": "a vs b", "pages": 3, "cells_compared": 0, "disagree_rate": null}\n')
    out = bench_report.render_table_section(p)
    assert "| a vs b | 3 | 0 | — |" in out
