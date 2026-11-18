import csv
import json
from pathlib import Path

import roster
from roster import (classify_series, extract_row, fetch_all, split_window,
                    write_roster, year_windows)

FIX = Path(__file__).parent / "fixtures"
REC = json.loads((FIX / "ntrs_record.json").read_text())
REC_NO_PDF = json.loads((FIX / "ntrs_record_no_pdf.json").read_text())


def test_classify_series():
    assert classify_series(["NACA-TR-1135"]) == "TR"
    assert classify_series(["NACA-TN-2645"]) == "TN"
    assert classify_series(["NACA-TM-312"]) == "TM"
    assert classify_series(["NACA-RM-A51J09"]) == "RM"
    assert classify_series(["NACA-WR-L-560", "NACA-ARR-3G13"]) == "WR"
    assert classify_series(["NASA-CR-1234"]) == "other"
    assert classify_series([]) == "other"


def test_extract_row_full_record():
    row = extract_row(REC)
    assert row["ntrs_id"] == str(REC["id"])
    assert row["pdf_url"].startswith("https://ntrs.nasa.gov/api/citations/")
    assert row["no_pdf"] is False and isinstance(row["published_year"], int)
    assert row["report_number"] == "NACA-TM-196"
    assert row["series"] == "TM"


def test_extract_row_no_downloads():
    row = extract_row(REC_NO_PDF)
    assert row["no_pdf"] is True and row["pdf_url"] == ""


def test_year_windows_cover_and_split():
    ws = year_windows(1900, 1975, step=10)
    assert ws[0][0].startswith("1900") and ws[-1][1] > ws[-1][0]
    assert all(a[1] == b[0] for a, b in zip(ws, ws[1:]))
    a, b = split_window(("1930-01-01", "1940-01-01"))
    assert a[1] == b[0] and a[0] == "1930-01-01" and b[1] == "1940-01-01"


def _fake_post(db, calls):
    """db: list of (date, id). Honors published window + page from/size."""
    def post(body):
        calls.append(body)
        w = body.get("published")
        rows = [(d, i) for d, i in db if w is None or w["gte"] <= d < w["lt"]]
        srt = body.get("sort")
        if srt:
            rows.sort(key=lambda x: (x[0], x[1]), reverse=srt["order"] == "desc")
        hits = [{"id": i} for d, i in rows]
        off, size = body["page"]["from"], body["page"]["size"]
        return {"results": hits[off:off + size], "stats": {"total": len(hits)}}
    return post


def test_fetch_all_dedupes_and_splits(monkeypatch):

    monkeypatch.setattr(roster, "MAX_WINDOW", 150)
    db = [(f"1925-{1 + i % 12:02d}-01", i) for i in range(250)]
    db += [("1925-01-01", 5)]  # duplicate id
    calls = []
    monkeypatch.setattr(roster, "year_windows", lambda *a, **k: [("1920-01-01", "1930-01-01")])
    recs = fetch_all(_fake_post(db, calls), union=False)
    assert len(recs) == len({r["id"] for r in recs}) == 250
    assert len(calls) > 3  # window was split and re-fetched


def test_year_windows_cover_1975_inclusive():
    ws = roster.year_windows()
    assert ws[-1][1] >= "1976-01-01"
    assert ws[0][0] == "1900-01-01"


def test_union_adds_records_windows_missed(monkeypatch):
    # records dated outside every window (e.g. missing/odd published) are only
    # reachable through the sorted sweeps; page window shrunk to exercise paging
    monkeypatch.setattr(roster, "PAGE_SIZE", 10)
    monkeypatch.setattr(roster, "UNION_WINDOW", 40)
    monkeypatch.setattr(roster, "year_windows", lambda *a, **k: [("1920-01-01", "1930-01-01")])
    db = [(f"1925-{1 + i % 12:02d}-01", i) for i in range(30)]
    db += [("1990-01-01", 100 + i) for i in range(5)]       # outside the windows
    calls = []
    recs = fetch_all(_fake_post(db, calls))
    ids = {r["id"] for r in recs}
    assert len(recs) == len(ids) == 35
    assert {100, 104} <= ids
    assert {c.get("sort", {}).get("field") for c in calls} >= {"published", "created"}


def test_union_total_mismatch_raises(monkeypatch):
    monkeypatch.setattr(roster, "UNION_WINDOW", 20)
    monkeypatch.setattr(roster, "year_windows", lambda *a, **k: [("1920-01-01", "1930-01-01")])
    db = [("1925-01-01", i) for i in range(5)]
    post = _fake_post(db, [])

    def lying(body):
        out = post(body)
        if "published" not in body and "sort" not in body:
            out["stats"]["total"] += 3
        return out
    try:
        fetch_all(lying)
    except RuntimeError as e:
        assert "total" in str(e)
        return
    raise AssertionError("expected RuntimeError")


def test_write_roster_refuses_fewer_rows_without_force(tmp_path, monkeypatch):
    monkeypatch.setattr(roster, "load_reports", lambda p=None: [])
    csvp, jp = tmp_path / "r.csv", tmp_path / "r.jsonl"
    write_roster([REC, REC_NO_PDF], csvp, jp)
    before = csvp.read_text()
    try:
        write_roster([REC], csvp, jp)
    except RuntimeError as e:
        assert "force" in str(e)
    else:
        raise AssertionError("expected refusal")
    assert csvp.read_text() == before and len(jp.read_text().splitlines()) == 2
    write_roster([REC], csvp, jp, force=True)
    assert len(list(csv.DictReader(open(csvp)))) == 1


def test_in_pilot_uses_pilot_csv_not_env(tmp_path, monkeypatch):
    decoy = tmp_path / "decoy.csv"
    extra = dict(REC, id=999999999)                  # in decoy only, not in the pilot
    decoy.write_text("ntrs_id\n999999999\n")
    monkeypatch.setenv("NACA_REPORTS_CSV", str(decoy))
    seen = []
    real = roster.load_reports
    monkeypatch.setattr(roster, "load_reports", lambda p=None: (seen.append(p), real(p))[1])
    write_roster([REC, extra], tmp_path / "r.csv", tmp_path / "r.jsonl")
    assert seen == [str(roster.common.CSV)]
    rows = list(csv.DictReader(open(tmp_path / "r.csv")))
    assert [r["in_pilot"] for r in rows] == ["True", "False"]


def test_extract_row_decade_and_pub_date():
    row = extract_row(REC)
    assert row["pub_date"] == "1923-03-01" and row["decade"] == "1920s"
    undated = extract_row({"id": 1, "title": "x"})
    assert undated["pub_date"] == "" and undated["decade"] == ""


def test_pdf_url_rejects_non_pdf_original():
    rec = {"id": 20230002622, "title": "video", "downloads": [
        {"mimetype": "video/mp4", "links": {"original": "/api/citations/20230002622/downloads/v.mp4"}}]}
    row = extract_row(rec)
    assert row["no_pdf"] is True and row["pdf_url"] == ""
    pptx = {"id": 2, "downloads": [{"mimetype": "application/octet-stream",
            "links": {"original": "/d/x.pptx"}}]}
    assert extract_row(pptx)["no_pdf"] is True
    # .pdf suffix with odd mimetype is accepted
    odd = {"id": 3, "downloads": [{"mimetype": "application/octet-stream",
           "links": {"original": "/d/x.PDF"}}]}
    assert extract_row(odd)["no_pdf"] is False


def test_throttle_is_four_seconds(monkeypatch):
    assert roster.THROTTLE == 4.0
    seen = []
    monkeypatch.setattr(roster.common, "ntrs_wait", lambda *a, **k: seen.append(a))

    class R:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b"{}"
    monkeypatch.setattr(roster.urllib.request, "urlopen", lambda *a, **k: R())
    roster.post_search({})
    assert seen == [(4.0,)]


def test_from_jsonl_rebuild_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(roster, "load_reports", lambda p=None: [])
    jp, csvp = tmp_path / "rec.jsonl", tmp_path / "r.csv"
    jp.write_text("".join(json.dumps(r) + "\n" for r in (REC, REC_NO_PDF)))
    summary = roster.rebuild_from_jsonl(jp, csvp)
    rows = list(csv.DictReader(open(csvp)))
    assert [r["ntrs_id"] for r in rows] == [str(REC["id"]), str(REC_NO_PDF["id"])]
    assert rows[0]["decade"] == "1920s" and summary["total"] == 2
    assert list(rows[0]) == roster.FIELDS
    assert jp.read_text().count("\n") == 2          # jsonl untouched
    assert sorted(p.name for p in tmp_path.iterdir()) == ["r.csv", "rec.jsonl"]


def test_write_roster(tmp_path, monkeypatch):
    monkeypatch.setattr(roster, "load_reports",
                        lambda p=None: [{"ntrs_id": str(REC["id"])}])
    csvp, jp = tmp_path / "r.csv", tmp_path / "r.jsonl"
    summary = write_roster([REC, REC_NO_PDF], csvp, jp)
    rows = list(csv.DictReader(open(csvp)))
    assert [r["ntrs_id"] for r in rows] == [str(REC["id"]), str(REC_NO_PDF["id"])]
    assert len(jp.read_text().splitlines()) == 2
    assert summary["total"] == 2 and summary["no_pdf"] == 1
    assert summary["by_series"]["TM"] == 2
    assert summary["by_decade"]["1920s"] == 2
    assert [r["in_pilot"] for r in rows] == ["True", "False"]
    assert summary["pilot_overlap"] == 1


def test_run_writes_atomically_and_summarizes(tmp_path, monkeypatch):
    monkeypatch.setattr(roster, "load_reports", lambda p=None: [])
    monkeypatch.setattr(roster, "year_windows",
                        lambda *a, **k: [("1920-01-01", "1930-01-01")])
    db = [("1925-01-01", i) for i in range(5)]
    csvp, jp = tmp_path / "roster.csv", tmp_path / "roster.jsonl"
    summary = roster.run(_fake_post(db, []), csvp, jp)
    assert summary["total"] == 5
    assert len(jp.read_text().splitlines()) == 5
    assert len(list(csv.DictReader(open(csvp)))) == 5
    assert sorted(p.name for p in tmp_path.iterdir()) == ["roster.csv", "roster.jsonl"]
