import csv
import json

import common
import pass1


def _write(path, rows, fields=("ntrs_id", "published_year", "series", "no_pdf")):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _row(i, year, series="TN", no_pdf="False"):
    return {"ntrs_id": str(i), "published_year": str(year), "series": series, "no_pdf": no_pdf}


def test_load_reports_default_is_pilot(monkeypatch):
    monkeypatch.delenv("NACA_REPORTS_CSV", raising=False)
    assert len(common.load_reports()) == len(common.load_reports(str(common.CSV)))


def test_load_reports_env_override(tmp_path, monkeypatch):
    p = tmp_path / "env.csv"
    _write(p, [_row(1, 1950)])
    monkeypatch.setenv("NACA_REPORTS_CSV", str(p))
    assert [r["ntrs_id"] for r in common.load_reports()] == ["1"]


def test_load_reports_explicit_arg_wins(tmp_path, monkeypatch):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _write(a, [_row(1, 1950)])
    _write(b, [_row(2, 1950)])
    monkeypatch.setenv("NACA_REPORTS_CSV", str(a))
    assert [r["ntrs_id"] for r in common.load_reports(str(b))] == ["2"]


def test_batch_selection_sorted_and_sliced():
    rows = [_row(i, y) for i, y in [(7, 1960), (3, 1920), (5, 1940), (1, 1910),
                                    (6, 1950), (2, 1915), (4, 1930)]]
    assert [r["ntrs_id"] for r in pass1.select_batch(rows, 1, 3)] == ["4", "5", "6"]
    assert [r["ntrs_id"] for r in pass1.select_batch(rows, 0, 3)] == ["1", "2", "3"]
    assert [r["ntrs_id"] for r in pass1.select_batch(rows, 2, 3)] == ["7"]


def test_batch_population_filter_and_sort_order():
    rows = [_row(1, 1980, "other"), _row(2, 1980, "TN"), _row(3, "", "RM"),
            _row(4, "", "other"), _row(5, 999, "other"), _row(6, 1950, "other")]
    got = [r["ntrs_id"] for r in pass1.select_batch(rows, 0, 100)]
    # year 1980+other dropped, empty+other dropped; years sorted as ints, empty last
    assert got == ["5", "6", "2", "3"]


def test_batch_requires_published_year():
    try:
        pass1.select_batch([{"ntrs_id": "1", "series": "TN"}], 0, 5)
    except SystemExit:
        return
    raise AssertionError("expected SystemExit")


def test_record_no_pdf_disposition(tmp_path, monkeypatch):
    monkeypatch.setattr(pass1, "report_dir", lambda i: tmp_path / i)
    called = []
    monkeypatch.setattr(pass1, "download", lambda r: called.append(r))
    rows = [_row(1, 1950, no_pdf="True"), _row(2, 1951)]
    with_pdf, no_pdf = pass1.split_no_pdf(rows)
    for r in no_pdf:
        pass1.record_no_pdf(r)
    assert [r["ntrs_id"] for r in with_pdf] == ["2"] and not called
    meta = json.loads((tmp_path / "1" / "meta.json").read_text())
    assert meta == {"pdf_status": "no_pdf_in_record", "csv_row": rows[0]}


# ---------------------------------------------------------------- manifests
def _roster(n=7):
    return [_row(i, 1900 + i) for i in range(1, n + 1)]


def test_manifest_written_on_first_selection(tmp_path):
    csvp = tmp_path / "roster.csv"
    _write(csvp, _roster())
    rows = common.load_reports(str(csvp))
    got = pass1.resolve_batch(rows, 0, 3, csvp, batches_dir=tmp_path / "batches")
    assert [r["ntrs_id"] for r in got] == ["1", "2", "3"]
    m = json.loads((tmp_path / "batches" / "batch_000.json").read_text())
    assert m["ids"] == ["1", "2", "3"] and m["batch"] == 0 and m["batch_size"] == 3
    assert m["roster_sha256"] == pass1.sha256_file(csvp)
    assert m["population_rule"] == "published_year<=1975 or series!=other"
    assert m["created_ts"]


def test_manifest_ids_used_and_mismatch_refused(tmp_path):
    csvp = tmp_path / "roster.csv"
    _write(csvp, _roster())
    bd = tmp_path / "batches"
    pass1.resolve_batch(common.load_reports(str(csvp)), 0, 3, csvp, batches_dir=bd)
    # an earlier-dated row appears: recomputed selection now differs from manifest
    _write(csvp, _roster() + [_row(99, 1850)])
    rows = common.load_reports(str(csvp))
    try:
        pass1.resolve_batch(rows, 0, 3, csvp, batches_dir=bd)
    except SystemExit as e:
        assert "--rebatch" in str(e.code)
    else:
        raise AssertionError("expected refusal")
    assert json.loads((bd / "batch_000.json").read_text())["ids"] == ["1", "2", "3"]


def test_rebatch_overrides_manifest(tmp_path):
    csvp = tmp_path / "roster.csv"
    _write(csvp, _roster())
    bd = tmp_path / "batches"
    pass1.resolve_batch(common.load_reports(str(csvp)), 0, 3, csvp, batches_dir=bd)
    _write(csvp, _roster() + [_row(99, 1850)])
    rows = common.load_reports(str(csvp))
    got = pass1.resolve_batch(rows, 0, 3, csvp, batches_dir=bd, rebatch=True)
    assert [r["ntrs_id"] for r in got] == ["99", "1", "2"]
    assert json.loads((bd / "batch_000.json").read_text())["ids"] == ["99", "1", "2"]


def test_matching_manifest_reused(tmp_path):
    csvp = tmp_path / "roster.csv"
    _write(csvp, _roster())
    bd = tmp_path / "batches"
    rows = common.load_reports(str(csvp))
    pass1.resolve_batch(rows, 1, 3, csvp, batches_dir=bd)
    before = (bd / "batch_001.json").read_text()
    got = pass1.resolve_batch(rows, 1, 3, csvp, batches_dir=bd)
    assert [r["ntrs_id"] for r in got] == ["4", "5", "6"]
    assert (bd / "batch_001.json").read_text() == before


def _fake_urlopen(monkeypatch, urls, fail_404=()):
    import io
    import urllib.error

    class Resp(io.BytesIO):
        status = 200

        def __init__(self, data, url):
            super().__init__(data)
            self._url = url

        def geturl(self):
            return self._url

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake(req, timeout=None):
        url = req.full_url
        urls.append(url)
        if url in fail_404:
            raise urllib.error.HTTPError(url, 404, "nf", {}, None)
        if "/api/citations/" in url and "/downloads/" not in url:
            return Resp(b'{"downloads": [{"links": {"pdf": "/api/citations/9/downloads/alt.pdf"}}]}', url)
        return Resp(b"%PDF-fake", url)

    monkeypatch.setattr(pass1.urllib.request, "urlopen", fake)
    monkeypatch.setattr(common, "ntrs_wait", lambda *a, **k: None)
    monkeypatch.setattr(pass1.time, "sleep", lambda s: None)


def test_download_uses_roster_pdf_url(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "DATA", tmp_path)
    urls = []
    _fake_urlopen(monkeypatch, urls)
    row = {"ntrs_id": "9", "pdf_url": "https://ntrs.nasa.gov/api/citations/9/downloads/9.pdf"}
    meta = pass1.download(row)
    assert urls == [row["pdf_url"]]
    assert meta["sha256"]


def test_download_falls_back_without_pdf_url(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "DATA", tmp_path)
    urls = []
    _fake_urlopen(monkeypatch, urls)
    meta = pass1.download({"ntrs_id": "9", "pdf_url": "", "no_pdf": "False"})
    assert urls[0] == "https://ntrs.nasa.gov/api/citations/9"
    assert urls[1].endswith("alt.pdf")
    assert meta["sha256"]


def test_download_roster_404_falls_back_to_citation(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "DATA", tmp_path)
    urls = []
    bad = "https://ntrs.nasa.gov/api/citations/9/downloads/9.pdf"
    _fake_urlopen(monkeypatch, urls, fail_404=(bad,))
    meta = pass1.download({"ntrs_id": "9", "pdf_url": bad})
    assert urls[0] == bad and urls[1] == "https://ntrs.nasa.gov/api/citations/9"
    assert urls[2].endswith("alt.pdf") and meta["sha256"]
