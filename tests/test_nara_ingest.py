import hashlib, json, sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import nara_ingest as ni
import units

CID = "nara-rg255"


def make_pdf(path, n_pages=2):
    ims = [Image.new("RGB", (200, 260), "white") for _ in range(n_pages)]
    ims[0].save(path, save_all=True, append_images=ims[1:], resolution=72)
    return Path(path).read_bytes()


def rec(na_id, restricted="Unrestricted", objs=None):
    if objs is None:
        objs = [{"objectType": "Portable Document File (PDF)", "objectFilename": f"{na_id}.pdf",
                 "objectFileSize": 1000, "objectUrl": f"https://example.test/{na_id}.pdf"}]
    return {"record": {"naId": na_id, "title": f"T{na_id}", "digitalObjects": objs,
                       "inclusiveStartDate": {"logicalDate": "1960-01-01"},
                       "useRestriction": {"status": restricted}}}


def write_meta(d, recs, extra_lines=()):
    d.mkdir(parents=True, exist_ok=True)
    (d / "m.jsonl").write_text("\n".join([json.dumps(r) for r in recs] + list(extra_lines)) + "\n")
    return d


@pytest.fixture
def root(tmp_path, monkeypatch):
    r = tmp_path / "nara"
    monkeypatch.setattr(units.registry, "root", lambda cid: r)
    return r


def fake_fetch_factory(tmp_path, n_pages=2):
    src = tmp_path / "src.pdf"
    data = make_pdf(src, n_pages)

    def fetch(url, dest):
        Path(dest).write_bytes(data)
    fetch.sha = hashlib.sha256(data).hexdigest()
    return fetch


def low_ocr(png):
    return {"engine": "fake", "config_hash": "x", "text": "t",
            "words": [{"text": "a", "conf": 10.0, "bbox": [0, 0, 1, 1]}]}


def high_ocr(png):
    return {"engine": "fake", "config_hash": "x", "text": "t",
            "words": [{"text": "a", "conf": 95.0, "bbox": [0, 0, 1, 1]}]}


def no_layout(rdir, n):
    return None


def test_selection_predicate_rights_and_pdf(tmp_path):
    meta = write_meta(tmp_path / "meta", [
        rec(1), rec(2, "Restricted - Possibly"), rec(3, "Undetermined"),
        rec(4, objs=[{"objectType": "Image (JPG)", "objectUrl": "u", "objectFilename": "a.jpg"}]),
        rec(5)])
    sel, counts = ni.select(meta, done=set())
    assert [r["naId"] for r in sel] == [1, 5]
    assert counts["excluded_restricted"] == 2
    assert counts["no_pdf"] == 1
    sel, counts = ni.select(meta, done={"nara255-1"})
    assert [r["naId"] for r in sel] == [5]
    assert counts["already_done"] == 1


def test_ingest_skips_nonpdf_and_malformed(tmp_path):
    meta = write_meta(tmp_path / "meta", [rec(1), {"record": {"title": "no id"}},
                                          {"record": {"naId": 9, "digitalObjects": "bogus",
                                                      "useRestriction": {"status": "Unrestricted"}}},
                                          {"nothing": 1}],
                      extra_lines=["{not json"])
    sel, counts = ni.select(meta, done=set())
    assert [r["naId"] for r in sel] == [1]
    assert counts["malformed"] == 4


def test_select_rejects_non_int_naid(tmp_path):
    meta = write_meta(tmp_path / "meta", [rec("../evil"), rec(2.5), rec(True), rec(3)])
    sel, counts = ni.select(meta, done=set())
    assert [r["naId"] for r in sel] == [3]
    assert counts["malformed"] == 3


def run_one(root, tmp_path, ocr=low_ocr, n_pages=2):
    fetch = fake_fetch_factory(tmp_path, n_pages)
    r = rec(7)["record"]
    out = ni.ingest_doc(r, root, fetch=fetch, ocr=ocr, layout=no_layout)
    return fetch, out


def test_doc_processed_layout(root, tmp_path):
    fetch, out = run_one(root, tmp_path)
    d = root / "nara255-7"
    assert sorted(p.name for p in (d / "pages").glob("*.png")) == ["1.png", "2.png"]
    assert sorted(p.name for p in (d / "ocr").glob("*.json")) == ["1.json", "2.json"]
    meta = json.loads((d / "meta.json").read_text())
    assert meta["naId"] == 7 and meta["pdf_sha256"] == fetch.sha and meta["n_pages"] == 2
    assert meta["objectUrl"] == "https://example.test/7.pdf"
    assert meta["gated_pages"] == [1, 2]
    lines = units.read_units(root / "units_pending.jsonl")
    assert [u["page"] for u in lines] == [1, 2]
    assert all(u["collection"] == CID and u["ntrs_id"] == "nara255-7" for u in lines)
    done = [json.loads(l) for l in (root / "ingest_done.jsonl").read_text().splitlines()]
    assert done[0]["doc_id"] == "nara255-7"


def test_ungated_pages_emit_no_units(root, tmp_path):
    run_one(root, tmp_path, ocr=high_ocr)
    assert not (root / "units_pending.jsonl").exists() or not units.read_units(root / "units_pending.jsonl")


def test_pdf_deleted_after_raster(root, tmp_path):
    run_one(root, tmp_path)
    assert not list(root.rglob("*.pdf"))
    assert list((root / "nara255-7" / "pages").glob("*.png"))


def test_disk_cap_pauses_selection(root, tmp_path, monkeypatch):
    pages = root / "nara255-1" / "pages"
    pages.mkdir(parents=True)
    (pages / "1.png").write_bytes(b"x" * 4096)
    monkeypatch.setenv("NARA_INFLIGHT_GB", str(1000 / 1e9))
    assert ni.inflight_bytes(root) >= 4096
    meta = write_meta(tmp_path / "meta", [rec(2)])
    fetch = fake_fetch_factory(tmp_path)
    res = ni.run_cycle(meta, root, limit=5, workers=1, fetch=fetch, ocr=low_ocr, layout=no_layout)
    assert res["paused"] is True and res["ingested"] == 0
    assert not (root / "nara255-2").exists()
    monkeypatch.setenv("NARA_INFLIGHT_GB", "100")
    res = ni.run_cycle(meta, root, limit=5, workers=1, fetch=fetch, ocr=low_ocr, layout=no_layout)
    assert res["ingested"] == 1 and not res["paused"]


def accept(d, n):
    (d / units.CAMPAIGN_LAYER).mkdir(exist_ok=True)
    (d / units.CAMPAIGN_LAYER / f"{n}.meta.json").write_text(
        json.dumps({"config_hash": next(iter(units.ACCEPTED_HASHES))}))


def test_no_delete_while_pages_pending(root, tmp_path):
    run_one(root, tmp_path)
    d = root / "nara255-7"
    accept(d, 1)
    assert ni.reap(root)["reaped"] == []
    assert (d / "pages").is_dir() and not (root / "retention.jsonl").exists()
    accept(d, 2)
    assert ni.reap(root)["reaped"] == ["nara255-7"]
    assert not (d / "pages").exists()


def test_reap_zero_gated_pages(root, tmp_path):
    run_one(root, tmp_path, ocr=high_ocr)
    assert ni.reap(root)["reaped"] == ["nara255-7"]
    assert not (root / "nara255-7" / "pages").exists()


def test_retention_ledger_line_fields(root, tmp_path):
    fetch, _ = run_one(root, tmp_path)
    d = root / "nara255-7"
    accept(d, 1)
    accept(d, 2)
    ni.reap(root)
    lines = [json.loads(l) for l in (root / "retention.jsonl").read_text().splitlines()]
    assert len(lines) == 1
    assert set(lines[0]) == {"doc_id", "n_pages", "pdf_sha256", "objectUrl", "deleted_ts"}
    assert lines[0]["doc_id"] == "nara255-7" and lines[0]["n_pages"] == 2
    assert lines[0]["pdf_sha256"] == fetch.sha
    assert lines[0]["objectUrl"] == "https://example.test/7.pdf"
    ni.reap(root)  # idempotent: no second line
    assert len((root / "retention.jsonl").read_text().splitlines()) == 1


def test_rewrite_object_url_stale_s3_to_medialz():
    stale = "https://s3.amazonaws.com/NARAprodstorage/lz/electronic-records/rg-255/x/y.pdf"
    assert ni.rewrite_object_url(stale) == (
        "https://catalog.archives.gov/medialz/electronic-records/rg-255/x/y.pdf")
    fresh = "https://catalog.archives.gov/medialz/a/b.pdf"
    assert ni.rewrite_object_url(fresh) == fresh


def _rec(naid=9001):
    return {"naId": naid, "title": "T", "useRestriction": {"status": "Unrestricted"},
            "digitalObjects": [{"objectType": "Portable Document File (PDF)",
                                "objectUrl": "https://x/pdf", "objectFileSize": 10}]}


def _fake_fetch_pdf(tmp_path):
    import pass1
    from PIL import Image
    src = tmp_path / "fixture.pdf"
    Image.new("L", (200, 300), 255).save(src, "PDF")
    def fetch(url, dest):
        Path(dest).write_bytes(src.read_bytes())
    return fetch


def test_text_layer_doc_skips_raster_ocr_and_units(tmp_path, monkeypatch):
    root = tmp_path / "root"; root.mkdir()
    def extract(pdf):
        return ["x" * 120]          # single page, rich text layer
    def boom(*a, **k):
        raise AssertionError("must not run for text-layer pages")
    meta = ni.ingest_doc(_rec(), root, fetch=_fake_fetch_pdf(tmp_path),
                         ocr=boom, layout=boom, extract=extract)
    d = root / meta["doc_id"]
    assert (d / "pdftext" / "1.txt").read_text() == "x" * 120
    assert not (d / "pages").exists() and not (d / "ocr").exists()
    assert meta["gated_pages"] == [] and meta["pdftext_pages"] == 1
    assert not (root / "units_pending.jsonl").exists()


def test_gate_is_confidence_only_no_layout_call(root, tmp_path):
    def extract(pdf):
        return [""]                 # no text layer -> OCR path
    def low_conf_ocr(png):
        return {"text": "t", "words": [{"conf": 10.0}]}
    def boom(*a, **k):
        raise AssertionError("layout must not be called")
    meta = ni.ingest_doc(_rec(9002), root, fetch=_fake_fetch_pdf(tmp_path),
                         ocr=low_conf_ocr, layout=boom, extract=extract)
    assert (root / meta["doc_id"] / "pages" / "1.png").exists()
    assert meta["gated_pages"] == [1] and meta["ocr_pages"] == 1
