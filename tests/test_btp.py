import csv, io, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench" / "btp"))
sys.path.insert(0, str(ROOT / "pipeline"))

import prep  # noqa: E402
import score_btp  # noqa: E402

TEXT = "The law of the land. " * 30


def _csv(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode()


def _row(i, item, status="completed", text=TEXT):
    return {"Campaign": "c", "Project": "p", "Item": f"item {item}", "ItemId": item,
            "Asset": f"asset-{i}", "AssetStatus": status,
            "DownloadUrl": f"https://tile.loc.gov/x/{i}.jpg", "Transcription": text}


def test_eligible_filters_status_length_and_empty():
    assert prep.eligible(_row(1, "a"))
    assert not prep.eligible(_row(1, "a", status="submitted"))
    assert not prep.eligible(_row(1, "a", text="short"))
    assert not prep.eligible(_row(1, "a", text="Nothing to transcribe" + " x" * 300))


def test_sample_seeded_one_page_per_item():
    rows = prep.load_rows(_csv([_row(i, str(i % 5)) for i in range(40)]))
    a, b = prep.sample(rows, 10, 0), prep.sample(rows, 10, 0)
    assert [r["Asset"] for r in a] == [r["Asset"] for r in b]
    assert len(a) == 5 and len({r["ItemId"] for r in a}) == 5


def test_stage_writes_pipeline_layout(tmp_path, monkeypatch):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("L", (40, 60), 255).save(buf, format="JPEG")
    monkeypatch.setattr(prep, "DATA", tmp_path)
    monkeypatch.setattr(prep, "PAUSE", 0)
    rec = prep.stage(_row(7, "z"), "2023446314", fetch_fn=lambda u: buf.getvalue())
    d = tmp_path / rec["doc"]
    assert (d / "pages" / "1.png").exists()
    meta = json.loads((d / "ground_truth" / "1.json").read_text())
    assert meta["status"] == "btp_reviewed"   # never "done": stays out of NACA scoring
    assert (d / "ground_truth" / "1.txt").read_text().startswith("The law")


def test_btp_gt_markup():
    assert score_btp.btp_gt("a [?] b [illegible] c") == "a   b   c"
    assert score_btp.btp_gt("the [Senate?] voted") == "the Senate voted"
    assert score_btp.btp_gt("see [1] above") == "see [1] above"


def test_score_page_and_summary(tmp_path, monkeypatch):
    import common, score
    monkeypatch.setattr(common, "DATA", tmp_path)
    monkeypatch.setattr(score, "report_dir", lambda d: tmp_path / d)
    monkeypatch.setattr(score_btp, "report_dir", lambda d: tmp_path / d)
    d = tmp_path / "btp-x"
    (d / "ground_truth").mkdir(parents=True)
    (d / "vision").mkdir()
    (d / "ground_truth" / "1.txt").write_text("hello [world?]")
    (d / "vision" / "1.txt").write_text("hello world")
    r = score_btp.score_page("btp-x", 1)
    assert r["engines"]["vision"]["cer"] == 0.0
    s = score_btp.summarize([r])
    assert s["vision"]["n"] == 1 and s["vision"]["median_cer"] == 0.0
