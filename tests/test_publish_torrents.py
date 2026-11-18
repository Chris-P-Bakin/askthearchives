import hashlib
import json
import os

import pytest
import torf

import publish
from test_publish import ID, _put, tree, escalate_meta  # noqa: F401


def _kw(tree):
    data, out, review, queue = tree
    return dict(out_root=out, data_dir=data, review_path=review, queue_path=queue)


@pytest.fixture
def shipped(tree):
    publish.export_batch(7, **_kw(tree))
    return tree


def _supersede(tree):
    _put(tree[0] / ID, "vision_235b", 1, "NEW ONE", escalate_meta())


def _snap(root):
    return {p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_errata_only_for_changed_pages(shipped):
    assert publish.gen_errata(7, **_kw(shipped)) == 0
    _supersede(shipped)
    # page 1 was shipped flagged (pending table_flags); the 235b layer resolves it -> flags file too
    assert publish.gen_errata(7, **_kw(shipped)) == 2
    er = shipped[1] / "errata" / "batch_007" / ID
    assert sorted(p.name for p in er.iterdir()) == ["1.flags.json", "1.meta.json", "1.txt"]
    assert (er / "1.txt").read_text() == "NEW ONE"
    m = json.loads((er / "1.meta.json").read_text())
    old = hashlib.sha256(b"TESS1").hexdigest()
    assert m["supersedes_sha256"] == old and m["reason"] == "superseded"
    assert m["source"] == "vision_235b" and m["config_hash"] == "h235"
    assert {"quality", "flagged", "ts"} <= set(m)


def test_errata_deterministic(shipped):
    _supersede(shipped)
    publish.gen_errata(7, **_kw(shipped))
    a = _snap(shipped[1] / "errata")
    publish.gen_errata(7, **_kw(shipped))
    assert _snap(shipped[1] / "errata") == a


def test_errata_never_touches_batch_tree(shipped):
    _supersede(shipped)
    root = shipped[1] / "batch_007"
    before = _snap(root)
    publish.gen_errata(7, **_kw(shipped))
    assert _snap(root) == before


def test_errata_removed_when_no_longer_superseded(shipped):
    _supersede(shipped)
    publish.gen_errata(7, **_kw(shipped))
    import shutil
    shutil.rmtree(shipped[0] / ID / "vision_235b")
    assert publish.gen_errata(7, **_kw(shipped)) == 0
    assert not list((shipped[1] / "errata").rglob("*.txt"))


def test_torrent_piece_hashes_cover_manifest_files(shipped):
    p = publish.make_torrent(7, **_kw(shipped))
    assert p == shipped[1] / "torrents" / "batch_007.torrent"
    t = torf.Torrent.read(p)
    root = shipped[1] / "batch_007"
    man = {json.loads(l)["path"] for l in (root / "manifest.jsonl").read_text().splitlines()}
    got = {"/".join(f.parts[1:]) for f in t.files}
    assert got == man | {"manifest.jsonl"}
    assert t.name == "batch_007" and not t.private and not t.trackers
    assert t.verify(root)  # piece hashes match the tree


def test_torrent_uses_trackers_file(shipped):
    (shipped[1] / "trackers.txt").write_text("http://t.example/announce\n\nudp://u.example:1/announce\n")
    t = torf.Torrent.read(publish.make_torrent(7, **_kw(shipped)))
    assert sorted(u for tier in t.trackers for u in tier) == [
        "http://t.example/announce", "udp://u.example:1/announce"]


def test_torrent_requires_committed_batch(tree):
    with pytest.raises(SystemExit):
        publish.make_torrent(7, **_kw(tree))


def test_cumulative_has_batches_errata_corpus(shipped):
    _supersede(shipped)
    publish.gen_errata(7, **_kw(shipped))
    p = publish.make_cumulative(date="20261114", no_embed=True, **_kw(shipped))
    assert p.name == "cumulative_20261114.torrent"
    t = torf.Torrent.read(p)
    names = {"/".join(f.parts[1:]) for f in t.files}
    assert "batch_007/manifest.jsonl" in names and "batch_007/index_shard.db" in names
    assert f"errata/batch_007/{ID}/1.txt" in names and "corpus.db" in names
    import sqlite3
    c = sqlite3.connect(shipped[1] / "corpus.db")
    srcs = {r[0] for r in c.execute("SELECT source FROM pages")}
    assert srcs <= {"tesseract", "vision_27b_v2", "vision_235b"}
    assert c.execute("SELECT count(*) FROM pages").fetchone()[0] == 3


def test_regression_logged_not_overlaid(tree):
    import shutil
    _put(tree[0] / ID, "vision_235b", 1, "GOOD", escalate_meta())
    publish.export_batch(7, **_kw(tree))
    shutil.rmtree(tree[0] / ID / "vision_235b")
    # the page regressed to a flagged tesseract page: flag erratum yes, text erratum no
    assert publish.gen_errata(7, **_kw(tree)) == 1
    er = tree[1] / "errata"
    assert not list((er / "batch_007").rglob("*.txt"))
    rows = [json.loads(l) for l in (er / "regressions.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["shipped_source"] == "vision_235b"
    assert rows[0]["current_source"] == "tesseract" and rows[0]["page"] == 1
    a = (er / "regressions.jsonl").read_bytes()
    publish.gen_errata(7, **_kw(tree))
    assert (er / "regressions.jsonl").read_bytes() == a


def test_retraction_tombstone(shipped):
    import shutil
    _supersede(shipped)
    publish.gen_errata(7, **_kw(shipped))
    shutil.rmtree(shipped[0] / ID / "vision_235b")
    publish.gen_errata(7, **_kw(shipped))
    er = shipped[1] / "errata"
    assert not (er / "batch_007").exists() or not any((er / "batch_007").rglob("*.txt"))
    rows = [json.loads(l) for l in (er / "retractions.jsonl").read_text().splitlines()]
    assert {r["kind"] for r in rows} == {"text", "flags"} and len(rows) == 2
    t = [r for r in rows if r["kind"] == "text"][0]
    assert t["page"] == 1 and t["retracted_sha256"] == hashlib.sha256(b"NEW ONE").hexdigest()
    publish.gen_errata(7, **_kw(shipped))
    assert len((er / "retractions.jsonl").read_text().splitlines()) == 2


def test_flag_change_writes_flags_only(shipped):
    review = shipped[2]
    review.write_text(review.read_text() + json.dumps(
        {"unit_id": f"{ID}-p2", "ntrs_id": ID, "page": 2, "reason": "table_digits",
         "source_layer": "vision_27b_v2", "config_hash": "h", "ts": "2026-11-01T00:00:00Z"}) + "\n")
    assert publish.gen_errata(7, **_kw(shipped)) == 1
    er = shipped[1] / "errata" / "batch_007" / ID
    assert sorted(p.name for p in er.iterdir()) == ["2.flags.json"]
    f = json.loads((er / "2.flags.json").read_text())
    assert f["flagged"] is True and f["was"] is False
    assert f["reasons"] == ["review:table_digits"] and f["ts"] == "2026-11-01T00:00:00Z"


def test_torrent_refuses_drift(shipped):
    root = shipped[1] / "batch_007"
    (root / ID / "text" / "stray.tmp").write_text("x")
    with pytest.raises(SystemExit, match="stray.tmp"):
        publish.make_torrent(7, **_kw(shipped))
    with pytest.raises(SystemExit):
        publish.make_cumulative(date="20261114", no_embed=True, **_kw(shipped))
    (root / ID / "text" / "stray.tmp").unlink()
    p = root / ID / "text" / "1.txt"
    p.write_text("tampered")
    with pytest.raises(SystemExit, match="1.txt"):
        publish.make_torrent(7, **_kw(shipped))


def test_cumulative_same_day_gets_suffix(shipped):
    a = publish.make_cumulative(date="20261114", no_embed=True, **_kw(shipped))
    before = a.read_bytes()
    b = publish.make_cumulative(date="20261114", no_embed=True, **_kw(shipped))
    assert a.name == "cumulative_20261114.torrent" and b.name == "cumulative_20261114-2.torrent"
    assert a.read_bytes() == before
    assert (shipped[1] / "cumulative_20261114").is_dir() and (shipped[1] / "cumulative_20261114-2").is_dir()


def test_cumulative_regenerates_errata_itself(shipped):
    _supersede(shipped)  # post-commit escalation; no separate gen_errata call
    publish.make_cumulative(date="20261114", no_embed=True, **_kw(shipped))
    stage = shipped[1] / "cumulative_20261114"
    assert (stage / "errata" / "batch_007" / ID / "1.txt").read_text() == "NEW ONE"


def test_cumulative_embeds_by_default(shipped, monkeypatch):
    seen = {}

    def fake_build(db, data, **kw):
        seen.update(kw)
        db.write_bytes(b"x")
    monkeypatch.setattr(publish, "build", fake_build)
    publish.make_cumulative(date="20261114", **_kw(shipped))
    assert seen["embed"] is True and seen["campaign_only"] is True
