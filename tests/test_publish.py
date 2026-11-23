import hashlib
import json
import os

import pytest

import escalate
import publish

ID = "111"


def _put(rdir, layer, n, text, meta):
    d = rdir / layer
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{n}.txt").write_text(text)
    (d / f"{n}.meta.json").write_text(json.dumps(meta))


@pytest.fixture
def tree(tmp_path, monkeypatch):
    empty = tmp_path / "roster.csv"
    empty.write_text("ntrs_id,title\n")
    monkeypatch.setenv("NACA_REPORTS_CSV", str(empty))
    data = tmp_path / "data"
    (data / "batches").mkdir(parents=True)
    (data / "batches" / "batch_007.json").write_text(
        json.dumps({"batch": 7, "ids": [ID, "222"]}))
    rdir = data / ID
    (rdir / "pages").mkdir(parents=True)
    (rdir / "ocr").mkdir()
    (rdir / "source.pdf").write_bytes(b"%PDF-fake")
    (rdir / "meta.json").write_text(json.dumps({"csv_row": {"title": "T"}}))
    for n in (1, 2, 3):
        (rdir / "pages" / f"{n}.png").write_bytes(b"PNG%d" % n)
        (rdir / "ocr" / f"{n}.json").write_text(json.dumps(
            {"engine": "tesseract", "text": f"TESS{n}", "config_hash": "t", "words": []}))
    (rdir / "pages" / "1.raw.png").write_bytes(b"RAW")
    # legacy layers that must never reach the export
    _put(rdir, "vision_27bq4", 1, "LEGACY27", {"config_hash": "x"})
    _put(rdir, "vision2", 2, "LEGACY2", {"config_hash": "x"})
    _put(rdir, "vision_haiku2", 3, "LEGACYH", {"config_hash": "x"})
    (rdir / "reconcile").mkdir()
    (rdir / "reconcile" / "1.json").write_text(json.dumps(
        {"page_source": "vision_27bq4", "confidence": "high"}))
    _put(rdir, "vision_27b_v2", 2, "T27 text", {"config_hash": escalate.ESC_27B_HASH})
    # no_pdf doc
    (data / "222").mkdir()
    (data / "222" / "meta.json").write_text(json.dumps({"disposition": "no_pdf"}))
    review = data / "escalation_review.jsonl"
    review.write_text(json.dumps({"unit_id": f"{ID}-p3", "ntrs_id": ID, "page": 3,
                                  "reason": "empty_output_27b", "source_layer": "vision_27b_v2",
                                  "config_hash": "h", "ts": "t"}) + "\n")
    queue = data / "escalation_queue.jsonl"
    queue.write_text(json.dumps({"unit_id": f"{ID}-p1", "ntrs_id": ID, "page": 1,
                                 "config_hash": "h", "ts": "t", "kind": "table_flags",
                                 "flags": [{"table": 1}]}) + "\n")
    return data, tmp_path / "publish", review, queue


def _export(tree):
    data, out, review, queue = tree
    return publish.export_batch(7, out_root=out, data_dir=data,
                                review_path=review, queue_path=queue)


def _files(root):
    return sorted(p for p in root.rglob("*") if p.is_file())


def test_export_contains_no_legacy_text(tree):
    s = _export(tree)
    root = tree[1] / "batch_007"
    texts = {n: (root / ID / "text" / f"{n}.txt").read_text() for n in (1, 2, 3)}
    assert texts == {1: "TESS1", 2: "T27 text", 3: "TESS3"}
    metas = [json.loads((root / ID / "text" / f"{n}.meta.json").read_text()) for n in (1, 2, 3)]
    assert [m["source"] for m in metas] == ["tesseract", "vision_27b_v2", "tesseract"]
    blob = b"".join(p.read_bytes() for p in _files(root) if p.suffix in (".txt", ".json", ".jsonl"))
    for bad in (b"LEGACY", b"vision_27bq4", b"vision_haiku2", b"reconcile"):
        assert bad not in blob
    assert s["pages"] == 3


def test_hardlinks_not_copies(tree):
    data = tree[0]
    _export(tree)
    root = tree[1] / "batch_007" / ID
    assert os.stat(root / "source.pdf").st_nlink > 1
    assert os.path.samefile(root / "source.pdf", data / ID / "source.pdf")
    assert os.path.samefile(root / "pages" / "2.png", data / ID / "pages" / "2.png")
    assert not (root / "pages" / "1.raw.png").exists()
    assert (root / "ocr" / "1.json").exists()


def test_manifest_covers_every_file_with_sha(tree):
    _export(tree)
    root = tree[1] / "batch_007"
    rows = [json.loads(l) for l in (root / "manifest.jsonl").read_text().splitlines()]
    listed = {r["path"] for r in rows}
    actual = {p.relative_to(root).as_posix() for p in _files(root)} - {"manifest.jsonl"}
    assert listed == actual
    assert {"index_shard.db", "BATCH_README.md", f"{ID}/flags.jsonl"} <= listed
    for r in rows:
        data = (root / r["path"]).read_bytes()
        assert r["sha256"] == hashlib.sha256(data).hexdigest() and r["bytes"] == len(data)
        assert not r["path"].startswith("/")


def test_idempotent_rerun_skips(tree):
    s1 = _export(tree)
    root = tree[1] / "batch_007"
    before = (root / "manifest.jsonl").read_bytes()
    s2 = _export(tree)
    assert s1["new_files"] > 0 and s2["new_files"] == 0
    assert (root / "manifest.jsonl").read_bytes() == before


def test_resume_partial_tree(tree):
    _export(tree)
    root = tree[1] / "batch_007"
    (root / "manifest.jsonl").unlink()
    (root / ID / "text" / "2.txt").unlink()
    s = _export(tree)
    assert s["new_files"] >= 1 and (root / ID / "text" / "2.txt").read_text() == "T27 text"
    assert (root / "manifest.jsonl").exists()


def test_no_abs_paths_in_export(tree):
    _export(tree)
    root = tree[1] / "batch_007"
    for p in _files(root):
        if p.suffix in (".txt", ".json", ".jsonl", ".md"):
            assert "/home/" not in p.read_text(), p
            assert str(tree[0]) not in p.read_text(), p


def test_flagged_page_listed_in_flags_jsonl(tree):
    _export(tree)
    root = tree[1] / "batch_007" / ID
    rows = [json.loads(l) for l in (root / "flags.jsonl").read_text().splitlines()]
    assert {(r["page"], r["status"]) for r in rows} == {(3, "review"), (1, "pending")}
    assert rows[0]["unit_id"] == f"{ID}-p3"  # review rows first, verbatim fields kept
    m = {n: json.loads((root / "text" / f"{n}.meta.json").read_text()) for n in (1, 2, 3)}
    assert m[1]["flagged"] and m[3]["flagged"] and not m[2]["flagged"]
    assert not (tree[1] / "batch_007" / "222").exists() or \
        not (tree[1] / "batch_007" / "222" / "flags.jsonl").exists()


def test_index_shard_campaign_only_no_embed(tree):
    import sqlite3
    s = _export(tree)
    c = sqlite3.connect(tree[1] / "batch_007" / "index_shard.db")
    srcs = {r[0] for r in c.execute("SELECT source FROM pages")}
    assert srcs <= {"tesseract", "vision_27b_v2"}
    assert dict(c.execute("SELECT key,value FROM index_meta"))["campaign_only"] == "True"
    assert s["no_pages_docs"] == 1


def test_committed_batch_not_rewritten(tree):
    data = tree[0]
    _export(tree)
    root = tree[1] / "batch_007"
    before = (root / "manifest.jsonl").read_bytes()
    _put(data / ID, "vision_235b", 1, "NEW", escalate_meta())
    s = _export(tree)
    assert s["committed"] and s["new_files"] == 0 and s["verified"]
    assert (root / ID / "text" / "1.txt").read_text() == "TESS1"
    assert (root / "manifest.jsonl").read_bytes() == before
    with pytest.raises(SystemExit):
        publish.export_batch(7, out_root=tree[1], data_dir=data, review_path=tree[2],
                             queue_path=tree[3], force=True)
    (root / ID / "text" / "1.txt").write_text("tampered")
    assert _export(tree)["mismatches"] == [f"{ID}/text/1.txt"]


def escalate_meta():
    return {"config": {"prompt_version": "file:x", "model": escalate.ENV_235B["NACA_VLM_MODEL"],
                       "loop_guard": {"version": "v2"}}, "config_hash": "h235"}


def test_crash_before_manifest_then_rerun_completes(tree, monkeypatch):
    real = publish._write

    def boom(dst, data):
        if dst.name == "manifest.jsonl":
            raise RuntimeError("crash")
        return real(dst, data)
    monkeypatch.setattr(publish, "_write", boom)
    with pytest.raises(RuntimeError):
        _export(tree)
    root = tree[1] / "batch_007"
    assert not (root / "manifest.jsonl").exists()
    monkeypatch.setattr(publish, "_write", real)
    s = _export(tree)
    assert not s["committed"] and (root / "manifest.jsonl").exists()


def test_docs_jsonl_covers_no_pdf_and_readme_reconciles(tree):
    _export(tree)
    root = tree[1] / "batch_007"
    docs = [json.loads(l) for l in (root / "docs.jsonl").read_text().splitlines()]
    d = {r["ntrs_id"]: r for r in docs}
    assert d["222"]["disposition"] == "no_pdf" and d["222"]["pages_exported"] == 0
    assert d[ID]["disposition"] == "ok" and d[ID]["pages_exported"] == 3
    man = {json.loads(l)["path"] for l in (root / "manifest.jsonl").read_text().splitlines()}
    assert "docs.jsonl" in man
    assert "2 documents (1 with pages, 1 no_pdf)" in (root / "BATCH_README.md").read_text()


def test_drained_table_page_resolved_and_wu_error_excluded(tree):
    data, out, review, queue = tree
    _put(data / ID, "vision_235b", 1, "FIXED", escalate_meta())
    with queue.open("a") as f:
        f.write(json.dumps({"unit_id": f"{ID}-p2", "ntrs_id": ID, "page": 2,
                            "config_hash": "h", "ts": "t", "kind": "wu_error"}) + "\n")
    _export(tree)
    root = out / "batch_007" / ID
    rows = [json.loads(l) for l in (root / "flags.jsonl").read_text().splitlines()]
    st = {(r["page"], r.get("kind")): r["status"] for r in rows}
    assert st[(1, "table_flags")] == "resolved"
    assert not any(r.get("kind") == "wu_error" for r in rows)
    assert json.loads((root / "text" / "1.meta.json").read_text())["flagged"] is False


def test_stale_shard_rebuilt_when_uncommitted(tree, monkeypatch):
    import sqlite3
    real = publish._write

    def boom(dst, data):
        if dst.name == "manifest.jsonl":
            raise RuntimeError("crash")
        return real(dst, data)
    monkeypatch.setattr(publish, "_write", boom)
    with pytest.raises(RuntimeError):
        _export(tree)
    monkeypatch.setattr(publish, "_write", real)
    shard = tree[1] / "batch_007" / "index_shard.db"
    shard.write_bytes(b"crash-partial garbage")  # tree files all unchanged on rerun
    _export(tree)
    c = sqlite3.connect(shard)
    assert c.execute("SELECT count(*) FROM pages").fetchone()[0] == 3
